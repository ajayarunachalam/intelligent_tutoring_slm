"""
preprocessor.py
───────────────
Converts MathDialExample objects into the Gemma 4 chat template format
required for supervised fine-tuning (SFT) with TRL/Unsloth.

Gemma 4 chat format:
    <bos><start_of_turn>user
    {user_message}<end_of_turn>
    <start_of_turn>model
    {assistant_message}<end_of_turn>

For tutoring: the model plays the TEACHER role.
We construct multi-turn conversations where:
  - system prompt  → tutoring instructions + problem context
  - user turns     → student messages
  - model turns    → teacher responses (what we train on)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from datasets import Dataset
from rich.console import Console
from rich.progress import track

from data_loader import MathDialExample, TutoringTurn

console = Console()
logger = logging.getLogger(__name__)


# ── System prompt template ─────────────────────────────────────────────────

TUTOR_SYSTEM_PROMPT = """\
You are an expert mathematics tutor helping a 7th-grade student solve a math problem through \
Socratic dialogue. Your goal is to GUIDE the student to discover the correct answer themselves, \
NOT to give them the answer directly.

Teaching principles you must follow:
1. **Scaffold** — break the problem into small steps and ask focused questions about each step.
2. **Diagnose** — identify the student's specific misconception before correcting it.
3. **Encourage** — acknowledge correct reasoning before addressing errors.
4. **Reveal as last resort** — only state the answer if the student has genuinely tried multiple times.
5. **Be concise** — keep each response to 1-3 sentences to maintain student engagement.

Dialog moves available to you:
- (focus)      Ask the student to re-read or focus on a specific part of the problem.
- (hint)       Provide a guiding hint without revealing the answer.
- (correction) Gently correct a specific error and ask a follow-up question.
- (approval)   Confirm correct reasoning and advance to the next step.
- (question)   Ask a probing question to diagnose or advance understanding.
- (solution)   [Last resort only] Reveal the answer after multiple failed attempts.

Problem context:
{problem}

Correct answer (for your reference only — do NOT reveal unless absolutely necessary):
{ground_truth}

Known student misconception:
{confusion}
"""


# ── Data structures ─────────────────────────────────────────────────────────

@dataclass
class ChatSample:
    """A formatted sample ready for SFT training."""
    qid: str
    messages: list[dict]   # [{"role": "system"|"user"|"assistant", "content": str}]
    raw_example: MathDialExample


# ── Preprocessor ────────────────────────────────────────────────────────────

class MathDialPreprocessor:
    """
    Transforms MathDialExample objects into chat-formatted training samples.

    Strategy: for each dialogue, we build one sample per teacher turn
    (teacher-response prediction), using full prior context as input.
    This maximises training signal from each dialogue.
    """

    def __init__(
        self,
        max_context_turns: int = 10,
        min_teacher_turns: int = 2,
        include_dialog_acts: bool = True,
        gemma_format: bool = True,
    ):
        """
        Args:
            max_context_turns:    max prior turns to include in context window
            min_teacher_turns:    skip dialogues with fewer teacher turns
            include_dialog_acts:  prefix teacher responses with (act) tags
            gemma_format:         apply Gemma 4 <start_of_turn> token wrapping
        """
        self.max_context_turns = max_context_turns
        self.min_teacher_turns = min_teacher_turns
        self.include_dialog_acts = include_dialog_acts
        self.gemma_format = gemma_format

    # ── Public API ─────────────────────────────────────────────────────────

    def process(self, examples: list[MathDialExample]) -> list[ChatSample]:
        """Convert a list of MathDialExample into ChatSample training instances."""
        samples = []
        skipped = 0
        for ex in track(examples, description="[cyan]Preprocessing dialogues..."):
            ex_samples = self._example_to_samples(ex)
            if ex_samples:
                samples.extend(ex_samples)
            else:
                skipped += 1

        console.print(
            f"[green]✓ Generated {len(samples)} training samples "
            f"from {len(examples)} dialogues "
            f"([yellow]{skipped} skipped[/yellow])[/green]"
        )
        return samples

    def to_hf_dataset(self, samples: list[ChatSample]) -> Dataset:
        """Convert ChatSamples to a HuggingFace Dataset with 'text' column."""
        rows = []
        for s in samples:
            text = self._format_for_gemma(s.messages) if self.gemma_format \
                   else self._format_as_json(s.messages)
            rows.append({
                "qid":      s.qid,
                "text":     text,
                "messages": json.dumps(s.messages),
            })
        return Dataset.from_list(rows)

    def process_to_hf(
        self,
        train_examples: list[MathDialExample],
        test_examples:  list[MathDialExample],
    ) -> tuple[Dataset, Dataset]:
        """Full pipeline: examples → HuggingFace Datasets for train & test."""
        console.print("\n[bold cyan]── Preprocessing train split ──[/bold cyan]")
        train_samples = self.process(train_examples)
        console.print("\n[bold cyan]── Preprocessing test split ──[/bold cyan]")
        test_samples  = self.process(test_examples)

        train_ds = self.to_hf_dataset(train_samples)
        test_ds  = self.to_hf_dataset(test_samples)

        console.print(f"\n[bold]Dataset sizes:[/bold] train={len(train_ds)}, test={len(test_ds)}")
        return train_ds, test_ds

    def save(self, train_ds: Dataset, test_ds: Dataset, output_dir: str = "./data/processed"):
        """Save preprocessed datasets to disk."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        train_ds.save_to_disk(str(out / "train"))
        test_ds.save_to_disk(str(out / "test"))
        console.print(f"[green]✓ Saved preprocessed data to {output_dir}[/green]")

    # ── Private helpers ─────────────────────────────────────────────────────

    def _example_to_samples(self, ex: MathDialExample) -> list[ChatSample]:
        """
        Generate one ChatSample per teacher turn in the dialogue.

        For each teacher turn at index i, the context is all turns 0..i-1,
        and the target is the teacher's response at turn i.
        """
        teacher_turns = [t for t in ex.turns if t.speaker == "Teacher"]
        if len(teacher_turns) < self.min_teacher_turns:
            return []

        system_msg = {
            "role": "system",
            "content": TUTOR_SYSTEM_PROMPT.format(
                problem=ex.question,
                ground_truth=ex.ground_truth,
                confusion=ex.teacher_described_confusion or ex.student_profile,
            ).strip()
        }

        samples = []
        # Build context progressively, generating one sample per teacher turn
        for i, turn in enumerate(ex.turns):
            if turn.speaker != "Teacher":
                continue

            # Context = all prior turns
            prior_turns = ex.turns[:i]
            # Limit context window
            if len(prior_turns) > self.max_context_turns:
                prior_turns = prior_turns[-self.max_context_turns:]

            messages = [system_msg]
            # Interleave user (student) and assistant (teacher) turns
            for pt in prior_turns:
                role = "user" if pt.speaker == "Student" else "assistant"
                content = self._format_turn_content(pt)
                # Merge consecutive same-role messages
                if messages and messages[-1]["role"] == role:
                    messages[-1]["content"] += "\n" + content
                else:
                    messages.append({"role": role, "content": content})

            # The target teacher response
            target = self._format_turn_content(turn)

            # The message list must end with a user turn for SFT
            # If context is empty or last turn was also teacher, add placeholder
            if len(messages) == 1:
                # First teacher turn with no student context yet
                messages.append({
                    "role": "user",
                    "content": f"[Session start] Student is working on the problem above."
                })
            elif messages[-1]["role"] == "assistant":
                # Back-to-back teacher turns (shouldn't happen much but handle it)
                messages.append({"role": "user", "content": "[continue]"})

            # Append the target teacher response as the final assistant turn
            messages.append({"role": "assistant", "content": target})

            samples.append(ChatSample(qid=ex.qid, messages=messages, raw_example=ex))

        return samples

    def _format_turn_content(self, turn: TutoringTurn) -> str:
        """Format a single turn's content, optionally including dialog act."""
        if self.include_dialog_acts and turn.speaker == "Teacher" and turn.dialog_act not in ("unknown", "response"):
            return f"({turn.dialog_act}) {turn.text}"
        return turn.text

    def _format_for_gemma(self, messages: list[dict]) -> str:
        """
        Apply Gemma 4 chat template manually.

        Format:
          <bos>
          [system turn — embedded into first user message for Gemma]
          <start_of_turn>user
          {user_content}<end_of_turn>
          <start_of_turn>model
          {assistant_content}<end_of_turn>
          <start_of_turn>model   ← training target (no <end_of_turn>, masked in loss)
        """
        parts = ["<bos>"]
        system_injected = False
        system_text = ""

        for msg in messages:
            role    = msg["role"]
            content = msg["content"]

            if role == "system":
                system_text = content
                continue

            if role == "user":
                # Inject system prompt into first user turn (Gemma style)
                if not system_injected and system_text:
                    content = f"{system_text}\n\n{content}"
                    system_injected = True
                parts.append(f"<start_of_turn>user\n{content}<end_of_turn>\n")

            elif role == "assistant":
                parts.append(f"<start_of_turn>model\n{content}<end_of_turn>\n")

        return "".join(parts)

    def _format_as_json(self, messages: list[dict]) -> str:
        """Plain JSON format for non-Gemma models."""
        return json.dumps(messages, ensure_ascii=False)

    # ── Quality checks ──────────────────────────────────────────────────────

    def quality_report(self, samples: list[ChatSample]) -> dict:
        """Compute basic quality statistics on the preprocessed samples."""
        lengths = [len(s.messages) for s in samples]
        target_lengths = [
            len(s.messages[-1]["content"].split())
            for s in samples
            if s.messages and s.messages[-1]["role"] == "assistant"
        ]
        act_dist: dict[str, int] = {}
        for s in samples:
            if s.raw_example:
                for t in s.raw_example.turns:
                    if t.speaker == "Teacher":
                        act_dist[t.dialog_act] = act_dist.get(t.dialog_act, 0) + 1

        report = {
            "total_samples": len(samples),
            "avg_turns_per_sample": sum(lengths) / max(len(lengths), 1),
            "avg_target_words": sum(target_lengths) / max(len(target_lengths), 1),
            "dialog_act_distribution": dict(sorted(act_dist.items(), key=lambda x: -x[1])),
        }

        console.print("\n[bold]Preprocessing Quality Report:[/bold]")
        console.print(f"  Total samples:          {report['total_samples']}")
        console.print(f"  Avg turns/sample:       {report['avg_turns_per_sample']:.1f}")
        console.print(f"  Avg target words:       {report['avg_target_words']:.1f}")
        console.print(f"  Dialog act dist:        {list(act_dist.items())[:5]}")

        return report


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_loader import MathDialLoader

    loader = MathDialLoader()
    train_ex, test_ex = loader.load(max_train=200, max_test=50)

    proc = MathDialPreprocessor(include_dialog_acts=True, gemma_format=True)
    train_ds, test_ds = proc.process_to_hf(train_ex, test_ex)
    proc.save(train_ds, test_ds)

    # Show a sample
    console.print("\n[bold]Sample formatted text (first 800 chars):[/bold]")
    console.print(train_ds[0]["text"][:800])
