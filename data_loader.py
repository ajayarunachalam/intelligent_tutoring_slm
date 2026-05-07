"""
data_loader.py
──────────────
Downloads the MathDial dataset from HuggingFace Hub and parses it into
structured Python objects ready for preprocessing.

MathDial: 2861 teacher-student tutoring dialogues grounded in
multi-step math reasoning problems (EMNLP 2023).
"""

import json
import os
import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import load_dataset, DatasetDict
from rich.console import Console
from rich.table import Table

console = Console()
logger = logging.getLogger(__name__)

# ── Data structures ────────────────────────────────────────────────────────

@dataclass
class TutoringTurn:
    """A single utterance in the tutoring dialogue."""
    speaker: str          # "Teacher" or "Student"
    dialog_act: str       # e.g. "focus", "hint", "correction", "solution"
    text: str             # utterance text


@dataclass
class MathDialExample:
    """One complete tutoring dialogue from MathDial."""
    qid: str
    scenario: int
    question: str                         # math word problem
    ground_truth: str                     # correct answer
    student_incorrect_solution: str       # what the student got wrong
    student_profile: str                  # misconception profile
    teacher_described_confusion: str      # teacher's annotation
    self_correctness: str                 # Yes / Yes, but I had to reveal / No
    self_typical_confusion: int           # Likert 1-5
    self_typical_interactions: int        # Likert 1-5
    turns: list[TutoringTurn] = field(default_factory=list)
    split: str = "train"                  # "train" or "test"


# ── Parser ─────────────────────────────────────────────────────────────────

def parse_conversation(raw_conversation: str) -> list[TutoringTurn]:
    """
    Parse the |EOM|-delimited conversation string into TutoringTurn objects.

    Format: "Teacher: (dialog_act) text|EOM|Student: text|EOM|Teacher: ..."
    """
    turns = []
    if not raw_conversation:
        return turns

    utterances = raw_conversation.split("|EOM|")
    for utt in utterances:
        utt = utt.strip()
        if not utt:
            continue

        # Extract speaker
        if utt.startswith("Teacher:"):
            speaker = "Teacher"
            rest = utt[len("Teacher:"):].strip()
        elif utt.startswith("Student:"):
            speaker = "Student"
            rest = utt[len("Student:"):].strip()
        else:
            # Fallback: treat whole utterance as unknown speaker
            turns.append(TutoringTurn(speaker="Unknown", dialog_act="unknown", text=utt))
            continue

        # Extract dialog act from (act) prefix (may be absent for student turns)
        act_match = re.match(r"^\(([^)]+)\)\s*(.*)", rest, re.DOTALL)
        if act_match:
            dialog_act = act_match.group(1).strip().lower()
            text = act_match.group(2).strip()
        else:
            dialog_act = "response" if speaker == "Student" else "unknown"
            text = rest.strip()

        turns.append(TutoringTurn(speaker=speaker, dialog_act=dialog_act, text=text))

    return turns


def row_to_example(row: dict, split: str = "train") -> MathDialExample:
    """Convert a raw dataset row to a MathDialExample."""
    turns = parse_conversation(str(row.get("conversation", "")))
    return MathDialExample(
        qid=str(row.get("qid", "")),
        scenario=int(row.get("scenario", 0)),
        question=str(row.get("question", "")),
        ground_truth=str(row.get("ground_truth", "")),
        student_incorrect_solution=str(row.get("student_incorrect_solution", "")),
        student_profile=str(row.get("student_profile", "")),
        teacher_described_confusion=str(row.get("teacher_described_confusion", "")),
        self_correctness=str(row.get("self-correctness", row.get("self_correctness", ""))),
        self_typical_confusion=int(row.get("self-typical-confusion", row.get("self_typical_confusion", 3))),
        self_typical_interactions=int(row.get("self-typical-interactions", row.get("self_typical_interactions", 3))),
        turns=turns,
        split=split,
    )


# ── Loader ─────────────────────────────────────────────────────────────────

class MathDialLoader:
    """
    Loads MathDial from HuggingFace Hub.

    Usage:
        loader = MathDialLoader(cache_dir="./data/mathdial")
        train, test = loader.load()
    """

    HF_DATASET_ID = "eth-nlped/mathdial"
    FALLBACK_URLS = {
        "train": "https://raw.githubusercontent.com/eth-nlped/mathdial/main/data/train.jsonl",
        "test":  "https://raw.githubusercontent.com/eth-nlped/mathdial/main/data/test.jsonl",
    }

    def __init__(self, cache_dir: str = "./data/mathdial"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._train: list[MathDialExample] = []
        self._test:  list[MathDialExample] = []

    # ── Public API ──────────────────────────────────────────────────────────

    def load(
        self,
        max_train: Optional[int] = None,
        max_test: Optional[int] = None,
        force_download: bool = False,
    ) -> tuple[list[MathDialExample], list[MathDialExample]]:
        """
        Download (or load from cache) and return (train_examples, test_examples).

        Args:
            max_train: cap training set size (useful for quick POC runs)
            max_test:  cap test set size
            force_download: ignore cache and re-download
        """
        cache_train = self.cache_dir / "train_parsed.json"
        cache_test  = self.cache_dir / "test_parsed.json"

        if not force_download and cache_train.exists() and cache_test.exists():
            console.print("[green]✓ Loading MathDial from local cache[/green]")
            self._train = self._load_cache(cache_train, "train")
            self._test  = self._load_cache(cache_test,  "test")
        else:
            console.print("[cyan]⬇  Downloading MathDial from HuggingFace Hub...[/cyan]")
            self._train, self._test = self._download_and_parse()
            self._save_cache(self._train, cache_train)
            self._save_cache(self._test,  cache_test)

        # Apply caps
        if max_train:
            self._train = self._train[:max_train]
        if max_test:
            self._test = self._test[:max_test]

        self._print_summary()
        return self._train, self._test

    def get_dataframe(self, split: str = "train") -> pd.DataFrame:
        """Return a pandas DataFrame for quick exploration."""
        examples = self._train if split == "train" else self._test
        rows = []
        for ex in examples:
            rows.append({
                "qid": ex.qid,
                "question": ex.question,
                "ground_truth": ex.ground_truth,
                "student_incorrect_solution": ex.student_incorrect_solution,
                "student_profile": ex.student_profile,
                "self_correctness": ex.self_correctness,
                "self_typical_confusion": ex.self_typical_confusion,
                "num_turns": len(ex.turns),
                "split": ex.split,
            })
        return pd.DataFrame(rows)

    # ── Private helpers ─────────────────────────────────────────────────────

    def _download_and_parse(self) -> tuple[list[MathDialExample], list[MathDialExample]]:
        """Attempt HF Hub first, fall back to raw GitHub URLs."""
        try:
            ds: DatasetDict = load_dataset(self.HF_DATASET_ID)
            train_examples = [row_to_example(row, "train") for row in ds["train"]]
            test_examples  = [row_to_example(row, "test")  for row in ds["test"]]
            console.print(f"[green]✓ Loaded from HuggingFace Hub ({self.HF_DATASET_ID})[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠  HF Hub failed ({e}), trying raw JSONL fallback...[/yellow]")
            train_examples = self._load_jsonl_url(self.FALLBACK_URLS["train"], "train")
            test_examples  = self._load_jsonl_url(self.FALLBACK_URLS["test"],  "test")

        return train_examples, test_examples

    def _load_jsonl_url(self, url: str, split: str) -> list[MathDialExample]:
        """Download a JSONL file from a URL and parse it."""
        import urllib.request
        local_path = self.cache_dir / f"{split}.jsonl"
        if not local_path.exists():
            console.print(f"  Downloading {url}")
            urllib.request.urlretrieve(url, local_path)
        examples = []
        with open(local_path, "r") as f:
            for line in f:
                row = json.loads(line.strip())
                examples.append(row_to_example(row, split))
        return examples

    def _save_cache(self, examples: list[MathDialExample], path: Path):
        """Serialise parsed examples to JSON for fast re-loading."""
        data = []
        for ex in examples:
            d = ex.__dict__.copy()
            d["turns"] = [t.__dict__ for t in ex.turns]
            data.append(d)
        with open(path, "w") as f:
            json.dump(data, f)

    def _load_cache(self, path: Path, split: str) -> list[MathDialExample]:
        """Deserialise examples from JSON cache."""
        with open(path) as f:
            data = json.load(f)
        examples = []
        for d in data:
            turns = [TutoringTurn(**t) for t in d.pop("turns", [])]
            d["turns"] = turns
            d["split"] = split
            examples.append(MathDialExample(**d))
        return examples

    def _print_summary(self):
        """Print a rich summary table of the loaded dataset."""
        table = Table(title="📚 MathDial Dataset Summary", show_header=True, header_style="bold cyan")
        table.add_column("Split",    style="cyan")
        table.add_column("Examples", justify="right")
        table.add_column("Avg turns / dialogue", justify="right")
        table.add_column("% Student solved", justify="right")

        for split_name, examples in [("train", self._train), ("test", self._test)]:
            if not examples:
                continue
            avg_turns = sum(len(e.turns) for e in examples) / len(examples)
            solved = sum(1 for e in examples if "yes" in e.self_correctness.lower())
            pct_solved = 100 * solved / len(examples)
            table.add_row(
                split_name,
                str(len(examples)),
                f"{avg_turns:.1f}",
                f"{pct_solved:.1f}%",
            )

        console.print(table)

        # Dialog act distribution
        if self._train:
            act_counts: dict[str, int] = {}
            for ex in self._train:
                for turn in ex.turns:
                    if turn.speaker == "Teacher":
                        act_counts[turn.dialog_act] = act_counts.get(turn.dialog_act, 0) + 1
            top_acts = sorted(act_counts.items(), key=lambda x: -x[1])[:8]
            console.print("\n[bold]Top teacher dialog acts (train):[/bold]")
            for act, cnt in top_acts:
                console.print(f"  {act:<20} {cnt:>5}")


# ── CLI helper ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = MathDialLoader()
    train, test = loader.load(max_train=500)

    console.print(f"\n[bold]Sample training example:[/bold]")
    ex = train[0]
    console.print(f"  Question:   {ex.question[:100]}...")
    console.print(f"  Answer:     {ex.ground_truth}")
    console.print(f"  Confusion:  {ex.teacher_described_confusion[:100]}...")
    console.print(f"  Turns:      {len(ex.turns)}")
    for t in ex.turns[:4]:
        console.print(f"    [{t.speaker}] ({t.dialog_act}) {t.text[:80]}")
