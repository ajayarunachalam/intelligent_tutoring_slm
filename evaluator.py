"""
evaluator.py
────────────
Evaluation suite for the Tutoring SLM.

Metrics computed:
  1. Automatic NLP:    BLEU-4, ROUGE-L, BERTScore (teacher response quality)
  2. Pedagogical:      solve-rate, hint-efficiency, answer-reveal rate
  3. Mastery tracking: trajectory correlation, convergence speed
  4. Dialog quality:   avg turns to solve, dialog act distribution

Reference: MathDial evaluation protocol (Macina et al., EMNLP 2023)
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import nltk
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from rich.console import Console
from rich.table import Table

from data_loader import MathDialExample
from inference import TutoringInferenceEngine, TutoringSession

# Download required NLTK data
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

console = Console()
logger = logging.getLogger(__name__)


# ── Result structures ──────────────────────────────────────────────────────

@dataclass
class TurnResult:
    """Result for a single teacher turn prediction."""
    qid: str
    turn_idx: int
    reference: str       # ground-truth teacher response
    hypothesis: str      # model-generated response
    dialog_act: str
    student_input: str


@dataclass
class SessionResult:
    """Result for a complete simulated session."""
    qid: str
    student_solved: bool
    answer_revealed: bool
    total_turns: int
    hint_count: int
    correction_count: int
    final_mastery: float
    mastery_trajectory: list[float]
    time_seconds: float


@dataclass
class EvaluationReport:
    """Aggregated evaluation report."""
    # NLP metrics
    bleu4: float              = 0.0
    rouge_l: float            = 0.0
    bert_score_f1: float      = 0.0

    # Pedagogical metrics
    solve_rate: float         = 0.0   # % of sessions where student solved
    reveal_rate: float        = 0.0   # % where answer was revealed (lower = better)
    avg_turns_to_solve: float = 0.0
    hint_efficiency: float    = 0.0   # hints per successful session

    # Dialog act distribution
    dialog_act_dist: dict     = field(default_factory=dict)

    # Mastery
    avg_final_mastery: float  = 0.0
    avg_mastery_gain: float   = 0.0

    # Session counts
    n_sessions: int           = 0
    n_turn_evals: int         = 0

    def print(self):
        """Print a rich formatted report."""
        table = Table(
            title="📊 Tutoring SLM Evaluation Report",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Metric", style="cyan", min_width=30)
        table.add_column("Value", justify="right", style="bold white")
        table.add_column("Notes", style="dim")

        def row(metric, value, notes=""):
            table.add_row(metric, str(value), notes)

        row("── NLP Metrics ──", "", "")
        row("BLEU-4",          f"{self.bleu4:.3f}",        "teacher response quality")
        row("ROUGE-L",         f"{self.rouge_l:.3f}",      "lexical overlap")
        row("BERTScore F1",    f"{self.bert_score_f1:.3f}","semantic similarity")

        row("── Pedagogical Metrics ──", "", "")
        row("Student solve rate",    f"{self.solve_rate*100:.1f}%",   "higher = better")
        row("Answer reveal rate",    f"{self.reveal_rate*100:.1f}%",  "lower = better")
        row("Avg turns to solve",    f"{self.avg_turns_to_solve:.1f}","lower = more efficient")
        row("Hint efficiency",       f"{self.hint_efficiency:.2f}",   "hints per solve")

        row("── Mastery Tracking ──", "", "")
        row("Avg final mastery",     f"{self.avg_final_mastery:.2f}", "0-1 scale")
        row("Avg mastery gain",      f"{self.avg_mastery_gain:.2f}",  "improvement over session")

        row("── Session Info ──", "", "")
        row("Sessions evaluated",    str(self.n_sessions),    "")
        row("Turn predictions",      str(self.n_turn_evals),  "")

        console.print(table)

        if self.dialog_act_dist:
            console.print("\n[bold]Dialog act distribution:[/bold]")
            total = sum(self.dialog_act_dist.values())
            for act, cnt in sorted(self.dialog_act_dist.items(), key=lambda x: -x[1]):
                pct = 100 * cnt / max(total, 1)
                console.print(f"  {act:<20} {cnt:>5}  ({pct:.1f}%)")

    def save(self, path: str = "./results/evaluation_report.json"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            d = self.__dict__.copy()
            json.dump(d, f, indent=2)
        console.print(f"[green]✓ Report saved → {path}[/green]")


# ── Evaluator ──────────────────────────────────────────────────────────────

class TutoringSLMEvaluator:
    """
    Evaluates the Tutoring SLM on:
      (A) Static turn-level: given context, predict teacher response → BLEU/ROUGE
      (B) Dynamic session: simulate full session with LLM student → solve rate

    Usage:
        evaluator = TutoringSLMEvaluator(engine)
        report = evaluator.evaluate(test_examples, mode="both")
    """

    def __init__(
        self,
        engine: TutoringInferenceEngine,
        student_model: str = "gemma4:latest",  # model simulating student errors
    ):
        self.engine        = engine
        self.student_model = student_model
        self._smoothing    = SmoothingFunction().method4

    # ── Public API ──────────────────────────────────────────────────────────

    def evaluate(
        self,
        test_examples: list[MathDialExample],
        mode: str = "both",   # "static" | "dynamic" | "both"
        max_examples: int = 50,
        verbose: bool = True,
    ) -> EvaluationReport:
        """
        Run evaluation and return an EvaluationReport.

        Args:
            test_examples: list of MathDialExample from test split
            mode:          evaluation mode
            max_examples:  cap for speed
            verbose:       print progress
        """
        examples = test_examples[:max_examples]
        report = EvaluationReport(n_sessions=len(examples))

        if mode in ("static", "both"):
            console.print(f"\n[bold cyan]── Static Turn-Level Evaluation ({len(examples)} examples) ──[/bold cyan]")
            turn_results = self._evaluate_static(examples)
            report = self._compute_nlp_metrics(turn_results, report)

        if mode in ("dynamic", "both"):
            console.print(f"\n[bold cyan]── Dynamic Session Evaluation ({len(examples)} examples) ──[/bold cyan]")
            session_results = self._evaluate_dynamic(examples)
            report = self._compute_pedagogical_metrics(session_results, report)

        if verbose:
            report.print()

        return report

    # ── Static evaluation ───────────────────────────────────────────────────

    def _evaluate_static(self, examples: list[MathDialExample]) -> list[TurnResult]:
        """
        For each teacher turn in the test set, generate a response
        given the prior context and compare to the reference.
        """
        turn_results = []
        for ex in examples:
            session = self.engine.new_session(
                question=ex.question,
                ground_truth=ex.ground_truth,
                student_profile=ex.student_profile,
                confusion=ex.teacher_described_confusion,
                session_id=ex.qid,
            )

            for i, turn in enumerate(ex.turns):
                if turn.speaker != "Teacher":
                    # Feed student turns into session history
                    if turn.speaker == "Student":
                        session.history.append({"role": "user", "content": turn.text})
                    continue

                # Generate model response given current context
                reference = turn.text
                # Build message without the reference
                messages = self.engine._build_messages(session)
                hypothesis = self.engine._generate(messages)

                turn_results.append(TurnResult(
                    qid=ex.qid,
                    turn_idx=i,
                    reference=reference,
                    hypothesis=hypothesis,
                    dialog_act=turn.dialog_act,
                    student_input=session.last_student_response,
                ))

                # Add reference (not hypothesis) to history for fair eval
                session.history.append({"role": "assistant", "content": reference})

        console.print(f"[green]✓ Static eval: {len(turn_results)} turn predictions[/green]")
        return turn_results

    def _compute_nlp_metrics(
        self,
        results: list[TurnResult],
        report: EvaluationReport,
    ) -> EvaluationReport:
        """Compute BLEU-4 and ROUGE-L."""
        if not results:
            return report

        report.n_turn_evals = len(results)

        # BLEU-4
        refs = [[nltk.word_tokenize(r.reference.lower())] for r in results]
        hyps = [nltk.word_tokenize(r.hypothesis.lower()) for r in results]
        try:
            report.bleu4 = corpus_bleu(refs, hyps, smoothing_function=self._smoothing)
        except Exception as e:
            logger.warning(f"BLEU computation failed: {e}")
            report.bleu4 = 0.0

        # ROUGE-L
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            rouge_scores = [
                scorer.score(r.reference, r.hypothesis)["rougeL"].fmeasure
                for r in results
            ]
            report.rouge_l = float(np.mean(rouge_scores))
        except ImportError:
            logger.warning("rouge-score not installed; skipping ROUGE-L")

        # BERTScore (optional — requires bert_score package)
        try:
            from bert_score import score as bert_score
            refs_str = [r.reference for r in results]
            hyps_str = [r.hypothesis for r in results]
            P, R, F1 = bert_score(hyps_str, refs_str, lang="en", verbose=False)
            report.bert_score_f1 = float(F1.mean())
        except Exception:
            report.bert_score_f1 = 0.0
            logger.info("BERTScore not available (pip install bert-score)")

        # Dialog act distribution
        for r in results:
            act = r.dialog_act or "unknown"
            report.dialog_act_dist[act] = report.dialog_act_dist.get(act, 0) + 1

        console.print(
            f"  BLEU-4: {report.bleu4:.3f}  "
            f"ROUGE-L: {report.rouge_l:.3f}  "
            f"BERTScore: {report.bert_score_f1:.3f}"
        )
        return report

    # ── Dynamic evaluation ──────────────────────────────────────────────────

    def _evaluate_dynamic(self, examples: list[MathDialExample]) -> list[SessionResult]:
        """
        Simulate complete tutoring sessions.
        An LLM acts as a confused student; the tutor engine tries to help.
        """
        session_results = []
        for ex in examples:
            result = self._simulate_session(ex)
            session_results.append(result)
            status = "✓ solved" if result.student_solved else "✗ unsolved"
            console.print(
                f"  [{status}] qid={ex.qid}  "
                f"turns={result.total_turns}  "
                f"mastery={result.final_mastery:.2f}  "
                f"time={result.time_seconds:.1f}s"
            )
        return session_results

    def _simulate_session(self, ex: MathDialExample, max_turns: int = 8) -> SessionResult:
        """Simulate one session: LLM student + tutor engine."""
        start = time.time()
        session = self.engine.new_session(
            question=ex.question,
            ground_truth=ex.ground_truth,
            student_profile=ex.student_profile,
            confusion=ex.teacher_described_confusion,
            session_id=ex.qid,
        )

        # Initial confused student response (from dataset)
        student_msg = (
            ex.student_incorrect_solution or
            f"I think the answer is {ex.student_incorrect_solution}, but I'm not sure."
        )

        for turn_num in range(max_turns):
            # Tutor responds
            tutor_response = self.engine.respond(session, student_msg)

            if session.student_solved or session.mastery_score >= 0.95:
                break

            # Simulate student response using LLM
            student_msg = self._simulate_student_response(
                question=ex.question,
                incorrect_solution=ex.student_incorrect_solution,
                profile=ex.student_profile,
                tutor_message=tutor_response,
                turn_num=turn_num,
            )

        elapsed = time.time() - start
        return SessionResult(
            qid=ex.qid,
            student_solved=session.student_solved,
            answer_revealed=session.answer_revealed,
            total_turns=session.turn_count,
            hint_count=session.hint_count,
            correction_count=session.correction_count,
            final_mastery=session.mastery_score,
            mastery_trajectory=session.mastery_history,
            time_seconds=elapsed,
        )

    def _simulate_student_response(
        self,
        question: str,
        incorrect_solution: str,
        profile: str,
        tutor_message: str,
        turn_num: int,
    ) -> str:
        """Use Ollama to simulate a confused 7th-grade student."""
        prompt = f"""You are a 7th-grade student working on this math problem:
{question}

Your initial (wrong) answer was: {incorrect_solution}
Your misconception profile: {profile}

The tutor just said: "{tutor_message}"

Respond as a realistic confused student would:
- Turn {turn_num}: Start mostly confused, gradually improve as the tutor helps
- Ask follow-up questions if confused
- Show partial understanding if you've received hints
- Keep response to 1-2 sentences
- If the answer is now clear after the tutor's help, show you understand

Student response:"""

        try:
            import ollama
            result = ollama.generate(
                model=self.student_model,
                prompt=prompt,
                options={"temperature": 0.8, "num_predict": 80},
            )
            return result["response"].strip()
        except Exception:
            # Fallback: use the original incorrect solution
            return f"I'm still confused. {incorrect_solution}"

    def _compute_pedagogical_metrics(
        self,
        results: list[SessionResult],
        report: EvaluationReport,
    ) -> EvaluationReport:
        """Aggregate session-level pedagogical metrics."""
        if not results:
            return report

        n = len(results)
        solved = [r for r in results if r.student_solved]
        revealed = [r for r in results if r.answer_revealed]

        report.solve_rate         = len(solved) / n
        report.reveal_rate        = len(revealed) / n
        report.avg_turns_to_solve = np.mean([r.total_turns for r in solved]) if solved else 0.0
        report.hint_efficiency    = (
            np.mean([r.hint_count for r in solved]) if solved else 0.0
        )
        report.avg_final_mastery  = float(np.mean([r.final_mastery for r in results]))
        report.avg_mastery_gain   = float(np.mean([
            r.mastery_history[-1] - r.mastery_history[0]
            if r.mastery_history else 0.0
            for r in results
        ]))

        return report


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from data_loader import MathDialLoader
    from inference import TutoringInferenceEngine

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="gemma4:latest")
    parser.add_argument("--mode",     default="both", choices=["static", "dynamic", "both"])
    parser.add_argument("--max",      type=int, default=20)
    parser.add_argument("--out",      default="./results/eval_report.json")
    args = parser.parse_args()

    loader = MathDialLoader()
    _, test_ex = loader.load(max_test=args.max)

    engine = TutoringInferenceEngine(model=args.model)
    evaluator = TutoringSLMEvaluator(engine, student_model=args.model)
    report = evaluator.evaluate(test_ex, mode=args.mode, max_examples=args.max)
    report.save(args.out)
