"""
pipeline.py
───────────
End-to-end orchestrator for the Tutoring SLM proof of concept.

Stages:
  1. Data      — Download MathDial, preprocess into Gemma 4 chat format
  2. Fine-tune — QLoRA SFT on DGX Spark (gemma4:31b or gemma4:latest)
  3. Evaluate  — BLEU, solve rate, pedagogical metrics
  4. Demo      — Interactive session or auto-demo

Modes:
  full      — run all 4 stages
  data      — stage 1 only
  finetune  — stages 1-2
  evaluate  — stages 1, 3
  demo      — stages 1, 4 (uses base Ollama model, no fine-tuning needed)
  quick     — compressed smoke-test (100 train samples, 1 epoch)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# ── Constants ──────────────────────────────────────────────────────────────

DEFAULT_TRAIN_SIZE  = 2400   # ~84% of MathDial train set
DEFAULT_TEST_SIZE   = 100
QUICK_TRAIN_SIZE    = 150
QUICK_TEST_SIZE     = 30
PROCESSED_DATA_DIR  = "./data/processed"
RESULTS_DIR         = "./results"


# ── Stage functions ────────────────────────────────────────────────────────

def stage_data(args) -> tuple:
    """Stage 1: Download and preprocess MathDial."""
    console.print(Rule("[bold cyan]Stage 1 — Data: MathDial Download & Preprocessing[/bold cyan]"))

    from data_loader import MathDialLoader
    from preprocessor import MathDialPreprocessor

    max_train = QUICK_TRAIN_SIZE if args.quick else DEFAULT_TRAIN_SIZE
    max_test  = QUICK_TEST_SIZE  if args.quick else DEFAULT_TEST_SIZE

    # 1a. Load raw data
    loader = MathDialLoader(cache_dir="./data/mathdial")
    train_ex, test_ex = loader.load(max_train=max_train, max_test=max_test)

    # 1b. Preprocess
    proc = MathDialPreprocessor(
        max_context_turns=10,
        min_teacher_turns=2,
        include_dialog_acts=True,
        gemma_format=True,
    )
    train_ds, test_ds = proc.process_to_hf(train_ex, test_ex)

    # 1c. Quality report
    train_samples = proc.process(train_ex)
    proc.quality_report(train_samples)

    # 1d. Save to disk
    proc.save(train_ds, test_ds, output_dir=PROCESSED_DATA_DIR)

    console.print(f"\n[green]✓ Stage 1 complete: {len(train_ds)} train / {len(test_ds)} test samples[/green]\n")
    return train_ex, test_ex, train_ds, test_ds


def stage_finetune(args, train_ds, test_ds):
    """Stage 2: QLoRA fine-tuning on DGX Spark."""
    console.print(Rule("[bold cyan]Stage 2 — Fine-tuning: QLoRA SFT on DGX Spark[/bold cyan]"))

    from fine_tuner import TutoringSLMFineTuner, TrainingConfig, LoRAConfig

    # Determine model name
    model_name = args.base_model

    cfg = TrainingConfig(
        base_model_name=model_name,
        num_train_epochs=1 if args.quick else 3,
        per_device_train_batch_size=2 if args.quick else 4,
        gradient_accumulation_steps=2 if args.quick else 4,
        output_dir="./checkpoints/tutoring_slm",
        adapter_dir="./models/tutoring_slm_adapter",
        final_model_dir="./models/tutoring_slm_final",
        report_to="wandb" if args.wandb else "none",
    )

    lora_cfg = LoRAConfig(r=16, lora_alpha=32)

    ft = TutoringSLMFineTuner(training_config=cfg, lora_config=lora_cfg)
    ft.train(train_ds, test_ds)
    ft.save()

    if args.merge:
        ft.merge_and_save()

    console.print(f"\n[green]✓ Stage 2 complete.[/green]\n")
    return ft


def stage_evaluate(args, test_ex):
    """Stage 3: Evaluate the Tutoring SLM."""
    console.print(Rule("[bold cyan]Stage 3 — Evaluation[/bold cyan]"))

    from inference import TutoringInferenceEngine
    from evaluator import TutoringSLMEvaluator

    # Use fine-tuned model if available, else base Ollama model
    model = args.inference_model or args.ollama_model

    engine = TutoringInferenceEngine(
        model=model,
        safety_check=True,
        mastery_tracking=True,
    )

    evaluator = TutoringSLMEvaluator(engine, student_model=model)
    max_eval = 15 if args.quick else 50

    report = evaluator.evaluate(
        test_ex,
        mode=args.eval_mode,
        max_examples=max_eval,
    )

    # Save report
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    report_path = f"{RESULTS_DIR}/eval_report.json"
    report.save(report_path)

    console.print(f"\n[green]✓ Stage 3 complete. Report → {report_path}[/green]\n")
    return report


def stage_demo(args, test_ex):
    """Stage 4: Interactive demo session."""
    console.print(Rule("[bold cyan]Stage 4 — Interactive Demo[/bold cyan]"))

    import random
    from app import run_interactive_session, run_demo_session, print_header
    from inference import TutoringInferenceEngine

    model = args.inference_model or args.ollama_model
    engine = TutoringInferenceEngine(
        model=model,
        safety_check=True,
        mastery_tracking=True,
    )

    # Pick a test problem
    example = random.choice(test_ex[:50])

    print_header()

    if args.auto_demo:
        run_demo_session(engine, example, delay=args.demo_delay)
    else:
        run_interactive_session(engine, example)

    console.print(f"\n[green]✓ Stage 4 complete.[/green]\n")


# ── Summary printer ────────────────────────────────────────────────────────

def print_pipeline_summary(args, elapsed: float):
    """Print a final summary of the pipeline run."""
    console.print(Rule("[bold]Pipeline Complete[/bold]"))

    table = Table(show_header=False, box=None, padding=(0, 3))
    table.add_column("", style="dim")
    table.add_column("", style="bold white")

    table.add_row("Mode",            args.mode)
    table.add_row("Ollama model",    args.ollama_model)
    table.add_row("Base model (HF)", args.base_model)
    table.add_row("Quick run",       "Yes" if args.quick else "No")
    table.add_row("Total time",      f"{elapsed:.1f}s  ({elapsed/60:.1f} min)")
    table.add_row("Processed data",  PROCESSED_DATA_DIR)
    table.add_row("Results",         RESULTS_DIR)

    console.print(table)
    console.print(Panel(
        "[bold green]🎓 Tutoring SLM Proof of Concept complete![/bold green]\n\n"
        "Next steps:\n"
        "  • Review evaluation report in ./results/eval_report.json\n"
        "  • Run interactive demo:  [cyan]python pipeline.py --mode demo[/cyan]\n"
        "  • Try custom problem:    [cyan]python app.py --custom 'If a train...' --answer 42[/cyan]\n"
        "  • Scale up:              [cyan]python pipeline.py --mode full --base-model gemma4:31b[/cyan]",
        border_style="green",
        padding=(1, 3),
    ))


# ── Main orchestrator ──────────────────────────────────────────────────────

def run_pipeline(args):
    """Run the pipeline according to args.mode."""
    start = time.time()
    mode  = args.mode

    console.print(Panel(
        f"[bold cyan]🚀 Tutoring SLM Pipeline[/bold cyan]\n"
        f"[dim]Mode: {mode}  ·  Model: {args.ollama_model}  ·  Quick: {args.quick}[/dim]",
        border_style="cyan",
        padding=(1, 4),
    ))

    # ── Stage 1: Always run data stage ──────────────────────────────────────
    if mode in ("full", "data", "finetune", "evaluate", "quick"):
        train_ex, test_ex, train_ds, test_ds = stage_data(args)
    elif mode == "demo":
        # Demo just needs test examples (raw, not preprocessed)
        from data_loader import MathDialLoader
        loader = MathDialLoader()
        _, test_ex = loader.load(max_test=50)
        train_ds = test_ds = None

    # ── Stage 2: Fine-tuning ────────────────────────────────────────────────
    if mode in ("full", "finetune", "quick"):
        stage_finetune(args, train_ds, test_ds)

    # ── Stage 3: Evaluation ─────────────────────────────────────────────────
    if mode in ("full", "evaluate", "quick"):
        stage_evaluate(args, test_ex)

    # ── Stage 4: Demo ───────────────────────────────────────────────────────
    if mode in ("full", "demo"):
        stage_demo(args, test_ex)

    elapsed = time.time() - start
    print_pipeline_summary(args, elapsed)


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Tutoring SLM End-to-End Pipeline (DGX Spark PoC)",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python pipeline.py --mode demo                     # Interactive demo (no training)
  python pipeline.py --mode quick                    # Smoke-test all stages fast
  python pipeline.py --mode evaluate                 # Evaluate only
  python pipeline.py --mode full --base-model google/gemma-2-9b-it
  python pipeline.py --mode full --base-model google/gemma-2-27b-it --merge
        """
    )

    parser.add_argument(
        "--mode",
        choices=["full", "data", "finetune", "evaluate", "demo", "quick"],
        default="demo",
        help=(
            "full     = data + finetune + evaluate + demo\n"
            "data     = download & preprocess only\n"
            "finetune = data + QLoRA fine-tune\n"
            "evaluate = data + evaluate (no training)\n"
            "demo     = interactive tutoring demo\n"
            "quick    = compressed smoke-test of all stages"
        ),
    )
    parser.add_argument(
        "--ollama-model",
        default="gemma4:latest",
        help="Ollama model for inference (already on DGX Spark)",
    )
    parser.add_argument(
        "--base-model",
        default="google/gemma-2-9b-it",
        help="HuggingFace model ID for fine-tuning (or local path)",
    )
    parser.add_argument(
        "--inference-model",
        default=None,
        help="Ollama model to use for evaluation (defaults to --ollama-model)",
    )
    parser.add_argument(
        "--eval-mode",
        choices=["static", "dynamic", "both"],
        default="both",
        help="Evaluation mode: static (BLEU/ROUGE) / dynamic (simulate sessions) / both",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick/smoke-test run: small dataset, 1 epoch, fast eval",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="After fine-tuning, merge LoRA adapters into the base model",
    )
    parser.add_argument(
        "--auto-demo",
        action="store_true",
        help="Run demo in automated mode (no keyboard input required)",
    )
    parser.add_argument(
        "--demo-delay",
        type=float,
        default=0.8,
        help="Delay between demo turns in seconds",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging during fine-tuning",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
