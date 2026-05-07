"""
app.py
──────
Interactive CLI tutoring application.
Allows a human to play the "student" role and experience the Tutoring SLM.

Also includes a demo mode that replays a MathDial session with the
model in teacher role, showing the dialogue with rich formatting.

Usage:
    python app.py                          # interactive session (pick random problem)
    python app.py --qid 123               # specific MathDial problem
    python app.py --model gemma4:31b       # use larger model
    python app.py --demo                   # replay a test session (automated)
    python app.py --custom "If ..."        # supply your own math problem
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.table import Table

from data_loader import MathDialLoader, MathDialExample
from inference import TutoringInferenceEngine, TutoringSession

console = Console()


# ── Styling helpers ────────────────────────────────────────────────────────

def print_header():
    console.print(Panel(
        "[bold cyan]🎓  Tutoring SLM — DGX Spark Proof of Concept[/bold cyan]\n"
        "[dim]Powered by Gemma 4 via Ollama  ·  MathDial Dataset[/dim]",
        border_style="cyan",
        padding=(1, 4),
    ))


def print_problem(question: str, session_id: str):
    console.print(Rule(f"[bold]Problem  [dim](session: {session_id})[/dim][/bold]"))
    console.print(Panel(
        f"[bold white]{question}[/bold white]",
        border_style="yellow",
        title="[yellow]📐 Math Problem[/yellow]",
        padding=(0, 2),
    ))


def print_tutor(text: str):
    console.print(Panel(
        Markdown(text),
        border_style="green",
        title="[green]🎓 Tutor[/green]",
        padding=(0, 2),
    ))


def print_student(text: str):
    console.print(Panel(
        f"[white]{text}[/white]",
        border_style="blue",
        title="[blue]👤 You (Student)[/blue]",
        padding=(0, 2),
    ))


def print_mastery_bar(score: float):
    """Print a visual mastery progress bar."""
    pct = int(score * 20)
    bar = "█" * pct + "░" * (20 - pct)
    color = "red" if score < 0.4 else "yellow" if score < 0.7 else "green"
    console.print(f"  [dim]Mastery:[/dim] [{color}]{bar}[/{color}] [{color}]{score*100:.0f}%[/{color}]")


def print_session_summary(session: TutoringSession):
    """Print end-of-session statistics."""
    console.print(Rule("[bold]Session Summary[/bold]"))
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("", style="dim")
    table.add_column("", style="bold white")

    table.add_row("Total turns",      str(session.turn_count))
    table.add_row("Hints given",      str(session.hint_count))
    table.add_row("Corrections",      str(session.correction_count))
    table.add_row("Student solved",   "✅ Yes" if session.student_solved else "❌ No")
    table.add_row("Answer revealed",  "⚠️  Yes" if session.answer_revealed else "✓  No")
    table.add_row("Final mastery",    f"{session.mastery_score*100:.0f}%")

    console.print(table)
    if session.student_solved:
        console.print("\n[bold green]🎉 Great work! The student solved the problem![/bold green]")
    elif session.answer_revealed:
        console.print("\n[yellow]📖 The tutor revealed the answer after many attempts.[/yellow]")
    else:
        console.print("\n[dim]Session ended. Keep practising![/dim]")


# ── Session modes ──────────────────────────────────────────────────────────

def run_interactive_session(
    engine: TutoringInferenceEngine,
    example: MathDialExample,
):
    """
    Interactive session: human plays the student.
    """
    print_problem(example.question, example.qid)
    console.print("\n[dim]You are the student. Type your answers below.")
    console.print("[dim]Commands: 'quit' to end, 'hint' to ask for a hint, 'answer' to reveal answer\n")

    session = engine.new_session(
        question=example.question,
        ground_truth=example.ground_truth,
        student_profile=example.student_profile,
        confusion=example.teacher_described_confusion,
        session_id=example.qid,
    )

    # Opening tutor message
    opening_messages = engine._build_messages(session)
    opening_messages.append({
        "role": "user",
        "content": "Hello! I'm ready to work on this problem."
    })
    session.history.append({"role": "user", "content": "Hello! I'm ready to work on this problem."})
    with Progress(SpinnerColumn(), TextColumn("[green]Tutor is thinking..."), transient=True) as p:
        p.add_task("")
        opener = engine._generate(opening_messages)
    session.add_turn("assistant", opener)
    print_tutor(opener)

    while True:
        try:
            student_input = console.input("\n[bold blue]You:[/bold blue] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Session ended.[/dim]")
            break

        if not student_input:
            continue

        if student_input.lower() in ("quit", "exit", "q"):
            break

        if student_input.lower() == "hint":
            student_input = "Can you give me a hint?"

        if student_input.lower() == "answer":
            student_input = "I give up, what's the answer?"

        print_student(student_input)

        # Generate tutor response
        with Progress(SpinnerColumn(), TextColumn("[green]Tutor is thinking..."), transient=True) as p:
            task = p.add_task("")
            response = engine.respond(session, student_input)

        print_tutor(response)

        if engine.mastery_tracking:
            print_mastery_bar(session.mastery_score)

        if session.student_solved:
            console.print("\n[bold green]✅ Excellent! You solved it![/bold green]")
            break

        if session.answer_revealed:
            console.print("\n[yellow]💡 The answer has been revealed.[/yellow]")
            break

    print_session_summary(session)
    return session


def run_demo_session(
    engine: TutoringInferenceEngine,
    example: MathDialExample,
    delay: float = 1.0,
):
    """
    Demo mode: replay the MathDial conversation automatically,
    with the model generating teacher responses.
    """
    console.print(Rule("[bold cyan]Demo Mode — Simulated Session[/bold cyan]"))
    print_problem(example.question, example.qid)
    console.print(f"\n[dim]Correct answer: {example.ground_truth}[/dim]")
    console.print(f"[dim]Student confusion: {example.teacher_described_confusion}[/dim]\n")

    session = engine.new_session(
        question=example.question,
        ground_truth=example.ground_truth,
        student_profile=example.student_profile,
        confusion=example.teacher_described_confusion,
        session_id=example.qid,
    )

    # Replay reference conversation vs model-generated
    ref_turns = example.turns
    console.print(f"[dim]Replaying {len(ref_turns)} reference turns + model responses...[/dim]\n")

    for i, turn in enumerate(ref_turns[:10]):  # cap at 10 turns for demo
        time.sleep(delay)

        if turn.speaker == "Student":
            print_student(f"[Reference] {turn.text}")
            # Feed into session
            session.history.append({"role": "user", "content": turn.text})
            session.last_student_response = turn.text
            session.turn_count += 1
            if engine.mastery_tracking:
                engine._update_mastery(session)

        elif turn.speaker == "Teacher":
            # Show reference response
            console.print(f"[dim]Reference teacher: {turn.text[:100]}...[/dim]" if len(turn.text) > 100 else f"[dim]Reference teacher: {turn.text}[/dim]")

            # Generate model response
            with Progress(SpinnerColumn(), TextColumn("[green]Model generating..."), transient=True) as p:
                p.add_task("")
                messages = engine._build_messages(session)
                model_resp = engine._generate(messages)

            print_tutor(f"[Model] {model_resp}")
            session.add_turn("assistant", model_resp)
            if engine.mastery_tracking:
                print_mastery_bar(session.mastery_score)

    print_session_summary(session)
    return session


# ── CLI entry point ────────────────────────────────────────────────────────

@click.command()
@click.option("--model",   default="gemma4:latest",    show_default=True, help="Ollama model name")
@click.option("--qid",     default=None,               help="Specific MathDial problem ID")
@click.option("--demo",    is_flag=True,               help="Run automated demo (no human input)")
@click.option("--custom",  default=None,               help="Provide your own math problem text")
@click.option("--answer",  default=None,               help="Answer for custom problem (used with --custom)")
@click.option("--delay",   default=0.5, type=float,    help="Delay between demo turns (seconds)")
@click.option("--max-tokens", default=300, type=int,   help="Max tokens per tutor response")
@click.option("--no-safety", is_flag=True,             help="Disable safety filter (faster)")
@click.option("--no-mastery", is_flag=True,            help="Disable mastery tracking (faster)")
@click.option("--save-session", default=None,          help="Save session JSON to file")
def main(model, qid, demo, custom, answer, delay, max_tokens, no_safety, no_mastery, save_session):
    """
    🎓 Tutoring SLM — Interactive Math Tutoring powered by Gemma 4 on DGX Spark
    """
    print_header()

    # ── Initialise engine ──
    engine = TutoringInferenceEngine(
        model=model,
        max_tokens=max_tokens,
        safety_check=not no_safety,
        mastery_tracking=not no_mastery,
    )

    # ── Pick a problem ──
    if custom:
        # Custom problem supplied by user
        if not answer:
            answer = console.input("[yellow]Enter the correct answer for your problem: [/yellow]")
        from data_loader import MathDialExample
        example = MathDialExample(
            qid="custom_001",
            scenario=1,
            question=custom,
            ground_truth=answer,
            student_incorrect_solution="I don't know",
            student_profile="general student",
            teacher_described_confusion="Unknown",
            self_correctness="",
            self_typical_confusion=3,
            self_typical_interactions=3,
        )
    else:
        # Load from MathDial
        loader = MathDialLoader()
        _, test_examples = loader.load(max_test=100)

        if qid:
            matches = [e for e in test_examples if e.qid == qid]
            if not matches:
                console.print(f"[red]Problem {qid} not found. Available IDs: {[e.qid for e in test_examples[:5]]}...[/red]")
                sys.exit(1)
            example = matches[0]
        else:
            example = random.choice(test_examples)
            console.print(f"[dim]Randomly selected problem: {example.qid}[/dim]")

    # ── Run session ──
    if demo:
        session = run_demo_session(engine, example, delay=delay)
    else:
        session = run_interactive_session(engine, example)

    # ── Save session ──
    if save_session:
        path = Path(save_session)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(session.to_dict(), f, indent=2)
        console.print(f"[green]Session saved → {save_session}[/green]")


if __name__ == "__main__":
    main()
