from __future__ import annotations
 
import argparse
import json
import random
import time
from pathlib import Path
 
import gradio as gr
 
# ── Lazy imports (allow app to load even if inference deps not installed) ──
try:
    from inference import TutoringInferenceEngine, TutoringSession
    from data_loader import MathDialLoader, MathDialExample
    DEPS_OK = True
except ImportError as e:
    DEPS_OK = False
    IMPORT_ERROR = str(e)
 
# ── Global state ──────────────────────────────────────────────────────────
_engine: TutoringInferenceEngine | None = None
_examples: list[MathDialExample] = []
_sessions: dict[str, TutoringSession] = {}   # session_id → session
 
 
# ── Colour / style constants (CSS injected into Gradio) ───────────────────
CUSTOM_CSS = """
/* ── Google Font imports ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
 
/* ── Root variables ── */
:root {
    --bg-deep:      #0C0F1A;
    --bg-card:      #131729;
    --bg-elevated:  #1A2035;
    --border:       #252D45;
    --accent-teal:  #00D4AA;
    --accent-amber: #F59E0B;
    --accent-red:   #EF4444;
    --accent-blue:  #60A5FA;
    --text-primary: #F0F4FF;
    --text-muted:   #6B7A9A;
    --text-dim:     #3D4A6A;
    --tutor-bg:     #0D2E26;
    --tutor-border: #00D4AA33;
    --student-bg:   #1A1F35;
    --student-border:#60A5FA33;
    --radius:       12px;
    --radius-sm:    8px;
    --font-display: 'DM Serif Display', serif;
    --font-body:    'DM Sans', sans-serif;
    --font-mono:    'JetBrains Mono', monospace;
}
 
/* ── Global resets ── */
* { box-sizing: border-box; }
body, .gradio-container {
    background: var(--bg-deep) !important;
    font-family: var(--font-body) !important;
    color: var(--text-primary) !important;
}
 
/* ── Header ── */
.app-header {
    background: linear-gradient(135deg, #0C0F1A 0%, #0D1F3C 50%, #0C1A14 100%);
    border-bottom: 1px solid var(--border);
    padding: 28px 40px 24px;
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, #00D4AA18 0%, transparent 70%);
    pointer-events: none;
}
.header-title {
    font-family: var(--font-display) !important;
    font-size: 2rem !important;
    font-weight: 400 !important;
    color: var(--text-primary) !important;
    margin: 0 !important;
    letter-spacing: -0.02em;
}
.header-title span { color: var(--accent-teal); }
.header-subtitle {
    font-size: 0.82rem;
    color: var(--text-muted);
    margin-top: 4px;
    font-weight: 300;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.header-badge {
    display: inline-block;
    background: #00D4AA18;
    border: 1px solid #00D4AA44;
    color: var(--accent-teal);
    font-size: 0.72rem;
    font-family: var(--font-mono);
    padding: 3px 10px;
    border-radius: 4px;
    margin-top: 10px;
}
 
/* ── Panels ── */
.panel {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
    margin-bottom: 12px;
}
.panel-title {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.panel-title::before {
    content: '';
    display: inline-block;
    width: 3px; height: 14px;
    border-radius: 2px;
    background: var(--accent-teal);
}
 
/* ── Problem card ── */
.problem-card {
    background: linear-gradient(135deg, #0F1E3A, #0D2A20);
    border: 1px solid #1E3A5F;
    border-left: 3px solid var(--accent-amber);
    border-radius: var(--radius-sm);
    padding: 16px 18px;
    font-size: 0.92rem;
    line-height: 1.7;
    color: var(--text-primary);
    margin: 8px 0;
}
 
/* ── Chat bubbles ── */
.message.svelte-1s78gho {
    font-family: var(--font-body) !important;
}
/* Tutor bubble */
.message.bot, div[data-testid="bot"] {
    background: var(--tutor-bg) !important;
    border: 1px solid var(--tutor-border) !important;
    border-radius: 0 var(--radius) var(--radius) var(--radius) !important;
    color: var(--text-primary) !important;
    font-size: 0.9rem !important;
}
/* Student bubble */
.message.user, div[data-testid="user"] {
    background: var(--student-bg) !important;
    border: 1px solid var(--student-border) !important;
    border-radius: var(--radius) 0 var(--radius) var(--radius) !important;
    color: var(--text-primary) !important;
    font-size: 0.9rem !important;
}
 
/* ── Mastery bar ── */
.mastery-container {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 14px 18px;
}
.mastery-label {
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 8px;
}
.mastery-bar-track {
    background: var(--bg-deep);
    border-radius: 4px;
    height: 8px;
    width: 100%;
    overflow: hidden;
}
.mastery-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
}
 
/* ── Stat pills ── */
.stat-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 10px;
}
.stat-pill {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 0.78rem;
    font-family: var(--font-mono);
    color: var(--text-muted);
}
.stat-pill strong { color: var(--text-primary); }
 
/* ── Buttons ── */
.btn-primary {
    background: linear-gradient(135deg, #00D4AA, #00A889) !important;
    color: #0A1A14 !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    font-family: var(--font-body) !important;
    letter-spacing: 0.02em !important;
    transition: opacity 0.2s, transform 0.15s !important;
}
.btn-primary:hover { opacity: 0.9 !important; transform: translateY(-1px) !important; }
 
.btn-secondary {
    background: var(--bg-elevated) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    font-family: var(--font-body) !important;
}
.btn-secondary:hover { border-color: var(--accent-teal) !important; }
 
/* ── Tabs ── */
.tab-nav button {
    font-family: var(--font-body) !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.04em !important;
    color: var(--text-muted) !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 10px 16px !important;
    transition: all 0.2s !important;
}
.tab-nav button.selected, .tab-nav button:hover {
    color: var(--accent-teal) !important;
    border-bottom-color: var(--accent-teal) !important;
}
 
/* ── Textboxes ── */
textarea, input[type="text"] {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
    font-size: 0.9rem !important;
}
textarea:focus, input:focus {
    border-color: var(--accent-teal) !important;
    box-shadow: 0 0 0 3px #00D4AA14 !important;
    outline: none !important;
}
 
/* ── Dropdowns ── */
.gr-dropdown select, select {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: var(--radius-sm) !important;
}
 
/* ── Status pill ── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.75rem;
    font-family: var(--font-mono);
    padding: 4px 10px;
    border-radius: 20px;
}
.status-online  { background:#00D4AA14; border:1px solid #00D4AA44; color:var(--accent-teal); }
.status-offline { background:#EF444414; border:1px solid #EF444444; color:var(--accent-red);  }
.status-dot { width:6px; height:6px; border-radius:50%; background:currentColor; }
 
/* ── Accordion ── */
.gr-accordion {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
}
 
/* ── Scrollbars ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-dim); }
"""
 
# ── HTML components ────────────────────────────────────────────────────────
 
def make_header_html(model_name: str) -> str:
    return f"""
<div class="app-header">
  <h1 class="header-title">🎓 Tutoring <span>SLM</span></h1>
  <div class="header-subtitle">DGX Spark · Gemma 4 · MathDial · Proof of Concept</div>
  <div class="header-badge">model: {model_name}</div>
</div>"""
 
 
def make_problem_html(question: str, qid: str = "") -> str:
    return f"""
<div class="panel">
  <div class="panel-title">Active Problem {f'· <span style="font-family:var(--font-mono);color:var(--accent-amber)">{qid}</span>' if qid else ''}</div>
  <div class="problem-card">{question}</div>
</div>"""
 
 
def make_stats_html(session: "TutoringSession | None") -> str:
    if session is None:
        return """<div class="panel" style="color:var(--text-dim);font-size:0.82rem;">
            No active session. Start a conversation to see live stats.
        </div>"""
 
    mastery = session.mastery_score
    pct = int(mastery * 100)
    bar_color = (
        "#EF4444" if pct < 35 else
        "#F59E0B" if pct < 65 else
        "#00D4AA"
    )
    status_icon = "✅" if session.student_solved else ("⚠️" if session.answer_revealed else "🔄")
    status_text = "Solved!" if session.student_solved else ("Answer revealed" if session.answer_revealed else "In progress")
 
    return f"""
<div class="panel">
  <div class="panel-title">Session Analytics</div>
  <div class="mastery-container">
    <div class="mastery-label">Mastery estimate</div>
    <div class="mastery-bar-track">
      <div class="mastery-bar-fill" style="width:{pct}%; background:{bar_color};"></div>
    </div>
    <div style="display:flex;justify-content:space-between;margin-top:6px;">
      <span style="font-size:0.72rem;color:var(--text-muted)">0%</span>
      <span style="font-size:0.82rem;font-family:var(--font-mono);color:{bar_color};font-weight:600">{pct}%</span>
      <span style="font-size:0.72rem;color:var(--text-muted)">100%</span>
    </div>
  </div>
  <div class="stat-row" style="margin-top:12px;">
    <div class="stat-pill"><strong>{session.turn_count}</strong>&nbsp;turns</div>
    <div class="stat-pill"><strong>{session.hint_count}</strong>&nbsp;hints</div>
    <div class="stat-pill"><strong>{session.correction_count}</strong>&nbsp;corrections</div>
    <div class="stat-pill">{status_icon}&nbsp;<strong>{status_text}</strong></div>
  </div>
</div>"""
 
 
# ── Engine helpers ─────────────────────────────────────────────────────────
 
def get_engine(model: str) -> "TutoringInferenceEngine":
    global _engine
    if _engine is None or _engine.model != model:
        _engine = TutoringInferenceEngine(
            model=model,
            safety_check=True,
            mastery_tracking=True,
        )
    return _engine
 
 
def load_examples(n: int = 200) -> list["MathDialExample"]:
    global _examples
    if not _examples:
        try:
            loader = MathDialLoader()
            _, test_ex = loader.load(max_test=n)
            _examples = test_ex
        except Exception as e:
            _examples = []
    return _examples
 
 
def get_example_choices(examples: list) -> list[str]:
    return [f"[{e.qid}] {e.question[:80]}…" for e in examples]
 
 
# ── Core chat function ─────────────────────────────────────────────────────
 
def chat(
    message: str,
    history: list,
    session_state: dict,
    model_name: str,
) -> tuple[list, dict, str, str]:
    """
    Main chat handler.
    Returns: (updated_history, updated_session_state, problem_html, stats_html)
    """
    if not DEPS_OK:
        history.append((message, f"⚠️ Dependencies not installed: `{IMPORT_ERROR}`\n\nRun: `pip install -r requirements.txt`"))
        return history, session_state, "", ""
 
    if not message.strip():
        return history, session_state, "", ""
 
    engine = get_engine(model_name)
 
    # Retrieve or create session
    session_id = session_state.get("session_id")
    session: TutoringSession | None = _sessions.get(session_id)
 
    if session is None or not session_id:
        # No active session — give a helpful nudge
        history.append((
            message,
            "👋 Welcome! Please select a problem from the **Problem Setup** tab first, "
            "then come back here to start your tutoring session."
        ))
        return history, session_state, "", make_stats_html(None)
 
    # Generate tutor response
    try:
        response = engine.respond(session, message)
    except Exception as e:
        response = f"⚠️ Error generating response: {e}\n\nPlease check that Ollama is running: `ollama serve`"
 
    history.append((message, response))
 
    problem_html = make_problem_html(session.question, session.session_id)
    stats_html   = make_stats_html(session)
 
    # Check if session is complete
    if session.student_solved:
        history.append((None, "🎉 **Excellent work! You solved it!** Feel free to try another problem from the Problem Setup tab."))
    elif session.answer_revealed:
        history.append((None, "💡 The full solution has been walked through. Try another problem to keep practising!"))
 
    return history, session_state, problem_html, stats_html
 
 
def start_session(
    problem_choice: str,
    custom_problem: str,
    custom_answer: str,
    model_name: str,
    session_state: dict,
) -> tuple[list, dict, str, str, str]:
    """
    Start a new tutoring session.
    Returns: (history, session_state, problem_html, stats_html, status_msg)
    """
    if not DEPS_OK:
        return [], session_state, "", "", f"❌ Missing deps: {IMPORT_ERROR}"
 
    engine = get_engine(model_name)
    examples = load_examples()
 
    # Determine problem source
    if custom_problem.strip():
        question    = custom_problem.strip()
        answer      = custom_answer.strip() or "See working below"
        confusion   = "Unknown — custom problem"
        profile     = "General student"
        qid         = f"custom_{int(time.time())}"
    elif problem_choice and examples:
        # Find selected example
        idx = next(
            (i for i, e in enumerate(examples)
             if problem_choice.startswith(f"[{e.qid}]")),
            0
        )
        ex = examples[idx]
        question  = ex.question
        answer    = ex.ground_truth
        confusion = ex.teacher_described_confusion
        profile   = ex.student_profile
        qid       = ex.qid
    else:
        # Random example
        if not examples:
            return [], session_state, "", "", "❌ No examples loaded. Check your data directory."
        ex = random.choice(examples)
        question  = ex.question
        answer    = ex.ground_truth
        confusion = ex.teacher_described_confusion
        profile   = ex.student_profile
        qid       = ex.qid
 
    # Create session
    session = engine.new_session(
        question=question,
        ground_truth=answer,
        student_profile=profile,
        confusion=confusion,
        session_id=qid,
    )
    _sessions[qid] = session
    session_state["session_id"] = qid
 
    # Generate opening tutor message
    opening_messages = engine._build_messages(session)
    opening_messages.append({
        "role": "user",
        "content": "Hello! I'm ready to start working on this problem."
    })
    session.history.append({
        "role": "user",
        "content": "Hello! I'm ready to start working on this problem."
    })
    try:
        opener = engine._generate(opening_messages)
    except Exception as e:
        opener = f"⚠️ Could not connect to Ollama. Start it with `ollama serve` then refresh.\n\nError: {e}"
    session.add_turn("assistant", opener)
 
    history    = [(None, opener)]
    prob_html  = make_problem_html(question, qid)
    stats_html = make_stats_html(session)
    status     = f"✅ Session started · Problem {qid} · Model: {model_name}"
 
    return history, session_state, prob_html, stats_html, status
 
 
def reset_session(session_state: dict) -> tuple[list, dict, str, str, str]:
    """Clear the current session."""
    session_id = session_state.get("session_id")
    if session_id and session_id in _sessions:
        del _sessions[session_id]
    new_state = {}
    return [], new_state, "", make_stats_html(None), "Session cleared. Pick a new problem to start."
 
 
def export_session(session_state: dict) -> str:
    """Export current session to JSON string."""
    session_id = session_state.get("session_id")
    session = _sessions.get(session_id)
    if not session:
        return "No active session to export."
    data = session.to_dict()
    return json.dumps(data, indent=2)
 
 
def demo_turn(
    history: list,
    session_state: dict,
    model_name: str,
) -> tuple[list, dict, str, str]:
    """Advance demo by one auto-generated student turn."""
    session_id = session_state.get("session_id")
    session = _sessions.get(session_id)
    if not session:
        history.append((None, "⚠️ No active session. Start one in Problem Setup first."))
        return history, session_state, "", make_stats_html(None)
 
    examples = load_examples()
    ex = next((e for e in examples if e.qid == session_id), None)
 
    engine = get_engine(model_name)
 
    if ex:
        # Pick next student turn from reference dialogue
        student_turns = [t for t in ex.turns if t.speaker == "Student"]
        turn_idx = min(session.turn_count // 2, len(student_turns) - 1)
        student_msg = student_turns[turn_idx].text if student_turns else ex.student_incorrect_solution
    else:
        student_msg = "I'm not sure how to approach this..."
 
    try:
        tutor_response = engine.respond(session, student_msg)
    except Exception as e:
        tutor_response = f"⚠️ Error: {e}"
 
    history.append((student_msg, tutor_response))
    prob_html  = make_problem_html(session.question, session.session_id)
    stats_html = make_stats_html(session)
    return history, session_state, prob_html, stats_html
 
 
# ── Build Gradio UI ────────────────────────────────────────────────────────
 
def build_app(model_name: str = "gemma4:latest") -> gr.Blocks:
 
    examples_list = load_examples(200) if DEPS_OK else []
    choices       = get_example_choices(examples_list)
 
    with gr.Blocks(
        css=CUSTOM_CSS,
        title="Tutoring SLM — DGX Spark",
        theme=gr.themes.Base(
            primary_hue="emerald",
            secondary_hue="slate",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("DM Sans"),
        ),
    ) as app:
 
        # ── State ──────────────────────────────────────────────────────────
        session_state = gr.State({})
 
        # ── Header ─────────────────────────────────────────────────────────
        gr.HTML(make_header_html(model_name))
 
        # ── Body: two-column layout ────────────────────────────────────────
        with gr.Row(equal_height=False):
 
            # ── Left column: problem setup + stats ─────────────────────────
            with gr.Column(scale=1, min_width=320):
 
                # Model selector
                model_selector = gr.Dropdown(
                    choices=["gemma4:latest", "gemma4:31b"],
                    value=model_name,
                    label="🤖  Model",
                    info="gemma4:latest = faster  ·  gemma4:31b = smarter",
                    elem_classes=["panel"],
                )
 
                # Tabs: Browse / Custom
                with gr.Tabs(elem_classes=["tab-nav"]):
 
                    with gr.Tab("📚  MathDial Problems"):
                        problem_dropdown = gr.Dropdown(
                            choices=choices,
                            label="Select a problem",
                            info=f"{len(choices)} test problems loaded from MathDial",
                            value=choices[0] if choices else None,
                        )
                        gr.HTML('<div style="font-size:0.75rem;color:var(--text-muted);margin-top:6px;">Problems from: MathDial (EMNLP 2023) · eth-nlped/mathdial</div>')
 
                    with gr.Tab("✏️  Custom Problem"):
                        custom_q = gr.Textbox(
                            label="Your math problem",
                            placeholder="E.g. A train travels 120 km in 2 hours. What is its speed in km/h?",
                            lines=3,
                        )
                        custom_a = gr.Textbox(
                            label="Correct answer",
                            placeholder="60 km/h",
                        )
 
                # Start / Reset buttons
                with gr.Row():
                    btn_start = gr.Button("▶  Start Session", variant="primary", elem_classes=["btn-primary"])
                    btn_reset = gr.Button("↺  Reset",         variant="secondary", elem_classes=["btn-secondary"])
 
                status_msg = gr.Markdown(
                    "_No active session. Select a problem and click Start._",
                    elem_classes=["panel"],
                )
 
                # Problem display
                problem_display = gr.HTML(
                    '<div class="panel" style="color:var(--text-dim);font-size:0.82rem;">No problem loaded.</div>'
                )
 
                # Live stats
                stats_display = gr.HTML(make_stats_html(None))
 
                # Demo mode
                with gr.Accordion("🎬  Demo Mode", open=False):
                    gr.Markdown(
                        "_Watch a simulated session. Click 'Next Turn' to advance the demo._",
                        elem_classes=["panel"],
                    )
                    btn_demo = gr.Button("⏭  Next Turn (auto student)", elem_classes=["btn-secondary"])
 
                # Export
                with gr.Accordion("📥  Export Session", open=False):
                    export_btn = gr.Button("Export as JSON", elem_classes=["btn-secondary"])
                    export_box = gr.Code(label="Session JSON", language="json", lines=10)
 
            # ── Right column: chat ─────────────────────────────────────────
            with gr.Column(scale=2):
 
                gr.HTML("""
                <div class="panel" style="margin-bottom:8px;">
                  <div class="panel-title">Tutoring Chat</div>
                  <div style="font-size:0.82rem;color:var(--text-muted);">
                    You are the <strong style="color:var(--accent-blue)">student</strong>.
                    The tutor will guide you to the answer — not give it away.
                    Type your working, guesses, or questions below.
                  </div>
                </div>""")
 
                chatbot = gr.Chatbot(
                    label="",
                    height=520,
                    show_label=False,
                    avatar_images=(
                        None,   # user: no avatar
                        None,   # bot: no avatar
                    ),
                    render_markdown=True,
                    elem_classes=["chat-window"],
                )
 
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Type your answer or question here…",
                        show_label=False,
                        scale=5,
                        container=False,
                    )
                    send_btn = gr.Button("Send →", scale=1, variant="primary", elem_classes=["btn-primary"])
 
                gr.HTML("""
                <div style="font-size:0.72rem;color:var(--text-dim);text-align:center;margin-top:6px;">
                  💡 Tips: type <code>hint</code> for a nudge · <code>explain</code> to ask for clarification
                </div>""")
 
        # ── Tips accordion ─────────────────────────────────────────────────
        with gr.Accordion("ℹ️  How to use this app", open=False):
            gr.Markdown("""
**For students:**
1. Pick a maths problem from the dropdown (or enter your own)
2. Click **Start Session** to begin
3. Type your answer in the chat — even if you're not sure!
4. The tutor will guide you with questions, never just give the answer
5. Try to solve it step by step
 
**For teachers / evaluators:**
- Use **Demo Mode** to watch a simulated session without typing
- **Export Session** saves the full conversation as JSON for analysis
- Switch between `gemma4:latest` (fast) and `gemma4:31b` (best quality)
 
**Mastery bar:**
🔴 < 35% · 🟡 35–65% · 🟢 > 65% · This is estimated by the model in real time
            """)
 
        # ── Event wiring ────────────────────────────────────────────────────
 
        btn_start.click(
            fn=start_session,
            inputs=[problem_dropdown, custom_q, custom_a, model_selector, session_state],
            outputs=[chatbot, session_state, problem_display, stats_display, status_msg],
        )
 
        btn_reset.click(
            fn=reset_session,
            inputs=[session_state],
            outputs=[chatbot, session_state, problem_display, stats_display, status_msg],
        )
 
        def _chat_wrap(msg, hist, state, model):
            hist, state, prob, stats = chat(msg, hist, state, model)
            return "", hist, state, prob, stats
 
        send_btn.click(
            fn=_chat_wrap,
            inputs=[msg_input, chatbot, session_state, model_selector],
            outputs=[msg_input, chatbot, session_state, problem_display, stats_display],
        )
 
        msg_input.submit(
            fn=_chat_wrap,
            inputs=[msg_input, chatbot, session_state, model_selector],
            outputs=[msg_input, chatbot, session_state, problem_display, stats_display],
        )
 
        btn_demo.click(
            fn=demo_turn,
            inputs=[chatbot, session_state, model_selector],
            outputs=[chatbot, session_state, problem_display, stats_display],
        )
 
        export_btn.click(
            fn=export_session,
            inputs=[session_state],
            outputs=[export_box],
        )
 
    return app
 
 
# ── Entry point ────────────────────────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser(description="Tutoring SLM Gradio App")
    parser.add_argument("--model",  default="gemma4:latest", help="Ollama model to use")
    parser.add_argument("--port",   default=7860, type=int,  help="Port to serve on")
    parser.add_argument("--host",   default="0.0.0.0",       help="Host to bind to")
    parser.add_argument("--share",  action="store_true",     help="Create a public Gradio share link")
    parser.add_argument("--debug",  action="store_true",     help="Enable Gradio debug mode")
    args = parser.parse_args()
 
    print(f"""
╔══════════════════════════════════════════════════════╗
║        🎓  Tutoring SLM — Gradio Interface           ║
╠══════════════════════════════════════════════════════╣
║  Model:   {args.model:<42} ║
║  URL:     http://localhost:{args.port:<26} ║
║  Share:   {"Yes — public link will be shown" if args.share else "No (local only)":<35} ║
╚══════════════════════════════════════════════════════╝
""")
 
    app = build_app(model_name=args.model)
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        show_api=False,
        favicon_path=None,
    )
 
 
if __name__ == "__main__":
    main()
