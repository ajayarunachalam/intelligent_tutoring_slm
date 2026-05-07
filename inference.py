"""
inference.py
────────────
Local inference engine using Ollama (gemma4:latest or gemma4:31b).
Implements the Socratic tutoring loop with:
  - Session state management (student mastery tracking)
  - Dialog act classification & selection
  - Safety filters (no premature answer reveal)
  - Streaming output for interactive CLI

DGX Spark: Ollama serves gemma4:latest / gemma4:31b locally.
No internet required at inference time.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Generator, Optional

import ollama
from rich.console import Console
from rich.markdown import Markdown

console = Console()
logger = logging.getLogger(__name__)


# ── Dialog act taxonomy ───────────────────────────────────────────────────

class DialogAct(Enum):
    FOCUS      = "focus"       # redirect student attention to problem element
    HINT       = "hint"        # guiding hint without revealing answer
    CORRECTION = "correction"  # gently correct a specific error
    APPROVAL   = "approval"    # confirm correct reasoning
    QUESTION   = "question"    # probing question for diagnosis/advancement
    SOLUTION   = "solution"    # last-resort answer reveal


# ── Session state ──────────────────────────────────────────────────────────

@dataclass
class TutoringSession:
    """Tracks the state of a single student tutoring session."""
    session_id: str
    question: str
    ground_truth: str
    student_profile: str
    confusion: str

    # Conversation history: list of {"role": str, "content": str}
    history: list[dict] = field(default_factory=list)

    # Pedagogical state
    turn_count: int         = 0
    hint_count: int         = 0
    correction_count: int   = 0
    student_solved: bool    = False
    answer_revealed: bool   = False
    last_student_response:  str = ""

    # Mastery tracking: 0.0 (no understanding) → 1.0 (solved)
    mastery_score: float    = 0.0
    mastery_history: list[float] = field(default_factory=list)

    def add_turn(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        self.turn_count += 1
        if role == "user":
            self.last_student_response = content

    def should_reveal_answer(self) -> bool:
        """Heuristic: reveal only if student has had many attempts."""
        return (
            self.turn_count > 12 and
            self.correction_count >= 3 and
            not self.student_solved
        )

    def to_dict(self) -> dict:
        return {
            "session_id":       self.session_id,
            "question":         self.question,
            "ground_truth":     self.ground_truth,
            "turn_count":       self.turn_count,
            "hint_count":       self.hint_count,
            "correction_count": self.correction_count,
            "student_solved":   self.student_solved,
            "answer_revealed":  self.answer_revealed,
            "mastery_score":    self.mastery_score,
            "history":          self.history,
        }


# ── Prompts ────────────────────────────────────────────────────────────────

TUTOR_SYSTEM = """\
You are an expert Socratic mathematics tutor for 7th-grade students.
Your ONLY goal is to guide the student to discover the correct answer themselves.

STRICT RULES:
1. NEVER give the answer directly unless explicitly instructed.
2. Ask ONE focused question per response.
3. Keep responses under 3 sentences.
4. Diagnose the student's specific error before correcting.
5. Celebrate correct steps enthusiastically but briefly.
6. If the student is completely stuck after many attempts, give a small hint — not the answer.

Current problem:
{question}

Correct answer (your reference — do NOT share): {ground_truth}

Student background: {profile}
Known confusion: {confusion}
"""

MASTERY_EVAL_PROMPT = """\
You are evaluating a student's mathematical understanding.

Problem: {question}
Correct answer: {ground_truth}
Student's latest response: "{student_response}"

Rate the student's current understanding on a scale 0.0 to 1.0:
- 0.0 = completely wrong / off-topic
- 0.3 = partial understanding, major errors
- 0.6 = mostly correct, minor errors  
- 0.9 = correct reasoning, small slip
- 1.0 = fully correct

Respond with ONLY a JSON object: {{"score": 0.7, "reasoning": "brief explanation"}}
"""

SAFETY_CHECK_PROMPT = """\
Does the following tutoring response reveal the answer to this math problem?
Problem: {question}
Answer: {ground_truth}
Response: "{response}"

Reply with ONLY: {{"reveals_answer": true/false}}
"""


# ── Inference engine ──────────────────────────────────────────────────────

class TutoringInferenceEngine:
    """
    Ollama-based tutoring inference engine.
    Uses gemma4:latest (fast) or gemma4:31b (best quality).

    Usage:
        engine = TutoringInferenceEngine(model="gemma4:latest")
        session = engine.new_session(question, answer, confusion)
        response = engine.respond(session, student_input)
    """

    def __init__(
        self,
        model: str = "gemma4:latest",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 300,
        safety_check: bool = True,
        mastery_tracking: bool = True,
    ):
        self.model            = model
        self.temperature      = temperature
        self.top_p            = top_p
        self.max_tokens       = max_tokens
        self.safety_check     = safety_check
        self.mastery_tracking = mastery_tracking
        self._client          = ollama.Client()
        self._verify_model()

    # ── Public API ──────────────────────────────────────────────────────────

    def new_session(
        self,
        question: str,
        ground_truth: str,
        student_profile: str = "",
        confusion: str = "",
        session_id: Optional[str] = None,
    ) -> TutoringSession:
        """Create and initialise a new tutoring session."""
        sid = session_id or f"session_{int(time.time())}"
        session = TutoringSession(
            session_id=sid,
            question=question,
            ground_truth=ground_truth,
            student_profile=student_profile,
            confusion=confusion,
        )
        return session

    def respond(
        self,
        session: TutoringSession,
        student_input: str,
        stream: bool = False,
    ) -> str:
        """
        Generate the tutor's next response given the student's input.

        Args:
            session:      current TutoringSession
            student_input: the student's message
            stream:        if True, print tokens as they arrive

        Returns:
            tutor_response (str)
        """
        # Record student turn
        session.add_turn("user", student_input)

        # Update mastery estimate
        if self.mastery_tracking:
            self._update_mastery(session)
            if session.mastery_score >= 0.95:
                session.student_solved = True

        # Check if we should reveal the answer
        force_reveal = session.should_reveal_answer()

        # Build the prompt
        messages = self._build_messages(session, force_reveal=force_reveal)

        # Generate response
        if stream:
            response = self._stream_response(messages)
        else:
            response = self._generate(messages)

        # Safety check: ensure answer isn't prematurely revealed
        if self.safety_check and not force_reveal and not session.should_reveal_answer():
            response = self._safety_filter(response, session)

        # Update session state
        session.add_turn("assistant", response)
        if "(correction)" in response.lower():
            session.correction_count += 1
        if "(hint)" in response.lower():
            session.hint_count += 1
        if "(solution)" in response.lower() or "the answer is" in response.lower():
            session.answer_revealed = True

        return response

    def respond_stream(
        self,
        session: TutoringSession,
        student_input: str,
    ) -> Generator[str, None, str]:
        """
        Streaming version of respond().
        Yields token chunks; returns final full response.
        """
        session.add_turn("user", student_input)

        if self.mastery_tracking:
            self._update_mastery(session)

        messages = self._build_messages(session)
        full_response = ""

        stream = self._client.chat(
            model=self.model,
            messages=messages,
            stream=True,
            options={
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_predict": self.max_tokens,
            }
        )

        for chunk in stream:
            token = chunk["message"]["content"]
            full_response += token
            yield token

        session.add_turn("assistant", full_response)
        if "(correction)" in full_response.lower():
            session.correction_count += 1
        if "(hint)" in full_response.lower():
            session.hint_count += 1

        return full_response

    # ── Private methods ─────────────────────────────────────────────────────

    def _verify_model(self):
        """Check that the requested model is available in Ollama."""
        try:
            models = self._client.list()
            available = [m["name"] for m in models.get("models", [])]
            if not any(self.model in m or m in self.model for m in available):
                console.print(
                    f"[yellow]⚠  Model '{self.model}' not found in Ollama. "
                    f"Available: {available}\n"
                    f"Pull it with: ollama pull {self.model}[/yellow]"
                )
            else:
                console.print(f"[green]✓ Model '{self.model}' ready[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠  Could not verify Ollama model: {e}[/yellow]")

    def _build_messages(
        self,
        session: TutoringSession,
        force_reveal: bool = False,
    ) -> list[dict]:
        """Build the message list for Ollama chat."""
        system_content = TUTOR_SYSTEM.format(
            question=session.question,
            ground_truth=session.ground_truth if force_reveal else "[HIDDEN]",
            profile=session.student_profile or "typical 7th grader",
            confusion=session.confusion or "unknown",
        )

        if force_reveal:
            system_content += (
                "\n\nIMPORTANT: The student has struggled for many turns. "
                "You may now gently walk them through the full solution step by step."
            )

        messages = [{"role": "system", "content": system_content}]

        # Add conversation history (last 10 turns to stay within context)
        history = session.history[-20:]  # 10 turns = 20 messages
        messages.extend(history)

        return messages

    def _generate(self, messages: list[dict]) -> str:
        """Non-streaming generation via Ollama."""
        try:
            response = self._client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "num_predict": self.max_tokens,
                    "stop": ["<end_of_turn>", "<eos>"],
                }
            )
            return response["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Ollama inference error: {e}")
            return "I'm having trouble connecting. Let's try again — can you re-read the problem?"

    def _stream_response(self, messages: list[dict]) -> str:
        """Stream response and return full text."""
        full = ""
        try:
            stream = self._client.chat(
                model=self.model,
                messages=messages,
                stream=True,
                options={
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "num_predict": self.max_tokens,
                    "stop": ["<end_of_turn>", "<eos>"],
                }
            )
            console.print("[bold green]Tutor:[/bold green] ", end="")
            for chunk in stream:
                token = chunk["message"]["content"]
                full += token
                console.print(token, end="", markup=False)
            console.print()
        except Exception as e:
            logger.error(f"Stream error: {e}")
            full = "Let me try a different approach. Can you tell me what you understand so far?"
        return full.strip()

    def _update_mastery(self, session: TutoringSession):
        """Estimate student mastery score using a lightweight LLM call."""
        if not session.last_student_response:
            return

        prompt = MASTERY_EVAL_PROMPT.format(
            question=session.question,
            ground_truth=session.ground_truth,
            student_response=session.last_student_response,
        )
        try:
            result = self._client.generate(
                model=self.model,
                prompt=prompt,
                options={"temperature": 0.1, "num_predict": 80},
            )
            raw = result["response"].strip()
            # Extract JSON from response (may have surrounding text)
            import re
            match = re.search(r'\{[^}]+\}', raw)
            if match:
                data = json.loads(match.group())
                score = float(data.get("score", session.mastery_score))
                session.mastery_score = max(0.0, min(1.0, score))
                session.mastery_history.append(session.mastery_score)
        except Exception as e:
            logger.debug(f"Mastery eval failed: {e}")

    def _safety_filter(self, response: str, session: TutoringSession) -> str:
        """
        Check if response prematurely reveals the answer.
        If so, regenerate with a stricter prompt.
        """
        check_prompt = SAFETY_CHECK_PROMPT.format(
            question=session.question,
            ground_truth=session.ground_truth,
            response=response[:300],
        )
        try:
            result = self._client.generate(
                model=self.model,
                prompt=check_prompt,
                options={"temperature": 0.0, "num_predict": 30},
            )
            import re
            raw = result["response"].strip()
            match = re.search(r'\{[^}]+\}', raw)
            if match:
                data = json.loads(match.group())
                if data.get("reveals_answer", False):
                    logger.warning("Safety filter triggered: regenerating without answer")
                    # Regenerate with explicit no-reveal instruction
                    messages = self._build_messages(session)
                    messages.append({
                        "role": "system",
                        "content": "IMPORTANT: Do NOT reveal the answer. Guide with a question only."
                    })
                    # Remove the last user turn to avoid duplication
                    messages = [m for m in messages if not (m == messages[-2])]
                    return self._generate(messages[:-1] if len(messages) > 2 else messages)
        except Exception:
            pass  # If safety check fails, return original response
        return response

    def get_session_summary(self, session: TutoringSession) -> dict:
        """Return a summary of the session for evaluation."""
        return {
            "session_id":       session.session_id,
            "total_turns":      session.turn_count,
            "hint_count":       session.hint_count,
            "correction_count": session.correction_count,
            "student_solved":   session.student_solved,
            "answer_revealed":  session.answer_revealed,
            "final_mastery":    session.mastery_score,
            "mastery_trajectory": session.mastery_history,
        }
