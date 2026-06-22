"""
Answer generation pipeline for RAGLab.

Takes retrieved context chunks and a user question, then:
1. Builds a structured prompt with conversation history.
2. Calls the LLM for a JSON-formatted answer.
3. Cleans, normalises, and optionally expands the answer.
4. Runs a self-check to verify grounding.

All functions prefixed with ``_`` are internal to this module.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from .retrieval import _llm_complete


# ===================================================================
# History serialisation
# ===================================================================
def _serialize_history(chat_history: list[Any]) -> str:
    """Format the last ~12 conversation turns as a compact text block."""
    if not chat_history:
        return ""
    lines: list[str] = []
    for turn in chat_history[-12:]:
        role = str(getattr(turn, "role", "")).strip().lower()
        content = str(getattr(turn, "content", "")).strip()
        if not content:
            continue
        # Keep memory useful but token-cheap to avoid rate-limit spikes.
        content = content[:320]
        if role not in {"user", "ai", "assistant"}:
            role = "user"
        role_label = "User" if role == "user" else "Assistant"
        lines.append(f"{role_label}: {content}")
    return "\n".join(lines)


# ===================================================================
# List / bullet helpers
# ===================================================================
def _list_to_bullets(items: list[Any]) -> str:
    """Convert a list of strings to a markdown bullet list."""
    return "\n".join(
        f"- {str(x).strip().lstrip('-').strip()}"
        for x in items
        if str(x).strip()
    )


def _wants_numbered_points(question: str) -> bool:
    """Check if the user explicitly asked for a numbered list."""
    q = (question or "").lower()
    return any(
        k in q
        for k in [
            "numbered",
            "ordered list",
            "1. 2. 3.",
            "1) 2) 3)",
            "number points",
        ]
    )


def _dedupe_and_cap_bullets(text: str, max_items: int = 8) -> str:
    """Remove duplicate bullet points and cap at *max_items*."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    cleaned: list[str] = []
    seen: set[str] = set()
    for ln in lines:
        ln = re.sub(r"^\d+\.\s+", "", ln).lstrip("-").strip()
        if not ln:
            continue
        key = re.sub(r"\s+", " ", ln.lower())
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(f"- {ln}")
        if len(cleaned) >= max_items:
            break
    return "\n".join(cleaned)


def _bullets_to_numbered(text: str) -> str:
    """Convert a bullet list to a numbered list."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    out: list[str] = []
    idx = 1
    for ln in lines:
        core = re.sub(r"^\d+\.\s+", "", ln).lstrip("-").strip()
        if not core:
            continue
        out.append(f"{idx}. {core}")
        idx += 1
    return "\n".join(out)


def _parse_listish_string(raw: str) -> list[str] | None:
    """Try to parse a string that looks like a Python list literal."""
    text = (raw or "").strip()
    if not text:
        return None
    candidates = [text]
    if text.startswith("- ["):
        candidates.append(text[2:].strip())
    for c in candidates:
        if not (c.startswith("[") and c.endswith("]")):
            continue
        try:
            value = ast.literal_eval(c)
            if isinstance(value, list):
                return [str(x).strip() for x in value if str(x).strip()]
        except Exception:
            continue
    return None


# ===================================================================
# Structured answer generation
# ===================================================================
async def _structured_answer(
    question: str,
    context_blocks: list[str],
    numeric_safe: bool,
    answer_length: str = "medium",
    chat_history: list[Any] | None = None,
) -> dict[str, Any]:
    """Generate a structured answer via the LLM.

    Returns a dict with keys:
        - ``direct_answer`` — the main answer text.
        - ``evidence_points`` — list of supporting evidence snippets.
        - ``assumptions`` — list of assumptions made.
        - ``cannot_find`` — boolean, true if the answer could not be found.
    """
    context = "\n\n---\n\n".join(context_blocks)
    history_text = _serialize_history(chat_history or [])
    q = (question or "").lower()

    wants_points = any(
        k in q
        for k in [
            "point",
            "points",
            "bullet",
            "bullets",
            "list",
            "key takeaways",
            "main points",
        ]
    )

    safe_note = (
        "The question appears numeric. Do not invent any number. "
        "If uncertain, set cannot_find=true."
        if numeric_safe
        else "If not found in context, set cannot_find=true."
    )

    length = (answer_length or "medium").lower()
    paragraph_shape = {
        "short": (
            "direct_answer must be a concise 1-2 sentence response "
            "with no bullet points."
        ),
        "medium": (
            "direct_answer must be a clear 2-4 sentence response "
            "with no bullet points."
        ),
        "detailed": (
            "direct_answer must be a detailed 4-7 sentence response "
            "with no bullet points."
        ),
    }.get(length, "direct_answer must be a clear 2-4 sentence response with no bullet points.")

    answer_shape = (
        "direct_answer must be 4-7 concise bullet points, "
        "each on a new line starting with '- '."
        if wants_points
        else paragraph_shape
    )

    prompt = (
        "You are a precise document QA assistant.\n"
        "Use only the provided context.\n"
        f"{safe_note}\n"
        "Respect the prior conversation if it is relevant "
        "to the current question.\n"
        "Return strict JSON with keys: "
        "direct_answer, evidence_points, assumptions, cannot_find.\n"
        f"{answer_shape}\n\n"
        f"Conversation so far:\n{history_text or 'N/A'}\n\n"
        f"Question: {question}\n\nContext:\n{context}\n\nJSON:"
    )

    raw = await _llm_complete(prompt)
    try:
        data = json.loads(raw)
        return {
            "direct_answer": data.get("direct_answer", ""),
            "evidence_points": data.get("evidence_points", []),
            "assumptions": data.get("assumptions", []),
            "cannot_find": bool(data.get("cannot_find", False)),
        }
    except Exception:
        return {
            "direct_answer": raw.strip(),
            "evidence_points": [],
            "assumptions": [
                "Model did not return strict JSON; using raw output fallback."
            ],
            "cannot_find": False,
        }


# ===================================================================
# Answer cleaning
# ===================================================================
def _extract_json_from_text(raw: str) -> dict[str, Any] | None:
    """Extract a JSON object from a string that may contain surrounding text."""
    text = (raw or "").strip()
    if not text:
        return None
    candidates = [text]
    if "```json" in text:
        fenced = text.split("```json", 1)[1]
        fenced = fenced.split("```", 1)[0]
        candidates.append(fenced.strip())
    if "{" in text and "}" in text:
        start = text.find("{")
        end = text.rfind("}")
        if start < end:
            candidates.append(text[start : end + 1].strip())
    for c in candidates:
        try:
            data = json.loads(c)
            if isinstance(data, dict):
                return data
        except Exception:
            continue
    return None


def _clean_answer_text(raw: Any) -> str:
    """Normalise the raw LLM output into clean answer text.

    Handles JSON-wrapped answers, list literals, and plain text.
    """
    if isinstance(raw, list):
        return _list_to_bullets(raw)
    if isinstance(raw, dict):
        raw = json.dumps(raw, ensure_ascii=False)

    data = _extract_json_from_text(raw)
    if data:
        direct_raw = data.get("direct_answer", "")
        if isinstance(direct_raw, list):
            direct = _list_to_bullets(direct_raw)
        else:
            direct = str(direct_raw).strip()
            parsed = _parse_listish_string(direct)
            if parsed:
                direct = _list_to_bullets(parsed)
        if direct:
            return direct

    cleaned = str(raw or "").replace("```json", "").replace("```", "").strip()
    # Handle malformed JSON-like spillover where keys leak into text.
    if (
        '"direct_answer"' in cleaned
        or cleaned.startswith("{")
        or cleaned.startswith("- {")
    ):
        payload = _extract_json_from_text(cleaned)
        if payload and payload.get("direct_answer") is not None:
            return _clean_answer_text(payload.get("direct_answer"))

    parsed = _parse_listish_string(cleaned)
    if parsed:
        return _list_to_bullets(parsed)
    return cleaned


# ===================================================================
# Format normalisation
# ===================================================================
def _normalize_points_format(
    text: str,
    prefer_numbered: bool = False,
) -> str:
    """Normalise bullet / numbered list formatting."""
    raw = (text or "").strip()
    if not raw:
        return raw

    parsed = _parse_listish_string(raw)
    if parsed:
        out = _dedupe_and_cap_bullets(_list_to_bullets(parsed))
        return _bullets_to_numbered(out) if prefer_numbered else out

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if any(ln.startswith("- ") or re.match(r"^\d+\.\s+", ln) for ln in lines):
        out = []
        for ln in lines:
            ln = re.sub(r"^\d+\.\s+", "", ln)
            if not ln.startswith("- "):
                ln = f"- {ln}"
            out.append(ln)
        norm = _dedupe_and_cap_bullets("\n".join(out))
        return _bullets_to_numbered(norm) if prefer_numbered else norm

    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", raw) if p.strip()]
    if len(parts) <= 1:
        return f"- {raw}"
    norm = _dedupe_and_cap_bullets(
        "\n".join([f"- {p}" for p in parts[:7]])
    )
    return _bullets_to_numbered(norm) if prefer_numbered else norm


# ===================================================================
# Answer expansion
# ===================================================================
def _should_expand_answer(
    question: str,
    base_answer: str,
    answer_length: str = "medium",
) -> bool:
    """Determine whether the answer needs to be expanded."""
    q = (question or "").lower()
    wants_detail = any(
        k in q
        for k in [
            "explain",
            "detailed",
            "in detail",
            "elaborate",
            "more",
            "why",
            "how",
            "steps",
        ]
    )
    word_count = len((base_answer or "").split())
    length = (answer_length or "medium").lower()
    floor = {"short": 20, "medium": 45, "detailed": 90}.get(length, 45)
    ceiling = {"short": 35, "medium": 90, "detailed": 160}.get(length, 90)
    if word_count < floor:
        return True
    return wants_detail and word_count < ceiling


def _dedupe_repeated_sentences(text: str) -> str:
    """Remove duplicate or near-duplicate sentences."""
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    out: list[str] = []
    seen: set[str] = set()
    for p in parts:
        key = re.sub(r"\s+", " ", p.strip().lower())
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(p.strip())
    return " ".join(out).strip()


async def _expand_answer_if_needed(
    question: str,
    answer: str,
    context_blocks: list[str],
    answer_length: str = "medium",
    max_rounds: int = 2,
) -> str:
    """Expand a short answer with additional sentences from the LLM."""
    text = (answer or "").strip()
    q = (question or "").lower()

    wants_points = any(
        k in q
        for k in [
            "point",
            "points",
            "bullet",
            "bullets",
            "list",
            "key takeaways",
            "main points",
        ]
    )
    wants_numbered = _wants_numbered_points(question)

    if wants_points:
        return _normalize_points_format(text, prefer_numbered=wants_numbered)
    if not _should_expand_answer(question, text, answer_length=answer_length):
        return text

    context = "\n\n---\n\n".join(context_blocks[:4])
    wants_detail = any(
        k in q
        for k in [
            "explain",
            "detailed",
            "in detail",
            "elaborate",
            "more",
            "why",
            "how",
            "steps",
        ]
    )
    length = (answer_length or "medium").lower()
    # Reduce extra LLM calls to stay under TPM limits.
    default_rounds = {"short": 0, "medium": 0, "detailed": 1}.get(length, 0)
    rounds = min(max_rounds, max(default_rounds, 1 if wants_detail else 0))

    for _ in range(rounds):
        prompt = (
            "You are extending an answer for a document-grounded QA response.\n"
            "Continue the answer with 2-4 additional sentences.\n"
            "Do not repeat existing wording. Do not return JSON.\n"
            "Use only the context below.\n\n"
            f"Question: {question}\n\n"
            f"Current answer:\n{text}\n\n"
            f"Context:\n{context}\n\n"
            "Write only the continuation:"
        )
        continuation = _clean_answer_text(
            await _llm_complete(prompt, timeout_seconds=18.0)
        )
        if not continuation:
            break
        text = f"{text} {continuation}".strip()
        text = _dedupe_repeated_sentences(text)
        if len(text.split()) >= 120:
            break
    return text


# ===================================================================
# Self-check grounding
# ===================================================================
async def _self_check_answer(
    question: str,
    answer: str,
    context_blocks: list[str],
) -> bool:
    """Ask the LLM whether the answer is properly grounded in context."""
    context = "\n\n---\n\n".join(context_blocks[:4])
    prompt = (
        "Check if the answer is grounded in context.\n"
        'Return JSON {"grounded": true/false} only.\n'
        f"Question: {question}\nAnswer: {answer}\nContext:\n{context}\nJSON:"
    )
    try:
        data = json.loads(await _llm_complete(prompt, timeout_seconds=12.0))
        return bool(data.get("grounded", False))
    except Exception:
        return True


# ===================================================================
# Answer length normalisation
# ===================================================================
def _normalize_answer_length(value: str) -> str:
    """Normalise the answer-length parameter to one of the three valid values."""
    v = (value or "").strip().lower()
    if v in {"short", "medium", "detailed"}:
        return v
    return "medium"