"""Shared utilities for parsing LLM judge responses.

Used by both ``llm_judge`` and ``multimodal`` scorers to extract numeric
scores from model output. Fixes #10: fragile JSON parsing with
``find("{")``/``rfind("}")`` that fails when model output contains
braces outside the target JSON object.
"""

from __future__ import annotations

import json
import re

from ai4sci_bench.core.logger import get_logger

logger = get_logger(__name__)


def parse_judge_json(raw: str, max_score: float = 10.0) -> float | None:
    """Extract a numeric ``score`` from a judge LLM response.

    Parsing priority:
      1. Fenced JSON block (````json { ... } ````)
      2. Reverse brace scan — find the last valid ``{"score": ...}`` object
      3. Regex fallback for common free-text patterns

    Returns:
        Parsed score clamped to ``[0, max_score]``, or ``None`` when
        parsing fails entirely (caller should treat as missing data,
        **not** as a zero score).
    """
    if not raw or not raw.strip():
        logger.warning("Empty judge response")
        return None

    score = _try_fenced_json(raw)
    if score is None:
        score = _try_brace_scan(raw)
    if score is None:
        score = _try_regex(raw)

    if score is not None:
        clamped = max(0.0, min(float(score), max_score))
        if clamped != score:
            logger.warning(
                "Judge score %.2f clamped to [0, %.1f] → %.2f",
                score, max_score, clamped,
            )
        return clamped

    logger.warning("Could not parse judge response: %s", raw[:200])
    return None


def _try_fenced_json(raw: str) -> float | None:
    """Try to extract score from a fenced JSON code block."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1))
            return float(data["score"])
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            pass
    return None


def _try_brace_scan(raw: str) -> float | None:
    """Scan from the end of the string for valid JSON objects with a "score" key."""
    end = len(raw)
    while True:
        j = raw.rfind("}", 0, end)
        if j < 0:
            break
        # Try progressively earlier opening braces
        i = raw.rfind("{", 0, j + 1)
        while i >= 0:
            candidate = raw[i : j + 1]
            try:
                data = json.loads(candidate)
                if "score" in data:
                    return float(data["score"])
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                pass
            i = raw.rfind("{", 0, i)
        end = j
    return None


def _try_regex(raw: str) -> float | None:
    """Try common free-text score patterns."""
    patterns = [
        # 9/10
        r"\b(\d+(?:\.\d+)?)\s*/\s*\d+",
        # Score: 9, **Score:** 9, Score = 9, Score - 9
        r"[*]*[Ss]core[*]*\s*[*:=\-]+\s*(\d+(?:\.\d+)?)",
        # "score": 9 (JSON fragment when outer parse failed)
        r"\"score\"\s*:\s*(\d+(?:\.\d+)?)",
        # "9 out of 10"
        r"\b(\d+(?:\.\d+)?)\s+out\s+of\s+\d+",
    ]
    for pat in patterns:
        m = re.search(pat, raw)
        if m:
            return float(m.group(1))
    return None
