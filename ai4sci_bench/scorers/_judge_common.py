"""Judge scorer public infrastructure shared by llm_judge, multimodal, and agent_judge."""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from ai4sci_bench.core.logger import get_logger
from ai4sci_bench.core.types import ScoreDetail

logger = get_logger(__name__)

MAX_RETRIES = 3
BACKOFF_BASE = 2.0

_scoring_defaults_cache: dict[str, Any] | None = None


def get_scoring_defaults() -> dict[str, Any]:
    """Load global scoring defaults from configs/scoring.yaml.

    Returns a dict with optional keys: api_base, api_key.
    The file is loaded once and cached for the process lifetime.
    """
    global _scoring_defaults_cache
    if _scoring_defaults_cache is not None:
        return _scoring_defaults_cache

    config_path = Path(__file__).resolve().parents[2] / "configs" / "scoring.yaml"
    if not config_path.exists():
        _scoring_defaults_cache = {}
        return _scoring_defaults_cache

    try:
        import yaml
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        _scoring_defaults_cache = data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning("Failed to load scoring defaults from %s: %s", config_path, exc)
        _scoring_defaults_cache = {}

    return _scoring_defaults_cache


def resolve_scorer_api_params(config: dict[str, Any]) -> tuple[str | None, str | None]:
    """Resolve api_base and api_key for a scorer, with global defaults as fallback.

    Priority: task.yaml scorer config > configs/scoring.yaml > None
    """
    defaults = get_scoring_defaults()
    api_base = config.get("api_base") or defaults.get("api_base")
    api_key = config.get("api_key") or defaults.get("api_key")
    return api_base, api_key

SUPPORTED_MODELS: dict[str, str] = {
    "gemini-3.1-pro-preview": "gemini/gemini-3.1-pro-preview",
    "claude-opus-4-6": "anthropic/claude-opus-4-6",
    "gpt-5.4": "openai/gpt-5.4",
    "openrouter/claude-opus-4-6": "openrouter/anthropic/claude-opus-4-6",
    "openrouter/gpt-5.4": "openrouter/openai/gpt-5.4",
    "openrouter/gemini-3.1-pro-preview": "openrouter/google/gemini-3.1-pro-preview",
}

DEFAULT_MODEL = "gemini/gemini-3.1-pro-preview"


def resolve_model(model: str) -> str:
    """Resolve a short model name to a litellm model identifier."""
    resolved = SUPPORTED_MODELS.get(model, model)
    if resolved.startswith("google/"):
        resolved = "gemini/" + resolved[len("google/"):]
    return resolved


def aggregate_judge_scores(
    scores: list[float | None],
    raw_responses: list[str],
    *,
    scorer_name: str,
    model: str,
    num_judges: int,
    max_score_value: float,
    weight: float,
    threshold: float,
    extra_details: dict[str, Any] | None = None,
) -> ScoreDetail:
    """Aggregate multiple judge scores into a single ScoreDetail.

    Shared by llm_judge, vlm_judge, and agent_judge.
    """
    valid_scores = [s for s in scores if s is not None]
    parse_failures = len(scores) - len(valid_scores)

    if not valid_scores:
        logger.error("All %d judge responses failed to parse", len(scores))
        median_score = 0.0
    else:
        median_score = statistics.median(valid_scores)
        if parse_failures > 0:
            logger.warning(
                "%d/%d judge responses failed to parse, "
                "median based on %d valid scores",
                parse_failures, len(scores), len(valid_scores),
            )

    if max_score_value <= 0:
        logger.error("max_score_value=%s is invalid, treating as 0 score", max_score_value)
        normalized = 0.0
    else:
        normalized = median_score / max_score_value
    final_score = normalized * weight
    passed = normalized >= threshold

    details: dict[str, Any] = {
        "model": model,
        "num_judges": num_judges,
        "individual_scores": [s if s is not None else "PARSE_FAILED" for s in scores],
        "raw_responses": raw_responses,
        "parse_failures": parse_failures,
        "valid_judge_count": len(valid_scores),
        "median_score": median_score,
        "normalized_score": round(normalized, 4),
        "threshold": threshold,
    }
    if extra_details:
        details.update(extra_details)

    return ScoreDetail(
        scorer_name=scorer_name,
        score=round(final_score, 4),
        max_score=weight,
        passed=passed,
        details=details,
        message=f"Judge median: {median_score}/{max_score_value}",
    )


def evaluator_unavailable_result(
    *,
    scorer_name: str,
    weight: float,
    error: Exception | str,
    model: str | None = None,
) -> ScoreDetail:
    """Return a zero score that is explicitly marked as evaluator infrastructure failure."""
    error_text = str(error)
    details: dict[str, Any] = {
        "error": error_text,
        "evaluator_status": "unavailable",
        "evaluator_unavailable": True,
    }
    if model:
        details["model"] = model

    return ScoreDetail(
        scorer_name=scorer_name,
        score=0.0,
        max_score=weight,
        passed=False,
        details=details,
        message=f"Evaluator unavailable: {error_text}",
    )
