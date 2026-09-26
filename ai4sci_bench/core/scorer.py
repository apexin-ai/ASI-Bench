"""Scorer base class and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from pathlib import Path
from typing import Any

from ai4sci_bench.core.types import ScoreDetail


def score_divisor(evaluation: dict[str, Any]) -> float:
    """Return the validated task-level divisor used before aggregation."""
    raw = evaluation.get("score_divisor", 1.0)
    if isinstance(raw, bool):
        raise ValueError(
            f"score_divisor must be a finite positive number, got {raw!r}"
        )
    try:
        divisor = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"score_divisor must be a finite positive number, got {raw!r}"
        ) from exc
    if not math.isfinite(divisor) or divisor <= 0.0:
        raise ValueError(
            f"score_divisor must be a finite positive number, got {raw!r}"
        )
    return divisor


def scoring_max_score(evaluation: dict[str, Any]) -> float:
    """Return the raw sum of configured scorer weights."""
    return float(
        sum(
            float(item.get("weight", 1.0))
            for item in evaluation.get("scoring", [])
        )
    )


def normalize_task_score(
    evaluation: dict[str, Any], final_score: float | None, max_score: float
) -> tuple[float | None, float]:
    """Normalize a task score and maximum without altering scorer details."""
    divisor = score_divisor(evaluation)
    return (
        None if final_score is None else float(final_score) / divisor,
        float(max_score) / divisor,
    )


def failure_score_metadata(
    evaluation: dict[str, Any], max_score: float
) -> tuple[float, float | None]:
    """Return failure-safe maximum/divisor metadata for an invalid evaluation."""
    try:
        _unused_final, normalized_max = normalize_task_score(
            evaluation, None, max_score
        )
        return normalized_max, score_divisor(evaluation)
    except (TypeError, ValueError):
        # A malformed evaluator contract is itself an evaluator error. Preserve
        # the raw maximum so reporting that failure cannot fail a second time.
        return float(max_score), None


class Scorer(ABC):
    """Base class for all scorers."""

    name: str  # set by @register_scorer

    @abstractmethod
    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        """Evaluate predictions against ground truth.

        Args:
            pred_dir: directory containing agent output files
            ref_dir:  directory containing ground-truth reference files
            config:   scorer-specific config from task.yaml

        Returns:
            ScoreDetail with score, pass/fail, and breakdown details
        """
        ...


_SCORER_REGISTRY: dict[str, type[Scorer]] = {}


def register_scorer(name: str):
    """Decorator to register a scorer class into the global registry."""
    def decorator(cls):
        _SCORER_REGISTRY[name] = cls
        cls.name = name
        return cls
    return decorator


def get_scorer(name: str) -> Scorer:
    """Return a scorer instance by registered name."""
    if name not in _SCORER_REGISTRY:
        raise KeyError(f"Scorer '{name}' not found. Registered: {list(_SCORER_REGISTRY)}")
    return _SCORER_REGISTRY[name]()


def list_scorers() -> list[str]:
    """Return all registered scorer names."""
    return list(_SCORER_REGISTRY.keys())
