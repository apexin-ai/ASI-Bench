"""Scorer base class and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ai4sci_bench.core.types import ScoreDetail


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
