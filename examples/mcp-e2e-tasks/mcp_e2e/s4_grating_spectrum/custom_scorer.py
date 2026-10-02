"""Scorers for the s4 grating-spectrum MCP E2E fake task.

They only compare the agent's result.json with the pre-generated reference and
never run S4 or an RCWA. Whether the numbers came from the MCP tools is checked
separately from the run artefacts by scripts/mcp/e2e/verify_run.py.

Submission problems (missing/invalid result.json) are ordinary zero scores.
A missing or unreadable reference is an evaluator failure and is reported with
``scorer_internal_error: true``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from ai4sci_bench.core.scorer import Scorer, register_scorer
from ai4sci_bench.core.types import ScoreDetail

KEYS = ("R_at_report", "T_at_report", "R_max", "wavelength_at_R_max_um")


class _PredictionError(ValueError):
    pass


class _ReferenceError(ValueError):
    pass


def _number(value, what: str) -> float:
    if isinstance(value, bool):
        raise _PredictionError(f"{what} must be a number, got {value!r}")
    try:
        number = float(str(value).strip()) if isinstance(value, str) else float(value)
    except (TypeError, ValueError):
        raise _PredictionError(f"{what} must be a number, got {value!r}") from None
    if not math.isfinite(number):
        raise _PredictionError(f"{what} is not finite: {number}")
    return number


def load_prediction(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise _PredictionError(f"{path.name} not found")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _PredictionError(f"{path.name} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise _PredictionError(f"{path.name} must contain a JSON object")
    return {key: _number(data.get(key), key) for key in KEYS}


def load_reference(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise _ReferenceError(f"reference not found: {path.name}")
    try:
        ref = json.loads(path.read_text(encoding="utf-8"))
        return {key: float(ref[key]) for key in KEYS}
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise _ReferenceError(f"unreadable reference: {exc}") from exc


def credit(abs_err: float, full_tol: float, zero_tol: float) -> float:
    """1 at <= full_tol, 0 at >= zero_tol, log-linear in between."""
    if abs_err <= full_tol:
        return 1.0
    if abs_err >= zero_tol:
        return 0.0
    span = math.log10(zero_tol) - math.log10(full_tol)
    return max(0.0, min(1.0, 1.0 - (math.log10(abs_err) - math.log10(full_tol)) / span))


def _zero(name: str, weight: float, message: str) -> ScoreDetail:
    return ScoreDetail(scorer_name=name, score=0.0, max_score=weight, passed=False,
                       details={"error": message}, message=message)


def _evaluator_failure(name: str, weight: float, message: str) -> ScoreDetail:
    return ScoreDetail(scorer_name=name, score=0.0, max_score=weight, passed=False,
                       details={"scorer_internal_error": True, "failure_kind": "missing_evaluator_input",
                                "error": message}, message=message)


@register_scorer("s4_e2e_schema")
class SpectrumSchema(Scorer):
    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        name = "s4_e2e_schema"
        weight = float(config.get("weight", 1.0))
        try:
            load_prediction(pred_dir / config.get("pred_file", "result.json"))
        except _PredictionError as exc:
            return _zero(name, weight, str(exc))
        return ScoreDetail(scorer_name=name, score=weight, max_score=weight, passed=True,
                           details={"keys": list(KEYS)}, message="result.json has all four finite numbers")


@register_scorer("s4_e2e_value")
class SpectrumValue(Scorer):
    """|pred - ref| for one key: full credit at <= full_score_tol, zero at >= zero_score_tol, log-linear."""

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        key = config.get("key")
        name = f"s4_e2e_value[{key}]"
        weight = float(config.get("weight", 1.0))
        if key not in KEYS:
            return ScoreDetail(scorer_name=name, score=0.0, max_score=weight, passed=False,
                               details={"scorer_internal_error": True, "failure_kind": "evaluator_runtime_error",
                                        "error": f"unknown key {key!r}"}, message=f"unknown key {key!r}")
        full_tol = float(config["full_score_tol"])
        zero_tol = float(config["zero_score_tol"])
        try:
            ref = load_reference(ref_dir / config.get("ref_file", "reference.json"))
        except _ReferenceError as exc:
            return _evaluator_failure(name, weight, str(exc))
        try:
            pred = load_prediction(pred_dir / config.get("pred_file", "result.json"))
        except _PredictionError as exc:
            return _zero(name, weight, str(exc))
        err = abs(pred[key] - ref[key])
        fraction = credit(err, full_tol, zero_tol)
        return ScoreDetail(scorer_name=name, score=weight * fraction, max_score=weight, passed=err <= full_tol,
                           details={"key": key, "predicted": pred[key], "reference": ref[key], "abs_error": err,
                                    "credit_fraction": fraction},
                           message=f"{key}: |d| = {err:.3e}")
