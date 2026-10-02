"""Scorers for the psi4 optimize → frequency MCP E2E fake task.

They only compare the agent's result.json with the pre-generated reference and
never run psi4 or PySCF. Whether the numbers came from the MCP tools is checked
separately from the run artefacts by scripts/mcp/e2e/verify_run.py.

Submission problems (missing/invalid result.json, wrong number of frequencies)
are ordinary zero scores. A missing or unreadable reference is an evaluator
failure and is reported with ``scorer_internal_error: true``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from ai4sci_bench.core.scorer import Scorer, register_scorer
from ai4sci_bench.core.types import ScoreDetail


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


def load_prediction(path: Path) -> dict:
    if not path.is_file():
        raise _PredictionError(f"{path.name} not found")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _PredictionError(f"{path.name} is not valid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise _PredictionError(f"{path.name} must hold a JSON object")
    freqs = data.get("frequencies_cm_inv")
    if not isinstance(freqs, list) or not freqs:
        raise _PredictionError("frequencies_cm_inv must be a non-empty list")
    return {
        "final_energy_hartree": _number(data.get("final_energy_hartree"), "final_energy_hartree"),
        "frequencies_cm_inv": [_number(f, f"frequencies_cm_inv[{k}]") for k, f in enumerate(freqs)],
        "zpe_hartree": _number(data.get("zpe_hartree"), "zpe_hartree"),
    }


def load_reference(path: Path) -> dict:
    try:
        ref = json.loads(path.read_text(encoding="utf-8"))
        out = {"final_energy_hartree": float(ref["final_energy_hartree"]),
               "frequencies_cm_inv": [float(f) for f in ref["frequencies_cm_inv"]],
               "zpe_hartree": float(ref["zpe_hartree"])}
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise _ReferenceError(f"unreadable reference: {exc}") from exc
    if not out["frequencies_cm_inv"]:
        raise _ReferenceError("reference has no frequencies")
    return out


def credit(abs_err: float, full_tol: float, zero_tol: float) -> float:
    """1 at <= full_tol, 0 at >= zero_tol, log-linear in between."""
    if abs_err <= full_tol:
        return 1.0
    if abs_err >= zero_tol:
        return 0.0
    span = math.log10(zero_tol) - math.log10(full_tol)
    return max(0.0, min(1.0, 1.0 - (math.log10(abs_err) - math.log10(full_tol)) / span))


def frequency_error(pred: list[float], ref: list[float]) -> float:
    """Max |Δν| after sorting both lists; inf when the mode count differs (e.g. translations/rotations included)."""
    if len(pred) != len(ref):
        return math.inf
    return max(abs(a - b) for a, b in zip(sorted(pred), sorted(ref)))


def _zero(name: str, weight: float, message: str) -> ScoreDetail:
    return ScoreDetail(scorer_name=name, score=0.0, max_score=weight, passed=False,
                       details={"error": message}, message=message)


def _evaluator_failure(name: str, weight: float, message: str) -> ScoreDetail:
    return ScoreDetail(scorer_name=name, score=0.0, max_score=weight, passed=False,
                       details={"scorer_internal_error": True, "failure_kind": "missing_evaluator_input",
                                "error": message}, message=message)


def _load_both(name: str, weight: float, pred_dir: Path, ref_dir: Path, config: dict):
    try:
        ref = load_reference(ref_dir / config.get("ref_file", "reference.json"))
    except _ReferenceError as exc:
        return None, None, _evaluator_failure(name, weight, str(exc))
    try:
        pred = load_prediction(pred_dir / config.get("pred_file", "result.json"))
    except _PredictionError as exc:
        return None, None, _zero(name, weight, str(exc))
    return pred, ref, None


@register_scorer("psi4_e2e_schema")
class OptFreqSchema(Scorer):
    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        name = "psi4_e2e_schema"
        weight = float(config.get("weight", 1.0))
        try:
            pred = load_prediction(pred_dir / config.get("pred_file", "result.json"))
        except _PredictionError as exc:
            return _zero(name, weight, str(exc))
        return ScoreDetail(scorer_name=name, score=weight, max_score=weight, passed=True,
                           details={"n_frequencies": len(pred["frequencies_cm_inv"])},
                           message="result.json has an energy, a frequency list and a ZPE")


class _ScalarScorer(Scorer):
    key = ""
    unit = ""

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        name = f"psi4_e2e_{self.key}"
        weight = float(config.get("weight", 1.0))
        pred, ref, failure = _load_both(name, weight, pred_dir, ref_dir, config)
        if failure is not None:
            return failure
        field = {"energy": "final_energy_hartree", "zpe": "zpe_hartree"}[self.key]
        err = abs(pred[field] - ref[field])
        fraction = credit(err, float(config["full_score_tol"]), float(config["zero_score_tol"]))
        return ScoreDetail(scorer_name=name, score=weight * fraction, max_score=weight,
                           passed=err <= float(config["full_score_tol"]),
                           details={"predicted": pred[field], "reference": ref[field], "abs_error": err,
                                    "credit_fraction": fraction},
                           message=f"{field}: |d| = {err:.3e} {self.unit}")


@register_scorer("psi4_e2e_energy")
class OptEnergy(_ScalarScorer):
    key = "energy"
    unit = "Eh"


@register_scorer("psi4_e2e_zpe")
class ZeroPointEnergy(_ScalarScorer):
    key = "zpe"
    unit = "Eh"


@register_scorer("psi4_e2e_frequencies")
class HarmonicFrequencies(Scorer):
    """Max |Δν| over the sorted modes; a different mode count scores zero."""

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        name = "psi4_e2e_frequencies"
        weight = float(config.get("weight", 1.0))
        pred, ref, failure = _load_both(name, weight, pred_dir, ref_dir, config)
        if failure is not None:
            return failure
        err = frequency_error(pred["frequencies_cm_inv"], ref["frequencies_cm_inv"])
        fraction = credit(err, float(config["full_score_tol"]), float(config["zero_score_tol"]))
        details = {"predicted": pred["frequencies_cm_inv"], "reference": ref["frequencies_cm_inv"],
                   "max_abs_error": None if math.isinf(err) else err, "credit_fraction": fraction}
        message = (f"{len(pred['frequencies_cm_inv'])} frequencies, reference has {len(ref['frequencies_cm_inv'])}"
                   if math.isinf(err) else f"max |Δν| = {err:.3f} cm^-1")
        return ScoreDetail(scorer_name=name, score=weight * fraction, max_score=weight,
                           passed=err <= float(config["full_score_tol"]), details=details, message=message)
