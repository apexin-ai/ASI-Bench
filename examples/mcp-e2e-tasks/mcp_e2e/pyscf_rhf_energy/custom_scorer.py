"""Scorers for the pyscf MCP E2E fake task.

These only compare the agent's result.json with the pre-generated reference;
they never recompute PySCF. Whether the agent actually called the MCP tool is
checked separately from the run artefacts by scripts/mcp/e2e/verify_run.py.

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


class _PredictionError(ValueError):
    pass


def _load_prediction(path: Path) -> dict:
    if not path.is_file():
        raise _PredictionError(f"{path.name} not found")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _PredictionError(f"{path.name} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise _PredictionError(f"{path.name} must contain a JSON object")
    energy = data.get("energy_hartree")
    if isinstance(energy, bool) or not isinstance(energy, (int, float)):
        # Accept a numeric string as well; agents often copy tool text verbatim.
        try:
            energy = float(str(energy).strip())
        except (TypeError, ValueError):
            raise _PredictionError(f"energy_hartree must be a number, got {data.get('energy_hartree')!r}") from None
    energy = float(energy)
    if not math.isfinite(energy):
        raise _PredictionError(f"energy_hartree is not finite: {energy}")
    return {**data, "energy_hartree": energy}


def _evaluator_failure(name: str, max_score: float, kind: str, message: str) -> ScoreDetail:
    return ScoreDetail(
        scorer_name=name, score=0.0, max_score=max_score, passed=False,
        details={"scorer_internal_error": True, "failure_kind": kind, "error": message},
        message=message,
    )


def credit(abs_err: float, full_tol: float, zero_tol: float) -> float:
    """1 at <= full_tol, 0 at >= zero_tol, log-linear in between."""
    if abs_err <= full_tol:
        return 1.0
    if abs_err >= zero_tol:
        return 0.0
    span = math.log10(zero_tol) - math.log10(full_tol)
    return max(0.0, min(1.0, 1.0 - (math.log10(abs_err) - math.log10(full_tol)) / span))


@register_scorer("pyscf_e2e_result_schema")
class PySCFResultSchema(Scorer):
    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        weight = float(config.get("weight", 1.0))
        try:
            data = _load_prediction(pred_dir / config.get("pred_file", "result.json"))
        except _PredictionError as exc:
            return ScoreDetail(scorer_name="pyscf_e2e_result_schema", score=0.0, max_score=weight,
                               passed=False, details={"error": str(exc)}, message=str(exc))
        return ScoreDetail(scorer_name="pyscf_e2e_result_schema", score=weight, max_score=weight,
                           passed=True, details={"energy_hartree": data["energy_hartree"]},
                           message="result.json has a finite energy_hartree")


@register_scorer("pyscf_e2e_energy")
class PySCFEnergy(Scorer):
    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        name = "pyscf_e2e_energy"
        weight = float(config.get("weight", 1.0))
        full_tol = float(config.get("full_score_tol", 1e-6))
        zero_tol = float(config.get("zero_score_tol", 1e-3))

        ref_path = ref_dir / config.get("ref_file", "reference.json")
        if not ref_path.is_file():
            return _evaluator_failure(name, weight, "missing_evaluator_input", f"reference not found: {ref_path.name}")
        try:
            reference = json.loads(ref_path.read_text(encoding="utf-8"))
            ref_energy = float(reference["energy_hartree"])
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            return _evaluator_failure(name, weight, "missing_evaluator_input", f"unreadable reference: {exc}")

        try:
            data = _load_prediction(pred_dir / config.get("pred_file", "result.json"))
        except _PredictionError as exc:
            return ScoreDetail(scorer_name=name, score=0.0, max_score=weight, passed=False,
                               details={"error": str(exc)}, message=str(exc))

        abs_err = abs(data["energy_hartree"] - ref_energy)
        fraction = credit(abs_err, full_tol, zero_tol)
        details = {
            "pred_energy_hartree": data["energy_hartree"],
            "ref_energy_hartree": ref_energy,
            "abs_error_hartree": abs_err,
            "credit_fraction": fraction,
            "basis_reported": data.get("basis"),
            "basis_expected": reference.get("basis"),
            "basis_match": str(data.get("basis", "")).lower() == str(reference.get("basis", "")).lower(),
        }
        return ScoreDetail(scorer_name=name, score=weight * fraction, max_score=weight,
                           passed=abs_err <= full_tol, details=details,
                           message=f"|dE| = {abs_err:.3e} Ha")
