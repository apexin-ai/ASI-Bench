"""Scorers for the pyscf bond-stretch MCP E2E fake task.

They only compare the agent's result.json with the pre-generated reference and
never recompute PySCF. Whether the numbers came from the MCP tools, and whether
the plotting tool was called with them, is checked separately from the run
artefacts by scripts/mcp/e2e/verify_run.py.

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


def _numbers(value, what: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise _PredictionError(f"{what} must be a non-empty list of numbers")
    return [_number(v, f"{what}[{k}]") for k, v in enumerate(value)]


def load_prediction(path: Path) -> dict:
    if not path.is_file():
        raise _PredictionError(f"{path.name} not found")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _PredictionError(f"{path.name} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise _PredictionError(f"{path.name} must contain a JSON object")
    lengths = _numbers(data.get("bond_lengths"), "bond_lengths")
    energies = _numbers(data.get("energies_hartree"), "energies_hartree")
    if len(lengths) != len(energies):
        raise _PredictionError(f"bond_lengths has {len(lengths)} values, energies_hartree {len(energies)}")
    return {**data, "bond_lengths": lengths, "energies_hartree": energies,
            "min_bond_length": _number(data.get("min_bond_length"), "min_bond_length"),
            "min_energy_hartree": _number(data.get("min_energy_hartree"), "min_energy_hartree")}


def load_reference(path: Path) -> dict:
    if not path.is_file():
        raise _ReferenceError(f"reference not found: {path.name}")
    try:
        ref = json.loads(path.read_text(encoding="utf-8"))
        return {"bond_lengths": [float(v) for v in ref["bond_lengths"]],
                "energies_hartree": [float(v) for v in ref["energies_hartree"]],
                "min_bond_length": float(ref["min_bond_length"]),
                "min_energy_hartree": float(ref["min_energy_hartree"])}
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


@register_scorer("pyscf_e2e_scan_schema")
class ScanSchema(Scorer):
    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        name = "pyscf_e2e_scan_schema"
        weight = float(config.get("weight", 1.0))
        try:
            data = load_prediction(pred_dir / config.get("pred_file", "result.json"))
        except _PredictionError as exc:
            return _zero(name, weight, str(exc))
        return ScoreDetail(scorer_name=name, score=weight, max_score=weight, passed=True,
                           details={"num_points": len(data["energies_hartree"])},
                           message="result.json has matching bond_lengths/energies_hartree and a minimum")


@register_scorer("pyscf_e2e_scan_energies")
class ScanEnergies(Scorer):
    """Max |ΔE| over the scan grid; the grid itself must match the reference."""

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        name = "pyscf_e2e_scan_energies"
        weight = float(config.get("weight", 1.0))
        full_tol = float(config.get("full_score_tol", 1e-5))
        zero_tol = float(config.get("zero_score_tol", 1e-3))
        grid_tol = float(config.get("grid_tol", 1e-6))
        try:
            ref = load_reference(ref_dir / config.get("ref_file", "reference.json"))
        except _ReferenceError as exc:
            return _evaluator_failure(name, weight, str(exc))
        try:
            data = load_prediction(pred_dir / config.get("pred_file", "result.json"))
        except _PredictionError as exc:
            return _zero(name, weight, str(exc))
        if len(data["bond_lengths"]) != len(ref["bond_lengths"]):
            return _zero(name, weight, f"{len(data['bond_lengths'])} scan points, expected {len(ref['bond_lengths'])}")
        grid_err = max(abs(a - b) for a, b in zip(data["bond_lengths"], ref["bond_lengths"]))
        if grid_err > grid_tol:
            return _zero(name, weight, f"bond_lengths differ from the requested grid by up to {grid_err:.3e} Å")
        errors = [abs(a - b) for a, b in zip(data["energies_hartree"], ref["energies_hartree"])]
        worst = max(errors)
        fraction = credit(worst, full_tol, zero_tol)
        return ScoreDetail(scorer_name=name, score=weight * fraction, max_score=weight, passed=worst <= full_tol,
                           details={"max_abs_error_hartree": worst, "abs_errors_hartree": errors,
                                    "credit_fraction": fraction},
                           message=f"max |dE| = {worst:.3e} Ha over {len(errors)} points")


@register_scorer("pyscf_e2e_scan_minimum")
class ScanMinimum(Scorer):
    """The reported minimum must be the lowest-energy grid point of the reference scan."""

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        name = "pyscf_e2e_scan_minimum"
        weight = float(config.get("weight", 1.0))
        length_tol = float(config.get("length_tol", 1e-6))
        energy_tol = float(config.get("energy_tol", 1e-5))
        try:
            ref = load_reference(ref_dir / config.get("ref_file", "reference.json"))
        except _ReferenceError as exc:
            return _evaluator_failure(name, weight, str(exc))
        try:
            data = load_prediction(pred_dir / config.get("pred_file", "result.json"))
        except _PredictionError as exc:
            return _zero(name, weight, str(exc))
        d_len = abs(data["min_bond_length"] - ref["min_bond_length"])
        d_e = abs(data["min_energy_hartree"] - ref["min_energy_hartree"])
        ok = d_len <= length_tol and d_e <= energy_tol
        return ScoreDetail(scorer_name=name, score=weight if ok else 0.0, max_score=weight, passed=ok,
                           details={"pred_min_bond_length": data["min_bond_length"],
                                    "ref_min_bond_length": ref["min_bond_length"],
                                    "abs_error_min_energy_hartree": d_e},
                           message=f"|dr_min| = {d_len:.3e} Å, |dE_min| = {d_e:.3e} Ha")
