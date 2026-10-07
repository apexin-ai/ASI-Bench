"""Scorers for the atomictoolkit vacancy MCP E2E fake task.

They compare the agent's ``result.json`` with the pre-generated reference,
using nothing but the standard library. Whether the numbers actually came from
the MCP tools is checked separately, from the run artefacts, by
scripts/mcp/e2e/verify_run.py.

Submission problems (a missing or malformed file, wrong values) are ordinary
zero scores. A missing or unreadable reference is an evaluator failure,
reported with ``scorer_internal_error: true``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from ai4sci_bench.core.scorer import Scorer, register_scorer
from ai4sci_bench.core.types import ScoreDetail

# result.json's contract, by how it is compared.
ENERGIES = ("energy_perfect_eV", "energy_vacancy_eV")
NUMBERS = (*ENERGIES, "vacancy_formation_energy_eV", "coordination_average", "bulk_modulus_GPa")
COUNTS = ("n_atoms_perfect",)


class _PredictionError(ValueError):
    pass


class _ReferenceError(ValueError):
    pass


# --------------------------------------------------------------------------
# result.json
# --------------------------------------------------------------------------

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


def _count(value, what: str) -> int:
    number = _number(value, what)
    if number != int(number) or number < 0:
        raise _PredictionError(f"{what} must be a non-negative whole number, got {value!r}")
    return int(number)


def load_result(path: Path) -> dict:
    if not path.is_file():
        raise _PredictionError(f"{path.name} not found")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _PredictionError(f"{path.name} is not valid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise _PredictionError(f"{path.name} must hold a JSON object")
    out: dict = {key: _number(data.get(key), key) for key in NUMBERS}
    out.update({key: _count(data.get(key), key) for key in COUNTS})
    return out


def load_reference(path: Path) -> dict:
    try:
        ref = json.loads(path.read_text(encoding="utf-8"))
        out = {key: float(ref[key]) for key in NUMBERS}
        out.update({key: int(ref[key]) for key in COUNTS})
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise _ReferenceError(f"unreadable reference: {exc}") from exc
    if not all(math.isfinite(out[key]) for key in NUMBERS) or out["n_atoms_perfect"] < 2:
        raise _ReferenceError("reference has a non-finite value or no supercell")
    return out


# --------------------------------------------------------------------------
# Credit
# --------------------------------------------------------------------------

def credit(error: float, full_tol: float, zero_tol: float) -> float:
    """1 at <= full_tol, 0 at >= zero_tol, log-linear in between."""
    if not math.isfinite(error):
        return 0.0
    if error <= full_tol:
        return 1.0
    if error >= zero_tol:
        return 0.0
    span = math.log10(zero_tol) - math.log10(full_tol)
    return max(0.0, min(1.0, 1.0 - (math.log10(error) - math.log10(full_tol)) / span))


def _zero(name: str, weight: float, message: str) -> ScoreDetail:
    return ScoreDetail(scorer_name=name, score=0.0, max_score=weight, passed=False,
                       details={"error": message}, message=message)


def _evaluator_failure(name: str, weight: float, message: str) -> ScoreDetail:
    return ScoreDetail(scorer_name=name, score=0.0, max_score=weight, passed=False,
                       details={"scorer_internal_error": True,
                                "failure_kind": "missing_evaluator_input", "error": message},
                       message=message)


def _load(name: str, weight: float, pred_dir: Path, ref_dir: Path, config: dict):
    """``(reference, prediction, failure)`` — at most one of the first two is None."""
    try:
        ref = load_reference(ref_dir / config.get("ref_file", "reference.json"))
    except _ReferenceError as exc:
        return None, None, _evaluator_failure(name, weight, str(exc))
    try:
        pred = load_result(pred_dir / config.get("pred_file", "result.json"))
    except _PredictionError as exc:
        return ref, None, _zero(name, weight, str(exc))
    return ref, pred, None


def _detail(name: str, weight: float, pred: dict, ref: dict, config: dict,
            numeric=(), exact=()) -> ScoreDetail:
    """Numbers by absolute error (log-linear credit), the rest all-or-nothing."""
    errors = {key: abs(pred[key] - ref[key]) for key in numeric}
    fractions = {key: credit(error, float(config["full_score_tol"]), float(config["zero_score_tol"]))
                 for key, error in errors.items()}
    fractions.update({key: 1.0 if pred[key] == ref[key] else 0.0 for key in exact})
    fraction = sum(fractions.values()) / len(fractions)
    keys = (*numeric, *exact)
    wrong = sorted(key for key, value in fractions.items() if value < 1.0)
    return ScoreDetail(scorer_name=name, score=weight * fraction, max_score=weight,
                       passed=not wrong,
                       details={"abs_errors": errors, "credit_fractions": fractions,
                                "predicted": {key: pred[key] for key in keys},
                                "reference": {key: ref[key] for key in keys}},
                       message="all match" if not wrong else
                       "; ".join(f"{key}: {pred[key]!r} vs {ref[key]!r}" for key in wrong))


class _Part(Scorer):
    NAME = ""
    NUMERIC: tuple[str, ...] = ()
    EXACT: tuple[str, ...] = ()

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        weight = float(config.get("weight", 1.0))
        ref, pred, failure = _load(self.NAME, weight, pred_dir, ref_dir, config)
        if failure is not None:
            return failure
        return _detail(self.NAME, weight, pred, ref, config, self.NUMERIC, self.EXACT)


# --------------------------------------------------------------------------
# Scorers
# --------------------------------------------------------------------------

@register_scorer("atomictoolkit_e2e_schema")
class Schema(Scorer):
    """Gate: result.json is complete and well typed."""

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        name = "atomictoolkit_e2e_schema"
        weight = float(config.get("weight", 1.0))
        try:
            load_result(pred_dir / config.get("pred_file", "result.json"))
        except _PredictionError as exc:
            return _zero(name, weight, str(exc))
        return ScoreDetail(scorer_name=name, score=weight, max_score=weight, passed=True,
                           details={"pred_file": config.get("pred_file", "result.json")},
                           message="result.json complete and well typed")


@register_scorer("atomictoolkit_e2e_energies")
class Energies(_Part):
    """The EMT energies of the perfect and the defective supercell."""
    NAME = "atomictoolkit_e2e_energies"
    NUMERIC = ENERGIES


@register_scorer("atomictoolkit_e2e_vacancy_formation")
class VacancyFormation(_Part):
    """E_vacancy - E_perfect * (N - 1) / N, from the two energies (unrelaxed)."""
    NAME = "atomictoolkit_e2e_vacancy_formation"
    NUMERIC = ("vacancy_formation_energy_eV",)


@register_scorer("atomictoolkit_e2e_bulk_modulus")
class BulkModulus(_Part):
    """The unit cell's bulk modulus as the server fits it, over the instance's strain range."""
    NAME = "atomictoolkit_e2e_bulk_modulus"
    NUMERIC = ("bulk_modulus_GPa",)


@register_scorer("atomictoolkit_e2e_coordination")
class Coordination(_Part):
    """The defective supercell's average coordination number, as the server counts it."""
    NAME = "atomictoolkit_e2e_coordination"
    NUMERIC = ("coordination_average",)


@register_scorer("atomictoolkit_e2e_supercell")
class Supercell(_Part):
    """The perfect supercell's atom count."""
    NAME = "atomictoolkit_e2e_supercell"
    EXACT = ("n_atoms_perfect",)
