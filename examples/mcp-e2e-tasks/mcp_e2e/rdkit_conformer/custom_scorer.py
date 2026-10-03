"""Scorers for the rdkit conformer MCP E2E fake task.

They only compare the agent's result.json and conformer.sdf with the
pre-generated reference and never run RDKit (the SDF is parsed here with the
standard library). Whether the values came from the MCP tools is checked
separately from the run artefacts by scripts/mcp/e2e/verify_run.py.

Submission problems (missing/invalid result.json or conformer.sdf, wrong atoms)
are ordinary zero scores. A missing or unreadable reference is an evaluator
failure and is reported with ``scorer_internal_error: true``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from ai4sci_bench.core.scorer import Scorer, register_scorer
from ai4sci_bench.core.types import ScoreDetail

CHARGES = ("max_partial_charge", "min_partial_charge")


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


def parse_molfile(text: str) -> list[tuple[str, float, float, float]]:
    """Atoms (symbol, x, y, z) of the first record of a V2000 molfile / SDF."""
    lines = text.split("$$$$")[0].splitlines()
    if len(lines) < 4 or "V2000" not in lines[3]:
        raise _PredictionError("not a V2000 molfile (no counts line)")
    try:
        count = int(lines[3][:3])
    except ValueError:
        raise _PredictionError(f"bad counts line {lines[3]!r}") from None
    if count <= 0 or len(lines) < 4 + count:
        raise _PredictionError(f"counts line announces {count} atoms, file has {len(lines) - 4} lines after it")
    atoms = []
    for line in lines[4:4 + count]:
        parts = line.split()
        if len(parts) < 4:
            raise _PredictionError(f"bad atom line {line!r}")
        try:
            x, y, z = (float(v) for v in parts[:3])
        except ValueError:
            raise _PredictionError(f"bad atom line {line!r}") from None
        atoms.append((parts[3], x, y, z))
    return atoms


def load_result(path: Path) -> dict:
    if not path.is_file():
        raise _PredictionError(f"{path.name} not found")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _PredictionError(f"{path.name} is not valid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise _PredictionError(f"{path.name} must hold a JSON object")
    return {key: _number(data.get(key), key) for key in CHARGES}


def load_conformer(path: Path) -> list[tuple[str, float, float, float]]:
    if not path.is_file():
        raise _PredictionError(f"{path.name} not found")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise _PredictionError(f"{path.name} is unreadable: {exc}") from None
    return parse_molfile(text)


def load_reference(path: Path) -> dict:
    try:
        ref = json.loads(path.read_text(encoding="utf-8"))
        out = {key: float(ref[key]) for key in CHARGES}
        out["atoms"] = [(str(a[0]), float(a[1]), float(a[2]), float(a[3])) for a in ref["atoms"]]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError, IndexError) as exc:
        raise _ReferenceError(f"unreadable reference: {exc}") from exc
    if not out["atoms"]:
        raise _ReferenceError("reference has no atoms")
    return out


def credit(abs_err: float, full_tol: float, zero_tol: float) -> float:
    """1 at <= full_tol, 0 at >= zero_tol, log-linear in between."""
    if abs_err <= full_tol:
        return 1.0
    if abs_err >= zero_tol:
        return 0.0
    span = math.log10(zero_tol) - math.log10(full_tol)
    return max(0.0, min(1.0, 1.0 - (math.log10(abs_err) - math.log10(full_tol)) / span))


def coordinate_error(pred: list[tuple], ref: list[tuple]) -> float:
    """Largest |Δx|, |Δy|, |Δz| over the atoms, without alignment (the seed fixes the frame);
    inf when the element sequence differs (other molecule, hydrogens added, atoms reordered)."""
    if [a[0] for a in pred] != [a[0] for a in ref]:
        return math.inf
    return max(abs(p - q) for a, b in zip(pred, ref) for p, q in zip(a[1:], b[1:]))


def _zero(name: str, weight: float, message: str) -> ScoreDetail:
    return ScoreDetail(scorer_name=name, score=0.0, max_score=weight, passed=False,
                       details={"error": message}, message=message)


def _evaluator_failure(name: str, weight: float, message: str) -> ScoreDetail:
    return ScoreDetail(scorer_name=name, score=0.0, max_score=weight, passed=False,
                       details={"scorer_internal_error": True, "failure_kind": "missing_evaluator_input",
                                "error": message}, message=message)


def _reference(name: str, weight: float, ref_dir: Path, config: dict):
    try:
        return load_reference(ref_dir / config.get("ref_file", "reference.json")), None
    except _ReferenceError as exc:
        return None, _evaluator_failure(name, weight, str(exc))


@register_scorer("rdkit_e2e_schema")
class ConformerSchema(Scorer):
    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        name = "rdkit_e2e_schema"
        weight = float(config.get("weight", 1.0))
        try:
            load_result(pred_dir / config.get("pred_file", "result.json"))
            atoms = load_conformer(pred_dir / config.get("sdf_file", "conformer.sdf"))
        except _PredictionError as exc:
            return _zero(name, weight, str(exc))
        return ScoreDetail(scorer_name=name, score=weight, max_score=weight, passed=True,
                           details={"n_atoms": len(atoms)},
                           message="result.json has both charges and conformer.sdf is a molfile")


@register_scorer("rdkit_e2e_conformer")
class SeededConformer(Scorer):
    """Coordinates of conformer.sdf against the seeded ETKDGv3 reference, atom by atom."""

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        name = "rdkit_e2e_conformer"
        weight = float(config.get("weight", 1.0))
        ref, failure = _reference(name, weight, ref_dir, config)
        if failure is not None:
            return failure
        try:
            atoms = load_conformer(pred_dir / config.get("sdf_file", "conformer.sdf"))
        except _PredictionError as exc:
            return _zero(name, weight, str(exc))
        err = coordinate_error(atoms, ref["atoms"])
        fraction = credit(err, float(config["full_score_tol"]), float(config["zero_score_tol"]))
        message = (f"atoms {''.join(a[0] for a in atoms)} != reference {''.join(a[0] for a in ref['atoms'])}"
                   if math.isinf(err) else f"max |dxyz| = {err:.2e} Angstrom")
        return ScoreDetail(scorer_name=name, score=weight * fraction, max_score=weight,
                           passed=err <= float(config["full_score_tol"]),
                           details={"n_atoms": len(atoms), "max_abs_error": None if math.isinf(err) else err,
                                    "credit_fraction": fraction}, message=message)


class _ChargeScorer(Scorer):
    key = ""

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        name = f"rdkit_e2e_{self.key}"
        weight = float(config.get("weight", 1.0))
        ref, failure = _reference(name, weight, ref_dir, config)
        if failure is not None:
            return failure
        try:
            pred = load_result(pred_dir / config.get("pred_file", "result.json"))
        except _PredictionError as exc:
            return _zero(name, weight, str(exc))
        err = abs(pred[self.key] - ref[self.key])
        fraction = credit(err, float(config["full_score_tol"]), float(config["zero_score_tol"]))
        return ScoreDetail(scorer_name=name, score=weight * fraction, max_score=weight,
                           passed=err <= float(config["full_score_tol"]),
                           details={"predicted": pred[self.key], "reference": ref[self.key], "abs_error": err,
                                    "credit_fraction": fraction}, message=f"{self.key}: |d| = {err:.3e} e")


@register_scorer("rdkit_e2e_max_partial_charge")
class MaxCharge(_ChargeScorer):
    key = "max_partial_charge"


@register_scorer("rdkit_e2e_min_partial_charge")
class MinCharge(_ChargeScorer):
    key = "min_partial_charge"
