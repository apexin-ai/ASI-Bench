"""Scorers for the AlphaFold DB isoform profile MCP E2E fake task.

They only compare the agent's result.json with the pre-generated reference and
never contact AlphaFold DB. Whether the values came from the MCP tools and
whether AlphaFold DB was reached some other way is checked separately from the
run artefacts by scripts/mcp/e2e/verify_run.py.

Numbers are compared numerically (``192`` == ``"192"``, ``0.999`` ==
``"0.9990"``) but to printing precision, and entry ids as written: the
verifier's ``answer_from_tool`` check requires the value the tool returned, so
the scorer must not accept a re-cased id or a score the tool never reported
(the tool already serves the rounded 4-decimal mean; no tool-derived answer
lies between two of its values).

The hard gate is structural only, like the other MCP E2E fake tasks: it asks
for the six keys with usable types, so a wrong ranking or score costs that
field's own weight instead of zeroing the instance.

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

INTEGER_FIELDS = ("canonical_length", "max_am_residue", "n_am_above_threshold")
SCORE_TOL = 1e-9                    # printing precision: the answer is the value the tool reported
AM_FIELDS = ("max_am_residue", "max_am_score", "n_am_above_threshold")


class _PredictionError(ValueError):
    pass


class _ReferenceError(ValueError):
    pass


def _number(value, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise _PredictionError(f"{what} must be a number, got {value!r}")
    try:
        number = float(str(value).strip())
    except ValueError:
        raise _PredictionError(f"{what} must be a number, got {value!r}") from None
    if not math.isfinite(number):
        raise _PredictionError(f"{what} must be finite, got {value!r}")
    return number


def _integer(value, what: str) -> int:
    number = _number(value, what)
    if number != int(number):
        raise _PredictionError(f"{what} must be a whole number, got {value!r}")
    return int(number)


def _entry_id(value, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _PredictionError(f"{what} must be a non-empty string, got {value!r}")
    return value.strip()


def _normalize(data: dict, where: str) -> dict:
    ranking = data.get("entry_ids_by_length_desc")
    if not isinstance(ranking, list) or not ranking:
        raise _PredictionError(f"{where}['entry_ids_by_length_desc'] must be a non-empty list, got {ranking!r}")
    out = {"entry_ids_by_length_desc": [_entry_id(v, f"{where}['entry_ids_by_length_desc'][{k}]")
                                        for k, v in enumerate(ranking)],
           "longest_entry_id": _entry_id(data.get("longest_entry_id"), f"{where}['longest_entry_id']"),
           "max_am_score": _number(data.get("max_am_score"), f"{where}['max_am_score']")}
    out.update({field: _integer(data.get(field), f"{where}[{field!r}]") for field in INTEGER_FIELDS})
    return out


def load_prediction(path: Path) -> dict:
    if not path.is_file():
        raise _PredictionError(f"{path.name} not found")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _PredictionError(f"{path.name} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise _PredictionError(f"{path.name} must contain a JSON object")
    return _normalize(data, path.name)


def load_reference(path: Path) -> dict:
    if not path.is_file():
        raise _ReferenceError(f"reference not found: {path.name}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("not a JSON object")
        return _normalize(data, path.name)
    except (OSError, UnicodeError, json.JSONDecodeError, _PredictionError, ValueError) as exc:
        raise _ReferenceError(f"unreadable reference: {exc}") from exc


def _zero(name: str, weight: float, message: str) -> ScoreDetail:
    return ScoreDetail(scorer_name=name, score=0.0, max_score=weight, passed=False,
                       details={"error": message}, message=message)


def _evaluator_failure(name: str, weight: float, message: str) -> ScoreDetail:
    return ScoreDetail(scorer_name=name, score=0.0, max_score=weight, passed=False,
                       details={"scorer_internal_error": True, "failure_kind": "missing_evaluator_input",
                                "error": message}, message=message)


def _load_both(name: str, weight: float, pred_dir: Path, ref_dir: Path, config: dict):
    """(prediction, reference, None) or (None, None, failure ScoreDetail)."""
    try:
        ref = load_reference(ref_dir / config.get("ref_file", "reference.json"))
    except _ReferenceError as exc:
        return None, None, _evaluator_failure(name, weight, str(exc))
    try:
        pred = load_prediction(pred_dir / config.get("pred_file", "result.json"))
    except _PredictionError as exc:
        return None, None, _zero(name, weight, str(exc))
    return pred, ref, None


def _same(field: str, pred: dict, ref: dict) -> bool:
    if field == "max_am_score":
        return abs(pred[field] - ref[field]) <= SCORE_TOL
    return pred[field] == ref[field]


def _per_field(name: str, weight: float, fields: tuple[str, ...], pred: dict, ref: dict) -> ScoreDetail:
    correct = {field: _same(field, pred, ref) for field in fields}
    fraction = sum(correct.values()) / len(fields)
    return ScoreDetail(scorer_name=name, score=weight * fraction, max_score=weight, passed=fraction == 1.0,
                       details={"per_field_correct": correct,
                                "prediction": {field: pred[field] for field in fields},
                                "reference": {field: ref[field] for field in fields}},
                       message=f"{sum(correct.values())}/{len(fields)} fields match the reference")


class _Fields(Scorer):
    NAME = ""
    FIELDS: tuple[str, ...] = ()

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        weight = float(config.get("weight", 1.0))
        pred, ref, failure = _load_both(self.NAME, weight, pred_dir, ref_dir, config)
        if failure:
            return failure
        return _per_field(self.NAME, weight, self.FIELDS, pred, ref)


@register_scorer("afdb_e2e_schema")
class AfdbSchema(Scorer):
    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        name = "afdb_e2e_schema"
        weight = float(config.get("weight", 1.0))
        try:
            data = load_prediction(pred_dir / config.get("pred_file", "result.json"))
        except _PredictionError as exc:
            return _zero(name, weight, str(exc))
        return ScoreDetail(scorer_name=name, score=weight, max_score=weight, passed=True,
                           details={"num_entries": len(data["entry_ids_by_length_desc"])},
                           message="result.json carries all six profile fields")


@register_scorer("afdb_e2e_isoforms")
class AfdbIsoforms(_Fields):
    """Equal credit for the full ranking (exact order, exactly the listed entries) and the longest entry."""
    NAME = "afdb_e2e_isoforms"
    FIELDS = ("entry_ids_by_length_desc", "longest_entry_id")


@register_scorer("afdb_e2e_canonical_length")
class AfdbCanonicalLength(_Fields):
    NAME = "afdb_e2e_canonical_length"
    FIELDS = ("canonical_length",)


@register_scorer("afdb_e2e_am_profile")
class AfdbAmProfile(_Fields):
    """Equal credit for the highest-scoring residue, its score and the count above the threshold."""
    NAME = "afdb_e2e_am_profile"
    FIELDS = AM_FIELDS
