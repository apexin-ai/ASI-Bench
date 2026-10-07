"""Scorers for the OpenROAD tiny-floorplan MCP E2E fake task.

They compare the agent's ``result.json`` with the pre-generated closed-form
reference (counts, die box, design area and utilisation, total HPWL, the area
report line) and never run openroad. Every quantity is an exact integer by
construction (see generate_gt.py), so each field is right or wrong. Whether the
numbers came from the MCP session is checked separately from the run artefacts
by scripts/mcp/e2e/verify_run.py.

Submission problems (a missing or malformed ``result.json``, wrong values) are
ordinary zero scores. A missing or unreadable reference is an evaluator
failure, reported with ``scorer_internal_error: true``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from ai4sci_bench.core.scorer import Scorer, register_scorer
from ai4sci_bench.core.types import ScoreDetail

NUMBERS = ("instance_count", "net_count", "io_pin_count", "die_width_dbu", "die_height_dbu",
           "design_area_um2", "utilization_percent", "hpwl_total_dbu")
TEXTS = ("design", "session_id", "area_report_line")


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
    for key in TEXTS:
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise _PredictionError(f"{key} must be a non-empty string, got {value!r}")
        out[key] = value.strip()
    return out


def load_reference(path: Path) -> dict:
    try:
        ref = json.loads(path.read_text(encoding="utf-8"))
        out: dict = {key: float(ref[key]) for key in NUMBERS}
        out.update({key: str(ref[key]) for key in TEXTS})
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise _ReferenceError(f"unreadable reference: {exc}") from exc
    if out["instance_count"] <= 0 or out["hpwl_total_dbu"] <= 0:
        raise _ReferenceError("reference has no design")
    return out


def _zero(name: str, weight: float, message: str) -> ScoreDetail:
    return ScoreDetail(scorer_name=name, score=0.0, max_score=weight, passed=False,
                       details={"error": message}, message=message)


def _evaluator_failure(name: str, weight: float, message: str) -> ScoreDetail:
    return ScoreDetail(scorer_name=name, score=0.0, max_score=weight, passed=False,
                       details={"scorer_internal_error": True,
                                "failure_kind": "missing_evaluator_input", "error": message},
                       message=message)


@register_scorer("openroad_e2e_schema")
class FloorplanSchema(Scorer):
    """Gate: result.json holds every field with the right type."""

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        name = "openroad_e2e_schema"
        weight = float(config.get("weight", 1.0))
        try:
            load_result(pred_dir / config.get("pred_file", "result.json"))
        except _PredictionError as exc:
            return _zero(name, weight, str(exc))
        return ScoreDetail(scorer_name=name, score=weight, max_score=weight, passed=True,
                           details={}, message="result.json complete")


class _Fields(Scorer):
    """Equal share per field; a number must equal the reference exactly (all are integers),
    a text must equal it after stripping."""

    NAME = ""
    FIELDS: tuple[str, ...] = ()

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        name, weight = self.NAME, float(config.get("weight", 1.0))
        try:
            ref = load_reference(ref_dir / config.get("ref_file", "reference.json"))
        except _ReferenceError as exc:
            return _evaluator_failure(name, weight, str(exc))
        try:
            pred = load_result(pred_dir / config.get("pred_file", "result.json"))
        except _PredictionError as exc:
            return _zero(name, weight, str(exc))
        verdicts = {key: pred[key] == ref[key] for key in self.FIELDS}
        wrong = [f"{key}={pred[key]!r} (expected {ref[key]!r})" for key, ok in verdicts.items() if not ok]
        fraction = sum(verdicts.values()) / len(verdicts)
        return ScoreDetail(scorer_name=name, score=weight * fraction, max_score=weight, passed=not wrong,
                           details={"correct": verdicts, "submitted": {k: pred[k] for k in self.FIELDS},
                                    "expected": {k: ref[k] for k in self.FIELDS}},
                           message="all correct" if not wrong else "wrong: " + "; ".join(wrong))


@register_scorer("openroad_e2e_counts")
class Counts(_Fields):
    """Instances, nets and I/O pins of the design read into the session."""
    NAME = "openroad_e2e_counts"
    FIELDS = ("instance_count", "net_count", "io_pin_count")


@register_scorer("openroad_e2e_die")
class Die(_Fields):
    """Die width and height in DBU, from the session's database."""
    NAME = "openroad_e2e_die"
    FIELDS = ("die_width_dbu", "die_height_dbu")


@register_scorer("openroad_e2e_area")
class Area(_Fields):
    """report_design_area in microns: the cell area and the utilisation of the core."""
    NAME = "openroad_e2e_area"
    FIELDS = ("design_area_um2", "utilization_percent")


@register_scorer("openroad_e2e_hpwl")
class Hpwl(_Fields):
    """Total half-perimeter wirelength over the pin-shape centres, in DBU."""
    NAME = "openroad_e2e_hpwl"
    FIELDS = ("hpwl_total_dbu",)


@register_scorer("openroad_e2e_report_line")
class ReportLine(_Fields):
    """The area report line, as grep_session_output found it in the retained output."""
    NAME = "openroad_e2e_report_line"
    FIELDS = ("area_report_line",)
