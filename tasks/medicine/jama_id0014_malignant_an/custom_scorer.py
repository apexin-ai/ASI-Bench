"""Deterministic scorer for multi-patient clinical synthesis.

Scores seed-specific decisions rather than prose keywords.  The structured
contract makes contradictions, over-testing, and copied one-size-fits-all
answers directly measurable.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ai4sci_bench.core.scorer import Scorer, register_scorer
from ai4sci_bench.core.types import ScoreDetail


def _load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, f"missing {path.name}"
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"invalid {path.name}: {exc}"
    if not isinstance(value, dict):
        return None, f"{path.name} must contain a JSON object"
    return value, None


def _validate_submission(pred: dict[str, Any], gt: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    rows = pred.get("assessments")
    if not isinstance(rows, list):
        return ["assessments must be a list"]
    expected_ids = {row["case_id"] for row in gt["assessments"]}
    actual_ids = [row.get("case_id") for row in rows if isinstance(row, dict)]
    if len(rows) != len(expected_ids) or set(actual_ids) != expected_ids or len(actual_ids) != len(set(actual_ids)):
        errors.append("assess every expected case exactly once")
    diagnoses = set(gt["diagnoses"])
    action_ids = set(gt["action_costs"])
    references = {row["case_id"]: row for row in gt["assessments"]}
    for index, row in enumerate(rows):
        label = f"assessment[{index}]"
        if not isinstance(row, dict):
            errors.append(f"{label} must be an object")
            continue
        if row.get("primary_diagnosis") not in diagnoses:
            errors.append(f"{label}.primary_diagnosis is invalid")
        probs = row.get("differential_probabilities")
        if not isinstance(probs, dict) or set(probs) != diagnoses:
            errors.append(f"{label}.differential_probabilities must contain all diagnosis IDs")
        else:
            try:
                values = [float(probs[key]) for key in diagnoses]
                if any(not math.isfinite(x) or x < 0 or x > 1 for x in values):
                    raise ValueError
                if abs(sum(values) - 1.0) > 0.01:
                    errors.append(f"{label}.differential_probabilities must sum to 1")
            except (TypeError, ValueError):
                errors.append(f"{label}.differential_probabilities must be finite values in [0,1]")
        for field, limit in (("evidence_for", 4), ("evidence_against", 3)):
            values = row.get(field)
            if not isinstance(values, list) or len(values) > limit or len(values) != len(set(values)):
                errors.append(f"{label}.{field} must be a unique list of at most {limit} IDs")
            elif row.get("case_id") in references and any(
                value not in references[row["case_id"]]["valid_event_ids"] for value in values
            ):
                errors.append(f"{label}.{field} contains an event ID from the wrong patient")
        actions = row.get("next_actions")
        if not isinstance(actions, list) or any(x not in action_ids for x in actions) or len(actions) != len(set(actions)):
            errors.append(f"{label}.next_actions contains invalid or duplicate IDs")
        cf = row.get("counterfactual")
        if not isinstance(cf, dict) or cf.get("revised_diagnosis") not in diagnoses or not isinstance(cf.get("remove_event_id"), str):
            errors.append(f"{label}.counterfactual is invalid")
        elif row.get("case_id") in references and cf["remove_event_id"] not in references[row["case_id"]]["valid_event_ids"]:
            errors.append(f"{label}.counterfactual contains an event ID from the wrong patient")
        rationale = row.get("rationale")
        if not isinstance(rationale, str) or len(rationale) > 500:
            errors.append(f"{label}.rationale must be a string of at most 500 characters")
    return errors


def _set_f1(predicted: Any, expected: Any) -> float:
    pred = set(predicted) if isinstance(predicted, list) else set()
    ref = set(expected) if isinstance(expected, list) else set()
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    overlap = len(pred & ref)
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def _calibration_score(pred: Any, ref: dict[str, float], diagnoses: list[str]) -> tuple[float, float]:
    try:
        values = {key: float(pred[key]) for key in diagnoses}
    except (KeyError, TypeError, ValueError):
        return 0.0, float("inf")
    brier = sum((values[key] - float(ref[key])) ** 2 for key in diagnoses) / len(diagnoses)
    # Full credit at the reference distribution, tapering to zero at RMSE 0.30.
    fraction = max(0.0, 1.0 - math.sqrt(brier) / 0.30)
    return fraction, brier


@register_scorer("jama_clinical_contract")
class ClinicalContractGate(Scorer):
    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        pred, pred_error = _load_json(pred_dir / "final_assessment.json")
        gt, gt_error = _load_json(ref_dir / "ground_truth.json")
        errors = [error for error in (pred_error, gt_error) if error]
        if pred is not None and gt is not None:
            errors.extend(_validate_submission(pred, gt))
        passed = not errors
        return ScoreDetail(
            scorer_name="jama_clinical_contract",
            score=1.0 if passed else 0.0,
            max_score=1.0,
            passed=passed,
            details={"errors": errors},
            message="valid structured clinical assessment" if passed else "; ".join(errors[:4]),
        )


@register_scorer("jama_clinical_reasoning")
class ClinicalReasoningScorer(Scorer):
    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        weight = float(config.get("weight", 100.0))
        pred, pred_error = _load_json(pred_dir / "final_assessment.json")
        gt, gt_error = _load_json(ref_dir / "ground_truth.json")
        if pred is None or gt is None:
            message = pred_error or gt_error or "could not load assessment"
            return ScoreDetail("jama_clinical_reasoning", 0.0, weight, False, {"error": message}, message)
        errors = _validate_submission(pred, gt)
        if errors:
            return ScoreDetail("jama_clinical_reasoning", 0.0, weight, False, {"errors": errors}, "invalid output contract")

        pred_by_id = {row["case_id"]: row for row in pred["assessments"]}
        costs = gt["action_costs"]
        budget = float(gt["budget"])
        diagnoses = list(gt["diagnoses"])
        case_details: dict[str, Any] = {}
        case_scores: list[float] = []
        for ref in gt["assessments"]:
            row = pred_by_id[ref["case_id"]]
            diagnosis = 30.0 if row["primary_diagnosis"] == ref["primary_diagnosis"] else 0.0
            calibration_fraction, brier = _calibration_score(
                row["differential_probabilities"], ref["differential_probabilities"], diagnoses
            )
            calibration = 15.0 * calibration_fraction
            evidence_for = 10.0 * _set_f1(row["evidence_for"], ref["evidence_for"])
            evidence_against = 5.0 * _set_f1(row["evidence_against"], ref["evidence_against"])
            action_cost = sum(float(costs[x]) for x in row["next_actions"])
            action_fraction = _set_f1(row["next_actions"], ref["next_actions"])
            actions = 0.0 if action_cost > budget else 20.0 * action_fraction
            occult = 10.0 if row.get("occult_primary") == ref["occult_primary"] else 0.0
            disposition = 5.0 if row.get("disposition") == ref["disposition"] else 0.0
            counter = row["counterfactual"]
            counterfactual = 5.0 if (
                counter.get("remove_event_id") == ref["counterfactual"]["remove_event_id"]
                and counter.get("revised_diagnosis") == ref["counterfactual"]["revised_diagnosis"]
            ) else 0.0
            points = diagnosis + calibration + evidence_for + evidence_against + actions + occult + disposition + counterfactual
            case_scores.append(points)
            case_details[ref["case_id"]] = {
                "score": points,
                "diagnosis": diagnosis,
                "calibration": calibration,
                "brier": brier,
                "evidence_for": evidence_for,
                "evidence_against": evidence_against,
                "actions": actions,
                "action_cost": action_cost,
                "occult_primary": occult,
                "disposition": disposition,
                "counterfactual": counterfactual,
            }

        # The weakest patient matters, preventing one excellent case from hiding a
        # copied or contradictory answer on another patient.
        mean_score = sum(case_scores) / len(case_scores)
        min_score = min(case_scores)
        aggregate = 0.75 * mean_score + 0.25 * min_score
        fraction = max(0.0, min(1.0, aggregate / 100.0))
        return ScoreDetail(
            scorer_name="jama_clinical_reasoning",
            score=weight * fraction,
            max_score=weight,
            passed=fraction > 0.0,
            details={
                "aggregation": "0.75 * mean(case scores) + 0.25 * minimum(case score)",
                "mean_case_score": mean_score,
                "minimum_case_score": min_score,
                "case_details": case_details,
            },
            message=f"clinical synthesis score {aggregate:.2f}/100",
        )
