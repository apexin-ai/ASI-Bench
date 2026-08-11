"""Structured scorer for decision-specific disclosure ethics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai4sci_bench.core.scorer import Scorer, register_scorer
from ai4sci_bench.core.types import ScoreDetail


def _load(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, f"missing {path.name}"
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"invalid {path.name}: {exc}"
    return (value, None) if isinstance(value, dict) else (None, f"{path.name} must be an object")


def _validate(pred: dict[str, Any], gt: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    rows = pred.get("cases")
    if not isinstance(rows, list):
        return ["cases must be a list"]
    refs = {row["case_id"]: row for row in gt["cases"]}
    ids = [row.get("case_id") for row in rows if isinstance(row, dict)]
    if len(rows) != len(refs) or set(ids) != set(refs) or len(ids) != len(set(ids)):
        errors.append("assess each expected case exactly once")
    for index, row in enumerate(rows):
        label = f"cases[{index}]"
        if not isinstance(row, dict):
            errors.append(f"{label} must be an object")
            continue
        if row.get("capacity_for_information_choice") not in gt["capacity_states"]:
            errors.append(f"{label}.capacity_for_information_choice is invalid")
        if row.get("capacity_for_treatment") not in gt["capacity_states"]:
            errors.append(f"{label}.capacity_for_treatment is invalid")
        if row.get("disclosure_plan") not in gt["disclosure_plans"]:
            errors.append(f"{label}.disclosure_plan is invalid")
        if row.get("surrogate_role") not in gt["surrogate_roles"]:
            errors.append(f"{label}.surrogate_role is invalid")
        for field, allowed, exact, maximum in (
            ("immediate_actions", set(gt["action_ids"]), None, 4),
            ("ethical_basis", set(gt["principle_ids"]), 3, 3),
        ):
            values = row.get(field)
            if not isinstance(values, list) or len(values) != len(set(values)) or any(value not in allowed for value in values):
                errors.append(f"{label}.{field} contains invalid or duplicate IDs")
            elif (exact is not None and len(values) != exact) or len(values) > maximum:
                errors.append(f"{label}.{field} has the wrong number of IDs")
        evidence = row.get("decisive_evidence")
        ref = refs.get(row.get("case_id"))
        if not isinstance(evidence, list) or len(evidence) != 3 or len(set(evidence)) != 3:
            errors.append(f"{label}.decisive_evidence must contain three unique IDs")
        elif ref and any(value not in ref["valid_event_ids"] for value in evidence):
            errors.append(f"{label}.decisive_evidence contains an ID from the wrong case")
        contingency = row.get("contingency")
        if not isinstance(contingency, dict) or not isinstance(contingency.get("trigger"), str) or not isinstance(contingency.get("action"), str):
            errors.append(f"{label}.contingency is invalid")
        justification = row.get("justification")
        if not isinstance(justification, str) or len(justification) > 500:
            errors.append(f"{label}.justification must be at most 500 characters")
    return errors


def _f1(predicted: Any, expected: Any) -> float:
    pred = set(predicted) if isinstance(predicted, list) else set()
    ref = set(expected) if isinstance(expected, list) else set()
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    overlap = len(pred & ref)
    return 0.0 if not overlap else 2 * overlap / (len(pred) + len(ref))


@register_scorer("ethics_decision_contract")
class EthicsDecisionContract(Scorer):
    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        pred, pred_error = _load(pred_dir / "ethics_plan.json")
        gt, gt_error = _load(ref_dir / "ground_truth.json")
        errors = [error for error in (pred_error, gt_error) if error]
        if pred is not None and gt is not None:
            errors.extend(_validate(pred, gt))
        passed = not errors
        return ScoreDetail(
            "ethics_decision_contract", 1.0 if passed else 0.0, 1.0, passed,
            {"errors": errors}, "valid ethics plan" if passed else "; ".join(errors[:4]),
        )


@register_scorer("ethics_decision_reasoning")
class EthicsDecisionReasoning(Scorer):
    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        weight = float(config.get("weight", 100.0))
        pred, pred_error = _load(pred_dir / "ethics_plan.json")
        gt, gt_error = _load(ref_dir / "ground_truth.json")
        if pred is None or gt is None:
            message = pred_error or gt_error or "load failure"
            return ScoreDetail("ethics_decision_reasoning", 0.0, weight, False, {"error": message}, message)
        errors = _validate(pred, gt)
        if errors:
            return ScoreDetail("ethics_decision_reasoning", 0.0, weight, False, {"errors": errors}, "invalid output contract")
        pred_by_id = {row["case_id"]: row for row in pred["cases"]}
        details: dict[str, Any] = {}
        scores: list[float] = []
        for ref in gt["cases"]:
            row = pred_by_id[ref["case_id"]]
            parts = {
                "information_capacity": 12.5 if row["capacity_for_information_choice"] == ref["capacity_for_information_choice"] else 0.0,
                "treatment_capacity": 12.5 if row["capacity_for_treatment"] == ref["capacity_for_treatment"] else 0.0,
                "disclosure_plan": 20.0 if row["disclosure_plan"] == ref["disclosure_plan"] else 0.0,
                "surrogate_role": 15.0 if row["surrogate_role"] == ref["surrogate_role"] else 0.0,
                "immediate_actions": 15.0 * _f1(row["immediate_actions"], ref["immediate_actions"]),
                "ethical_basis": 10.0 * _f1(row["ethical_basis"], ref["ethical_basis"]),
                "decisive_evidence": 10.0 * _f1(row["decisive_evidence"], ref["decisive_evidence"]),
                "contingency": 5.0 if row["contingency"] == ref["contingency"] else 0.0,
            }
            total = sum(parts.values())
            scores.append(total)
            details[ref["case_id"]] = {"score": total, **parts}
        mean_score = sum(scores) / len(scores)
        minimum = min(scores)
        aggregate = 0.7 * mean_score + 0.3 * minimum
        fraction = max(0.0, min(1.0, aggregate / 100.0))
        return ScoreDetail(
            "ethics_decision_reasoning", weight * fraction, weight, fraction > 0,
            {"aggregation": "0.7*mean + 0.3*minimum", "mean": mean_score, "minimum": minimum, "cases": details},
            f"ethics decision score {aggregate:.2f}/100",
        )
