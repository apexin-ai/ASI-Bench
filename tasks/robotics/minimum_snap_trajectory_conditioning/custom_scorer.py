"""Custom scorer for robotics.minimum_snap_trajectory_conditioning."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ai4sci_bench.core.scorer import Scorer, register_scorer
from ai4sci_bench.core.types import ScoreDetail


AXES = ("x", "y", "z")
SCORE_WEIGHTS = {
    "query": 30.0,
    "trajectory": 41.0,
    "objective": 19.75,
    "time": 6.0,
    "constraint": 1.5,
    "outlier": 0.75,
    "report": 0.5,
    "figure": 0.5,
}


def _load_dropout_windows(ref_dir: Path) -> list[list[float]]:
    try:
        payload = json.loads((ref_dir / "residual_field_ref.json").read_text(encoding="utf-8"))
        return [[float(a), float(b)] for a, b in payload.get("windows", [])]
    except Exception:
        return []


def _linear_desc(value: float, full: float, zero: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    if value <= full:
        return 1.0
    if value >= zero:
        return 0.0
    return float((zero - value) / max(zero - full, 1.0e-30))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _read_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"missing file: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = required - fields
        if missing:
            raise ValueError(f"{path} missing columns {sorted(missing)}")
        return list(reader)


def _falling(power: int, order: int) -> float:
    out = 1.0
    for value in range(power - order + 1, power + 1):
        out *= float(value)
    return out


def _basis_row(degree: int, delta: float, derivative_order: int) -> np.ndarray:
    row = np.zeros(degree + 1, dtype=np.float64)
    for power in range(derivative_order, degree + 1):
        row[power] = _falling(power, derivative_order) * (delta ** (power - derivative_order))
    return row


def _load_waypoints(ref_dir: Path) -> np.ndarray:
    rows = _read_rows(ref_dir / "waypoints_ref.csv", {"waypoint_id", "x", "y", "z"})
    rows = sorted(rows, key=lambda row: int(row["waypoint_id"]))
    return np.asarray([[float(row[axis]) for axis in AXES] for row in rows], dtype=np.float64)


def _load_bounds(ref_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = _read_rows(
        ref_dir.parent / "data" / "segment_time_bounds.csv",
        {"segment", "min_duration", "max_duration"},
    )
    rows = sorted(rows, key=lambda row: int(row["segment"]))
    lower = np.asarray([float(row["min_duration"]) for row in rows], dtype=np.float64)
    upper = np.asarray([float(row["max_duration"]) for row in rows], dtype=np.float64)
    return lower, upper


def _load_windows(ref_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = _read_rows(
        ref_dir.parent / "data" / "waypoint_windows.csv",
        {"waypoint_id", "t_min", "t_max"},
    )
    rows = sorted(rows, key=lambda row: int(row["waypoint_id"]))
    t_min = np.asarray([float(row["t_min"]) for row in rows], dtype=np.float64)
    t_max = np.asarray([float(row["t_max"]) for row in rows], dtype=np.float64)
    return t_min, t_max


def _load_constraints(ref_dir: Path) -> list[dict[str, Any]]:
    rows = _read_rows(
        ref_dir.parent / "data" / "constraints.csv",
        {"waypoint_id", "axis", "derivative_order", "value"},
    )
    out = []
    for row in rows:
        out.append(
            {
                "waypoint_id": int(row["waypoint_id"]),
                "axis": row["axis"],
                "derivative_order": int(row["derivative_order"]),
                "value": float(row["value"]),
            }
        )
    return out


def _load_coeff(path: Path, expected_shape: tuple[int, int, int]) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"missing coefficient file: {path}")
    coeff = np.load(path).astype(np.float64)
    if coeff.shape != expected_shape:
        raise ValueError(f"segment_coefficients shape {coeff.shape} != expected {expected_shape}")
    if not np.isfinite(coeff).all():
        raise ValueError("segment_coefficients contains non-finite values")
    return coeff


def _load_times(path: Path, expected_count: int) -> np.ndarray:
    rows = _read_rows(path, {"waypoint_id", "t"})
    rows = sorted(rows, key=lambda row: int(row["waypoint_id"]))
    if len(rows) != expected_count:
        raise ValueError(f"knot_times row count {len(rows)} != expected {expected_count}")
    ids = [int(row["waypoint_id"]) for row in rows]
    if ids != list(range(expected_count)):
        raise ValueError("knot_times waypoint_id values must be consecutive from zero")
    times = np.asarray([float(row["t"]) for row in rows], dtype=np.float64)
    if not np.isfinite(times).all():
        raise ValueError("knot_times contains non-finite values")
    return times


def _load_query_table(path: Path) -> tuple[np.ndarray, np.ndarray]:
    cols = ["x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az", "jx", "jy", "jz"]
    rows = _read_rows(path, {"query_id", "t", *cols})
    rows = sorted(rows, key=lambda row: int(row["query_id"]))
    ids = [int(row["query_id"]) for row in rows]
    if ids != list(range(len(rows))):
        raise ValueError("query_id values must be consecutive from zero")
    times = np.asarray([float(row["t"]) for row in rows], dtype=np.float64)
    values = np.asarray([[float(row[col]) for col in cols] for row in rows], dtype=np.float64)
    if not np.isfinite(times).all() or not np.isfinite(values).all():
        raise ValueError("query table contains non-finite values")
    return times, values


def _load_outlier_scores(path: Path, expected_count: int) -> np.ndarray:
    rows = _read_rows(path, {"observation_id", "outlier_score"})
    rows = sorted(rows, key=lambda row: int(row["observation_id"]))
    ids = [int(row["observation_id"]) for row in rows]
    if ids != list(range(expected_count)):
        raise ValueError("outlier_scores observation_id values must cover every observation")
    scores = np.asarray([float(row["outlier_score"]) for row in rows], dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("outlier_scores contains non-finite values")
    return np.clip(scores, 0.0, 1.0)


def _load_outlier_labels(path: Path) -> np.ndarray:
    rows = _read_rows(path, {"observation_id", "is_outlier"})
    rows = sorted(rows, key=lambda row: int(row["observation_id"]))
    ids = [int(row["observation_id"]) for row in rows]
    if ids != list(range(len(rows))):
        raise ValueError("outlier_labels observation_id values must be consecutive from zero")
    return np.asarray([int(row["is_outlier"]) for row in rows], dtype=np.int8)


def _time_feasibility_errors(
    times: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    total_time: float,
    window_min: np.ndarray,
    window_max: np.ndarray,
) -> dict[str, float]:
    durations = np.diff(times)
    start_error = abs(float(times[0]))
    total_error = abs(float(times[-1]) - total_time)
    monotonic_violation = max(0.0, float(np.max(-durations)) if len(durations) else 0.0)
    lower_violation = float(np.max(np.maximum(lower - durations, 0.0)))
    upper_violation = float(np.max(np.maximum(durations - upper, 0.0)))
    window_violation = float(
        np.max(np.maximum(np.maximum(window_min - times, times - window_max), 0.0))
    )
    return {
        "time_start_error": start_error,
        "time_total_error": total_error,
        "time_monotonic_violation": monotonic_violation,
        "duration_bound_violation": max(lower_violation, upper_violation),
        "window_violation": window_violation,
        "max_time_feasibility_error": max(
            start_error, total_error, monotonic_violation, lower_violation, upper_violation, window_violation
        ),
    }


def _eval(coeff: np.ndarray, times: np.ndarray, segment: int, tau: float, order: int) -> np.ndarray:
    degree = coeff.shape[-1] - 1
    duration = float(times[segment + 1] - times[segment])
    delta = float(np.clip(tau, 0.0, 1.0) * duration)
    row = _basis_row(degree, delta, order)
    return coeff[segment] @ row


def _eval_global(coeff: np.ndarray, times: np.ndarray, time_value: float, order: int) -> np.ndarray:
    if time_value <= times[0]:
        segment = 0
    elif time_value >= times[-1]:
        segment = len(times) - 2
    else:
        segment = int(np.searchsorted(times, time_value, side="right") - 1)
        segment = min(max(segment, 0), len(times) - 2)
    duration = float(times[segment + 1] - times[segment])
    if duration <= 0.0:
        return np.full(coeff.shape[1], np.nan, dtype=np.float64)
    tau = float((time_value - times[segment]) / duration)
    return _eval(coeff, times, segment, tau, order)


def _objectives(coeff: np.ndarray, times: np.ndarray) -> dict[str, float]:
    per_axis = {axis: 0.0 for axis in AXES}
    degree = coeff.shape[-1] - 1
    durations = np.diff(times)
    for segment, duration in enumerate(durations):
        for axis_index, axis in enumerate(AXES):
            value = 0.0
            c = coeff[segment, axis_index, :]
            for p in range(4, degree + 1):
                fp = _falling(p, 4)
                for r in range(4, degree + 1):
                    value += (
                        c[p]
                        * c[r]
                        * fp
                        * _falling(r, 4)
                        * (duration ** (p + r - 7))
                        / float(p + r - 7)
                    )
            per_axis[axis] += float(value)
    per_axis["total"] = float(sum(per_axis[axis] for axis in AXES))
    return per_axis


def _sample_error(pred: np.ndarray, ref: np.ndarray, pred_times: np.ndarray, ref_times: np.ndarray,
                  audit: np.ndarray, windows: list[list[float]] | None = None) -> float:
    """Backwards-compatible scalar: position-only combined error (meters)."""
    inw, outw = _sample_errors_split(pred, ref, pred_times, ref_times, audit, windows or [])
    return 0.7 * inw + 0.3 * outw


def _sample_errors_split(pred: np.ndarray, ref: np.ndarray, pred_times: np.ndarray,
                         ref_times: np.ndarray, audit: np.ndarray,
                         windows: list[list[float]]) -> tuple[float, float]:
    """Absolute position RMS (meters) at audit times, split into the
    tracking-dropout windows (where the knowledge premium lives) and the
    observed remainder. Derivative orders are not scored: with the dropout
    deviation in play they are noise-dominated for every solver and would
    only dilute the signal (v0.18 lesson)."""
    pred_values = np.asarray(
        [_eval_global(pred, pred_times, float(t), 0) for t in audit], dtype=np.float64)
    ref_values = np.asarray(
        [_eval_global(ref, ref_times, float(t), 0) for t in audit], dtype=np.float64)
    if not np.isfinite(pred_values).all():
        return float("inf"), float("inf")
    inw = np.array([any(a <= t <= b for a, b in windows) for t in audit]) if windows else np.zeros(len(audit), dtype=bool)
    err = np.sqrt(np.sum((pred_values - ref_values) ** 2, axis=1))

    def rms(mask):
        if not mask.any():
            return 0.0
        return float(np.sqrt(np.mean(err[mask] ** 2)))

    return rms(inw), rms(~inw)


def _query_error(pred_values: np.ndarray, ref_values: np.ndarray) -> float:
    blocks = [
        (slice(0, 3), 0.85, 0.05),
        (slice(3, 6), 0.09, 0.20),
        (slice(6, 9), 0.04, 1.00),
        (slice(9, 12), 0.02, 5.00),
    ]
    total = 0.0
    for block, weight, floor in blocks:
        ref_block = ref_values[:, block]
        pred_block = pred_values[:, block]
        centered = ref_block - np.mean(ref_block, axis=0, keepdims=True)
        scale = max(float(np.sqrt(np.mean(centered**2))), floor)
        rms = float(np.sqrt(np.mean((pred_block - ref_block) ** 2)))
        total += weight * rms / scale
    return float(total)


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return 0.5
    wins = 0.0
    for value in positives:
        wins += float(np.sum(value > negatives))
        wins += 0.5 * float(np.sum(value == negatives))
    return float(wins / (len(positives) * len(negatives)))


def _constraint_errors(
    coeff: np.ndarray,
    times: np.ndarray,
    constraints: list[dict[str, Any]],
) -> dict[str, float]:
    n_segments = coeff.shape[0]
    continuity_error = 0.0
    for knot in range(1, n_segments):
        for order in [0, 1, 2, 3]:
            continuity_error = max(
                continuity_error,
                float(np.max(np.abs(_eval(coeff, times, knot - 1, 1.0, order) - _eval(coeff, times, knot, 0.0, order)))),
            )

    fixed_error = 0.0
    for constraint in constraints:
        axis_index = AXES.index(constraint["axis"])
        knot = int(constraint["waypoint_id"])
        order = int(constraint["derivative_order"])
        target = float(constraint["value"])
        values = []
        if knot == 0:
            values.append(_eval(coeff, times, 0, 0.0, order)[axis_index])
        elif knot == n_segments:
            values.append(_eval(coeff, times, n_segments - 1, 1.0, order)[axis_index])
        else:
            values.append(_eval(coeff, times, knot - 1, 1.0, order)[axis_index])
            values.append(_eval(coeff, times, knot, 0.0, order)[axis_index])
        for value in values:
            fixed_error = max(fixed_error, abs(float(value) - target))

    return {
        "continuity_error": float(continuity_error),
        "fixed_constraint_error": float(fixed_error),
        "max_polynomial_constraint_error": float(max(continuity_error, fixed_error)),
    }


def _time_error(pred_times: np.ndarray, ref_times: np.ndarray, total_time: float) -> float:
    return float(np.linalg.norm(pred_times - ref_times) / max(total_time, 1.0e-12))


def _report_consistency(
    pred_dir: Path,
    coeff: np.ndarray,
    times: np.ndarray,
    computed_objective: dict[str, float],
) -> tuple[float, dict[str, Any]]:
    # The quality-metric definition is no longer disclosed to agents, so the
    # numeric content of results/objective.json is not scored (agents without
    # the formula cannot compute it); only knot-derivative consistency counts.
    knot_score = 0.0
    details: dict[str, Any] = {}

    try:
        rows = _read_rows(
            pred_dir / "results" / "knot_derivatives.csv",
            {"waypoint_id", "axis", "side", "derivative_order", "value"},
        )
        seen = {}
        for row in rows:
            key = (
                int(row["waypoint_id"]),
                row["axis"],
                row["side"],
                int(row["derivative_order"]),
            )
            seen[key] = float(row["value"])
        max_error = 0.0
        n_expected = 0
        n_segments = coeff.shape[0]
        for knot in range(n_segments + 1):
            if knot == 0:
                sides = [("right", 0, 0.0)]
            elif knot == n_segments:
                sides = [("left", n_segments - 1, 1.0)]
            else:
                sides = [("left", knot - 1, 1.0), ("right", knot, 0.0)]
            for side, segment, tau in sides:
                for axis_index, axis in enumerate(AXES):
                    for order in range(5):
                        n_expected += 1
                        key = (knot, axis, side, order)
                        if key not in seen:
                            max_error = float("inf")
                            continue
                        expected = _eval(coeff, times, segment, tau, order)[axis_index]
                        max_error = max(max_error, abs(float(seen[key]) - float(expected)))
        coverage = min(1.0, len(seen) / max(1, n_expected))
        knot_score = SCORE_WEIGHTS["report"] * coverage * _linear_desc(max_error, 2.5e-5, 5.0e-4)
        details["knot_report_max_abs_error"] = float(max_error)
        details["knot_report_coverage"] = float(coverage)
    except Exception as exc:  # noqa: BLE001
        details["knot_report_error"] = f"{type(exc).__name__}: {exc}"

    return knot_score, details


@register_scorer("minimum_snap_trajectory_score")
class MinimumSnapTrajectoryScorer(Scorer):
    """Aggregate result-based scorer for the trajectory timing task."""

    def score(self, pred_dir: Path, ref_dir: Path, config: dict[str, Any]) -> ScoreDetail:
        weight = float(config.get("weight", 100.0))
        # Reference-side artifacts: a failure here is an instance/infrastructure
        # problem, not the agent's fault, so it still fails the whole scorer.
        try:
            waypoints = _load_waypoints(ref_dir)
            lower, upper = _load_bounds(ref_dir)
            window_min, window_max = _load_windows(ref_dir)
            constraints = _load_constraints(ref_dir)
            task_info = _read_json(ref_dir.parent / "data" / "task_info.json")
            total_time = float(task_info["total_time"])
            expected_shape = tuple(int(value) for value in task_info["coefficient_file_shape"])
            n_segments = expected_shape[0]
            ref = _load_coeff(ref_dir / config.get("reference_coefficients_file", "segment_coefficients_ref.npy"), expected_shape)
            ref_times = _load_times(ref_dir / config.get("reference_knot_times_file", "knot_times_ref.csv"), n_segments + 1)
            audit = np.load(ref_dir / config.get("audit_file", "audit_times.npy")).astype(np.float64)
            ref_query_times, ref_query = _load_query_table(ref_dir / config.get("reference_query_file", "query_predictions_ref.csv"))
            labels = _load_outlier_labels(ref_dir / config.get("outlier_label_file", "labels_hidden.csv"))
        except Exception as exc:  # noqa: BLE001
            return ScoreDetail(
                scorer_name="minimum_snap_trajectory_score",
                score=0.0,
                max_score=float(weight),
                passed=False,
                details={"error": f"{type(exc).__name__}: {exc}"},
                message=f"Scorer exception: {type(exc).__name__}: {exc}",
            )

        # Prediction-side artifacts are scored per component: a malformed file
        # zeroes only the components that depend on it instead of the whole
        # submission.
        details: dict[str, Any] = {}
        time_score = trajectory_score = objective_score = constraint_score = report_score = 0.0
        query_score = outlier_score = 0.0

        pred = None
        pred_times = None
        try:
            pred = _load_coeff(pred_dir / config.get("coefficients_file", "results/segment_coefficients.npy"), expected_shape)
            pred_times = _load_times(pred_dir / config.get("knot_times_file", "results/knot_times.csv"), n_segments + 1)
        except Exception as exc:  # noqa: BLE001
            details["core_error"] = f"{type(exc).__name__}: {exc}"

        time_credit = 0.0
        sample_credit = 0.0
        if pred is not None and pred_times is not None:
            time_details = _time_feasibility_errors(pred_times, lower, upper, total_time, window_min, window_max)
            time_rel = _time_error(pred_times, ref_times, total_time)
            time_credit = _linear_desc(time_rel, 2.0e-3, 4.0e-2)
            time_score = SCORE_WEIGHTS["time"] * time_credit
            time_score *= _linear_desc(time_details["max_time_feasibility_error"], 1.0e-7, 2.0e-3)

            dropout_windows = _load_dropout_windows(ref_dir)
            inwin_rms, outwin_rms = _sample_errors_split(pred, ref, pred_times, ref_times, audit, dropout_windows)
            sample_rel = 0.7 * inwin_rms + 0.3 * outwin_rms
            # bands measured with scripts/minsnap_v020_oracle.py: disclosed-law
            # reconstruction reaches 0.02-0.08 m inside the windows, the best
            # structure-free bridge sits at 0.17-0.19 m
            inwin_credit = _linear_desc(inwin_rms, 5.0e-2, 1.6e-1)
            outwin_credit = _linear_desc(outwin_rms, 2.0e-2, 1.0e-1)
            sample_credit = 0.75 * inwin_credit + 0.25 * outwin_credit
            trajectory_score = SCORE_WEIGHTS["trajectory"] * sample_credit

            constraint_details = _constraint_errors(pred, pred_times, constraints)
            max_constraint_error = max(
                constraint_details["max_polynomial_constraint_error"],
                time_details["max_time_feasibility_error"],
            )
            constraint_score = SCORE_WEIGHTS["constraint"] * _linear_desc(max_constraint_error, 7.5e-5, 5.0e-3)

            details.update(
                {
                    "knot_time_relative_error": time_rel,
                    "sample_relative_error": sample_rel,
                    "inwindow_position_rms": inwin_rms,
                    "outwindow_position_rms": outwin_rms,
                    **time_details,
                    **constraint_details,
                }
            )

        query_credit = 0.0
        try:
            pred_query_times, pred_query = _load_query_table(pred_dir / config.get("query_file", "results/query_predictions.csv"))
            if len(pred_query_times) != len(ref_query_times):
                raise ValueError("query_predictions row count does not match query_times")
            query_time_rel = float(np.linalg.norm(pred_query_times - ref_query_times) / max(total_time, 1.0e-12))
            query_rel = _query_error(pred_query, ref_query)
            query_credit = _linear_desc(query_rel, 1.0e-2, 1.4e-1)
            query_score = SCORE_WEIGHTS["query"] * query_credit
            query_score *= _linear_desc(query_time_rel, 1.0e-9, 1.0e-4)
            details["query_relative_error"] = query_rel
            details["query_time_relative_error"] = query_time_rel
        except Exception as exc:  # noqa: BLE001
            details["query_error"] = f"{type(exc).__name__}: {exc}"

        if pred is not None and pred_times is not None:
            pred_objective = _objectives(pred, pred_times)
            ref_objective = _objectives(ref, ref_times)
            objective_rel = abs(pred_objective["total"] - ref_objective["total"]) / max(1.0, abs(ref_objective["total"]))
            objective_context = min(time_credit, sample_credit, query_credit)
            # band anchored to the v0.20 deviation world: the objective is dominated
            # by the bump snap energy - a structure-free fit misses it entirely
            # (rel ~1.0) while the disclosed-law reconstruction reaches 0.02-0.28
            objective_score = SCORE_WEIGHTS["objective"] * _linear_desc(objective_rel, 5.0e-2, 8.5e-1) * objective_context

            report_score, report_details = _report_consistency(pred_dir, pred, pred_times, pred_objective)
            details.update(
                {
                    "objective_relative_error": objective_rel,
                    "objective_context_credit": objective_context,
                    "pred_objective_total": pred_objective["total"],
                    "ref_objective_total": ref_objective["total"],
                    **report_details,
                }
            )

        try:
            outlier_scores = _load_outlier_scores(
                pred_dir / config.get("outlier_file", "results/outlier_scores.csv"),
                len(labels),
            )
            outlier_auc = _auc(labels, outlier_scores)
            outlier_score = SCORE_WEIGHTS["outlier"] * _linear_desc(1.0 - outlier_auc, 5.0e-2, 3.8e-1)
            details["outlier_auc"] = outlier_auc
        except Exception as exc:  # noqa: BLE001
            details["outlier_error"] = f"{type(exc).__name__}: {exc}"

        figure_file = pred_dir / config.get("figure_file", "results/trajectory_diagnostics.png")
        figure_exists = figure_file.exists()
        figure_size = figure_file.stat().st_size if figure_exists else 0
        figure_score = SCORE_WEIGHTS["figure"] if figure_exists and figure_size >= 1000 else 0.0

        raw = query_score + trajectory_score + time_score + outlier_score + constraint_score + objective_score + report_score + figure_score
        scaled = raw * weight / 100.0
        details.update(
            {
                "query_score": query_score,
                "time_score": time_score,
                "trajectory_score": trajectory_score,
                "outlier_score": outlier_score,
                "objective_score": objective_score,
                "constraint_score": constraint_score,
                "report_score": report_score,
                "figure_score": figure_score,
                "figure_exists": figure_exists,
                "figure_size": figure_size,
            }
        )
        component_errors = sorted(key for key in details if key.endswith("_error") and isinstance(details[key], str))
        message = f"minimum_snap_trajectory={scaled:.2f}/{weight:.2f}"
        if component_errors:
            message += f" (component failures: {', '.join(component_errors)})"
        return ScoreDetail(
            scorer_name="minimum_snap_trajectory_score",
            score=float(scaled),
            max_score=float(weight),
            passed=True,
            details=details,
            message=message,
        )
