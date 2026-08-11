"""Ground-truth generator for robotics.minimum_snap_trajectory_conditioning."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ai4sci_mplconfig")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


INPUT_SPEC = [
    {"name": "data/observations.csv", "type": "data"},
    {"name": "data/waypoint_windows.csv", "type": "data"},
    {"name": "data/constraints.csv", "type": "data"},
    {"name": "data/segment_time_bounds.csv", "type": "data"},
    {"name": "data/query_times.csv", "type": "data"},
    {"name": "data/task_info.json", "type": "data"},
]

OUTPUT_SPEC = [
    {"name": "analysis.py", "type": "code"},
    {"name": "results/knot_times.csv", "type": "data"},
    {"name": "results/segment_coefficients.npy", "type": "data"},
    {"name": "results/knot_derivatives.csv", "type": "data"},
    {"name": "results/query_predictions.csv", "type": "data"},
    {"name": "results/outlier_scores.csv", "type": "data"},
    {"name": "results/objective.json", "type": "data"},
    {"name": "results/trajectory_diagnostics.png", "type": "figure"},
]

DEFAULT_PARAMS = {
    "n_segments": 10,
    "degree": 11,
    "seed": 0,
}

AXES = ("x", "y", "z")
TOTAL_TIME = 18.0
CORE_DEGREE = 7


def _merge_params(params: dict[str, Any] | None) -> dict[str, int]:
    merged = dict(DEFAULT_PARAMS)
    if params:
        merged.update(params)
    merged["n_segments"] = int(merged["n_segments"])
    merged["degree"] = int(merged["degree"])
    merged["seed"] = int(merged["seed"])
    if merged["n_segments"] < 6:
        raise ValueError("n_segments must be at least 6")
    if merged["degree"] < CORE_DEGREE:
        raise ValueError("degree must be at least 7")
    return merged


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


def _hermite_matrix() -> np.ndarray:
    matrix = np.zeros((8, 8), dtype=np.float64)
    row = 0
    for u in (0.0, 1.0):
        for order in range(4):
            for power in range(order, 8):
                if u == 0.0 and power > order:
                    continue
                matrix[row, power] = _falling(power, order) * (u ** (power - order))
            row += 1
    return matrix


def _unit_snap_matrix() -> np.ndarray:
    q = np.zeros((8, 8), dtype=np.float64)
    for p in range(4, 8):
        fp = _falling(p, 4)
        for r in range(4, 8):
            q[p, r] = fp * _falling(r, 4) / float(p + r - 7)
    return q


HERMITE_INV = np.linalg.inv(_hermite_matrix())
Q_UNIT = _unit_snap_matrix()


def _make_waypoints(n_segments: int, rng: np.random.Generator) -> np.ndarray:
    u = np.linspace(0.0, 1.0, n_segments + 1)
    phases = rng.uniform(-0.35, 0.35, size=3)
    local = np.column_stack(
        [
            2.2 * np.sin(2.0 * np.pi * (u + phases[0]))
            + 0.50 * np.sin(8.0 * np.pi * u + 0.2)
            + 0.35 * u,
            1.7 * np.cos(2.0 * np.pi * (u + phases[1]))
            + 0.42 * np.sin(5.0 * np.pi * u - 0.5)
            - 0.20 * u,
            1.1
            + 0.78 * np.sin(3.0 * np.pi * (u + phases[2]))
            + 0.25 * np.cos(11.0 * np.pi * u),
        ]
    )
    local += rng.normal(scale=[0.04, 0.04, 0.03], size=local.shape)
    offset = np.array([1.0e6, -7.5e5, 2.5e5], dtype=np.float64)
    scale = np.array([1.0, 0.75, 0.55], dtype=np.float64)
    return (offset + scale * local).astype(np.float64)


def _make_duration_bounds(n_segments: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    # moderate spread: a localized in-segment deviation must be
    # representable (and auditable) in every segment, so segment durations
    # stay within roughly one order of magnitude
    base = np.geomspace(0.55, 3.2, n_segments)
    center = base[rng.permutation(n_segments)] * rng.uniform(0.75, 1.25, size=n_segments)
    center *= TOTAL_TIME / float(center.sum())
    lower = 0.18 * center
    upper = 3.60 * center
    lower *= min(0.88, 0.34 * TOTAL_TIME / float(lower.sum()))
    upper *= max(1.12, 1.74 * TOTAL_TIME / float(upper.sum()))
    if not (float(lower.sum()) < TOTAL_TIME < float(upper.sum())):
        raise RuntimeError("duration bounds are infeasible")
    return lower.astype(np.float64), upper.astype(np.float64)


def _build_constraints(n_waypoints: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    constraints: list[dict[str, Any]] = []
    for waypoint_id in [0, n_waypoints - 1]:
        for axis in AXES:
            for order in [1, 2, 3]:
                constraints.append(
                    {
                        "waypoint_id": waypoint_id,
                        "axis": axis,
                        "derivative_order": order,
                        "value": 0.0,
                    }
                )
    interior = sorted(rng.choice(np.arange(2, n_waypoints - 2), size=3, replace=False).tolist())
    for idx, waypoint_id in enumerate(interior):
        axis = AXES[idx % len(AXES)]
        constraints.append(
            {
                "waypoint_id": int(waypoint_id),
                "axis": axis,
                "derivative_order": 1,
                "value": float(rng.uniform(-0.16, 0.16)),
            }
        )
    return constraints


def _fixed_map(axis: str, constraints: list[dict[str, Any]]) -> dict[tuple[int, int], float]:
    fixed: dict[tuple[int, int], float] = {}
    for row in constraints:
        if row["axis"] != axis:
            continue
        order = int(row["derivative_order"])
        if order == 0:
            continue
        key = (int(row["waypoint_id"]), order)
        value = float(row["value"])
        if key in fixed and not math.isclose(fixed[key], value, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"conflicting derivative constraint for {axis} at {key}")
        fixed[key] = value
    return fixed


def _solve_axis_reduced(
    durations: np.ndarray,
    positions: np.ndarray,
    axis: str,
    constraints: list[dict[str, Any]],
    degree: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    n_segments = len(durations)
    n_knots = n_segments + 1
    position_origin = float(positions[0])
    local_positions = positions - position_origin
    fixed = _fixed_map(axis, constraints)
    free_keys = [
        (waypoint_id, order)
        for waypoint_id in range(n_knots)
        for order in range(1, 4)
        if (waypoint_id, order) not in fixed
    ]
    free_index = {key: idx for idx, key in enumerate(free_keys)}
    n_free = len(free_keys)

    hessian = np.zeros((n_free, n_free), dtype=np.float64)
    linear = np.zeros(n_free, dtype=np.float64)
    segment_maps: list[tuple[float, np.ndarray, np.ndarray]] = []

    for segment, duration in enumerate(durations):
        endpoint_state = [
            (segment, 0),
            (segment, 1),
            (segment, 2),
            (segment, 3),
            (segment + 1, 0),
            (segment + 1, 1),
            (segment + 1, 2),
            (segment + 1, 3),
        ]
        e_const = np.zeros(8, dtype=np.float64)
        e_free = np.zeros((8, n_free), dtype=np.float64)
        for row, (waypoint_id, order) in enumerate(endpoint_state):
            scale = duration ** order
            if order == 0:
                e_const[row] = float(local_positions[waypoint_id])
            elif (waypoint_id, order) in fixed:
                e_const[row] = scale * fixed[(waypoint_id, order)]
            else:
                e_free[row, free_index[(waypoint_id, order)]] = scale
        coeff_const = HERMITE_INV @ e_const
        coeff_free = HERMITE_INV @ e_free
        q_segment = Q_UNIT / (duration**7)
        hessian += coeff_free.T @ q_segment @ coeff_free
        linear += coeff_free.T @ q_segment @ coeff_const
        segment_maps.append((float(duration), e_const, e_free))

    if n_free:
        hessian = 0.5 * (hessian + hessian.T)
        try:
            free_values = np.linalg.solve(hessian, -linear)
        except np.linalg.LinAlgError:
            scale = max(float(np.linalg.norm(hessian, ord=np.inf)), 1.0)
            regularized = hessian + (1.0e-12 * scale) * np.eye(n_free)
            free_values = np.linalg.lstsq(regularized, -linear, rcond=None)[0]
    else:
        free_values = np.zeros(0, dtype=np.float64)

    knot_derivatives = np.zeros((n_knots, 4), dtype=np.float64)
    knot_derivatives[:, 0] = positions
    for waypoint_id in range(n_knots):
        for order in range(1, 4):
            key = (waypoint_id, order)
            if key in fixed:
                knot_derivatives[waypoint_id, order] = fixed[key]
            else:
                knot_derivatives[waypoint_id, order] = free_values[free_index[key]]

    coeffs = np.zeros((n_segments, degree + 1), dtype=np.float64)
    objective = 0.0
    powers = np.arange(8, dtype=np.float64)
    for segment, (duration, e_const, e_free) in enumerate(segment_maps):
        normalized = HERMITE_INV @ (e_const + e_free @ free_values)
        coeffs[segment, :8] = normalized / (duration**powers)
        coeffs[segment, 0] += position_origin
        objective += float(normalized @ (Q_UNIT / (duration**7)) @ normalized)
    return coeffs, knot_derivatives, objective


def _solve_for_durations(
    durations: np.ndarray,
    waypoints: np.ndarray,
    constraints: list[dict[str, Any]],
    degree: int,
) -> tuple[np.ndarray, dict[str, float], dict[str, np.ndarray]]:
    coeffs = np.zeros((len(durations), 3, degree + 1), dtype=np.float64)
    per_axis: dict[str, float] = {}
    knot_derivatives: dict[str, np.ndarray] = {}
    for axis_index, axis in enumerate(AXES):
        axis_coeffs, axis_knot_derivatives, objective = _solve_axis_reduced(
            durations,
            waypoints[:, axis_index],
            axis,
            constraints,
            degree,
        )
        coeffs[:, axis_index, :] = axis_coeffs
        knot_derivatives[axis] = axis_knot_derivatives
        per_axis[axis] = float(objective)
    return coeffs, per_axis, knot_derivatives


# Relative scale of the de-idealization noise added to the free knot
# derivatives. The latent trajectory is deliberately NOT an exact constrained
# minimizer of the smoothness functional: an exact minimizer collapses the
# problem to ~36 identifiable dof and lets structure-aware solvers recover the
# curve far below the observation-noise floor (measured 1.7e-4 vs 0.11).
# Calibrated so the structure-assuming shortcut bottoms out near the general
# robust-fit noise floor.
DEIDEALIZATION_EPS = 0.05


def _perturb_knot_derivatives(
    knot_derivatives: dict[str, np.ndarray],
    constraints: list[dict[str, Any]],
    rng: np.random.Generator,
    eps: float,
) -> dict[str, np.ndarray]:
    perturbed: dict[str, np.ndarray] = {}
    for axis, states in knot_derivatives.items():
        fixed = _fixed_map(axis, constraints)
        out = states.copy()
        n_knots = out.shape[0]
        for order in range(1, 4):
            for knot in range(n_knots):
                if (knot, order) in fixed:
                    continue
                # Multiplicative noise adapts to the local derivative
                # magnitude, so knots bordering very short segments are not
                # over-perturbed by a global scale.
                out[knot, order] *= 1.0 + float(rng.normal(0.0, eps))
        perturbed[axis] = out
    return perturbed




# --- v0.20 structured tracking-residual field -------------------------------
# The latent trajectory deviates from the (de-idealized) plan inside short
# "aggressive maneuver" windows: an asymmetric-lag response to the plan's
# thresholded acceleration magnitude, applied along the acceleration
# direction rotated by an undisclosed per-instance azimuth. The controller
# re-anchors at every waypoint, so within each segment the deviation is
# projected onto a basis that vanishes to 3rd order at both knots - this
# keeps the truth exactly inside the contract polynomial space and keeps
# knot states, continuity, and fixed constraints untouched. Observations
# drop out while the deviation is large (tracking loses lock), leaving only
# shallow (~1-3 sigma) tails observable near window edges: the structure is
# discoverable in principle but the hypothesis search is CV-degenerate.
RESIDUAL_GRID_DT = 0.01
RESIDUAL_ACT_QUANTILE = 0.90
RESIDUAL_AMP_RANGE = (0.38, 0.62)     # meters; the level-crossing inversion needs amp well above the occlusion level
RESIDUAL_P_RANGE = (1.15, 1.65)
RESIDUAL_OCCLUSION_LEVEL = 0.07       # meters: no observations where |r| exceeds this
RESIDUAL_OCCLUSION_MARGIN = 0.12      # seconds of extra masking on each side


def _make_residual_params(rng: np.random.Generator) -> dict[str, Any]:
    return {
        "n_events": int(rng.integers(2, 4)),
        "amp_range_m": list(RESIDUAL_AMP_RANGE),
        "skew": float(rng.uniform(-0.9, 0.9)),
        "azimuth_deg": float(rng.uniform(0.0, 360.0)),
        "z_comp": float(rng.uniform(-0.4, 0.4)),
    }


def _comb(n: int, k: int) -> float:
    out = 1.0
    for i in range(k):
        out = out * (n - i) / (i + 1)
    return out


def _bump_coeff_rows(duration: float) -> np.ndarray:
    """Power-basis coefficients (ascending) of delta^4 (d-delta)^4 delta^k / d^8."""
    d = float(duration)
    base = np.zeros(9)
    for j in range(5):
        base[4 + j] = _comb(4, j) * (d ** (4 - j)) * ((-1.0) ** j)
    base = base / max(d, 1e-9) ** 8
    rows = np.zeros((4, 12))
    for k in range(4):
        rows[k, k:k + 9] = base
    return rows


def _bump_shape(delta: np.ndarray, d: float, skew: float) -> np.ndarray:
    """Unit-peak C3-vanishing skewed envelope on [0, d]."""
    env = (delta ** 4) * ((d - delta) ** 4) / max(d, 1e-9) ** 8 * 256.0
    return env * (1.0 + skew * (delta / d - 0.5))


def _event_direction(chord_xy: np.ndarray, azimuth_deg: float, z_comp: float) -> np.ndarray:
    phi = np.deg2rad(azimuth_deg)
    u = chord_xy / max(float(np.linalg.norm(chord_xy)), 1e-12)
    rot = np.array([np.cos(phi) * u[0] - np.sin(phi) * u[1],
                    np.sin(phi) * u[0] + np.cos(phi) * u[1],
                    z_comp])
    return rot / max(float(np.linalg.norm(rot)), 1e-12)


def _apply_residual_field(coeffs: np.ndarray, times: np.ndarray,
                          rp: dict[str, Any],
                          rng: np.random.Generator | None = None,
                          event_amps: dict[int, float] | None = None,
                          ) -> tuple[np.ndarray, dict[str, Any]]:
    """Tracking-dropout deviation, law v3 (position-level inputs only).

    n_events segments (duration >= 1.4 s, chosen at generation) each carry a
    deviation bump: unit-peak C3-vanishing skewed envelope times a random
    amplitude, along a direction defined by the segment's knot-to-knot chord
    (xy) rotated by an undisclosed azimuth plus an undisclosed z component.
    The bump vanishes to 3rd order at both knots, so knot states,
    continuity, and constraints are untouched and the truth stays exactly
    inside the contract space. Observations drop out where |deviation|
    exceeds RESIDUAL_OCCLUSION_LEVEL: the dropout-gap edges are therefore
    the bump's level crossings, and (given the law) the gap geometry alone
    determines each amplitude - every quantity a solver needs is
    position-level robust.
    """
    n_segments = coeffs.shape[0]
    durations = np.diff(times)
    if event_amps is None:
        assert rng is not None
        eligible = [seg for seg in range(n_segments) if durations[seg] >= 1.4]
        rng.shuffle(eligible)
        chosen = sorted(eligible[: rp["n_events"]])
        event_amps = {seg: float(rng.uniform(*rp["amp_range_m"])) for seg in chosen}

    out = coeffs.copy()
    tgrid = np.arange(0.0, TOTAL_TIME, RESIDUAL_GRID_DT)
    seg_idx = np.clip(np.searchsorted(times, tgrid, side="right") - 1, 0, n_segments - 1)
    field = np.zeros((len(tgrid), 3))
    for seg, amp in event_amps.items():
        d = float(durations[seg])
        chord = np.array([
            _evaluate(coeffs[seg, ax, :], d, 0) - _evaluate(coeffs[seg, ax, :], 0.0, 0)
            for ax in range(3)
        ])
        direction = _event_direction(chord[:2], rp["azimuth_deg"], rp["z_comp"])
        rows = _bump_coeff_rows(d)
        c0 = 256.0 * (1.0 - 0.5 * rp["skew"])
        c1 = 256.0 * rp["skew"] / d
        for ax in range(3):
            out[seg, ax, :] += amp * direction[ax] * (c0 * rows[0] + c1 * rows[1])
        m = seg_idx == seg
        delta = tgrid[m] - float(times[seg])
        field[m] = amp * _bump_shape(delta, d, rp["skew"])[:, None] * direction[None, :]

    pm = np.linalg.norm(field, axis=1)
    hot = pm > RESIDUAL_OCCLUSION_LEVEL
    k = max(1, int(RESIDUAL_OCCLUSION_MARGIN / RESIDUAL_GRID_DT))
    hot = np.convolve(hot.astype(float), np.ones(2 * k + 1), mode="same") > 0
    windows: list[list[float]] = []
    i = 0
    while i < len(hot):
        if hot[i]:
            j = i
            while j + 1 < len(hot) and hot[j + 1]:
                j += 1
            windows.append([float(tgrid[i]), float(tgrid[j])])
            i = j + 1
        else:
            i += 1
    meta: dict[str, Any] = {
        "windows": windows,
        "peak_m": float(pm.max()) if len(pm) else 0.0,
        "event_amps": {str(k_): float(v) for k_, v in event_amps.items()},
        "segment_amplitudes_m": [float(event_amps.get(seg, 0.0)) for seg in range(n_segments)],
    }
    return out, meta


def _in_windows(value: float, windows: list[list[float]]) -> bool:
    return any(a <= value <= b for a, b in windows)


def _states_to_coeffs_axis(durations: np.ndarray, states: np.ndarray, degree: int) -> np.ndarray:
    origin = float(states[0, 0])
    n_segments = len(durations)
    coeffs = np.zeros((n_segments, degree + 1), dtype=np.float64)
    powers = np.arange(8, dtype=np.float64)
    for segment, duration in enumerate(durations):
        endpoint_state = [
            (segment, 0), (segment, 1), (segment, 2), (segment, 3),
            (segment + 1, 0), (segment + 1, 1), (segment + 1, 2), (segment + 1, 3),
        ]
        e = np.zeros(8, dtype=np.float64)
        for row, (knot, order) in enumerate(endpoint_state):
            value = float(states[knot, order])
            if order == 0:
                value -= origin
            e[row] = (duration**order) * value
        normalized = HERMITE_INV @ e
        coeffs[segment, :8] = normalized / (duration**powers)
        coeffs[segment, 0] += origin
    return coeffs


def _states_to_coeffs(durations: np.ndarray, knot_derivatives: dict[str, np.ndarray], degree: int) -> np.ndarray:
    coeffs = np.zeros((len(durations), 3, degree + 1), dtype=np.float64)
    for axis_index, axis in enumerate(AXES):
        coeffs[:, axis_index, :] = _states_to_coeffs_axis(durations, knot_derivatives[axis], degree)
    return coeffs


def _project_bounded_sum(
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    total: float,
) -> np.ndarray:
    out = np.clip(np.asarray(values, dtype=np.float64), lower, upper)
    for _ in range(100):
        diff = float(total - out.sum())
        if abs(diff) < 1.0e-12:
            break
        if diff > 0:
            slack = upper - out
            active = slack > 1.0e-12
            if not np.any(active):
                break
            out[active] += diff * slack[active] / float(slack[active].sum())
        else:
            slack = out - lower
            active = slack > 1.0e-12
            if not np.any(active):
                break
            out[active] += diff * slack[active] / float(slack[active].sum())
        out = np.clip(out, lower, upper)
    return out


def _random_feasible(
    lower: np.ndarray,
    upper: np.ndarray,
    total: float,
    rng: np.random.Generator,
) -> np.ndarray:
    raw = lower + rng.random(len(lower)) * (upper - lower)
    return _project_bounded_sum(raw, lower, upper, total)


def _optimize_durations(
    waypoints: np.ndarray,
    constraints: list[dict[str, Any]],
    lower: np.ndarray,
    upper: np.ndarray,
    degree: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    cache: dict[tuple[float, ...], float] = {}

    def actual_objective(durations: np.ndarray) -> float:
        key = tuple(np.round(durations, 10))
        if key not in cache:
            _, per_axis, _ = _solve_for_durations(durations, waypoints, constraints, degree)
            cache[key] = float(sum(per_axis.values()))
        return cache[key]

    def log_objective(durations: np.ndarray) -> float:
        if np.any(durations < lower - 1.0e-10) or np.any(durations > upper + 1.0e-10):
            return 1.0e30
        if abs(float(durations.sum()) - TOTAL_TIME) > 1.0e-6:
            return 1.0e30
        return float(math.log(max(actual_objective(durations), 1.0e-300)))

    def local_pair_search(start: np.ndarray) -> np.ndarray:
        x = _project_bounded_sum(start, lower, upper, TOTAL_TIME)
        current = log_objective(x)
        n = len(x)
        base_step = float(np.max(upper - lower))
        for outer in range(12):
            improved = False
            step = base_step * (0.45 ** outer)
            for i in range(n):
                for j in range(i + 1, n):
                    lo_delta = max(lower[i] - x[i], x[j] - upper[j])
                    hi_delta = min(upper[i] - x[i], x[j] - lower[j])
                    if hi_delta - lo_delta <= 1.0e-12:
                        continue
                    candidates = np.linspace(lo_delta, hi_delta, 9)
                    if step > 0.0:
                        focused = np.array([-step, -0.5 * step, 0.5 * step, step], dtype=np.float64)
                        focused = focused[(focused >= lo_delta) & (focused <= hi_delta)]
                        candidates = np.unique(np.concatenate([candidates, focused]))
                    best_delta = 0.0
                    best_value = current
                    for delta in candidates:
                        if abs(float(delta)) < 1.0e-14:
                            continue
                        candidate = x.copy()
                        candidate[i] += float(delta)
                        candidate[j] -= float(delta)
                        value = log_objective(candidate)
                        if value < best_value - 1.0e-10:
                            best_value = value
                            best_delta = float(delta)
                    if best_delta != 0.0:
                        x[i] += best_delta
                        x[j] -= best_delta
                        current = best_value
                        improved = True
            if not improved and step < 1.0e-5:
                break
        return _project_bounded_sum(x, lower, upper, TOTAL_TIME)

    midpoint = _project_bounded_sum(0.5 * (lower + upper), lower, upper, TOTAL_TIME)
    duration_weighted = _project_bounded_sum(np.sqrt(lower * upper), lower, upper, TOTAL_TIME)
    starts = [midpoint, duration_weighted]
    for _ in range(22):
        starts.append(_random_feasible(lower, upper, TOTAL_TIME, rng))

    best_x = midpoint
    best_obj = actual_objective(midpoint)
    for start in starts:
        candidate = local_pair_search(start)
        value = actual_objective(candidate)
        if value < best_obj:
            best_x = candidate
            best_obj = value
    return best_x.astype(np.float64), float(best_obj)


def _evaluate(coeff: np.ndarray, delta: float, order: int) -> float:
    degree = coeff.shape[-1] - 1
    return float(_basis_row(degree, delta, order) @ coeff)


def _objective_from_coeffs(coeffs: np.ndarray, durations: np.ndarray) -> dict[str, float]:
    per_axis = {axis: 0.0 for axis in AXES}
    degree = coeffs.shape[-1] - 1
    for segment, duration in enumerate(durations):
        for axis_index, axis in enumerate(AXES):
            value = 0.0
            c = coeffs[segment, axis_index, :]
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


def _write_waypoints(path: Path, waypoints: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["waypoint_id", "x", "y", "z"])
        writer.writeheader()
        for idx, point in enumerate(waypoints):
            writer.writerow(
                {
                    "waypoint_id": idx,
                    "x": f"{float(point[0]):.17e}",
                    "y": f"{float(point[1]):.17e}",
                    "z": f"{float(point[2]):.17e}",
                }
            )


def _write_constraints(path: Path, constraints: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["waypoint_id", "axis", "derivative_order", "value"])
        writer.writeheader()
        for row in constraints:
            writer.writerow(
                {
                    "waypoint_id": row["waypoint_id"],
                    "axis": row["axis"],
                    "derivative_order": row["derivative_order"],
                    "value": f"{float(row['value']):.17e}",
                }
            )


def _write_segment_bounds(path: Path, lower: np.ndarray, upper: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["segment", "min_duration", "max_duration"])
        writer.writeheader()
        for segment, (lo, hi) in enumerate(zip(lower, upper, strict=True)):
            writer.writerow(
                {
                    "segment": segment,
                    "min_duration": f"{float(lo):.17e}",
                    "max_duration": f"{float(hi):.17e}",
                }
            )


def _write_knot_times(path: Path, times: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["waypoint_id", "t"])
        writer.writeheader()
        for idx, value in enumerate(times):
            writer.writerow({"waypoint_id": idx, "t": f"{float(value):.17e}"})


def _write_knot_derivatives(path: Path, coeffs: np.ndarray, times: np.ndarray) -> None:
    n_segments = coeffs.shape[0]
    durations = np.diff(times)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["waypoint_id", "axis", "side", "derivative_order", "value"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for knot in range(n_segments + 1):
            if knot == 0:
                sides = [("right", 0, 0.0)]
            elif knot == n_segments:
                sides = [("left", n_segments - 1, float(durations[n_segments - 1]))]
            else:
                sides = [
                    ("left", knot - 1, float(durations[knot - 1])),
                    ("right", knot, 0.0),
                ]
            for side, segment, delta in sides:
                for axis_index, axis in enumerate(AXES):
                    for order in range(5):
                        value = _evaluate(coeffs[segment, axis_index, :], delta, order)
                        writer.writerow(
                            {
                                "waypoint_id": knot,
                                "axis": axis,
                                "side": side,
                                "derivative_order": order,
                                "value": f"{value:.17e}",
                            }
                        )


def _eval_global_state(coeffs: np.ndarray, times: np.ndarray, time_value: float, max_order: int) -> np.ndarray:
    if time_value <= times[0]:
        segment = 0
    elif time_value >= times[-1]:
        segment = len(times) - 2
    else:
        segment = int(np.searchsorted(times, time_value, side="right") - 1)
        segment = min(max(segment, 0), len(times) - 2)
    delta = float(time_value - times[segment])
    values = []
    for order in range(max_order + 1):
        values.append([_evaluate(coeffs[segment, axis, :], delta, order) for axis in range(3)])
    return np.asarray(values, dtype=np.float64)


def _make_observations(
    coeffs: np.ndarray,
    times: np.ndarray,
    rng: np.random.Generator,
    windows: list[list[float]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    windows = windows or []
    rows: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    n_segments = coeffs.shape[0]
    obs_id = 0

    # v0.20: the tracking-dropout windows are the only observation gaps;
    # the old random sparse segments would create spurious gaps that mimic
    # deviation events
    normal_times: list[float] = []
    for segment in range(n_segments):
        count = 16
        low = float(times[segment])
        high = float(times[segment + 1])
        u = np.linspace(0.04, 0.96, count)
        u += rng.normal(0.0, 0.018, size=count)
        u = np.clip(u, 0.025, 0.975)
        for value in (low + u * (high - low)).tolist():
            # tracking dropout: no fixes while the deviation is large; a
            # rejected sample is re-drawn outside the windows so the
            # per-segment density stays balanced
            for _ in range(40):
                if not _in_windows(value, windows):
                    break
                value = float(low + rng.uniform(0.025, 0.975) * (high - low))
            if not _in_windows(value, windows):
                normal_times.append(value)
    for value in (times + rng.normal(0.0, 0.018, size=len(times))).clip(0.0, TOTAL_TIME).tolist():
        if not _in_windows(value, windows):
            normal_times.append(value)

    for value in normal_times:
        state = _eval_global_state(coeffs, times, float(value), 0)[0]
        sigma = float(rng.uniform(0.018, 0.055))
        noisy = state + rng.normal(0.0, sigma, size=3)
        rows.append(
            {
                "observation_id": obs_id,
                "t": float(value),
                "x": float(noisy[0]),
                "y": float(noisy[1]),
                "z": float(noisy[2]),
                "sigma": sigma,
            }
        )
        labels.append({"observation_id": obs_id, "is_outlier": 0})
        obs_id += 1

    n_outliers = max(18, int(0.16 * len(rows)))
    for _ in range(n_outliers):
        value = float(rng.uniform(0.0, TOTAL_TIME))
        for _attempt in range(60):
            if not _in_windows(value, windows):
                break
            value = float(rng.uniform(0.0, TOTAL_TIME))
        if _in_windows(value, windows):
            continue
        state = _eval_global_state(coeffs, times, value, 0)[0]
        if rng.random() < 0.45:
            shifted = float(np.clip(value + rng.normal(0.0, 1.8), 0.0, TOTAL_TIME))
            false = _eval_global_state(coeffs, times, shifted, 0)[0]
            offset = false - state
        else:
            direction = rng.normal(size=3)
            direction /= max(float(np.linalg.norm(direction)), 1.0e-12)
            offset = direction * float(rng.uniform(0.45, 1.55))
        sigma = float(rng.uniform(0.018, 0.06))
        noisy = state + offset + rng.normal(0.0, 2.0 * sigma, size=3)
        rows.append(
            {
                "observation_id": obs_id,
                "t": value,
                "x": float(noisy[0]),
                "y": float(noisy[1]),
                "z": float(noisy[2]),
                "sigma": sigma,
            }
        )
        labels.append({"observation_id": obs_id, "is_outlier": 1})
        obs_id += 1

    order = np.argsort([row["t"] for row in rows], kind="stable")
    sorted_rows: list[dict[str, Any]] = []
    sorted_labels: list[dict[str, Any]] = []
    for new_id, old_index in enumerate(order):
        row = dict(rows[int(old_index)])
        label = dict(labels[int(old_index)])
        row["observation_id"] = new_id
        label["observation_id"] = new_id
        sorted_rows.append(row)
        sorted_labels.append(label)
    return sorted_rows, sorted_labels


def _write_observations(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["observation_id", "t", "x", "y", "z", "sigma"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "observation_id": int(row["observation_id"]),
                    "t": f"{float(row['t']):.17e}",
                    "x": f"{float(row['x']):.17e}",
                    "y": f"{float(row['y']):.17e}",
                    "z": f"{float(row['z']):.17e}",
                    "sigma": f"{float(row['sigma']):.17e}",
                }
            )


def _write_outlier_labels(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["observation_id", "is_outlier"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"observation_id": int(row["observation_id"]), "is_outlier": int(row["is_outlier"])})


def _write_outlier_score_reference(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["observation_id", "outlier_score"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"observation_id": int(row["observation_id"]), "outlier_score": int(row["is_outlier"])})


def _write_waypoint_windows(path: Path, times: np.ndarray, rng: np.random.Generator) -> None:
    # Windows must contain the latent knot time but must NOT be centered on
    # it: independent left/right margins keep the midpoint from encoding the
    # answer (copying midpoints once earned the full knot-time credit).
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["waypoint_id", "t_min", "t_max"])
        writer.writeheader()
        for idx, value in enumerate(times):
            width = float(rng.uniform(0.10, 0.32))
            if 0 < idx < len(times) - 1:
                width += 0.10 * min(float(times[idx] - times[idx - 1]), float(times[idx + 1] - times[idx]))
            left = width * float(rng.uniform(0.30, 1.0))
            right = width * float(rng.uniform(0.30, 1.0))
            writer.writerow(
                {
                    "waypoint_id": idx,
                    "t_min": f"{max(0.0, float(value) - left):.17e}",
                    "t_max": f"{min(TOTAL_TIME, float(value) + right):.17e}",
                }
            )


def _write_query_times(path: Path, query_times: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "t"])
        writer.writeheader()
        for idx, value in enumerate(query_times):
            writer.writerow({"query_id": idx, "t": f"{float(value):.17e}"})


def _write_query_reference(path: Path, coeffs: np.ndarray, times: np.ndarray, query_times: np.ndarray) -> None:
    fieldnames = [
        "query_id",
        "t",
        "x",
        "y",
        "z",
        "vx",
        "vy",
        "vz",
        "ax",
        "ay",
        "az",
        "jx",
        "jy",
        "jz",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, value in enumerate(query_times):
            state = _eval_global_state(coeffs, times, float(value), 3)
            row = {"query_id": idx, "t": f"{float(value):.17e}"}
            for prefix, order in [("", 0), ("v", 1), ("a", 2), ("j", 3)]:
                for axis_index, axis in enumerate(AXES):
                    row[f"{prefix}{axis}"] = f"{float(state[order, axis_index]):.17e}"
            writer.writerow(row)


def _audit_times(times: np.ndarray, rng: np.random.Generator,
                 windows: list[list[float]] | None = None) -> np.ndarray:
    rows = []
    for segment in range(len(times) - 1):
        duration = float(times[segment + 1] - times[segment])
        for tau in [0.09, 0.27, 0.48, 0.71, 0.92]:
            jitter = float(rng.uniform(-0.012, 0.012))
            u = float(np.clip(tau + jitter, 0.04, 0.96))
            rows.append(float(times[segment] + u * duration))
    # concentrate audit mass inside the tracking-dropout windows: the
    # knowledge premium is scored where the deviation lives
    for a, b in windows or []:
        span = b - a
        for u in np.linspace(0.12, 0.88, 6):
            jitter = float(rng.uniform(-0.02, 0.02)) * span
            rows.append(float(np.clip(a + u * span + jitter, 0.0, TOTAL_TIME - 1e-6)))
    return np.asarray(sorted(rows), dtype=np.float64)


def _make_figure(path: Path, coeffs: np.ndarray, times: np.ndarray, waypoints: np.ndarray) -> None:
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(121, projection="3d")
    dense_points = []
    for segment in range(coeffs.shape[0]):
        duration = float(times[segment + 1] - times[segment])
        for u in np.linspace(0.0, 1.0, 45):
            delta = u * duration
            dense_points.append([_evaluate(coeffs[segment, axis, :], delta, 0) for axis in range(3)])
    dense = np.asarray(dense_points, dtype=np.float64)
    ax.plot(dense[:, 0], dense[:, 1], dense[:, 2], lw=1.8)
    ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], s=18)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("trajectory")

    ax2 = fig.add_subplot(122)
    durations = np.diff(times)
    ax2.bar(np.arange(len(durations)), durations)
    ax2.set_xlabel("segment")
    ax2.set_ylabel("duration")
    ax2.set_title("optimized timing")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _reference_diagnostics(
    coeffs: np.ndarray,
    times: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    nominal_objective: float,
    optimal_objective: float,
) -> dict[str, Any]:
    n_segments = coeffs.shape[0]
    durations = np.diff(times)
    max_jerk_jump = 0.0
    for knot in range(1, n_segments):
        for axis in range(3):
            left = _evaluate(coeffs[knot - 1, axis, :], float(durations[knot - 1]), 3)
            right = _evaluate(coeffs[knot, axis, :], 0.0, 3)
            max_jerk_jump = max(max_jerk_jump, abs(left - right))
    active = int(np.sum(np.isclose(durations, lower, rtol=0.0, atol=2.0e-5)))
    active += int(np.sum(np.isclose(durations, upper, rtol=0.0, atol=2.0e-5)))
    return {
        "duration_ratio": float(np.max(durations) / np.min(durations)),
        "bound_ratio": float(np.max(upper) / np.min(lower)),
        "active_duration_bounds": active,
        "max_jerk_jump": float(max_jerk_jump),
        "nominal_objective": float(nominal_objective),
        "optimal_objective": float(optimal_objective),
        "objective_improvement_ratio": float(nominal_objective / max(optimal_objective, 1.0e-300)),
    }


def _render_prompts(output_dir: Path, subs: dict[str, str] | None = None) -> None:
    """Copy prompts, rendering {{ key }} placeholders with instance values.
    Only prompt_b1.md carries placeholders (the disclosed residual law); a
    leftover unrendered placeholder is a hard error."""
    task_dir = Path(__file__).parent
    for level in ["b1", "b2", "b3", "b4"]:
        text = (task_dir / f"prompt_{level}.md").read_text(encoding="utf-8")
        for key, value in (subs or {}).items():
            text = text.replace("{{ " + key + " }}", value).replace("{{" + key + "}}", value)
        if "{{" in text:
            raise ValueError(f"unrendered placeholder left in prompt_{level}.md")
        (output_dir / f"prompt_{level}.md").write_text(text, encoding="utf-8")


def generate(output_dir: Path, params: dict[str, Any] | None = None) -> dict[str, Any]:
    p = _merge_params(params)
    start = time.time()
    rng = np.random.default_rng(int(p["seed"]))

    output_dir = Path(output_dir)
    data_dir = output_dir / "data"
    ref_dir = output_dir / "reference"
    data_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)

    n_segments = int(p["n_segments"])
    degree = int(p["degree"])
    waypoints = _make_waypoints(n_segments, rng)
    constraints = _build_constraints(n_segments + 1, rng)
    lower, upper = _make_duration_bounds(n_segments, rng)
    nominal = _project_bounded_sum(0.5 * (lower + upper), lower, upper, TOTAL_TIME)
    _, nominal_per_axis, _ = _solve_for_durations(nominal, waypoints, constraints, degree)
    nominal_objective = float(sum(nominal_per_axis.values()))
    durations, _ = _optimize_durations(waypoints, constraints, lower, upper, degree, rng)
    times = np.concatenate([[0.0], np.cumsum(durations)])
    times[-1] = TOTAL_TIME
    _, per_axis, knot_states = _solve_for_durations(durations, waypoints, constraints, degree)
    clean_optimal_objective = float(sum(per_axis.values()))
    perturbed_states = _perturb_knot_derivatives(knot_states, constraints, rng, DEIDEALIZATION_EPS)
    coeffs = _states_to_coeffs(durations, perturbed_states, degree)
    residual_params = _make_residual_params(rng)
    coeffs_plan = coeffs.copy()
    coeffs, residual_meta = _apply_residual_field(coeffs, times, residual_params, rng=rng)
    windows = residual_meta["windows"]
    computed_objective = _objective_from_coeffs(coeffs, durations)
    total_objective = float(computed_objective["total"])
    audit = _audit_times(times, rng, windows)
    query_times = _audit_times(times, rng, windows)
    observations, outlier_labels = _make_observations(coeffs, times, rng, windows)

    _write_observations(data_dir / "observations.csv", observations)
    _write_waypoint_windows(data_dir / "waypoint_windows.csv", times, rng)
    _write_constraints(data_dir / "constraints.csv", constraints)
    _write_segment_bounds(data_dir / "segment_time_bounds.csv", lower, upper)
    _write_query_times(data_dir / "query_times.csv", query_times)
    task_info = {
        "schema_version": 3,
        "degree": degree,
        "n_segments": n_segments,
        "axes": list(AXES),
        "time_unit": "seconds",
        "total_time": TOTAL_TIME,
        "basis": "segment_time_offset_power",
        "coefficient_order": "ascending",
        "metric_label": "trajectory_quality_index",
        "quality_metric": {
            "name": "trajectory_quality_index",
            "objective": (
                "lower is generally better; scored by agreement with the "
                "latent trajectory's own value, not by absolute minimization"
            ),
            "definition": (
                "a fixed smoothness functional of the trajectory; its exact "
                "form is not disclosed"
            ),
        },
        "continuity_derivative_orders": [0, 1, 2, 3],
        "reported_derivative_orders": [0, 1, 2, 3, 4],
        "coefficient_file_shape": [n_segments, 3, degree + 1],
        "query_prediction_columns": [
            "query_id",
            "t",
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "vz",
            "ax",
            "ay",
            "az",
            "jx",
            "jy",
            "jz",
        ],
        "outlier_score_columns": ["observation_id", "outlier_score"],
        "outlier_score_semantics": "larger values indicate more likely outliers",
        "outlier_score_range": (
            "scores are clipped to [0, 1] before ranking; map your scores "
            "monotonically into [0, 1] or values above 1 collapse into ties"
        ),
        "knot_derivative_columns": ["waypoint_id", "axis", "side", "derivative_order", "value"],
        "knot_derivative_sides": (
            "side is 'left' or 'right'; the first waypoint reports only "
            "'right', the last waypoint reports only 'left', interior "
            "waypoints report both sides, each with derivative orders 0-4 "
            "per axis"
        ),
        "required_outputs": [entry["name"] for entry in OUTPUT_SPEC],
        "evaluation": {
            "summary": (
                "Result-based scoring against a hidden reference trajectory. "
                "Partial credit interpolates between per-component error "
                "thresholds; the thresholds, component weights, and the exact "
                "held-out sample times are not disclosed."
            ),
            "components": [
                {
                    "name": "trajectory_agreement",
                    "description": (
                        "Your coefficients and knot times are evaluated at "
                        "hidden held-out times across the full duration - "
                        "including time intervals where observations are "
                        "unavailable - and compared with the latent reference "
                        "positions using RMS error."
                    ),
                },
                {
                    "name": "query_agreement",
                    "description": (
                        "results/query_predictions.csv is compared with hidden "
                        "clean reference states at the public query times. The "
                        "submitted t column must reproduce the public query "
                        "times essentially exactly."
                    ),
                },
                {
                    "name": "objective_agreement",
                    "description": (
                        "The undisclosed quality metric is recomputed from "
                        "your submitted coefficients and compared with the "
                        "latent reference value; this component is also scaled "
                        "by how well the trajectory, query, and knot-time "
                        "components score."
                    ),
                },
                {
                    "name": "knot_times",
                    "description": (
                        "results/knot_times.csv is compared with the reference "
                        "knot times and must satisfy the window, duration-bound, "
                        "monotonicity, and total-time requirements."
                    ),
                },
                {
                    "name": "constraints",
                    "description": (
                        "Worst-case continuity and fixed derivative-constraint "
                        "residual evaluated from the submitted coefficients."
                    ),
                },
                {
                    "name": "outlier_ranking",
                    "description": (
                        "results/outlier_scores.csv is ranked against hidden "
                        "ground-truth outlier labels (AUC-style). Scores are "
                        "clipped to [0, 1] before ranking."
                    ),
                },
                {
                    "name": "report_consistency",
                    "description": (
                        "results/knot_derivatives.csv must match values "
                        "recomputed from your submitted coefficients; the "
                        "diagnostic figure must exist."
                    ),
                },
            ],
        },
    }
    (data_dir / "task_info.json").write_text(json.dumps(task_info, indent=2), encoding="utf-8")

    _write_knot_times(ref_dir / "knot_times_ref.csv", times)
    _write_waypoints(ref_dir / "waypoints_ref.csv", waypoints)
    np.save(ref_dir / "segment_coefficients_ref.npy", coeffs)
    np.save(ref_dir / "audit_times.npy", audit)
    np.save(ref_dir / "segment_coefficients_plan_ref.npy", coeffs_plan)
    (ref_dir / "residual_field_ref.json").write_text(json.dumps({
        "params": residual_params,
        "event_amps": residual_meta["event_amps"],
        "windows": residual_meta["windows"],
        "peak_m": residual_meta["peak_m"],
        "segment_amplitudes_m": residual_meta["segment_amplitudes_m"],
        "occlusion_level_m": RESIDUAL_OCCLUSION_LEVEL,
    }, indent=2), encoding="utf-8")
    _write_knot_derivatives(ref_dir / "knot_derivatives_ref.csv", coeffs, times)
    _write_query_reference(ref_dir / "query_predictions_ref.csv", coeffs, times, query_times)
    _write_outlier_labels(ref_dir / "labels_hidden.csv", outlier_labels)
    _write_outlier_score_reference(ref_dir / "outlier_scores.csv", outlier_labels)
    _write_outlier_score_reference(ref_dir / "outlier_scores_ref.csv", outlier_labels)
    objective = {
        "metric": "trajectory_quality_index",
        "basis": "segment_time_offset_power",
        "per_axis": {axis: float(computed_objective[axis]) for axis in AXES},
        "total": total_objective,
    }
    (ref_dir / "objective_ref.json").write_text(json.dumps(objective, indent=2), encoding="utf-8")
    _make_figure(ref_dir / "trajectory_diagnostics_ref.png", coeffs, times, waypoints)
    diagnostics = _reference_diagnostics(
        coeffs,
        times,
        lower,
        upper,
        nominal_objective,
        clean_optimal_objective,
    )
    diagnostics["deidealization_eps"] = DEIDEALIZATION_EPS
    diagnostics["deidealized_objective_ratio"] = float(total_objective / max(clean_optimal_objective, 1.0e-300))
    diagnostics["generation_seconds"] = float(time.time() - start)
    (ref_dir / "reference_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    _render_prompts(output_dir, {
        "residual_skew": f"{residual_params['skew']:.3f}",
        "residual_azimuth_deg": f"{residual_params['azimuth_deg']:.1f}",
        "residual_z_comp": f"{residual_params['z_comp']:.3f}",
        "residual_occlusion_level_m": f"{RESIDUAL_OCCLUSION_LEVEL:.3f}",
    })

    meta = {
        "params_used": p,
        "input_files": [item["name"] for item in INPUT_SPEC],
        "reference_files": [
            "knot_times_ref.csv",
            "waypoints_ref.csv",
            "segment_coefficients_ref.npy",
            "knot_derivatives_ref.csv",
            "query_predictions_ref.csv",
            "outlier_scores.csv",
            "outlier_scores_ref.csv",
            "objective_ref.json",
            "trajectory_diagnostics_ref.png",
            "audit_times.npy",
            "reference_diagnostics.json",
        ],
        "generation_time_seconds": round(time.time() - start, 2),
    }
    (output_dir / "instance_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--params", type=str, default="{}")
    args = parser.parse_args()
    params = json.loads(args.params)
    print(json.dumps(generate(args.output_dir, params), indent=2))


if __name__ == "__main__":
    main()
