"""Ground-truth generator for the anonymized BSSE / asymptotic-limit task.

Version 3 updates the task in two directions:
1. restore the minimal public row-family contract that v2 over-hid in B3/B4
2. add a pairing-sensitive low-level diagnostic and harder limit-fitting workflow
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


HARTREE_TO_KCAL_MOL = 627.5094740631
KCAL_MOL_TO_HARTREE = 1.0 / HARTREE_TO_KCAL_MOL
HF_ALPHA_DOMINANT = 1.63
HF_ALPHA_SUBLEADING = 3.05

LADDER_LEVELS = [
    ("level_02", 2),
    ("level_03", 3),
    ("level_04", 4),
    ("level_05", 5),
]

OUTPUT_CHANNEL_ROLES = ("reference", "total")
SEMANTIC_STATES = (
    "dimer_full",
    "frag_a_native",
    "frag_b_native",
    "frag_a_balanced",
    "frag_b_balanced",
)

SCAN_COLUMNS = [
    "case_id",
    "channel_role",
    "level_label",
    "level_index",
    "path_0_kcal_mol",
    "path_1_kcal_mol",
    "path_gap_kcal_mol",
]

SHIFT_COLUMNS = [
    "case_id",
    "channel_role",
    "level_label",
    "level_index",
    "shift_small_kcal_mol",
    "shift_large_kcal_mol",
]

SUMMARY_COLUMNS = [
    "case_id",
    "reference_limit_path_0_kcal_mol",
    "reference_limit_path_1_kcal_mol",
    "delta_limit_path_0_kcal_mol",
    "delta_limit_path_1_kcal_mol",
    "total_limit_path_0_kcal_mol",
    "total_limit_path_1_kcal_mol",
]

INPUT_SPEC = [
    {
        "name": "case_index.csv",
        "description": "Per-case geometry index with a generic distance axis in atomic units.",
    },
    {
        "name": "ladder_spec.json",
        "description": "Generic ladder metadata with level labels, value units, and channel-column names.",
    },
    {
        "name": "channel_ledger.csv",
        "description": (
            "Synthetic precomputed energy ledger modeled after HF/reference and "
            "correlated total basis-set convergence for five anonymous row codes "
            "across the ladder."
        ),
    },
]

OUTPUT_SPEC = [
    {
        "name": "results/path_scan.csv",
        "description": "Per-case path scan for the reference-like and total channels across the ladder.",
    },
    {
        "name": "results/pair_shift_scan.csv",
        "description": "Per-case sorted within-family stabilization shifts for both channels.",
    },
    {
        "name": "results/asymptotic_summary.csv",
        "description": "Per-case asymptotic limits for the reference, residual, and total channels.",
    },
    {
        "name": "results/convergence_plot.png",
        "description": "Reference plot showing path_0 vs path_1 convergence across the ladder.",
    },
]

DEFAULT_PARAMS = {
    "seed": 0,
}


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sample_cases(rng: np.random.Generator) -> list[dict[str, Any]]:
    base = float(rng.uniform(5.45, 5.75))
    separations = np.array(
        [
            base - 0.46 + rng.uniform(-0.05, 0.04),
            base - 0.11 + rng.uniform(-0.04, 0.03),
            base + 0.24 + rng.uniform(-0.03, 0.04),
            base + 0.72 + rng.uniform(-0.04, 0.05),
        ],
        dtype=float,
    )
    separations.sort()
    cases: list[dict[str, Any]] = []
    for idx, separation in enumerate(separations):
        cases.append(
            {
                "case_id": f"case_{idx:03d}",
                "geometry_label": f"sample_{idx}",
                "distance_axis_au": round(float(separation), 3),
            }
        )
    return cases


def _fit_line(xs: list[float], ys: list[float]) -> tuple[float, float]:
    count = float(len(xs))
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xx = sum(x * x for x in xs)
    sum_xy = sum(x * y for x, y in zip(xs, ys, strict=True))
    denominator = count * sum_xx - sum_x * sum_x
    if abs(denominator) < 1.0e-16:
        raise ValueError("degenerate linear regression")
    intercept = (sum_y * sum_xx - sum_x * sum_xy) / denominator
    slope = (count * sum_xy - sum_x * sum_y) / denominator
    return intercept, slope


def _make_code_maps(rng: np.random.Generator) -> tuple[dict[str, str], dict[str, str]]:
    row_codes = [f"slot_{value}" for value in rng.permutation(np.arange(5))]
    semantic_to_code = {semantic: code for semantic, code in zip(SEMANTIC_STATES, row_codes, strict=True)}
    code_to_semantic = {code: semantic for semantic, code in semantic_to_code.items()}
    return semantic_to_code, code_to_semantic


def _make_channel_map(rng: np.random.Generator) -> dict[str, str]:
    if bool(rng.integers(0, 2)):
        return {
            "reference": "channel_0_hartree",
            "total": "channel_1_hartree",
        }
    return {
        "reference": "channel_1_hartree",
        "total": "channel_0_hartree",
    }


def _monomer_energies(
    rng: np.random.Generator,
    cases: list[dict[str, Any]],
) -> tuple[
    dict[str, dict[int, float]],
    dict[str, dict[int, float]],
    dict[str, dict[int, float]],
    dict[str, dict[int, float]],
]:
    hf_base_inf = -76.0605 + float(rng.normal(0.0, 0.0012))
    corr_base_inf = -0.2042 + float(rng.normal(0.0, 0.0006))
    hf_offset = float(rng.uniform(-0.0010, 0.0010))
    corr_offset = float(rng.uniform(-0.00045, 0.00045))
    hf_amp = 0.60 + float(rng.uniform(-0.05, 0.05))
    corr_amp = 0.34 + float(rng.uniform(-0.03, 0.03))

    r_center = sum(float(case["distance_axis_au"]) for case in cases) / len(cases)
    hf_case_a_lin = float(rng.uniform(-2.2e-5, 2.2e-5))
    hf_case_b_lin = float(rng.uniform(-2.2e-5, 2.2e-5))
    hf_case_a_quad = float(rng.uniform(-6.0e-6, 6.0e-6))
    hf_case_b_quad = float(rng.uniform(-6.0e-6, 6.0e-6))
    corr_case_a_lin = float(rng.uniform(-8.0e-6, 8.0e-6))
    corr_case_b_lin = float(rng.uniform(-8.0e-6, 8.0e-6))
    corr_case_a_quad = float(rng.uniform(-2.5e-6, 2.5e-6))
    corr_case_b_quad = float(rng.uniform(-2.5e-6, 2.5e-6))

    mono_a_hf: dict[str, dict[int, float]] = {str(case["case_id"]): {} for case in cases}
    mono_b_hf: dict[str, dict[int, float]] = {str(case["case_id"]): {} for case in cases}
    mono_a_total: dict[str, dict[int, float]] = {str(case["case_id"]): {} for case in cases}
    mono_b_total: dict[str, dict[int, float]] = {str(case["case_id"]): {} for case in cases}

    for case in cases:
        case_id = str(case["case_id"])
        delta_r = float(case["distance_axis_au"]) - r_center
        hf_case_a = hf_case_a_lin * delta_r + hf_case_a_quad * delta_r * delta_r
        hf_case_b = hf_case_b_lin * delta_r + hf_case_b_quad * delta_r * delta_r
        corr_case_a = corr_case_a_lin * delta_r + corr_case_a_quad * delta_r * delta_r
        corr_case_b = corr_case_b_lin * delta_r + corr_case_b_quad * delta_r * delta_r

        for _, level_index in LADDER_LEVELS:
            hf_err = hf_amp * math.exp(-HF_ALPHA_DOMINANT * level_index)
            corr_err = corr_amp * (level_index ** -3)

            mono_a_hf[case_id][level_index] = hf_base_inf + hf_offset + hf_case_a + hf_err
            mono_b_hf[case_id][level_index] = hf_base_inf - hf_offset + hf_case_b + hf_err
            mono_a_total[case_id][level_index] = (
                mono_a_hf[case_id][level_index] + corr_base_inf + corr_offset + corr_case_a + corr_err
            )
            mono_b_total[case_id][level_index] = (
                mono_b_hf[case_id][level_index] + corr_base_inf - corr_offset + corr_case_b + corr_err
            )

    return mono_a_hf, mono_b_hf, mono_a_total, mono_b_total


def _interaction_model(
    rng: np.random.Generator,
    distance_axis_au: float,
    r_min: float,
) -> dict[str, float]:
    scale = math.exp(-0.55 * (distance_axis_au - r_min))

    ref_inf = 0.11 + 0.13 * math.exp(-0.85 * (distance_axis_au - r_min)) + float(rng.uniform(0.0, 0.02))
    total_inf = -(
        1.45 * math.exp(-0.93 * (distance_axis_au - r_min))
        + 0.10
        + float(rng.uniform(0.0, 0.03))
    )
    delta_inf = total_inf - ref_inf

    ref_path1_primary = 8.6 * scale + 1.2 + float(rng.uniform(0.0, 0.4))
    ref_gap_primary = 2.3 + 0.7 * scale + float(rng.uniform(0.0, 0.2))
    ref_path0_primary = ref_path1_primary - ref_gap_primary
    ref_path1_secondary = 0.38 + 0.22 * scale + float(rng.uniform(0.0, 0.07))
    ref_gap_secondary = 0.08 + 0.05 * scale + float(rng.uniform(0.0, 0.03))
    ref_path0_secondary = ref_path1_secondary - ref_gap_secondary

    delta_path1_primary = 2.0 + 1.2 * scale + float(rng.uniform(0.0, 0.2))
    delta_gap_primary = 1.0 + 0.85 * scale + float(rng.uniform(0.0, 0.1))
    delta_path0_primary = delta_path1_primary - delta_gap_primary
    delta_path1_secondary = 1.6 + 0.4 * scale + float(rng.uniform(0.0, 0.15))
    delta_gap_secondary = 0.55 + 0.25 * scale + float(rng.uniform(0.0, 0.08))
    delta_path0_secondary = delta_path1_secondary - delta_gap_secondary

    return {
        "ref_inf": ref_inf,
        "delta_inf": delta_inf,
        "total_inf": total_inf,
        "ref_path0_primary": ref_path0_primary,
        "ref_path1_primary": ref_path1_primary,
        "ref_path0_secondary": ref_path0_secondary,
        "ref_path1_secondary": ref_path1_secondary,
        "delta_path0_primary": delta_path0_primary,
        "delta_path1_primary": delta_path1_primary,
        "delta_path0_secondary": delta_path0_secondary,
        "delta_path1_secondary": delta_path1_secondary,
    }


def _reference_path_value(model: dict[str, float], level_index: int, *, path1: bool) -> float:
    primary = model["ref_path1_primary"] if path1 else model["ref_path0_primary"]
    secondary = model["ref_path1_secondary"] if path1 else model["ref_path0_secondary"]
    return (
        model["ref_inf"]
        + primary * math.exp(-HF_ALPHA_DOMINANT * level_index)
        + secondary * math.exp(-HF_ALPHA_SUBLEADING * level_index)
    )


def _delta_path_value(model: dict[str, float], level_index: int, *, path1: bool) -> float:
    primary = model["delta_path1_primary"] if path1 else model["delta_path0_primary"]
    secondary = model["delta_path1_secondary"] if path1 else model["delta_path0_secondary"]
    return model["delta_inf"] + primary * (level_index ** -3) + secondary * (level_index ** -5)


def _reference_fit_sse(alpha: float, series_list: list[dict[int, float]]) -> float:
    total_sse = 0.0
    for series in series_list:
        xs = [math.exp(-alpha * level_index) for _, level_index in LADDER_LEVELS]
        ys = [float(series[level_index]) for _, level_index in LADDER_LEVELS]
        limit, amplitude = _fit_line(xs, ys)
        for x_value, y_value in zip(xs, ys, strict=True):
            residual = y_value - (limit + amplitude * x_value)
            total_sse += residual * residual
    return total_sse


def _fit_shared_reference_alpha(series_list: list[dict[int, float]]) -> float:
    left = 0.25
    right = 3.5
    phi = 0.5 * (1.0 + math.sqrt(5.0))
    c = right - (right - left) / phi
    d = left + (right - left) / phi
    f_c = _reference_fit_sse(c, series_list)
    f_d = _reference_fit_sse(d, series_list)

    for _ in range(140):
        if f_c <= f_d:
            right = d
            d = c
            f_d = f_c
            c = right - (right - left) / phi
            f_c = _reference_fit_sse(c, series_list)
        else:
            left = c
            c = d
            f_c = f_d
            d = left + (right - left) / phi
            f_d = _reference_fit_sse(d, series_list)

    return 0.5 * (left + right)


def _reference_limit_from_series(series: dict[int, float], alpha: float) -> float:
    xs = [math.exp(-alpha * level_index) for _, level_index in LADDER_LEVELS]
    ys = [float(series[level_index]) for _, level_index in LADDER_LEVELS]
    limit, _ = _fit_line(xs, ys)
    return limit


def _delta_limit_from_series(series: dict[int, float]) -> float:
    xs = [float(level_index ** -3) for _, level_index in LADDER_LEVELS if level_index >= 3]
    ys = [float(series[level_index]) for _, level_index in LADDER_LEVELS if level_index >= 3]
    limit, _ = _fit_line(xs, ys)
    return limit


def _make_energy_ledger(
    rng: np.random.Generator,
    cases: list[dict[str, Any]],
    semantic_to_code: dict[str, str],
    channel_map: dict[str, str],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    mono_a_ref, mono_b_ref, mono_a_total, mono_b_total = _monomer_energies(rng, cases)
    r_min = min(float(case["distance_axis_au"]) for case in cases)

    ledger_rows: list[dict[str, Any]] = []
    scan_rows: list[dict[str, Any]] = []
    shift_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    model_debug: dict[str, Any] = {}

    ref_col = channel_map["reference"]
    total_col = channel_map["total"]
    reference_series_all: list[dict[int, float]] = []

    for case in cases:
        case_id = str(case["case_id"])
        model = _interaction_model(rng, float(case["distance_axis_au"]), r_min)
        model_debug[case_id] = {"levels": {}}

        ref_path0_series: dict[int, float] = {}
        ref_path1_series: dict[int, float] = {}
        delta_path0_series: dict[int, float] = {}
        delta_path1_series: dict[int, float] = {}
        total_path0_series: dict[int, float] = {}
        total_path1_series: dict[int, float] = {}

        for level_label, level_index in LADDER_LEVELS:
            ref_path0 = _reference_path_value(model, level_index, path1=False)
            ref_path1 = _reference_path_value(model, level_index, path1=True)
            delta_path0 = _delta_path_value(model, level_index, path1=False)
            delta_path1 = _delta_path_value(model, level_index, path1=True)
            total_path0 = ref_path0 + delta_path0
            total_path1 = ref_path1 + delta_path1

            ref_gap = ref_path1 - ref_path0
            total_gap = total_path1 - total_path0

            split_ref = float(rng.uniform(0.38, 0.62))
            split_total = float(rng.uniform(0.35, 0.65))

            delta_a_ref = split_ref * ref_gap * KCAL_MOL_TO_HARTREE
            delta_b_ref = (1.0 - split_ref) * ref_gap * KCAL_MOL_TO_HARTREE
            delta_a_total = split_total * total_gap * KCAL_MOL_TO_HARTREE
            delta_b_total = (1.0 - split_total) * total_gap * KCAL_MOL_TO_HARTREE

            frag_a_native_ref = mono_a_ref[case_id][level_index]
            frag_b_native_ref = mono_b_ref[case_id][level_index]
            frag_a_native_total = mono_a_total[case_id][level_index]
            frag_b_native_total = mono_b_total[case_id][level_index]

            frag_a_balanced_ref = frag_a_native_ref - delta_a_ref
            frag_b_balanced_ref = frag_b_native_ref - delta_b_ref
            frag_a_balanced_total = frag_a_native_total - delta_a_total
            frag_b_balanced_total = frag_b_native_total - delta_b_total

            dimer_ref = frag_a_balanced_ref + frag_b_balanced_ref + ref_path1 * KCAL_MOL_TO_HARTREE
            dimer_total = frag_a_balanced_total + frag_b_balanced_total + total_path1 * KCAL_MOL_TO_HARTREE

            semantic_states = {
                "dimer_full": (dimer_ref, dimer_total),
                "frag_a_native": (frag_a_native_ref, frag_a_native_total),
                "frag_b_native": (frag_b_native_ref, frag_b_native_total),
                "frag_a_balanced": (frag_a_balanced_ref, frag_a_balanced_total),
                "frag_b_balanced": (frag_b_balanced_ref, frag_b_balanced_total),
            }

            state_rows: list[dict[str, Any]] = []
            for semantic_state in SEMANTIC_STATES:
                row_code = semantic_to_code[semantic_state]
                ref_value, total_value = semantic_states[semantic_state]
                row = {
                    "case_id": case_id,
                    "level_label": level_label,
                    "level_index": level_index,
                    "row_code": row_code,
                    "channel_0_hartree": "",
                    "channel_1_hartree": "",
                }
                row[ref_col] = f"{ref_value:.12f}"
                row[total_col] = f"{total_value:.12f}"
                state_rows.append(row)
            state_rows.sort(key=lambda row: str(row["row_code"]))
            ledger_rows.extend(state_rows)

            scan_rows.extend(
                [
                    {
                        "case_id": case_id,
                        "channel_role": "reference",
                        "level_label": level_label,
                        "level_index": level_index,
                        "path_0_kcal_mol": f"{ref_path0:.8f}",
                        "path_1_kcal_mol": f"{ref_path1:.8f}",
                        "path_gap_kcal_mol": f"{ref_gap:.8f}",
                    },
                    {
                        "case_id": case_id,
                        "channel_role": "total",
                        "level_label": level_label,
                        "level_index": level_index,
                        "path_0_kcal_mol": f"{total_path0:.8f}",
                        "path_1_kcal_mol": f"{total_path1:.8f}",
                        "path_gap_kcal_mol": f"{total_gap:.8f}",
                    },
                ]
            )

            ref_shift_pair = sorted(
                [delta_a_ref * HARTREE_TO_KCAL_MOL, delta_b_ref * HARTREE_TO_KCAL_MOL]
            )
            total_shift_pair = sorted(
                [delta_a_total * HARTREE_TO_KCAL_MOL, delta_b_total * HARTREE_TO_KCAL_MOL]
            )
            shift_rows.extend(
                [
                    {
                        "case_id": case_id,
                        "channel_role": "reference",
                        "level_label": level_label,
                        "level_index": level_index,
                        "shift_small_kcal_mol": f"{ref_shift_pair[0]:.8f}",
                        "shift_large_kcal_mol": f"{ref_shift_pair[1]:.8f}",
                    },
                    {
                        "case_id": case_id,
                        "channel_role": "total",
                        "level_label": level_label,
                        "level_index": level_index,
                        "shift_small_kcal_mol": f"{total_shift_pair[0]:.8f}",
                        "shift_large_kcal_mol": f"{total_shift_pair[1]:.8f}",
                    },
                ]
            )

            ref_path0_series[level_index] = ref_path0
            ref_path1_series[level_index] = ref_path1
            delta_path0_series[level_index] = delta_path0
            delta_path1_series[level_index] = delta_path1
            total_path0_series[level_index] = total_path0
            total_path1_series[level_index] = total_path1

            model_debug[case_id]["levels"][level_index] = {
                "reference_path_0": ref_path0,
                "reference_path_1": ref_path1,
                "delta_path_0": delta_path0,
                "delta_path_1": delta_path1,
                "total_path_0": total_path0,
                "total_path_1": total_path1,
                "reference_shift_small": ref_shift_pair[0],
                "reference_shift_large": ref_shift_pair[1],
                "total_shift_small": total_shift_pair[0],
                "total_shift_large": total_shift_pair[1],
            }

        reference_series_all.extend([ref_path0_series, ref_path1_series])
        model_debug[case_id]["reference_path_0_series"] = ref_path0_series
        model_debug[case_id]["reference_path_1_series"] = ref_path1_series
        model_debug[case_id]["delta_path_0_series"] = delta_path0_series
        model_debug[case_id]["delta_path_1_series"] = delta_path1_series
        model_debug[case_id]["total_path_0_series"] = total_path0_series
        model_debug[case_id]["total_path_1_series"] = total_path1_series

    shared_alpha = _fit_shared_reference_alpha(reference_series_all)

    for case in cases:
        case_id = str(case["case_id"])
        ref_path0_series = model_debug[case_id]["reference_path_0_series"]
        ref_path1_series = model_debug[case_id]["reference_path_1_series"]
        delta_path0_series = model_debug[case_id]["delta_path_0_series"]
        delta_path1_series = model_debug[case_id]["delta_path_1_series"]

        ref_limit_path0 = _reference_limit_from_series(ref_path0_series, shared_alpha)
        ref_limit_path1 = _reference_limit_from_series(ref_path1_series, shared_alpha)
        delta_limit_path0 = _delta_limit_from_series(delta_path0_series)
        delta_limit_path1 = _delta_limit_from_series(delta_path1_series)

        model_debug[case_id]["fitted_reference_limit_path_0"] = ref_limit_path0
        model_debug[case_id]["fitted_reference_limit_path_1"] = ref_limit_path1
        model_debug[case_id]["fitted_delta_limit_path_0"] = delta_limit_path0
        model_debug[case_id]["fitted_delta_limit_path_1"] = delta_limit_path1

        summary_rows.append(
            {
                "case_id": case_id,
                "reference_limit_path_0_kcal_mol": f"{ref_limit_path0:.8f}",
                "reference_limit_path_1_kcal_mol": f"{ref_limit_path1:.8f}",
                "delta_limit_path_0_kcal_mol": f"{delta_limit_path0:.8f}",
                "delta_limit_path_1_kcal_mol": f"{delta_limit_path1:.8f}",
                "total_limit_path_0_kcal_mol": f"{(ref_limit_path0 + delta_limit_path0):.8f}",
                "total_limit_path_1_kcal_mol": f"{(ref_limit_path1 + delta_limit_path1):.8f}",
            }
        )

    scan_rows.sort(
        key=lambda row: (str(row["case_id"]), str(row["channel_role"]), int(row["level_index"]))
    )
    shift_rows.sort(
        key=lambda row: (str(row["case_id"]), str(row["channel_role"]), int(row["level_index"]))
    )
    summary_rows.sort(key=lambda row: str(row["case_id"]))

    model_debug["shared_reference_alpha_fit"] = shared_alpha
    model_debug["dominant_reference_alpha_truth"] = HF_ALPHA_DOMINANT
    return ledger_rows, scan_rows, shift_rows, summary_rows, model_debug


def _make_ladder_spec(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "task_family": "two_path_family_ladder",
        "value_units": "hartree",
        "report_units": "kcal/mol",
        "ladder_levels": [
            {"level_label": level_label, "level_index": level_index}
            for level_label, level_index in LADDER_LEVELS
        ],
        "channel_columns": ["channel_0_hartree", "channel_1_hartree"],
        "row_code_count": len(SEMANTIC_STATES),
        "case_count": len(cases),
    }


def _make_reference_plot(
    cases: list[dict[str, Any]],
    scan_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    out_path: Path,
) -> None:
    case_order = [str(case["case_id"]) for case in cases]
    level_order = [level_index for _, level_index in LADDER_LEVELS]
    level_ticks = [f"L{level_index}" for _, level_index in LADDER_LEVELS]
    colors = ["#005f73", "#9b2226", "#ca6702", "#6c757d"]

    scan_map: dict[tuple[str, str, int], dict[str, float]] = {}
    for row in scan_rows:
        scan_map[(str(row["case_id"]), str(row["channel_role"]), int(row["level_index"]))] = {
            "path0": float(row["path_0_kcal_mol"]),
            "path1": float(row["path_1_kcal_mol"]),
        }

    summary_map = {
        str(row["case_id"]): {
            "reference_path0": float(row["reference_limit_path_0_kcal_mol"]),
            "reference_path1": float(row["reference_limit_path_1_kcal_mol"]),
            "total_path0": float(row["total_limit_path_0_kcal_mol"]),
            "total_path1": float(row["total_limit_path_1_kcal_mol"]),
        }
        for row in summary_rows
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)
    for ax, channel_role in zip(axes, OUTPUT_CHANNEL_ROLES, strict=True):
        for color, case_id in zip(colors, case_order, strict=True):
            path0_vals = [scan_map[(case_id, channel_role, idx)]["path0"] for idx in level_order]
            path1_vals = [scan_map[(case_id, channel_role, idx)]["path1"] for idx in level_order]
            limit_prefix = "reference" if channel_role == "reference" else "total"
            ax.plot(level_order, path0_vals, marker="o", color=color, linewidth=1.7, alpha=0.95)
            ax.plot(
                level_order,
                path1_vals,
                marker="s",
                color=color,
                linewidth=1.7,
                linestyle="--",
                alpha=0.95,
            )
            ax.scatter(
                [5.28],
                [summary_map[case_id][f"{limit_prefix}_path0"]],
                marker="D",
                color=color,
                s=28,
            )
            ax.scatter(
                [5.40],
                [summary_map[case_id][f"{limit_prefix}_path1"]],
                marker="^",
                color=color,
                s=28,
            )
        ax.set_xticks(level_order, level_ticks)
        ax.set_xlabel("Ladder level")
        ax.set_ylabel("Interaction energy (kcal/mol)")
        ax.set_title(f"{channel_role} channel: path_0 vs path_1")
        ax.grid(alpha=0.25, linestyle=":")
        ax.set_xlim(level_order[0] - 0.1, 5.52)

    fig.suptitle("Two-path weak-dimer convergence across the ladder", fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _render_b1_prompt(
    task_dir: Path,
    semantic_to_code: dict[str, str],
    channel_map: dict[str, str],
) -> str:
    base = (task_dir / "prompt_b1.md").read_text(encoding="utf-8")
    code_lines = [
        f"- `{semantic_to_code['dimer_full']}` = combined-system row",
        f"- `{semantic_to_code['frag_a_native']}` and `{semantic_to_code['frag_b_native']}` = local/native fragment rows",
        f"- `{semantic_to_code['frag_a_balanced']}` and `{semantic_to_code['frag_b_balanced']}` = expanded/balanced fragment rows",
    ]
    channel_lines = [
        f"- `{channel_map['reference']}` is the lower-order reference-like channel",
        f"- `{channel_map['total']}` is the higher-order total channel",
    ]
    preamble = "\n".join(
        [
            "Instance-specific decoding for this seed:",
            "",
            "Row-code map:",
            *code_lines,
            "",
            "Channel-column map:",
            *channel_lines,
            "",
            "Use `path_0` for the native/local pair and `path_1` for the expanded/balanced pair.",
            "Use `path_gap_kcal_mol = path_1_kcal_mol - path_0_kcal_mol` exactly.",
            "Also write `results/pair_shift_scan.csv` with the two positive within-family stabilization shifts for each case/channel/level, sorted ascending as `shift_small_kcal_mol` and `shift_large_kcal_mol`.",
            "For the reference-like channel, fit a single dominant-alpha model `E(X) = E_inf + A * exp(-alpha * X)` across levels 2, 3, 4, 5.",
            "Recover one shared `alpha` from all reference-path series together, then fit `E_inf` separately for each case/path at that fixed alpha.",
            "For the residual channel `total - reference`, fit `E(X) = E_inf + B * X^-3` by linear regression over levels 3, 4, 5.",
            "",
        ]
    )
    return f"{preamble}\n{base}"


def generate(output_dir: Path, params: dict) -> dict[str, Any]:
    merged = {**DEFAULT_PARAMS, **params}
    start = time.time()
    rng = np.random.default_rng(int(merged["seed"]))

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    ref_dir = output_dir / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)

    cases = _sample_cases(rng)
    semantic_to_code, code_to_semantic = _make_code_maps(rng)
    channel_map = _make_channel_map(rng)
    ladder_spec = _make_ladder_spec(cases)
    ledger_rows, scan_rows, shift_rows, summary_rows, model_debug = _make_energy_ledger(
        rng,
        cases,
        semantic_to_code,
        channel_map,
    )

    _write_csv(
        data_dir / "case_index.csv",
        ["case_id", "geometry_label", "distance_axis_au"],
        cases,
    )
    (data_dir / "ladder_spec.json").write_text(json.dumps(ladder_spec, indent=2), encoding="utf-8")
    _write_csv(
        data_dir / "channel_ledger.csv",
        [
            "case_id",
            "level_label",
            "level_index",
            "row_code",
            "channel_0_hartree",
            "channel_1_hartree",
        ],
        ledger_rows,
    )

    _write_csv(ref_dir / "path_scan_ref.csv", SCAN_COLUMNS, scan_rows)
    _write_csv(ref_dir / "pair_shift_scan_ref.csv", SHIFT_COLUMNS, shift_rows)
    _write_csv(ref_dir / "asymptotic_summary_ref.csv", SUMMARY_COLUMNS, summary_rows)
    _make_reference_plot(cases, scan_rows, summary_rows, ref_dir / "convergence_plot_ref.png")

    task_dir = Path(__file__).parent
    (output_dir / "prompt_b1.md").write_text(
        _render_b1_prompt(task_dir, semantic_to_code, channel_map),
        encoding="utf-8",
    )
    for level in ("b2", "b3", "b4"):
        prompt_text = (task_dir / f"prompt_{level}.md").read_text(encoding="utf-8")
        (output_dir / f"prompt_{level}.md").write_text(prompt_text, encoding="utf-8")

    (ref_dir / "reference_truth.json").write_text(
        json.dumps(
            {
                "dominant_reference_alpha_truth": HF_ALPHA_DOMINANT,
                "ladder_levels": LADDER_LEVELS,
                "cases": cases,
                "row_code_map": semantic_to_code,
                "row_code_inverse": code_to_semantic,
                "channel_map": channel_map,
                "model_debug": model_debug,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    meta = {
        "params_used": merged,
        "input_files": [spec["name"] for spec in INPUT_SPEC],
        "reference_files": [
            "path_scan_ref.csv",
            "pair_shift_scan_ref.csv",
            "asymptotic_summary_ref.csv",
            "convergence_plot_ref.png",
            "reference_truth.json",
        ],
        "generation_time_seconds": round(time.time() - start, 2),
    }
    (output_dir / "instance_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--params", type=str, default="{}")
    args = parser.parse_args()
    result = generate(args.output_dir, json.loads(args.params))
    print(json.dumps(result, indent=2))
