from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from ai4sci_bench.core.scorer import Scorer, register_scorer
from ai4sci_bench.core.types import ScoreDetail


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

LEVEL_ORDER = {
    "level_02": 2,
    "level_03": 3,
    "level_04": 4,
    "level_05": 5,
}
CHANNEL_ROLE_ORDER = {"reference": 0, "total": 1}


def _read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _safe_float(value: object) -> float:
    try:
        number = float(value)
    except Exception:
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _score_exp_lower_better(error: float, scale: float) -> float:
    if not math.isfinite(error):
        return 0.0
    if error <= 0.0:
        return 100.0
    return float(max(0.0, min(100.0, 100.0 * math.exp(-error / max(scale, 1.0e-12)))))


def _rmse(pred: np.ndarray, ref: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if pred.shape != ref.shape:
        return float("inf")
    diff = pred - ref
    return float(np.sqrt(np.mean(diff * diff)))


def _load_scan_table(path: Path) -> tuple[list[tuple[str, str, str]], np.ndarray]:
    rows, columns = _read_csv_rows(path)
    if columns != SCAN_COLUMNS:
        raise ValueError(f"path_scan.csv columns mismatch: expected {SCAN_COLUMNS}, got {columns}")

    parsed: list[tuple[tuple[str, str, str], list[float]]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        case_id = str(row["case_id"]).strip()
        channel_role = str(row["channel_role"]).strip()
        level_label = str(row["level_label"]).strip()
        key = (case_id, channel_role, level_label)
        if key in seen:
            raise ValueError(f"duplicate scan row for {key}")
        seen.add(key)

        level_index = int(row["level_index"])
        if level_label not in LEVEL_ORDER:
            raise ValueError(f"unexpected level label: {level_label}")
        if channel_role not in CHANNEL_ROLE_ORDER:
            raise ValueError(f"unexpected channel role: {channel_role}")
        if level_index != LEVEL_ORDER[level_label]:
            raise ValueError(f"level_index mismatch for {level_label}: {level_index}")

        path0 = _safe_float(row["path_0_kcal_mol"])
        path1 = _safe_float(row["path_1_kcal_mol"])
        path_gap = _safe_float(row["path_gap_kcal_mol"])
        if not all(math.isfinite(v) for v in (path0, path1, path_gap)):
            raise ValueError(f"non-finite scan row for {key}")
        parsed.append((key, [path0, path1, path_gap]))

    parsed.sort(key=lambda item: (item[0][0], CHANNEL_ROLE_ORDER[item[0][1]], LEVEL_ORDER[item[0][2]]))
    keys = [item[0] for item in parsed]
    values = np.asarray([item[1] for item in parsed], dtype=float)
    return keys, values


def _load_shift_table(path: Path) -> tuple[list[tuple[str, str, str]], np.ndarray]:
    rows, columns = _read_csv_rows(path)
    if columns != SHIFT_COLUMNS:
        raise ValueError(f"pair_shift_scan.csv columns mismatch: expected {SHIFT_COLUMNS}, got {columns}")

    parsed: list[tuple[tuple[str, str, str], list[float]]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        case_id = str(row["case_id"]).strip()
        channel_role = str(row["channel_role"]).strip()
        level_label = str(row["level_label"]).strip()
        key = (case_id, channel_role, level_label)
        if key in seen:
            raise ValueError(f"duplicate shift row for {key}")
        seen.add(key)

        level_index = int(row["level_index"])
        if level_label not in LEVEL_ORDER:
            raise ValueError(f"unexpected level label: {level_label}")
        if channel_role not in CHANNEL_ROLE_ORDER:
            raise ValueError(f"unexpected channel role: {channel_role}")
        if level_index != LEVEL_ORDER[level_label]:
            raise ValueError(f"level_index mismatch for {level_label}: {level_index}")

        shift_small = _safe_float(row["shift_small_kcal_mol"])
        shift_large = _safe_float(row["shift_large_kcal_mol"])
        if not all(math.isfinite(v) for v in (shift_small, shift_large)):
            raise ValueError(f"non-finite shift row for {key}")
        parsed.append((key, [shift_small, shift_large]))

    parsed.sort(key=lambda item: (item[0][0], CHANNEL_ROLE_ORDER[item[0][1]], LEVEL_ORDER[item[0][2]]))
    keys = [item[0] for item in parsed]
    values = np.asarray([item[1] for item in parsed], dtype=float)
    return keys, values


def _load_summary_table(path: Path) -> tuple[list[str], np.ndarray]:
    rows, columns = _read_csv_rows(path)
    if columns != SUMMARY_COLUMNS:
        raise ValueError(
            f"asymptotic_summary.csv columns mismatch: expected {SUMMARY_COLUMNS}, got {columns}"
        )

    parsed: list[tuple[str, list[float]]] = []
    seen: set[str] = set()
    for row in rows:
        case_id = str(row["case_id"]).strip()
        if case_id in seen:
            raise ValueError(f"duplicate summary row for {case_id}")
        seen.add(case_id)

        vals = [
            _safe_float(row["reference_limit_path_0_kcal_mol"]),
            _safe_float(row["reference_limit_path_1_kcal_mol"]),
            _safe_float(row["delta_limit_path_0_kcal_mol"]),
            _safe_float(row["delta_limit_path_1_kcal_mol"]),
            _safe_float(row["total_limit_path_0_kcal_mol"]),
            _safe_float(row["total_limit_path_1_kcal_mol"]),
        ]
        if not all(math.isfinite(v) for v in vals):
            raise ValueError(f"non-finite summary row for {case_id}")
        parsed.append((case_id, vals))

    parsed.sort(key=lambda item: item[0])
    case_ids = [item[0] for item in parsed]
    values = np.asarray([item[1] for item in parsed], dtype=float)
    return case_ids, values


@register_scorer("bsse_output_sanity")
class BsseOutputSanityScorer(Scorer):
    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        weight = float(config.get("weight", 1.0))
        scan_file = str(config.get("scan_file", "results/path_scan.csv"))
        shift_file = str(config.get("shift_file", "results/pair_shift_scan.csv"))
        summary_file = str(config.get("summary_file", "results/asymptotic_summary.csv"))
        expected_case_count = int(config.get("expected_case_count", 4))
        expected_level_labels = list(config.get("expected_level_labels", list(LEVEL_ORDER)))

        try:
            scan_keys, scan_values = _load_scan_table(pred_dir / scan_file)
            shift_keys, shift_values = _load_shift_table(pred_dir / shift_file)
            case_ids, summary_values = _load_summary_table(pred_dir / summary_file)
        except Exception as exc:
            return ScoreDetail(
                scorer_name="bsse_output_sanity",
                score=0.0,
                max_score=weight,
                passed=False,
                details={"error": str(exc)},
                message=f"output sanity failure: {exc}",
            )

        level_labels = sorted({level for _, _, level in scan_keys})
        channel_roles = sorted({role for _, role, _ in scan_keys})
        expected_rows = expected_case_count * len(expected_level_labels) * len(CHANNEL_ROLE_ORDER)
        gap_error = float(np.max(np.abs(scan_values[:, 2] - (scan_values[:, 1] - scan_values[:, 0]))))

        passed = (
            len(scan_keys) == expected_rows
            and len(shift_keys) == expected_rows
            and scan_keys == shift_keys
            and len(case_ids) == expected_case_count
            and level_labels == sorted(expected_level_labels)
            and channel_roles == ["reference", "total"]
            and gap_error <= 1.0e-7
        )
        return ScoreDetail(
            scorer_name="bsse_output_sanity",
            score=weight if passed else 0.0,
            max_score=weight,
            passed=passed,
            details={
                "scan_row_count": len(scan_keys),
                "shift_row_count": len(shift_keys),
                "summary_row_count": len(case_ids),
                "level_labels": level_labels,
                "channel_roles": channel_roles,
                "shared_row_keys": scan_keys == shift_keys,
                "internal_gap_error": gap_error,
                "shift_shape": list(shift_values.shape),
                "summary_shape": list(summary_values.shape),
            },
            message=(
                f"scan_rows={len(scan_keys)}, shift_rows={len(shift_keys)}, "
                f"summary_rows={len(case_ids)}, gap_error={gap_error:.2e}"
            ),
        )


@register_scorer("bsse_scan_score")
class BsseScanScoreScorer(Scorer):
    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        weight = float(config.get("weight", 1.0))
        scan_file = str(config.get("scan_file", "results/path_scan.csv"))
        reference_file = str(config.get("reference_file", "path_scan_ref.csv"))

        try:
            pred_keys, pred_values = _load_scan_table(pred_dir / scan_file)
            ref_keys, ref_values = _load_scan_table(ref_dir / reference_file)
        except Exception as exc:
            return ScoreDetail(
                scorer_name="bsse_scan_score",
                score=0.0,
                max_score=weight,
                passed=False,
                details={"error": str(exc)},
                message=f"scan score failure: {exc}",
            )

        if pred_keys != ref_keys:
            return ScoreDetail(
                scorer_name="bsse_scan_score",
                score=0.0,
                max_score=weight,
                passed=False,
                details={"pred_keys": pred_keys, "ref_keys": ref_keys},
                message="scan row-key mismatch",
            )

        path0_rmse = _rmse(pred_values[:, 0], ref_values[:, 0])
        path1_rmse = _rmse(pred_values[:, 1], ref_values[:, 1])
        gap_rmse = _rmse(pred_values[:, 2], ref_values[:, 2])

        path0_score = _score_exp_lower_better(path0_rmse, float(config.get("path0_scale", 0.05)))
        path1_score = _score_exp_lower_better(path1_rmse, float(config.get("path1_scale", 0.05)))
        gap_score = _score_exp_lower_better(gap_rmse, float(config.get("gap_scale", 0.03)))

        score_100 = 0.20 * path0_score + 0.25 * path1_score + 0.55 * gap_score
        return ScoreDetail(
            scorer_name="bsse_scan_score",
            score=weight * score_100 / 100.0,
            max_score=weight,
            passed=True,
            details={
                "path0_rmse_kcal_mol": path0_rmse,
                "path1_rmse_kcal_mol": path1_rmse,
                "gap_rmse_kcal_mol": gap_rmse,
                "component_score_100": {
                    "path0": path0_score,
                    "path1": path1_score,
                    "gap": gap_score,
                    "combined": score_100,
                },
            },
            message=(
                f"scan score={score_100:.1f}/100; "
                f"path0_rmse={path0_rmse:.4f}, path1_rmse={path1_rmse:.4f}, gap_rmse={gap_rmse:.4f}"
            ),
        )


@register_scorer("bsse_shift_score")
class BsseShiftScoreScorer(Scorer):
    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        weight = float(config.get("weight", 1.0))
        shift_file = str(config.get("shift_file", "results/pair_shift_scan.csv"))
        reference_file = str(config.get("reference_file", "pair_shift_scan_ref.csv"))
        scan_file = str(config.get("scan_file", "results/path_scan.csv"))

        try:
            pred_keys, pred_values = _load_shift_table(pred_dir / shift_file)
            ref_keys, ref_values = _load_shift_table(ref_dir / reference_file)
            scan_keys, scan_values = _load_scan_table(pred_dir / scan_file)
        except Exception as exc:
            return ScoreDetail(
                scorer_name="bsse_shift_score",
                score=0.0,
                max_score=weight,
                passed=False,
                details={"error": str(exc)},
                message=f"shift score failure: {exc}",
            )

        if pred_keys != ref_keys:
            return ScoreDetail(
                scorer_name="bsse_shift_score",
                score=0.0,
                max_score=weight,
                passed=False,
                details={"pred_keys": pred_keys, "ref_keys": ref_keys},
                message="shift row-key mismatch",
            )
        if pred_keys != scan_keys:
            return ScoreDetail(
                scorer_name="bsse_shift_score",
                score=0.0,
                max_score=weight,
                passed=False,
                details={"shift_keys": pred_keys, "scan_keys": scan_keys},
                message="shift/scan row-key mismatch",
            )

        scan_gap_by_key = {key: value[2] for key, value in zip(scan_keys, scan_values, strict=True)}
        pred_gap_sums = np.asarray(
            [pred_values[idx, 0] + pred_values[idx, 1] for idx in range(pred_values.shape[0])],
            dtype=float,
        )
        paired_gap = np.asarray([scan_gap_by_key[key] for key in pred_keys], dtype=float)

        shift_small_rmse = _rmse(pred_values[:, 0], ref_values[:, 0])
        shift_large_rmse = _rmse(pred_values[:, 1], ref_values[:, 1])
        shift_sum_consistency_rmse = _rmse(pred_gap_sums, paired_gap)

        shift_small_score = _score_exp_lower_better(
            shift_small_rmse,
            float(config.get("shift_small_scale", 0.02)),
        )
        shift_large_score = _score_exp_lower_better(
            shift_large_rmse,
            float(config.get("shift_large_scale", 0.02)),
        )
        shift_sum_consistency_score = _score_exp_lower_better(
            shift_sum_consistency_rmse,
            float(config.get("sum_consistency_scale", 0.01)),
        )

        raw_score_100 = 0.50 * shift_small_score + 0.50 * shift_large_score
        capped_score_100 = raw_score_100 * (0.15 + 0.85 * shift_sum_consistency_score / 100.0)

        return ScoreDetail(
            scorer_name="bsse_shift_score",
            score=weight * capped_score_100 / 100.0,
            max_score=weight,
            passed=True,
            details={
                "shift_small_rmse_kcal_mol": shift_small_rmse,
                "shift_large_rmse_kcal_mol": shift_large_rmse,
                "shift_sum_consistency_rmse_kcal_mol": shift_sum_consistency_rmse,
                "component_score_100": {
                    "shift_small": shift_small_score,
                    "shift_large": shift_large_score,
                    "sum_consistency": shift_sum_consistency_score,
                    "raw_combined": raw_score_100,
                    "capped_combined": capped_score_100,
                },
            },
            message=(
                f"shift score={capped_score_100:.1f}/100; "
                f"small_rmse={shift_small_rmse:.4f}, large_rmse={shift_large_rmse:.4f}, "
                f"sum_consistency_rmse={shift_sum_consistency_rmse:.4f}"
            ),
        )


@register_scorer("bsse_asymptotic_score")
class BsseAsymptoticScoreScorer(Scorer):
    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        weight = float(config.get("weight", 1.0))
        summary_file = str(config.get("summary_file", "results/asymptotic_summary.csv"))
        reference_file = str(config.get("reference_file", "asymptotic_summary_ref.csv"))
        scan_file = str(config.get("scan_file", "results/path_scan.csv"))
        scan_reference_file = str(config.get("scan_reference_file", "path_scan_ref.csv"))
        shift_file = str(config.get("shift_file", "results/pair_shift_scan.csv"))
        shift_reference_file = str(config.get("shift_reference_file", "pair_shift_scan_ref.csv"))

        try:
            pred_case_ids, pred_values = _load_summary_table(pred_dir / summary_file)
            ref_case_ids, ref_values = _load_summary_table(ref_dir / reference_file)
            pred_scan_keys, pred_scan_values = _load_scan_table(pred_dir / scan_file)
            ref_scan_keys, ref_scan_values = _load_scan_table(ref_dir / scan_reference_file)
            pred_shift_keys, pred_shift_values = _load_shift_table(pred_dir / shift_file)
            ref_shift_keys, ref_shift_values = _load_shift_table(ref_dir / shift_reference_file)
        except Exception as exc:
            return ScoreDetail(
                scorer_name="bsse_asymptotic_score",
                score=0.0,
                max_score=weight,
                passed=False,
                details={"error": str(exc)},
                message=f"asymptotic score failure: {exc}",
            )

        if pred_case_ids != ref_case_ids:
            return ScoreDetail(
                scorer_name="bsse_asymptotic_score",
                score=0.0,
                max_score=weight,
                passed=False,
                details={"pred_case_ids": pred_case_ids, "ref_case_ids": ref_case_ids},
                message="asymptotic case_id mismatch",
            )
        if pred_scan_keys != ref_scan_keys:
            return ScoreDetail(
                scorer_name="bsse_asymptotic_score",
                score=0.0,
                max_score=weight,
                passed=False,
                details={"pred_scan_keys": pred_scan_keys, "ref_scan_keys": ref_scan_keys},
                message="scan keys mismatch while capping asymptotic score",
            )
        if pred_shift_keys != ref_shift_keys:
            return ScoreDetail(
                scorer_name="bsse_asymptotic_score",
                score=0.0,
                max_score=weight,
                passed=False,
                details={"pred_shift_keys": pred_shift_keys, "ref_shift_keys": ref_shift_keys},
                message="shift keys mismatch while capping asymptotic score",
            )

        ref_path0_rmse = _rmse(pred_values[:, 0], ref_values[:, 0])
        ref_path1_rmse = _rmse(pred_values[:, 1], ref_values[:, 1])
        delta_path0_rmse = _rmse(pred_values[:, 2], ref_values[:, 2])
        delta_path1_rmse = _rmse(pred_values[:, 3], ref_values[:, 3])
        total_path0_rmse = _rmse(pred_values[:, 4], ref_values[:, 4])
        total_path1_rmse = _rmse(pred_values[:, 5], ref_values[:, 5])

        ref_scale = float(config.get("reference_scale", 0.03))
        delta_scale = float(config.get("delta_scale", 0.035))
        total_scale = float(config.get("total_scale", 0.04))

        ref_path0_score = _score_exp_lower_better(ref_path0_rmse, ref_scale)
        ref_path1_score = _score_exp_lower_better(ref_path1_rmse, ref_scale)
        delta_path0_score = _score_exp_lower_better(delta_path0_rmse, delta_scale)
        delta_path1_score = _score_exp_lower_better(delta_path1_rmse, delta_scale)
        total_path0_score = _score_exp_lower_better(total_path0_rmse, total_scale)
        total_path1_score = _score_exp_lower_better(total_path1_rmse, total_scale)

        raw_score_100 = (
            0.12 * ref_path0_score
            + 0.12 * ref_path1_score
            + 0.22 * delta_path0_score
            + 0.22 * delta_path1_score
            + 0.16 * total_path0_score
            + 0.16 * total_path1_score
        )

        scan_path0_rmse = _rmse(pred_scan_values[:, 0], ref_scan_values[:, 0])
        scan_path1_rmse = _rmse(pred_scan_values[:, 1], ref_scan_values[:, 1])
        scan_gap_rmse = _rmse(pred_scan_values[:, 2], ref_scan_values[:, 2])
        shift_small_rmse = _rmse(pred_shift_values[:, 0], ref_shift_values[:, 0])
        shift_large_rmse = _rmse(pred_shift_values[:, 1], ref_shift_values[:, 1])

        scan_path0_score = _score_exp_lower_better(scan_path0_rmse, float(config.get("scan_path0_scale", 0.05)))
        scan_path1_score = _score_exp_lower_better(scan_path1_rmse, float(config.get("scan_path1_scale", 0.05)))
        scan_gap_score = _score_exp_lower_better(scan_gap_rmse, float(config.get("scan_gap_scale", 0.03)))
        shift_small_score = _score_exp_lower_better(
            shift_small_rmse,
            float(config.get("shift_small_scale", 0.02)),
        )
        shift_large_score = _score_exp_lower_better(
            shift_large_rmse,
            float(config.get("shift_large_scale", 0.02)),
        )

        low_level_factor = min(
            scan_path0_score,
            scan_path1_score,
            scan_gap_score,
            shift_small_score,
            shift_large_score,
        ) / 100.0
        capped_score_100 = raw_score_100 * (0.10 + 0.90 * low_level_factor)

        return ScoreDetail(
            scorer_name="bsse_asymptotic_score",
            score=weight * capped_score_100 / 100.0,
            max_score=weight,
            passed=True,
            details={
                "reference_path0_rmse_kcal_mol": ref_path0_rmse,
                "reference_path1_rmse_kcal_mol": ref_path1_rmse,
                "delta_path0_rmse_kcal_mol": delta_path0_rmse,
                "delta_path1_rmse_kcal_mol": delta_path1_rmse,
                "total_path0_rmse_kcal_mol": total_path0_rmse,
                "total_path1_rmse_kcal_mol": total_path1_rmse,
                "low_level_consistency": {
                    "scan_path0_rmse_kcal_mol": scan_path0_rmse,
                    "scan_path1_rmse_kcal_mol": scan_path1_rmse,
                    "scan_gap_rmse_kcal_mol": scan_gap_rmse,
                    "shift_small_rmse_kcal_mol": shift_small_rmse,
                    "shift_large_rmse_kcal_mol": shift_large_rmse,
                    "scan_path0_score_100": scan_path0_score,
                    "scan_path1_score_100": scan_path1_score,
                    "scan_gap_score_100": scan_gap_score,
                    "shift_small_score_100": shift_small_score,
                    "shift_large_score_100": shift_large_score,
                    "low_level_factor": low_level_factor,
                },
                "component_score_100": {
                    "reference_path0": ref_path0_score,
                    "reference_path1": ref_path1_score,
                    "delta_path0": delta_path0_score,
                    "delta_path1": delta_path1_score,
                    "total_path0": total_path0_score,
                    "total_path1": total_path1_score,
                    "raw_combined": raw_score_100,
                    "capped_combined": capped_score_100,
                },
            },
            message=(
                f"asymptotic score={capped_score_100:.1f}/100 "
                f"(raw={raw_score_100:.1f}, low_level_factor={low_level_factor:.2f})"
            ),
        )
