"""Public, task-local scorers for the option-surface reconstruction task."""

from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ai4sci_bench.core.scorer import Scorer, register_scorer
from ai4sci_bench.core.types import ScoreDetail


QUERY_COLUMNS = [
    "query_id",
    "call_price",
    "implied_volatility",
    "total_variance",
    "risk_neutral_density",
    "digital_survival_probability",
]
QUOTE_COLUMNS = [
    "quote_id",
    "fitted_call_price",
    "standardized_residual",
    "outlier_score",
]
QUERY_INPUT_COLUMNS = [
    "query_id",
    "maturity",
    "log_moneyness",
    "strike",
    "forward",
    "discount_factor",
]
QUOTE_INPUT_COLUMNS = [
    "quote_id",
    "maturity",
    "log_moneyness",
    "strike",
    "forward",
    "discount_factor",
    "bid",
    "ask",
]
AUDIT_KEYS = [
    "max_bound_violation",
    "max_monotonicity_violation",
    "max_convexity_violation",
    "max_calendar_violation",
    "minimum_density",
    "max_self_consistency_error",
    "max_digital_range_violation",
]


def _detail(
    name: str,
    weight: float,
    fraction: float,
    passed: bool,
    details: dict[str, Any],
    message: str,
) -> ScoreDetail:
    fraction = float(max(0.0, min(1.0, fraction)))
    return ScoreDetail(
        scorer_name=name,
        score=weight * fraction,
        max_score=weight,
        passed=passed,
        details={**details, "score_fraction": fraction},
        message=message,
    )


def _lower_better(value: float, full: float, zero: float) -> float:
    if not math.isfinite(value) or zero <= full:
        return 0.0
    if value <= full:
        return 1.0
    if value >= zero:
        return 0.0
    return float((zero - value) / (zero - full))


def _higher_better(value: float, zero: float, full: float) -> float:
    if not math.isfinite(value) or full <= zero:
        return 0.0
    if value <= zero:
        return 0.0
    if value >= full:
        return 1.0
    return float((value - zero) / (full - zero))


def _safe_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except Exception as exc:
        raise ValueError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite")
    return number


def _load_table(
    path: Path,
    expected_columns: list[str],
    id_column: str,
) -> tuple[list[str], dict[str, np.ndarray]]:
    if not path.is_file():
        raise ValueError(f"missing {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        if columns != expected_columns:
            raise ValueError(
                f"{path.name} columns mismatch: expected {expected_columns}, got {columns}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path.name} is empty")

    ids: list[str] = []
    values: dict[str, list[float]] = {
        column: [] for column in expected_columns if column != id_column
    }
    seen: set[str] = set()
    for row_index, row in enumerate(rows):
        identifier = str(row[id_column]).strip()
        if not identifier:
            raise ValueError(f"{path.name} row {row_index} has an empty {id_column}")
        if identifier in seen:
            raise ValueError(f"{path.name} contains duplicate {id_column}={identifier}")
        seen.add(identifier)
        ids.append(identifier)
        for column in values:
            values[column].append(
                _safe_float(row[column], f"{path.name}:{identifier}:{column}")
            )
    return ids, {key: np.asarray(value, dtype=np.float64) for key, value in values.items()}


def _aligned_values(
    pred_path: Path,
    ref_path: Path,
    columns: list[str],
    id_column: str,
) -> tuple[list[str], dict[str, np.ndarray], dict[str, np.ndarray]]:
    pred_ids, pred = _load_table(pred_path, columns, id_column)
    ref_ids, ref = _load_table(ref_path, columns, id_column)
    if set(pred_ids) != set(ref_ids):
        missing = sorted(set(ref_ids) - set(pred_ids))[:5]
        extra = sorted(set(pred_ids) - set(ref_ids))[:5]
        raise ValueError(f"ID mismatch; missing={missing}, extra={extra}")
    pred_index = {identifier: idx for idx, identifier in enumerate(pred_ids)}
    order = np.asarray([pred_index[identifier] for identifier in ref_ids], dtype=int)
    pred_aligned = {key: value[order] for key, value in pred.items()}
    return ref_ids, pred_aligned, ref


def _load_query_input(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    return _load_table(path, QUERY_INPUT_COLUMNS, "query_id")


def _load_quote_input(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    return _load_table(path, QUOTE_INPUT_COLUMNS, "quote_id")


def _align_input(
    expected_ids: list[str],
    input_ids: list[str],
    values: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if set(expected_ids) != set(input_ids):
        raise ValueError("public input IDs do not match the reference contract")
    index = {identifier: idx for idx, identifier in enumerate(input_ids)}
    order = np.asarray([index[identifier] for identifier in expected_ids], dtype=int)
    return {key: value[order] for key, value in values.items()}


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    if y.size < 2 or x.size != y.size:
        return 0.0
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x)))


def _normal_cdf(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).ravel()
    result = np.fromiter(
        (0.5 * math.erfc(-float(value) / math.sqrt(2.0)) for value in flat),
        dtype=np.float64,
        count=flat.size,
    )
    return result.reshape(np.asarray(values).shape)


def _black_call(
    total_variance: np.ndarray,
    log_moneyness: np.ndarray,
    forward: np.ndarray,
    discount: np.ndarray,
) -> np.ndarray:
    w = np.maximum(np.asarray(total_variance, dtype=np.float64), 1.0e-16)
    k = np.asarray(log_moneyness, dtype=np.float64)
    fwd = np.asarray(forward, dtype=np.float64)
    disc = np.asarray(discount, dtype=np.float64)
    sqrt_w = np.sqrt(w)
    d1 = -k / sqrt_w + 0.5 * sqrt_w
    d2 = d1 - sqrt_w
    strike = fwd * np.exp(k)
    return disc * (fwd * _normal_cdf(d1) - strike * _normal_cdf(d2))


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels >= 0.5, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    i = 0
    while i < scores.size:
        j = i + 1
        while j < scores.size and scores[order[j]] == scores[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * ((i + 1) + j)
        i = j
    rank_sum = float(ranks[labels == 1].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels >= 0.5, dtype=np.int8)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")
    sorted_labels = labels[order]
    precision = np.cumsum(sorted_labels) / np.arange(1, labels.size + 1)
    return float(np.sum(precision * sorted_labels) / n_pos)


@register_scorer("option_surface_contract")
class OptionSurfaceContractScorer(Scorer):
    """Hard gate for schemas, identities, finiteness, and gross numeric ranges."""

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        weight = float(config.get("weight", 1.0))
        try:
            query_ids, query, _ = _aligned_values(
                pred_dir / config.get("pred_file", "results/query_surface.csv"),
                ref_dir / config.get("ref_file", "query_surface_ref.csv"),
                QUERY_COLUMNS,
                "query_id",
            )
            quote_ids, quote, _ = _aligned_values(
                pred_dir / config.get("quote_file", "results/quote_diagnostics.csv"),
                ref_dir / config.get("reference_quote_file", "quote_diagnostics_ref.csv"),
                QUOTE_COLUMNS,
                "quote_id",
            )
            public_query_ids, public_query = _load_query_input(pred_dir / "data/query_grid.csv")
            public_quote_ids, public_quotes = _load_quote_input(pred_dir / "data/quotes.csv")
            public_query = _align_input(query_ids, public_query_ids, public_query)
            public_quotes = _align_input(quote_ids, public_quote_ids, public_quotes)

            upper = public_query["discount_factor"] * public_query["forward"]
            if np.any(query["call_price"] < -1.0e-8):
                raise ValueError("query call_price contains materially negative values")
            if np.any(query["call_price"] > 1.05 * upper + 1.0e-8):
                raise ValueError("query call_price grossly exceeds the vanilla upper bound")
            if np.any((query["implied_volatility"] <= 0.0) | (query["implied_volatility"] > 5.0)):
                raise ValueError("implied_volatility must lie in (0,5]")
            if np.any((query["total_variance"] <= 0.0) | (query["total_variance"] > 10.0)):
                raise ValueError("total_variance must lie in (0,10]")
            if np.any((query["risk_neutral_density"] < -5.0) | (query["risk_neutral_density"] > 100.0)):
                raise ValueError("risk_neutral_density is outside the gross safety range")
            if np.any(
                (query["digital_survival_probability"] < 0.0)
                | (query["digital_survival_probability"] > 1.0)
            ):
                raise ValueError("digital_survival_probability must be in [0,1]")
            if np.any((quote["outlier_score"] < 0.0) | (quote["outlier_score"] > 1.0)):
                raise ValueError("outlier_score must be in [0,1]")
            quote_upper = public_quotes["discount_factor"] * public_quotes["forward"]
            if np.any(quote["fitted_call_price"] < -1.0e-8) or np.any(
                quote["fitted_call_price"] > 1.05 * quote_upper + 1.0e-8
            ):
                raise ValueError("fitted_call_price is outside gross vanilla bounds")

            audit_path = pred_dir / config.get("audit_file", "results/arbitrage_audit.json")
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if not isinstance(audit, dict):
                raise ValueError("arbitrage_audit.json must contain a JSON object")
            for key in AUDIT_KEYS:
                _safe_float(audit.get(key), f"arbitrage_audit.json:{key}")

            figure_path = pred_dir / config.get("figure_file", "results/surface_diagnostics.png")
            image = figure_path.read_bytes()
            if len(image) < 1024 or not image.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("surface_diagnostics.png must be a non-trivial PNG")
        except Exception as exc:
            return _detail(self.name, weight, 0.0, False, {"error": str(exc)}, str(exc))
        return _detail(
            self.name,
            weight,
            1.0,
            True,
            {"query_rows": len(query_ids), "quote_rows": len(quote_ids)},
            "",
        )


@register_scorer("option_surface_query_accuracy")
class OptionSurfaceQueryAccuracyScorer(Scorer):
    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        weight = float(config.get("weight", 1.0))
        try:
            ids, pred, ref = _aligned_values(
                pred_dir / config.get("pred_file", "results/query_surface.csv"),
                ref_dir / config.get("ref_file", "query_surface_ref.csv"),
                QUERY_COLUMNS,
                "query_id",
            )
            input_ids, query_input = _load_query_input(
                pred_dir / config.get("query_file", "data/query_grid.csv")
            )
            query_input = _align_input(ids, input_ids, query_input)
            unit = query_input["discount_factor"] * query_input["forward"]
            price_scale = np.maximum(0.010 * unit + 0.05 * np.abs(ref["call_price"]), 1.0e-8)
            price_error = float(np.sqrt(np.mean(np.square((pred["call_price"] - ref["call_price"]) / price_scale))))
            variance_error = float(
                np.sqrt(
                    np.mean(
                        np.square(
                            (pred["total_variance"] - ref["total_variance"])
                            / np.maximum(ref["total_variance"], 1.0e-5)
                        )
                    )
                )
            )
            iv_error = float(
                np.sqrt(
                    np.mean(
                        np.square(
                            (pred["implied_volatility"] - ref["implied_volatility"])
                            / np.maximum(ref["implied_volatility"], 1.0e-5)
                        )
                    )
                )
            )
        except Exception as exc:
            return _detail(self.name, weight, 0.0, False, {"error": str(exc)}, str(exc))

        parts = {
            "price": _lower_better(price_error, float(config["full_price_error"]), float(config["zero_price_error"])),
            "total_variance": _lower_better(variance_error, float(config["full_variance_error"]), float(config["zero_variance_error"])),
            "implied_volatility": _lower_better(iv_error, float(config["full_iv_error"]), float(config["zero_iv_error"])),
        }
        fraction = 0.50 * parts["price"] + 0.25 * parts["total_variance"] + 0.25 * parts["implied_volatility"]
        message = f"price_nrmse={price_error:.4g}, variance_rel_rmse={variance_error:.4g}, iv_rel_rmse={iv_error:.4g}"
        return _detail(
            self.name,
            weight,
            fraction,
            fraction > 0.0,
            {"price_error": price_error, "variance_error": variance_error, "iv_error": iv_error, "component_fractions": parts},
            message,
        )


@register_scorer("option_surface_distribution_accuracy")
class OptionSurfaceDistributionAccuracyScorer(Scorer):
    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        weight = float(config.get("weight", 1.0))
        try:
            ids, pred, ref = _aligned_values(
                pred_dir / config.get("pred_file", "results/query_surface.csv"),
                ref_dir / config.get("ref_file", "query_surface_ref.csv"),
                QUERY_COLUMNS,
                "query_id",
            )
            input_ids, query_input = _load_query_input(
                pred_dir / config.get("query_file", "data/query_grid.csv")
            )
            query_input = _align_input(ids, input_ids, query_input)
            l1_errors: list[float] = []
            mass_errors: list[float] = []
            digital_errors: list[float] = []
            tail_digital_errors: list[float] = []
            for maturity in sorted(set(query_input["maturity"].tolist())):
                mask = np.isclose(query_input["maturity"], maturity, rtol=0.0, atol=1.0e-10)
                order = np.argsort(query_input["strike"][mask])
                strike = query_input["strike"][mask][order]
                p = pred["risk_neutral_density"][mask][order]
                q = ref["risk_neutral_density"][mask][order]
                ref_mass = max(_trapezoid(np.abs(q), strike), 1.0e-12)
                l1_errors.append(_trapezoid(np.abs(p - q), strike) / ref_mass)
                mass_errors.append(abs(_trapezoid(p, strike) - _trapezoid(q, strike)) / ref_mass)
                digital_diff = (
                    pred["digital_survival_probability"][mask][order]
                    - ref["digital_survival_probability"][mask][order]
                )
                digital_errors.append(float(np.sqrt(np.mean(np.square(digital_diff)))))
                log_moneyness = query_input["log_moneyness"][mask][order]
                tail_mask = np.abs(log_moneyness) >= 0.65
                tail_digital_errors.append(
                    float(np.sqrt(np.mean(np.square(digital_diff[tail_mask]))))
                )
            l1_error = float(np.mean(l1_errors))
            mass_error = float(np.mean(mass_errors))
            digital_error = float(np.mean(digital_errors))
            tail_digital_error = float(np.mean(tail_digital_errors))
        except Exception as exc:
            return _detail(self.name, weight, 0.0, False, {"error": str(exc)}, str(exc))
        l1_fraction = _lower_better(l1_error, float(config["full_l1_error"]), float(config["zero_l1_error"]))
        mass_fraction = _lower_better(mass_error, float(config["full_mass_error"]), float(config["zero_mass_error"]))
        digital_fraction = _lower_better(
            digital_error,
            float(config["full_digital_error"]),
            float(config["zero_digital_error"]),
        )
        tail_digital_fraction = _lower_better(
            tail_digital_error,
            float(config["full_tail_digital_error"]),
            float(config["zero_tail_digital_error"]),
        )
        fraction = (
            0.65 * l1_fraction
            + 0.05 * mass_fraction
            + 0.25 * digital_fraction
            + 0.05 * tail_digital_fraction
        )
        return _detail(
            self.name,
            weight,
            fraction,
            fraction > 0.0,
            {
                "mean_relative_l1": l1_error,
                "mean_mass_error": mass_error,
                "digital_rmse": digital_error,
                "tail_digital_rmse": tail_digital_error,
            },
            (
                f"density_relative_L1={l1_error:.4g}, density_mass_error={mass_error:.4g}, "
                f"digital_rmse={digital_error:.4g}, tail_digital_rmse={tail_digital_error:.4g}"
            ),
        )


@register_scorer("option_surface_quote_fit")
class OptionSurfaceQuoteFitScorer(Scorer):
    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        weight = float(config.get("weight", 1.0))
        try:
            ids, pred, ref = _aligned_values(
                pred_dir / config.get("pred_file", "results/quote_diagnostics.csv"),
                ref_dir / config.get("ref_file", "quote_diagnostics_ref.csv"),
                QUOTE_COLUMNS,
                "quote_id",
            )
            quote_ids, quotes = _load_quote_input(
                pred_dir / config.get("quotes_file", "data/quotes.csv")
            )
            quotes = _align_input(ids, quote_ids, quotes)
            clean = ref["outlier_score"] < 0.5
            if int(clean.sum()) == 0:
                raise ValueError("reference contains no clean quotes")
            half = np.maximum(0.5 * (quotes["ask"] - quotes["bid"]), 1.0e-12)
            fit = pred["fitted_call_price"]
            interval_violation = np.maximum.reduce((quotes["bid"] - fit, fit - quotes["ask"], np.zeros_like(fit))) / half
            interval_error = float(np.sqrt(np.mean(np.square(interval_violation[clean]))))
            true_price_error = float(
                np.sqrt(
                    np.mean(
                        np.square(
                            (fit[clean] - ref["fitted_call_price"][clean]) / half[clean]
                        )
                    )
                )
            )
        except Exception as exc:
            return _detail(self.name, weight, 0.0, False, {"error": str(exc)}, str(exc))
        interval_fraction = _lower_better(interval_error, float(config["full_interval_error"]), float(config["zero_interval_error"]))
        true_fraction = _lower_better(true_price_error, float(config["full_true_price_error"]), float(config["zero_true_price_error"]))
        fraction = 0.60 * interval_fraction + 0.40 * true_fraction
        return _detail(
            self.name,
            weight,
            fraction,
            fraction > 0.0,
            {"clean_quote_count": int(clean.sum()), "interval_rmse": interval_error, "true_price_spread_rmse": true_price_error},
            f"clean_interval_rmse={interval_error:.4g}, clean_true_price_spread_rmse={true_price_error:.4g}",
        )


@register_scorer("option_surface_outlier_detection")
class OptionSurfaceOutlierDetectionScorer(Scorer):
    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        weight = float(config.get("weight", 1.0))
        try:
            _, pred, ref = _aligned_values(
                pred_dir / config.get("pred_file", "results/quote_diagnostics.csv"),
                ref_dir / config.get("ref_file", "quote_diagnostics_ref.csv"),
                QUOTE_COLUMNS,
                "quote_id",
            )
            labels = ref["outlier_score"]
            auc = _auc(pred["outlier_score"], labels)
            average_precision = _average_precision(pred["outlier_score"], labels)
        except Exception as exc:
            return _detail(self.name, weight, 0.0, False, {"error": str(exc)}, str(exc))
        auc_fraction = _higher_better(auc, float(config["zero_auc"]), float(config["full_auc"]))
        ap_fraction = _higher_better(average_precision, float(config["zero_average_precision"]), float(config["full_average_precision"]))
        fraction = 0.70 * auc_fraction + 0.30 * ap_fraction
        return _detail(
            self.name,
            weight,
            fraction,
            fraction > 0.0,
            {"auc": auc, "average_precision": average_precision, "prevalence": float(np.mean(labels >= 0.5))},
            f"outlier_AUC={auc:.4f}, average_precision={average_precision:.4f}",
        )


@register_scorer("option_surface_arbitrage")
class OptionSurfaceArbitrageScorer(Scorer):
    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        weight = float(config.get("weight", 1.0))
        try:
            ids, pred, _ = _aligned_values(
                pred_dir / config.get("pred_file", "results/query_surface.csv"),
                ref_dir / config.get("ref_file", "query_surface_ref.csv"),
                QUERY_COLUMNS,
                "query_id",
            )
            input_ids, query_input = _load_query_input(
                pred_dir / config.get("query_file", "data/query_grid.csv")
            )
            query_input = _align_input(ids, input_ids, query_input)

            lower = query_input["discount_factor"] * np.maximum(
                query_input["forward"] - query_input["strike"], 0.0
            )
            upper = query_input["discount_factor"] * query_input["forward"]
            unit = np.maximum(upper, 1.0e-12)
            bound_error = float(
                np.max(
                    np.maximum.reduce(
                        (
                            (lower - pred["call_price"]) / unit,
                            (pred["call_price"] - upper) / unit,
                            np.zeros_like(unit),
                        )
                    )
                )
            )

            monotonicity_error = 0.0
            convexity_error = 0.0
            digital_monotonicity_error = 0.0
            digital_consistency_errors: list[float] = []
            negative_mass_errors: list[float] = []
            maturity_values = sorted(set(query_input["maturity"].tolist()))
            for maturity in maturity_values:
                mask = np.isclose(query_input["maturity"], maturity, rtol=0.0, atol=1.0e-10)
                order = np.argsort(query_input["strike"][mask])
                strike = query_input["strike"][mask][order]
                call = pred["call_price"][mask][order]
                discount = query_input["discount_factor"][mask][order]
                group_unit = float(np.median(unit[mask][order]))
                monotonicity_error = max(
                    monotonicity_error,
                    float(np.max(np.maximum(np.diff(call), 0.0))) / group_unit,
                )
                slopes = np.diff(call) / np.diff(strike)
                convexity_error = max(
                    convexity_error,
                    float(np.max(np.maximum(-np.diff(slopes), 0.0)))
                    / max(float(np.median(discount)), 1.0e-12),
                )
                density = pred["risk_neutral_density"][mask][order]
                abs_mass = max(_trapezoid(np.abs(density), strike), 1.0e-12)
                negative_mass = _trapezoid(np.maximum(-density, 0.0), strike)
                negative_mass_errors.append(negative_mass / abs_mass)
                digital = pred["digital_survival_probability"][mask][order]
                digital_monotonicity_error = max(
                    digital_monotonicity_error,
                    float(np.max(np.maximum(np.diff(digital), 0.0))),
                )
                digital_from_call = -slopes / max(
                    float(np.median(discount)), 1.0e-12
                )
                digital_midpoint = 0.5 * (digital[:-1] + digital[1:])
                digital_consistency_errors.append(
                    float(
                        np.sqrt(
                            np.mean(np.square(digital_from_call - digital_midpoint))
                        )
                    )
                )

            # Query grid is rectangular in maturity and log-moneyness.
            calendar_error = 0.0
            for k in sorted(set(query_input["log_moneyness"].tolist())):
                mask = np.isclose(query_input["log_moneyness"], k, rtol=0.0, atol=1.0e-10)
                order = np.argsort(query_input["maturity"][mask])
                variance = pred["total_variance"][mask][order]
                relative_drop = np.maximum(-np.diff(variance), 0.0) / np.maximum(variance[:-1], 1.0e-8)
                if relative_drop.size:
                    calendar_error = max(calendar_error, float(np.max(relative_drop)))

            negative_density_mass = float(np.mean(negative_mass_errors))
            w_from_iv = np.square(pred["implied_volatility"]) * query_input["maturity"]
            variance_consistency = float(
                np.max(
                    np.abs(pred["total_variance"] - w_from_iv)
                    / np.maximum(pred["total_variance"], 1.0e-8)
                )
            )
            call_from_w = _black_call(
                pred["total_variance"],
                query_input["log_moneyness"],
                query_input["forward"],
                query_input["discount_factor"],
            )
            price_consistency = float(
                np.max(np.abs(pred["call_price"] - call_from_w) / unit)
            )
            consistency_error = max(variance_consistency, price_consistency)
            digital_consistency_error = float(np.mean(digital_consistency_errors))
        except Exception as exc:
            return _detail(self.name, weight, 0.0, False, {"error": str(exc)}, str(exc))

        errors = {
            "bound_error": bound_error,
            "monotonicity_error": monotonicity_error,
            "convexity_error": convexity_error,
            "calendar_error": calendar_error,
            "density_negative_mass": negative_density_mass,
            "consistency_error": consistency_error,
            "digital_monotonicity_error": digital_monotonicity_error,
            "digital_consistency_error": digital_consistency_error,
        }
        fractions = {
            "bounds": _lower_better(bound_error, float(config["full_bound_error"]), float(config["zero_bound_error"])),
            "monotonicity": _lower_better(monotonicity_error, float(config["full_monotonicity_error"]), float(config["zero_monotonicity_error"])),
            "convexity": _lower_better(convexity_error, float(config["full_convexity_error"]), float(config["zero_convexity_error"])),
            "calendar": _lower_better(calendar_error, float(config["full_calendar_error"]), float(config["zero_calendar_error"])),
            "density": _lower_better(negative_density_mass, float(config["full_density_negative_mass"]), float(config["zero_density_negative_mass"])),
            "consistency": _lower_better(consistency_error, float(config["full_consistency_error"]), float(config["zero_consistency_error"])),
            "digital_monotonicity": _lower_better(
                digital_monotonicity_error,
                float(config["full_digital_monotonicity_error"]),
                float(config["zero_digital_monotonicity_error"]),
            ),
            "digital_consistency": _lower_better(
                digital_consistency_error,
                float(config["full_digital_consistency_error"]),
                float(config["zero_digital_consistency_error"]),
            ),
        }
        fraction = (
            0.10 * fractions["bounds"]
            + 0.10 * fractions["monotonicity"]
            + 0.15 * fractions["convexity"]
            + 0.15 * fractions["calendar"]
            + 0.10 * fractions["density"]
            + 0.15 * fractions["consistency"]
            + 0.10 * fractions["digital_monotonicity"]
            + 0.15 * fractions["digital_consistency"]
        )
        message = ", ".join(f"{key}={value:.3g}" for key, value in errors.items())
        return _detail(
            self.name,
            weight,
            fraction,
            fraction > 0.0,
            {**errors, "component_fractions": fractions},
            message,
        )


def _hidden_component_fraction(
    scorer: Scorer,
    prediction: Path,
    reference: Path,
    config: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    detail = scorer.score(prediction, reference, {**config, "weight": 1.0})
    fraction = float(detail.score / detail.max_score) if detail.max_score else 0.0
    return fraction, {
        "fraction": fraction,
        "passed": bool(detail.passed),
        "message": detail.message,
        "details": detail.details,
    }


@register_scorer("option_surface_hidden_replay_generalization")
class OptionSurfaceHiddenReplayGeneralizationScorer(Scorer):
    """Rerun the submitted program on pre-materialized evaluator inputs."""

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        weight = float(config.get("weight", 1.0))
        analysis = pred_dir / str(config.get("analysis_file", "analysis.py"))
        if not analysis.is_file():
            return _detail(
                self.name,
                weight,
                0.0,
                False,
                {"error": "analysis.py is missing"},
                "analysis.py is missing for hidden replay",
            )
        replay_ids = config.get("replay_ids", ["replay_1"])
        timeout = float(config.get("timeout_seconds", 2400.0))
        replay_root = ref_dir / "replay"
        if not replay_ids or any(
            not isinstance(name, str) or Path(name).name != name
            or name in {".", ".."}
            or not (replay_root / name / "data").is_dir()
            or not (replay_root / name / "reference").is_dir()
            for name in replay_ids
        ):
            return ScoreDetail(
                scorer_name=self.name, score=0.0, max_score=weight, passed=False,
                details={"scorer_internal_error": True,
                         "failure_kind": "missing_evaluator_input"},
                message="Missing or invalid pre-materialized replay bundle",
            )

        replay_details: list[dict[str, Any]] = []
        raw_scores: list[float] = []
        with tempfile.TemporaryDirectory(prefix="option_surface_hidden_replay_") as root:
            temporary_root = Path(root)
            for replay_id in replay_ids:
                workspace = temporary_root / f"{replay_id}"
                reference = temporary_root / f"reference_{replay_id}"
                try:
                    workspace.mkdir()
                    bundle = replay_root / replay_id
                    shutil.copytree(bundle / "data", workspace / "data")
                    shutil.copytree(bundle / "reference", reference)
                    instance_meta = workspace / "instance_meta.json"
                    if instance_meta.exists():
                        instance_meta.unlink()
                    shutil.copy2(analysis, workspace / "analysis.py")
                    completed = subprocess.run(
                        [sys.executable, "analysis.py"],
                        cwd=workspace,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=timeout,
                    )
                    if completed.returncode != 0:
                        replay_details.append(
                            {
                                "replay_id": replay_id,
                                "raw_score": 0.0,
                                "error": f"analysis.py exited with {completed.returncode}",
                                "stderr_tail": completed.stderr[-1200:],
                            }
                        )
                        raw_scores.append(0.0)
                        continue

                    contract, contract_detail = _hidden_component_fraction(
                        OptionSurfaceContractScorer(),
                        workspace,
                        reference,
                        {
                            "pred_file": "results/query_surface.csv",
                            "ref_file": "query_surface_ref.csv",
                            "quote_file": "results/quote_diagnostics.csv",
                            "reference_quote_file": "quote_diagnostics_ref.csv",
                            "audit_file": "results/arbitrage_audit.json",
                            "figure_file": "results/surface_diagnostics.png",
                        },
                    )
                    if contract < 1.0:
                        replay_details.append(
                            {
                                "replay_id": replay_id,
                                "raw_score": 0.0,
                                "contract": contract_detail,
                            }
                        )
                        raw_scores.append(0.0)
                        continue

                    query, query_detail = _hidden_component_fraction(
                        OptionSurfaceQueryAccuracyScorer(),
                        workspace,
                        reference,
                        {
                            "pred_file": "results/query_surface.csv",
                            "ref_file": "query_surface_ref.csv",
                            "query_file": "data/query_grid.csv",
                            "full_price_error": 0.004,
                            "zero_price_error": 0.075,
                            "full_variance_error": 0.006,
                            "zero_variance_error": 0.12,
                            "full_iv_error": 0.003,
                            "zero_iv_error": 0.060,
                        },
                    )
                    distribution, distribution_detail = _hidden_component_fraction(
                        OptionSurfaceDistributionAccuracyScorer(),
                        workspace,
                        reference,
                        {
                            "pred_file": "results/query_surface.csv",
                            "ref_file": "query_surface_ref.csv",
                            "query_file": "data/query_grid.csv",
                            "full_l1_error": 0.007,
                            "zero_l1_error": 0.025,
                            "full_mass_error": 0.001,
                            "zero_mass_error": 0.020,
                            "full_digital_error": 0.0004,
                            "zero_digital_error": 0.0012,
                            "full_tail_digital_error": 0.00015,
                            "zero_tail_digital_error": 0.00035,
                        },
                    )
                    quote, quote_detail = _hidden_component_fraction(
                        OptionSurfaceQuoteFitScorer(),
                        workspace,
                        reference,
                        {
                            "pred_file": "results/quote_diagnostics.csv",
                            "ref_file": "quote_diagnostics_ref.csv",
                            "quotes_file": "data/quotes.csv",
                            "full_interval_error": 0.03,
                            "zero_interval_error": 1.75,
                            "full_true_price_error": 0.04,
                            "zero_true_price_error": 1.10,
                        },
                    )
                    outlier, outlier_detail = _hidden_component_fraction(
                        OptionSurfaceOutlierDetectionScorer(),
                        workspace,
                        reference,
                        {
                            "pred_file": "results/quote_diagnostics.csv",
                            "ref_file": "quote_diagnostics_ref.csv",
                            "full_auc": 0.97,
                            "zero_auc": 0.55,
                            "full_average_precision": 0.85,
                            "zero_average_precision": 0.12,
                        },
                    )
                    arbitrage, arbitrage_detail = _hidden_component_fraction(
                        OptionSurfaceArbitrageScorer(),
                        workspace,
                        reference,
                        {
                            "pred_file": "results/query_surface.csv",
                            "ref_file": "query_surface_ref.csv",
                            "query_file": "data/query_grid.csv",
                            "full_bound_error": 1.0e-08,
                            "zero_bound_error": 2.0e-03,
                            "full_monotonicity_error": 1.0e-08,
                            "zero_monotonicity_error": 2.0e-03,
                            "full_convexity_error": 2.0e-06,
                            "zero_convexity_error": 0.08,
                            "full_calendar_error": 1.0e-08,
                            "zero_calendar_error": 0.04,
                            "full_density_negative_mass": 1.0e-08,
                            "zero_density_negative_mass": 0.08,
                            "full_consistency_error": 2.0e-05,
                            "zero_consistency_error": 0.08,
                            "full_digital_monotonicity_error": 1.0e-08,
                            "zero_digital_monotonicity_error": 0.03,
                            "full_digital_consistency_error": 1.5e-03,
                            "zero_digital_consistency_error": 0.08,
                        },
                    )
                    raw = float(
                        0.30 * query
                        + 0.55 * distribution
                        + 0.05 * quote
                        + 0.03 * outlier
                        + 0.05 * arbitrage
                        + 0.02
                    )
                    raw_scores.append(raw)
                    replay_details.append(
                        {
                            "replay_id": replay_id,
                            "raw_score": raw,
                            "query": query_detail,
                            "distribution": distribution_detail,
                            "quote": quote_detail,
                            "outlier": outlier_detail,
                            "arbitrage": arbitrage_detail,
                        }
                    )
                except subprocess.TimeoutExpired as exc:
                    replay_details.append(
                        {
                            "replay_id": replay_id,
                            "raw_score": 0.0,
                            "error": f"analysis.py timed out after {timeout:g}s",
                            "stdout_tail": (exc.stdout or "")[-1200:]
                            if isinstance(exc.stdout, str)
                            else "",
                            "stderr_tail": (exc.stderr or "")[-1200:]
                            if isinstance(exc.stderr, str)
                            else "",
                        }
                    )
                    raw_scores.append(0.0)
                except Exception as exc:
                    replay_details.append(
                        {"replay_id": replay_id, "raw_score": 0.0, "error": str(exc)}
                    )
                    raw_scores.append(0.0)

        raw_score = float(np.mean(raw_scores)) if raw_scores else 0.0
        fraction = _higher_better(
            raw_score,
            float(config.get("zero_replay_score", 0.80)),
            float(config.get("full_replay_score", 0.95)),
        )
        return _detail(
            self.name,
            weight,
            fraction,
            fraction > 0.0,
            {"raw_replay_score": raw_score, "hidden_instances": replay_details},
            f"hidden_replay_raw_score={raw_score:.4f}",
        )
