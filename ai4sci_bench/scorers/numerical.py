"""Numerical comparison scorer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ai4sci_bench.core.scorer import Scorer, register_scorer
from ai4sci_bench.core.types import ScoreDetail


@register_scorer("numerical")
class NumericalScorer(Scorer):
    """General-purpose numerical comparison scorer.

    Supported metrics:
      exact_match, absolute_error, relative_error, relative_l2,
      per_frame_relative_l2, mean_relative_l2, max_ratio,
      cosine_similarity, correlation, rmse, mape
    """

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        metric = config.get("metric", "relative_l2")
        scorer_name = f"numerical:{metric}"

        try:
            result = getattr(self, f"_metric_{metric}")(pred_dir, ref_dir, config)
        except AttributeError:
            return ScoreDetail(
                scorer_name=scorer_name,
                score=0.0,
                max_score=config.get("weight", 1.0),
                passed=False,
                details={"error": f"Unknown metric: {metric}"},
                message=f"Unknown metric: {metric}",
            )
        except Exception as e:
            return ScoreDetail(
                scorer_name=scorer_name,
                score=0.0,
                max_score=config.get("weight", 1.0),
                passed=False,
                details={"error": str(e)},
                message=f"Scoring error: {e}",
            )

        return result

    def _load_arrays(
        self, pred_dir: Path, ref_dir: Path, config: dict
    ) -> tuple[np.ndarray, np.ndarray]:
        pred_file = config.get("pred_file", config.get("file"))
        ref_file = config.get("ref_file", config.get("file"))
        column = config.get("column")

        if str(pred_file).endswith(".csv") or str(ref_file).endswith(".csv"):
            import pandas as pd
            if column is None:
                raise ValueError(
                    f"numerical scorer: `column` is required when pred_file/ref_file is a CSV "
                    f"(got pred_file={pred_file!r}, ref_file={ref_file!r})"
                )
            pred = pd.read_csv(pred_dir / pred_file)[column].to_numpy(dtype=np.float64)
            ref = pd.read_csv(ref_dir / ref_file)[column].to_numpy(dtype=np.float64)
        else:
            pred = np.load(pred_dir / pred_file)
            ref = np.load(ref_dir / ref_file)
            row = config.get("row")
            if row is not None:
                pred = pred[row : row + 1]
                ref = ref[row : row + 1]

        if pred.shape != ref.shape:
            raise ValueError(
                f"Shape mismatch between {pred_file} and {ref_file}: "
                f"predicted {pred.shape}, reference {ref.shape}"
            )
        return pred.astype(np.float64), ref.astype(np.float64)

    def _apply_scoring(self, error: float, config: dict, weight: float) -> tuple[float, bool, dict[str, Any]]:
        """Apply scoring strategy (threshold or linear interpolation)."""
        scoring = config.get("scoring")
        threshold = config.get("threshold")

        if not np.isfinite(error):
            return 0.0, False, {
                "scoring_mode": "non_finite_error",
                "raw_fraction": 0.0,
            }

        if scoring == "linear_interpolation":
            full_thresh = config.get("full_score_threshold", 0.0)
            zero_thresh = config.get("zero_score_threshold", 1.0)
            if error <= full_thresh:
                raw = 1.0
            elif error >= zero_thresh:
                raw = 0.0
            else:
                raw = 1.0 - (error - full_thresh) / (zero_thresh - full_thresh)
            return raw * weight, True, {
                "scoring_mode": "linear_interpolation",
                "raw_fraction": raw,
                "full_score_threshold": full_thresh,
                "zero_score_threshold": zero_thresh,
            }  # always "passes" for scoring layer
        elif threshold is not None:
            passed = error <= threshold
            return (weight if passed else 0.0), passed, {
                "scoring_mode": "threshold",
                "raw_fraction": 1.0 if passed else 0.0,
                "threshold": threshold,
            }
        else:
            # No threshold → pass and return full score scaled by (1 - error)
            raw = max(0.0, 1.0 - error)
            return raw * weight, True, {
                "scoring_mode": "proportional",
                "raw_fraction": raw,
            }

    def _build_score_message(
        self,
        metric_name: str,
        error: float,
        score: float,
        weight: float,
        scoring_meta: dict[str, Any],
    ) -> str:
        mode = scoring_meta.get("scoring_mode", "unknown")
        if mode == "non_finite_error":
            return f"{metric_name}=non-finite; score=0.00/{weight:.2f}"
        if mode == "linear_interpolation":
            return (
                f"{metric_name}={error:.6f}; score={score:.2f}/{weight:.2f}; "
                f"linear interpolation with full<={scoring_meta['full_score_threshold']:.6f}, "
                f"zero>={scoring_meta['zero_score_threshold']:.6f}"
            )
        if mode == "threshold":
            passed = error <= scoring_meta["threshold"]
            comparator = "<=" if passed else ">"
            return (
                f"{metric_name}={error:.6f} {comparator} threshold "
                f"{scoring_meta['threshold']:.6f}; score={score:.2f}/{weight:.2f}"
            )
        return f"{metric_name}={error:.6f}; score={score:.2f}/{weight:.2f}"

    def _metric_relative_l2(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        pred, ref = self._load_arrays(pred_dir, ref_dir, config)
        if config.get("nan_policy") == "omit":
            mask = np.isfinite(pred) & np.isfinite(ref)
            pred = pred[mask]
            ref = ref[mask]
        ref_norm = np.linalg.norm(ref)
        if ref_norm == 0:
            error = float(np.linalg.norm(pred))
        else:
            error = float(np.linalg.norm(pred - ref) / ref_norm)

        weight = config.get("weight", 1.0)
        sc, passed, scoring_meta = self._apply_scoring(error, config, weight)
        return ScoreDetail(
            scorer_name=f"numerical:relative_l2",
            score=sc,
            max_score=weight,
            passed=passed,
            details={
                "relative_l2": error,
                "pred_file": config.get("pred_file", config.get("file")),
                **scoring_meta,
            },
            message=self._build_score_message("relative_l2", error, sc, weight, scoring_meta),
        )

    def _metric_per_frame_relative_l2(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        pred, ref = self._load_arrays(pred_dir, ref_dir, config)
        weight = config.get("weight", 1.0)
        threshold = config.get("threshold", 0.30)
        per_item_score = config.get("per_item_score", weight / pred.shape[0])
        n_items = config.get("items", pred.shape[0])

        alignment_cfg = config.get("alignment", {})
        do_align = alignment_cfg.get("enabled", False)
        max_shift = alignment_cfg.get("max_shift", 0)
        periodic = alignment_cfg.get("periodic", False)

        frame_details = {}
        total_score = 0.0
        for i in range(n_items):
            p_frame = pred[i]
            r_frame = ref[i]

            best_err = float("inf")
            best_shift = (0, 0)

            if do_align and max_shift > 0:
                for dx in range(-max_shift, max_shift + 1):
                    for dy in range(-max_shift, max_shift + 1):
                        shifted = np.roll(np.roll(p_frame, dx, axis=0), dy, axis=1)
                        ref_norm = np.linalg.norm(r_frame)
                        err = float(np.linalg.norm(shifted - r_frame) / ref_norm) if ref_norm > 0 else float(np.linalg.norm(shifted))
                        if err < best_err:
                            best_err = err
                            best_shift = (dx, dy)
            else:
                ref_norm = np.linalg.norm(r_frame)
                best_err = float(np.linalg.norm(p_frame - r_frame) / ref_norm) if ref_norm > 0 else float(np.linalg.norm(p_frame))

            pts = per_item_score if best_err <= threshold else 0.0
            total_score += pts
            frame_details[f"frame_{i}"] = {
                "rel_err": round(best_err, 6),
                "shift": list(best_shift),
                "pts": pts,
            }

        return ScoreDetail(
            scorer_name="numerical:per_frame_relative_l2",
            score=total_score,
            max_score=weight,
            passed=True,
            details=frame_details,
            message=(
                f"per_frame_relative_l2: awarded {total_score:.2f}/{weight:.2f} "
                f"across {n_items} frame(s) with threshold <= {threshold:.6f}"
            ),
        )

    def _metric_mean_relative_l2(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        pred, ref = self._load_arrays(pred_dir, ref_dir, config)
        weight = config.get("weight", 1.0)
        n_frames = pred.shape[0]

        errors = []
        for i in range(n_frames):
            ref_norm = np.linalg.norm(ref[i])
            if ref_norm > 0:
                errors.append(float(np.linalg.norm(pred[i] - ref[i]) / ref_norm))
            else:
                errors.append(float(np.linalg.norm(pred[i])))

        mean_err = float(np.mean(errors))
        sc, passed, scoring_meta = self._apply_scoring(mean_err, config, weight)
        return ScoreDetail(
            scorer_name="numerical:mean_relative_l2",
            score=sc,
            max_score=weight,
            passed=passed,
            details={"mean_relative_l2": mean_err, "per_frame_errors": errors, **scoring_meta},
            message=self._build_score_message("mean_relative_l2", mean_err, sc, weight, scoring_meta),
        )

    def _metric_max_ratio(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        file_name = config.get("file")
        pred = np.load(pred_dir / file_name).astype(np.float64)
        threshold = config.get("threshold", 2.0)
        weight = config.get("weight", 1.0)

        if pred[0] == 0:
            ratio = float("inf") if np.max(np.abs(pred)) > 0 else 1.0
        else:
            ratio = float(np.max(np.abs(pred)) / np.abs(pred[0]))

        passed = ratio <= threshold
        return ScoreDetail(
            scorer_name="numerical:max_ratio",
            score=weight if passed else 0.0,
            max_score=weight,
            passed=passed,
            details={"ke_ratio": round(ratio, 4)},
            message=(
                f"ke_ratio={ratio:.4f} {'<=' if passed else '>'} threshold {threshold:.4f}; "
                f"score={(weight if passed else 0.0):.2f}/{weight:.2f}"
            ),
        )

    def _metric_exact_match(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        pred, ref = self._load_arrays(pred_dir, ref_dir, config)
        weight = config.get("weight", 1.0)
        match = np.array_equal(pred, ref)
        return ScoreDetail(
            scorer_name="numerical:exact_match",
            score=weight if match else 0.0,
            max_score=weight,
            passed=match,
            details={"match": match},
        )

    def _metric_absolute_error(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        pred, ref = self._load_arrays(pred_dir, ref_dir, config)
        weight = config.get("weight", 1.0)
        tolerance = config.get("tolerance", 1e-6)
        error = float(np.max(np.abs(pred - ref)))
        passed = error <= tolerance
        return ScoreDetail(
            scorer_name="numerical:absolute_error",
            score=weight if passed else 0.0,
            max_score=weight,
            passed=passed,
            details={"max_absolute_error": error, "tolerance": tolerance},
        )

    def _metric_relative_error(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        pred, ref = self._load_arrays(pred_dir, ref_dir, config)
        weight = config.get("weight", 1.0)
        denom = np.abs(ref)
        denom = np.where(denom == 0, 1.0, denom)
        error = float(np.max(np.abs(pred - ref) / denom))
        if "scoring" in config or "threshold" in config:
            sc, passed, scoring_meta = self._apply_scoring(error, config, weight)
        else:
            tolerance = config.get("tolerance", 1e-4)
            passed = error <= tolerance
            sc = weight if passed else 0.0
            scoring_meta = {
                "scoring_mode": "threshold",
                "raw_fraction": 1.0 if passed else 0.0,
                "threshold": tolerance,
            }
        return ScoreDetail(
            scorer_name="numerical:relative_error",
            score=sc,
            max_score=weight,
            passed=passed,
            details={"max_relative_error": error, **scoring_meta},
            message=self._build_score_message("relative_error", error, sc, weight, scoring_meta),
        )

    def _metric_cosine_similarity(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        pred, ref = self._load_arrays(pred_dir, ref_dir, config)
        weight = config.get("weight", 1.0)
        threshold = config.get("threshold", 0.99)
        pred_flat = pred.flatten()
        ref_flat = ref.flatten()
        denom = np.linalg.norm(pred_flat) * np.linalg.norm(ref_flat)
        sim = float(np.dot(pred_flat, ref_flat) / denom) if denom > 0 else 0.0
        passed = sim >= threshold
        return ScoreDetail(
            scorer_name="numerical:cosine_similarity",
            score=weight if passed else 0.0,
            max_score=weight,
            passed=passed,
            details={"cosine_similarity": sim, "threshold": threshold},
        )

    def _metric_rmse(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        pred, ref = self._load_arrays(pred_dir, ref_dir, config)
        weight = config.get("weight", 1.0)
        rmse = float(np.sqrt(np.mean((pred - ref) ** 2)))
        sc, passed, scoring_meta = self._apply_scoring(rmse, config, weight)
        return ScoreDetail(
            scorer_name="numerical:rmse",
            score=sc,
            max_score=weight,
            passed=passed,
            details={"rmse": rmse, **scoring_meta},
            message=self._build_score_message("rmse", rmse, sc, weight, scoring_meta),
        )

    def _metric_correlation(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        pred, ref = self._load_arrays(pred_dir, ref_dir, config)
        weight = config.get("weight", 1.0)
        corr = float(np.corrcoef(pred.flatten(), ref.flatten())[0, 1])

        # Correlation is "higher is better". When the task config opts into
        # `scoring: linear_interpolation` with `full_score_threshold` / `zero_score_threshold`
        # (interpreted as correlation values, not error magnitudes), do partial credit
        # so that a near-but-not-perfect signal still earns points. Otherwise fall back
        # to the legacy boolean threshold (default 0.95).
        if config.get("scoring") == "linear_interpolation":
            full_corr = config.get("full_score_threshold", 0.95)
            zero_corr = config.get("zero_score_threshold", 0.50)
            if not np.isfinite(corr):
                raw = 0.0
            elif corr >= full_corr:
                raw = 1.0
            elif corr <= zero_corr:
                raw = 0.0
            else:
                raw = (corr - zero_corr) / (full_corr - zero_corr)
            sc = raw * weight
            return ScoreDetail(
                scorer_name="numerical:correlation",
                score=sc,
                max_score=weight,
                passed=True,
                details={
                    "correlation": corr,
                    "scoring_mode": "linear_interpolation",
                    "raw_fraction": raw,
                    "full_score_threshold": full_corr,
                    "zero_score_threshold": zero_corr,
                },
                message=f"correlation={corr:.4f}; score={sc:.2f}/{weight:.2f}; linear interpolation [{zero_corr}, {full_corr}]",
            )

        threshold = config.get("threshold", 0.95)
        passed = np.isfinite(corr) and corr >= threshold
        return ScoreDetail(
            scorer_name="numerical:correlation",
            score=weight if passed else 0.0,
            max_score=weight,
            passed=passed,
            details={"correlation": corr, "threshold": threshold},
        )

    def _metric_mape(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        pred, ref = self._load_arrays(pred_dir, ref_dir, config)
        weight = config.get("weight", 1.0)
        denom = np.abs(ref)
        mask = denom > 0
        if mask.sum() == 0:
            mape = 0.0
        else:
            mape = float(np.mean(np.abs(pred[mask] - ref[mask]) / denom[mask]))
        sc, passed, scoring_meta = self._apply_scoring(mape, config, weight)
        return ScoreDetail(
            scorer_name="numerical:mape",
            score=sc,
            max_score=weight,
            passed=passed,
            details={"mape": mape, **scoring_meta},
            message=self._build_score_message("mape", mape, sc, weight, scoring_meta),
        )
