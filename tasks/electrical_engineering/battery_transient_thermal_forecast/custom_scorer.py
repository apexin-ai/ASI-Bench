"""Forecast-only scoring contract for battery transient extrapolation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ai4sci_bench.core.scorer import Scorer, register_scorer
from ai4sci_bench.core.types import ScoreDetail


def _fraction(error: float, full: float, zero: float) -> float:
    return float(np.clip((zero - error) / (zero - full), 0.0, 1.0))


@register_scorer("battery_transient_thermal_forecast_v3")
class BatteryTransientThermalForecastScorer(Scorer):
    """Score the public task objective, not an undisclosed model parameterization."""

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        weight = float(config.get("weight", 100.0))
        try:
            predicted_forecast = pd.read_csv(pred_dir / "results/forecast.csv")
            reference_forecast = pd.read_csv(ref_dir / "results/forecast.csv")
            if list(predicted_forecast.columns) != ["time_s", "voltage_v", "temperature_c"]:
                raise ValueError("forecast.csv must have time_s,voltage_v,temperature_c columns in that order")
            if len(predicted_forecast) != len(reference_forecast):
                raise ValueError("forecast row count does not match the requested current profile")
            if not np.allclose(predicted_forecast["time_s"], reference_forecast["time_s"], atol=1e-9, rtol=0.0):
                raise ValueError("forecast timestamps do not match the requested profile")
            prediction_values = predicted_forecast[["voltage_v", "temperature_c"]].to_numpy(dtype=np.float64)
            reference_values = reference_forecast[["voltage_v", "temperature_c"]].to_numpy(dtype=np.float64)
            if not np.isfinite(prediction_values).all():
                raise ValueError("forecast contains non-finite values")
            voltage_rmse_mv = float(np.sqrt(np.mean((prediction_values[:, 0] - reference_values[:, 0]) ** 2)) * 1000.0)
            temperature_rmse_c = float(np.sqrt(np.mean((prediction_values[:, 1] - reference_values[:, 1]) ** 2)))
            # Full credit is at a near-noise-floor forecast. A forecast at or beyond
            # 10 mV / 0.15 C receives no component credit; these tolerances are public.
            voltage_score = _fraction(voltage_rmse_mv, 2.5, 10.0)
            temperature_score = _fraction(temperature_rmse_c, 0.030, 0.150)
            fraction = 0.70 * voltage_score + 0.30 * temperature_score
            score = weight * fraction
            return ScoreDetail(
                scorer_name="battery_transient_thermal_forecast_v3",
                score=score,
                max_score=weight,
                passed=fraction >= 0.5,
                details={
                    "voltage_rmse_mv": voltage_rmse_mv,
                    "temperature_rmse_c": temperature_rmse_c,
                    "voltage_fraction": voltage_score,
                    "temperature_fraction": temperature_score,
                },
                message=(
                    f"voltage RMSE={voltage_rmse_mv:.2f}mV, "
                    f"temperature RMSE={temperature_rmse_c:.3f}C"
                ),
            )
        except Exception as exc:
            return ScoreDetail(
                scorer_name="battery_transient_thermal_forecast_v3",
                score=0.0,
                max_score=weight,
                passed=False,
                details={"error": str(exc)},
                message=f"invalid task output: {exc}",
            )
