"""Forecast-only scoring contract for snowmelt runoff extrapolation."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from ai4sci_bench.core.scorer import Scorer, register_scorer

def _fraction(error: float, full: float, zero: float) -> float:
    return float(np.clip((zero-error)/(zero-full),0.0,1.0))

@register_scorer("hysteretic_snowmelt_runoff_forecast_v2")
class HystereticSnowmeltRunoffForecastScorer(Scorer):
    """Evaluate public daily-runoff forecasts, not a hidden catchment implementation."""
    def score(self,pred_dir:Path,ref_dir:Path,config:dict):
        from ai4sci_bench.core.types import ScoreDetail
        weight=float(config.get("weight",100.0))
        try:
            predicted=pd.read_csv(pred_dir/"results/forecast.csv");reference=pd.read_csv(ref_dir/"results/forecast.csv")
            if list(predicted.columns)!=["day","runoff_mm_per_day"]: raise ValueError("forecast.csv must have day,runoff_mm_per_day columns in that order")
            if len(predicted)!=len(reference): raise ValueError("forecast row count does not match forecast_forcings.csv")
            if not np.array_equal(predicted["day"].to_numpy(),reference["day"].to_numpy()): raise ValueError("forecast day values do not match forecast_forcings.csv")
            values=predicted["runoff_mm_per_day"].to_numpy(dtype=np.float64);truth=reference["runoff_mm_per_day"].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all() or (values<0).any(): raise ValueError("forecast runoff must be finite and non-negative")
            rmse=float(np.sqrt(np.mean((values-truth)**2)))
            fraction=_fraction(rmse,0.105,1.65)
            return ScoreDetail(scorer_name="hysteretic_snowmelt_runoff_forecast_v2",score=weight*fraction,max_score=weight,passed=fraction>=0.5,details={"runoff_rmse_mm_per_day":rmse,"runoff_fraction":fraction},message=f"runoff RMSE={rmse:.3f} mm/day")
        except Exception as exc:
            return ScoreDetail(scorer_name="hysteretic_snowmelt_runoff_forecast_v2",score=0.0,max_score=weight,passed=False,details={"error":str(exc)},message=f"invalid task output: {exc}")
