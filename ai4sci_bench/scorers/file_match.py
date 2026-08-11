"""File match scorer — checks file existence, shape, dtype, and value properties."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ai4sci_bench.core.scorer import Scorer, register_scorer
from ai4sci_bench.core.types import ScoreDetail


@register_scorer("file_match")
class FileMatchScorer(Scorer):
    """Check output file existence, format, shape, and value properties."""

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        checks = config.get("checks", [])
        weight = config.get("weight", 1.0)
        scoring_mode = config.get("scoring_mode", "all_or_nothing")
        results = []
        all_passed = True

        for check in checks:
            if "check" in check:
                result = self._check_property(pred_dir, check)
            elif "file" in check:
                result = self._check_file(pred_dir, check)
            else:
                result = {"passed": False, "message": f"Unknown check: {check}"}

            results.append(result)
            if not result["passed"]:
                all_passed = False

        score = self._compute_score(results, weight, scoring_mode)
        return ScoreDetail(
            scorer_name="file_match",
            score=score,
            max_score=weight,
            passed=all_passed,
            details={"checks": results, "scoring_mode": scoring_mode},
            message=self._build_message(results, all_passed),
        )

    def _compute_score(self, results: list[dict], weight: float, scoring_mode: str) -> float:
        if scoring_mode == "all_or_nothing":
            return weight if all(result.get("passed", False) for result in results) else 0.0
        if scoring_mode == "per_check":
            if not results:
                return weight
            passed = sum(1 for result in results if result.get("passed", False))
            return weight * passed / len(results)
        raise ValueError(f"Unknown file_match scoring_mode: {scoring_mode}")

    def _build_message(self, results: list[dict], all_passed: bool) -> str:
        if all_passed:
            return ""
        failed = [result.get("message") for result in results if not result.get("passed") and result.get("message")]
        if failed:
            return "; ".join(failed)
        return "Some file checks failed"

    def _check_file(self, pred_dir: Path, check: dict) -> dict:
        file_name = check["file"]
        file_path = pred_dir / file_name

        if not file_path.exists():
            return {"passed": False, "file": file_name, "message": f"File not found: {file_name}"}

        result = {"passed": True, "file": file_name}

        if file_name.endswith(".npy"):
            try:
                arr = np.load(file_path)
            except Exception as e:
                return {"passed": False, "file": file_name, "message": f"Cannot load: {e}"}

            result["actual_shape"] = list(arr.shape)
            result["actual_dtype"] = str(arr.dtype)

            if "shape" in check:
                expected_shape = check["shape"]
                if isinstance(expected_shape, str):
                    # Parse parametric shape like "{5,grid_size,grid_size}"
                    expected_shape = [int(x.strip()) for x in expected_shape.strip("{}").split(",")]
                elif isinstance(expected_shape, int):
                    expected_shape = [expected_shape]
                result["expected_shape"] = list(expected_shape)
                expected_shape = tuple(expected_shape)
                if arr.shape != expected_shape:
                    result["passed"] = False
                    result["message"] = f"Shape mismatch: expected {expected_shape}, got {arr.shape}"
                    return result

            if "dtype" in check:
                expected_dtype = np.dtype(check["dtype"])
                result["expected_dtype"] = str(expected_dtype)
                if arr.dtype != expected_dtype:
                    result["passed"] = False
                    result["message"] = f"Dtype mismatch: expected {expected_dtype}, got {arr.dtype}"
                    return result

        return result

    def _check_property(self, pred_dir: Path, check: dict) -> dict:
        check_type = check["check"]

        if check_type == "no_nan_inf":
            return self._check_no_nan_inf(pred_dir)
        elif check_type == "value_range":
            return self._check_value_range(pred_dir, check)
        else:
            return {"passed": False, "message": f"Unknown check type: {check_type}"}

    def _check_no_nan_inf(self, pred_dir: Path) -> dict:
        for npy_file in pred_dir.glob("*.npy"):
            try:
                arr = np.load(npy_file)
                if np.any(np.isnan(arr)):
                    return {"passed": False, "check": "no_nan_inf", "message": f"NaN found in {npy_file.name}"}
                if np.any(np.isinf(arr)):
                    return {"passed": False, "check": "no_nan_inf", "message": f"Inf found in {npy_file.name}"}
            except Exception:
                pass
        return {"passed": True, "check": "no_nan_inf"}

    def _check_value_range(self, pred_dir: Path, check: dict) -> dict:
        file_name = check.get("file")
        if not file_name:
            return {"passed": False, "check": "value_range", "message": "No file specified"}

        file_path = pred_dir / file_name
        if not file_path.exists():
            return {"passed": False, "check": "value_range", "message": f"File not found: {file_name}"}

        arr = np.load(file_path)
        min_val = check.get("min")
        max_val = check.get("max")

        if min_val is not None and np.min(arr) < min_val:
            return {"passed": False, "check": "value_range",
                    "message": f"Values below minimum {min_val}: min={np.min(arr):.6f}"}
        if max_val is not None and np.max(arr) > max_val:
            return {"passed": False, "check": "value_range",
                    "message": f"Values above maximum {max_val}: max={np.max(arr):.6f}"}

        return {"passed": True, "check": "value_range"}
