"""Tests for scorer implementations."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from ai4sci_bench.core.scorer import get_scorer, list_scorers, register_scorer, Scorer
from ai4sci_bench.core.types import ScoreDetail

# Ensure scorers are registered
import ai4sci_bench.scorers  # noqa: F401


class TestScorerRegistry:
    def test_registered_scorers(self):
        names = list_scorers()
        assert "numerical" in names
        assert "file_match" in names
        assert "code_analysis" in names
        assert "composite" in names

    def test_get_scorer(self):
        scorer = get_scorer("numerical")
        assert scorer.name == "numerical"

    def test_get_unknown_scorer(self):
        with pytest.raises(KeyError, match="not found"):
            get_scorer("nonexistent_scorer")

    def test_register_custom_scorer(self):
        @register_scorer("test_custom")
        class TestScorer(Scorer):
            def score(self, pred_dir, ref_dir, config):
                return ScoreDetail(
                    scorer_name="test_custom",
                    score=1.0, max_score=1.0, passed=True, details={}
                )

        scorer = get_scorer("test_custom")
        assert scorer.name == "test_custom"


class TestNumericalScorer:
    def test_relative_l2_perfect(self, matching_pred_ref):
        pred_dir, ref_dir = matching_pred_ref
        scorer = get_scorer("numerical")
        result = scorer.score(pred_dir, ref_dir, {
            "metric": "relative_l2",
            "pred_file": "vorticity_frames.npy",
            "ref_file": "vorticity_frames_ref.npy",
            "threshold": 0.1,
            "weight": 50.0,
        })
        assert result.passed is True
        assert result.score == 50.0
        assert result.details["relative_l2"] == 0.0

    def test_relative_l2_different(self, tmp_dir):
        pred_dir = tmp_dir / "pred"
        ref_dir = tmp_dir / "ref"
        pred_dir.mkdir()
        ref_dir.mkdir()

        ref = np.ones(100, dtype=np.float32)
        pred = np.ones(100, dtype=np.float32) * 1.5
        np.save(pred_dir / "data.npy", pred)
        np.save(ref_dir / "data_ref.npy", ref)

        scorer = get_scorer("numerical")
        result = scorer.score(pred_dir, ref_dir, {
            "metric": "relative_l2",
            "pred_file": "data.npy",
            "ref_file": "data_ref.npy",
            "threshold": 0.1,
            "weight": 20.0,
        })
        assert result.passed is False
        assert result.score == 0.0
        assert result.details["relative_l2"] == pytest.approx(0.5, abs=1e-5)

    def test_relative_l2_linear_interpolation(self, tmp_dir):
        pred_dir = tmp_dir / "pred"
        ref_dir = tmp_dir / "ref"
        pred_dir.mkdir()
        ref_dir.mkdir()

        ref = np.ones(100, dtype=np.float32)
        pred = ref * 1.15  # 15% relative error
        np.save(pred_dir / "data.npy", pred)
        np.save(ref_dir / "data_ref.npy", ref)

        scorer = get_scorer("numerical")
        result = scorer.score(pred_dir, ref_dir, {
            "metric": "relative_l2",
            "pred_file": "data.npy",
            "ref_file": "data_ref.npy",
            "scoring": "linear_interpolation",
            "full_score_threshold": 0.10,
            "zero_score_threshold": 0.30,
            "weight": 30.0,
        })
        # error ≈ 0.15, between 0.10 and 0.30 → partial score
        assert 0 < result.score < 30.0

    def test_max_ratio_pass(self, tmp_dir):
        pred_dir = tmp_dir / "pred"
        pred_dir.mkdir()

        ke = np.array([1.0, 1.1, 1.2, 1.3, 1.1], dtype=np.float32)
        np.save(pred_dir / "kinetic_energy.npy", ke)

        scorer = get_scorer("numerical")
        result = scorer.score(pred_dir, tmp_dir, {
            "metric": "max_ratio",
            "file": "kinetic_energy.npy",
            "threshold": 2.0,
            "weight": 1.0,
        })
        assert result.passed is True
        assert result.details["ke_ratio"] == pytest.approx(1.3, abs=0.01)

    def test_max_ratio_fail(self, tmp_dir):
        pred_dir = tmp_dir / "pred"
        pred_dir.mkdir()

        ke = np.array([1.0, 5.0, 10.0], dtype=np.float32)
        np.save(pred_dir / "kinetic_energy.npy", ke)

        scorer = get_scorer("numerical")
        result = scorer.score(pred_dir, tmp_dir, {
            "metric": "max_ratio",
            "file": "kinetic_energy.npy",
            "threshold": 2.0,
            "weight": 1.0,
        })
        assert result.passed is False

    def test_per_frame_relative_l2(self, matching_pred_ref):
        pred_dir, ref_dir = matching_pred_ref
        scorer = get_scorer("numerical")
        result = scorer.score(pred_dir, ref_dir, {
            "metric": "per_frame_relative_l2",
            "pred_file": "vorticity_frames.npy",
            "ref_file": "vorticity_frames_ref.npy",
            "threshold": 0.30,
            "weight": 50.0,
            "items": 5,
            "per_item_score": 10.0,
        })
        assert result.score == 50.0  # All frames match perfectly

    def test_mean_relative_l2(self, matching_pred_ref):
        pred_dir, ref_dir = matching_pred_ref
        scorer = get_scorer("numerical")
        result = scorer.score(pred_dir, ref_dir, {
            "metric": "mean_relative_l2",
            "pred_file": "vorticity_frames.npy",
            "ref_file": "vorticity_frames_ref.npy",
            "scoring": "linear_interpolation",
            "full_score_threshold": 0.10,
            "zero_score_threshold": 0.30,
            "weight": 30.0,
        })
        assert result.score == 30.0  # Perfect match

    def test_exact_match(self, matching_pred_ref):
        pred_dir, ref_dir = matching_pred_ref
        scorer = get_scorer("numerical")
        result = scorer.score(pred_dir, ref_dir, {
            "metric": "exact_match",
            "pred_file": "vorticity_frames.npy",
            "ref_file": "vorticity_frames_ref.npy",
            "weight": 10.0,
        })
        assert result.passed is True

    def test_absolute_error(self, tmp_dir):
        pred_dir = tmp_dir / "pred"
        ref_dir = tmp_dir / "ref"
        pred_dir.mkdir()
        ref_dir.mkdir()

        ref = np.zeros(10, dtype=np.float32)
        pred = np.full(10, 0.001, dtype=np.float32)
        np.save(pred_dir / "d.npy", pred)
        np.save(ref_dir / "d.npy", ref)

        scorer = get_scorer("numerical")
        result = scorer.score(pred_dir, ref_dir, {
            "metric": "absolute_error",
            "pred_file": "d.npy",
            "ref_file": "d.npy",
            "tolerance": 0.01,
            "weight": 5.0,
        })
        assert result.passed is True

    def test_cosine_similarity(self, tmp_dir):
        pred_dir = tmp_dir / "pred"
        ref_dir = tmp_dir / "ref"
        pred_dir.mkdir()
        ref_dir.mkdir()

        ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        pred = np.array([1.0, 0.01, 0.0], dtype=np.float32)
        np.save(pred_dir / "v.npy", pred)
        np.save(ref_dir / "v.npy", ref)

        scorer = get_scorer("numerical")
        result = scorer.score(pred_dir, ref_dir, {
            "metric": "cosine_similarity",
            "pred_file": "v.npy",
            "ref_file": "v.npy",
            "threshold": 0.99,
            "weight": 5.0,
        })
        assert result.passed is True

    def test_rmse(self, matching_pred_ref):
        pred_dir, ref_dir = matching_pred_ref
        scorer = get_scorer("numerical")
        result = scorer.score(pred_dir, ref_dir, {
            "metric": "rmse",
            "pred_file": "kinetic_energy.npy",
            "ref_file": "kinetic_energy_ref.npy",
            "threshold": 0.01,
            "weight": 10.0,
        })
        assert result.score == 10.0  # Perfect match

    def test_correlation(self, tmp_dir):
        pred_dir = tmp_dir / "pred"
        ref_dir = tmp_dir / "ref"
        pred_dir.mkdir()
        ref_dir.mkdir()

        x = np.arange(100, dtype=np.float32)
        np.save(pred_dir / "c.npy", x * 2 + 1)  # Linear transform → perfect corr
        np.save(ref_dir / "c.npy", x)

        scorer = get_scorer("numerical")
        result = scorer.score(pred_dir, ref_dir, {
            "metric": "correlation",
            "pred_file": "c.npy",
            "ref_file": "c.npy",
            "threshold": 0.99,
            "weight": 5.0,
        })
        assert result.passed is True

    def test_unknown_metric(self, tmp_dir):
        scorer = get_scorer("numerical")
        result = scorer.score(tmp_dir, tmp_dir, {
            "metric": "nonexistent_metric",
            "weight": 1.0,
        })
        assert result.passed is False
        assert "Unknown metric" in result.message

    def test_missing_file_error(self, tmp_dir):
        scorer = get_scorer("numerical")
        result = scorer.score(tmp_dir, tmp_dir, {
            "metric": "relative_l2",
            "pred_file": "missing.npy",
            "ref_file": "missing.npy",
            "weight": 10.0,
        })
        assert result.passed is False
        assert result.score == 0.0

    def test_relative_l2_shape_mismatch_reports_clean_error(self, tmp_dir):
        pred_dir = tmp_dir / "pred"
        ref_dir = tmp_dir / "ref"
        pred_dir.mkdir()
        ref_dir.mkdir()

        np.save(pred_dir / "field.npy", np.zeros((96, 96, 2), dtype=np.float32))
        np.save(ref_dir / "field_ref.npy", np.zeros((2, 96, 96), dtype=np.float32))

        scorer = get_scorer("numerical")
        result = scorer.score(pred_dir, ref_dir, {
            "metric": "relative_l2",
            "pred_file": "field.npy",
            "ref_file": "field_ref.npy",
            "weight": 10.0,
        })

        assert result.passed is False
        assert result.score == 0.0
        assert "Shape mismatch between field.npy and field_ref.npy" in result.message
        assert "broadcast" not in result.message

    def test_mean_relative_l2_shape_mismatch_reports_clean_error(self, tmp_dir):
        pred_dir = tmp_dir / "pred"
        ref_dir = tmp_dir / "ref"
        pred_dir.mkdir()
        ref_dir.mkdir()

        np.save(pred_dir / "field.npy", np.zeros((96, 96, 2), dtype=np.float32))
        np.save(ref_dir / "field_ref.npy", np.zeros((2, 96, 96), dtype=np.float32))

        scorer = get_scorer("numerical")
        result = scorer.score(pred_dir, ref_dir, {
            "metric": "mean_relative_l2",
            "pred_file": "field.npy",
            "ref_file": "field_ref.npy",
            "weight": 10.0,
        })

        assert result.passed is False
        assert result.score == 0.0
        assert "Shape mismatch between field.npy and field_ref.npy" in result.message
        assert "broadcast" not in result.message

    def test_relative_l2_nan_prediction_returns_zero_not_nan(self, tmp_dir):
        pred_dir = tmp_dir / "pred"
        ref_dir = tmp_dir / "ref"
        pred_dir.mkdir()
        ref_dir.mkdir()

        np.save(pred_dir / "data.npy", np.array([np.nan, np.nan], dtype=np.float32))
        np.save(ref_dir / "data_ref.npy", np.array([1.0, 2.0], dtype=np.float32))

        scorer = get_scorer("numerical")
        result = scorer.score(pred_dir, ref_dir, {
            "metric": "relative_l2",
            "pred_file": "data.npy",
            "ref_file": "data_ref.npy",
            "scoring": "linear_interpolation",
            "full_score_threshold": 0.01,
            "zero_score_threshold": 0.10,
            "weight": 10.0,
        })

        assert result.passed is False
        assert result.score == 0.0
        assert result.details["scoring_mode"] == "non_finite_error"
        assert result.details["raw_fraction"] == 0.0
        assert "non-finite" in result.message

    def test_per_frame_with_alignment(self, tmp_dir):
        pred_dir = tmp_dir / "pred"
        ref_dir = tmp_dir / "ref"
        pred_dir.mkdir()
        ref_dir.mkdir()

        # Create data where pred is shifted by 1 pixel
        ref = np.zeros((3, 10, 10), dtype=np.float32)
        ref[:, 5, 5] = 1.0
        pred = np.zeros((3, 10, 10), dtype=np.float32)
        pred[:, 5, 6] = 1.0  # shifted by (0,1)

        np.save(pred_dir / "frames.npy", pred)
        np.save(ref_dir / "frames_ref.npy", ref)

        scorer = get_scorer("numerical")
        result = scorer.score(pred_dir, ref_dir, {
            "metric": "per_frame_relative_l2",
            "pred_file": "frames.npy",
            "ref_file": "frames_ref.npy",
            "threshold": 0.01,
            "weight": 30.0,
            "items": 3,
            "per_item_score": 10.0,
            "alignment": {"enabled": True, "max_shift": 2, "periodic": True},
        })
        # With alignment, the shift should be corrected
        assert result.score == 30.0


class TestFileMatchScorer:
    def test_file_exists(self, sample_pred_dir):
        scorer = get_scorer("file_match")
        result = scorer.score(sample_pred_dir, sample_pred_dir, {
            "checks": [
                {"file": "vorticity_frames.npy", "shape": (5, 10, 10), "dtype": "float32"},
                {"file": "kinetic_energy.npy"},
            ],
            "weight": 1.0,
        })
        assert result.passed is True

    def test_file_missing(self, tmp_dir):
        pred_dir = tmp_dir / "pred"
        pred_dir.mkdir()

        scorer = get_scorer("file_match")
        result = scorer.score(pred_dir, pred_dir, {
            "checks": [{"file": "missing.npy"}],
            "weight": 1.0,
        })
        assert result.passed is False

    def test_shape_mismatch(self, tmp_dir):
        pred_dir = tmp_dir / "pred"
        pred_dir.mkdir()
        np.save(pred_dir / "data.npy", np.zeros((3, 5)))

        scorer = get_scorer("file_match")
        result = scorer.score(pred_dir, pred_dir, {
            "checks": [{"file": "data.npy", "shape": (5, 5), "dtype": "float64"}],
            "weight": 1.0,
        })
        assert result.passed is False
        check = result.details["checks"][0]
        assert check["expected_shape"] == [5, 5]
        assert check["actual_shape"] == [3, 5]

    def test_no_nan_inf(self, sample_pred_dir):
        scorer = get_scorer("file_match")
        result = scorer.score(sample_pred_dir, sample_pred_dir, {
            "checks": [{"check": "no_nan_inf"}],
            "weight": 1.0,
        })
        assert result.passed is True

    def test_nan_detected(self, tmp_dir):
        pred_dir = tmp_dir / "pred"
        pred_dir.mkdir()
        arr = np.array([1.0, float("nan"), 3.0])
        np.save(pred_dir / "bad.npy", arr)

        scorer = get_scorer("file_match")
        result = scorer.score(pred_dir, pred_dir, {
            "checks": [{"check": "no_nan_inf"}],
            "weight": 1.0,
        })
        assert result.passed is False

    def test_dtype_mismatch(self, tmp_dir):
        pred_dir = tmp_dir / "pred"
        pred_dir.mkdir()
        np.save(pred_dir / "data.npy", np.zeros(10, dtype=np.float64))

        scorer = get_scorer("file_match")
        result = scorer.score(pred_dir, pred_dir, {
            "checks": [{"file": "data.npy", "dtype": "float32"}],
            "weight": 1.0,
        })
        assert result.passed is False
        check = result.details["checks"][0]
        assert check["expected_dtype"] == "float32"
        assert check["actual_dtype"] == "float64"

    def test_shape_as_int_scalar(self, tmp_dir):
        """After template resolution, a 1-D shape like '{n_frames+1}' becomes an int."""
        pred_dir = tmp_dir / "pred"
        pred_dir.mkdir()
        np.save(pred_dir / "data.npy", np.zeros((201,), dtype=np.float32))

        scorer = get_scorer("file_match")
        result = scorer.score(pred_dir, pred_dir, {
            "checks": [{"file": "data.npy", "shape": 201}],
            "weight": 1.0,
        })
        assert result.passed is True

    def test_shape_as_int_scalar_mismatch(self, tmp_dir):
        """Int shape that doesn't match should fail."""
        pred_dir = tmp_dir / "pred"
        pred_dir.mkdir()
        np.save(pred_dir / "data.npy", np.zeros((100,), dtype=np.float32))

        scorer = get_scorer("file_match")
        result = scorer.score(pred_dir, pred_dir, {
            "checks": [{"file": "data.npy", "shape": 201}],
            "weight": 1.0,
        })
        assert result.passed is False

    def test_shape_as_resolved_list(self, tmp_dir):
        """After template resolution, a multi-dim shape becomes a list of ints."""
        pred_dir = tmp_dir / "pred"
        pred_dir.mkdir()
        np.save(pred_dir / "data.npy", np.zeros((5, 10, 10), dtype=np.float32))

        scorer = get_scorer("file_match")
        result = scorer.score(pred_dir, pred_dir, {
            "checks": [{"file": "data.npy", "shape": [5, 10, 10]}],
            "weight": 1.0,
        })
        assert result.passed is True

    def test_shape_as_string_with_ints(self, tmp_dir):
        """String shape with numeric values (not parametric) should still parse."""
        pred_dir = tmp_dir / "pred"
        pred_dir.mkdir()
        np.save(pred_dir / "data.npy", np.zeros((5, 10, 10), dtype=np.float32))

        scorer = get_scorer("file_match")
        result = scorer.score(pred_dir, pred_dir, {
            "checks": [{"file": "data.npy", "shape": "{5,10,10}"}],
            "weight": 1.0,
        })
        assert result.passed is True

    def test_no_nan_inf_all_clean(self, tmp_dir):
        """no_nan_inf should pass when all .npy files are clean."""
        pred_dir = tmp_dir / "pred"
        pred_dir.mkdir()
        np.save(pred_dir / "a.npy", np.ones(10, dtype=np.float32))
        np.save(pred_dir / "b.npy", np.zeros(5, dtype=np.float64))

        scorer = get_scorer("file_match")
        result = scorer.score(pred_dir, pred_dir, {
            "checks": [{"check": "no_nan_inf"}],
            "weight": 1.0,
        })
        assert result.passed is True

    def test_value_range_check_passes(self, tmp_dir):
        """value_range should pass when all values are within bounds."""
        pred_dir = tmp_dir / "pred"
        pred_dir.mkdir()
        np.save(pred_dir / "data.npy", np.array([0.5, 1.0, 1.5], dtype=np.float32))

        scorer = get_scorer("file_match")
        result = scorer.score(pred_dir, pred_dir, {
            "checks": [{"check": "value_range", "file": "data.npy", "min": 0.0, "max": 2.0}],
            "weight": 1.0,
        })
        assert result.passed is True

    def test_value_range_below_min(self, tmp_dir):
        """value_range should fail when a value is below the minimum."""
        pred_dir = tmp_dir / "pred"
        pred_dir.mkdir()
        np.save(pred_dir / "data.npy", np.array([-1.0, 0.5], dtype=np.float32))

        scorer = get_scorer("file_match")
        result = scorer.score(pred_dir, pred_dir, {
            "checks": [{"check": "value_range", "file": "data.npy", "min": 0.0}],
            "weight": 1.0,
        })
        assert result.passed is False

    def test_value_range_above_max(self, tmp_dir):
        """value_range should fail when a value exceeds the maximum."""
        pred_dir = tmp_dir / "pred"
        pred_dir.mkdir()
        np.save(pred_dir / "data.npy", np.array([0.5, 9.9], dtype=np.float32))

        scorer = get_scorer("file_match")
        result = scorer.score(pred_dir, pred_dir, {
            "checks": [{"check": "value_range", "file": "data.npy", "max": 5.0}],
            "weight": 1.0,
        })
        assert result.passed is False

    def test_inf_detected(self, tmp_dir):
        pred_dir = tmp_dir / "pred"
        pred_dir.mkdir()
        np.save(pred_dir / "inf.npy", np.array([1.0, float("inf"), 3.0]))

        scorer = get_scorer("file_match")
        result = scorer.score(pred_dir, pred_dir, {
            "checks": [{"check": "no_nan_inf"}],
            "weight": 1.0,
        })
        assert result.passed is False

    def test_per_check_scoring_mode(self, tmp_dir):
        pred_dir = tmp_dir / "pred"
        pred_dir.mkdir()
        np.save(pred_dir / "good.npy", np.zeros((2,), dtype=np.float32))

        scorer = get_scorer("file_match")
        result = scorer.score(pred_dir, pred_dir, {
            "checks": [
                {"file": "good.npy", "shape": [2], "dtype": "float32"},
                {"file": "missing.npy"},
            ],
            "scoring_mode": "per_check",
            "weight": 10.0,
        })
        assert result.passed is False
        assert result.score == pytest.approx(5.0)
        assert result.details["scoring_mode"] == "per_check"

    def test_invalid_scoring_mode_raises_scoring_error(self, tmp_dir):
        pred_dir = tmp_dir / "pred"
        pred_dir.mkdir()
        np.save(pred_dir / "good.npy", np.zeros((2,), dtype=np.float32))

        scorer = get_scorer("file_match")
        with pytest.raises(ValueError, match="Unknown file_match scoring_mode"):
            scorer.score(pred_dir, pred_dir, {
                "checks": [{"file": "good.npy"}],
                "scoring_mode": "bogus",
                "weight": 1.0,
            })


class TestCodeAnalysisScorer:
    def test_required_pattern_found(self, sample_code_file):
        scorer = get_scorer("code_analysis")
        result = scorer.score(sample_code_file, sample_code_file, {
            "target_file": "simulation.py",
            "checks": [
                {"pattern": "import taichi", "required": True},
                {"pattern": "ti.init\\(", "required": True},
                {"pattern": "@ti.kernel", "min_count": 1},
            ],
            "weight": 1.0,
        })
        assert result.passed is True

    def test_required_pattern_missing(self, tmp_dir):
        pred_dir = tmp_dir / "pred"
        pred_dir.mkdir()
        (pred_dir / "simulation.py").write_text("import numpy as np\nprint('hello')")

        scorer = get_scorer("code_analysis")
        result = scorer.score(pred_dir, pred_dir, {
            "target_file": "simulation.py",
            "checks": [
                {"pattern": "import taichi", "required": True},
            ],
            "weight": 1.0,
        })
        assert result.passed is False
        assert "Required pattern not found" in result.message

    def test_required_output_pattern_satisfied_by_existing_file(self, tmp_dir):
        pred_dir = tmp_dir / "pred"
        (pred_dir / "results").mkdir(parents=True)
        (pred_dir / "analysis.py").write_text("import numpy as np\n")
        (pred_dir / "results" / "overview.png").write_bytes(b"png")

        scorer = get_scorer("code_analysis")
        result = scorer.score(pred_dir, pred_dir, {
            "target_file": "analysis.py",
            "checks": [
                {"pattern": "overview\\.png|savefig\\(|Image\\.save\\(", "required": True},
            ],
            "weight": 1.0,
        })

        assert result.passed is True
        assert result.details["checks"][0]["satisfied_by_file"] == "results/overview.png"

    def test_min_count(self, sample_code_file):
        scorer = get_scorer("code_analysis")
        result = scorer.score(sample_code_file, sample_code_file, {
            "target_file": "simulation.py",
            "checks": [
                {"pattern": "@ti.kernel", "min_count": 2},
            ],
            "weight": 1.0,
        })
        assert result.passed is True

    def test_file_not_found(self, tmp_dir):
        scorer = get_scorer("code_analysis")
        result = scorer.score(tmp_dir, tmp_dir, {
            "target_file": "missing.py",
            "checks": [],
            "weight": 1.0,
        })
        assert result.passed is False

    def test_forbidden_pattern(self, tmp_dir):
        pred_dir = tmp_dir / "pred"
        pred_dir.mkdir()
        (pred_dir / "code.py").write_text('answer = "hardcoded answer"\n')

        scorer = get_scorer("code_analysis")
        result = scorer.score(pred_dir, pred_dir, {
            "target_file": "code.py",
            "checks": [
                {"pattern": "hardcoded", "forbidden": True},
            ],
            "weight": 1.0,
        })
        assert result.passed is False

    def test_forbidden_imports_ignores_comments_and_docstrings(self, tmp_dir):
        pred_dir = tmp_dir / "pred"
        pred_dir.mkdir()
        (pred_dir / "code.py").write_text(
            '"""No hmmlearn / pomegranate / sklearn / Biopython are used."""\n'
            "# Also no tensorflow, torch, or jax here.\n"
            "import numpy as np\n"
            "value = np.array([1.0])\n",
            encoding="utf-8",
        )

        scorer = get_scorer("code_analysis")
        result = scorer.score(pred_dir, pred_dir, {
            "target_file": "code.py",
            "checks": [
                {
                    "forbidden_imports": [
                        "hmmlearn",
                        "pomegranate",
                        "sklearn",
                        "tensorflow",
                        "torch",
                        "jax",
                        "Bio.SeqIO",
                    ],
                },
            ],
            "weight": 1.0,
        })

        assert result.passed is True
        assert result.details["checks"][0]["matched_imports"] == []

    def test_forbidden_imports_detects_real_imports(self, tmp_dir):
        pred_dir = tmp_dir / "pred"
        pred_dir.mkdir()
        (pred_dir / "code.py").write_text(
            "import sklearn.metrics as metrics\n"
            "from Bio import SeqIO\n",
            encoding="utf-8",
        )

        scorer = get_scorer("code_analysis")
        result = scorer.score(pred_dir, pred_dir, {
            "target_file": "code.py",
            "checks": [
                {"forbidden_imports": ["sklearn", "Bio.SeqIO"]},
            ],
            "weight": 1.0,
        })

        check = result.details["checks"][0]
        assert result.passed is False
        assert "sklearn.metrics" in check["matched_imports"]
        assert "Bio.SeqIO" in check["matched_imports"]

    def test_per_check_scoring_mode(self, tmp_dir):
        pred_dir = tmp_dir / "pred"
        pred_dir.mkdir()
        (pred_dir / "code.py").write_text("import numpy as np\nvalue = 1\n")

        scorer = get_scorer("code_analysis")
        result = scorer.score(pred_dir, pred_dir, {
            "target_file": "code.py",
            "checks": [
                {"pattern": "import numpy", "required": True},
                {"pattern": "scipy", "required": True},
            ],
            "scoring_mode": "per_check",
            "weight": 12.0,
        })
        assert result.passed is False
        assert result.score == pytest.approx(6.0)
        assert result.details["scoring_mode"] == "per_check"

    def test_invalid_scoring_mode_raises(self, tmp_dir):
        pred_dir = tmp_dir / "pred"
        pred_dir.mkdir()
        (pred_dir / "code.py").write_text("print('ok')\n")

        scorer = get_scorer("code_analysis")
        with pytest.raises(ValueError, match="Unknown code_analysis scoring_mode"):
            scorer.score(pred_dir, pred_dir, {
                "target_file": "code.py",
                "checks": [],
                "scoring_mode": "bogus",
                "weight": 1.0,
            })


class TestCompositeScorer:
    def test_composite(self, matching_pred_ref):
        pred_dir, ref_dir = matching_pred_ref
        scorer = get_scorer("composite")
        result = scorer.score(pred_dir, ref_dir, {
            "weight": 100.0,
            "sub_scorers": [
                {
                    "scorer": "numerical",
                    "weight": 0.6,
                    "config": {
                        "metric": "relative_l2",
                        "pred_file": "vorticity_frames.npy",
                        "ref_file": "vorticity_frames_ref.npy",
                        "threshold": 0.1,
                    },
                },
                {
                    "scorer": "numerical",
                    "weight": 0.4,
                    "config": {
                        "metric": "relative_l2",
                        "pred_file": "kinetic_energy.npy",
                        "ref_file": "kinetic_energy_ref.npy",
                        "threshold": 0.1,
                    },
                },
            ],
        })
        assert result.score == 100.0
        assert result.passed is True

    def test_composite_no_sub_scorers(self, tmp_dir):
        scorer = get_scorer("composite")
        result = scorer.score(tmp_dir, tmp_dir, {"weight": 10.0, "sub_scorers": []})
        assert result.passed is False


class TestNumericalScorerDiagnostics:
    def test_linear_interpolation_details_and_message(self, tmp_dir):
        pred_dir = tmp_dir / "pred"
        ref_dir = tmp_dir / "ref"
        pred_dir.mkdir()
        ref_dir.mkdir()

        np.save(pred_dir / "energy.npy", np.array([1.0, 2.0], dtype=np.float32))
        np.save(ref_dir / "energy_ref.npy", np.array([1.0, 1.0], dtype=np.float32))

        scorer = get_scorer("numerical")
        result = scorer.score(pred_dir, ref_dir, {
            "metric": "relative_l2",
            "pred_file": "energy.npy",
            "ref_file": "energy_ref.npy",
            "weight": 20.0,
            "scoring": "linear_interpolation",
            "full_score_threshold": 0.05,
            "zero_score_threshold": 0.20,
        })

        assert result.score == 0.0
        assert result.details["scoring_mode"] == "linear_interpolation"
        assert result.details["full_score_threshold"] == 0.05
        assert result.details["zero_score_threshold"] == 0.20
        assert "linear interpolation" in result.message

    def test_per_frame_message_summarizes_award(self, tmp_dir):
        pred_dir = tmp_dir / "pred"
        ref_dir = tmp_dir / "ref"
        pred_dir.mkdir()
        ref_dir.mkdir()

        arr = np.zeros((2, 4, 4), dtype=np.float32)
        np.save(pred_dir / "frames.npy", arr)
        np.save(ref_dir / "frames_ref.npy", arr.copy())

        scorer = get_scorer("numerical")
        result = scorer.score(pred_dir, ref_dir, {
            "metric": "per_frame_relative_l2",
            "pred_file": "frames.npy",
            "ref_file": "frames_ref.npy",
            "weight": 20.0,
            "threshold": 0.30,
            "items": 2,
            "per_item_score": 10.0,
        })

        assert result.score == 20.0
        assert "awarded 20.00/20.00" in result.message


class TestNumericalScorerCSV:
    """Regression tests for CSV column-based scoring (issue: np.load choked on CSVs)."""

    def _write_csv(self, path: Path, data: dict) -> None:
        pd.DataFrame(data).to_csv(path, index=False)

    def test_relative_l2_csv_perfect_match(self, tmp_dir):
        pred_dir = tmp_dir / "pred"
        ref_dir = tmp_dir / "ref"
        pred_dir.mkdir()
        ref_dir.mkdir()

        vals = np.linspace(1.0, 10.0, 50)
        self._write_csv(pred_dir / "fluxes.csv", {"A_sun": vals, "other": vals * 2})
        self._write_csv(ref_dir / "fluxes_ref.csv", {"A_sun": vals, "other": vals * 2})

        scorer = get_scorer("numerical")
        result = scorer.score(pred_dir, ref_dir, {
            "metric": "relative_l2",
            "pred_file": "fluxes.csv",
            "ref_file": "fluxes_ref.csv",
            "column": "A_sun",
            "scoring": "linear_interpolation",
            "full_score_threshold": 0.05,
            "zero_score_threshold": 0.30,
            "weight": 25.0,
        })
        assert result.score == pytest.approx(25.0)
        assert result.details.get("relative_l2", 1.0) == pytest.approx(0.0, abs=1e-9)

    def test_relative_l2_csv_partial_score(self, tmp_dir):
        pred_dir = tmp_dir / "pred"
        ref_dir = tmp_dir / "ref"
        pred_dir.mkdir()
        ref_dir.mkdir()

        ref_vals = np.ones(50)
        pred_vals = ref_vals * 1.15  # ~15% error, between 5% and 30%
        self._write_csv(pred_dir / "fluxes.csv", {"A_sun": pred_vals})
        self._write_csv(ref_dir / "fluxes_ref.csv", {"A_sun": ref_vals})

        scorer = get_scorer("numerical")
        result = scorer.score(pred_dir, ref_dir, {
            "metric": "relative_l2",
            "pred_file": "fluxes.csv",
            "ref_file": "fluxes_ref.csv",
            "column": "A_sun",
            "scoring": "linear_interpolation",
            "full_score_threshold": 0.05,
            "zero_score_threshold": 0.30,
            "weight": 25.0,
        })
        assert 0 < result.score < 25.0

    def test_correlation_csv(self, tmp_dir):
        pred_dir = tmp_dir / "pred"
        ref_dir = tmp_dir / "ref"
        pred_dir.mkdir()
        ref_dir.mkdir()

        t = np.linspace(0, 2 * np.pi, 100)
        ref_vals = np.sin(t)
        pred_vals = np.sin(t) + np.random.default_rng(0).normal(0, 0.05, 100)
        self._write_csv(pred_dir / "fluxes.csv", {"A_sun": pred_vals})
        self._write_csv(ref_dir / "fluxes_ref.csv", {"A_sun": ref_vals})

        scorer = get_scorer("numerical")
        result = scorer.score(pred_dir, ref_dir, {
            "metric": "correlation",
            "pred_file": "fluxes.csv",
            "ref_file": "fluxes_ref.csv",
            "column": "A_sun",
            "scoring": "linear_interpolation",
            "full_score_threshold": 0.95,
            "zero_score_threshold": 0.50,
            "weight": 10.0,
        })
        assert result.score > 0, "high-correlation CSV pair should score above zero"

    def test_csv_missing_column_raises(self, tmp_dir):
        pred_dir = tmp_dir / "pred"
        ref_dir = tmp_dir / "ref"
        pred_dir.mkdir()
        ref_dir.mkdir()

        self._write_csv(pred_dir / "fluxes.csv", {"A_sun": [1.0, 2.0]})
        self._write_csv(ref_dir / "fluxes_ref.csv", {"A_sun": [1.0, 2.0]})

        scorer = get_scorer("numerical")
        # column: omitted -> scorer should return a Scoring error, not raise
        result = scorer.score(pred_dir, ref_dir, {
            "metric": "relative_l2",
            "pred_file": "fluxes.csv",
            "ref_file": "fluxes_ref.csv",
            "weight": 10.0,
        })
        assert result.score == 0.0
        assert "column" in result.message.lower() or "column" in str(result.details).lower()

    def test_csv_wrong_column_name_raises(self, tmp_dir):
        pred_dir = tmp_dir / "pred"
        ref_dir = tmp_dir / "ref"
        pred_dir.mkdir()
        ref_dir.mkdir()

        self._write_csv(pred_dir / "fluxes.csv", {"A_sun": [1.0, 2.0]})
        self._write_csv(ref_dir / "fluxes_ref.csv", {"A_sun": [1.0, 2.0]})

        scorer = get_scorer("numerical")
        result = scorer.score(pred_dir, ref_dir, {
            "metric": "relative_l2",
            "pred_file": "fluxes.csv",
            "ref_file": "fluxes_ref.csv",
            "column": "nonexistent_col",
            "weight": 10.0,
        })
        assert result.score == 0.0
        assert "Scoring error" in result.message


class TestCorrelationLinearInterpolation:
    """Regression tests: correlation metric should honor linear_interpolation scoring."""

    def test_correlation_full_credit_above_threshold(self, tmp_dir):
        pred_dir = tmp_dir / "pred"
        ref_dir = tmp_dir / "ref"
        pred_dir.mkdir()
        ref_dir.mkdir()

        t = np.linspace(0, 2 * np.pi, 200)
        ref = np.sin(t)
        pred = np.sin(t) + np.random.default_rng(0).normal(0, 0.001, 200)
        np.save(pred_dir / "x.npy", pred)
        np.save(ref_dir / "x_ref.npy", ref)

        scorer = get_scorer("numerical")
        result = scorer.score(pred_dir, ref_dir, {
            "metric": "correlation",
            "pred_file": "x.npy",
            "ref_file": "x_ref.npy",
            "scoring": "linear_interpolation",
            "full_score_threshold": 0.95,
            "zero_score_threshold": 0.50,
            "weight": 10.0,
        })
        assert result.score == pytest.approx(10.0, abs=0.05)
        assert result.details["scoring_mode"] == "linear_interpolation"

    def test_correlation_partial_credit_between_thresholds(self, tmp_dir):
        pred_dir = tmp_dir / "pred"
        ref_dir = tmp_dir / "ref"
        pred_dir.mkdir()
        ref_dir.mkdir()

        rng = np.random.default_rng(0)
        t = np.linspace(0, 2 * np.pi, 1000)
        ref = np.sin(t)
        pred = 0.7 * np.sin(t) + 0.3 * rng.normal(0, 1, 1000)
        np.save(pred_dir / "x.npy", pred)
        np.save(ref_dir / "x_ref.npy", ref)

        scorer = get_scorer("numerical")
        result = scorer.score(pred_dir, ref_dir, {
            "metric": "correlation",
            "pred_file": "x.npy",
            "ref_file": "x_ref.npy",
            "scoring": "linear_interpolation",
            "full_score_threshold": 0.95,
            "zero_score_threshold": 0.50,
            "weight": 10.0,
        })
        # corr ~0.85, partial credit between 0.50 and 0.95
        assert 0 < result.score < 10.0, f"expected partial credit, got {result.score}"

    def test_correlation_zero_below_threshold(self, tmp_dir):
        pred_dir = tmp_dir / "pred"
        ref_dir = tmp_dir / "ref"
        pred_dir.mkdir()
        ref_dir.mkdir()

        rng = np.random.default_rng(1)
        ref = rng.normal(0, 1, 500)
        pred = rng.normal(0, 1, 500)   # independent → corr ≈ 0
        np.save(pred_dir / "x.npy", pred)
        np.save(ref_dir / "x_ref.npy", ref)

        scorer = get_scorer("numerical")
        result = scorer.score(pred_dir, ref_dir, {
            "metric": "correlation",
            "pred_file": "x.npy",
            "ref_file": "x_ref.npy",
            "scoring": "linear_interpolation",
            "full_score_threshold": 0.95,
            "zero_score_threshold": 0.50,
            "weight": 10.0,
        })
        assert result.score == 0.0

    def test_correlation_legacy_threshold_still_works(self, tmp_dir):
        """Pre-existing tasks that use `threshold:` only should keep working."""
        pred_dir = tmp_dir / "pred"
        ref_dir = tmp_dir / "ref"
        pred_dir.mkdir()
        ref_dir.mkdir()

        t = np.linspace(0, 1, 100)
        ref = t * 2 + 1
        np.save(pred_dir / "x.npy", ref.copy())
        np.save(ref_dir / "x_ref.npy", ref)

        scorer = get_scorer("numerical")
        result = scorer.score(pred_dir, ref_dir, {
            "metric": "correlation",
            "pred_file": "x.npy",
            "ref_file": "x_ref.npy",
            "threshold": 0.95,
            "weight": 5.0,
        })
        assert result.passed is True
        assert result.score == 5.0
