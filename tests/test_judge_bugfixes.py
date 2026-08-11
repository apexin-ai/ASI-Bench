"""Tests exposing bugs found in code review of judge scorer implementation.

Each test is expected to FAIL before the fix and PASS after.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai4sci_bench.core.scorer import get_scorer
from ai4sci_bench.scorers._judge_common import aggregate_judge_scores

import ai4sci_bench.scorers  # noqa: F401


def _setup_dirs(tmp_path):
    pred_dir = tmp_path / "pred"
    ref_dir = tmp_path / "ref"
    pred_dir.mkdir()
    ref_dir.mkdir()
    (pred_dir / "analysis.py").write_text("print('hello')")
    (ref_dir / "reference.npy").write_text("ref data")
    return pred_dir, ref_dir


def _make_subprocess_result(stdout="", stderr="", returncode=0):
    r = MagicMock(spec=subprocess.CompletedProcess)
    r.stdout = stdout
    r.stderr = stderr
    r.returncode = returncode
    return r


def _base_config(**overrides):
    cfg = {
        "rubric": "Evaluate the solution. Score 0-10.",
        "timeout": 120,
        "weight": 10.0,
        "max_score_value": 10,
        "threshold": 0.5,
    }
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# Bug 1: Score source misattribution
# When score.json exists but is invalid and score comes from stdout fallback,
# the score_source incorrectly reports "score.json" instead of "stdout_parse".
# ---------------------------------------------------------------------------

class TestBug1ScoreSourceMisattribution:
    @patch("ai4sci_bench.scorers.agent_judge.subprocess.run")
    def test_invalid_score_json_reports_stdout_source(self, mock_run, tmp_path):
        """If score.json is invalid and score comes from stdout,
        score_source must be 'stdout_parse', not 'score.json'."""
        pred_dir, ref_dir = _setup_dirs(tmp_path)

        def side_effect(cmd, **kwargs):
            ws = Path(kwargs["cwd"])
            (ws / "score.json").write_text("INVALID JSON {{{")
            return _make_subprocess_result(
                stdout='My evaluation: {"score": 7, "reasoning": "good"}'
            )

        mock_run.side_effect = side_effect

        scorer = get_scorer("agent_judge")
        result = scorer.score(pred_dir, ref_dir, _base_config(num_judges=1))

        assert result.details["median_score"] == 7.0
        assert result.details["score_sources"] == ["stdout_parse"]

    @patch("ai4sci_bench.scorers.agent_judge.subprocess.run")
    def test_score_json_missing_key_reports_stdout_source(self, mock_run, tmp_path):
        """If score.json is valid JSON but has no 'score' key and score comes
        from stdout, score_source must be 'stdout_parse'."""
        pred_dir, ref_dir = _setup_dirs(tmp_path)

        def side_effect(cmd, **kwargs):
            ws = Path(kwargs["cwd"])
            (ws / "score.json").write_text('{"reasoning": "good but no score key"}')
            return _make_subprocess_result(
                stdout='{"score": 6, "reasoning": "from stdout"}'
            )

        mock_run.side_effect = side_effect

        scorer = get_scorer("agent_judge")
        result = scorer.score(pred_dir, ref_dir, _base_config(num_judges=1))

        assert result.details["median_score"] == 6.0
        assert result.details["score_sources"] == ["stdout_parse"]


# ---------------------------------------------------------------------------
# Bug 2: Workspace leak and crash when _prepare_workspace fails
# If _prepare_workspace raises after mkdtemp, the temp dir leaks and
# score() crashes with an unhandled exception.
# ---------------------------------------------------------------------------

class TestBug2WorkspaceLeakOnPrepareFailure:
    @patch("ai4sci_bench.scorers.agent_judge.subprocess.run")
    def test_prepare_workspace_error_returns_score_detail(self, mock_run, tmp_path):
        """If _prepare_workspace fails (e.g. pred_dir symlink error),
        score() should return a ScoreDetail with score=0, not crash."""
        pred_dir = tmp_path / "nonexistent_pred"
        ref_dir = tmp_path / "ref"
        ref_dir.mkdir()

        scorer = get_scorer("agent_judge")
        result = scorer.score(pred_dir, ref_dir, _base_config(num_judges=1))

        assert result.score == 0.0
        assert result.details["parse_failures"] >= 1

    @patch("ai4sci_bench.scorers.agent_judge.subprocess.run")
    def test_prepare_workspace_error_cleans_temp_dir(self, mock_run, tmp_path):
        """Even if _prepare_workspace fails mid-way, any created temp dirs
        should be cleaned up."""
        pred_dir, ref_dir = _setup_dirs(tmp_path)

        original_mkdtemp = tempfile.mkdtemp
        created_dirs = []

        def tracking_mkdtemp(**kwargs):
            d = original_mkdtemp(**kwargs)
            created_dirs.append(d)
            return d

        with patch("ai4sci_bench.scorers.agent_judge.tempfile.mkdtemp", side_effect=tracking_mkdtemp):
            with patch.object(Path, "symlink_to", side_effect=OSError("permission denied")):
                scorer = get_scorer("agent_judge")
                result = scorer.score(pred_dir, ref_dir, _base_config(num_judges=1))

        assert result.score == 0.0
        for d in created_dirs:
            assert not Path(d).exists(), f"Temp dir leaked: {d}"


# ---------------------------------------------------------------------------
# Bug 3: max_score_value=0 causes ZeroDivisionError in aggregate
# ---------------------------------------------------------------------------

class TestBug3MaxScoreZeroDivision:
    def test_aggregate_max_score_zero(self):
        """aggregate_judge_scores with max_score_value=0 should not crash."""
        result = aggregate_judge_scores(
            scores=[5.0],
            raw_responses=["resp"],
            scorer_name="test",
            model="test/model",
            num_judges=1,
            max_score_value=0,
            weight=10.0,
            threshold=0.5,
        )
        assert result.score == 0.0
        assert result.passed is False


# ---------------------------------------------------------------------------
# Bug 5: Preflight runs even when scorer registration fails
# ---------------------------------------------------------------------------

class TestBug5PreflightOnFailedRegistration:
    def test_preflight_skipped_when_scorer_unknown(self):
        """If get_scorer fails for an unknown scorer that happens to be named
        'agent_judge' in a broken config, the preflight should still not crash."""
        from ai4sci_bench.cli import _validate_registered_scorers

        metadata = {
            "evaluation": {
                "scoring": [
                    {"scorer": "agent_judge", "config": {"rubric": "", "timeout": 10}},
                ],
            },
        }
        errors = []
        _validate_registered_scorers(metadata, errors)

        # agent_judge IS registered so get_scorer succeeds, but preflight should
        # still catch the bad config. Both rubric and timeout errors should appear.
        assert any("rubric" in e for e in errors)
        assert any("too low" in e for e in errors)
