"""Tests for _judge_common.py — shared judge infrastructure."""

import pytest

from ai4sci_bench.scorers._judge_common import (
    DEFAULT_MODEL,
    SUPPORTED_MODELS,
    aggregate_judge_scores,
    resolve_model,
)


class TestResolveModel:
    def test_resolve_model_known(self):
        assert resolve_model("gpt-5.4") == "openai/gpt-5.4"
        assert resolve_model("claude-opus-4-6") == "anthropic/claude-opus-4-6"
        assert resolve_model("gemini-3.1-pro-preview") == "gemini/gemini-3.1-pro-preview"

    def test_resolve_model_passthrough(self):
        assert resolve_model("openai/gpt-4o") == "openai/gpt-4o"
        assert resolve_model("some/custom-model") == "some/custom-model"


class TestAggregateJudgeScores:
    def _make(self, scores, raw_responses=None, **kwargs):
        if raw_responses is None:
            raw_responses = [f"resp_{i}" for i in range(len(scores))]
        defaults = dict(
            scorer_name="test_scorer",
            model="test/model",
            num_judges=len(scores),
            max_score_value=10.0,
            weight=10.0,
            threshold=0.5,
        )
        defaults.update(kwargs)
        return aggregate_judge_scores(scores=scores, raw_responses=raw_responses, **defaults)

    def test_aggregate_all_valid(self):
        result = self._make([7.0, 8.0, 9.0])
        assert result.details["median_score"] == 8.0
        assert result.details["parse_failures"] == 0
        assert result.details["valid_judge_count"] == 3
        assert result.score == pytest.approx(8.0)  # 8/10 * 10

    def test_aggregate_with_none(self):
        result = self._make([7.0, None, 9.0])
        assert result.details["median_score"] == 8.0
        assert result.details["parse_failures"] == 1
        assert result.details["valid_judge_count"] == 2

    def test_aggregate_all_none(self):
        result = self._make([None, None, None])
        assert result.score == 0.0
        assert result.details["parse_failures"] == 3
        assert result.details["valid_judge_count"] == 0

    def test_aggregate_single_score(self):
        result = self._make([8.0])
        assert result.details["median_score"] == 8.0

    def test_aggregate_clamp(self):
        """Scores from judges are pre-clamped by parse_judge_json, but
        aggregate should work correctly with any valid float values."""
        result = self._make([0.0, 10.0, 5.0])
        assert result.details["median_score"] == 5.0

    def test_aggregate_score_detail_fields(self):
        result = self._make(
            [7.0, 8.0, 9.0],
            scorer_name="my_scorer",
            weight=20.0,
            threshold=0.6,
        )
        assert result.scorer_name == "my_scorer"
        assert result.max_score == 20.0
        assert result.passed is True  # 8/10 = 0.8 >= 0.6
        assert result.details["threshold"] == 0.6
        assert result.details["model"] == "test/model"
        assert result.details["num_judges"] == 3

    def test_aggregate_extra_details_merged(self):
        result = self._make(
            [8.0],
            extra_details={"agent_type": "codex", "effort": "xhigh"},
        )
        assert result.details["agent_type"] == "codex"
        assert result.details["effort"] == "xhigh"
        assert result.details["median_score"] == 8.0

    def test_aggregate_normalized_score(self):
        result = self._make([5.0], max_score_value=10.0, weight=20.0)
        assert result.details["normalized_score"] == 0.5
        assert result.score == pytest.approx(10.0)

    def test_aggregate_individual_scores_format(self):
        result = self._make([7.0, None, 9.0])
        assert result.details["individual_scores"] == [7.0, "PARSE_FAILED", 9.0]

    def test_aggregate_pass_fail(self):
        result_pass = self._make([8.0], threshold=0.5)
        assert result_pass.passed is True

        result_fail = self._make([3.0], threshold=0.5)
        assert result_fail.passed is False
