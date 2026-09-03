"""Tests for _judge_common.py — shared judge infrastructure."""

import pytest
from unittest.mock import patch

from ai4sci_bench.core.judge_api import (
    JudgeAPIConfigurationError,
    get_judge_api_override,
    resolve_judge_api_override,
    use_judge_api_override,
)
from ai4sci_bench.scorers._judge_common import (
    DEFAULT_MODEL,
    SUPPORTED_MODELS,
    aggregate_judge_scores,
    judge_completion_api_kwargs,
    resolve_judge_api,
    resolve_model,
    sanitize_judge_error,
)


class TestResolveModel:
    def test_resolve_model_known(self):
        assert resolve_model("gpt-5.4") == "openai/gpt-5.4"
        assert resolve_model("claude-opus-4-6") == "anthropic/claude-opus-4-6"
        assert resolve_model("gemini-3.1-pro-preview") == "gemini/gemini-3.1-pro-preview"

    def test_resolve_model_passthrough(self):
        assert resolve_model("openai/gpt-4o") == "openai/gpt-4o"
        assert resolve_model("some/custom-model") == "some/custom-model"

    def test_openai_protocol_keeps_gateway_model_id(self):
        assert resolve_model(
            "google/gemini-3.5-flash", api_protocol="openai"
        ) == "openai/google/gemini-3.5-flash"
        assert resolve_model(
            "openai/google/gemini-3.5-flash", api_protocol="openai"
        ) == "openai/google/gemini-3.5-flash"

    def test_openai_protocol_namespaces_bare_gemini_model(self):
        assert resolve_model(
            "gemini-3.5-flash", api_protocol="openai"
        ) == "openai/google/gemini-3.5-flash"

    def test_native_protocol_namespaces_bare_gemini_model(self):
        assert resolve_model("gemini-3.5-flash") == "gemini/gemini-3.5-flash"


class TestJudgeAPIOverride:
    def test_manual_override_cannot_bypass_validation(self):
        from ai4sci_bench.core.judge_api import JudgeAPIOverride

        with pytest.raises(JudgeAPIConfigurationError, match="protocol"):
            JudgeAPIOverride(
                api_base="https://api.example.test/v1",
                api_key_env="JUDGE_KEY",
            )
        with pytest.raises(JudgeAPIConfigurationError, match="key-env"):
            JudgeAPIOverride(
                api_base="https://api.example.test/v1",
                api_protocol="openai",
            )

    def test_override_requires_key_and_protocol_for_custom_base(self):
        with pytest.raises(JudgeAPIConfigurationError, match="key environment"):
            resolve_judge_api_override(
                "https://api.example.test/v1", "MISSING_KEY", "openai",
                environ={},
            )
        with pytest.raises(JudgeAPIConfigurationError, match="protocol"):
            resolve_judge_api_override(
                "https://api.example.test/v1", "KEY", None,
                environ={"KEY": "secret"},
            )

    def test_override_validates_url_and_env_name(self):
        with pytest.raises(JudgeAPIConfigurationError, match="HTTP"):
            resolve_judge_api_override(
                "not-a-url", "KEY", "openai", environ={"KEY": "secret"}
            )
        with pytest.raises(JudgeAPIConfigurationError, match="environment variable name"):
            resolve_judge_api_override(
                "https://api.example.test/v1", "not-valid-name", "openai",
                environ={"not-valid-name": "secret"},
            )
        with pytest.raises(JudgeAPIConfigurationError, match="valid HTTP"):
            resolve_judge_api_override(
                "https://api.example.test:bad/v1", "KEY", "openai",
                environ={"KEY": "secret"},
            )

    @pytest.mark.parametrize("candidate", ["sk-raw-secret", "AIzaRawSecretValue"])
    def test_key_env_validation_never_echoes_accidental_secret(self, candidate):
        with pytest.raises(JudgeAPIConfigurationError) as exc_info:
            resolve_judge_api_override(
                "https://api.example.test/v1",
                candidate,
                "openai",
                environ={},
            )
        assert candidate not in str(exc_info.value)

    def test_override_metadata_never_contains_secret(self):
        override = resolve_judge_api_override(
            "https://api.example.test/v1", "TOKENROUTER_API_KEY", "openai",
            environ={"TOKENROUTER_API_KEY": "secret-value"},
        )
        assert override is not None
        assert override.public_metadata() == {
            "api_base": "https://api.example.test/v1",
            "api_protocol": "openai",
            "api_key_env": "TOKENROUTER_API_KEY",
        }
        assert "secret-value" not in repr(override)

    def test_runtime_override_routes_and_authenticates(self):
        with patch.dict("os.environ", {"TOKENROUTER_API_KEY": "secret-value"}, clear=False):
            override = resolve_judge_api_override(
                "https://api.example.test/v1", "TOKENROUTER_API_KEY", "openai"
            )
            assert override is not None
            with use_judge_api_override(override):
                resolved = resolve_judge_api({"model": "google/gemini-3.5-flash"})
        assert resolved.model == "openai/google/gemini-3.5-flash"
        assert resolved.api_base == "https://api.example.test/v1"
        assert resolved.api_key == "secret-value"
        assert resolved.public_metadata()["api_key_env"] == "TOKENROUTER_API_KEY"

    def test_key_only_override_supports_native_provider(self):
        override = resolve_judge_api_override(
            None,
            "GEMINI_JUDGE_KEY",
            None,
            environ={"GEMINI_JUDGE_KEY": "secret-value"},
        )
        assert override is not None
        with patch.dict("os.environ", {"GEMINI_JUDGE_KEY": "secret-value"}, clear=False):
            with use_judge_api_override(override):
                resolved = resolve_judge_api({"model": "google/gemini-3.5-flash"})
        assert resolved.model == "gemini/gemini-3.5-flash"
        assert resolved.api_base is None
        assert resolved.api_key == "secret-value"
        assert resolved.api_protocol is None

    def test_native_provider_key_is_resolved_for_redaction(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "native-secret"}, clear=False):
            with use_judge_api_override(None):
                resolved = resolve_judge_api({"model": "google/gemini-3.5-flash"})
        assert resolved.model == "gemini/gemini-3.5-flash"
        assert resolved.api_key == "native-secret"
        assert resolved.api_key_env == "GEMINI_API_KEY"
        assert "native-secret" not in repr(resolved)

    def test_explicit_none_scope_suppresses_ambient_override(self):
        environment = {
            "ASIBENCH_JUDGE_API_BASE": "https://api.example.test/v1",
            "ASIBENCH_JUDGE_API_KEY_ENV": "AMBIENT_JUDGE_KEY",
            "ASIBENCH_JUDGE_API_PROTOCOL": "openai",
            "AMBIENT_JUDGE_KEY": "ambient-secret",
        }
        with patch.dict("os.environ", environment, clear=False):
            ambient = get_judge_api_override()
            assert ambient is not None
            with use_judge_api_override(None):
                assert get_judge_api_override() is None
            assert get_judge_api_override() == ambient

    def test_custom_endpoint_does_not_use_openrouter_fallback(self):
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "wrong-key"}, clear=False):
            assert judge_completion_api_kwargs(
                model="openrouter/google/gemini-3.5-flash",
                api_base="https://api.example.test/v1",
                api_key=None,
            ) == {"api_base": "https://api.example.test/v1"}

    def test_openrouter_route_keeps_legacy_environment_fallback(self):
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "provider-key"}, clear=False):
            assert judge_completion_api_kwargs(
                model="openrouter/google/gemini-3.5-flash",
                api_base=None,
                api_key=None,
            ) == {"api_key": "provider-key"}

    def test_error_sanitization(self):
        assert sanitize_judge_error(
            "request failed with secret-value", secrets=("secret-value",)
        ) == "request failed with <redacted>"

    def test_legacy_openrouter_fallback_is_redacted_from_raw_response(self):
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "provider-key"}, clear=False):
            result = aggregate_judge_scores(
                scores=[8.0],
                raw_responses=["provider echoed provider-key"],
                scorer_name="llm_judge",
                model="openrouter/google/gemini-3.5-flash",
                num_judges=1,
                max_score_value=10,
                weight=1,
                threshold=0.5,
            )
        assert result.details["raw_responses"] == ["provider echoed <redacted>"]


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
