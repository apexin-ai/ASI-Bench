"""Tests for LLM Judge and Multimodal scorers."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ai4sci_bench.core.scorer import get_scorer, list_scorers
from ai4sci_bench.core.types import ScoreDetail
from ai4sci_bench.scorers.llm_judge import (
    DEFAULT_MODEL,
    SUPPORTED_MODELS,
    LLMJudgeScorer,
    _resolve_model,
)

# Ensure scorers are registered
import ai4sci_bench.scorers  # noqa: F401


def _make_litellm_response(score: float, reasoning: str = "Good work") -> MagicMock:
    """Create a mock litellm response."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = json.dumps({"score": score, "reasoning": reasoning})
    return resp


class TestLLMJudgeRegistration:
    def test_registered(self):
        names = list_scorers()
        assert "llm_judge" in names
        assert "multimodal" in names

    def test_get_llm_judge(self):
        scorer = get_scorer("llm_judge")
        assert scorer.name == "llm_judge"

    def test_get_multimodal(self):
        scorer = get_scorer("multimodal")
        assert scorer.name == "multimodal"


class TestResolveModel:
    def test_short_name(self):
        assert _resolve_model("gemini-3.1-pro-preview") == "gemini/gemini-3.1-pro-preview"
        assert _resolve_model("claude-opus-4-6") == "anthropic/claude-opus-4-6"
        assert _resolve_model("gpt-5.4") == "openai/gpt-5.4"

    def test_passthrough(self):
        assert _resolve_model("openai/gpt-4o") == "openai/gpt-4o"
        assert _resolve_model("some/custom-model") == "some/custom-model"

    def test_openrouter_short_name(self):
        assert _resolve_model("openrouter/claude-opus-4-6") == "openrouter/anthropic/claude-opus-4-6"
        assert _resolve_model("openrouter/gpt-5.4") == "openrouter/openai/gpt-5.4"
        assert _resolve_model("openrouter/gemini-3.1-pro-preview") == "openrouter/google/gemini-3.1-pro-preview"

    def test_openrouter_full_passthrough(self):
        assert _resolve_model("openrouter/anthropic/claude-opus-4-6") == "openrouter/anthropic/claude-opus-4-6"


class TestLLMJudgeScorer:
    def test_missing_pred_file(self, tmp_path):
        scorer = get_scorer("llm_judge")
        result = scorer.score(tmp_path, tmp_path, {
            "pred_file": "missing.txt",
            "weight": 10.0,
        })
        assert result.passed is False
        assert result.score == 0.0
        assert "not found" in result.message

    @patch("ai4sci_bench.scorers.llm_judge.litellm")
    def test_single_judge_pass(self, mock_litellm, tmp_path):
        # Setup
        pred_dir = tmp_path / "pred"
        pred_dir.mkdir()
        (pred_dir / "output.txt").write_text("Agent output content")

        ref_dir = tmp_path / "ref"
        ref_dir.mkdir()
        (ref_dir / "reference.txt").write_text("Reference content")

        mock_litellm.completion.return_value = _make_litellm_response(8.0)

        scorer = get_scorer("llm_judge")
        result = scorer.score(pred_dir, ref_dir, {
            "pred_file": "output.txt",
            "ref_file": "reference.txt",
            "weight": 20.0,
            "threshold": 0.5,
        })

        assert result.passed is True
        assert result.score == pytest.approx(16.0)  # 8/10 * 20
        assert result.details["median_score"] == 8.0
        assert result.details["num_judges"] == 1
        mock_litellm.completion.assert_called_once()

    @patch("ai4sci_bench.scorers.llm_judge.litellm")
    def test_multi_judge_median(self, mock_litellm, tmp_path):
        pred_dir = tmp_path / "pred"
        pred_dir.mkdir()
        (pred_dir / "output.txt").write_text("Agent output")

        # Return different scores for 3 judges
        mock_litellm.completion.side_effect = [
            _make_litellm_response(6.0),
            _make_litellm_response(8.0),
            _make_litellm_response(7.0),
        ]

        scorer = get_scorer("llm_judge")
        result = scorer.score(pred_dir, tmp_path, {
            "pred_file": "output.txt",
            "num_judges": 3,
            "weight": 10.0,
            "threshold": 0.5,
        })

        assert result.details["median_score"] == 7.0  # median of [6, 8, 7]
        assert result.details["individual_scores"] == [6.0, 8.0, 7.0]
        assert mock_litellm.completion.call_count == 3

    @patch("ai4sci_bench.scorers.llm_judge.litellm")
    def test_judge_fail_threshold(self, mock_litellm, tmp_path):
        pred_dir = tmp_path / "pred"
        pred_dir.mkdir()
        (pred_dir / "output.txt").write_text("Bad output")

        mock_litellm.completion.return_value = _make_litellm_response(2.0)

        scorer = get_scorer("llm_judge")
        result = scorer.score(pred_dir, tmp_path, {
            "pred_file": "output.txt",
            "weight": 10.0,
            "threshold": 0.5,
        })

        assert result.passed is False
        assert result.details["normalized_score"] == pytest.approx(0.2)

    @patch("ai4sci_bench.scorers.llm_judge.litellm")
    def test_custom_model(self, mock_litellm, tmp_path):
        pred_dir = tmp_path / "pred"
        pred_dir.mkdir()
        (pred_dir / "output.txt").write_text("Output")

        mock_litellm.completion.return_value = _make_litellm_response(9.0)

        scorer = get_scorer("llm_judge")
        result = scorer.score(pred_dir, tmp_path, {
            "pred_file": "output.txt",
            "model": "claude-opus-4-6",
            "weight": 10.0,
        })

        call_args = mock_litellm.completion.call_args
        assert call_args.kwargs["model"] == "anthropic/claude-opus-4-6"

    @patch("ai4sci_bench.scorers.llm_judge.litellm")
    def test_openrouter_model_with_api_key(self, mock_litellm, tmp_path):
        pred_dir = tmp_path / "pred"
        pred_dir.mkdir()
        (pred_dir / "output.txt").write_text("Output")

        mock_litellm.completion.return_value = _make_litellm_response(8.0)

        scorer = get_scorer("llm_judge")
        result = scorer.score(pred_dir, tmp_path, {
            "pred_file": "output.txt",
            "model": "openrouter/claude-opus-4-6",
            "weight": 10.0,
            "api_key": "sk-or-test-key",
            "api_base": "https://openrouter.ai/api/v1",
        })

        call_args = mock_litellm.completion.call_args
        assert call_args.kwargs["model"] == "openrouter/anthropic/claude-opus-4-6"
        assert call_args.kwargs["api_key"] == "sk-or-test-key"
        assert call_args.kwargs["api_base"] == "https://openrouter.ai/api/v1"
        assert result.passed is True

    @patch("ai4sci_bench.scorers.llm_judge.litellm")
    def test_runtime_judge_override_uses_openai_compatible_route(self, mock_litellm, tmp_path):
        pred_dir = tmp_path / "pred"
        pred_dir.mkdir()
        (pred_dir / "output.txt").write_text("Output")
        mock_litellm.completion.return_value = _make_litellm_response(8.0)

        with patch.dict(os.environ, {"TOKENROUTER_API_KEY": "tokenrouter-secret"}, clear=False):
            from ai4sci_bench.core.judge_api import resolve_judge_api_override, use_judge_api_override
            override = resolve_judge_api_override(
                "https://api.tokenrouter.com/v1", "TOKENROUTER_API_KEY", "openai"
            )
            with use_judge_api_override(override):
                result = get_scorer("llm_judge").score(pred_dir, tmp_path, {
                    "pred_file": "output.txt",
                    "model": "google/gemini-3.5-flash",
                    "weight": 10.0,
                })

        kwargs = mock_litellm.completion.call_args.kwargs
        assert kwargs["model"] == "openai/google/gemini-3.5-flash"
        assert kwargs["api_base"] == "https://api.tokenrouter.com/v1"
        assert kwargs["api_key"] == "tokenrouter-secret"
        assert result.details["judge_api"] == {
            "api_base": "https://api.tokenrouter.com/v1",
            "api_protocol": "openai",
            "api_key_env": "TOKENROUTER_API_KEY",
        }
        assert "tokenrouter-secret" not in str(result.details)

    @patch("ai4sci_bench.scorers.llm_judge.litellm")
    def test_native_provider_environment_key_is_redacted(self, mock_litellm, tmp_path):
        pred_dir = tmp_path / "pred"
        pred_dir.mkdir()
        (pred_dir / "output.txt").write_text("Output")
        secret = "native-gemini-secret"
        mock_litellm.completion.return_value = _make_litellm_response(
            8.0, reasoning=f"provider echoed {secret}"
        )

        from ai4sci_bench.core.judge_api import use_judge_api_override

        with patch.dict(os.environ, {"GEMINI_API_KEY": secret}, clear=False):
            with use_judge_api_override(None):
                result = get_scorer("llm_judge").score(pred_dir, tmp_path, {
                    "pred_file": "output.txt",
                    "model": "google/gemini-3.5-flash",
                    "weight": 10.0,
                })

        kwargs = mock_litellm.completion.call_args.kwargs
        assert kwargs["api_key"] == secret
        assert secret not in str(result.details)
        assert "<redacted>" in result.details["raw_responses"][0]
        assert result.details["judge_api"] == {"api_key_env": "GEMINI_API_KEY"}

    @patch("ai4sci_bench.scorers.llm_judge.litellm")
    def test_api_key_not_passed_when_none(self, mock_litellm, tmp_path):
        pred_dir = tmp_path / "pred"
        pred_dir.mkdir()
        (pred_dir / "output.txt").write_text("Output")

        mock_litellm.completion.return_value = _make_litellm_response(7.0)

        scorer = get_scorer("llm_judge")
        scorer.score(pred_dir, tmp_path, {
            "pred_file": "output.txt",
            "weight": 10.0,
        })

        call_args = mock_litellm.completion.call_args
        assert "api_key" not in call_args.kwargs
        assert "api_base" not in call_args.kwargs

    @patch("ai4sci_bench.scorers.llm_judge.litellm")
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-env-test"})
    def test_openrouter_api_key_from_env(self, mock_litellm, tmp_path):
        """OpenRouter model auto-reads OPENROUTER_API_KEY from .env / env var."""
        pred_dir = tmp_path / "pred"
        pred_dir.mkdir()
        (pred_dir / "output.txt").write_text("Output")

        mock_litellm.completion.return_value = _make_litellm_response(9.0)

        scorer = get_scorer("llm_judge")
        scorer.score(pred_dir, tmp_path, {
            "pred_file": "output.txt",
            "model": "openrouter/claude-opus-4-6",
            "weight": 10.0,
        })

        call_args = mock_litellm.completion.call_args
        assert call_args.kwargs["model"] == "openrouter/anthropic/claude-opus-4-6"
        assert call_args.kwargs["api_key"] == "sk-or-env-test"

    def test_openrouter_fallback_key_is_redacted_from_retry_log(self, caplog):
        secret = "sk-or-retry-secret"
        scorer = get_scorer("llm_judge")
        with (
            patch.dict(os.environ, {"OPENROUTER_API_KEY": secret}, clear=False),
            patch(
                "ai4sci_bench.scorers.llm_judge.litellm.completion",
                side_effect=Exception(f"request rejected for {secret}"),
            ),
            patch("ai4sci_bench.scorers.llm_judge.time.sleep"),
            caplog.at_level("WARNING"),
            pytest.raises(Exception),
        ):
            scorer._call_with_retry(
                "openrouter/google/gemini-3.5-flash",
                "judge prompt",
                0.0,
                100,
            )

        assert secret not in caplog.text
        assert "<redacted>" in caplog.text

    @patch("ai4sci_bench.scorers.llm_judge.litellm")
    def test_explicit_api_key_overrides_env(self, mock_litellm, tmp_path):
        """Explicit api_key in config takes precedence over env var."""
        pred_dir = tmp_path / "pred"
        pred_dir.mkdir()
        (pred_dir / "output.txt").write_text("Output")

        mock_litellm.completion.return_value = _make_litellm_response(8.0)

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-env"}):
            scorer = get_scorer("llm_judge")
            scorer.score(pred_dir, tmp_path, {
                "pred_file": "output.txt",
                "model": "openrouter/claude-opus-4-6",
                "weight": 10.0,
                "api_key": "sk-or-explicit",
            })

        call_args = mock_litellm.completion.call_args
        assert call_args.kwargs["api_key"] == "sk-or-explicit"

    @patch("ai4sci_bench.scorers.llm_judge.litellm")
    def test_api_error_handled(self, mock_litellm, tmp_path):
        pred_dir = tmp_path / "pred"
        pred_dir.mkdir()
        (pred_dir / "output.txt").write_text("Output")

        mock_litellm.completion.side_effect = Exception("API rate limit exceeded")

        scorer = get_scorer("llm_judge")
        result = scorer.score(pred_dir, tmp_path, {
            "pred_file": "output.txt",
            "weight": 10.0,
        })

        assert result.passed is False
        assert "API rate limit" in result.details["error"]
        assert result.details["evaluator_status"] == "unavailable"
        assert result.details["evaluator_unavailable"] is True
        assert "unavailable" in result.message.lower()

    def test_parse_judge_response_json(self):
        from ai4sci_bench.scorers._parse_utils import parse_judge_json
        assert parse_judge_json('{"score": 7.5, "reasoning": "ok"}') == 7.5

    def test_parse_judge_response_with_text(self):
        from ai4sci_bench.scorers._parse_utils import parse_judge_json
        raw = 'Based on my analysis, here is my evaluation:\n{"score": 6, "reasoning": "decent"}\n'
        assert parse_judge_json(raw) == 6.0

    def test_parse_judge_response_score_format(self):
        from ai4sci_bench.scorers._parse_utils import parse_judge_json
        assert parse_judge_json("Score: 8.5 out of 10") == 8.5

    def test_parse_judge_response_fraction(self):
        from ai4sci_bench.scorers._parse_utils import parse_judge_json
        assert parse_judge_json("I give this a 7/10") == 7.0

    def test_parse_judge_response_fallback(self):
        from ai4sci_bench.scorers._parse_utils import parse_judge_json
        assert parse_judge_json("This is garbage text with no score") is None

    @patch("ai4sci_bench.scorers.llm_judge.litellm")
    def test_no_ref_file(self, mock_litellm, tmp_path):
        """LLM judge should work without a reference file."""
        pred_dir = tmp_path / "pred"
        pred_dir.mkdir()
        (pred_dir / "output.txt").write_text("Agent output")

        mock_litellm.completion.return_value = _make_litellm_response(7.0)

        scorer = get_scorer("llm_judge")
        result = scorer.score(pred_dir, tmp_path, {
            "pred_file": "output.txt",
            "weight": 10.0,
        })

        assert result.passed is True
        assert result.score == pytest.approx(7.0)


class TestMultimodalScorer:
    def test_unknown_mode(self, tmp_path):
        scorer = get_scorer("multimodal")
        result = scorer.score(tmp_path, tmp_path, {
            "mode": "unknown_mode",
            "weight": 10.0,
        })
        assert result.passed is False
        assert "unknown_mode" in result.message

    def test_vlm_judge_missing_pred_image_config(self, tmp_path):
        scorer = get_scorer("multimodal")
        result = scorer.score(tmp_path, tmp_path, {
            "mode": "vlm_judge",
            "weight": 10.0,
        })
        assert result.passed is False
        assert "pred_image not specified" in result.message

    def test_vlm_judge_missing_pred_image_file(self, tmp_path):
        scorer = get_scorer("multimodal")
        result = scorer.score(tmp_path, tmp_path, {
            "mode": "vlm_judge",
            "pred_image": "missing.png",
            "weight": 10.0,
        })
        assert result.passed is False
        assert "not found" in result.message

    @patch("ai4sci_bench.scorers.multimodal.litellm")
    def test_vlm_judge_pass(self, mock_litellm, tmp_path):
        pred_dir = tmp_path / "pred"
        pred_dir.mkdir()
        ref_dir = tmp_path / "ref"
        ref_dir.mkdir()

        # Create minimal PNG files (1x1 pixel)
        _create_minimal_png(pred_dir / "plot.png")
        _create_minimal_png(ref_dir / "ref_plot.png")

        mock_litellm.completion.return_value = _make_litellm_response(8.0)

        scorer = get_scorer("multimodal")
        result = scorer.score(pred_dir, ref_dir, {
            "mode": "vlm_judge",
            "pred_image": "plot.png",
            "ref_image": "ref_plot.png",
            "weight": 20.0,
            "threshold": 0.5,
        })

        assert result.passed is True
        assert result.score == pytest.approx(16.0)
        assert result.details["median_score"] == 8.0

    @patch("ai4sci_bench.scorers.multimodal.litellm")
    def test_vlm_runtime_judge_override_matches_text_route(self, mock_litellm, tmp_path):
        pred_dir = tmp_path / "pred"
        pred_dir.mkdir()
        _create_minimal_png(pred_dir / "plot.png")
        mock_litellm.completion.return_value = _make_litellm_response(8.0)

        with patch.dict(os.environ, {"TOKENROUTER_API_KEY": "tokenrouter-secret"}, clear=False):
            from ai4sci_bench.core.judge_api import resolve_judge_api_override, use_judge_api_override
            override = resolve_judge_api_override(
                "https://api.tokenrouter.com/v1", "TOKENROUTER_API_KEY", "openai"
            )
            with use_judge_api_override(override):
                result = get_scorer("multimodal").score(tmp_path / "pred", tmp_path, {
                    "mode": "vlm_judge",
                    "model": "google/gemini-3.5-flash",
                    "pred_image": "plot.png",
                    "weight": 10.0,
                })

        kwargs = mock_litellm.completion.call_args.kwargs
        assert kwargs["model"] == "openai/google/gemini-3.5-flash"
        assert kwargs["api_base"] == "https://api.tokenrouter.com/v1"
        assert kwargs["api_key"] == "tokenrouter-secret"
        assert result.details["judge_api"]["api_protocol"] == "openai"

    @patch("ai4sci_bench.scorers.multimodal.litellm")
    def test_vlm_judge_multi_judge(self, mock_litellm, tmp_path):
        pred_dir = tmp_path / "pred"
        pred_dir.mkdir()
        _create_minimal_png(pred_dir / "plot.png")

        mock_litellm.completion.side_effect = [
            _make_litellm_response(5.0),
            _make_litellm_response(7.0),
            _make_litellm_response(6.0),
        ]

        scorer = get_scorer("multimodal")
        result = scorer.score(pred_dir, tmp_path, {
            "mode": "vlm_judge",
            "pred_image": "plot.png",
            "num_judges": 3,
            "weight": 10.0,
        })

        assert result.details["median_score"] == 6.0
        assert mock_litellm.completion.call_count == 3

    def test_vlm_openrouter_fallback_key_is_redacted_from_retry_log(self, caplog):
        secret = "sk-or-vlm-retry-secret"
        scorer = get_scorer("multimodal")
        with (
            patch.dict(os.environ, {"OPENROUTER_API_KEY": secret}, clear=False),
            patch(
                "ai4sci_bench.scorers.multimodal.litellm.completion",
                side_effect=Exception(f"request rejected for {secret}"),
            ),
            patch("ai4sci_bench.scorers.multimodal.time.sleep"),
            caplog.at_level("WARNING"),
            pytest.raises(Exception),
        ):
            scorer._call_vlm_with_retry(
                "openrouter/google/gemini-3.5-flash",
                [],
                0.0,
                100,
            )

        assert secret not in caplog.text
        assert "<redacted>" in caplog.text

    @patch("ai4sci_bench.scorers.multimodal.litellm")
    def test_vlm_api_error_marked_unavailable(self, mock_litellm, tmp_path):
        pred_dir = tmp_path / "pred"
        pred_dir.mkdir()
        _create_minimal_png(pred_dir / "plot.png")
        mock_litellm.completion.side_effect = Exception("OpenRouter 401 Unauthorized")

        scorer = get_scorer("multimodal")
        result = scorer.score(pred_dir, tmp_path, {
            "mode": "vlm_judge",
            "pred_image": "plot.png",
            "weight": 10.0,
        })

        assert result.passed is False
        assert result.details["evaluator_status"] == "unavailable"
        assert result.details["evaluator_unavailable"] is True
        assert "OpenRouter 401" in result.details["error"]

    def test_pixel_sim_missing_config(self, tmp_path):
        scorer = get_scorer("multimodal")
        result = scorer.score(tmp_path, tmp_path, {
            "mode": "pixel_sim",
            "weight": 10.0,
        })
        assert result.passed is False
        assert "not specified" in result.message

    def test_pixel_sim_missing_file(self, tmp_path):
        scorer = get_scorer("multimodal")
        result = scorer.score(tmp_path, tmp_path, {
            "mode": "pixel_sim",
            "pred_image": "missing.png",
            "ref_image": "missing_ref.png",
            "weight": 10.0,
        })
        assert result.passed is False
        assert "not found" in result.message

    def test_pixel_sim_ssim_identical(self, tmp_path):
        """SSIM of identical images should be ~1.0."""
        pytest.importorskip("PIL")

        pred_dir = tmp_path / "pred"
        ref_dir = tmp_path / "ref"
        pred_dir.mkdir()
        ref_dir.mkdir()

        _create_test_image(pred_dir / "plot.png", color=(100, 150, 200))
        _create_test_image(ref_dir / "ref_plot.png", color=(100, 150, 200))

        scorer = get_scorer("multimodal")
        result = scorer.score(pred_dir, ref_dir, {
            "mode": "pixel_sim",
            "pixel_metric": "ssim",
            "pred_image": "plot.png",
            "ref_image": "ref_plot.png",
            "weight": 10.0,
            "threshold": 0.9,
        })

        assert result.passed is True
        assert result.details["similarity"] == pytest.approx(1.0, abs=0.01)

    def test_pixel_sim_psnr_identical(self, tmp_path):
        """PSNR of identical images should be very high."""
        pytest.importorskip("PIL")

        pred_dir = tmp_path / "pred"
        ref_dir = tmp_path / "ref"
        pred_dir.mkdir()
        ref_dir.mkdir()

        _create_test_image(pred_dir / "plot.png", color=(100, 150, 200))
        _create_test_image(ref_dir / "ref_plot.png", color=(100, 150, 200))

        scorer = get_scorer("multimodal")
        result = scorer.score(pred_dir, ref_dir, {
            "mode": "pixel_sim",
            "pixel_metric": "psnr",
            "pred_image": "plot.png",
            "ref_image": "ref_plot.png",
            "weight": 10.0,
            "threshold": 0.9,
        })

        assert result.passed is True

    def test_pixel_sim_ssim_different(self, tmp_path):
        """SSIM of very different images should be low."""
        pytest.importorskip("PIL")

        pred_dir = tmp_path / "pred"
        ref_dir = tmp_path / "ref"
        pred_dir.mkdir()
        ref_dir.mkdir()

        _create_test_image(pred_dir / "plot.png", color=(0, 0, 0))
        _create_test_image(ref_dir / "ref_plot.png", color=(255, 255, 255))

        scorer = get_scorer("multimodal")
        result = scorer.score(pred_dir, ref_dir, {
            "mode": "pixel_sim",
            "pixel_metric": "ssim",
            "pred_image": "plot.png",
            "ref_image": "ref_plot.png",
            "weight": 10.0,
            "threshold": 0.9,
        })

        assert result.passed is False

    def test_pixel_sim_shape_mismatch(self, tmp_path):
        """Different-sized images should fail."""
        pytest.importorskip("PIL")

        pred_dir = tmp_path / "pred"
        ref_dir = tmp_path / "ref"
        pred_dir.mkdir()
        ref_dir.mkdir()

        _create_test_image(pred_dir / "plot.png", size=(50, 50))
        _create_test_image(ref_dir / "ref_plot.png", size=(100, 100))

        scorer = get_scorer("multimodal")
        result = scorer.score(pred_dir, ref_dir, {
            "mode": "pixel_sim",
            "pixel_metric": "ssim",
            "pred_image": "plot.png",
            "ref_image": "ref_plot.png",
            "weight": 10.0,
        })

        assert result.passed is False
        assert "mismatch" in result.message.lower()

    def test_pixel_sim_unknown_metric(self, tmp_path):
        pytest.importorskip("PIL")

        pred_dir = tmp_path / "pred"
        ref_dir = tmp_path / "ref"
        pred_dir.mkdir()
        ref_dir.mkdir()

        _create_test_image(pred_dir / "plot.png")
        _create_test_image(ref_dir / "ref_plot.png")

        scorer = get_scorer("multimodal")
        result = scorer.score(pred_dir, ref_dir, {
            "mode": "pixel_sim",
            "pixel_metric": "unknown_metric",
            "pred_image": "plot.png",
            "ref_image": "ref_plot.png",
            "weight": 10.0,
        })

        assert result.passed is False
        assert "Unknown pixel" in result.message


class TestLogger:
    def test_get_logger(self):
        from ai4sci_bench.core.logger import get_logger
        log = get_logger("test_module")
        assert log.name == "ai4sci_bench.test_module"

    def test_get_logger_already_namespaced(self):
        from ai4sci_bench.core.logger import get_logger
        log = get_logger("ai4sci_bench.something")
        assert log.name == "ai4sci_bench.something"

    def test_setup_logging(self, tmp_path):
        import logging
        from ai4sci_bench.core.logger import setup_logging

        # Clear handlers for test isolation
        root = logging.getLogger("ai4sci_bench")
        root.handlers.clear()

        log_file = tmp_path / "test.log"
        setup_logging(level="DEBUG", log_file=str(log_file))

        assert root.level == logging.DEBUG
        assert len(root.handlers) == 2  # console + file

        # Cleanup
        root.handlers.clear()

    def test_setup_logging_no_duplicate_handlers(self):
        import logging
        from ai4sci_bench.core.logger import setup_logging

        root = logging.getLogger("ai4sci_bench")
        root.handlers.clear()

        setup_logging(level="INFO")
        handler_count = len(root.handlers)
        setup_logging(level="INFO")  # second call should not add more handlers
        assert len(root.handlers) == handler_count

        # Cleanup
        root.handlers.clear()


# --- Helpers ---

def _create_minimal_png(path: Path) -> None:
    """Create a minimal valid PNG file (bytes)."""
    # Minimal 1x1 red pixel PNG
    import struct
    import zlib

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + c + crc

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    raw_data = b"\x00\xff\x00\x00"  # filter=none, R=255, G=0, B=0
    idat = _chunk(b"IDAT", zlib.compress(raw_data))
    iend = _chunk(b"IEND", b"")

    path.write_bytes(signature + ihdr + idat + iend)


def _create_test_image(path: Path, size: tuple[int, int] = (50, 50), color: tuple[int, int, int] = (100, 150, 200)) -> None:
    """Create a test image using Pillow."""
    from PIL import Image
    img = Image.new("RGB", size, color)
    img.save(path)
