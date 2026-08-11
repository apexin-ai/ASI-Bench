"""Tests for agent_judge scorer — comprehensive coverage per design doc."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from ai4sci_bench.core.scorer import get_scorer, list_scorers
from ai4sci_bench.core.types import ToolMode

import ai4sci_bench.scorers  # noqa: F401  — trigger registration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
# Registration
# ---------------------------------------------------------------------------

class TestAgentJudgeRegistration:
    def test_registered(self):
        assert "agent_judge" in list_scorers()

    def test_get_scorer(self):
        scorer = get_scorer("agent_judge")
        assert scorer.name == "agent_judge"

    def test_scorer_registry_orchestrator(self):
        scorer = get_scorer("agent_judge")
        assert hasattr(scorer, "score")


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------

class TestAgentJudgeCommands:
    @patch("ai4sci_bench.scorers.agent_judge.subprocess.run")
    def test_score_codex_default_config(self, mock_run, tmp_path):
        pred_dir, ref_dir = _setup_dirs(tmp_path)
        score_json = json.dumps({"score": 8, "reasoning": "good"})

        def side_effect(cmd, **kwargs):
            ws = Path(kwargs["cwd"])
            (ws / "score.json").write_text(score_json)
            return _make_subprocess_result(stdout="done")

        mock_run.side_effect = side_effect

        scorer = get_scorer("agent_judge")
        result = scorer.score(pred_dir, ref_dir, _base_config(num_judges=1))

        cmd = mock_run.call_args_list[0][0][0]
        assert "codex" in cmd
        assert "exec" in cmd
        assert "--model" in cmd
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "gpt-5.5"
        assert "--effort" not in cmd
        assert "--sandbox" in cmd
        idx = cmd.index("--sandbox")
        assert cmd[idx + 1] == "workspace-write"
        assert "--skip-git-repo-check" in cmd
        cmd_str = " ".join(cmd)
        assert 'model_reasoning_effort="xhigh"' in cmd_str

    @patch("ai4sci_bench.scorers.agent_judge.subprocess.run")
    def test_score_claude_code_config(self, mock_run, tmp_path):
        pred_dir, ref_dir = _setup_dirs(tmp_path)
        score_json = json.dumps({"score": 7, "reasoning": "ok"})

        def side_effect(cmd, **kwargs):
            ws = Path(kwargs["cwd"])
            (ws / "score.json").write_text(score_json)
            return _make_subprocess_result(stdout="done")

        mock_run.side_effect = side_effect

        scorer = get_scorer("agent_judge")
        result = scorer.score(pred_dir, ref_dir, _base_config(
            agent="claude_code", num_judges=1,
        ))

        cmd = mock_run.call_args_list[0][0][0]
        assert "claude" in cmd
        assert "--model" in cmd
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "claude-opus-4-6"
        assert "--permission-mode" in cmd
        idx = cmd.index("--permission-mode")
        assert cmd[idx + 1] == "bypassPermissions"
        assert "--max-turns" not in cmd

    @patch("ai4sci_bench.scorers.agent_judge.subprocess.run")
    def test_score_custom_model_and_reasoning_effort(self, mock_run, tmp_path):
        pred_dir, ref_dir = _setup_dirs(tmp_path)

        def side_effect(cmd, **kwargs):
            ws = Path(kwargs["cwd"])
            (ws / "score.json").write_text('{"score": 5}')
            return _make_subprocess_result()

        mock_run.side_effect = side_effect

        scorer = get_scorer("agent_judge")
        scorer.score(pred_dir, ref_dir, _base_config(
            model="gpt-4o", reasoning_effort="high", num_judges=1,
        ))

        cmd = mock_run.call_args_list[0][0][0]
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "gpt-4o"
        cmd_str = " ".join(cmd)
        assert 'model_reasoning_effort="high"' in cmd_str


# ---------------------------------------------------------------------------
# Tool mode isolation
# ---------------------------------------------------------------------------

class TestToolModeIsolation:
    @patch("ai4sci_bench.scorers.agent_judge.subprocess.run")
    def test_tool_mode_unrestricted_default(self, mock_run, tmp_path):
        pred_dir, ref_dir = _setup_dirs(tmp_path)

        def side_effect(cmd, **kwargs):
            ws = Path(kwargs["cwd"])
            (ws / "score.json").write_text('{"score": 8}')
            return _make_subprocess_result()

        mock_run.side_effect = side_effect

        scorer = get_scorer("agent_judge")
        scorer.score(pred_dir, ref_dir, _base_config(num_judges=1))

        cmd = mock_run.call_args_list[0][0][0]
        assert "--ignore-user-config" not in cmd
        assert "--disable" not in cmd

    @patch("ai4sci_bench.scorers.agent_judge.subprocess.run")
    def test_tool_mode_restricted_codex(self, mock_run, tmp_path):
        pred_dir, ref_dir = _setup_dirs(tmp_path)

        def side_effect(cmd, **kwargs):
            ws = Path(kwargs["cwd"])
            (ws / "score.json").write_text('{"score": 8}')
            return _make_subprocess_result()

        mock_run.side_effect = side_effect

        scorer = get_scorer("agent_judge")
        scorer.score(pred_dir, ref_dir, _base_config(
            tool_mode="restricted", num_judges=1,
        ))

        cmd = mock_run.call_args_list[0][0][0]
        assert "--ignore-user-config" in cmd
        assert "--disable" in cmd
        cmd_str = " ".join(cmd)
        assert 'web_search="disabled"' in cmd_str

    @patch("ai4sci_bench.scorers.agent_judge.subprocess.run")
    def test_tool_mode_restricted_claude(self, mock_run, tmp_path):
        pred_dir, ref_dir = _setup_dirs(tmp_path)

        def side_effect(cmd, **kwargs):
            ws = Path(kwargs["cwd"])
            (ws / "score.json").write_text('{"score": 8}')
            return _make_subprocess_result()

        mock_run.side_effect = side_effect

        scorer = get_scorer("agent_judge")
        scorer.score(pred_dir, ref_dir, _base_config(
            agent="claude_code", tool_mode="restricted", num_judges=1,
        ))

        cmd = mock_run.call_args_list[0][0][0]
        assert "--tools" in cmd
        assert "--strict-mcp-config" in cmd
        assert "--disable-slash-commands" in cmd
        from ai4sci_bench.adapters.claude_code_cli import CLAUDE_CORE_TOOLS
        idx = cmd.index("--tools")
        assert cmd[idx + 1] == CLAUDE_CORE_TOOLS

    @patch("ai4sci_bench.scorers.agent_judge.subprocess.run")
    def test_tool_mode_search_codex(self, mock_run, tmp_path):
        pred_dir, ref_dir = _setup_dirs(tmp_path)

        def side_effect(cmd, **kwargs):
            ws = Path(kwargs["cwd"])
            (ws / "score.json").write_text('{"score": 8}')
            return _make_subprocess_result()

        mock_run.side_effect = side_effect

        scorer = get_scorer("agent_judge")
        scorer.score(pred_dir, ref_dir, _base_config(
            tool_mode="search", num_judges=1,
        ))

        cmd = mock_run.call_args_list[0][0][0]
        assert "--ignore-user-config" in cmd
        cmd_str = " ".join(cmd)
        assert 'web_search="disabled"' not in cmd_str

    @patch("ai4sci_bench.scorers.agent_judge.subprocess.run")
    def test_tool_mode_search_claude(self, mock_run, tmp_path):
        pred_dir, ref_dir = _setup_dirs(tmp_path)

        def side_effect(cmd, **kwargs):
            ws = Path(kwargs["cwd"])
            (ws / "score.json").write_text('{"score": 8}')
            return _make_subprocess_result()

        mock_run.side_effect = side_effect

        scorer = get_scorer("agent_judge")
        scorer.score(pred_dir, ref_dir, _base_config(
            agent="claude_code", tool_mode="search", num_judges=1,
        ))

        cmd = mock_run.call_args_list[0][0][0]
        from ai4sci_bench.adapters.claude_code_cli import CLAUDE_SEARCH_TOOLS
        idx = cmd.index("--tools")
        assert cmd[idx + 1] == CLAUDE_SEARCH_TOOLS


# ---------------------------------------------------------------------------
# Workspace layout
# ---------------------------------------------------------------------------

class TestWorkspaceLayout:
    @patch("ai4sci_bench.scorers.agent_judge.subprocess.run")
    def test_workspace_layout(self, mock_run, tmp_path):
        pred_dir, ref_dir = _setup_dirs(tmp_path)
        workspaces = []

        def side_effect(cmd, **kwargs):
            ws = Path(kwargs["cwd"])
            workspaces.append(ws)
            assert (ws / "JUDGE_INSTRUCTIONS.md").exists()
            assert (ws / "agent_output").is_symlink()
            assert (ws / "reference").is_symlink()
            assert (ws / "agent_output").resolve() == pred_dir.resolve()
            assert (ws / "reference").resolve() == ref_dir.resolve()
            (ws / "score.json").write_text('{"score": 8}')
            return _make_subprocess_result()

        mock_run.side_effect = side_effect

        scorer = get_scorer("agent_judge")
        scorer.score(pred_dir, ref_dir, _base_config(num_judges=1))

        assert len(workspaces) == 1

    @patch("ai4sci_bench.scorers.agent_judge.subprocess.run")
    def test_workspace_independent_per_judge(self, mock_run, tmp_path):
        pred_dir, ref_dir = _setup_dirs(tmp_path)
        workspaces = []

        def side_effect(cmd, **kwargs):
            ws = Path(kwargs["cwd"])
            workspaces.append(ws)
            (ws / "score.json").write_text('{"score": 8}')
            return _make_subprocess_result()

        mock_run.side_effect = side_effect

        scorer = get_scorer("agent_judge")
        scorer.score(pred_dir, ref_dir, _base_config(num_judges=2))

        assert len(workspaces) == 2
        assert workspaces[0] != workspaces[1]

    @patch("ai4sci_bench.scorers.agent_judge.subprocess.run")
    def test_workspace_cleanup(self, mock_run, tmp_path):
        pred_dir, ref_dir = _setup_dirs(tmp_path)
        workspaces = []

        def side_effect(cmd, **kwargs):
            ws = Path(kwargs["cwd"])
            workspaces.append(ws)
            (ws / "score.json").write_text('{"score": 8}')
            return _make_subprocess_result()

        mock_run.side_effect = side_effect

        scorer = get_scorer("agent_judge")
        scorer.score(pred_dir, ref_dir, _base_config(num_judges=1))

        assert len(workspaces) == 1
        assert not workspaces[0].exists()


# ---------------------------------------------------------------------------
# Score extraction
# ---------------------------------------------------------------------------

class TestScoreExtraction:
    @patch("ai4sci_bench.scorers.agent_judge.subprocess.run")
    def test_extract_score_from_json(self, mock_run, tmp_path):
        pred_dir, ref_dir = _setup_dirs(tmp_path)

        def side_effect(cmd, **kwargs):
            ws = Path(kwargs["cwd"])
            (ws / "score.json").write_text('{"score": 9, "reasoning": "excellent"}')
            return _make_subprocess_result(stdout="some agent output")

        mock_run.side_effect = side_effect

        scorer = get_scorer("agent_judge")
        result = scorer.score(pred_dir, ref_dir, _base_config(num_judges=1))

        assert result.details["median_score"] == 9.0
        assert result.details["score_sources"] == ["score.json"]

    @patch("ai4sci_bench.scorers.agent_judge.subprocess.run")
    def test_extract_score_fallback_stdout(self, mock_run, tmp_path):
        pred_dir, ref_dir = _setup_dirs(tmp_path)

        mock_run.return_value = _make_subprocess_result(
            stdout='Here is my evaluation:\n{"score": 7, "reasoning": "good"}'
        )

        scorer = get_scorer("agent_judge")
        result = scorer.score(pred_dir, ref_dir, _base_config(num_judges=1))

        assert result.details["median_score"] == 7.0
        assert result.details["score_sources"] == ["stdout_parse"]

    @patch("ai4sci_bench.scorers.agent_judge.subprocess.run")
    def test_extract_score_invalid_json(self, mock_run, tmp_path):
        pred_dir, ref_dir = _setup_dirs(tmp_path)

        def side_effect(cmd, **kwargs):
            ws = Path(kwargs["cwd"])
            (ws / "score.json").write_text("not valid json {{{")
            return _make_subprocess_result(
                stdout='Fallback: {"score": 6}'
            )

        mock_run.side_effect = side_effect

        scorer = get_scorer("agent_judge")
        result = scorer.score(pred_dir, ref_dir, _base_config(num_judges=1))

        assert result.details["median_score"] == 6.0

    @patch("ai4sci_bench.scorers.agent_judge.subprocess.run")
    def test_extract_score_clamp(self, mock_run, tmp_path):
        pred_dir, ref_dir = _setup_dirs(tmp_path)

        def side_effect(cmd, **kwargs):
            ws = Path(kwargs["cwd"])
            (ws / "score.json").write_text('{"score": 15}')
            return _make_subprocess_result()

        mock_run.side_effect = side_effect

        scorer = get_scorer("agent_judge")
        result = scorer.score(pred_dir, ref_dir, _base_config(
            num_judges=1, max_score_value=10,
        ))

        assert result.details["median_score"] == 10.0

    @patch("ai4sci_bench.scorers.agent_judge.subprocess.run")
    def test_extract_score_clamp_negative(self, mock_run, tmp_path):
        pred_dir, ref_dir = _setup_dirs(tmp_path)

        def side_effect(cmd, **kwargs):
            ws = Path(kwargs["cwd"])
            (ws / "score.json").write_text('{"score": -5}')
            return _make_subprocess_result()

        mock_run.side_effect = side_effect

        scorer = get_scorer("agent_judge")
        result = scorer.score(pred_dir, ref_dir, _base_config(num_judges=1))

        assert result.details["median_score"] == 0.0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    @patch("ai4sci_bench.scorers.agent_judge.subprocess.run")
    def test_timeout_handling(self, mock_run, tmp_path):
        pred_dir, ref_dir = _setup_dirs(tmp_path)
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="codex", timeout=120)

        scorer = get_scorer("agent_judge")
        result = scorer.score(pred_dir, ref_dir, _base_config(num_judges=1))

        assert result.score == 0.0
        assert result.details["parse_failures"] == 1
        assert result.details["score_sources"] == ["timeout"]

    @patch("ai4sci_bench.scorers.agent_judge.subprocess.run")
    def test_crash_handling(self, mock_run, tmp_path):
        pred_dir, ref_dir = _setup_dirs(tmp_path)

        mock_run.return_value = _make_subprocess_result(
            stdout='{"score": 5}', returncode=1,
        )

        scorer = get_scorer("agent_judge")
        result = scorer.score(pred_dir, ref_dir, _base_config(num_judges=1))

        assert result.details["median_score"] == 5.0

    @patch("ai4sci_bench.scorers.agent_judge.subprocess.run")
    def test_all_judges_fail(self, mock_run, tmp_path):
        pred_dir, ref_dir = _setup_dirs(tmp_path)
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="codex", timeout=120)

        scorer = get_scorer("agent_judge")
        result = scorer.score(pred_dir, ref_dir, _base_config(num_judges=3))

        assert result.score == 0.0
        assert result.details["parse_failures"] == 3
        assert result.details["valid_judge_count"] == 0

    def test_unsupported_agent_type(self, tmp_path):
        pred_dir, ref_dir = _setup_dirs(tmp_path)

        scorer = get_scorer("agent_judge")
        with pytest.raises(ValueError, match="Unsupported agent type"):
            scorer.score(pred_dir, ref_dir, _base_config(agent="unknown"))


# ---------------------------------------------------------------------------
# num_judges and aggregation
# ---------------------------------------------------------------------------

class TestNumJudgesAndAggregation:
    @patch("ai4sci_bench.scorers.agent_judge.subprocess.run")
    def test_num_judges_default_3(self, mock_run, tmp_path):
        pred_dir, ref_dir = _setup_dirs(tmp_path)

        def side_effect(cmd, **kwargs):
            ws = Path(kwargs["cwd"])
            (ws / "score.json").write_text('{"score": 8}')
            return _make_subprocess_result()

        mock_run.side_effect = side_effect

        scorer = get_scorer("agent_judge")
        result = scorer.score(pred_dir, ref_dir, _base_config())

        assert mock_run.call_count == 3
        assert result.details["num_judges"] == 3

    @patch("ai4sci_bench.scorers.agent_judge.subprocess.run")
    def test_num_judges_custom(self, mock_run, tmp_path):
        pred_dir, ref_dir = _setup_dirs(tmp_path)

        def side_effect(cmd, **kwargs):
            ws = Path(kwargs["cwd"])
            (ws / "score.json").write_text('{"score": 8}')
            return _make_subprocess_result()

        mock_run.side_effect = side_effect

        scorer = get_scorer("agent_judge")
        result = scorer.score(pred_dir, ref_dir, _base_config(num_judges=5))

        assert mock_run.call_count == 5
        assert result.details["num_judges"] == 5

    @patch("ai4sci_bench.scorers.agent_judge.subprocess.run")
    def test_median_aggregation(self, mock_run, tmp_path):
        pred_dir, ref_dir = _setup_dirs(tmp_path)
        scores_to_return = iter([6, 8, 10])

        def side_effect(cmd, **kwargs):
            ws = Path(kwargs["cwd"])
            s = next(scores_to_return)
            (ws / "score.json").write_text(json.dumps({"score": s}))
            return _make_subprocess_result()

        mock_run.side_effect = side_effect

        scorer = get_scorer("agent_judge")
        result = scorer.score(pred_dir, ref_dir, _base_config(num_judges=3))

        assert result.details["median_score"] == 8.0

    @patch("ai4sci_bench.scorers.agent_judge.subprocess.run")
    def test_partial_failure_median(self, mock_run, tmp_path):
        pred_dir, ref_dir = _setup_dirs(tmp_path)
        call_count = [0]

        def side_effect(cmd, **kwargs):
            ws = Path(kwargs["cwd"])
            call_count[0] += 1
            if call_count[0] == 2:
                raise subprocess.TimeoutExpired(cmd="codex", timeout=120)
            s = 6 if call_count[0] == 1 else 10
            (ws / "score.json").write_text(json.dumps({"score": s}))
            return _make_subprocess_result()

        mock_run.side_effect = side_effect

        scorer = get_scorer("agent_judge")
        result = scorer.score(pred_dir, ref_dir, _base_config(num_judges=3))

        assert result.details["median_score"] == 8.0
        assert result.details["parse_failures"] == 1


# ---------------------------------------------------------------------------
# ScoreDetail extra details
# ---------------------------------------------------------------------------

class TestScoreDetailExtraDetails:
    @patch("ai4sci_bench.scorers.agent_judge.subprocess.run")
    def test_score_detail_extra_details(self, mock_run, tmp_path):
        pred_dir, ref_dir = _setup_dirs(tmp_path)

        def side_effect(cmd, **kwargs):
            ws = Path(kwargs["cwd"])
            (ws / "score.json").write_text('{"score": 8}')
            return _make_subprocess_result()

        mock_run.side_effect = side_effect

        scorer = get_scorer("agent_judge")
        result = scorer.score(pred_dir, ref_dir, _base_config(num_judges=1))

        assert result.details["agent_type"] == "codex"
        assert result.details["tool_mode"] == "unrestricted"
        assert "score_sources" in result.details


# ---------------------------------------------------------------------------
# Encoding safety (issue #30)
# ---------------------------------------------------------------------------

class TestEncodingSafety:
    @patch("ai4sci_bench.scorers.agent_judge.subprocess.run")
    def test_encoding_utf8(self, mock_run, tmp_path):
        pred_dir, ref_dir = _setup_dirs(tmp_path)

        def side_effect(cmd, **kwargs):
            assert kwargs.get("encoding") == "utf-8"
            assert kwargs.get("errors") == "replace"
            ws = Path(kwargs["cwd"])
            (ws / "score.json").write_text('{"score": 8}')
            return _make_subprocess_result()

        mock_run.side_effect = side_effect

        scorer = get_scorer("agent_judge")
        scorer.score(pred_dir, ref_dir, _base_config(num_judges=1))


# ---------------------------------------------------------------------------
# Judge instructions content
# ---------------------------------------------------------------------------

class TestJudgeInstructions:
    @patch("ai4sci_bench.scorers.agent_judge.subprocess.run")
    def test_judge_instructions_content(self, mock_run, tmp_path):
        pred_dir, ref_dir = _setup_dirs(tmp_path)
        rubric_text = "Check if the solution uses Euler method correctly."
        instructions_content = [None]

        def side_effect(cmd, **kwargs):
            ws = Path(kwargs["cwd"])
            content = (ws / "JUDGE_INSTRUCTIONS.md").read_text()
            instructions_content[0] = content
            (ws / "score.json").write_text('{"score": 8}')
            return _make_subprocess_result()

        mock_run.side_effect = side_effect

        scorer = get_scorer("agent_judge")
        scorer.score(pred_dir, ref_dir, _base_config(
            rubric=rubric_text, num_judges=1,
        ))

        content = instructions_content[0]
        assert rubric_text in content
        assert "score.json" in content
        assert "agent_output" in content
        assert "reference" in content


# ---------------------------------------------------------------------------
# Preflight tests
# ---------------------------------------------------------------------------

class TestPreflightAgentJudge:
    def test_preflight_codex_available(self):
        from ai4sci_bench.scorers.agent_judge import preflight_agent_judge
        errors = []
        with patch("ai4sci_bench.scorers.agent_judge.shutil.which", return_value="/usr/bin/codex"):
            preflight_agent_judge({
                "agent": "codex",
                "rubric": "Evaluate the solution on a scale of 0-10.",
                "timeout": 120,
            }, errors)
        assert errors == []

    def test_preflight_codex_missing(self):
        from ai4sci_bench.scorers.agent_judge import preflight_agent_judge
        errors = []
        with patch("ai4sci_bench.scorers.agent_judge.shutil.which", return_value=None):
            preflight_agent_judge({
                "agent": "codex",
                "rubric": "Evaluate the solution.",
                "timeout": 120,
            }, errors)
        assert any("codex CLI not found" in e for e in errors)

    def test_preflight_claude_available(self):
        from ai4sci_bench.scorers.agent_judge import preflight_agent_judge
        errors = []
        with patch("ai4sci_bench.scorers.agent_judge.shutil.which", return_value="/usr/bin/claude"):
            preflight_agent_judge({
                "agent": "claude_code",
                "rubric": "Evaluate the solution.",
                "timeout": 120,
            }, errors)
        assert errors == []

    def test_preflight_empty_rubric(self):
        from ai4sci_bench.scorers.agent_judge import preflight_agent_judge
        errors = []
        with patch("ai4sci_bench.scorers.agent_judge.shutil.which", return_value="/usr/bin/codex"):
            preflight_agent_judge({
                "rubric": "",
                "timeout": 120,
            }, errors)
        assert any("rubric must not be empty" in e for e in errors)

    def test_preflight_timeout_too_low(self):
        from ai4sci_bench.scorers.agent_judge import preflight_agent_judge
        errors = []
        with patch("ai4sci_bench.scorers.agent_judge.shutil.which", return_value="/usr/bin/codex"):
            preflight_agent_judge({
                "rubric": "Evaluate.",
                "timeout": 30,
            }, errors)
        assert any("too low" in e for e in errors)

    def test_preflight_invalid_tool_mode(self):
        from ai4sci_bench.scorers.agent_judge import preflight_agent_judge
        errors = []
        with patch("ai4sci_bench.scorers.agent_judge.shutil.which", return_value="/usr/bin/codex"):
            preflight_agent_judge({
                "rubric": "Evaluate.",
                "timeout": 120,
                "tool_mode": "invalid",
            }, errors)
        assert any("invalid tool_mode" in e for e in errors)

    def test_preflight_num_judges_zero(self):
        from ai4sci_bench.scorers.agent_judge import preflight_agent_judge
        errors = []
        with patch("ai4sci_bench.scorers.agent_judge.shutil.which", return_value="/usr/bin/codex"):
            preflight_agent_judge({
                "rubric": "Evaluate.",
                "timeout": 120,
                "num_judges": 0,
            }, errors)
        assert any("num_judges=0" in e for e in errors)

    def test_preflight_unsupported_agent(self):
        from ai4sci_bench.scorers.agent_judge import preflight_agent_judge
        errors = []
        preflight_agent_judge({
            "agent": "unknown_agent",
            "rubric": "Evaluate.",
            "timeout": 120,
        }, errors)
        assert any("unsupported agent type" in e for e in errors)


# ---------------------------------------------------------------------------
# Regression: existing scorers unchanged
# ---------------------------------------------------------------------------

class TestRegressionExistingScorers:
    def test_llm_judge_unchanged(self):
        scorer = get_scorer("llm_judge")
        assert scorer.name == "llm_judge"

    def test_multimodal_unchanged(self):
        scorer = get_scorer("multimodal")
        assert scorer.name == "multimodal"

    def test_llm_judge_imports_from_common(self):
        from ai4sci_bench.scorers.llm_judge import SUPPORTED_MODELS, DEFAULT_MODEL
        assert "gpt-5.4" in SUPPORTED_MODELS
        assert DEFAULT_MODEL == "gemini/gemini-3.1-pro-preview"

    def test_multimodal_no_longer_imports_llm_judge_private(self):
        import inspect
        import ai4sci_bench.scorers.multimodal as mm
        source = inspect.getsource(mm)
        assert "from ai4sci_bench.scorers.llm_judge import" not in source
