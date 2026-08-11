"""Tests for all issue fixes (#9, #10, #11, #12, #13, #14, #15)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure scorers are registered
import ai4sci_bench.scorers  # noqa: F401


# ── Issue #10: Robust judge response parsing ──────��───────────────────────────

from ai4sci_bench.scorers._parse_utils import parse_judge_json


class TestParseJudgeJson:
    """Test the shared parse_judge_json() utility from _parse_utils."""

    def test_plain_json(self):
        assert parse_judge_json('{"score": 7.5, "reasoning": "ok"}') == 7.5

    def test_json_with_surrounding_text(self):
        raw = 'Based on analysis:\n{"score": 6, "reasoning": "decent"}\nDone.'
        assert parse_judge_json(raw) == 6.0

    def test_fenced_json_block(self):
        raw = 'Here is my evaluation:\n```json\n{"score": 9, "reasoning": "excellent"}\n```'
        assert parse_judge_json(raw) == 9.0

    def test_fenced_json_no_lang_tag(self):
        raw = '```\n{"score": 8, "reasoning": "good"}\n```'
        assert parse_judge_json(raw) == 8.0

    def test_prefix_text_with_braces(self):
        """The key bug from Issue #10 — braces in prefix text."""
        raw = 'Here is my evaluation of the {agent\'s work}:\n\n{"score": 9, "reasoning": "good"}'
        assert parse_judge_json(raw) == 9.0

    def test_multiple_json_objects_picks_score(self):
        raw = '{"context": "test"} and then {"score": 7, "reasoning": "ok"}'
        assert parse_judge_json(raw) == 7.0

    def test_score_fraction_format(self):
        raw = "I give this a 7/10"
        assert parse_judge_json(raw) == 7.0

    def test_score_colon_format(self):
        raw = "Score: 8.5"
        assert parse_judge_json(raw) == 8.5

    def test_score_bold_markdown(self):
        raw = "**Score:** 9"
        assert parse_judge_json(raw) == 9.0

    def test_score_out_of_format(self):
        raw = "I would rate this 8 out of 10"
        assert parse_judge_json(raw) == 8.0

    def test_json_fragment_in_text(self):
        raw = 'The evaluation gives "score": 7 for this work'
        assert parse_judge_json(raw) == 7.0

    def test_clamped_to_max(self):
        raw = '{"score": 15, "reasoning": "overflow"}'
        assert parse_judge_json(raw, max_score=10.0) == 10.0

    def test_clamped_to_zero(self):
        raw = '{"score": -3, "reasoning": "underflow"}'
        assert parse_judge_json(raw, max_score=10.0) == 0.0

    def test_returns_none_on_garbage(self):
        assert parse_judge_json("This is garbage text with no score") is None

    def test_returns_none_on_empty(self):
        assert parse_judge_json("") is None
        assert parse_judge_json("   ") is None

    def test_score_as_string_in_json(self):
        raw = '{"score": "9", "reasoning": "ok"}'
        assert parse_judge_json(raw) == 9.0

    def test_deterministic(self):
        """Same input always gives same output."""
        raw = '{"score": 7, "reasoning": "ok"}'
        results = [parse_judge_json(raw) for _ in range(10)]
        assert all(r == 7.0 for r in results)


class TestLLMJudgeScorerParseFailureHandling:
    """Test that llm_judge properly handles parse failures in aggregation."""

    @patch("ai4sci_bench.scorers.llm_judge.litellm")
    def test_parse_failure_filtered_from_median(self, mock_litellm, tmp_path):
        """When 1/3 judges fail to parse, median should use only valid scores."""
        pred_dir = tmp_path / "pred"
        pred_dir.mkdir()
        (pred_dir / "output.txt").write_text("Agent output")

        mock_litellm.completion.side_effect = [
            _make_response("garbage no score here"),  # parse failure
            _make_response('{"score": 9, "reasoning": "good"}'),
            _make_response('{"score": 8, "reasoning": "ok"}'),
        ]

        scorer = ai4sci_bench.scorers.LLMJudgeScorer()
        result = scorer.score(pred_dir, tmp_path, {
            "pred_file": "output.txt",
            "num_judges": 3,
            "weight": 10.0,
        })

        assert result.details["parse_failures"] == 1
        assert result.details["valid_judge_count"] == 2
        assert result.details["median_score"] == 8.5  # median of [9, 8]
        assert "PARSE_FAILED" in result.details["individual_scores"]

    @patch("ai4sci_bench.scorers.llm_judge.litellm")
    def test_all_parse_failures_gives_zero(self, mock_litellm, tmp_path):
        pred_dir = tmp_path / "pred"
        pred_dir.mkdir()
        (pred_dir / "output.txt").write_text("Agent output")

        mock_litellm.completion.side_effect = [
            _make_response("no score"),
            _make_response("also no score"),
            _make_response("still no score"),
        ]

        scorer = ai4sci_bench.scorers.LLMJudgeScorer()
        result = scorer.score(pred_dir, tmp_path, {
            "pred_file": "output.txt",
            "num_judges": 3,
            "weight": 10.0,
        })

        assert result.details["parse_failures"] == 3
        assert result.details["valid_judge_count"] == 0
        assert result.score == 0.0

    @patch("ai4sci_bench.scorers.llm_judge.litellm")
    def test_raw_responses_preserved(self, mock_litellm, tmp_path):
        pred_dir = tmp_path / "pred"
        pred_dir.mkdir()
        (pred_dir / "output.txt").write_text("Agent output")

        raw_text = '{"score": 7, "reasoning": "good"}'
        mock_litellm.completion.return_value = _make_response(raw_text)

        scorer = ai4sci_bench.scorers.LLMJudgeScorer()
        result = scorer.score(pred_dir, tmp_path, {
            "pred_file": "output.txt",
            "weight": 10.0,
        })

        assert result.details["raw_responses"] == [raw_text]

    @patch("ai4sci_bench.scorers.llm_judge.litellm")
    def test_retry_on_api_error(self, mock_litellm, tmp_path):
        """API errors should be retried before failing."""
        pred_dir = tmp_path / "pred"
        pred_dir.mkdir()
        (pred_dir / "output.txt").write_text("Agent output")

        # First call fails, second succeeds
        mock_litellm.completion.side_effect = [
            Exception("rate limit"),
            _make_response('{"score": 8, "reasoning": "ok"}'),
        ]

        scorer = ai4sci_bench.scorers.LLMJudgeScorer()
        result = scorer.score(pred_dir, tmp_path, {
            "pred_file": "output.txt",
            "weight": 10.0,
        })

        assert result.details["median_score"] == 8.0
        assert mock_litellm.completion.call_count == 2

    def test_read_content_truncation_warning(self, tmp_path):
        scorer = ai4sci_bench.scorers.LLMJudgeScorer()
        (tmp_path / "long.txt").write_text("x" * 20000)
        content = scorer._read_content(tmp_path, "long.txt", max_chars=100)
        assert len(content) > 100  # includes truncation message
        assert "truncated" in content


class TestMultimodalScorerParseFailureHandling:
    """Test multimodal scorer parse failure handling."""

    @patch("ai4sci_bench.scorers.multimodal.litellm")
    def test_vlm_parse_failure_filtered(self, mock_litellm, tmp_path):
        pred_dir = tmp_path / "pred"
        pred_dir.mkdir()
        _create_minimal_png(pred_dir / "plot.png")

        mock_litellm.completion.side_effect = [
            _make_response("no score here"),
            _make_response('{"score": 9, "reasoning": "great"}'),
            _make_response('{"score": 8, "reasoning": "ok"}'),
        ]

        scorer = ai4sci_bench.scorers.MultimodalScorer()
        result = scorer.score(pred_dir, tmp_path, {
            "mode": "vlm_judge",
            "pred_image": "plot.png",
            "num_judges": 3,
            "weight": 10.0,
        })

        assert result.details["parse_failures"] == 1
        assert result.details["valid_judge_count"] == 2
        assert result.details["median_score"] == 8.5

    @patch("ai4sci_bench.scorers.multimodal.litellm")
    def test_vlm_raw_responses_preserved(self, mock_litellm, tmp_path):
        pred_dir = tmp_path / "pred"
        pred_dir.mkdir()
        _create_minimal_png(pred_dir / "plot.png")

        mock_litellm.completion.return_value = _make_response('{"score": 7, "reasoning": "ok"}')

        scorer = ai4sci_bench.scorers.MultimodalScorer()
        result = scorer.score(pred_dir, tmp_path, {
            "mode": "vlm_judge",
            "pred_image": "plot.png",
            "weight": 10.0,
        })

        assert "raw_responses" in result.details
        assert len(result.details["raw_responses"]) == 1


# ── Issue #11: Anonymized task_id ──────────────────���──────────────────────────

class TestAnonymizeTaskId:
    """Test that task_id is anonymized in agent-facing task_info."""

    def test_anonymize_is_deterministic(self):
        from ai4sci_bench.generators.instance_generator import InstanceGenerator
        a = InstanceGenerator._anonymize_task_id("physics.ks_2d_anisotropic")
        b = InstanceGenerator._anonymize_task_id("physics.ks_2d_anisotropic")
        assert a == b

    def test_anonymize_starts_with_task_prefix(self):
        from ai4sci_bench.generators.instance_generator import InstanceGenerator
        result = InstanceGenerator._anonymize_task_id("physics.test")
        assert result.startswith("task_")

    def test_anonymize_hides_original(self):
        from ai4sci_bench.generators.instance_generator import InstanceGenerator
        result = InstanceGenerator._anonymize_task_id("physics.ks_2d_anisotropic")
        assert "ks" not in result
        assert "anisotropic" not in result
        assert "physics" not in result

    def test_different_ids_give_different_hashes(self):
        from ai4sci_bench.generators.instance_generator import InstanceGenerator
        a = InstanceGenerator._anonymize_task_id("physics.task_a")
        b = InstanceGenerator._anonymize_task_id("physics.task_b")
        assert a != b


# ── Issue #12: Codex prompt via stdin ────────────────────���────────────────────

class TestCodexStdin:
    """Test that codex_cli passes prompt via stdin, not command line."""

    def test_build_command_uses_stdin_sentinel(self, tmp_path):
        from ai4sci_bench.adapters.codex_cli import CodexCLIAdapter
        from ai4sci_bench.core.types import TaskInstance, PromptLevel

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "prompt.md").write_text("test prompt content")

        adapter = CodexCLIAdapter()
        instance = TaskInstance(
            task_id="test",
            instance_id="test_id",
            task_dir=tmp_path,
            workspace_dir=workspace,
            reference_dir=tmp_path,
            prompt_level=PromptLevel.B1,
            parameters={},
            metadata={"output": {"files": []}},
        )
        cmd = adapter._build_command(instance, None)

        # The last args should be ["--", "-"] (stdin sentinel)
        assert cmd[-2:] == ["--", "-"]
        # prompt should NOT appear as a command-line argument
        assert "test prompt content" not in cmd

    def test_get_stdin_input_returns_prompt(self, tmp_path):
        from ai4sci_bench.adapters.codex_cli import CodexCLIAdapter
        from ai4sci_bench.core.types import TaskInstance, PromptLevel

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "prompt.md").write_text("test prompt content")

        adapter = CodexCLIAdapter()
        instance = TaskInstance(
            task_id="test",
            instance_id="test_id",
            task_dir=tmp_path,
            workspace_dir=workspace,
            reference_dir=tmp_path,
            prompt_level=PromptLevel.B1,
            parameters={},
            metadata={"output": {"files": []}},
        )
        adapter._build_command(instance, None)
        stdin = adapter._get_stdin_input(instance, None)
        assert stdin == "test prompt content"

    def test_get_stdin_input_idempotent(self, tmp_path):
        from ai4sci_bench.adapters.codex_cli import CodexCLIAdapter
        from ai4sci_bench.core.types import TaskInstance, PromptLevel

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "prompt.md").write_text("prompt")

        adapter = CodexCLIAdapter()
        instance = TaskInstance(
            task_id="test",
            instance_id="test_id",
            task_dir=tmp_path,
            workspace_dir=workspace,
            reference_dir=tmp_path,
            prompt_level=PromptLevel.B1,
            parameters={},
            metadata={"output": {"files": []}},
        )
        # Thread-safe: repeated calls return the same result
        assert adapter._get_stdin_input(instance, None) == "prompt"
        assert adapter._get_stdin_input(instance, None) == "prompt"


# ── Issue #27: Claude Code prompt via stdin (Windows argv length) ─────────────

class TestClaudeCodeCliStdin:
    """Test that claude_code_cli passes prompt via stdin, not command line.

    Long (≳ 8 KB) B1 prompts were failing on Windows with "参数太长" because
    the adapter appended the full prompt as a trailing argv element and
    ``cmd.exe`` / ``CreateProcess`` enforces an ~8191-char command-line
    limit. Routing the prompt through stdin removes the argv payload.
    """

    def _make_instance(self, tmp_path, prompt_text):
        from ai4sci_bench.core.types import PromptLevel, TaskInstance

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "prompt.md").write_text(prompt_text, encoding="utf-8")
        return TaskInstance(
            task_id="test",
            instance_id="test_id",
            task_dir=tmp_path,
            workspace_dir=workspace,
            reference_dir=tmp_path,
            prompt_level=PromptLevel.B1,
            parameters={},
            metadata={"output": {"files": []}},
        )

    # ── argv-shape contract ──────────────────────────────────────────────

    def test_build_command_omits_prompt_from_argv(self, tmp_path):
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter

        adapter = ClaudeCodeCLIAdapter()
        instance = self._make_instance(tmp_path, "test prompt content")
        cmd = adapter._build_command(instance, None)

        # The prompt must not appear anywhere in argv (was previously ["--", prompt]).
        assert "test prompt content" not in cmd
        # And the argv must not end with ``--`` followed by the prompt tail.
        assert "--" not in cmd

    def test_build_command_survives_8kb_prompt(self, tmp_path):
        """Prompts ≥ 8 KB must not make it into the command-line argv."""
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter

        big_prompt = "x" * 9000  # just over cmd.exe's ~8191-char ceiling
        adapter = ClaudeCodeCLIAdapter()
        instance = self._make_instance(tmp_path, big_prompt)
        cmd = adapter._build_command(instance, None)

        # Total argv length should stay well below the Windows limit because
        # the prompt is now piped via stdin.
        assert sum(len(a) for a in cmd) < 2000
        assert big_prompt not in cmd
        assert adapter._get_stdin_input(instance, None) == big_prompt

    def test_build_command_survives_50kb_prompt(self, tmp_path):
        """Extreme-length prompt (far beyond any argv limit) still routes via stdin."""
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter

        huge_prompt = "y" * 50_000
        adapter = ClaudeCodeCLIAdapter()
        instance = self._make_instance(tmp_path, huge_prompt)
        cmd = adapter._build_command(instance, None)

        # Even 50 KB prompts must not leak into argv.
        assert sum(len(a) for a in cmd) < 2000
        assert huge_prompt not in cmd
        assert adapter._get_stdin_input(instance, None) == huge_prompt

    def test_cjk_and_emoji_prompt_preserved_via_stdin(self, tmp_path):
        """UTF-8 content (中文 / emoji) round-trips through stdin without mangling."""
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter

        prompt = "你好世界 🌍 — Solve task with UTF-8 chars. 日本語テスト 한국어 тест"
        adapter = ClaudeCodeCLIAdapter()
        instance = self._make_instance(tmp_path, prompt)
        adapter._build_command(instance, None)

        assert adapter._get_stdin_input(instance, None) == prompt

    def test_empty_prompt_still_routes_through_stdin(self, tmp_path):
        """An empty prompt should still be handed via stdin, not omitted."""
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter

        adapter = ClaudeCodeCLIAdapter()
        instance = self._make_instance(tmp_path, "")
        cmd = adapter._build_command(instance, None)

        assert "--" not in cmd
        # Empty string is distinct from None — the consumer must see "".
        assert adapter._get_stdin_input(instance, None) == ""

    def test_get_stdin_input_returns_prompt(self, tmp_path):
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter

        adapter = ClaudeCodeCLIAdapter()
        instance = self._make_instance(tmp_path, "test prompt content")

        assert adapter._get_stdin_input(instance, None) == "test prompt content"

    def test_get_stdin_input_idempotent(self, tmp_path):
        """Thread-safe: repeated calls with same instance return same result."""
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter

        adapter = ClaudeCodeCLIAdapter()
        instance = self._make_instance(tmp_path, "prompt")
        assert adapter._get_stdin_input(instance, None) == "prompt"
        assert adapter._get_stdin_input(instance, None) == "prompt"

    def test_get_stdin_input_returns_none_without_instance(self):
        """Without task_instance, stdin returns None."""
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter

        adapter = ClaudeCodeCLIAdapter()
        assert adapter._get_stdin_input() is None

    # ── Compatibility with existing adapter options ──────────────────────

    def test_permission_mode_still_forwarded_after_stdin_fix(self, tmp_path):
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter

        adapter = ClaudeCodeCLIAdapter(permission_mode="bypassPermissions")
        instance = self._make_instance(tmp_path, "prompt")
        cmd = adapter._build_command(instance, None)

        assert "--permission-mode" in cmd
        assert cmd[cmd.index("--permission-mode") + 1] == "bypassPermissions"

    def test_disallowed_tools_still_applied_after_stdin_fix(self, tmp_path):
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter

        adapter = ClaudeCodeCLIAdapter(allow_external_tools=False)
        instance = self._make_instance(tmp_path, "prompt")
        cmd = adapter._build_command(instance, None)

        assert "--tools" in cmd
        tools = cmd[cmd.index("--tools") + 1].split(",")
        assert "WebSearch" not in tools
        assert "WebFetch" not in tools
        assert "--strict-mcp-config" in cmd
        assert "--disable-slash-commands" in cmd

    def test_external_tools_allowed_leaves_disallowed_flag_off(self, tmp_path):
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter

        adapter = ClaudeCodeCLIAdapter(allow_external_tools=True)
        instance = self._make_instance(tmp_path, "prompt")
        cmd = adapter._build_command(instance, None)

        assert "--disallowed-tools" not in cmd

    # ── End-to-end: solve() actually pipes via subprocess stdin ─────────

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_solve_pipes_prompt_to_subprocess_stdin(self, mock_run, tmp_path):
        """solve() must pass the prompt via subprocess.run(input=...) not argv."""
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        adapter = ClaudeCodeCLIAdapter()
        instance = self._make_instance(tmp_path, "x" * 10_000)
        adapter.solve(instance)

        call = mock_run.call_args
        argv = call.args[0]
        stdin_input = call.kwargs.get("input")

        # Prompt MUST arrive via stdin, never argv — this is the Issue #27 contract.
        assert stdin_input == "x" * 10_000
        assert "x" * 10_000 not in argv
        # Windows-facing argv length stays tiny after the fix.
        assert sum(len(a) for a in argv) < 2000

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_solve_second_call_re_reads_prompt_from_workspace(
        self, mock_run, tmp_path
    ):
        """solve() twice on the same adapter must not lose the second prompt.

        Guards against a regression where the second call would lose the
        prompt (e.g. stale state from a previous call).
        """
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        adapter = ClaudeCodeCLIAdapter()
        instance = self._make_instance(tmp_path, "first prompt payload")
        adapter.solve(instance)
        # Rewrite prompt.md and run again — the adapter should always pick up
        # the current workspace contents via _get_stdin_input(task_instance).
        (instance.workspace_dir / "prompt.md").write_text(
            "second prompt payload", encoding="utf-8"
        )
        adapter.solve(instance)

        second_call = mock_run.call_args
        assert second_call.kwargs.get("input") == "second prompt payload"
        assert "second prompt payload" not in second_call.args[0]

    # ── Scope marker: OS sandbox path is intentionally unchanged ────────

    def test_os_sandbox_path_still_uses_argv_by_design(self, tmp_path):
        """`_build_os_agent_cmd` keeps prompt-in-argv on purpose.

        The Issue #27 reporter confirmed ``--sandbox os`` is unaffected
        because ``docker.exe`` uses ``CreateProcessW`` (32K ceiling) rather
        than ``cmd.exe`` (8K). This test pins that scope decision so a
        future refactor does not accidentally change OS sandbox behavior
        without an explicit issue of its own.
        """
        from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "prompt.md").write_text("os-sandbox prompt body", encoding="utf-8")

        adapter = ClaudeCodeCLIAdapter()
        cmd = adapter._build_os_agent_cmd(workspace)

        # The OS-sandbox path still ends with ``--`` + prompt as argv.
        assert cmd[-2] == "--"
        assert cmd[-1] == "os-sandbox prompt body"


# ── Issue #13: Workspace temp-dir isolation ───────────────────────────────────

class TestWorkspaceTempDirIsolation:
    """Test that workspaces are isolated in temp directories."""

    def test_different_prompt_levels_in_separate_dirs(self, tmp_path):
        """B1 and B3 workspaces must not be siblings."""
        from ai4sci_bench.generators.instance_generator import InstanceGenerator
        from ai4sci_bench.core.types import PromptLevel

        gen = InstanceGenerator(tmp_path)
        ws_b1 = gen._create_isolated_workspace(tmp_path, "test_instance", PromptLevel.B1)
        ws_b3 = gen._create_isolated_workspace(tmp_path, "test_instance", PromptLevel.B3)

        # They should be in different parent directories
        assert ws_b1.parent != ws_b3.parent
        # Neither should be able to see the other via parent traversal
        assert not (ws_b1.parent / "workspace_b3").exists()
        assert not (ws_b3.parent / "workspace_b1").exists()

    def test_workspace_is_in_temp_dir(self, tmp_path):
        from ai4sci_bench.generators.instance_generator import InstanceGenerator
        from ai4sci_bench.core.types import PromptLevel

        gen = InstanceGenerator(tmp_path)
        ws = gen._create_isolated_workspace(tmp_path, "test_instance", PromptLevel.B1)

        # Should be under system temp, not under output_dir
        import tempfile
        assert str(ws).startswith(tempfile.gettempdir()) or "ai4sci_ws_" in str(ws)

    def test_debug_symlink_exists(self, tmp_path):
        from ai4sci_bench.generators.instance_generator import InstanceGenerator
        from ai4sci_bench.core.types import PromptLevel

        gen = InstanceGenerator(tmp_path)
        ws = gen._create_isolated_workspace(tmp_path, "test_instance", PromptLevel.B1)

        link_path = tmp_path / "_workspaces" / "test_instance" / "workspace_b1"
        # Either a symlink or a .path file should exist
        assert link_path.is_symlink() or (link_path.parent / "workspace_b1.path").exists()


# ── Issue #14: PowerShell quote fallback ────────────────────────���─────────────

class TestParseAgentConfig:
    """Test YAML fallback for PowerShell-stripped quotes."""

    def test_valid_json(self):
        from ai4sci_bench.cli import _parse_agent_config
        result = _parse_agent_config('{"model": "gpt-5.4"}')
        assert result == {"model": "gpt-5.4"}

    def test_empty_json(self):
        from ai4sci_bench.cli import _parse_agent_config
        result = _parse_agent_config('{}')
        assert result == {}

    def test_unquoted_keys_yaml_fallback(self):
        from ai4sci_bench.cli import _parse_agent_config
        # PowerShell strips quotes: {model:gpt-5.4}
        result = _parse_agent_config('{model: gpt-5.4}')
        assert result == {"model": "gpt-5.4"}

    def test_unquoted_boolean_yaml_fallback(self):
        from ai4sci_bench.cli import _parse_agent_config
        result = _parse_agent_config('{allow_external_tools: true}')
        assert result == {"allow_external_tools": True}

    def test_garbage_raises_error(self):
        import click
        from ai4sci_bench.cli import _parse_agent_config
        with pytest.raises(click.BadParameter, match="Cannot parse"):
            _parse_agent_config("not valid at all [[[")


# ── Issue #15: Codex absolute path ───────────────────────────────────��────────

class TestCodexAbsolutePath:
    """Test that codex_cli always uses absolute paths for --cd."""

    def test_build_command_uses_absolute_path(self, tmp_path):
        from ai4sci_bench.adapters.codex_cli import CodexCLIAdapter
        from ai4sci_bench.core.types import TaskInstance, PromptLevel

        # Create a workspace with a relative-like path
        workspace = tmp_path / "rel" / "workspace"
        workspace.mkdir(parents=True)
        (workspace / "prompt.md").write_text("test prompt")

        adapter = CodexCLIAdapter()
        instance = TaskInstance(
            task_id="test",
            instance_id="test_id",
            task_dir=tmp_path,
            workspace_dir=workspace,
            reference_dir=tmp_path,
            prompt_level=PromptLevel.B1,
            parameters={},
            metadata={"output": {"files": []}},
        )
        cmd = adapter._build_command(instance, None)

        # Find the --cd argument value
        cd_idx = cmd.index("--cd")
        cd_path = cmd[cd_idx + 1]
        assert os.path.isabs(cd_path), f"--cd should be absolute, got: {cd_path}"

    def test_get_cwd_returns_absolute(self, tmp_path):
        from ai4sci_bench.adapters.subprocess_base import SubprocessAgentAdapter
        from ai4sci_bench.core.types import TaskInstance, PromptLevel

        class DummyAdapter(SubprocessAgentAdapter):
            def _build_command(self, ti, te):
                return ["echo"]

        adapter = DummyAdapter()
        instance = TaskInstance(
            task_id="test",
            instance_id="test_id",
            task_dir=tmp_path,
            workspace_dir=tmp_path / "workspace",
            reference_dir=tmp_path,
            prompt_level=PromptLevel.B1,
            parameters={},
            metadata={},
        )
        (tmp_path / "workspace").mkdir(exist_ok=True)
        cwd = adapter._get_cwd(instance)
        assert cwd.is_absolute()


# ── Issue #9: Windows path length check ───────────────────────────────────────

class TestWindowsPathLengthCheck:
    """Test Windows MAX_PATH preflight check."""

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific test")
    def test_long_path_raises_on_windows(self, tmp_path):
        from ai4sci_bench.generators.instance_generator import InstanceGenerator
        from ai4sci_bench.core.types import PromptLevel

        gen = InstanceGenerator(tmp_path)
        # Create instance_dir with long data file
        instance_dir = tmp_path / "instances" / "long_instance_id"
        instance_dir.mkdir(parents=True)
        data_dir = instance_dir / "data"
        data_dir.mkdir()
        (data_dir / ("a" * 200 + ".npy")).write_bytes(b"data")

        workspace_dir = tmp_path / "workspace"
        workspace_dir.mkdir()

        with pytest.raises(RuntimeError, match="MAX_PATH"):
            gen._prepare_workspace(
                instance_dir, workspace_dir,
                "long_instance_id", PromptLevel.B1,
                {"id": "test", "output": {"files": []}}, {},
            )

    def test_short_path_passes(self, tmp_path):
        """Non-Windows or short paths should not raise."""
        from ai4sci_bench.generators.instance_generator import InstanceGenerator
        from ai4sci_bench.core.types import PromptLevel

        gen = InstanceGenerator(tmp_path)
        instance_dir = tmp_path / "instances" / "short_id"
        instance_dir.mkdir(parents=True)
        data_dir = instance_dir / "data"
        data_dir.mkdir()
        (data_dir / "input.npy").write_bytes(b"data")
        (instance_dir / "prompt_b1.md").write_text("prompt")

        workspace_dir = tmp_path / "workspace"
        workspace_dir.mkdir()

        # Should not raise
        gen._prepare_workspace(
            instance_dir, workspace_dir,
            "short_id", PromptLevel.B1,
            {"id": "test", "output": {"files": [{"name": "output.npy", "type": "data"}]}},
            {},
        )
        assert (workspace_dir / "data" / "input.npy").exists()


# ── Helpers ────────────────────────────────���────────────────────���─────────────

def _make_response(content: str) -> MagicMock:
    """Create a mock litellm response with given content."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def _create_minimal_png(path: Path) -> None:
    """Create a minimal valid PNG file."""
    import struct
    import zlib

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + c + crc

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    raw_data = b"\x00\xff\x00\x00"
    idat = _chunk(b"IDAT", zlib.compress(raw_data))
    iend = _chunk(b"IEND", b"")

    path.write_bytes(signature + ihdr + idat + iend)
