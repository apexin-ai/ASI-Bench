"""Comprehensive tool isolation tests.

These tests are designed to be "mutation-proof" — if any single line of
tool isolation logic is removed or changed, at least one test here must fail.
They cover:

1. Exact command output for every ToolMode × adapter × path combination
2. Consistency between _build_command() and _build_os_agent_cmd()
3. Negative tests (forbidden flags/tools must NOT appear)
4. Boundary conditions (ToolMode resolution, invalid inputs)
5. Full CLI→adapter→command integration path
6. Provenance metadata correctness
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai4sci_bench.adapters.claude_code_cli import (
    CLAUDE_CORE_TOOLS,
    CLAUDE_SEARCH_TOOLS,
    ClaudeCodeCLIAdapter,
)
from ai4sci_bench.adapters.codex_cli import (
    CODEX_RESTRICTED_DISABLE_FEATURES,
    OS_SANDBOX_PYTHON,
    CodexCLIAdapter,
)
from ai4sci_bench.cli import (
    _build_agent,
    _build_agent_metadata,
    _resolve_tool_mode,
)
from ai4sci_bench.core.types import ToolMode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "prompt.md").write_text("test prompt")
    return ws


# ---------------------------------------------------------------------------
# ToolMode resolution
# ---------------------------------------------------------------------------


class TestToolModeResolution:
    """Verify ToolMode is correctly resolved from various input combinations."""

    def test_default_is_restricted(self):
        assert ClaudeCodeCLIAdapter().tool_mode == ToolMode.RESTRICTED
        assert CodexCLIAdapter().tool_mode == ToolMode.RESTRICTED

    def test_allow_external_tools_true_gives_search(self):
        assert ClaudeCodeCLIAdapter(allow_external_tools=True).tool_mode == ToolMode.SEARCH
        assert CodexCLIAdapter(allow_external_tools=True).tool_mode == ToolMode.SEARCH

    def test_explicit_tool_mode_string(self):
        for mode_str in ("restricted", "search", "unrestricted"):
            assert ClaudeCodeCLIAdapter(tool_mode=mode_str).tool_mode == ToolMode(mode_str)
            assert CodexCLIAdapter(tool_mode=mode_str).tool_mode == ToolMode(mode_str)

    def test_explicit_tool_mode_enum(self):
        assert ClaudeCodeCLIAdapter(tool_mode=ToolMode.SEARCH).tool_mode == ToolMode.SEARCH
        assert CodexCLIAdapter(tool_mode=ToolMode.UNRESTRICTED).tool_mode == ToolMode.UNRESTRICTED

    def test_explicit_overrides_allow_external_tools(self):
        a = ClaudeCodeCLIAdapter(allow_external_tools=True, tool_mode="restricted")
        assert a.tool_mode == ToolMode.RESTRICTED
        b = CodexCLIAdapter(allow_external_tools=False, tool_mode="search")
        assert b.tool_mode == ToolMode.SEARCH

    def test_invalid_tool_mode_raises(self):
        with pytest.raises(ValueError):
            ClaudeCodeCLIAdapter(tool_mode="invalid")
        with pytest.raises(ValueError):
            CodexCLIAdapter(tool_mode="bogus")


# ---------------------------------------------------------------------------
# Claude Code: exact command verification
# ---------------------------------------------------------------------------


class TestClaudeToolIsolationExact:
    """Verify the EXACT flags produced by Claude Code for each mode."""

    # -- Restricted mode --

    def test_restricted_os_cmd_exact_tools(self, workspace):
        adapter = ClaudeCodeCLIAdapter()
        cmd = adapter._build_os_agent_cmd(workspace)
        idx = cmd.index("--tools")
        assert cmd[idx + 1] == CLAUDE_CORE_TOOLS

    def test_restricted_os_cmd_has_strict_mcp(self, workspace):
        adapter = ClaudeCodeCLIAdapter()
        cmd = adapter._build_os_agent_cmd(workspace)
        assert "--strict-mcp-config" in cmd

    def test_restricted_os_cmd_has_disable_slash(self, workspace):
        adapter = ClaudeCodeCLIAdapter()
        cmd = adapter._build_os_agent_cmd(workspace)
        assert "--disable-slash-commands" in cmd

    def test_restricted_os_cmd_no_disallowed_tools(self, workspace):
        adapter = ClaudeCodeCLIAdapter()
        cmd = adapter._build_os_agent_cmd(workspace)
        assert "--disallowed-tools" not in cmd

    # -- Search mode --

    def test_search_os_cmd_exact_tools(self, workspace):
        adapter = ClaudeCodeCLIAdapter(tool_mode="search")
        cmd = adapter._build_os_agent_cmd(workspace)
        idx = cmd.index("--tools")
        assert cmd[idx + 1] == CLAUDE_SEARCH_TOOLS

    def test_search_os_cmd_has_strict_mcp(self, workspace):
        adapter = ClaudeCodeCLIAdapter(tool_mode="search")
        cmd = adapter._build_os_agent_cmd(workspace)
        assert "--strict-mcp-config" in cmd

    def test_search_os_cmd_has_disable_slash(self, workspace):
        adapter = ClaudeCodeCLIAdapter(tool_mode="search")
        cmd = adapter._build_os_agent_cmd(workspace)
        assert "--disable-slash-commands" in cmd

    # -- Unrestricted mode --

    def test_unrestricted_os_cmd_no_tools_flag(self, workspace):
        adapter = ClaudeCodeCLIAdapter(tool_mode="unrestricted")
        cmd = adapter._build_os_agent_cmd(workspace)
        assert "--tools" not in cmd

    def test_unrestricted_os_cmd_no_strict_mcp(self, workspace):
        adapter = ClaudeCodeCLIAdapter(tool_mode="unrestricted")
        cmd = adapter._build_os_agent_cmd(workspace)
        assert "--strict-mcp-config" not in cmd

    def test_unrestricted_os_cmd_no_disable_slash(self, workspace):
        adapter = ClaudeCodeCLIAdapter(tool_mode="unrestricted")
        cmd = adapter._build_os_agent_cmd(workspace)
        assert "--disable-slash-commands" not in cmd

    # -- Tool whitelist content --

    def test_core_tools_exactly_15(self):
        tools = CLAUDE_CORE_TOOLS.split(",")
        assert len(tools) == 15

    def test_search_tools_exactly_17(self):
        tools = CLAUDE_SEARCH_TOOLS.split(",")
        assert len(tools) == 17

    def test_search_tools_superset_of_core(self):
        core = set(CLAUDE_CORE_TOOLS.split(","))
        search = set(CLAUDE_SEARCH_TOOLS.split(","))
        assert core.issubset(search)
        assert search - core == {"WebSearch", "WebFetch"}

    @pytest.mark.parametrize("forbidden", [
        "WebSearch", "WebFetch", "Skill", "AskUserQuestion",
        "EnterPlanMode", "ExitPlanMode", "ScheduleWakeup",
        "CronCreate", "CronDelete", "CronList",
        "PushNotification", "RemoteTrigger", "ShareOnboardingGuide",
    ])
    def test_core_tools_excludes_forbidden(self, forbidden):
        assert forbidden not in CLAUDE_CORE_TOOLS.split(",")

    @pytest.mark.parametrize("required", [
        "Bash", "Read", "Write", "Edit", "Glob", "Grep",
        "TodoWrite", "Agent", "NotebookEdit", "ToolSearch",
        "Monitor", "TaskOutput", "TaskStop",
        "EnterWorktree", "ExitWorktree",
    ])
    def test_core_tools_includes_required(self, required):
        assert required in CLAUDE_CORE_TOOLS.split(",")


# ---------------------------------------------------------------------------
# Claude Code: cross-path consistency
# ---------------------------------------------------------------------------


class TestClaudeCrossPathConsistency:
    """_build_command() must produce the same isolation as _build_os_agent_cmd()."""

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def _get_build_command_cmd(self, mock_run, workspace, mode):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        from ai4sci_bench.core.types import TaskInstance, PromptLevel
        ti = TaskInstance(
            task_id="t", instance_id="i", task_dir=workspace.parent,
            workspace_dir=workspace, reference_dir=workspace.parent / "ref",
            prompt_level=PromptLevel.B2, parameters={},
            metadata={"output": {"files": [{"name": "output.npy", "type": "numpy"}]}},
        )
        adapter = ClaudeCodeCLIAdapter(tool_mode=mode)
        adapter.solve(ti)
        return mock_run.call_args.args[0]

    @pytest.mark.parametrize("mode", ["restricted", "search", "unrestricted"])
    def test_tools_flag_consistent(self, workspace, mode):
        os_cmd = ClaudeCodeCLIAdapter(tool_mode=mode)._build_os_agent_cmd(workspace)
        local_cmd = self._get_build_command_cmd(workspace=workspace, mode=mode)

        os_has_tools = "--tools" in os_cmd
        local_has_tools = "--tools" in local_cmd
        assert os_has_tools == local_has_tools

        if os_has_tools:
            os_tools = os_cmd[os_cmd.index("--tools") + 1]
            local_tools = local_cmd[local_cmd.index("--tools") + 1]
            assert os_tools == local_tools

    @pytest.mark.parametrize("mode", ["restricted", "search", "unrestricted"])
    def test_mcp_flag_consistent(self, workspace, mode):
        os_cmd = ClaudeCodeCLIAdapter(tool_mode=mode)._build_os_agent_cmd(workspace)
        local_cmd = self._get_build_command_cmd(workspace=workspace, mode=mode)
        assert ("--strict-mcp-config" in os_cmd) == ("--strict-mcp-config" in local_cmd)

    @pytest.mark.parametrize("mode", ["restricted", "search", "unrestricted"])
    def test_slash_flag_consistent(self, workspace, mode):
        os_cmd = ClaudeCodeCLIAdapter(tool_mode=mode)._build_os_agent_cmd(workspace)
        local_cmd = self._get_build_command_cmd(workspace=workspace, mode=mode)
        assert ("--disable-slash-commands" in os_cmd) == ("--disable-slash-commands" in local_cmd)


# ---------------------------------------------------------------------------
# Codex: exact command verification
# ---------------------------------------------------------------------------


class TestCodexToolIsolationExact:
    """Verify the EXACT flags produced by Codex for each mode."""

    # -- Restricted mode --

    def test_restricted_os_cmd_has_ignore_user_config(self, workspace):
        adapter = CodexCLIAdapter()
        cmd = adapter._build_os_agent_cmd(workspace)
        assert "--ignore-user-config" in cmd

    def test_restricted_os_cmd_disables_all_required_features(self, workspace):
        adapter = CodexCLIAdapter()
        cmd = adapter._build_os_agent_cmd(workspace)
        disabled = [cmd[i + 1] for i, v in enumerate(cmd) if v == "--disable"]
        for feature in CODEX_RESTRICTED_DISABLE_FEATURES:
            assert feature in disabled, f"missing --disable {feature}"

    def test_restricted_os_cmd_disables_web_search(self, workspace):
        adapter = CodexCLIAdapter()
        cmd = adapter._build_os_agent_cmd(workspace)
        idx = cmd.index("--config")
        assert cmd[idx + 1] == 'web_search="disabled"'

    def test_restricted_os_cmd_does_not_disable_multi_agent(self, workspace):
        adapter = CodexCLIAdapter()
        cmd = adapter._build_os_agent_cmd(workspace)
        disabled = [cmd[i + 1] for i, v in enumerate(cmd) if v == "--disable"]
        assert "multi_agent" not in disabled

    def test_os_cmd_can_inject_explicit_task_python_helper(self, workspace):
        adapter = CodexCLIAdapter()
        cmd = adapter._build_os_agent_cmd(
            workspace,
            python_executable=OS_SANDBOX_PYTHON,
        )

        prompt = cmd[-1]
        helper = workspace / "_ai4sci_task_python"
        assert helper.exists()
        assert f'exec "{OS_SANDBOX_PYTHON}" "$@"' in helper.read_text()
        assert f"runs `{OS_SANDBOX_PYTHON}`" in prompt
        assert "Do not rely on bare `python` or `python3`" in prompt

    # -- Search mode --

    def test_search_os_cmd_no_web_search_disabled(self, workspace):
        adapter = CodexCLIAdapter(tool_mode="search")
        cmd = adapter._build_os_agent_cmd(workspace)
        assert 'web_search="disabled"' not in " ".join(cmd)

    def test_search_os_cmd_still_disables_features(self, workspace):
        adapter = CodexCLIAdapter(tool_mode="search")
        cmd = adapter._build_os_agent_cmd(workspace)
        disabled = [cmd[i + 1] for i, v in enumerate(cmd) if v == "--disable"]
        for feature in CODEX_RESTRICTED_DISABLE_FEATURES:
            assert feature in disabled, f"search mode missing --disable {feature}"

    def test_search_os_cmd_has_ignore_user_config(self, workspace):
        adapter = CodexCLIAdapter(tool_mode="search")
        cmd = adapter._build_os_agent_cmd(workspace)
        assert "--ignore-user-config" in cmd

    # -- Unrestricted mode --

    def test_unrestricted_os_cmd_no_disable(self, workspace):
        adapter = CodexCLIAdapter(tool_mode="unrestricted")
        cmd = adapter._build_os_agent_cmd(workspace)
        disabled = [cmd[i + 1] for i, v in enumerate(cmd) if v == "--disable"]
        assert disabled == []

    def test_unrestricted_os_cmd_no_ignore_user_config(self, workspace):
        adapter = CodexCLIAdapter(tool_mode="unrestricted")
        cmd = adapter._build_os_agent_cmd(workspace)
        assert "--ignore-user-config" not in cmd

    def test_unrestricted_os_cmd_no_web_search_disabled(self, workspace):
        adapter = CodexCLIAdapter(tool_mode="unrestricted")
        cmd = adapter._build_os_agent_cmd(workspace)
        assert 'web_search="disabled"' not in " ".join(cmd)

    # -- --full-auto removal --

    def test_build_command_never_uses_full_auto(self, workspace):
        from ai4sci_bench.core.types import TaskInstance, PromptLevel
        ti = TaskInstance(
            task_id="t", instance_id="i", task_dir=workspace.parent,
            workspace_dir=workspace, reference_dir=workspace.parent / "ref",
            prompt_level=PromptLevel.B2, parameters={},
            metadata={"output": {"files": []}},
        )
        for full_auto in (True, False):
            adapter = CodexCLIAdapter(full_auto=full_auto)
            cmd = adapter._build_command(ti, task_env=None)
            assert "--full-auto" not in cmd

    def test_build_command_task_sandbox_uses_workspace_write(self, workspace):
        from ai4sci_bench.core.types import TaskInstance, PromptLevel
        ti = TaskInstance(
            task_id="t", instance_id="i", task_dir=workspace.parent,
            workspace_dir=workspace, reference_dir=workspace.parent / "ref",
            prompt_level=PromptLevel.B2, parameters={},
            metadata={"output": {"files": []}},
        )
        adapter = CodexCLIAdapter()
        adapter.setup({"sandbox": "task"})
        cmd = adapter._build_command(ti, task_env=None)
        idx = cmd.index("--sandbox")
        assert cmd[idx + 1] == "workspace-write"

    @pytest.mark.parametrize("sandbox", ["none", "linux_ns"])
    def test_build_command_outer_sandbox_uses_bypass(self, workspace, sandbox):
        from ai4sci_bench.core.types import TaskInstance, PromptLevel
        ti = TaskInstance(
            task_id="t", instance_id="i", task_dir=workspace.parent,
            workspace_dir=workspace, reference_dir=workspace.parent / "ref",
            prompt_level=PromptLevel.B2, parameters={},
            metadata={"output": {"files": []}},
        )
        adapter = CodexCLIAdapter()
        adapter.sandbox = sandbox
        cmd = adapter._build_command(ti, task_env=None)
        assert "--dangerously-bypass-approvals-and-sandbox" in cmd
        assert "--sandbox" not in cmd

    # -- Feature list completeness --

    @pytest.mark.parametrize("feature", [
        "plugins",
        "tool_call_mcp_elicitation",
        "skill_mcp_dependency_install",
        "image_generation",
        "codex_hooks",
        "computer_use",
    ])
    def test_feature_in_disable_list(self, feature):
        assert feature in CODEX_RESTRICTED_DISABLE_FEATURES

    def test_multi_agent_not_in_disable_list(self):
        assert "multi_agent" not in CODEX_RESTRICTED_DISABLE_FEATURES

    def test_disable_list_has_exactly_6_features(self):
        assert len(CODEX_RESTRICTED_DISABLE_FEATURES) == 6


# ---------------------------------------------------------------------------
# Codex: cross-path consistency
# ---------------------------------------------------------------------------


class TestCodexCrossPathConsistency:
    """_build_command() and _build_os_agent_cmd() isolation must be identical."""

    def _extract_isolation_flags(self, cmd):
        """Extract isolation-relevant flags from a command list."""
        disabled = sorted(cmd[i + 1] for i, v in enumerate(cmd) if v == "--disable")
        has_ignore_user_config = "--ignore-user-config" in cmd
        web_search_disabled = '--config' in cmd and 'web_search="disabled"' in " ".join(cmd)
        return {
            "disabled": disabled,
            "ignore_user_config": has_ignore_user_config,
            "web_search_disabled": web_search_disabled,
        }

    @pytest.mark.parametrize("mode", ["restricted", "search", "unrestricted"])
    def test_isolation_flags_match(self, workspace, mode):
        from ai4sci_bench.core.types import TaskInstance, PromptLevel
        ti = TaskInstance(
            task_id="t", instance_id="i", task_dir=workspace.parent,
            workspace_dir=workspace, reference_dir=workspace.parent / "ref",
            prompt_level=PromptLevel.B2, parameters={},
            metadata={"output": {"files": []}},
        )
        adapter = CodexCLIAdapter(tool_mode=mode)
        os_cmd = adapter._build_os_agent_cmd(workspace)
        local_cmd = adapter._build_command(ti, task_env=None)

        os_flags = self._extract_isolation_flags(os_cmd)
        local_flags = self._extract_isolation_flags(local_cmd)
        assert os_flags == local_flags, (
            f"Mismatch for mode={mode}:\n  os={os_flags}\n  local={local_flags}"
        )

    @patch("ai4sci_bench.adapters.subprocess_base.validate_sandbox_mode")
    @patch("ai4sci_bench.adapters.subprocess_base.LinuxNSSandbox")
    @pytest.mark.parametrize("mode", ["restricted", "search"])
    def test_linux_ns_isolation_matches_os(
        self, mock_linux_ns_cls, mock_validate, workspace, mode
    ):
        from ai4sci_bench.core.types import TaskInstance, PromptLevel
        ti = TaskInstance(
            task_id="t", instance_id="i", task_dir=workspace.parent,
            workspace_dir=workspace, reference_dir=workspace.parent / "ref",
            prompt_level=PromptLevel.B2, parameters={},
            metadata={"output": {"files": []}},
        )
        adapter = CodexCLIAdapter(tool_mode=mode)
        adapter.setup({"sandbox": "linux_ns"})
        linux_ns_cmd = adapter._build_command(ti, task_env=None)

        adapter2 = CodexCLIAdapter(tool_mode=mode)
        os_cmd = adapter2._build_os_agent_cmd(workspace)

        ns_flags = self._extract_isolation_flags(linux_ns_cmd)
        os_flags = self._extract_isolation_flags(os_cmd)
        assert ns_flags == os_flags


# ---------------------------------------------------------------------------
# Negative tests: things that must NEVER appear
# ---------------------------------------------------------------------------


class TestNegativeConstraints:
    """Verify things that must NEVER be present in commands."""

    def test_claude_never_has_disallowed_tools(self, workspace):
        for mode in ("restricted", "search", "unrestricted"):
            cmd = ClaudeCodeCLIAdapter(tool_mode=mode)._build_os_agent_cmd(workspace)
            assert "--disallowed-tools" not in cmd, f"mode={mode} has --disallowed-tools"

    def test_claude_restricted_never_has_websearch(self, workspace):
        cmd = ClaudeCodeCLIAdapter(tool_mode="restricted")._build_os_agent_cmd(workspace)
        idx = cmd.index("--tools")
        tools = cmd[idx + 1].split(",")
        assert "WebSearch" not in tools
        assert "WebFetch" not in tools

    def test_codex_never_has_full_auto(self, workspace):
        from ai4sci_bench.core.types import TaskInstance, PromptLevel
        ti = TaskInstance(
            task_id="t", instance_id="i", task_dir=workspace.parent,
            workspace_dir=workspace, reference_dir=workspace.parent / "ref",
            prompt_level=PromptLevel.B2, parameters={},
            metadata={"output": {"files": []}},
        )
        for mode in ("restricted", "search", "unrestricted"):
            adapter = CodexCLIAdapter(tool_mode=mode)
            cmd = adapter._build_command(ti, task_env=None)
            assert "--full-auto" not in cmd, f"mode={mode} has --full-auto"

    def test_codex_never_disables_multi_agent(self, workspace):
        for mode in ("restricted", "search", "unrestricted"):
            cmd = CodexCLIAdapter(tool_mode=mode)._build_os_agent_cmd(workspace)
            disabled = [cmd[i + 1] for i, v in enumerate(cmd) if v == "--disable"]
            assert "multi_agent" not in disabled, f"mode={mode} disables multi_agent"

    def test_codex_unrestricted_has_no_isolation(self, workspace):
        cmd = CodexCLIAdapter(tool_mode="unrestricted")._build_os_agent_cmd(workspace)
        assert "--ignore-user-config" not in cmd
        assert "--disable" not in cmd
        assert "--config" not in cmd


# ---------------------------------------------------------------------------
# CLI integration: _resolve_tool_mode and _build_agent_metadata
# ---------------------------------------------------------------------------


class TestCLIToolModeIntegration:
    """End-to-end: CLI flags → adapter → correct tool_mode."""

    def test_resolve_default(self):
        assert _resolve_tool_mode(None, False) == "restricted"

    def test_resolve_allow_external_tools(self):
        assert _resolve_tool_mode(None, True) == "search"

    def test_resolve_explicit_overrides(self):
        assert _resolve_tool_mode("unrestricted", False) == "unrestricted"
        assert _resolve_tool_mode("restricted", True) == "restricted"

    def test_build_agent_passes_tool_mode_to_claude(self):
        adapter = _build_agent(None, "claude_code_cli", {}, tool_mode="search")
        assert adapter.tool_mode == ToolMode.SEARCH

    def test_build_agent_passes_tool_mode_to_codex(self):
        adapter = _build_agent(None, "codex_cli", {}, tool_mode="unrestricted")
        assert adapter.tool_mode == ToolMode.UNRESTRICTED

    def test_build_agent_default_restricted(self):
        adapter = _build_agent(None, "claude_code_cli", {})
        assert adapter.tool_mode == ToolMode.RESTRICTED

    def test_metadata_records_tool_mode(self):
        for mode in ("restricted", "search", "unrestricted"):
            meta = _build_agent_metadata(None, "claude_code_cli", {}, tool_mode=mode)
            assert meta["tool_mode"] == mode

    def test_metadata_derives_from_allow_external_tools(self):
        meta = _build_agent_metadata(None, "codex_cli", {}, allow_external_tools=True)
        assert meta["tool_mode"] == "search"

    def test_metadata_explicit_overrides_allow_external_tools(self):
        meta = _build_agent_metadata(
            None, "codex_cli", {},
            allow_external_tools=True,
            tool_mode="restricted",
        )
        assert meta["tool_mode"] == "restricted"


# ---------------------------------------------------------------------------
# Mutation resistance: each isolation mechanism has dedicated tests
# ---------------------------------------------------------------------------


class TestMutationResistance:
    """If any single isolation mechanism is removed, these tests must fail.

    Each test checks exactly ONE mechanism in isolation, so if the
    implementation is mutated (e.g. a line deleted), exactly the right
    test fails and pinpoints the regression.
    """

    def test_claude_apply_tool_isolation_called_in_os_cmd(self, workspace):
        """If _apply_tool_isolation is removed from _build_os_agent_cmd, this fails."""
        cmd = ClaudeCodeCLIAdapter()._build_os_agent_cmd(workspace)
        assert "--tools" in cmd

    @patch("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout")
    def test_claude_apply_tool_isolation_called_in_build_command(self, mock_run, workspace):
        """If _apply_tool_isolation is removed from _build_command, this fails."""
        from ai4sci_bench.core.types import TaskInstance, PromptLevel
        ti = TaskInstance(
            task_id="t", instance_id="i", task_dir=workspace.parent,
            workspace_dir=workspace, reference_dir=workspace.parent / "ref",
            prompt_level=PromptLevel.B2, parameters={},
            metadata={"output": {"files": []}},
        )
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        ClaudeCodeCLIAdapter().solve(ti)
        cmd = mock_run.call_args.args[0]
        assert "--tools" in cmd

    def test_codex_apply_tool_isolation_called_in_os_cmd(self, workspace):
        """If _apply_tool_isolation is removed from _build_os_agent_cmd, this fails."""
        cmd = CodexCLIAdapter()._build_os_agent_cmd(workspace)
        assert "--disable" in cmd

    def test_codex_apply_tool_isolation_called_in_build_command(self, workspace):
        """If _apply_tool_isolation is removed from _build_command, this fails."""
        from ai4sci_bench.core.types import TaskInstance, PromptLevel
        ti = TaskInstance(
            task_id="t", instance_id="i", task_dir=workspace.parent,
            workspace_dir=workspace, reference_dir=workspace.parent / "ref",
            prompt_level=PromptLevel.B2, parameters={},
            metadata={"output": {"files": []}},
        )
        cmd = CodexCLIAdapter()._build_command(ti, task_env=None)
        assert "--disable" in cmd

    def test_codex_ignore_user_config_present(self, workspace):
        """If --ignore-user-config line is removed, this fails."""
        cmd = CodexCLIAdapter()._build_os_agent_cmd(workspace)
        assert "--ignore-user-config" in cmd

    def test_codex_web_search_disabled_in_restricted(self, workspace):
        """If web_search disable is removed, this fails."""
        cmd = CodexCLIAdapter(tool_mode="restricted")._build_os_agent_cmd(workspace)
        assert 'web_search="disabled"' in " ".join(cmd)

    def test_codex_web_search_not_disabled_in_search(self, workspace):
        """If search mode accidentally disables web_search, this fails."""
        cmd = CodexCLIAdapter(tool_mode="search")._build_os_agent_cmd(workspace)
        assert 'web_search="disabled"' not in " ".join(cmd)

    def test_claude_unrestricted_early_return(self, workspace):
        """If the early return for UNRESTRICTED is removed, this fails."""
        cmd = ClaudeCodeCLIAdapter(tool_mode="unrestricted")._build_os_agent_cmd(workspace)
        assert "--tools" not in cmd

    def test_codex_unrestricted_early_return(self, workspace):
        """If the early return for UNRESTRICTED is removed, this fails."""
        cmd = CodexCLIAdapter(tool_mode="unrestricted")._build_os_agent_cmd(workspace)
        assert "--ignore-user-config" not in cmd
        assert "--disable" not in cmd


# ---------------------------------------------------------------------------
# CODEX_RESTRICTED_DISABLE_FEATURES tuple integrity
# ---------------------------------------------------------------------------


class TestFeatureListIntegrity:
    """Ensure the feature list cannot be accidentally modified."""

    def test_features_is_tuple(self):
        assert isinstance(CODEX_RESTRICTED_DISABLE_FEATURES, tuple)

    def test_features_are_strings(self):
        for f in CODEX_RESTRICTED_DISABLE_FEATURES:
            assert isinstance(f, str)
            assert len(f) > 0

    def test_no_duplicate_features(self):
        assert len(CODEX_RESTRICTED_DISABLE_FEATURES) == len(set(CODEX_RESTRICTED_DISABLE_FEATURES))

    def test_features_match_known_set(self):
        expected = {
            "plugins",
            "tool_call_mcp_elicitation",
            "skill_mcp_dependency_install",
            "image_generation",
            "codex_hooks",
            "computer_use",
        }
        assert set(CODEX_RESTRICTED_DISABLE_FEATURES) == expected


# ---------------------------------------------------------------------------
# CLAUDE tool constant integrity
# ---------------------------------------------------------------------------


class TestClaudeToolConstantIntegrity:
    """Ensure tool constants cannot be accidentally modified."""

    def test_core_tools_is_string(self):
        assert isinstance(CLAUDE_CORE_TOOLS, str)

    def test_search_tools_is_string(self):
        assert isinstance(CLAUDE_SEARCH_TOOLS, str)

    def test_core_tools_no_spaces(self):
        assert " " not in CLAUDE_CORE_TOOLS

    def test_search_tools_no_spaces(self):
        assert " " not in CLAUDE_SEARCH_TOOLS

    def test_core_tools_match_known_set(self):
        expected = {
            "Bash", "Read", "Write", "Edit", "Glob", "Grep",
            "TodoWrite", "Agent", "NotebookEdit", "ToolSearch",
            "Monitor", "TaskOutput", "TaskStop",
            "EnterWorktree", "ExitWorktree",
        }
        assert set(CLAUDE_CORE_TOOLS.split(",")) == expected

    def test_search_tools_match_known_set(self):
        expected = {
            "Bash", "Read", "Write", "Edit", "Glob", "Grep",
            "TodoWrite", "Agent", "NotebookEdit", "ToolSearch",
            "Monitor", "TaskOutput", "TaskStop",
            "EnterWorktree", "ExitWorktree",
            "WebSearch", "WebFetch",
        }
        assert set(CLAUDE_SEARCH_TOOLS.split(",")) == expected


# ---------------------------------------------------------------------------
# Default model verification
# ---------------------------------------------------------------------------


class TestDefaultModels:
    def test_codex_default_model_is_gpt55(self):
        assert CodexCLIAdapter().model == "gpt-5.5"

    def test_claude_default_model_is_opus(self):
        assert ClaudeCodeCLIAdapter().model == "claude-opus-4-6"
