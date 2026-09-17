"""Regression coverage for CC transport configuration and terminal evidence."""

import json
import subprocess
from unittest.mock import Mock

import pytest

from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
from ai4sci_bench.core.types import PromptLevel, RunStatus, TaskInstance


def task(workspace):
    return TaskInstance(
        task_id="fixture", instance_id="fixture__1", task_dir=workspace,
        workspace_dir=workspace, reference_dir=workspace,
        prompt_level=PromptLevel.B1, parameters={}, metadata={},
        effective_timeout_seconds=2,
    )


def terminal(**updates):
    event = {"type": "result", "subtype": "success", "is_error": False,
             "result": "", "stop_reason": "end_turn"}
    event.update(updates)
    return json.dumps(event) + "\n"


@pytest.mark.parametrize("stdout", [
    "", '{"type":"system","subtype":"init"}\n',
    '{"type":"error","error":{"type":"rate_limit_error"}}\n',
    '{"type":"assistant","message":{"content":[]}}\n',
    '[]\n', 'null\n', '{"type":"result",',
])
def test_missing_terminal_is_explicit_error(stdout):
    error = ClaudeCodeCLIAdapter._extract_terminal_error_from_jsonl(stdout)
    assert error is not None


def test_valid_empty_success_is_not_blanket_rejected():
    assert ClaudeCodeCLIAdapter._extract_terminal_error_from_jsonl(terminal()) is None


def test_recovered_error_before_success_is_allowed():
    stdout = '{"type":"error","error":{"message":"retrying"}}\n' + terminal(result="done")
    assert ClaudeCodeCLIAdapter._extract_terminal_error_from_jsonl(stdout) is None


def test_result_cannot_hide_incomplete_later_turn():
    stdout = terminal(result="first") + '{"type":"assistant","message":{"content":[]}}\n'
    assert ClaudeCodeCLIAdapter._extract_terminal_error_from_jsonl(stdout) is not None


def test_truncated_tail_after_success_is_not_complete():
    assert ClaudeCodeCLIAdapter._extract_terminal_error_from_jsonl(terminal() + '{"type":') is not None


@pytest.mark.parametrize("event", [
    {"type": "result"},
    {"type": "result", "is_error": False, "subtype": "error_during_execution"},
])
def test_incomplete_or_error_result_schema_is_rejected(event):
    assert ClaudeCodeCLIAdapter._extract_terminal_error_from_jsonl(json.dumps(event)) is not None


def test_explicit_terminal_429_is_preserved():
    error = ClaudeCodeCLIAdapter._extract_terminal_error_from_jsonl(
        terminal(is_error=True, api_error_status=429, result="rate limited"))
    assert "429" in error


def test_os_forwards_runtime_env_and_keeps_success_text(tmp_path, monkeypatch):
    (tmp_path / "prompt.md").write_text("fixture")
    configured = {
        "API_TIMEOUT_MS": "123456", "ANTHROPIC_MAX_RETRIES": "0",
        "CLAUDE_CODE_MAX_RETRIES": "0", "CLAUDE_CODE_RETRY_WATCHDOG": "0",
        "CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK": "1",
        "CLAUDE_ENABLE_BYTE_WATCHDOG": "1",
        "CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS": "10000",
    }
    for key, value in configured.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("UNRELATED_SECRET", "do-not-forward")
    adapter = ClaudeCodeCLIAdapter(model="fixture")
    adapter.sandbox = "os"
    raw = terminal(result="A previous tool timed out; the task is now complete.")
    sandbox = Mock()
    sandbox.run_agent.return_value = (True, raw, raw, "", "image-fixture")
    adapter._os_sandbox = sandbox
    result = adapter.solve(task(tmp_path))
    kwargs = sandbox.run_agent.call_args.kwargs
    for key, value in configured.items():
        assert kwargs["extra_env"][key] == value
    assert "UNRELATED_SECRET" not in kwargs["extra_env"]
    assert kwargs["raise_on_timeout"] is True
    assert result.status == RunStatus.COMPLETED


def test_os_timeout_keeps_partial_evidence_and_status(tmp_path):
    (tmp_path / "prompt.md").write_text("fixture")
    adapter = ClaudeCodeCLIAdapter(model="fixture")
    adapter.sandbox = "os"
    sandbox = Mock()
    exc = subprocess.TimeoutExpired(["fixture"], 2, output=b'{"type":"system"}\n', stderr=b"waiting")
    exc.image_identity = "image-fixture"
    sandbox.run_agent.side_effect = exc
    adapter._os_sandbox = sandbox
    result = adapter.solve(task(tmp_path))
    assert result.status == RunStatus.TIMEOUT
    assert result.raw_stdout == '{"type":"system"}\n'
    assert result.raw_stderr == "waiting"
    assert "timed out" in result.error_message


def test_os_and_host_submit_identical_proxy_prompt(tmp_path):
    (tmp_path / "prompt.md").write_text("fixture")
    adapter = ClaudeCodeCLIAdapter(model="Qwen/test", api_protocol="openai",
                                   api_base="http://fixture.invalid/v1")
    assert adapter._build_os_agent_cmd(tmp_path)[-1] == adapter._get_stdin_input(task(tmp_path))


@pytest.mark.parametrize("raise_on_timeout", [False, True])
def test_sandbox_timeout_signal_is_structured_and_backwards_compatible(tmp_path, monkeypatch, raise_on_timeout):
    from ai4sci_bench.runner.os_sandbox import OSSandbox

    sandbox = Mock()
    sandbox.image_builder.ensure_image.return_value = "image"
    sandbox.image_builder.get_image_identity.return_value = "sha256:fixture"
    sandbox._verify_workspace_writable.return_value = None
    sandbox.enable_workspace_audit = False
    sandbox._build_docker_cmd.return_value = ["docker", "fixture"]
    monkeypatch.setattr(subprocess, "run", Mock(side_effect=subprocess.TimeoutExpired(
        ["docker"], 2, output=b"partial", stderr=b"details")))
    kwargs = dict(task_metadata={}, agent_cmd=["fixture"], workspace=tmp_path, timeout=2,
                  raise_on_timeout=raise_on_timeout)
    if raise_on_timeout:
        with pytest.raises(subprocess.TimeoutExpired) as caught:
            OSSandbox.run_agent(sandbox, **kwargs)
        assert caught.value.image_identity == "sha256:fixture"
        assert caught.value.stdout == b"partial"
    else:
        result = OSSandbox.run_agent(sandbox, **kwargs)
        assert result[0] is False
        assert "timed out" in result[1]
    sandbox._stop_container.assert_called_once()
    sandbox._remove_container.assert_called_once()
