"""Regression tests for issue #30.

On Windows 中文 locale (GBK / CP936), ``subprocess.run(..., text=True)`` without an
explicit ``encoding`` falls back to ``locale.getpreferredencoding(False)`` and
crashes the internal stdout reader thread when the subprocess emits UTF-8 bytes
that are illegal under GBK (e.g. Claude CLI JSONL containing Chinese punctuation
or emoji). The reader thread dies silently, ``communicate()`` then blocks
forever on EOF, and the framework only recovers via the outer ``--timeout``
kill — producing a spurious ``status: timeout`` / score 0 result.

The fix is to always pass ``encoding="utf-8", errors="replace"`` alongside
``text=True`` in every ``subprocess.run`` site that decodes subprocess output
as text. These tests pin that contract for the known call sites so the bug
cannot regress.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai4sci_bench.core.types import PromptLevel, TaskInstance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_call(mock_run: MagicMock, *, first_arg_prefix: str):
    """Return the first recorded call whose positional cmd starts with the given token.

    On macOS the OSSandbox path performs auxiliary ``subprocess.run`` calls
    (e.g. ``security dump-keychain`` for Claude credential discovery) before
    the actual ``docker run`` invocation we want to assert against. Filter by
    the command prefix so the test is platform-independent.
    """
    for call in mock_run.call_args_list:
        args = call.args[0] if call.args else call.kwargs.get("args")
        if isinstance(args, (list, tuple)) and args and args[0] == first_arg_prefix:
            return call
    raise AssertionError(
        f"no subprocess.run call started with {first_arg_prefix!r}; "
        f"recorded calls: {[c.args for c in mock_run.call_args_list]}"
    )


def _assert_utf8_replace_call(call) -> None:
    """Assert the given mock call decodes subprocess output as UTF-8 with replace."""
    kwargs = call.kwargs
    assert kwargs.get("text") is True, (
        f"expected text=True, got kwargs={kwargs}"
    )
    assert kwargs.get("encoding") == "utf-8", (
        f"subprocess.run must pass encoding='utf-8' to avoid GBK fallback on "
        f"Windows 中文 locale (issue #30). Got kwargs={kwargs}"
    )
    assert kwargs.get("errors") == "replace", (
        f"subprocess.run must pass errors='replace' so illegal bytes do not "
        f"kill the stdout reader thread (issue #30). Got kwargs={kwargs}"
    )


def _make_instance(workspace: Path) -> TaskInstance:
    return TaskInstance(
        task_id="dummy.task",
        instance_id="inst",
        task_dir=workspace,
        workspace_dir=workspace,
        reference_dir=workspace,
        prompt_level=PromptLevel.B1,
        parameters={},
        metadata={"difficulty": {}, "output": {"files": []}},
    )


# ---------------------------------------------------------------------------
# os_sandbox.py: the exact call sites that triggered #30
# ---------------------------------------------------------------------------


class TestOSSandboxEncoding:
    """Both Docker paths in ``OSSandbox`` must decode subprocess output as UTF-8."""

    def _make_sandbox(self, tmp_path: Path):
        from ai4sci_bench.runner.os_sandbox import OSSandbox

        builder = MagicMock()
        builder.ensure_image.return_value = "ai4sci-bench-base:latest"
        builder.get_image_identity.return_value = "sha256:abc"
        builder.check_daemon_health.return_value = True
        return OSSandbox(tmp_path, image_builder=builder)

    @patch("ai4sci_bench.runner.os_sandbox.subprocess.run")
    def test_run_agent_uses_utf8_with_replace(self, mock_run, tmp_path: Path):
        sandbox = self._make_sandbox(tmp_path)
        workspace = tmp_path / "ws"
        workspace.mkdir()
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")

        sandbox.run_agent(
            {"difficulty": {}},
            agent_cmd=["claude", "--print", "solve"],
            workspace=workspace,
            timeout=60,
            agent_type="claude_code",
        )

        _assert_utf8_replace_call(_find_call(mock_run, first_arg_prefix="docker"))

    @patch("ai4sci_bench.runner.os_sandbox.subprocess.run")
    def test_execute_python_uses_utf8_with_replace(self, mock_run, tmp_path: Path):
        sandbox = self._make_sandbox(tmp_path)
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "solve.py").write_text("print('hi')")
        mock_run.return_value = MagicMock(returncode=0, stdout="hi\n", stderr="")

        sandbox.execute_python(
            {"difficulty": {}},
            workspace=workspace,
            code_file="solve.py",
            timeout=60,
        )

        _assert_utf8_replace_call(_find_call(mock_run, first_arg_prefix="docker"))


# ---------------------------------------------------------------------------
# direct_llm.py: host-side python execution (sandbox == "none")
# ---------------------------------------------------------------------------


class TestDirectLLMExecutionEncoding:
    @patch("ai4sci_bench.adapters.direct_llm.subprocess.run")
    def test_host_python_execution_uses_utf8(self, mock_run, tmp_path: Path):
        from ai4sci_bench.adapters.direct_llm import DirectLLMAdapter

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "solve.py").write_text("print('ok')")

        adapter = DirectLLMAdapter()
        adapter.setup({"sandbox": "none"})
        adapter._execute(_make_instance(workspace), "solve.py")

        # direct_llm's host path only issues the single python subprocess call.
        assert mock_run.call_count == 1
        _assert_utf8_replace_call(mock_run.call_args_list[0])


# ---------------------------------------------------------------------------
# docker_agent.py: legacy docker adapter
# ---------------------------------------------------------------------------


class TestDockerAgentEncoding:
    @patch("ai4sci_bench.adapters.docker_agent.subprocess.run")
    def test_docker_agent_solve_uses_utf8(self, mock_run, tmp_path: Path):
        from ai4sci_bench.adapters.docker_agent import DockerAgentAdapter

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        workspace = tmp_path / "ws"
        workspace.mkdir()

        adapter = DockerAgentAdapter(image="dummy:latest")
        adapter.setup({})
        adapter.solve(_make_instance(workspace))

        _assert_utf8_replace_call(_find_call(mock_run, first_arg_prefix="docker"))


# ---------------------------------------------------------------------------
# instance_generator.py: generate_gt.py subprocess fallback
# ---------------------------------------------------------------------------


class TestInstanceGeneratorEncoding:
    @patch("ai4sci_bench.generators.instance_generator.subprocess.run")
    def test_generate_gt_subprocess_uses_utf8(self, mock_run, tmp_path: Path):
        from ai4sci_bench.generators.instance_generator import InstanceGenerator

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        tasks_dir = tmp_path / "tasks"
        task_dir = tasks_dir / "dummy"
        task_dir.mkdir(parents=True)
        (task_dir / "generate_gt.py").write_text(
            "def generate(output_dir, params):\n    pass\n"
        )

        output_dir = tmp_path / "out"
        output_dir.mkdir()

        gen = InstanceGenerator(tasks_dir=tasks_dir, repo_root=tmp_path)
        # Force the subprocess fallback by making the in-process module load fail.
        with patch.object(
            gen.task_loader,
            "load_generate_gt_module",
            side_effect=Exception("force subprocess path"),
        ):
            gen._run_generate_gt(
                task_metadata={"id": "dummy", "_task_dir": task_dir},
                output_dir=output_dir,
                params={},
            )

        assert mock_run.called, "instance_generator must hit the subprocess path"
        # The only subprocess call here runs the python interpreter directly.
        assert mock_run.call_count == 1
        _assert_utf8_replace_call(mock_run.call_args_list[0])


# ---------------------------------------------------------------------------
# End-to-end: prove UTF-8 bytes that crash under GBK decode cleanly with our kwargs.
# ---------------------------------------------------------------------------


class TestEndToEndUtf8Decoding:
    """Bytes that kill the Windows GBK decoder must survive under our config."""

    def test_utf8_bytes_that_break_gbk_decode_cleanly(self):
        # A naked 0x88 followed by ASCII is illegal in GBK — exactly the kind
        # of byte the issue report observed at position 65317.
        payload = (
            b"hello \xe4\xb8\xad\xe6\x96\x87 world \xf0\x9f\x9a\x80 \x88tail"
        )
        with pytest.raises(UnicodeDecodeError):
            payload.decode("gbk")

        decoded = payload.decode("utf-8", errors="replace")
        assert "中文" in decoded
        assert "🚀" in decoded
        assert "tail" in decoded
