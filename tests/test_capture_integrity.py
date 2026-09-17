"""Capture must retain every emitted byte while exposing a live log."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import Mock

import pytest

from ai4sci_bench.runner.proc_util import run_subprocess_with_graceful_timeout


def capture(tmp_path, program, **kwargs):
    return run_subprocess_with_graceful_timeout(
        [sys.executable, "-c", program], timeout=kwargs.pop("timeout", 5),
        live_stdout_path=tmp_path / "live.stdout",
        live_stderr_path=tmp_path / "live.stderr", **kwargs,
    )


@pytest.mark.parametrize("repeat", range(5))
def test_live_and_returned_output_are_identical(tmp_path, repeat):
    expected = "".join(f"event-{i}\n" for i in range(30)) + "terminal-429\n"
    program = "import sys,time\nfor line in " + repr(expected.splitlines()) + ":\n print(line,flush=True)\n time.sleep(.002)\n"
    result = capture(tmp_path, program, input="fixture")
    assert result.returncode == 0
    assert result.stdout == expected
    assert (tmp_path / "live.stdout").read_text() == result.stdout


def test_large_bidirectional_io_without_newlines(tmp_path):
    payload = "i" * 200000
    result = capture(tmp_path,
        "import sys; sys.stdout.write('o'*200000); sys.stdout.flush(); "
        "sys.stderr.write('e'*200000); sys.stderr.flush(); "
        "sys.stdout.write(sys.stdin.read())", input=payload)
    assert result.stdout == "o" * 200000 + payload
    assert result.stderr == "e" * 200000
    assert (tmp_path / "live.stdout").read_text() == result.stdout
    assert (tmp_path / "live.stderr").read_text() == result.stderr


def test_split_utf8_and_invalid_bytes_match_returned_text(tmp_path):
    result = capture(tmp_path,
        "import os,time; "
        "[(os.write(1, bytes([b])),time.sleep(.002)) for b in bytes.fromhex('e4b8ade69687ff')]")
    assert result.stdout == "\u4e2d\u6587\ufffd"
    assert (tmp_path / "live.stdout").read_bytes().decode("utf-8", errors="replace") == result.stdout


def test_live_output_is_available_before_child_exit(tmp_path):
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(capture, tmp_path,
                             "import os,time; os.write(1,b'visible'); time.sleep(.8)")
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            path = tmp_path / "live.stdout"
            if path.exists() and path.read_bytes() == b"visible":
                break
            time.sleep(.01)
        else:
            pytest.fail("no incremental capture")
        assert not future.done()
        assert future.result().stdout == "visible"


@pytest.mark.skipif(sys.platform == "win32", reason="requires SIGTERM handler")
def test_timeout_preserves_final_sigterm_flush(tmp_path):
    program = (
        "import os,signal,time\n"
        "def stop(*_):\n os.write(1,b'FINAL\\n'); raise SystemExit(0)\n"
        "signal.signal(signal.SIGTERM,stop)\n"
        "os.write(1,b'INITIAL\\n')\n"
        "time.sleep(30)\n"
    )
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        capture(tmp_path, program, timeout=.3, grace=2)
    assert caught.value.stdout == "INITIAL\nFINAL\n"
    assert caught.value.forced_kill is False
    assert (tmp_path / "live.stdout").read_text() == caught.value.stdout


def test_live_paths_are_unique_per_level_and_attempt(tmp_path, monkeypatch):
    from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
    from ai4sci_bench.core.types import PromptLevel, TaskInstance

    (tmp_path / "prompt.md").write_text("fixture")
    adapter = ClaudeCodeCLIAdapter(model="fixture")
    adapter.live_log_dir = tmp_path / "live"
    monkeypatch.setattr(adapter, "_build_run_env", lambda *_: {})
    run = Mock(return_value=subprocess.CompletedProcess([], 0,
               '{"type":"result","is_error":false,"result":""}\n', ""))
    monkeypatch.setattr("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout", run)
    for level in (PromptLevel.B1, PromptLevel.B2, PromptLevel.B1):
        instance = TaskInstance(task_id="fixture", instance_id="same__1", task_dir=tmp_path,
            workspace_dir=tmp_path, reference_dir=tmp_path, prompt_level=level,
            parameters={}, metadata={})
        adapter.solve(instance)
    paths = [call.kwargs["live_stdout_path"] for call in run.call_args_list]
    assert len(set(paths)) == 3
