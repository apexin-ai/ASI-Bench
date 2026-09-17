"""Independent review regressions: process lifetime and malformed evidence."""

import json
import os
import subprocess
import sys
import time
from unittest.mock import Mock

import pytest

from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
from ai4sci_bench.core.types import PromptLevel, RunStatus, TaskInstance
from ai4sci_bench.runner.proc_util import run_subprocess_with_graceful_timeout


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group lifetime")
@pytest.mark.parametrize("timed_out", [False, True])
@pytest.mark.parametrize("ignore_term", [False, True])
def test_no_descendant_mutates_workspace_after_return(tmp_path, timed_out, ignore_term):
    ready = tmp_path / "ready"
    marker = tmp_path / "late-output"
    child = (
        "import signal,time\nfrom pathlib import Path\n"
        + ("signal.signal(signal.SIGTERM,signal.SIG_IGN)\n" if ignore_term else "")
        + f"Path({str(ready)!r}).write_text('ready')\n"
        + "time.sleep(.65)\n"
        + f"Path({str(marker)!r}).write_text('outside budget')\n"
    )
    parent = (
        "import subprocess,sys,time\nfrom pathlib import Path\n"
        + f"subprocess.Popen([sys.executable,'-c',{child!r}])\n"
        + f"while not Path({str(ready)!r}).exists(): time.sleep(.005)\n"
        + "print('parent ready',flush=True)\n"
        + ("time.sleep(5)\n" if timed_out else "")
    )
    kwargs = dict(timeout=.2 if timed_out else 3, grace=.15,
                  live_stdout_path=tmp_path / "stdout",
                  live_stderr_path=tmp_path / "stderr")
    if timed_out:
        with pytest.raises(subprocess.TimeoutExpired) as exc:
            run_subprocess_with_graceful_timeout([sys.executable, "-c", parent], **kwargs)
        assert "parent ready" in exc.value.stdout
    else:
        result = run_subprocess_with_graceful_timeout([sys.executable, "-c", parent], **kwargs)
        assert result.returncode == 0
        assert "parent ready" in result.stdout
    assert ready.exists()
    time.sleep(.75)
    assert not marker.exists(), "a descendant wrote output after the attempt returned"


@pytest.mark.parametrize("malformed", [[], None, "unexpected string", 1, {"type": []}, {"type": {}}, {"type": "assistant", "message": None}, {"type": "assistant", "message": {"content": [None]}}])
def test_malformed_jsonl_preserves_raw_evidence(tmp_path, monkeypatch, malformed):
    raw = json.dumps(malformed) + '\n{"type":"result","is_error":false,"result":"done"}\n'
    adapter = ClaudeCodeCLIAdapter(model="fixture")
    monkeypatch.setattr(adapter, "_build_command", lambda *_: ["fixture"])
    monkeypatch.setattr(adapter, "_build_run_env", lambda *_: None)
    monkeypatch.setattr(adapter, "_get_stdin_input", lambda *_: None)
    monkeypatch.setattr("ai4sci_bench.adapters.subprocess_base.run_subprocess_with_graceful_timeout",
                        Mock(return_value=subprocess.CompletedProcess([], 0, raw, "stderr evidence")))
    instance = TaskInstance(task_id="fixture", instance_id="fixture", task_dir=tmp_path,
                            workspace_dir=tmp_path, reference_dir=tmp_path,
                            prompt_level=PromptLevel.B1, parameters={}, metadata={})
    output = adapter.solve(instance)
    assert output.raw_stdout == raw
    assert output.raw_stderr == "stderr evidence"
    assert output.status == RunStatus.FAILED
    assert "incomplete" in output.error_message
