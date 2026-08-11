"""Tests for run_subprocess_with_graceful_timeout (Fix A — graceful shutdown).

Covers the SIGTERM → grace → SIGKILL escalation that gives CLI agents a window
to flush credentials before being killed (docs/cc_session_stability_fix.md).
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from ai4sci_bench.runner.proc_util import (
    GRACEFUL_SHUTDOWN_SECONDS,
    run_subprocess_with_graceful_timeout,
)


def _py(code: str) -> list[str]:
    return [sys.executable, "-c", code]


class TestNormalCompletion:
    def test_returns_completed_process(self):
        result = run_subprocess_with_graceful_timeout(
            _py("print('hello')"), timeout=30
        )
        assert isinstance(result, subprocess.CompletedProcess)
        assert result.returncode == 0
        assert result.stdout.strip() == "hello"
        assert result.stderr == ""

    def test_nonzero_exit_code(self):
        result = run_subprocess_with_graceful_timeout(
            _py("import sys; sys.exit(3)"), timeout=30
        )
        assert result.returncode == 3

    def test_stdin_input_is_fed(self):
        result = run_subprocess_with_graceful_timeout(
            _py("import sys; sys.stdout.write(sys.stdin.read().upper())"),
            timeout=30,
            input="abc",
        )
        assert result.stdout == "ABC"

    def test_shell_mode(self):
        result = run_subprocess_with_graceful_timeout(
            "echo shellworks", timeout=30, shell=True
        )
        assert "shellworks" in result.stdout

    def test_decodes_invalid_utf8_without_crashing(self):
        # Emit raw bytes that are not valid UTF-8; errors="replace" must apply.
        result = run_subprocess_with_graceful_timeout(
            _py(
                "import sys; sys.stdout.buffer.write(b'\\xff\\xfe'); "
                "sys.stdout.flush()"
            ),
            timeout=30,
        )
        assert result.returncode == 0
        assert "�" in result.stdout


class TestGracefulTimeout:
    def test_default_sigterm_terminates_within_grace(self):
        # A plain sleeper dies on the default SIGTERM handler — no escalation.
        with pytest.raises(subprocess.TimeoutExpired) as ei:
            run_subprocess_with_graceful_timeout(
                _py("import time; time.sleep(60)"), timeout=1, grace=5
            )
        assert ei.value.forced_kill is False

    def test_partial_output_captured_on_timeout(self):
        with pytest.raises(subprocess.TimeoutExpired) as ei:
            run_subprocess_with_graceful_timeout(
                _py(
                    "import time; print('started', flush=True); time.sleep(60)"
                ),
                timeout=1,
                grace=5,
            )
        assert "started" in (ei.value.output or "")

    def test_sigterm_ignored_escalates_to_sigkill(self):
        # Child traps SIGTERM and keeps running → must escalate to SIGKILL.
        code = (
            "import signal, time; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "print('ready', flush=True); time.sleep(60)"
        )
        with pytest.raises(subprocess.TimeoutExpired) as ei:
            run_subprocess_with_graceful_timeout(_py(code), timeout=1, grace=1)
        assert ei.value.forced_kill is True
        assert "ready" in (ei.value.output or "")


def test_default_grace_constant():
    assert GRACEFUL_SHUTDOWN_SECONDS == 10
