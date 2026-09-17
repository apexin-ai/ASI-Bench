"""Subprocess execution helpers with graceful timeout handling.

Low-level utility shared by the subprocess-based adapters and the
namespace/Docker sandboxes.  Lives in its own module (no dependency on the
adapter layer) so it can be imported from both ``adapters`` and ``runner``
without creating an import cycle.

The key behaviour is :func:`run_subprocess_with_graceful_timeout`, which
escalates a timeout as **SIGTERM → grace period → SIGKILL** instead of the
immediate ``SIGKILL`` that ``subprocess.run(timeout=...)`` performs.  This
gives CLI agents (notably Claude Code) a window to flush OAuth credentials and
session state, avoiding the corrupted ``~/.claude/.credentials.json`` /
spurious logout problem documented in ``docs/cc_session_stability_fix.md``.
"""

from __future__ import annotations

from contextlib import ExitStack
import io
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

# Grace period (seconds) a child gets after SIGTERM before we escalate to
# SIGKILL.  10s is enough for a token flush but keeps the worst-case wall-clock
# overshoot bounded (effective timeout ≈ timeout + GRACEFUL_SHUTDOWN_SECONDS).
GRACEFUL_SHUTDOWN_SECONDS = 10


def run_subprocess_with_graceful_timeout(
    cmd: list[str] | str,
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
    shell: bool = False,
    input: str | None = None,
    grace: int = GRACEFUL_SHUTDOWN_SECONDS,
    live_stdout_path: str | Path | None = None,
    live_stderr_path: str | Path | None = None,
) -> subprocess.CompletedProcess:
    """Run a subprocess, escalating SIGTERM → grace → SIGKILL on timeout.

    Drop-in replacement for the ``subprocess.run(timeout=...)`` call sites that
    capture stdout/stderr as UTF-8 text. The child writes directly to capture
    files, which are decoded with ``errors="replace"`` (matching issue #30).

    On timeout the child first receives ``SIGTERM`` (catchable — the agent can
    clean up), then ``SIGKILL`` if it does not exit within ``grace`` seconds.
    A :class:`subprocess.TimeoutExpired` is raised in **both** cases (so callers
    keep treating the run as a timeout), carrying any captured partial output.
    The raised exception has a ``forced_kill`` attribute: ``True`` when
    escalation to SIGKILL was required, ``False`` when the child exited within
    the grace period. On POSIX, the CLI and tools remaining in its process
    group are terminated together, including tools left after normal CLI exit.

    Returns a :class:`subprocess.CompletedProcess` on normal completion.
    """
    with ExitStack() as stack:
        captures = []
        for path in (live_stdout_path, live_stderr_path):
            if path is None:
                capture = stack.enter_context(tempfile.TemporaryFile(mode="w+b"))
            else:
                live_path = Path(path)
                live_path.parent.mkdir(parents=True, exist_ok=True)
                capture = stack.enter_context(live_path.open("w+b"))
            captures.append(capture)

        def captured_text():
            values = []
            for capture in captures:
                capture.seek(0)
                # Match subprocess(text=True), including replacement decoding
                # and universal newlines, while retaining raw bytes on disk.
                with io.TextIOWrapper(io.BytesIO(capture.read()), encoding="utf-8",
                                      errors="replace") as reader:
                    values.append(reader.read())
            return values

        # The child writes directly to the retained capture. Live observers
        # and final parsing never compete for bytes from a pipe.
        proc = subprocess.Popen(
            cmd, cwd=str(cwd) if cwd is not None else None,
            stdout=captures[0], stderr=captures[1],
            stdin=subprocess.PIPE if input is not None else None,
            env=env, shell=shell, text=True, encoding="utf-8", errors="replace",
            start_new_session=os.name == "posix",
        )

        def group_alive():
            if os.name != "posix":
                return proc.poll() is None
            try:
                os.killpg(proc.pid, 0)
                return True
            except ProcessLookupError:
                return False
            except PermissionError:
                # Darwin can briefly report EPERM for an exiting orphaned
                # group. Keep waiting; an actual signal permission failure is
                # still surfaced by signal_group rather than ignored.
                return True

        def signal_group(force=False):
            try:
                if os.name == "posix":
                    os.killpg(proc.pid, signal.SIGKILL if force else signal.SIGTERM)
                elif proc.poll() is None:
                    proc.kill() if force else proc.terminate()
            except ProcessLookupError:
                pass

        def stop_group():
            """Stop tools as well as the CLI before returning its artifacts."""
            if not group_alive():
                return False
            deadline = time.monotonic() + grace
            signal_group()
            try:
                proc.communicate(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
            while group_alive() and time.monotonic() < deadline:
                time.sleep(min(.01, max(0, deadline - time.monotonic())))
            forced = group_alive()
            if forced:
                signal_group(force=True)
            proc.communicate()
            return forced

        try:
            proc.communicate(input=input, timeout=timeout)
        except subprocess.TimeoutExpired:
            forced = stop_group()
            stdout, stderr = captured_text()
            exc = subprocess.TimeoutExpired(cmd, timeout, output=stdout, stderr=stderr)
            exc.forced_kill = forced
            raise exc
        except BaseException:
            stop_group()
            raise
        # A CLI may exit while a background tool still owns the capture files
        # or workspace. End that tool's lifetime before taking the snapshot.
        stop_group()
        stdout, stderr = captured_text()
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
