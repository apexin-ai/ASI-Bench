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

import subprocess
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
) -> subprocess.CompletedProcess:
    """Run a subprocess, escalating SIGTERM → grace → SIGKILL on timeout.

    Drop-in replacement for the ``subprocess.run(timeout=...)`` call sites that
    capture stdout/stderr as UTF-8 text.  stdout and stderr are always piped
    and decoded with ``errors="replace"`` (matching issue #30).

    On timeout the child first receives ``SIGTERM`` (catchable — the agent can
    clean up), then ``SIGKILL`` if it does not exit within ``grace`` seconds.
    A :class:`subprocess.TimeoutExpired` is raised in **both** cases (so callers
    keep treating the run as a timeout), carrying any captured partial output.
    The raised exception has a ``forced_kill`` attribute: ``True`` when
    escalation to SIGKILL was required, ``False`` when the child exited within
    the grace period.

    Returns a :class:`subprocess.CompletedProcess` on normal completion.
    """
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE if input is not None else None,
        env=env,
        shell=shell,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        stdout, stderr = proc.communicate(input=input, timeout=timeout)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=proc.returncode,
            stdout=stdout,
            stderr=stderr,
        )
    except subprocess.TimeoutExpired:
        forced = False
        # Phase 1: SIGTERM — let the child flush credentials / session state.
        proc.terminate()
        try:
            stdout, stderr = proc.communicate(timeout=grace)
        except subprocess.TimeoutExpired:
            # Phase 2: SIGKILL — child ignored SIGTERM within the grace period.
            forced = True
            proc.kill()
            stdout, stderr = proc.communicate()
        exc = subprocess.TimeoutExpired(
            cmd, timeout, output=stdout or "", stderr=stderr or ""
        )
        exc.forced_kill = forced
        raise exc
