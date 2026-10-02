"""Golden snapshot of every ``verify_run.verify_one`` result in the MCP E2E tests.

The per-test assertions only check what each test cares about (usually one
verdict or one check). This snapshot pins the *whole* stable outcome of every
verifier call those tests make — verdict, failure, every check status, every
per-call status, the observed tool sequence — so that a refactor of
``scripts/mcp/e2e/verify_run.py`` cannot silently change a check nobody
asserted on.

Every test module in ``tests/mcp_e2e/`` that has a module-level ``verify``
(``support.verify``) is covered automatically: an autouse fixture wraps
``verify.verify_one`` and, when the test passes, the recorded snapshots must
equal ``tests/mcp_e2e/golden.json[<nodeid>]``. A full run also fails on
stale entries (tests that no longer exist or no longer call the verifier).

Intended behaviour changes are recorded by regenerating the file; the change
then shows up in the diff::

    MCP_E2E_GOLDEN=update uv run --frozen pytest -q tests/mcp_e2e

Update mode only writes after a green run of whole test files (no ``-k`` or
``file::test`` selections), because it replaces those files' entries.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "golden.json"
DIR_PREFIX = "tests/mcp_e2e/"            # nodeid prefix of the covered test files
ENV = "MCP_E2E_GOLDEN"

_state: dict = {"golden": None, "recorded": {}, "callers": set(), "passed": set(), "deselected": set()}


def _update_mode() -> bool:
    return os.environ.get(ENV, "").strip().lower() == "update"


def _golden() -> dict[str, list[dict]]:
    if _state["golden"] is None:
        _state["golden"] = json.loads(GOLDEN.read_text(encoding="utf-8")) if GOLDEN.is_file() else {}
    return _state["golden"]


def _file_of(nodeid: str) -> str:
    return nodeid.split("::", 1)[0]


def snapshot(row: dict) -> dict:
    """The stable part of a verify_one row (no paths, no free-text details)."""
    checks = row.get("checks") or {}
    per_call = (checks.get("tool_correct") or {}).get("per_call") or {}
    return {
        "verdict": row.get("verdict"),
        "failure": row.get("failure"),
        "checks": {name: check.get("status") for name, check in checks.items()},
        "per_call": {name: call.get("status") for name, call in per_call.items()},
        "tool_sequence": row.get("tool_sequence"),
    }


def _covered(request) -> bool:
    module = request.module
    return Path(module.__file__).resolve().parent == HERE \
        and hasattr(getattr(module, "verify", None), "verify_one")


def _diff(nodeid: str, recorded: list[dict], expected: list[dict]) -> str:
    lines = [f"verify_one outcome changed for {nodeid} (regenerate with {ENV}=update if intended):"]
    for i in range(max(len(recorded), len(expected))):
        got = recorded[i] if i < len(recorded) else None
        want = expected[i] if i < len(expected) else None
        if got == want:
            continue
        if got is None or want is None:
            lines.append(f"  call #{i}: expected {want}, got {got}")
            continue
        for key in sorted(set(got) | set(want)):
            if got.get(key) != want.get(key):
                lines.append(f"  call #{i} {key}: expected {want.get(key)!r}, got {got.get(key)!r}")
    return "\n".join(lines)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when == "call" and report.passed:
        _state["passed"].add(item.nodeid)


@pytest.fixture(autouse=True)
def mcp_e2e_verify_golden(request, monkeypatch):
    if not _covered(request):
        yield
        return
    verify = request.module.verify
    original = verify.verify_one
    recorded: list[dict] = []

    def recording(*args, **kwargs):
        row = original(*args, **kwargs)
        recorded.append(snapshot(row))
        return row

    monkeypatch.setattr(verify, "verify_one", recording)
    yield
    nodeid = request.node.nodeid
    if nodeid not in _state["passed"] or not recorded:
        return  # failed or skipped tests are reported by pytest; non-callers are checked as stale
    _state["callers"].add(nodeid)
    if _update_mode():
        _state["recorded"][nodeid] = recorded
        return
    expected = _golden().get(nodeid)
    if expected is None:
        pytest.fail(f"no golden verify_one snapshot for {nodeid}; regenerate with {ENV}=update")
    if recorded != expected:
        pytest.fail(_diff(nodeid, recorded, expected))


def pytest_deselected(items) -> None:
    _state["deselected"].update(item.nodeid for item in items)


def _partial_selection(config) -> bool:
    return bool(config.option.keyword) or any("::" in str(arg) for arg in config.args)


def _report(config, message: str) -> None:
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    if reporter:
        reporter.write_line(message, red=True)
    else:
        print(message)


def pytest_sessionfinish(session, exitstatus) -> None:
    config = session.config
    ran = {item.nodeid for item in session.items}
    files = {_file_of(n) for n in ran if _file_of(n).startswith(DIR_PREFIX)}
    if not files:
        return
    deselected = {n for n in _state["deselected"] if _file_of(n) in files}
    if _update_mode():
        if exitstatus != 0 or _partial_selection(config) or deselected:
            _report(config, f"{ENV}=update: not writing {GOLDEN.name} (needs a green run of whole test files)")
            session.exitstatus = exitstatus or 1
            return
        kept = {k: v for k, v in _golden().items() if _file_of(k) not in files}
        merged = {**kept, **_state["recorded"]}
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        body = ",\n".join(f"  {json.dumps(k)}: [\n" + ",\n".join(f"    {json.dumps(s, sort_keys=True)}" for s in v)
                          + "\n  ]" for k, v in sorted(merged.items()))
        GOLDEN.write_text("{\n" + body + "\n}\n", encoding="utf-8")
        _report(config, f"wrote {len(merged)} golden verify_one entries to {GOLDEN}")
        return
    if exitstatus != 0 or _partial_selection(config):
        return
    root = Path(config.rootpath)
    stale = sorted(
        k for k in _golden()
        if not (root / _file_of(k)).is_file()                                    # module deleted or renamed
        or (_file_of(k) in files and k not in deselected and k not in ran)       # test deleted or renamed
        or (k in _state["passed"] and k not in _state["callers"]))               # test no longer verifies
    if stale:
        session.exitstatus = 1
        _report(config, f"{len(stale)} stale golden verify_one entr{'y' if len(stale) == 1 else 'ies'} "
                        f"(regenerate with {ENV}=update): " + ", ".join(stale[:5]))
