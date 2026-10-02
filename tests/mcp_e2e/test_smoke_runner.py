"""The shared smoke run (e2e_smoke/runner.py, client.py, smoke.py): stdio client, generic L0/L1
checks around the server hooks against a fake server, result classification helpers."""
import json
import subprocess
import sys
import textwrap

import pytest

from . import support
from .support import (
    BUNDLE,
    StubClient,
    report_statuses,
    rpc_text,
    runner,
    setup,
)

stdio_client = support.client


FAKE_SERVER = textwrap.dedent('''
    import json, sys
    for line in sys.stdin:
        msg = json.loads(line)
        if "id" not in msg:
            continue
        method = msg["method"]
        if method == "initialize":
            result = {"protocolVersion": msg["params"]["protocolVersion"], "capabilities": {"tools": {}},
                      "serverInfo": {"name": "fake", "version": "0"}}
        elif method == "tools/list":
            if msg["params"].get("cursor"):
                result = {"tools": [{"name": "b", "inputSchema": {"type": "object"}}]}
            else:
                result = {"tools": [{"name": "a", "inputSchema": {"type": "object"}}], "nextCursor": "p2"}
        elif method == "tools/call":
            print("library chatter on stdout", flush=True)
            if msg["params"]["name"] == "a":
                result = {"content": [{"type": "text", "text": "42"}], "isError": False}
            else:
                result = {"content": [{"type": "text", "text": "unknown"}], "isError": True}
        else:
            result = {}
        print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result}), flush=True)
''')


def _git_checkout(path):
    path.mkdir()
    for args in (["init", "-q"], ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty",
                                  "-m", "pin"]):
        subprocess.run(["git", "-C", str(path), *args], check=True)
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True,
                          text=True).stdout.strip()


def _demo_run(tmp_path, monkeypatch, smoke, *, edit_config=None, argv=()):
    """Run runner.main for a manifest entry 'demo' served by FAKE_SERVER from a pinned checkout."""
    checkout = tmp_path / "demo"
    revision = _git_checkout(checkout)
    (checkout / "fake_server.py").write_text(FAKE_SERVER)
    (checkout / ".venv/bin").mkdir(parents=True)
    (checkout / ".venv/bin/python").symlink_to(sys.executable)
    entry = {"id": "demo", "catalog_id": "demo", "repository": "https://example.invalid/demo.git",
             "revision": revision, "python": "3.12", "install": "uv-sync-frozen",
             "launch": {"command": ".venv/bin/python", "args": ["{checkout}/fake_server.py"],
                        "env": {"DEMO_FLAG": "1"}},
             "expected_tools": ["a", "b"]}
    setup.validate_entry(entry)
    monkeypatch.setattr(setup, "load_manifest", lambda *a: {"demo": entry})
    monkeypatch.setattr(runner, "load_setup", lambda: setup)
    config = setup.render_config(entry, checkout)
    if edit_config:
        edit_config(config["mcpServers"]["demo"])
    config_path = tmp_path / "demo.mcp.json"
    config_path.write_text(json.dumps(config))
    report_path = tmp_path / "report.json"
    code = runner.main(smoke, ["--config", str(config_path), "--report", str(report_path), *argv])
    document = json.loads(report_path.read_text())
    return code, document, {c["name"]: c["status"] for c in document["checks"]}


def test_stdio_client_paginates_and_records_stdout_pollution(tmp_path):
    script = tmp_path / "fake_server.py"
    script.write_text(FAKE_SERVER)
    client = stdio_client.StdioMCP(sys.executable, [str(script)], cwd=tmp_path,
                                   env={"PATH": "/usr/bin:/bin"}, stderr_path=tmp_path / "err.log")
    try:
        assert client.initialize(timeout=30)["serverInfo"]["name"] == "fake"
        assert [t["name"] for t in client.list_tools()] == ["a", "b"]
        ok = client.call_tool("a", {}, timeout=30)
        assert stdio_client.text_of(ok["result"]) == "42"
        assert client.call_tool("zzz", {}, timeout=30)["result"]["isError"] is True
    finally:
        client.close()
    assert client.non_json_stdout == ["library chatter on stdout"] * 2


@pytest.mark.parametrize("server", sorted(setup.load_manifest()))
def test_smoke_covers_every_manifest_tool(server):
    entry = setup.load_manifest()[server]
    path = BUNDLE / "e2e_smoke" / "servers" / f"{server}.py"
    source = path.read_text()
    for tool in entry["expected_tools"]:
        assert f'"{tool}"' in source, f"{path.name} never calls {tool}"


def test_runner_performs_the_shared_checks_around_the_server_hooks(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    seen = {}

    def prepare(session):
        session.state["prepared"] = True
        assert session.env["DEMO_FLAG"] == "1" and "ANTHROPIC_API_KEY" not in session.env
        assert session.env["HOME"] == str(session.home) and session.env["TMPDIR"] == str(session.scratch)

    def run_l1(session):
        assert session.state["prepared"]
        result = session.call("a works", "a", {})
        session.report.add("L1", "a works", "PASS" if runner.text_of(result) == "42" else "FAIL")
        runner.check_rejected(session.call, "b rejects", "b", {})
        (session.cwd / "declared.html").write_text("x")
        (session.cwd / "stray.txt").write_text("x")

    def after(session):
        with session.spawn("probe") as client:
            seen["probe_tools"] = len(client.initialize() and client.list_tools())
        seen["flag"] = session.args.flag

    smoke = runner.Smoke(server="demo", run_l1=run_l1, prepare=prepare, after=after,
                         expected_cwd_files=("declared.html",), packages=("pytest",),
                         add_arguments=lambda p: p.add_argument("--flag", action="store_true"),
                         report_fields=lambda session: {"extra": session.state["prepared"]})
    code, document, statuses = _demo_run(tmp_path, monkeypatch, smoke, argv=["--flag"])
    assert code == 0, document["checks"]
    assert statuses == {
        "config matches manifest": "PASS", "initialize": "PASS", "tools/list": "PASS",
        "tools/list stable": "PASS", "a works": "PASS", "b rejects": "PASS", "unknown tool is an error": "PASS",
        "server alive after calls": "PASS", "stdout is pure JSON-RPC": "WARN", "server leaves cwd untouched": "WARN"}
    checks = {c["name"]: c for c in document["checks"]}
    assert checks["stdout is pure JSON-RPC"]["non_json_stdout_by_tool"] == {"a": 1, "b": 1}
    assert checks["server leaves cwd untouched"]["detail"] == "created ['stray.txt']"
    assert seen == {"probe_tools": 2, "flag": True}
    assert document["extra"] is True and document["environment"]["pytest"]
    assert document["server"] == "demo" and document["result"] == "PASS"


def test_runner_fails_a_config_that_is_not_the_rendered_manifest(tmp_path, monkeypatch):
    smoke = runner.Smoke(server="demo", run_l1=lambda session: None)
    code, document, statuses = _demo_run(tmp_path, monkeypatch, smoke,
                                         edit_config=lambda server: server["env"].update(DEMO_FLAG="0"))
    assert code == 1 and statuses["config matches manifest"] == "FAIL"
    detail = next(c["detail"] for c in document["checks"] if c["name"] == "config matches manifest")
    assert "env" in detail and "regenerate with setup.py" in detail


@pytest.mark.parametrize("response,in_band,expected", [
    (rpc_text("bad", is_error=True), None, "PASS"),
    (rpc_text('{"ok": false}'), lambda r: "ok=false" if '"ok": false' in runner.text_of(r) else None, "WARN"),
    (rpc_text("fine"), None, "FAIL"),
])
def test_check_rejected_classification(response, in_band, expected):
    report = runner.Report()
    runner.check_rejected(runner.Caller(StubClient({"t": response}), report), "t[bad]", "t", {}, in_band=in_band)
    assert report_statuses(report) == {"t[bad]": expected}
    report = runner.Report()
    runner.check_rejected(runner.Caller(StubClient({"t": rpc_text("fine")}), report), "t[bad]", "t", {},
                          on_accept=lambda r: ("WARN", "accepted", {"text": runner.text_of(r)}))
    assert report.checks[0]["status"] == "WARN" and report.checks[0]["text"] == "fine"


def test_json_result_marks_tool_errors_and_fails_non_json():
    report = runner.Report()
    assert runner.json_result(report, "c", rpc_text("x", is_error=True)["result"]) == {"_isError": True, "_text": "x"}
    assert runner.json_result(report, "c", rpc_text('{"a": 1}')["result"]) == {"a": 1}
    structured = {**rpc_text("ignored")["result"], "structuredContent": {"a": 2}}
    assert runner.json_result(report, "c", structured, structured=True) == {"a": 2}
    assert runner.json_result(report, "c", rpc_text("nope")["result"]) is None
    assert report_statuses(report) == {"c": "FAIL"}


def test_smoke_cli_knows_every_manifest_server():
    cli = support.load(BUNDLE / "smoke.py", "mcp_e2e_smoke_cli")
    for sid in setup.load_manifest():
        assert cli.load(sid).server == sid
    with pytest.raises(SystemExit, match="no smoke for 'nope'"):
        cli.load("nope")
    assert cli.main([]) == 2
