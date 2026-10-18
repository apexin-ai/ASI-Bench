"""OS MCP deployment boundaries, transport readiness and per-run staging."""
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
from ai4sci_bench.adapters.codex_cli import CodexCLIAdapter
from ai4sci_bench.mcp_config import load_mcp_config
from ai4sci_bench.runner import mcp_container
from ai4sci_bench.runner.mcp_sandbox import stage_os_mcp, validate_os_mcp, mcp_provenance

FIXTURE = Path(__file__).parent / "fixtures/mcp/server.py"


def instance(**extra):
    return {"command": "/opt/venv/bin/python", "args": ["/opt/mcp/server.py"], "sandbox": {"deployment": "instance"}, **extra}


def service():
    return {"url": "http://science:8765/mcp", "sandbox": {"deployment": "service", "service": "science"}}


@pytest.mark.parametrize("entry", [
    {"command": "/home/user/venv/bin/python"},
    instance(command="python"), instance(command="/opt/../home/agent/python"),
    instance(cwd="/home/user"), instance(sandbox={"deployment": "instance", "unknown": 1}),
    instance(sandbox={"deployment": "instance", "startup_timeout_seconds": True}),
    instance(sandbox={"deployment": "instance", "env_from": {"LD_PRELOAD": "KEY"}}),
    {"url": "http://localhost:8765/mcp", "sandbox": {"deployment": "service", "service": "science"}},
    {"url": "https://science:8765/mcp", "sandbox": {"deployment": "service", "service": "science"}},
    {"url": "http://science:8765/mcp", "headers": {"Authorization": "secret"}, "sandbox": {"deployment": "service", "service": "science"}},
])
def test_invalid_deployment_fails_closed(entry):
    with pytest.raises(ValueError):
        validate_os_mcp({"science": entry})


def test_service_binds_only_declared_name_and_port():
    validate_os_mcp({"science": service()}, {"runtime": {"services": [{"name": "science", "image": "image:1", "port": 8765}]}})
    for metadata in [{}, {"runtime": {"services": [{"name": "science", "image": "image:1", "port": 1234}]}}]:
        with pytest.raises(ValueError, match="declared runtime.services"):
            validate_os_mcp({"science": service()}, metadata)


@pytest.mark.parametrize("adapter_cls", [ClaudeCodeCLIAdapter, CodexCLIAdapter])
@pytest.mark.parametrize("mode", ["restricted", "search"])
def test_mcp_web_search_independent(adapter_cls, mode, tmp_path, sample_task_instance):
    config = tmp_path / "mcp.json"
    config.write_text(json.dumps({"mcpServers": {"science": instance()}}))
    adapter = adapter_cls(tool_mode=mode, mcp_config=str(config))
    adapter.setup({"sandbox": "os"})
    cmd = adapter._build_command(sample_task_instance, None)
    if adapter_cls is ClaudeCodeCLIAdapter:
        tools = cmd[cmd.index("--tools") + 1]
        assert "mcp__science__*" in tools and "ToolSearch" in tools
        assert ("WebSearch" in tools) == (mode == "search")
    else:
        assert ('web_search="disabled"' in cmd) == (mode == "restricted")
    adapter.teardown()


@pytest.mark.parametrize("agent,cmd", [("claude_code", ["claude", "--mcp-config", "/host/input.json"]), ("codex", ["codex", "exec"])])
def test_staging_is_local_per_run_and_removed(agent, cmd, monkeypatch):
    monkeypatch.setenv("SCIENCE_KEY", "private-test-key")
    entry = instance(sandbox={"deployment": "instance", "env_from": {"API_KEY": "SCIENCE_KEY"}})
    original = json.dumps(entry, sort_keys=True)
    def stage():
        with stage_os_mcp({"science": entry}, {}, agent, cmd, None) as (command, env, mounts):
            directory = Path(mounts[1].split(":")[0])
            assert directory.is_dir()
            assert "/host/input.json" not in command
            source = json.loads((directory / "source.json").read_text())
            assert source["science"]["env"]["API_KEY"] == "private-test-key"
            assert "private-test-key" not in repr(command) + repr(env) + repr(mcp_provenance({"science": entry}))
            assert not (directory / "source.json").stat().st_mode & 0o077
            result = str(directory)
        assert not directory.exists()
        return result
    with ThreadPoolExecutor(max_workers=2) as pool:
        paths = list(pool.map(lambda _: stage(), range(2)))
    assert paths[0] != paths[1]
    assert json.dumps(entry, sort_keys=True) == original


def test_missing_credential_reference_is_failure(monkeypatch):
    monkeypatch.delenv("NO_SUCH_SCIENCE_KEY", raising=False)
    entry = instance(sandbox={"deployment": "instance", "env_from": {"KEY": "NO_SUCH_SCIENCE_KEY"}})
    with pytest.raises(ValueError, match="is unset"):
        with stage_os_mcp({"science": entry}, {}, "codex", ["codex", "exec"], None):
            pytest.fail("must fail before starting container")


def test_probe_stdio_handshake_and_tools_and_env(monkeypatch):
    monkeypatch.setenv("ASIBENCH_TEST_AGENT_SECRET", "must-not-leak")
    mcp_container.probe_stdio(instance(command=sys.executable, args=[str(FIXTURE)], cwd="/tmp"))


@pytest.mark.parametrize("flag", ["--bad", "--hang"])
def test_probe_stdio_failure_or_timeout(flag):
    entry = instance(command=sys.executable, args=[str(FIXTURE), flag], cwd="/tmp", sandbox={"deployment": "instance", "startup_timeout_seconds": 1})
    with pytest.raises((ValueError, TimeoutError)):
        mcp_container.probe_stdio(entry)


def test_preflight_blocks_agent_on_server_failure(tmp_path):
    runtime = tmp_path / "runtime.py"
    runtime.write_text(Path(mcp_container.__file__).read_text())
    (tmp_path / "source.json").write_text(json.dumps({"science": instance(command="/nonexistent", cwd="/tmp")}))
    marker = tmp_path / "agent-started"
    result = subprocess.run([sys.executable, str(runtime), "preflight", "--", sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"], capture_output=True, text=True)
    assert result.returncode == 78
    assert result.stderr.strip() == "ASIBENCH_MCP_STARTUP_FAILED: science"
    assert not marker.exists()


def test_native_codex_provider_is_kept_without_secret_cli_args(tmp_path):
    source = tmp_path / "native-home"
    source.mkdir()
    (source / "config.toml").write_text('model_provider = "test"\n[model_providers.test]\nname = "Test"\nbase_url = "https://example.test/v1"\nexperimental_bearer_token = "native-provider-secret"\n')
    with stage_os_mcp({"science": instance()}, {}, "codex", ["codex", "exec"], {"CODEX_HOME": str(source)}) as (cmd, env, mounts):
        assert "CODEX_HOME" not in env
        assert "native-provider-secret" not in repr(cmd)
        assert any("/home/agent/.codex/config.toml:ro" in arg for arg in mounts)
        root = Path(mounts[1].split(":")[0])
        assert "native-provider-secret" in (root / "config.toml").read_text()
        assert not (root / "config.toml").stat().st_mode & 0o077


def test_ambient_codex_config_cannot_add_other_mcp_servers(tmp_path):
    (tmp_path / "config.toml").write_text('[mcp_servers.ambient]\ncommand = "unwanted"\n')
    with pytest.raises(ValueError, match="ambient CODEX_HOME"):
        with stage_os_mcp({"science": instance()}, {}, "codex", ["codex", "exec"], {"CODEX_HOME": str(tmp_path)}):
            pytest.fail("ambient config must not be used")


def test_sandbox_config_not_executed_on_host(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"mcpServers": {"science": instance()}}))
    for cls in (ClaudeCodeCLIAdapter, CodexCLIAdapter):
        with pytest.raises(ValueError, match="requires --sandbox os"):
            cls(mcp_config=str(config)).setup({"sandbox": "none"})


def test_legacy_sse_rejected_before_startup(tmp_path):
    config = tmp_path / "config.json"
    entry = service()
    entry["type"] = "sse"
    config.write_text(json.dumps({"mcpServers": {"science": entry}}))
    with pytest.raises(ValueError, match="not legacy SSE"):
        load_mcp_config(config)


def test_difficulty_check_exposes_mcp_option():
    from click.testing import CliRunner
    from ai4sci_bench.cli import cli
    result = CliRunner().invoke(cli, ["difficulty-check", "--help"])
    assert result.exit_code == 0
    assert "--mcp-config" in result.output


@pytest.mark.parametrize("stream", [False, True])
def test_http_readiness_negotiates_session_and_closes_it(stream, monkeypatch):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import threading
    seen = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((request["method"], self.headers.get("Mcp-Session-Id"), self.headers.get("MCP-Protocol-Version")))
            if "id" not in request:
                self.send_response(202)
                self.end_headers()
                return
            result = ({"capabilities": {}, "serverInfo": {"name": "test", "version": "1"}, "protocolVersion": "2025-03-26"}
                      if request["method"] == "initialize" else {"tools": []})
            reply = json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}, indent=2)
            self.send_response(200)
            self.send_header("Mcp-Session-Id", "test-session")
            self.send_header("Content-Type", "text/event-stream" if stream else "application/json")
            self.end_headers()
            payload = "\n".join("data: " + line for line in reply.splitlines()) + "\n\n" if stream else reply
            self.wfile.write(payload.encode())
        def do_DELETE(self):
            seen.append(("DELETE", self.headers.get("Mcp-Session-Id"), self.headers.get("MCP-Protocol-Version")))
            self.send_response(204)
            self.end_headers()
        def log_message(self, *args):
            pass
    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("HTTP_PROXY", "http://invalid-proxy.test:1234")
    try:
        entry = service()
        entry["url"] = f"http://127.0.0.1:{httpd.server_port}/mcp"
        mcp_container.probe_http(entry)
    finally:
        httpd.shutdown()
        thread.join()
        httpd.server_close()
    assert [row[0] for row in seen] == ["initialize", "notifications/initialized", "tools/list", "DELETE"]
    assert seen[2][1:] == ("test-session", "2025-03-26")


def test_explicit_mcp_does_not_mount_host_claude_settings(tmp_path, monkeypatch):
    from ai4sci_bench.runner.os_sandbox import OSSandbox
    sandbox = OSSandbox(tmp_path)
    monkeypatch.setattr(sandbox, "_prepare_auth_mounts", lambda _: [
        "-v", "/host/credentials:/home/agent/.claude/.credentials.json:ro",
        "-v", "/host/settings:/home/agent/.claude/settings.json:ro",
    ])
    cmd = sandbox._build_docker_cmd(container_name="test", image="image", workspace=tmp_path, exec_args=["claude"], requires_network=False, requires_gpu=False, agent_type="claude_code", isolate_mcp=True)
    assert "/host/credentials:/home/agent/.claude/.credentials.json:ro" in cmd
    assert "/host/settings:/home/agent/.claude/settings.json:ro" not in cmd
    assert cmd.count("-v") == 2  # workspace plus auth, no orphan -v
