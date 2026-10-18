"""Opt-in real Docker tests. Uses a protocol client stand-in, not a live model."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid

import pytest

from ai4sci_bench.runner.os_sandbox import OSSandbox

BASE = os.environ.get("ASIBENCH_MCP_DOCKER_IMAGE")
pytestmark = pytest.mark.skipif(not BASE, reason="set ASIBENCH_MCP_DOCKER_IMAGE to a task image with /opt/venv/bin/python")


@pytest.fixture(scope="module")
def docker_image(tmp_path_factory):
    root = tmp_path_factory.mktemp("mcp-image")
    fixtures = Path(__file__).parent / "fixtures/mcp"
    for name in ("server.py", "agent.py"):
        shutil.copyfile(fixtures / name, root / name)
    (root / "Dockerfile").write_text(f'''FROM {BASE}
COPY server.py agent.py /opt/mcp/
HEALTHCHECK --interval=1s --timeout=2s --retries=5 CMD /opt/venv/bin/python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/health')"
CMD ["/opt/venv/bin/python", "/opt/mcp/server.py", "--http"]
''')
    image = "asibench-mcp-test:" + uuid.uuid4().hex[:12]
    subprocess.run(["docker", "build", "-q", "-t", image, str(root)], check=True, capture_output=True, timeout=180)
    yield image
    subprocess.run(["docker", "image", "rm", image], capture_output=True, timeout=30)


class ImageBuilder:
    def __init__(self, image):
        self.image = image
    def ensure_image(self, *args, **kwargs):
        return self.image
    def get_image_identity(self, image):
        return subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", image], text=True).strip()


@pytest.mark.parametrize("agent_type", ["codex", "claude_code"])
@pytest.mark.parametrize("deployment", ["instance", "service"])
def test_real_container_tool_call_and_cleanup(tmp_path, docker_image, monkeypatch, agent_type, deployment):
    monkeypatch.delenv("AI4SCI_DOCKER_NETWORK", raising=False)
    sandbox = OSSandbox(tmp_path, image_builder=ImageBuilder(docker_image))
    # No host credentials needed or exposed by this test.
    monkeypatch.setattr(sandbox, "_prepare_auth_mounts", lambda _: [])
    metadata = {}
    if deployment == "instance":
        entry = {"command": "/opt/venv/bin/python", "args": ["/opt/mcp/server.py"], "sandbox": {"deployment": "instance"}}
    else:
        metadata = {"runtime": {"services": [{"name": "science", "image": docker_image, "port": 8765}]}}
        entry = {"url": "http://science:8765/mcp", "sandbox": {"deployment": "service", "service": "science"}}
    cmd = ["/opt/venv/bin/python", "/opt/mcp/agent.py"]
    if agent_type == "claude_code":
        cmd += ["--mcp-config", "/host/mcp.json"]
    before = subprocess.check_output(["docker", "ps", "-aq", "--filter", "name=ai4sci-"], text=True).splitlines()
    runtime = {}
    success, log, stdout, stderr, identity = sandbox.run_agent(metadata, workspace=tmp_path, agent_cmd=cmd, timeout=30, agent_type=agent_type, extra_env={"ASIBENCH_TEST_AGENT_SECRET": "never-forward-to-server"}, mcp_servers={"science": entry}, mcp_runtime=runtime)
    assert success, log
    assert "MCP_PROTOCOL_CLIENT_PASS" in stdout
    assert json.loads((tmp_path / "result.json").read_text())["answer"] == 42
    after = subprocess.check_output(["docker", "ps", "-aq", "--filter", "name=ai4sci-"], text=True).splitlines()
    assert set(after) <= set(before)
    assert identity.startswith("sha256:")
    assert runtime["instance_image"] == identity
    if deployment == "service":
        assert runtime["services"]["science"]["image_identity"].startswith("sha256:")


def test_real_preflight_failure_does_not_start_agent(tmp_path, docker_image, monkeypatch):
    sandbox = OSSandbox(tmp_path, image_builder=ImageBuilder(docker_image))
    monkeypatch.setattr(sandbox, "_prepare_auth_mounts", lambda _: [])
    entry = {"command": "/opt/does-not-exist", "sandbox": {"deployment": "instance"}}
    success, log, *_ = sandbox.run_agent({}, workspace=tmp_path, agent_cmd=["/opt/venv/bin/python", "/opt/mcp/agent.py"], timeout=10, agent_type="codex", mcp_servers={"science": entry})
    assert not success
    assert "ASIBENCH_MCP_STARTUP_FAILED: science" in log
    assert not (tmp_path / "result.json").exists()


def test_real_timeout_cleans_service_and_network(tmp_path, docker_image, monkeypatch):
    monkeypatch.delenv("AI4SCI_DOCKER_NETWORK", raising=False)
    sandbox = OSSandbox(tmp_path, image_builder=ImageBuilder(docker_image))
    monkeypatch.setattr(sandbox, "_prepare_auth_mounts", lambda _: [])
    original_run = subprocess.run
    def short_agent_timeout(command, *args, **kwargs):
        if command[:2] == ["docker", "start"]:
            kwargs["timeout"] = 0.5
        return original_run(command, *args, **kwargs)
    metadata = {"runtime": {"services": [{"name": "science", "image": docker_image, "port": 8765}]}}
    entry = {"url": "http://science:8765/mcp", "sandbox": {"deployment": "service", "service": "science"}}
    before = subprocess.check_output(["docker", "network", "ls", "--format", "{{.Name}}"], text=True).splitlines()
    monkeypatch.setattr(subprocess, "run", short_agent_timeout)
    success, log, *_ = sandbox.run_agent(metadata, workspace=tmp_path, agent_cmd=["/opt/venv/bin/python", "/opt/mcp/agent.py", "--hang-agent"], timeout=1, agent_type="codex", mcp_servers={"science": entry})
    assert not success and "OS sandbox agent execution timed out" in log
    after = subprocess.check_output(["docker", "network", "ls", "--format", "{{.Name}}"], text=True).splitlines()
    assert set(after) <= set(before)
    assert not (tmp_path / "result.json").exists()


def test_actual_codex_cli_accepts_generated_config(tmp_path, docker_image, monkeypatch):
    available = subprocess.run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "/bin/sh", docker_image, "-c", "command -v codex"], capture_output=True, timeout=30)
    if available.returncode:
        pytest.skip("selected base image has no Codex CLI")
    sandbox = OSSandbox(tmp_path, image_builder=ImageBuilder(docker_image))
    monkeypatch.setattr(sandbox, "_prepare_auth_mounts", lambda _: [])
    entry = {"command": "/opt/venv/bin/python", "args": ["/opt/mcp/server.py"], "sandbox": {"deployment": "instance"}}
    runtime = {}
    success, log, stdout, *_ = sandbox.run_agent({}, workspace=tmp_path, agent_cmd=["codex", "mcp", "list", "--json"], timeout=20, agent_type="codex", mcp_servers={"science": entry}, mcp_runtime=runtime)
    assert success, log
    listed = json.loads(stdout)
    assert [server["name"] for server in listed] == ["science"]
    assert runtime["servers"]["science"]["deployment"] == "instance"
    assert runtime["instance_image"].startswith("sha256:")
