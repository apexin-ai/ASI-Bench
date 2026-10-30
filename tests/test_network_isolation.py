"""Model-only relay and OS network boundary regression tests."""
from pathlib import Path
from unittest.mock import MagicMock

from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter


def test_restricted_claude_os_selects_model_only_by_default(tmp_path):
    adapter = ClaudeCodeCLIAdapter(api_key="fixture-key")
    adapter.setup({"sandbox": "os", "repo_root": str(tmp_path)})
    assert adapter.effective_network_policy == "model-only"
    adapter.teardown()

import concurrent.futures
from contextlib import contextmanager
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import socket
import subprocess
import threading

import pytest

from ai4sci_bench.runner.model_relay import model_relay
from ai4sci_bench.runner.os_sandbox import OSSandbox


class UnixConnection(http.client.HTTPConnection):
    def __init__(self, path):
        super().__init__("localhost", timeout=5)
        self.path = str(path)

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(5)
        self.sock.connect(self.path)


@contextmanager
def upstream_server():
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            requests.append(("GET", self.path, dict(self.headers), b""))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"external-canary")

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            requests.append(("POST", self.path, dict(self.headers), body))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b'data: {"type":"message_stop"}\n\n')
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1], requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def call_relay(root, method="POST", path="/v1/messages", payload=None, headers=None):
    connection = UnixConnection(root / "model.sock")
    body = json.dumps({"model": "fixture", "max_tokens": 16, "messages": []} if payload is None else payload)
    # Send the complete fixture request together: an immediate policy rejection
    # may close the connection before HTTPConnection's separate body write.
    # That is correct server behavior, not a failed isolation assertion.
    encoded = body.encode() if method == "POST" else b""
    request_headers = {"Host": "localhost", "Content-Type": "application/json",
                       "Content-Length": str(len(encoded)), **(headers or {})}
    wire = f"{method} {path} HTTP/1.1\r\n" + "".join(
        f"{key}: {value}\r\n" for key, value in request_headers.items()) + "\r\n"
    connection.connect()
    connection.sock.sendall(wire.encode() + encoded)
    response = http.client.HTTPResponse(connection.sock)
    response.begin()
    result = response.status, response.read()
    response.close()
    connection.close()
    return result


def test_relay_pins_target_auth_and_preserves_streaming(tmp_path):
    runtime = {}
    with upstream_server() as (port, requests):
        with model_relay({"ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}/prefix", "ANTHROPIC_API_KEY": "host-only-secret"}, runtime) as root:
            status, body = call_relay(root, headers={"Host": "evil.example", "x-api-key": "forged", "Authorization": "Bearer forged"})
            assert status == 200
            assert body == b'data: {"type":"message_stop"}\n\n'
            assert requests[0][1] == "/prefix/v1/messages"
            assert requests[0][2]["x-api-key"] == "host-only-secret"
            assert "Authorization" not in requests[0][2]
            assert "host-only-secret" not in json.dumps(runtime)
        assert not root.exists()
    assert runtime["forwarded_requests"] == 1


@pytest.mark.parametrize("method,path,payload", [
    ("GET", "/", {}), ("CONNECT", "example.com:443", {}),
    ("POST", "http://example.com/v1/messages", {}),
    ("POST", "/v1/messages?url=http://example.com", {}),
    ("POST", "/v1/../files", {}), ("POST", "/v1/files", {}),
    ("POST", "/v1/messages", {"messages": [], "tools": [{"type": "web_search_20250305", "name": "web_search"}]}),
    ("POST", "/v1/messages", {"messages": [], "mcp_servers": [{"url": "http://example.com"}]}),
    ("POST", "/v1/messages", {"messages": [{"content": [{"type": "image", "source": {"type": "url", "url": "https://example.com"}}]}]}),
])
def test_relay_denies_forwarding_bypasses(method, path, payload):
    with upstream_server() as (port, requests):
        runtime = {}
        with model_relay({"ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}", "ANTHROPIC_API_KEY": "secret"}, runtime) as root:
            assert call_relay(root, method, path, payload)[0] in {400, 403}
        assert requests == []
        assert runtime["denied_requests"] == 1


@pytest.mark.parametrize("mode,expected", [("search", "inherit"), ("unrestricted", "inherit"), ("restricted", "model-only")])
def test_claude_network_default_follows_tool_mode(tmp_path, mode, expected):
    adapter = ClaudeCodeCLIAdapter(tool_mode=mode)
    adapter.setup({"sandbox": "os", "repo_root": str(tmp_path)})
    assert adapter.effective_network_policy == expected
    adapter.teardown()


def test_explicit_legacy_policy_and_non_os_rejection(tmp_path):
    adapter = ClaudeCodeCLIAdapter(network_policy="inherit")
    adapter.setup({"sandbox": "os", "repo_root": str(tmp_path)})
    assert adapter.effective_network_policy == "inherit"
    adapter.teardown()
    with pytest.raises(ValueError, match="requires --sandbox os"):
        ClaudeCodeCLIAdapter(network_policy="model-only").setup({"sandbox": "none"})


@pytest.mark.parametrize("conflict", ["host-network", "bridge-network", "services", "service-mcp", "mount"])
def test_conflicts_fail_before_image_build(tmp_path, monkeypatch, conflict):
    builder = MagicMock()
    sandbox = OSSandbox(tmp_path, image_builder=builder)
    metadata, options = {}, {}
    if conflict.endswith("network"):
        monkeypatch.setenv("AI4SCI_DOCKER_NETWORK", "host" if conflict == "host-network" else "offline-bridge")
    elif conflict == "services":
        # No parsing/building of task services should precede the rejection.
        monkeypatch.setattr(sandbox, "_task_services", lambda _: [{"name": "external"}])
    elif conflict == "service-mcp":
        options["mcp_servers"] = {"fetch": {"sandbox": {"deployment": "service"}}}
    else:
        options["extra_mounts"] = ["-v", "/var/run/docker.sock:/var/run/docker.sock"]
    with pytest.raises(ValueError, match="model-only"):
        sandbox.run_agent(metadata, agent_cmd=["true"], workspace=tmp_path, timeout=5,
                          agent_type="claude_code", network_policy="model-only", **options)
    builder.ensure_image.assert_not_called()


def test_model_only_command_removes_network_capabilities_and_auth_mounts(tmp_path, monkeypatch):
    sandbox = OSSandbox(tmp_path, image_builder=MagicMock())
    monkeypatch.delenv("AI4SCI_DOCKER_NETWORK", raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://host-proxy:8080")
    monkeypatch.setattr(sandbox, "_prepare_auth_mounts", lambda _: pytest.fail("must not mount login/settings"))
    command = sandbox._build_docker_cmd(container_name="probe", image="fixture", workspace=tmp_path,
        exec_args=["true"], requires_network=False, requires_gpu=False, agent_type="claude_code", model_only=True)
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert "no-new-privileges:true" in command
    assert "http://host-proxy:8080" not in " ".join(command)


@pytest.fixture(scope="module")
def probe_image():
    if os.environ.get("ASIBENCH_NETWORK_DOCKER") != "1":
        pytest.skip("Set ASIBENCH_NETWORK_DOCKER=1 for actual Docker isolation probes")
    override = os.environ.get("ASIBENCH_NETWORK_AGENT_IMAGE")
    if override:
        yield override
        return
    import uuid
    tag = "asibench-network-probe:" + uuid.uuid4().hex[:12]
    # No downloads: a local, stdlib-only image with the normal agent UID.
    dockerfile = "FROM python:3.12-slim\nRUN python -m venv /opt/venv && useradd -m -u 1000 agent\nUSER agent\n"
    built = subprocess.run(["docker", "build", "--network", "none", "-t", tag, "-"], input=dockerfile, text=True, capture_output=True, timeout=120)
    assert built.returncode == 0, built.stderr
    try:
        yield tag
    finally:
        subprocess.run(["docker", "image", "rm", tag], capture_output=True, timeout=30)


def test_docker_model_relay_works_while_all_ip_egress_is_blocked(probe_image, tmp_path, monkeypatch):
    monkeypatch.delenv("AI4SCI_DOCKER_NETWORK", raising=False)
    bridge = json.loads(subprocess.check_output(["docker", "network", "inspect", "bridge"]))[0]
    gateway = bridge["IPAM"]["Config"][0]["Gateway"]
    builder = MagicMock()
    builder.ensure_image.return_value = probe_image
    builder.get_image_identity.return_value = subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", probe_image], text=True).strip()
    sandbox = OSSandbox(tmp_path, image_builder=builder)
    with upstream_server() as (port, requests):
        canary = f"http://{gateway}:{port}/canary"
        control = subprocess.run(["docker", "run", "--rm", "--network", "bridge", probe_image,
            "/opt/venv/bin/python", "-c", f"import urllib.request; print(urllib.request.urlopen('{canary}', timeout=5).read().decode())"], capture_output=True, text=True, timeout=20)
        assert control.returncode == 0 and "external-canary" in control.stdout, control.stderr
        requests.clear()
        # Deliberately inherited host proxies cannot create container egress.
        monkeypatch.setenv("HTTPS_PROXY", f"http://{gateway}:{port}")
        program = r'''
import json,os,socket,subprocess,sys,urllib.request,urllib.error
from pathlib import Path
assert 'host-only-secret' not in repr(dict(os.environ))
assert not any(k.lower() in {'http_proxy','https_proxy','all_proxy','no_proxy'} for k in os.environ)
interfaces={line.split(':')[0].strip() for line in Path('/proc/net/dev').read_text().splitlines() if ':' in line}
assert interfaces == {'lo'}, interfaces
Path('/workspace/computed.txt').write_text(str(sum(range(100))))
request=urllib.request.Request(os.environ['ANTHROPIC_BASE_URL']+'/v1/messages',data=json.dumps({'model':'fixture','messages':[],'max_tokens':16,'stream':True}).encode(),headers={'Content-Type':'application/json'})
assert b'message_stop' in urllib.request.urlopen(request,timeout=10).read()
try:
 urllib.request.urlopen(os.environ['ANTHROPIC_BASE_URL']+'/v1/files',timeout=5)
 raise AssertionError('arbitrary relay route was accessible')
except urllib.error.HTTPError as e:
 assert e.code==403
for family,address in [(socket.AF_INET,('GATEWAY',PORT)),(socket.AF_INET6,('::ffff:GATEWAY',PORT)),(socket.AF_INET6,('2001:db8::1',PORT))]:
 with socket.socket(family,socket.SOCK_STREAM) as s:
  s.settimeout(2)
  assert s.connect_ex(address)!=0, address
# Subprocess/Python HTTP, DNS hostnames, and shell children share the same boundary.
for target in ['http://GATEWAY:PORT/canary','http://example.com/','https://example.com/']:
 child=subprocess.run([sys.executable,'-c',"import urllib.request; urllib.request.urlopen("+repr(target)+",timeout=2).read()"],capture_output=True,timeout=5)
 assert child.returncode!=0,target
# When the full task image includes curl, cover its subprocess path too.
import shutil
if shutil.which('curl'):
 result=subprocess.run(['curl','--noproxy','*','--max-time','2','http://GATEWAY:PORT/canary'],capture_output=True,timeout=5)
 assert result.returncode!=0
print('isolation-ok')
'''.replace('GATEWAY', gateway).replace('PORT', str(port))
        def attempt(index):
            workspace = tmp_path / f"attempt-{index}"
            workspace.mkdir()
            runtime = {}
            result = sandbox.run_agent({}, agent_cmd=["/opt/venv/bin/python", "-c", program],
                workspace=workspace, timeout=30, agent_type="claude_code", network_policy="model-only",
                extra_env={"ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}", "ANTHROPIC_API_KEY": "host-only-secret"},
                network_runtime=runtime)
            assert result[0], result[1]
            assert "isolation-ok" in result[2]
            assert (workspace / "computed.txt").read_text() == "4950"
            assert runtime["enforced"] is True
            assert runtime["startup_check"] == "docker-inspect-passed"
            assert runtime["forwarded_requests"] == runtime["denied_requests"] == 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(attempt, range(2)))
        assert len(requests) == 2
        assert all(method == "POST" and path == "/v1/messages" for method,path,_,_ in requests)


def test_relay_does_not_follow_redirects():
    class Redirect(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass
        def do_POST(self):
            self.send_response(302)
            self.send_header("Location", self.server.destination)
            self.end_headers()
    server = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with upstream_server() as (port, requests):
            server.destination = f"http://127.0.0.1:{port}/canary"
            with model_relay({"ANTHROPIC_BASE_URL": f"http://127.0.0.1:{server.server_port}", "ANTHROPIC_API_KEY": "secret"}, {}) as root:
                assert call_relay(root)[0] == 502
            assert requests == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_relay_rejects_ambiguous_request_framing():
    with upstream_server() as (port, requests):
        with model_relay({"ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}", "ANTHROPIC_API_KEY": "secret"}, {}) as root:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(5)
                sock.connect(str(root / "model.sock"))
                sock.sendall(b'POST /v1/messages HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\nContent-Length: 200\r\n\r\n{}')
                assert b"400" in sock.recv(1024)
        assert requests == []


def test_real_claude_cli_uses_bash_offline_and_messages_relay(tmp_path, monkeypatch):
    image = os.environ.get("ASIBENCH_NETWORK_AGENT_IMAGE")
    if os.environ.get("ASIBENCH_NETWORK_DOCKER") != "1" or not image:
        pytest.skip("Requires ASIBENCH_NETWORK_AGENT_IMAGE containing Claude and curl")
    monkeypatch.delenv("AI4SCI_DOCKER_NETWORK", raising=False)
    gateway = json.loads(subprocess.check_output(["docker", "network", "inspect", "bridge"]))[0]["IPAM"]["Config"][0]["Gateway"]
    calls = []
    with upstream_server() as (canary_port, canary_requests):
        # Both Python and curl are requested by an actual Messages tool_use.
        command = f"python -c \"from pathlib import Path; Path('answer.txt').write_text(str(6*7))\"; curl --noproxy '*' --max-time 2 http://{gateway}:{canary_port}/canary; echo PROBE_DONE"
        class Messages(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                calls.append(body)
                completed = any(isinstance(m.get("content"), list) and any(isinstance(b,dict) and b.get("type")=="tool_result" for b in m["content"]) for m in body.get("messages",[]))
                if self.path.split("?")[0].endswith("count_tokens"):
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"input_tokens":32}')
                    return
                block = {"type":"text","text":"The offline calculation is complete."} if completed else {"type":"tool_use","id":"tool_probe","name":"Bash","input":{"command":command,"description":"Check network isolation and calculate locally"}}
                stop = "end_turn" if completed else "tool_use"
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                events = [
                    ("message_start", {"type":"message_start","message":{"id":"msg_probe","type":"message","role":"assistant","model":body.get("model","fixture"),"content":[],"stop_reason":None,"stop_sequence":None,"usage":{"input_tokens":32,"output_tokens":1}}}),
                    ("content_block_start", {"type":"content_block_start","index":0,"content_block":({**block,"input":{}} if not completed else {"type":"text","text":""})}),
                    ("content_block_delta", {"type":"content_block_delta","index":0,"delta":({"type":"input_json_delta","partial_json":json.dumps(block["input"])} if not completed else {"type":"text_delta","text":block["text"]})}),
                    ("content_block_stop", {"type":"content_block_stop","index":0}),
                    ("message_delta", {"type":"message_delta","delta":{"stop_reason":stop,"stop_sequence":None},"usage":{"output_tokens":16}}),
                    ("message_stop", {"type":"message_stop"}),
                ]
                for kind,event in events:
                    self.wfile.write(f"event: {kind}\ndata: {json.dumps(event)}\n\n".encode())
                    self.wfile.flush()
        server = ThreadingHTTPServer(("127.0.0.1",0),Messages)
        thread = threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        try:
            builder=MagicMock()
            builder.ensure_image.return_value=image
            builder.get_image_identity.return_value="fixture-claude"
            sandbox=OSSandbox(tmp_path,image_builder=builder)
            runtime={}
            outcome=sandbox.run_agent({},agent_cmd=["claude","--print","--verbose","--output-format","stream-json","--dangerously-skip-permissions","--tools","Bash","--model","claude-sonnet-4-6","Calculate locally and verify isolation."],workspace=tmp_path,timeout=90,agent_type="claude_code",network_policy="model-only",extra_env={"ANTHROPIC_BASE_URL":f"http://127.0.0.1:{server.server_port}","ANTHROPIC_API_KEY":"fixture-not-a-real-key"},network_runtime=runtime)
            assert outcome[0],outcome[1]
            assert (tmp_path/"answer.txt").read_text()=="42",outcome[1]
            assert any("PROBE_DONE" in json.dumps(call) for call in calls)
            assert canary_requests==[]
            assert runtime["enforced"] is True
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


def test_adapter_persists_rejected_policy_without_running_agent(tmp_path, monkeypatch):
    from ai4sci_bench.core.types import TaskInstance, PromptLevel, RunStatus
    adapter = ClaudeCodeCLIAdapter(api_key="fixture-key")
    adapter.setup({"sandbox": "os", "repo_root": str(tmp_path)})
    monkeypatch.setattr(adapter, "_build_os_agent_cmd", lambda _: ["true"])
    monkeypatch.setenv("AI4SCI_DOCKER_NETWORK", "host")
    instance = TaskInstance(task_id="fixture", instance_id="fixture__seed31415", task_dir=tmp_path,
        workspace_dir=tmp_path, reference_dir=tmp_path, prompt_level=PromptLevel.B1, parameters={}, metadata={})
    output = adapter.solve(instance)
    assert output.status == RunStatus.FAILED
    assert output.network_runtime["effective"] == "model-only"
    assert output.network_runtime["enforced"] is False
    assert output.network_runtime["startup_check"] == "rejected"
    assert "AI4SCI_DOCKER_NETWORK" in output.error_message
    adapter.teardown()


def test_messages_beta_query_is_the_only_allowed_query():
    with upstream_server() as (port, requests):
        with model_relay({"ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}", "ANTHROPIC_API_KEY": "secret"}, {}) as root:
            assert call_relay(root, path="/v1/messages?beta=true")[0] == 200
            assert call_relay(root, path="/v1/messages?beta=true&url=https://example.com")[0] == 403
        assert len(requests) == 1


def test_docker_preflight_rejects_network_mismatch_before_start(tmp_path, monkeypatch):
    from subprocess import CompletedProcess
    sandbox = OSSandbox(tmp_path, image_builder=MagicMock())
    monkeypatch.delenv("AI4SCI_DOCKER_NETWORK", raising=False)
    commands = []
    def docker(command, **_kwargs):
        commands.append(command)
        if command[:2] == ["docker", "inspect"]:
            return CompletedProcess(command, 0, json.dumps([{"HostConfig": {
                "NetworkMode": "bridge", "Privileged": False, "CapDrop": ["ALL"],
                "SecurityOpt": ["no-new-privileges:true"],
            }}]), "")
        return CompletedProcess(command, 0, "", "")
    monkeypatch.setattr("ai4sci_bench.runner.os_sandbox.subprocess.run", docker)
    runtime = {"enforced": False}
    outcome = sandbox._run_task_container({}, services=[], image="fixture", image_identity="fixture", container_name="probe",
        workspace=tmp_path, exec_args=["true"], timeout=5, requires_network=False, requires_gpu=False,
        allow_external_tools=False, agent_type="claude_code", model_only=True, network_runtime=runtime, timeout_label="fixture")
    assert outcome[0] is False
    assert "isolation preflight failed" in outcome[1]
    assert runtime["enforced"] is False
    assert not any(command[:2] == ["docker", "start"] for command in commands)
    assert any(command[:2] == ["docker", "rm"] for command in commands)


def test_relay_cleans_up_after_aborted_agent():
    root = None
    with pytest.raises(RuntimeError, match="abort"):
        with model_relay({"ANTHROPIC_API_KEY": "fixture"}, {}) as root:
            assert (root / "model.sock").is_socket()
            raise RuntimeError("abort")
    assert not root.exists()


def test_local_tool_schema_source_property_does_not_count_as_url_attachment():
    with upstream_server() as (port, requests):
        with model_relay({"ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}", "ANTHROPIC_API_KEY": "secret"}, {}) as root:
            payload = {"messages": [], "tools": [{"name": "offline_science", "input_schema": {
                "type": "object", "properties": {"source": {"type": "string"}},
            }}]}
            assert call_relay(root, payload=payload)[0] == 200
        assert len(requests) == 1


def test_network_policy_audit_survives_result_serialization(tmp_path):
    from ai4sci_bench.core.types import AgentOutput, EvalResult, PromptLevel, RunStatus
    from ai4sci_bench.runner.orchestrator import BenchmarkOrchestrator, RunConfig
    from tests.test_runner import DummyAgent
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    orchestrator = BenchmarkOrchestrator(RunConfig(agent=DummyAgent(), tasks_dir=str(tasks), output_dir=str(tmp_path / "results")))
    runtime = {"policy": "model-only-unix-v1", "effective": "model-only", "enforced": True,
        "startup_check": "docker-inspect-passed", "forwarded_requests": 2, "denied_requests": 1,
        "allowed_routes": ["/v1/messages", "/v1/messages/count_tokens", "/private/host/path"], "api_key": "never-persist"}
    result = EvalResult(instance_id="fixture__seed31415", task_id="fixture", prompt_level=PromptLevel.B1,
        agent_name="fixture", parameters={}, gate_results=[], gates_passed=True, score_results=[], final_score=0,
        agent_output=AgentOutput(instance_id="fixture__seed31415", output_dir=tmp_path, code_files=[], data_files=[],
            log="fixture", execution_time_seconds=0, status=RunStatus.COMPLETED, network_runtime=runtime))
    orchestrator._save_result(result)
    saved = json.loads((tmp_path / "results/fixture/fixture__seed31415__b1.json").read_text())
    audit = saved["provenance"]["network"]
    assert audit == saved["agent_output"]["network_runtime"]
    assert audit["allowed_routes"] == ["/v1/messages", "/v1/messages/count_tokens"]
    assert audit["enforced"] is True
    assert audit["api_key"] == "<redacted>"
    assert "never-persist" not in json.dumps(saved)
