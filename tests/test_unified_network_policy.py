"""All OS harnesses must make the same explicit network-policy decision."""

import pytest
from pathlib import Path

from ai4sci_bench.adapters import (
    ClaudeCodeCLIAdapter,
    CodexCLIAdapter,
    PiCLIAdapter,
    OpenCodeCLIAdapter,
    KimiCodeCLIAdapter,
    MiMoCodeCLIAdapter,
    AntigravityCLIAdapter,
    OpenHandsAdapter,
)

HARNESSES = [
    ClaudeCodeCLIAdapter,
    CodexCLIAdapter,
    PiCLIAdapter,
    OpenCodeCLIAdapter,
    KimiCodeCLIAdapter,
    MiMoCodeCLIAdapter,
    AntigravityCLIAdapter,
    OpenHandsAdapter,
]


@pytest.mark.parametrize("adapter_class", HARNESSES)
@pytest.mark.parametrize(
    "mode,expected",
    [("restricted", "model-only"), ("search", "inherit"), ("unrestricted", "inherit")],
)
def test_all_os_harnesses_select_same_default(adapter_class, mode, expected, tmp_path):
    adapter = adapter_class(tool_mode=mode)
    try:
        adapter.setup({"sandbox": "os", "repo_root": str(tmp_path)})
        assert adapter.network_policy == "auto"
        assert adapter.effective_network_policy == expected
    finally:
        adapter.teardown()


@pytest.mark.parametrize("adapter_class", HARNESSES)
def test_explicit_policy_is_validated_consistently(adapter_class, tmp_path):
    with pytest.raises(ValueError, match="network_policy"):
        adapter_class(network_policy="open")
    adapter = adapter_class(network_policy="model-only")
    with pytest.raises(ValueError, match="requires --sandbox os"):
        adapter.setup({"sandbox": "none", "repo_root": str(tmp_path)})
    adapter.teardown()
    adapter = adapter_class(network_policy="inherit")
    adapter.setup({"sandbox": "os", "repo_root": str(tmp_path)})
    assert adapter.effective_network_policy == "inherit"
    adapter.teardown()


from unittest.mock import MagicMock
from ai4sci_bench.core.types import TaskInstance, PromptLevel, RunStatus
from ai4sci_bench.runner.model_relay import model_relay, PROTOCOL_ROUTES
from tests.test_network_isolation import upstream_server, call_relay


@pytest.mark.parametrize(
    "protocol,path,payload",
    [
        (
            "openai",
            "/v1/chat/completions",
            {
                "model": "fixture",
                "messages": [],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "local", "parameters": {"type": "object"}},
                    }
                ],
            },
        ),
        (
            "openai_responses",
            "/v1/responses",
            {
                "model": "fixture",
                "input": [],
                "tools": [
                    {"type": "custom", "name": "local", "format": {"type": "text"}}
                ],
                "store": False,
            },
        ),
        (
            "openai_responses",
            "/v1/responses/compact",
            {"model": "fixture", "input": []},
        ),
    ],
)
def test_openai_relay_fixed_route_and_host_key(protocol, path, payload):
    with upstream_server() as (port, requests):
        runtime = {}
        with model_relay(
            {},
            runtime,
            {
                "protocol": protocol,
                "base_url": f"http://127.0.0.1:{port}/api/v1",
                "api_key": "host-secret",
            },
        ) as root:
            assert (
                call_relay(
                    root,
                    path=path,
                    payload=payload,
                    headers={"Authorization": "Bearer forged"},
                )[0]
                == 200
            )
            assert "host-secret" not in (root / "client.json").read_text()
            assert call_relay(root, path="/v1/files", payload=payload)[0] == 403
            assert (
                call_relay(
                    root, path=path + "?url=https://example.com", payload=payload
                )[0]
                == 403
            )
        assert len(requests) == 1
        assert requests[0][1] == "/api" + path
        assert requests[0][2]["Authorization"] == "Bearer host-secret"
        assert runtime["allowed_routes"] == sorted(PROTOCOL_ROUTES[protocol])


@pytest.mark.parametrize(
    "protocol,path,content_key",
    [
        ("openai", "/v1/chat/completions", "messages"),
        ("openai_responses", "/v1/responses", "input"),
    ],
)
@pytest.mark.parametrize(
    "extra",
    [
        {"tools": [{"type": "web_search_preview"}]},
        {"tools": [{"type": "mcp", "server_url": "https://example.com"}]},
        {"tools": [{"type": "file_search", "vector_store_ids": ["vs_remote"]}]},
        {"web_search_options": {}},
        {"previous_response_id": "resp_remote"},
        {"conversation": "conv_remote"},
        {"store": True},
        {"tool_choice": {"type": "web_search"}},
    ],
)
def test_hosted_retrieval_cannot_cross_relay(protocol, path, content_key, extra):
    with upstream_server() as (port, requests):
        with model_relay(
            {},
            {},
            {
                "protocol": protocol,
                "base_url": f"http://127.0.0.1:{port}/v1",
                "api_key": "secret",
            },
        ) as root:
            assert (
                call_relay(root, path=path, payload={content_key: [], **extra})[0]
                == 400
            )
        assert requests == []


@pytest.mark.parametrize(
    "content",
    [
        {"type": "input_image", "image_url": "https://example.com/image.png"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
        {"type": "input_file", "file_url": "https://example.com/file.pdf"},
        {"type": "file", "file": {"file_id": "file_remote"}},
        {"type": "item_reference", "id": "remote"},
    ],
)
def test_remote_content_rejected(content):
    from ai4sci_bench.runner.model_relay import validate_payload

    with pytest.raises(ValueError):
        validate_payload(
            {"input": [{"role": "user", "content": [content]}]}, "openai_responses"
        )


@pytest.mark.parametrize(
    "adapter_class,agent_type",
    [
        (CodexCLIAdapter, "codex"),
        (PiCLIAdapter, "pi"),
        (OpenCodeCLIAdapter, "opencode"),
        (KimiCodeCLIAdapter, "kimi_code"),
        (MiMoCodeCLIAdapter, "mimo_code"),
        (OpenHandsAdapter, "openhands"),
    ],
)
def test_adapters_forward_policy_and_audit_without_secret_configs(
    adapter_class, agent_type, tmp_path
):
    adapter = adapter_class(api_key="host-only-secret")
    adapter.setup({"sandbox": "os", "repo_root": str(tmp_path)})
    (tmp_path / "prompt.md").write_text("Calculate 6*7 locally.")
    instance = TaskInstance(
        task_id="fixture",
        instance_id="fixture__seed31415",
        task_dir=tmp_path,
        workspace_dir=tmp_path,
        reference_dir=tmp_path,
        prompt_level=PromptLevel.B1,
        parameters={},
        metadata={},
    )

    def run(**kwargs):
        assert kwargs["network_policy"] == "model-only"
        assert kwargs["model_connection"]["agent_type"] == agent_type
        assert kwargs["model_connection"]["api_key"] == "host-only-secret"
        assert not kwargs.get("extra_mounts")
        kwargs["network_runtime"].update(enforced=True, startup_check="fixture")
        return True, "fixture", None, None, "fixture-image"

    adapter._os_sandbox.run_agent = run
    try:
        output = adapter.solve(instance)
        assert output.network_runtime["enforced"] is True
        assert output.status == RunStatus.COMPLETED
        for path in tmp_path.rglob("*"):
            if path.is_file():
                assert "host-only-secret" not in path.read_text(errors="replace")
    finally:
        adapter.teardown()


@pytest.mark.parametrize("adapter_class", HARNESSES[1:])
def test_missing_transport_or_auth_never_falls_back(
    adapter_class, tmp_path, monkeypatch
):
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "MOONSHOT_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    adapter = adapter_class()
    # OpenHands resolves its fallback key in __init__; all envs removed first.
    adapter.setup({"sandbox": "os", "repo_root": str(tmp_path)})
    adapter._os_sandbox.run_agent = MagicMock()
    runtime = {}
    result = adapter._run_os_agent(
        network_runtime=runtime,
        agent_type={
            CodexCLIAdapter: "codex",
            PiCLIAdapter: "pi",
            OpenCodeCLIAdapter: "opencode",
            KimiCodeCLIAdapter: "kimi_code",
            MiMoCodeCLIAdapter: "mimo_code",
            AntigravityCLIAdapter: "agy",
            OpenHandsAdapter: "openhands",
        }[adapter_class],
    )
    assert result[0] is False
    assert runtime["enforced"] is False and runtime["startup_check"] == "rejected"
    adapter._os_sandbox.run_agent.assert_not_called()
    adapter.teardown()


import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ai4sci_bench.runner.os_sandbox import OSSandbox
from tests.test_network_isolation import probe_image


@pytest.mark.parametrize(
    "protocol,path,content_key",
    [
        ("openai", "/v1/chat/completions", "messages"),
        ("openai_responses", "/v1/responses", "input"),
    ],
)
def test_openai_docker_network_boundary(
    probe_image, tmp_path, monkeypatch, protocol, path, content_key
):
    monkeypatch.delenv("AI4SCI_DOCKER_NETWORK", raising=False)
    builder = MagicMock()
    builder.ensure_image.return_value = probe_image
    builder.get_image_identity.return_value = "fixture-openai"
    sandbox = OSSandbox(tmp_path, image_builder=builder)
    gateway = json.loads(
        subprocess.check_output(["docker", "network", "inspect", "bridge"])
    )[0]["IPAM"]["Config"][0]["Gateway"]
    with upstream_server() as (port, requests):
        control = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "bridge",
                probe_image,
                "/opt/venv/bin/python",
                "-c",
                f"import urllib.request; print(urllib.request.urlopen('http://{gateway}:{port}/canary',timeout=5).read())",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert control.returncode == 0 and "external-canary" in control.stdout
        requests.clear()
        program = f"""
import os,json,urllib.request,subprocess,sys
from pathlib import Path
assert 'host-only-secret' not in repr(dict(os.environ))
assert {{line.split(':')[0].strip() for line in Path('/proc/net/dev').read_text().splitlines() if ':' in line}}=={{'lo'}}
req=urllib.request.Request(os.environ['OPENAI_BASE_URL']+{path[3:]!r},data=json.dumps({{{content_key!r}:[],'model':'fixture','store':False}}).encode(),headers={{'Content-Type':'application/json'}})
assert urllib.request.urlopen(req,timeout=10).status==200
child=subprocess.run([sys.executable,'-c',"import urllib.request; urllib.request.urlopen('http://{gateway}:{port}/canary',timeout=2).read()"],capture_output=True,timeout=5)
assert child.returncode!=0
Path('answer.txt').write_text('42')
"""
        runtime = {}
        outcome = sandbox.run_agent(
            {},
            agent_cmd=["/opt/venv/bin/python", "-c", program],
            workspace=tmp_path,
            timeout=30,
            agent_type="mimo_code",
            network_policy="model-only",
            model_connection={
                "protocol": protocol,
                "base_url": f"http://127.0.0.1:{port}/v1",
                "api_key": "host-only-secret",
                "agent_type": "mimo_code",
            },
            network_runtime=runtime,
        )
        assert outcome[0], outcome[1]
        assert (tmp_path / "answer.txt").read_text() == "42"
        assert runtime["enforced"] is True
        assert len(requests) == 1 and requests[0][1] == path


@pytest.mark.parametrize("use_mcp", [False, True])
def test_real_codex_cli_uses_responses_and_local_shell_offline(
    tmp_path, monkeypatch, use_mcp
):
    image = os.environ.get("ASIBENCH_NETWORK_CODEX_IMAGE")
    if os.environ.get("ASIBENCH_NETWORK_DOCKER") != "1" or not image:
        pytest.skip("Requires local Codex/curl image via ASIBENCH_NETWORK_CODEX_IMAGE")
    monkeypatch.delenv("AI4SCI_DOCKER_NETWORK", raising=False)
    gateway = json.loads(
        subprocess.check_output(["docker", "network", "inspect", "bridge"])
    )[0]["IPAM"]["Config"][0]["Gateway"]
    calls = []
    from ai4sci_bench.runner import model_relay as relay_module

    original_validate = relay_module.validate_payload

    def capture_payload(payload, protocol):
        (tmp_path / "fixture-request.json").write_text(json.dumps(payload, indent=2))
        return original_validate(payload, protocol)

    monkeypatch.setattr(relay_module, "validate_payload", capture_payload)
    with upstream_server() as (canary_port, canary_requests):
        command = f"python -c \"from pathlib import Path; Path('answer.txt').write_text(str(6*7))\"; curl --noproxy '*' --max-time 2 http://{gateway}:{canary_port}/canary; echo PROBE_DONE"

        class Responses(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                calls.append(body)
                done = any(
                    item.get("type")
                    in {"function_call_output", "custom_tool_call_output"}
                    for item in body.get("input", [])
                    if isinstance(item, dict)
                )
                tools = body.get("tools", [])
                tool = next(
                    (
                        t
                        for t in tools
                        if t.get("name") in {"exec_command", "shell_command", "shell"}
                    ),
                    None,
                )
                if not done and tool:
                    name = tool["name"]
                    arguments = (
                        {"cmd": command}
                        if name == "exec_command"
                        else (
                            {"command": command}
                            if name == "shell_command"
                            else {"command": ["bash", "-lc", command]}
                        )
                    )
                    item = {
                        "type": "function_call",
                        "id": "fc_probe",
                        "call_id": "call_probe",
                        "name": name,
                        "arguments": json.dumps(arguments),
                        "status": "completed",
                    }
                else:
                    item = {
                        "type": "message",
                        "id": "msg_probe",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Completed locally.",
                                "annotations": [],
                            }
                        ],
                    }
                response = {
                    "id": f"resp_{len(calls)}",
                    "object": "response",
                    "created_at": 1,
                    "model": body.get("model", "fixture"),
                    "status": "completed",
                    "output": [item],
                    "usage": {
                        "input_tokens": 32,
                        "output_tokens": 16,
                        "total_tokens": 48,
                    },
                }
                events = [
                    {
                        "type": "response.created",
                        "response": {**response, "status": "in_progress", "output": []},
                    },
                    {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": item,
                    },
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": item,
                    },
                    {"type": "response.completed", "response": response},
                ]
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for event in events:
                    self.wfile.write(
                        (
                            "event: "
                            + event["type"]
                            + "\ndata: "
                            + json.dumps(event)
                            + "\n\n"
                        ).encode()
                    )
                    self.wfile.flush()

        server = ThreadingHTTPServer(("127.0.0.1", 0), Responses)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        mcp_options = {}
        if use_mcp:
            source = (Path(__file__).parent / "fixtures/mcp/server.py").read_text()
            probe = f"import socket\ntry:\n s=socket.create_connection(('{gateway}',{canary_port}),timeout=1)\nexcept OSError: pass\nelse:\n s.close(); raise RuntimeError('MCP egress escaped')\n"
            (tmp_path / "mcp_server.py").write_text(probe + source)
            config_path = tmp_path / "mcp.json"
            config_path.write_text(
                json.dumps(
                    {
                        "mcpServers": {
                            "science": {
                                "command": "/opt/venv/bin/python",
                                "args": ["/workspace/mcp_server.py"],
                                "sandbox": {"deployment": "instance"},
                            }
                        }
                    }
                )
            )
            mcp_options["mcp_config"] = str(config_path)
        adapter = CodexCLIAdapter(
            api_key="fixture-only-key",
            api_base=f"http://127.0.0.1:{server.server_port}/v1",
            provider="openai-compatible",
            model="gpt-5.1-codex",
            **mcp_options,
        )
        adapter.setup({"sandbox": "os", "repo_root": str(tmp_path)})
        builder = MagicMock()
        builder.ensure_image.return_value = image
        builder.get_image_identity.return_value = "fixture-codex"
        adapter._os_sandbox = OSSandbox(tmp_path, image_builder=builder)
        (tmp_path / "prompt.md").write_text(
            "Calculate 6*7 locally and verify isolation."
        )
        instance = TaskInstance(
            task_id="fixture",
            instance_id="fixture__seed31415",
            task_dir=tmp_path,
            workspace_dir=tmp_path,
            reference_dir=tmp_path,
            prompt_level=PromptLevel.B1,
            parameters={},
            metadata={},
        )
        try:
            output = adapter.solve(instance)
            # Failure diagnostics contain fixture requests, never real credentials.
            assert output.status == RunStatus.COMPLETED, (
                output.log,
                output.error_message,
                calls,
            )
            assert (tmp_path / "answer.txt").read_text() == "42", output.log
            assert any("PROBE_DONE" in json.dumps(c) for c in calls)
            assert canary_requests == []
            assert output.network_runtime["enforced"] is True
        finally:
            adapter.teardown()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


@pytest.mark.parametrize("adapter_class", [PiCLIAdapter, OpenCodeCLIAdapter])
def test_real_native_cli_uses_chat_and_bash_offline(
    adapter_class, tmp_path, monkeypatch
):
    image = os.environ.get("ASIBENCH_NETWORK_NATIVE_IMAGE")
    if os.environ.get("ASIBENCH_NETWORK_DOCKER") != "1" or not image:
        pytest.skip("Requires pi/opencode/curl image via ASIBENCH_NETWORK_NATIVE_IMAGE")
    monkeypatch.delenv("AI4SCI_DOCKER_NETWORK", raising=False)
    gateway = json.loads(
        subprocess.check_output(["docker", "network", "inspect", "bridge"])
    )[0]["IPAM"]["Config"][0]["Gateway"]
    calls = []
    with upstream_server() as (canary_port, canary_requests):
        command = f"python -c \"from pathlib import Path; Path('answer.txt').write_text(str(6*7))\"; curl --noproxy '*' --max-time 2 http://{gateway}:{canary_port}/canary; echo PROBE_DONE"

        class Chat(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                calls.append(body)
                done = any(m.get("role") == "tool" for m in body.get("messages", []))
                tool = next(
                    (
                        t["function"]["name"]
                        for t in body.get("tools", [])
                        if t.get("function", {}).get("name", "").lower() == "bash"
                    ),
                    None,
                )
                delta = (
                    {"content": "Completed locally."}
                    if done or not tool
                    else {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_probe",
                                "type": "function",
                                "function": {
                                    "name": tool,
                                    "arguments": json.dumps(
                                        {
                                            "command": command,
                                            "description": "Calculate locally and check isolation",
                                        }
                                    ),
                                },
                            }
                        ]
                    }
                )
                finish = "stop" if done or not tool else "tool_calls"
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                chunks = [
                    {
                        "id": "chat_probe",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": body["model"],
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", **delta},
                                "finish_reason": None,
                            }
                        ],
                    },
                    {
                        "id": "chat_probe",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": body["model"],
                        "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                        "usage": {
                            "prompt_tokens": 32,
                            "completion_tokens": 16,
                            "total_tokens": 48,
                        },
                    },
                ]
                for chunk in chunks:
                    self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")

        server = ThreadingHTTPServer(("127.0.0.1", 0), Chat)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        kwargs = {"reasoning": False} if adapter_class == PiCLIAdapter else {}
        adapter = adapter_class(
            api_key="fixture-only-key",
            api_base=f"http://127.0.0.1:{server.server_port}/v1",
            api_protocol="openai",
            model="fixture",
            **kwargs,
        )
        adapter.setup({"sandbox": "os", "repo_root": str(tmp_path)})
        builder = MagicMock()
        builder.ensure_image.return_value = image
        builder.get_image_identity.return_value = "fixture-native"
        adapter._os_sandbox = OSSandbox(tmp_path, image_builder=builder)
        (tmp_path / "prompt.md").write_text(
            "Calculate 6*7 locally and verify isolation."
        )
        instance = TaskInstance(
            task_id="fixture",
            instance_id="fixture__seed31415",
            task_dir=tmp_path,
            workspace_dir=tmp_path,
            reference_dir=tmp_path,
            prompt_level=PromptLevel.B1,
            parameters={},
            metadata={},
        )
        try:
            output = adapter.solve(instance)
            assert output.status == RunStatus.COMPLETED, (
                output.log,
                output.error_message,
                calls,
            )
            assert (tmp_path / "answer.txt").read_text() == "42", output.log
            assert any("PROBE_DONE" in json.dumps(c) for c in calls)
            assert canary_requests == []
            assert output.network_runtime["enforced"] is True
        finally:
            adapter.teardown()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


@pytest.mark.parametrize("adapter_name", ["hermes", "codewhale"])
def test_host_only_adapters_reject_os_instead_of_ignoring_it(adapter_name):
    from ai4sci_bench.adapters import HermesAgentAdapter, CodeWhaleAdapter

    cls = {"hermes": HermesAgentAdapter, "codewhale": CodeWhaleAdapter}[adapter_name]
    with pytest.raises(ValueError, match="does not support"):
        cls().setup({"sandbox": "os"})


def test_nested_hosted_tool_and_unknown_content_fail_closed():
    from ai4sci_bench.runner.model_relay import validate_payload

    with pytest.raises(ValueError):
        validate_payload(
            {
                "input": [],
                "tools": [
                    {
                        "type": "namespace",
                        "name": "hidden",
                        "tools": [{"type": "web_search"}],
                    }
                ],
            },
            "openai_responses",
        )
    with pytest.raises(ValueError):
        validate_payload(
            {
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_video", "video_url": "https://example.com"}
                        ],
                    }
                ]
            },
            "openai_responses",
        )
    validate_payload(
        {
            "input": [],
            "tools": [
                {
                    "type": "namespace",
                    "name": "offline",
                    "tools": [
                        {
                            "type": "function",
                            "name": "local",
                            "parameters": {
                                "type": "object",
                                "properties": {"source": {"type": "string"}},
                            },
                        }
                    ],
                }
            ],
        },
        "openai_responses",
    )


def test_codex_custom_env_references_stay_on_host(tmp_path, monkeypatch):
    from ai4sci_bench.runner.network_policy import model_connection

    monkeypatch.setenv("ASIBENCH_TEST_KEY", "fixture-secret")
    monkeypatch.setenv("ASIBENCH_TEST_BASE", "http://127.0.0.1:1/v1")
    adapter = CodexCLIAdapter(
        api_key_env="ASIBENCH_TEST_KEY",
        api_base_env="ASIBENCH_TEST_BASE",
        provider="openai-compatible",
    )
    adapter.setup({"sandbox": "os", "repo_root": str(tmp_path)})
    connection = model_connection(adapter, "codex")
    assert connection["api_key"] == "fixture-secret"
    assert connection["base_url"] == "http://127.0.0.1:1/v1"
    assert adapter._temp_codex_home is None
    adapter.teardown()


def test_direct_llm_os_uses_network_none_and_audit(tmp_path, monkeypatch):
    from ai4sci_bench.adapters import DirectLLMAdapter

    adapter = DirectLLMAdapter()
    adapter.setup({"sandbox": "os", "repo_root": str(tmp_path)})
    (tmp_path / "solver.py").write_text("print(42)")
    instance = TaskInstance(
        task_id="fixture",
        instance_id="fixture__seed31415",
        task_dir=tmp_path,
        workspace_dir=tmp_path,
        reference_dir=tmp_path,
        prompt_level=PromptLevel.B1,
        parameters={},
        metadata={},
    )
    observed = {}

    def execute(self, metadata, **kwargs):
        observed.update(kwargs)
        kwargs["network_runtime"].update(enforced=True)
        return True, "ok", "42", "", "fixture-image"

    monkeypatch.setattr(OSSandbox, "execute_python", execute)
    runtime = {}
    assert adapter._execute(instance, "solver.py", network_runtime=runtime)[0]
    assert observed["network_policy"] == "model-only"
    assert runtime["enforced"] is True and runtime["model_transport"] == "host"


def test_openhands_helper_reports_dependency_failure_as_process_failure(tmp_path):
    import sys
    from ai4sci_bench.adapters.openhands_agent import _HELPER_SCRIPT

    # Force a deterministic missing SDK, independent of installed packages.
    (tmp_path / "openhands").mkdir()
    (tmp_path / "openhands/__init__.py").write_text(
        'raise ImportError("fixture unavailable SDK")'
    )
    config = {
        "workspace": str(tmp_path),
        "prompt": "fixture",
        "result_path": str(tmp_path / "result.json"),
        "model": "openai/fixture",
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "runner.py").write_text(_HELPER_SCRIPT)
    result = subprocess.run(
        [sys.executable, str(tmp_path / "runner.py"), str(tmp_path / "config.json")],
        capture_output=True,
        text=True,
        timeout=10,
        cwd=tmp_path,
    )
    assert result.returncode == 1
    assert json.loads((tmp_path / "result.json").read_text())["status"] == "failed"


@pytest.mark.parametrize(
    "protocol,routes",
    [
        ("openai", ["/v1/chat/completions"]),
        ("openai_responses", ["/v1/responses", "/v1/responses/compact"]),
    ],
)
def test_openai_routes_survive_result_audit(protocol, routes, tmp_path):
    from ai4sci_bench.core.types import AgentOutput, EvalResult
    from ai4sci_bench.runner.orchestrator import BenchmarkOrchestrator, RunConfig
    from tests.test_runner import DummyAgent

    (tmp_path / "tasks").mkdir()
    runner = BenchmarkOrchestrator(
        RunConfig(
            agent=DummyAgent(),
            tasks_dir=str(tmp_path / "tasks"),
            output_dir=str(tmp_path / "results"),
        )
    )
    runtime = {
        "effective": "model-only",
        "protocol": protocol,
        "enforced": True,
        "allowed_routes": routes + ["/private/host/path"],
    }
    output = AgentOutput(
        instance_id="fixture__seed31415",
        output_dir=tmp_path,
        code_files=[],
        data_files=[],
        log="",
        execution_time_seconds=0,
        status=RunStatus.COMPLETED,
        network_runtime=runtime,
    )
    result = EvalResult(
        instance_id="fixture__seed31415",
        task_id="fixture",
        prompt_level=PromptLevel.B1,
        agent_name="fixture",
        parameters={},
        gate_results=[],
        gates_passed=True,
        score_results=[],
        final_score=0,
        agent_output=output,
    )
    runner._save_result(result)
    saved = json.loads(
        (tmp_path / "results/fixture/fixture__seed31415__b1.json").read_text()
    )
    assert saved["provenance"]["network"]["allowed_routes"] == routes
    assert saved["provenance"]["network"] == saved["agent_output"]["network_runtime"]


def test_direct_code_docker_boundary(probe_image, tmp_path, monkeypatch):
    monkeypatch.delenv("AI4SCI_DOCKER_NETWORK", raising=False)
    builder = MagicMock()
    builder.ensure_image.return_value = probe_image
    builder.get_image_identity.return_value = "fixture-code"
    sandbox = OSSandbox(tmp_path, image_builder=builder)
    gateway = json.loads(
        subprocess.check_output(["docker", "network", "inspect", "bridge"])
    )[0]["IPAM"]["Config"][0]["Gateway"]
    with upstream_server() as (port, requests):
        (tmp_path / "solver.py").write_text(
            f"import socket\nfrom pathlib import Path\nassert {{line.split(':')[0].strip() for line in Path('/proc/net/dev').read_text().splitlines() if ':' in line}}=={{'lo'}}\ns=socket.socket();s.settimeout(2);assert s.connect_ex(('{gateway}',{port}))!=0\nPath('answer.txt').write_text('42')\n"
        )
        runtime = {}
        outcome = sandbox.execute_python(
            {},
            workspace=tmp_path,
            code_file="solver.py",
            timeout=20,
            network_policy="model-only",
            network_runtime=runtime,
        )
        assert outcome[0], outcome[1]
        assert (tmp_path / "answer.txt").read_text() == "42"
        assert runtime["enforced"] is True and runtime["allowed_routes"] == []
        assert requests == []


def test_local_tool_arguments_do_not_become_remote_attachments():
    from ai4sci_bench.runner.model_relay import validate_payload

    validate_payload(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "local",
                            "name": "inspect_json",
                            "input": {
                                "type": "image",
                                "source": {"type": "url", "url": "https://example.com"},
                            },
                        }
                    ],
                }
            ]
        },
        "anthropic",
    )


@pytest.mark.parametrize("adapter_class", [PiCLIAdapter, OpenCodeCLIAdapter])
@pytest.mark.parametrize(
    "provider",
    ["anthropic", "openai", "deepseek", "openrouter", "groq", "mistral", "xai", "zai"],
)
def test_known_key_only_provider_can_use_fixed_relay(adapter_class, provider):
    from ai4sci_bench.runner.network_policy import model_connection

    adapter = adapter_class(model=provider + "/fixture", api_key="fixture-only-key")
    connection = model_connection(
        adapter, "pi" if adapter_class == PiCLIAdapter else "opencode"
    )
    assert connection["model"] == "fixture"
    assert connection["protocol"] in {"anthropic", "openai", "openai_responses"}
    assert connection["api_key"] == "fixture-only-key"
    adapter.teardown()


def test_kimi_official_chat_shape_maps_to_chat_relay():
    from ai4sci_bench.runner.network_policy import model_connection

    adapter = KimiCodeCLIAdapter(
        api_key="fixture-key", api_base="https://api.moonshot.ai/v1"
    )
    assert model_connection(adapter, "kimi_code")["protocol"] == "openai"
    adapter.teardown()


def test_openhands_uses_selected_provider_key_without_ambient_openai_override(
    monkeypatch,
):
    from ai4sci_bench.runner.network_policy import model_connection

    monkeypatch.setenv("OPENAI_API_KEY", "wrong-provider-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-fixture-key")
    adapter = OpenHandsAdapter(model="anthropic/claude-sonnet-4-6")
    connection = model_connection(adapter, "openhands")
    assert connection["protocol"] == "anthropic"
    assert connection["api_key"] == "anthropic-fixture-key"
