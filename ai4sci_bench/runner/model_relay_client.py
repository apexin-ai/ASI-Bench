"""Container-local TCP/Unix bridge. The host relay enforces all authorization.

This stdlib-only file is mounted read-only; it runs before the CLI or MCP
preflight and provides the CLI's expected HTTP transport on loopback.
"""

import os
import json
from pathlib import Path
import tempfile
import select
import socket
import socketserver
import subprocess
import sys
import threading


class Bridge(socketserver.BaseRequestHandler):
    def handle(self):
        try:
            self.relay()
        except OSError:
            pass  # Disconnected CLI requests are routine; no traceback noise.

    def relay(self):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as upstream:
            upstream.connect("/opt/asibench-network/socket/model.sock")
            peers = {self.request: upstream, upstream: self.request}
            while True:
                readable, _, _ = select.select(list(peers), [], [], 300)
                if not readable:
                    return
                for source in readable:
                    data = source.recv(64 * 1024)
                    if not data:
                        return
                    peers[source].sendall(data)


class Server(socketserver.ThreadingTCPServer):
    daemon_threads = True


def configure_client(
    config: dict, base: str, command: list[str], root: Path
) -> list[str]:
    """Generate fresh, placeholder-only CLI configuration inside the container."""
    agent = config["agent_type"]
    protocol = config["protocol"]
    model = config["model"]
    key = "asibench-model-relay"
    home = root / "home"
    home.mkdir()
    os.environ["HOME"] = str(home)
    for name, directory in (
        ("XDG_CONFIG_HOME", "config"),
        ("XDG_CACHE_HOME", "cache"),
        ("XDG_DATA_HOME", "data"),
        ("XDG_STATE_HOME", "state"),
    ):
        path = home / directory
        path.mkdir()
        os.environ[name] = str(path)
    os.environ.update(
        ANTHROPIC_BASE_URL=base,
        ANTHROPIC_API_KEY=key,
        OPENAI_BASE_URL=base + "/v1",
        OPENAI_API_KEY=key,
    )
    if agent == "codex":
        home = root / ".codex"
        home.mkdir()
        os.environ["CODEX_HOME"] = str(home)
        # Command-line provider config works with --ignore-user-config and
        # avoids inheriting auth.json, cached model catalogs or user providers.
        options = [
            'model_provider="asibench"',
            'model_providers.asibench.name="ASI-Bench relay"',
            "model_providers.asibench.base_url=" + json.dumps(base + "/v1"),
            'model_providers.asibench.env_key="OPENAI_API_KEY"',
            'model_providers.asibench.wire_api="responses"',
            "model_providers.asibench.supports_websockets=false",
            "model_providers.asibench.requires_openai_auth=false",
            'model_providers.asibench.capabilities.remote_compaction="unsupported"',
        ]
        start = command.index("codex")
        index = command.index("--", start) if "--" in command[start:] else len(command)
        command[index:index] = [arg for option in options for arg in ("-c", option)]
    elif agent == "pi":
        os.environ["PI_CODING_AGENT_DIR"] = str(root)
        api = {
            "anthropic": "anthropic-messages",
            "openai": "openai-completions",
            "openai_responses": "openai-responses",
        }[protocol]
        (root / "models.json").write_text(
            json.dumps(
                {
                    "providers": {
                        "bench": {
                            "baseUrl": base
                            if protocol == "anthropic"
                            else base + "/v1",
                            "api": api,
                            "apiKey": key,
                            "models": [
                                {
                                    "id": model,
                                    "name": model,
                                    "reasoning": config.get("reasoning", True),
                                    "input": ["text"],
                                    "contextWindow": 262144,
                                    "maxTokens": 32768,
                                }
                            ],
                        }
                    }
                }
            )
        )
        command[command.index("--model") + 1] = "bench/" + model
    elif agent == "opencode":
        npm = {
            "anthropic": "@ai-sdk/anthropic",
            "openai": "@ai-sdk/openai-compatible",
            "openai_responses": "@ai-sdk/openai",
        }[protocol]
        os.environ["OPENCODE_CONFIG_CONTENT"] = json.dumps(
            {
                "provider": {
                    "bench": {
                        "npm": npm,
                        "name": "ASI-Bench relay",
                        "options": {"baseURL": base + "/v1", "apiKey": key},
                        "models": {model: {"name": model}},
                    }
                },
                "agent": {
                    "build": {
                        "tools": {
                            "webfetch": False,
                            "websearch": False,
                            "codesearch": False,
                        }
                    }
                },
            }
        )
        os.environ["OPENCODE_DISABLE_MODELS_FETCH"] = "true"
        command[command.index("-m") + 1] = "bench/" + model
    elif agent == "kimi_code":
        os.environ["KIMI_CODE_HOME"] = str(root)
        (root / "config.toml").write_text(
            'default_model = "bench-model"\n[providers.bench]\n'
            + "type = "
            + json.dumps(protocol)
            + "\nbase_url = "
            + json.dumps(base if protocol == "anthropic" else base + "/v1")
            + "\napi_key = "
            + json.dumps(key)
            + '\n[models.bench-model]\nprovider = "bench"\nmodel = '
            + json.dumps(model)
            + "\nmax_context_size = 262144\n"
        )
        command[command.index("-m") + 1] = "bench-model"
    return command


if __name__ == "__main__":
    config = json.loads(Path("/opt/asibench-network/socket/client.json").read_text())
    with (
        Server(("127.0.0.1", 0), Bridge) as server,
        tempfile.TemporaryDirectory(prefix="asibench-model-client-") as directory,
    ):
        base = f"http://127.0.0.1:{server.server_address[1]}"
        for key in list(os.environ):
            if key.lower() in {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}:
                os.environ.pop(key)
        command = configure_client(config, base, sys.argv[1:], Path(directory))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        sys.exit(subprocess.call(command))
