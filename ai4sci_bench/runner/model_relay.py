"""Host-owned, fixed-route model relay over a per-attempt Unix socket.

The untrusted container has no IP network. It can only submit validated model
requests to this socket; upstream location and credentials never enter it.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler
import json
import os
from pathlib import Path
import socketserver
import tempfile
import threading
from urllib.parse import urlsplit

POLICY_VERSION = "model-only-unix-v2"
ROUTES = frozenset({"/v1/messages", "/v1/messages/count_tokens"})
PROTOCOL_ROUTES = {
    "anthropic": ROUTES,
    "openai": frozenset({"/v1/chat/completions"}),
    "openai_responses": frozenset({"/v1/responses", "/v1/responses/compact"}),
}
ALL_ROUTES = frozenset().union(*PROTOCOL_ROUTES.values())
MAX_BODY = 32 * 1024 * 1024


def validate_payload(payload, protocol: str) -> None:
    """Permit model inference and local tools, rejecting hosted retrieval.

    Never inspect arbitrary tool JSON schemas or user text as request options.
    Remote attachment checks apply to message/input content, not local schemas.
    """
    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object")
    shared = {
        "model",
        "tools",
        "tool_choice",
        "stream",
        "metadata",
        "temperature",
        "top_p",
        "service_tier",
    }
    allowed = {
        "anthropic": shared
        | {
            "messages",
            "max_tokens",
            "system",
            "stop_sequences",
            "top_k",
            "thinking",
            "output_config",
            "context_management",
        },
        "openai": shared
        | {
            "messages",
            "max_tokens",
            "max_completion_tokens",
            "stop",
            "n",
            "seed",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "logprobs",
            "top_logprobs",
            "response_format",
            "parallel_tool_calls",
            "reasoning_effort",
            "stream_options",
            "user",
            "store",
            "prompt_cache_key",
            "verbosity",
        },
        "openai_responses": shared
        | {
            "input",
            "instructions",
            "max_output_tokens",
            "parallel_tool_calls",
            "reasoning",
            "text",
            "include",
            "store",
            "stream_options",
            "truncation",
            "prompt_cache_key",
            "prompt_cache_retention",
            "safety_identifier",
            "context_management",
            "client_metadata",
        },
    }[protocol]
    if set(payload) - allowed:
        raise ValueError("Unknown model request options")
    content = payload.get("input" if protocol == "openai_responses" else "messages")
    if not isinstance(content, (list, str) if protocol == "openai_responses" else list):
        raise ValueError("Missing model input")
    if payload.get("store") not in {None, False}:
        raise ValueError("Remote stored conversations are disabled")
    tools = payload.get("tools", [])
    tool_types = (
        {"custom"}
        if protocol == "anthropic"
        else (
            {"function"}
            if protocol == "openai"
            else {"function", "custom", "local_shell"}
        )
    )

    def local_tool(tool):
        if not isinstance(tool, dict):
            return False
        if protocol == "openai_responses" and tool.get("type") == "namespace":
            children = tool.get("tools")
            return isinstance(children, list) and all(
                local_tool(child) for child in children
            )
        return tool.get("type", "custom") in tool_types

    if not isinstance(tools, list) or not all(local_tool(tool) for tool in tools):
        raise ValueError("Hosted tools are disabled")
    choice = payload.get("tool_choice")
    if isinstance(choice, dict) and choice.get("type") not in tool_types | {
        "auto",
        "any",
        "none",
        "tool",
        "required",
    }:
        raise ValueError("Hosted tool selection is disabled")

    def external_source(value):
        if isinstance(value, dict):
            kind = value.get("type")
            source = value.get("source")
            if (
                kind in {"image", "document"}
                and isinstance(source, dict)
                and source.get("type") not in {"base64", "text"}
            ):
                return True
            if kind in {"image_url", "input_image"}:
                url = value.get("image_url", "")
                if isinstance(url, dict):
                    url = url.get("url", "")
                if not isinstance(url, str) or not url.startswith("data:image/"):
                    return True
            if kind in {"file", "input_file"}:
                file = value.get("file", value)
                if (
                    not isinstance(file, dict)
                    or file.get("file_id")
                    or file.get("file_url")
                ):
                    return True
            if kind in {
                "item_reference",
                "mcp_call",
                "mcp_list_tools",
                "web_search_call",
                "file_search_call",
            }:
                return True
        return False

    def check_blocks(blocks):
        if isinstance(blocks, str) or blocks is None:
            return
        if not isinstance(blocks, list):
            raise ValueError("Invalid model content")
        block_types = {
            "anthropic": {
                "text",
                "image",
                "document",
                "thinking",
                "redacted_thinking",
                "tool_use",
                "tool_result",
            },
            "openai": {"text", "image_url", "input_audio", "refusal", "file"},
            "openai_responses": {
                "input_text",
                "input_image",
                "input_file",
                "output_text",
                "refusal",
            },
        }[protocol]
        for block in blocks:
            if not isinstance(block, dict) or block.get("type") not in block_types:
                raise ValueError("Unsupported model content block")
            if external_source(block):
                raise ValueError("Remote model content is disabled")
            if block.get("type") == "tool_result":
                check_blocks(block.get("content"))

    if isinstance(content, list):
        input_types = {
            "message",
            "function_call",
            "function_call_output",
            "custom_tool_call",
            "custom_tool_call_output",
            "reasoning",
            "compaction",
            "local_shell_call",
            "local_shell_call_output",
        }
        for item in content:
            if not isinstance(item, dict):
                raise ValueError("Invalid model input item")
            if (
                protocol == "openai_responses"
                and item.get("type", "message") not in input_types
            ):
                raise ValueError("Unsupported Responses input item")
            if (
                protocol != "openai_responses"
                or item.get("type", "message") == "message"
            ):
                check_blocks(item.get("content"))
            elif item.get("type") in {
                "function_call_output",
                "custom_tool_call_output",
            } and isinstance(item.get("output"), list):
                check_blocks(item["output"])


class _UnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    block_on_close = False

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            with self.audit_lock:
                self.audit["denied_requests"] += 1
            self.shutdown_request(request)
            return
        with self.audit_lock:
            self.clients.add(request)
        try:
            super().process_request(request, client_address)
        except BaseException:
            with self.audit_lock:
                self.clients.discard(request)
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            with self.audit_lock:
                self.clients.discard(request)
            self.slots.release()


class RelayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def setup(self):
        super().setup()
        self.connection.settimeout(60)

    def log_message(self, *_args):
        pass  # Never log prompts, headers, upstream URLs, or credentials.

    def _deny(self, status=403):
        with self.server.audit_lock:
            self.server.audit["denied_requests"] += 1
        self.send_error(status, "Request rejected by model-only policy")

    def do_CONNECT(self):
        self._deny()

    do_GET = do_HEAD = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = do_CONNECT

    def do_POST(self):
        routes = PROTOCOL_ROUTES[self.server.model_protocol]
        allowed_paths = routes
        if self.server.model_protocol == "anthropic":
            allowed_paths = routes | {route + "?beta=true" for route in routes}
        if self.path not in allowed_paths or self.headers.get("Transfer-Encoding"):
            self._deny()
            return
        lengths = self.headers.get_all("Content-Length", [])
        try:
            if len(lengths) != 1:
                raise ValueError()
            length = int(lengths[0])
            if not 0 < length <= MAX_BODY:
                raise ValueError()
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError()
            payload = json.loads(body)
            validate_payload(payload, self.server.model_protocol)
        except (ValueError, TypeError, UnicodeDecodeError, RecursionError):
            self._deny(400)
            return
        upstream = self.server.upstream
        connection_type = (
            http.client.HTTPSConnection
            if upstream.scheme == "https"
            else http.client.HTTPConnection
        )
        connection = connection_type(upstream.hostname, upstream.port, timeout=300)
        headers = {"Content-Type": "application/json"}
        if self.server.model_protocol == "anthropic":
            headers.update(
                {
                    "anthropic-version": self.headers.get(
                        "anthropic-version", "2023-06-01"
                    ),
                    "x-api-key": self.server.api_key,
                }
            )
            beta = self.headers.get("anthropic-beta")
            if beta:
                headers["anthropic-beta"] = beta
        else:
            headers["Authorization"] = "Bearer " + self.server.api_key
        sent_headers = False
        with self.server.audit_lock:
            self.server.connections.add(connection)
        try:
            connection.request(
                "POST", self.server.upstream_prefix + self.path, body, headers
            )
            response = connection.getresponse()
            # Never follow upstream redirects to arbitrary targets.
            if 300 <= response.status < 400:
                self._deny(502)
                return
            with self.server.audit_lock:
                self.server.audit["forwarded_requests"] += 1
            self.send_response(response.status)
            self.send_header(
                "Content-Type", response.getheader("Content-Type", "application/json")
            )
            self.send_header("Connection", "close")
            self.end_headers()
            sent_headers = True
            # read1 preserves streaming SSE without waiting for an entire buffer.
            while chunk := response.read1(64 * 1024):
                self.wfile.write(chunk)
                self.wfile.flush()
        except (OSError, http.client.HTTPException):
            # Fixed diagnostic; do not leak upstream errors/keys into agent logs.
            if not sent_headers:
                self.send_error(502, "Model upstream unavailable")
            self.close_connection = True
        finally:
            connection.close()
            with self.server.audit_lock:
                self.server.connections.discard(connection)


@contextmanager
def model_relay(api_env: dict[str, str], runtime: dict, connection: dict | None = None):
    connection = connection or {}
    protocol = connection.get("protocol", "anthropic")
    if protocol not in PROTOCOL_ROUTES:
        raise ValueError("model-only: unsupported model protocol")
    base = connection.get("base_url") or api_env.get(
        "ANTHROPIC_BASE_URL", "https://api.anthropic.com"
    )
    base = base.rstrip("/")
    upstream = urlsplit(base)
    if (
        upstream.scheme not in {"http", "https"}
        or not upstream.hostname
        or upstream.username
        or upstream.password
        or upstream.query
        or upstream.fragment
        or any(part in {".", ".."} for part in upstream.path.split("/"))
    ):
        raise ValueError("model-only requires a fixed HTTP(S) model endpoint")
    api_key = connection.get("api_key") or api_env.get("ANTHROPIC_API_KEY")
    if not api_key or "\r" in api_key or "\n" in api_key:
        raise ValueError(
            "model-only requires explicit API-key authentication; OAuth login is unsupported"
        )
    # Resolve port now, so malformed URLs fail before starting a container.
    _ = upstream.port
    runtime.update(
        policy=POLICY_VERSION,
        effective="model-only",
        enforced=False,
        mechanism="docker-network-none+unix-model-relay",
        protocol=protocol,
        allowed_routes=sorted(PROTOCOL_ROUTES[protocol]),
        upstream_id=hashlib.sha256(base.encode()).hexdigest(),
        forwarded_requests=0,
        denied_requests=0,
    )
    with tempfile.TemporaryDirectory(prefix="asibench-model-relay-") as directory:
        root = Path(directory)
        server = _UnixServer(str(root / "model.sock"), RelayHandler)
        server.upstream, server.api_key = upstream, api_key
        server.model_protocol = protocol
        prefix = upstream.path.rstrip("/")
        # Client routes use a canonical /v1 prefix; SDK bases normally include it.
        server.upstream_prefix = prefix[:-3] if prefix.endswith("/v1") else prefix
        (root / "client.json").write_text(
            json.dumps(
                {
                    "agent_type": connection.get("agent_type", "claude_code"),
                    "protocol": protocol,
                    "model": connection.get("model", ""),
                    "reasoning": connection.get("reasoning", True),
                }
            ),
            encoding="utf-8",
        )
        server.audit, server.audit_lock = runtime, threading.Lock()
        server.slots = threading.BoundedSemaphore(4)
        server.clients, server.connections = set(), set()
        # The container uses the host UID. No other host users need access.
        os.chmod(root / "model.sock", 0o600)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield root
        finally:
            server.shutdown()
            server.server_close()
            with server.audit_lock:
                connections, clients = list(server.connections), list(server.clients)
            for connection in connections:
                connection.close()
            for client in clients:
                try:
                    client.shutdown(2)
                except OSError:
                    pass
                client.close()
            thread.join(timeout=5)
