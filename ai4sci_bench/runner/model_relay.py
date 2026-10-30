"""Host-owned, fixed-route Anthropic relay over a per-attempt Unix socket.

The untrusted container has no IP network. It can only submit JSON Messages
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

POLICY_VERSION = "model-only-unix-v1"
ROUTES = frozenset({"/v1/messages", "/v1/messages/count_tokens"})
MAX_BODY = 32 * 1024 * 1024


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
        allowed_paths = ROUTES | {route + "?beta=true" for route in ROUTES}
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
            if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
                raise ValueError()
            # Only client-side tools: reject provider-hosted search, fetch,
            # computer-use and unknown tool types rather than enabling egress.
            tools = payload.get("tools", [])
            if not isinstance(tools, list) or any(
                not isinstance(tool, dict) or tool.get("type", "custom") != "custom"
                for tool in tools
            ):
                raise ValueError()
            allowed = {
                "model", "messages", "max_tokens", "system", "tools", "tool_choice",
                "stream", "metadata", "stop_sequences", "temperature", "top_k", "top_p",
                "thinking", "output_config", "service_tier", "context_management",
            }
            if set(payload) - allowed:
                raise ValueError()
            def external_source(value):
                if isinstance(value, dict):
                    source = value.get("source")
                    if value.get("type") in {"image", "document"} and isinstance(source, dict) and source.get("type") not in {"base64", "text"}:
                        return True
                    return any(external_source(item) for item in value.values())
                return isinstance(value, list) and any(external_source(item) for item in value)
            if external_source(payload):
                raise ValueError()
        except (ValueError, TypeError, UnicodeDecodeError, RecursionError):
            self._deny(400)
            return
        upstream = self.server.upstream
        connection_type = http.client.HTTPSConnection if upstream.scheme == "https" else http.client.HTTPConnection
        connection = connection_type(upstream.hostname, upstream.port, timeout=300)
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": self.headers.get("anthropic-version", "2023-06-01"),
            "x-api-key": self.server.api_key,
        }
        beta = self.headers.get("anthropic-beta")
        if beta:
            headers["anthropic-beta"] = beta
        sent_headers = False
        with self.server.audit_lock:
            self.server.connections.add(connection)
        try:
            connection.request("POST", upstream.path.rstrip("/") + self.path, body, headers)
            response = connection.getresponse()
            # Never follow upstream redirects to arbitrary targets.
            if 300 <= response.status < 400:
                self._deny(502)
                return
            with self.server.audit_lock:
                self.server.audit["forwarded_requests"] += 1
            self.send_response(response.status)
            self.send_header("Content-Type", response.getheader("Content-Type", "application/json"))
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
def model_relay(api_env: dict[str, str], runtime: dict):
    base = api_env.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
    upstream = urlsplit(base)
    if (upstream.scheme not in {"http", "https"} or not upstream.hostname
            or upstream.username or upstream.password or upstream.query or upstream.fragment
            or any(part in {".", ".."} for part in upstream.path.split("/"))):
        raise ValueError("model-only requires a fixed HTTP(S) Anthropic endpoint")
    api_key = api_env.get("ANTHROPIC_API_KEY")
    if not api_key or "\r" in api_key or "\n" in api_key:
        raise ValueError("model-only requires explicit API-key authentication; OAuth login is unsupported")
    # Resolve port now, so malformed URLs fail before starting a container.
    _ = upstream.port
    runtime.update(
        policy=POLICY_VERSION, effective="model-only", enforced=False,
        mechanism="docker-network-none+unix-model-relay", allowed_routes=sorted(ROUTES),
        upstream_id=hashlib.sha256(base.encode()).hexdigest(),
        forwarded_requests=0, denied_requests=0,
    )
    with tempfile.TemporaryDirectory(prefix="asibench-model-relay-") as directory:
        root = Path(directory)
        server = _UnixServer(str(root / "model.sock"), RelayHandler)
        server.upstream, server.api_key = upstream, api_key
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
