"""Opt-in native Messages stream guard; no protocol conversion or model retries.

The guard preserves native request and accepted SSE bytes, but fails closed on
unsupported event/block types. A proxy lifetime is the lifetime of its in-memory
session failure boundary. Only /v1/messages streaming requests are supported.
"""
from __future__ import annotations

import http.client
import http.server
import json
import logging
import math
import os
import select
import socket
import threading
from typing import Any
from urllib.parse import urlsplit
import uuid
import zlib

logger = logging.getLogger(__name__)
MAX_BODY = 16 * 1024 * 1024
MAX_EVENT = 2 * 1024 * 1024
MAX_BLOCK = 4 * 1024 * 1024
MAX_BLOCKS = 4096
MAX_SESSIONS = 10000
MAX_ERROR_BODY = 64 * 1024
EMPTY_RECOVERY = "[Your previous response had no visible output. Please continue and produce a user-visible response.]"


class StreamIntegrityError(ValueError):
    """A native response cannot be accepted as a completed generation."""


class _SSEValidator:
    def __init__(self) -> None:
        self.buffer = bytearray()
        self.started = False
        self.stopped = False
        self.stop_reason: str | None = None
        self.blocks: dict[int, dict[str, Any]] = {}
        self.visible = False
        self.has_tool = False

    def feed(self, chunk: bytes) -> list[tuple[bytes, bool]]:
        self.buffer.extend(chunk)
        out: list[tuple[bytes, bool]] = []
        while True:
            # Accept LF and CRLF without rewriting the bytes forwarded to CC.
            end = None
            position = 0
            while True:
                newline = self.buffer.find(b"\n", position)
                if newline < 0:
                    break
                if self.buffer[position:newline].rstrip(b"\r") == b"":
                    end = newline + 1
                    break
                position = newline + 1
            if end is None:
                if len(self.buffer) > MAX_EVENT:
                    raise StreamIntegrityError("event_too_large")
                break
            if end > MAX_EVENT:
                raise StreamIntegrityError("event_too_large")
            raw = bytes(self.buffer[:end])
            del self.buffer[:end]
            self._event(raw)
            out.append((raw, self.stopped))
        return out

    def _event(self, raw: bytes) -> None:
        try:
            lines = raw.decode("utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise StreamIntegrityError("invalid_utf8") from exc
        data: list[str] = []
        event_name = ""
        for line in lines:
            if not line or line.startswith(":"):
                continue
            field, _, value = line.partition(":")
            value = value[1:] if value.startswith(" ") else value
            if field == "data":
                data.append(value)
            elif field == "event":
                event_name = value
        if not data:
            return  # SSE comments/keepalives carry no model result.
        try:
            event = json.loads("\n".join(data))
        except (json.JSONDecodeError, ValueError) as exc:
            raise StreamIntegrityError("invalid_event_json") from exc
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            raise StreamIntegrityError("invalid_event_shape")
        kind = event["type"]
        if event_name and event_name != kind:
            raise StreamIntegrityError("event_type_mismatch")
        if kind == "ping":
            return
        if self.stopped:
            raise StreamIntegrityError("event_after_message_stop")
        if kind == "error":
            raise StreamIntegrityError("upstream_sse_error")
        if kind == "message_start":
            message = event.get("message")
            if self.started or not isinstance(message, dict) or message.get("role") != "assistant" or message.get("content") != []:
                raise StreamIntegrityError("invalid_message_start")
            self.started = True
            return
        if not self.started:
            raise StreamIntegrityError("event_before_message_start")
        if kind == "message_delta":
            if any(not block["closed"] for block in self.blocks.values()):
                raise StreamIntegrityError("message_delta_with_open_block")
            delta = event.get("delta")
            if not isinstance(delta, dict):
                raise StreamIntegrityError("invalid_message_delta")
            reason = delta.get("stop_reason")
            if reason is not None:
                if not isinstance(reason, str) or not reason or self.stop_reason is not None:
                    raise StreamIntegrityError("invalid_stop_reason")
                self.stop_reason = reason
            return
        if kind == "message_stop":
            if not self.stop_reason or any(not block["closed"] for block in self.blocks.values()):
                raise StreamIntegrityError("incomplete_message_stop")
            self.stopped = True
            return
        if self.stop_reason is not None:
            raise StreamIntegrityError("content_after_stop_reason")
        index = event.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise StreamIntegrityError("invalid_block_index")
        if kind == "content_block_start":
            block = event.get("content_block")
            if index != len(self.blocks) or index >= MAX_BLOCKS or not isinstance(block, dict):
                raise StreamIntegrityError("invalid_block_start")
            block_type = block.get("type")
            if block_type not in {"text", "thinking", "redacted_thinking", "tool_use"}:
                raise StreamIntegrityError("unsupported_block_type")
            state: dict[str, Any] = {"type": block_type, "closed": False, "size": 0, "json": [], "initial": block.get("input")}
            if block_type == "text":
                value = block.get("text")
                if not isinstance(value, str):
                    raise StreamIntegrityError("invalid_text_block")
                self.visible |= bool(value.strip())
            elif block_type == "thinking" and not isinstance(block.get("thinking"), str):
                raise StreamIntegrityError("invalid_thinking_block")
            elif block_type == "redacted_thinking" and not isinstance(block.get("data"), str):
                raise StreamIntegrityError("invalid_redacted_thinking_block")
            elif block_type == "tool_use":
                if not all(isinstance(block.get(key), str) and block[key] for key in ("id", "name")) or not isinstance(block.get("input"), dict):
                    raise StreamIntegrityError("invalid_tool_block")
                self.has_tool = True
            self.blocks[index] = state
            return
        block = self.blocks.get(index)
        if block is None or block["closed"]:
            raise StreamIntegrityError("event_without_open_block")
        if kind == "content_block_delta":
            delta = event.get("delta")
            if not isinstance(delta, dict):
                raise StreamIntegrityError("invalid_block_delta")
            mapping = {"text_delta": ("text", "text"), "thinking_delta": ("thinking", "thinking"),
                       "signature_delta": ("thinking", "signature"), "input_json_delta": ("tool_use", "partial_json")}
            expected = mapping.get(delta.get("type"))
            if expected is None or expected[0] != block["type"] or not isinstance(delta.get(expected[1]), str):
                raise StreamIntegrityError("delta_block_type_mismatch")
            value = delta[expected[1]]
            block["size"] += len(value.encode("utf-8"))
            if block["size"] > MAX_BLOCK:
                raise StreamIntegrityError("block_too_large")
            if block["type"] == "text":
                self.visible |= bool(value.strip())
            if block["type"] == "tool_use":
                block["json"].append(value)
            return
        if kind == "content_block_stop":
            if block["type"] == "tool_use" and block["json"]:
                try:
                    value = json.loads("".join(block["json"]))
                except ValueError as exc:
                    raise StreamIntegrityError("invalid_tool_json") from exc
                if not isinstance(value, dict) or block["initial"]:
                    raise StreamIntegrityError("invalid_tool_json")
                block["json"] = []
            block["closed"] = True
            return
        raise StreamIntegrityError("unsupported_event_type")

    def finish(self) -> None:
        if self.buffer.strip():
            raise StreamIntegrityError("incomplete_sse_frame")
        if not self.stopped:
            raise StreamIntegrityError("missing_message_stop")


class NativeAnthropicGuard:
    """Native streaming-only proxy with a sticky failure boundary per CC UUID."""

    def __init__(self, api_base: str, api_key: str | None, port: int = 0,
                 timeout_seconds: float | None = None) -> None:
        base = urlsplit(api_base.rstrip("/"))
        if base.scheme not in {"http", "https"} or not base.hostname or base.username or base.password or base.query or base.fragment:
            raise ValueError("Native guard requires a plain HTTP(S) API base URL")
        if not api_key:
            raise ValueError("Native guard requires an explicit API key or api_key_env")
        timeout = float(timeout_seconds if timeout_seconds is not None else os.getenv("ASIBENCH_STREAM_IDLE_TIMEOUT_SECONDS", "600"))
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Native guard timeout must be finite and positive")
        path = base.path.rstrip("/")
        self._upstream_path = path + ("/messages" if path.endswith("/v1") else "/v1/messages")
        self._base = base
        self._api_key = api_key
        self._port = port
        self._timeout = timeout
        self._states: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._connections: set[http.client.HTTPConnection] = set()
        self._sockets: set[socket.socket] = set()
        self._server: http.server.ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._stopping = False

    @property
    def local_url(self) -> str:
        if self._server is None:
            raise RuntimeError("Native guard not started")
        return f"http://127.0.0.1:{self._server.server_port}"

    def start(self) -> str:
        if self._server is not None:
            return self.local_url
        guard = self
        class Handler(_NativeHandler):
            owner = guard
        self._stopping = False
        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", self._port), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self.local_url

    def attempt_failure(self, session_id: str) -> str | None:
        with self._lock:
            state = self._states.get(session_id)
            if not state:
                return None
            return state["failure"] or ("upstream_response_not_complete" if state["active"] else None)

    @staticmethod
    def _close_connection(connection: http.client.HTTPConnection, wire_socket: socket.socket | None = None) -> None:
        wire_socket = wire_socket or connection.sock
        if wire_socket is not None:
            try:
                wire_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        connection.close()

    def stop(self) -> None:
        with self._lock:
            self._stopping = True
            connections = list(self._connections)
            sockets = list(self._sockets)
            for state in self._states.values():
                if state["active"] and not state["failure"]:
                    state["failure"] = "proxy_stopped"
        for wire_socket in sockets:
            try:
                wire_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        for connection in connections:
            self._close_connection(connection)
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None


class _NativeHandler(http.server.BaseHTTPRequestHandler):
    owner: NativeAnthropicGuard
    protocol_version = "HTTP/1.1"

    def log_message(self, *_: Any) -> None:
        pass

    def _error(self, status: int, reason: str, headers: Any = None, upstream_body: Any = None) -> None:
        document: dict[str, Any] = {"type": "error", "error": {"type": "api_error", "message": reason}}
        if upstream_body is not None:
            document["upstream_error"] = upstream_body
        payload = json.dumps(document).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        if headers is not None:
            for name in ("retry-after", "request-id", "x-request-id"):
                if headers.get(name):
                    self.send_header(name, self._sanitize(headers[name]))
        self.end_headers()
        self.close_connection = True
        self.wfile.write(payload)

    def _sanitize(self, value: str) -> str:
        for secret in (self.owner._api_key, self.headers.get("Authorization"), self.headers.get("x-api-key")):
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return value

    def _upstream_error(self, response: http.client.HTTPResponse) -> None:
        raw = bytearray()
        incomplete = False
        try:
            while len(raw) <= MAX_ERROR_BODY:
                chunk = response.read1(min(8192, MAX_ERROR_BODY + 1 - len(raw)))
                if not chunk:
                    break
                raw.extend(chunk)
        except (OSError, http.client.HTTPException):
            # The upstream status remains authoritative even if its optional
            # diagnostic body is truncated or fails to finish.
            incomplete = True
        truncated = len(raw) > MAX_ERROR_BODY
        text = self._sanitize(bytes(raw[:MAX_ERROR_BODY]).decode("utf-8", errors="replace"))
        try:
            upstream = json.loads(text)
        except ValueError:
            upstream = text
        reason = f"Native upstream returned HTTP {response.status}"
        if isinstance(upstream, dict):
            error = upstream.get("error")
            message = error.get("message") if isinstance(error, dict) else upstream.get("message", error)
            if isinstance(message, str) and message:
                reason += ": " + message[:2048]
        elif text:
            reason += ": " + text[:2048]
        detail = {"body": upstream, "truncated": truncated}
        if incomplete:
            detail["incomplete"] = True
        self._error(response.status, reason, response.headers, detail)

    def do_GET(self) -> None:
        self._error(404, "Native stream guard only serves POST /v1/messages")

    def do_POST(self) -> None:
        self.close_connection = True
        target = urlsplit(self.path)
        if target.path != "/v1/messages":
            self._error(404, "Native stream guard only serves /v1/messages")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY or self.headers.get("Transfer-Encoding"):
                self._error(413, "Invalid or oversized request body")
                return
            self.connection.settimeout(min(self.owner._timeout, 30))
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("incomplete_body")
            encoding = self.headers.get("Content-Encoding", "identity").lower()
            decoded = raw
            if encoding == "gzip":
                decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
                decoded = decoder.decompress(raw, MAX_BODY + 1)
                if len(decoded) > MAX_BODY or not decoder.eof or decoder.unused_data:
                    raise ValueError("invalid_compressed_body")
            elif encoding != "identity":
                raise ValueError("unsupported_request_encoding")
            body = json.loads(decoded)
            user = body.get("metadata", {}).get("user_id")
            user = json.loads(user) if isinstance(user, str) else user
            session = str(uuid.UUID(user["session_id"]))
        except (ValueError, TypeError, KeyError, AttributeError, zlib.error, OSError):
            self._error(400, "Native guard requires valid JSON and CC metadata.user_id.session_id")
            return
        owner = self.owner
        with owner._lock:
            state = owner._states.get(session)
            if state is None:
                if len(owner._states) >= MAX_SESSIONS or owner._stopping:
                    self._error(503, "Native guard session capacity unavailable")
                    return
                state = {"lock": threading.Lock(), "failure": None, "active": False, "empty": False}
                owner._states[session] = state
        with state["lock"]:
            if state["failure"] or owner._stopping:
                self._error(400, "Evaluation attempt closed after upstream failure; use a new session for declared recovery")
                return
            if body.get("stream") is not True:
                state["failure"] = "nonstream_request_rejected"
                self._error(400, "Native guard requires stream=true; it does not convert nonstream requests")
                return
            if state["empty"] and self._is_empty_recovery(body):
                state["failure"] = "client_recovery_after_empty_response"
                self._error(400, "Hidden client recovery after an empty response is not a declared attempt")
                return
            state["active"] = True
            try:
                self._forward(raw, target.query, state)
            finally:
                state["active"] = False
            logger.info("native guard session=%s outcome=%s empty=%s", session, state["failure"] or "complete", state["empty"])

    @staticmethod
    def _is_empty_recovery(body: dict[str, Any]) -> bool:
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages or not isinstance(messages[-1], dict) or messages[-1].get("role") != "user":
            return False
        content = messages[-1].get("content")
        if isinstance(content, str):
            return EMPTY_RECOVERY in content
        return isinstance(content, list) and any(isinstance(block, dict) and block.get("type") == "text" and EMPTY_RECOVERY in str(block.get("text", "")) for block in content)

    def _forward(self, raw: bytes, query: str, state: dict[str, Any]) -> None:
        owner = self.owner
        connection_type = http.client.HTTPSConnection if owner._base.scheme == "https" else http.client.HTTPConnection
        connection = connection_type(owner._base.hostname, owner._base.port, timeout=owner._timeout)
        with owner._lock:
            owner._connections.add(connection)
        hop = {"host", "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade", "content-length", "authorization", "x-api-key", "accept-encoding"}
        hop.update(value.strip().lower() for value in self.headers.get("Connection", "").split(","))
        headers = {key: value for key, value in self.headers.items() if key.lower() not in hop}
        headers.update({"x-api-key": owner._api_key, "Authorization": f"Bearer {owner._api_key}", "Accept-Encoding": "identity", "Connection": "close"})
        path = owner._upstream_path + ("?" + query if query else "")
        started = False
        monitor_done = threading.Event()
        monitor: threading.Thread | None = None
        response = None
        wire_socket: socket.socket | None = None

        def watch_client() -> None:
            while not monitor_done.wait(.05):
                try:
                    readable, _, _ = select.select([self.connection], [], [], 0)
                    if readable and self.connection.recv(1, socket.MSG_PEEK) == b"":
                        state["failure"] = state["failure"] or "client_disconnected"
                        owner._close_connection(connection, wire_socket)
                        return
                except (OSError, ValueError):
                    return

        try:
            monitor = threading.Thread(target=watch_client, daemon=True)
            monitor.start()
            connection.connect()
            wire_socket = connection.sock
            with owner._lock:
                if wire_socket is not None:
                    owner._sockets.add(wire_socket)
                if owner._stopping or state["failure"]:
                    raise StreamIntegrityError(state["failure"] or "proxy_stopped")
            connection.request("POST", path, body=raw, headers=headers)
            response = connection.getresponse()
            if response.status != 200:
                state["failure"] = f"upstream_http_{response.status}"
                self._upstream_error(response)
                return
            if response.headers.get_content_type() != "text/event-stream" or response.headers.get("Content-Encoding", "identity").lower() != "identity":
                raise StreamIntegrityError("upstream_not_native_sse")
            validator = _SSEValidator()
            held = bytearray()
            while True:
                chunk = response.read1(65536)
                if not chunk:
                    break
                for frame, terminal in validator.feed(chunk):
                    if terminal:
                        held.extend(frame)
                        if len(held) > MAX_EVENT:
                            raise StreamIntegrityError("terminal_trailer_too_large")
                        continue
                    if not started:
                        self.send_response(200)
                        for key, value in response.headers.items():
                            if key.lower() not in {"connection", "content-length", "transfer-encoding", "keep-alive"}:
                                self.send_header(key, value)
                        self.send_header("Connection", "close")
                        self.end_headers()
                        started = True
                    self.wfile.write(frame)
                    self.wfile.flush()
            validator.finish()
            monitor_done.set()
            if monitor is not None:
                monitor.join(timeout=1)
            if state["failure"]:
                raise StreamIntegrityError(state["failure"])
            if not started:
                raise StreamIntegrityError("empty_native_stream")
            # Do not expose success until EOF proves there was no trailing
            # protocol corruption after message_stop.
            state["empty"] = not validator.visible and not validator.has_tool
            # Upstream validation is already complete. Mark it before exposing
            # message_stop so a fast CLI exit cannot race the adapter lookup.
            state["active"] = False
            self.wfile.write(held)
            self.wfile.flush()
        except Exception as exc:
            reason = str(exc) if isinstance(exc, StreamIntegrityError) else ("upstream_timeout" if isinstance(exc, TimeoutError) else "native_transport_error")
            state["failure"] = state["failure"] or reason
            try:
                if started:
                    error = {"type": "error", "error": {"type": "api_error", "message": state["failure"]}}
                    self.wfile.write(("event: error\ndata: " + json.dumps(error) + "\n\n").encode())
                    self.wfile.flush()
                else:
                    self._error(502, state["failure"])
            except OSError:
                pass
        finally:
            monitor_done.set()
            owner._close_connection(connection, wire_socket)
            if response is not None:
                response.close()
            if monitor is not None:
                monitor.join(timeout=1)
            with owner._lock:
                owner._connections.discard(connection)
                if wire_socket is not None:
                    owner._sockets.discard(wire_socket)
