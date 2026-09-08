"""Local litellm proxy for third-party API services.

Provides two proxy classes that let CLI-based agents (Claude Code, Codex)
transparently use any of litellm's 120+ supported backends:

* **LiteLLMProxy** — accepts Anthropic Messages API (``/v1/messages``),
  translates via ``litellm.messages.create()``.  Used by Claude Code CLI.

* **LiteLLMOpenAIProxy** — accepts OpenAI Chat Completions API
  (``/v1/chat/completions``), translates via ``litellm.completion()``.
  Used by Codex CLI.

Usage from adapter code::

    # Anthropic format (Claude Code CLI)
    proxy = LiteLLMProxy(model="openai/deepseek-chat", api_base="...", api_key="...")
    proxy.start()   # → http://127.0.0.1:<port>

    # OpenAI format (Codex CLI)
    proxy = LiteLLMOpenAIProxy(model="anthropic/claude-opus-4-6", api_base="...", api_key="...")
    proxy.start()   # → http://127.0.0.1:<port>

Also provides ``resolve_litellm_model()`` to build the litellm model
identifier from a user-supplied model name and explicit ``api_protocol``.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import socket
import logging
import os
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

logger = logging.getLogger(__name__)

VALID_API_PROTOCOLS = ("openai", "anthropic")
GEMINI_THOUGHT_SIGNATURE_FALLBACK = "skip_thought_signature_validator"
TEXT_ONLY_IMAGE_NOTICE = (
    "[Image attachment omitted: the configured model endpoint is text-only. "
    "Do not retry attaching this image; inspect it with local text-based tools "
    "or continue without visual input.]"
)

_IMAGE_CONTENT_TYPES = frozenset({
    "image",
    "image_url",
    "input_image",
})


GEMINI_UNSUPPORTED_SCHEMA_KEYS = {
    "$schema",
    "$defs",
    "additionalItems",
    "additionalProperties",
    "allOf",
    "anyOf",
    "const",
    "contains",
    "default",
    "definitions",
    "dependentRequired",
    "dependentSchemas",
    "else",
    "examples",
    "exclusiveMaximum",
    "exclusiveMinimum",
    "if",
    "maxContains",
    "maxLength",
    "maxProperties",
    "minContains",
    "minLength",
    "minProperties",
    "multipleOf",
    "not",
    "oneOf",
    "pattern",
    "patternProperties",
    "prefixItems",
    "propertyNames",
    "then",
    "title",
    "unevaluatedItems",
    "unevaluatedProperties",
    "uniqueItems",
}


def replace_unsupported_image_inputs(value: Any) -> tuple[Any, int]:
    """Replace image content blocks with text safe for non-visual endpoints.

    Agent runtimes can add images to later model turns after a file-reading
    tool opens a generated PNG. Sending those blocks to a text-only endpoint
    can produce a permanent 4xx response that the agent retries until the
    benchmark timeout. This recursively handles OpenAI Responses, OpenAI Chat
    Completions, and Anthropic-style image block shapes while preserving the
    surrounding conversation and ordinary fields named ``image_url`` (for
    example in a tool's JSON schema).

    The input is not mutated. The returned count is useful for logging and
    tests without ever logging the image data itself.
    """
    if isinstance(value, list):
        replaced = 0
        rewritten: list[Any] = []
        for item in value:
            new_item, item_count = replace_unsupported_image_inputs(item)
            rewritten.append(new_item)
            replaced += item_count
        return rewritten, replaced

    if not isinstance(value, dict):
        return value, 0

    block_type = value.get("type")
    if isinstance(block_type, str) and block_type.lower() in _IMAGE_CONTENT_TYPES:
        text_type = "input_text" if block_type.lower() == "input_image" else "text"
        return {"type": text_type, "text": TEXT_ONLY_IMAGE_NOTICE}, 1

    replaced = 0
    rewritten_dict: dict[str, Any] = {}
    for key, item in value.items():
        new_item, item_count = replace_unsupported_image_inputs(item)
        rewritten_dict[key] = new_item
        replaced += item_count
    return rewritten_dict, replaced


def prepare_model_input_for_endpoint(
    body: dict[str, Any],
    *,
    supports_image_input: bool,
) -> dict[str, Any]:
    """Apply endpoint input-capability filters without inspecting model names."""
    if supports_image_input:
        return body
    rewritten, replaced = replace_unsupported_image_inputs(body)
    if replaced:
        logger.warning(
            "text-only endpoint: replaced %d image attachment(s) before forwarding",
            replaced,
        )
    return rewritten


def strip_gemini_unsupported_schema_keys(value: Any) -> Any:
    """Return ``value`` with JSON Schema keywords unsupported by Gemini removed.

    Claude Code emits Draft-07-ish schemas for tools. TokenRouter's Gemini path
    converts those to Gemini function declarations, whose Schema type accepts a
    smaller OpenAPI-like subset. Drop validation metadata that Gemini rejects
    while preserving names, descriptions, object shapes, required fields, enums,
    and simple min/max constraints.
    """
    return _strip_gemini_unsupported_schema_keys(value, parent_key=None)


def _strip_gemini_unsupported_schema_keys(value: Any, parent_key: str | None) -> Any:
    if isinstance(value, dict):
        stripped: dict[str, Any] = {}
        for key, item in value.items():
            # Inside ``properties`` the keys are parameter names, not schema
            # keywords. Claude Code has a real tool argument named "pattern".
            if parent_key != "properties" and key in GEMINI_UNSUPPORTED_SCHEMA_KEYS:
                continue
            stripped[key] = _strip_gemini_unsupported_schema_keys(item, key)
        return stripped
    if isinstance(value, list):
        return [_strip_gemini_unsupported_schema_keys(item, parent_key=None) for item in value]
    return value


def rewrite_anthropic_payload_for_tokenrouter(
    body: dict[str, Any],
    model: str,
    effort: str,
    gemini_thought_signature_cache: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Apply minimal TokenRouter compatibility rewrites to a Messages payload."""
    model_l = model.lower()
    rewritten = strip_gemini_unsupported_schema_keys(body) if "gemini" in model_l else dict(body)

    if "gemini" in model_l:
        _textualize_gemini_tool_history(rewritten)
        _attach_gemini_tool_thought_signatures(
            rewritten,
            signature_cache=gemini_thought_signature_cache,
        )
        _append_gemini_continue_after_terminal_tool_result(rewritten)

    if "claude" in model_l and rewritten.get("thinking") is not None:
        thinking = rewritten.get("thinking")
        if isinstance(thinking, dict):
            thinking = dict(thinking)
            thinking["type"] = "adaptive"
            thinking.pop("budget_tokens", None)
        else:
            thinking = {"type": "adaptive"}
        rewritten["thinking"] = thinking

        output_config = rewritten.get("output_config")
        if not isinstance(output_config, dict):
            output_config = {}
        else:
            output_config = dict(output_config)
        output_config["effort"] = _tokenrouter_output_effort(effort)
        rewritten["output_config"] = output_config

    return rewritten


def _textualize_gemini_tool_history(body: dict[str, Any]) -> None:
    """Send executed tool history to Gemini as text instead of replayed calls.

    Gemini 3 validates every historical functionCall part for a thought
    signature. TokenRouter's Anthropic route does not reliably preserve custom
    signature fields on Claude Code's replayed ``tool_use`` blocks over long
    trajectories, so keep Claude Code's local trajectory intact while sending a
    textual transcript upstream. Future model responses can still create fresh
    tool calls because the tools list is unchanged.
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        rewritten_content: list[Any] = []
        converted_tool_result = False
        for block in content:
            if not isinstance(block, dict):
                rewritten_content.append(block)
                continue
            btype = block.get("type")
            if btype == "tool_use":
                rewritten_content.append({
                    "type": "text",
                    "text": _format_tool_use_as_text(block),
                })
            elif btype == "tool_result":
                converted_tool_result = True
                rewritten_content.append({
                    "type": "text",
                    "text": _format_tool_result_as_text(block),
                })
            else:
                rewritten_content.append(block)
        if converted_tool_result and message.get("role") == "user":
            rewritten_content.append({
                "type": "text",
                "text": (
                    "Continue from this observation. If you need another command, file read, "
                    "or file write, use the provided tool-calling interface; do not describe "
                    "a tool call in plain text."
                ),
            })
        message["content"] = rewritten_content


def _format_tool_use_as_text(block: dict[str, Any]) -> str:
    return (
        "A prior assistant tool request was already executed by the runtime. "
        "Use the following runtime observation as its result."
    )


def _format_tool_result_as_text(block: dict[str, Any]) -> str:
    content = block.get("content")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        content_text = "\n".join(parts)
    else:
        content_text = str(content)
    if block.get("is_error"):
        return f"Runtime observation for the prior tool execution (error):\n{content_text}"
    return f"Runtime observation for the prior tool execution:\n{content_text}"


def _anthropic_messages_to_openai_standard(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Translate Anthropic Messages content to standard OpenAI chat roles."""
    messages: list[dict[str, Any]] = []
    system = body.get("system")
    if isinstance(system, list):
        system = "\n".join(str(b.get("text", "")) for b in system
                           if isinstance(b, dict) and b.get("type") == "text")
    if system:
        messages.append({"role": "system", "content": str(system)})
    for message in body.get("messages", []):
        if not isinstance(message, dict):
            continue
        role, content = message.get("role"), message.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            messages.append({"role": role, "content": str(content)})
            continue
        if role == "assistant":
            text: list[str] = []
            calls: list[dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    text.append(str(block.get("text", "")))
                elif block.get("type") == "tool_use":
                    calls.append({"id": str(block.get("id", "")), "type": "function",
                                  "function": {"name": str(block.get("name", "")),
                                  "arguments": json.dumps(block.get("input", {}), ensure_ascii=False)}})
            item: dict[str, Any] = {"role": "assistant", "content": "\n".join(text) or None}
            if calls:
                item["tool_calls"] = calls
            messages.append(item)
        else:
            text: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    text.append(str(block.get("text", "")))
                elif block.get("type") == "tool_result":
                    if text:
                        messages.append({"role": "user", "content": "\n".join(text)})
                        text = []
                    result = block.get("content", "")
                    if isinstance(result, list):
                        result = "\n".join(str(x.get("text", "")) if isinstance(x, dict) else str(x) for x in result)
                    messages.append({"role": "tool", "tool_call_id": str(block.get("tool_use_id", "")),
                                     "content": str(result)})
            if text:
                messages.append({"role": "user", "content": "\n".join(text)})
    return messages


def _attach_gemini_tool_thought_signatures(
    body: dict[str, Any],
    signature_cache: dict[str, str] | None = None,
) -> None:
    """Ensure historical assistant tool calls carry Gemini thought signatures.

    TokenRouter's Gemini route accepts Anthropic Messages payloads, but its
    Gemini backend requires each historical ``functionCall`` part to include a
    ``thought_signature``. Claude Code replays prior tool calls as Anthropic
    ``tool_use`` blocks and does not preserve Gemini-specific fields, so the
    second request in a tool loop can fail with HTTP 400. Prefer a cached real
    signature when available; otherwise use Gemini's documented validator-skip
    marker so the benchmark can continue.
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return

    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            if block.get("thought_signature") or block.get("thoughtSignature"):
                continue
            signature = None
            tool_id = block.get("id")
            if isinstance(tool_id, str) and signature_cache:
                signature = signature_cache.get(tool_id)
            signature = signature or GEMINI_THOUGHT_SIGNATURE_FALLBACK
            block["thought_signature"] = signature
            block["thoughtSignature"] = signature


def _append_gemini_continue_after_terminal_tool_result(body: dict[str, Any]) -> None:
    """Work around TokenRouter/Gemini rejecting a terminal tool_result turn.

    Anthropic clients send ``assistant(tool_use)`` followed by
    ``user(tool_result)`` and then expect the assistant to continue. The Gemini
    route currently rejects that terminal function-response shape as missing a
    thought signature. Adding a tiny text part after the tool result preserves
    the intended "continue after observing the tool output" semantics while
    satisfying the upstream converter.
    """
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return
    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") != "user":
        return
    content = last.get("content")
    if not isinstance(content, list) or not content:
        return
    has_tool_result = False
    has_text = False
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_result":
            has_tool_result = True
        elif block.get("type") == "text" and str(block.get("text", "")).strip():
            has_text = True
    if has_tool_result and not has_text:
        content.append({"type": "text", "text": "Continue."})


def _cache_gemini_tool_thought_signatures(
    data: bytes,
    cache: dict[str, str],
    lock: threading.Lock | None = None,
) -> None:
    """Cache any Gemini thought signatures exposed by an upstream response."""
    signatures = _extract_gemini_tool_thought_signatures(data)
    if not signatures:
        return
    if lock is None:
        cache.update(signatures)
    else:
        with lock:
            cache.update(signatures)


def _extract_gemini_tool_thought_signatures(data: bytes) -> dict[str, str]:
    text = data.decode("utf-8", errors="ignore")
    signatures: dict[str, str] = {}

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if payload is not None:
        _collect_gemini_tool_thought_signatures(payload, signatures)
        return signatures

    # Streaming Anthropic responses are SSE frames. Track content block index so
    # a later signature delta can be associated with the tool_use block id.
    block_ids: dict[int, str] = {}
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        raw = line[len("data:"):].strip()
        if not raw or raw == "[DONE]":
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        index = event.get("index")
        if event.get("type") == "content_block_start":
            block = event.get("content_block")
            if isinstance(index, int) and isinstance(block, dict):
                tool_id = block.get("id")
                if block.get("type") == "tool_use" and isinstance(tool_id, str):
                    block_ids[index] = tool_id
                signature = _get_signature_value(block)
                if isinstance(tool_id, str) and signature:
                    signatures[tool_id] = signature
        elif event.get("type") == "content_block_delta":
            delta = event.get("delta")
            signature = _get_signature_value(delta)
            if isinstance(index, int) and signature and index in block_ids:
                signatures[block_ids[index]] = signature

    return signatures


def _rewrite_gemini_response_tool_use_ids(
    data: bytes,
    next_tool_id: Any,
) -> bytes:
    """Rewrite Gemini adapter tool_use ids so Claude Code sees unique ids.

    TokenRouter's Gemini Anthropic route can emit ``call_gemini_0`` for every
    function call in a multi-turn conversation. Claude Code replays those ids in
    later ``tool_result`` blocks, so duplicates can make the tool loop terminate
    as ``[Tool use interrupted]``. The proxy is the boundary Claude Code sees,
    so rewrite each upstream tool_use id once before streaming it to the CLI.
    """
    text = data.decode("utf-8", errors="ignore")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if payload is not None:
        changed = _rewrite_json_gemini_tool_use_ids(payload, next_tool_id)
        if changed:
            return json.dumps(payload).encode()
        return data

    changed = False
    out_lines: list[str] = []
    for line in text.splitlines(keepends=True):
        newline = ""
        body = line
        if body.endswith("\r\n"):
            body = body[:-2]
            newline = "\r\n"
        elif body.endswith("\n"):
            body = body[:-1]
            newline = "\n"

        if not body.startswith("data:"):
            out_lines.append(line)
            continue
        raw = body[len("data:"):].strip()
        if not raw or raw == "[DONE]":
            out_lines.append(line)
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            out_lines.append(line)
            continue
        if _rewrite_json_gemini_tool_use_ids(event, next_tool_id):
            changed = True
            out_lines.append("data: " + json.dumps(event) + newline)
        else:
            out_lines.append(line)

    if not changed:
        return data
    return "".join(out_lines).encode()


def _rewrite_gemini_response_sse_line(
    line: bytes,
    next_tool_id: Any,
) -> bytes:
    decoded = line.decode("utf-8", errors="ignore")
    newline = ""
    body = decoded
    if body.endswith("\r\n"):
        body = body[:-2]
        newline = "\r\n"
    elif body.endswith("\n"):
        body = body[:-1]
        newline = "\n"

    if not body.startswith("data:"):
        return line
    raw = body[len("data:"):].strip()
    if not raw or raw == "[DONE]":
        return line
    try:
        event = json.loads(raw)
    except json.JSONDecodeError:
        return line
    if not _rewrite_json_gemini_tool_use_ids(event, next_tool_id):
        return line
    return ("data: " + json.dumps(event) + newline).encode()


def _rewrite_json_gemini_tool_use_ids(value: Any, next_tool_id: Any) -> bool:
    changed = False
    if isinstance(value, dict):
        if value.get("type") == "tool_use" and isinstance(value.get("id"), str):
            value["id"] = next_tool_id()
            changed = True
        for item in value.values():
            changed = _rewrite_json_gemini_tool_use_ids(item, next_tool_id) or changed
    elif isinstance(value, list):
        for item in value:
            changed = _rewrite_json_gemini_tool_use_ids(item, next_tool_id) or changed
    return changed


def _collect_gemini_tool_thought_signatures(value: Any, signatures: dict[str, str]) -> None:
    if isinstance(value, dict):
        tool_id = value.get("id")
        signature = _get_signature_value(value)
        if isinstance(tool_id, str) and signature:
            signatures[tool_id] = signature
        for item in value.values():
            _collect_gemini_tool_thought_signatures(item, signatures)
    elif isinstance(value, list):
        for item in value:
            _collect_gemini_tool_thought_signatures(item, signatures)


def _get_signature_value(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    for key in ("thought_signature", "thoughtSignature", "signature"):
        signature = value.get(key)
        if isinstance(signature, str) and signature:
            return signature
    return None


def _tokenrouter_output_effort(effort: str) -> str:
    """Map Claude Code effort labels to TokenRouter output_config labels."""
    if effort in ("xhigh", "max"):
        return "high"
    if effort in ("low", "medium", "high"):
        return effort
    return "medium"


def _openai_reasoning_effort_from_claude(effort: str) -> str:
    """Map Claude Code effort labels to OpenAI-compatible reasoning_effort."""
    if effort in ("xhigh", "max", "high"):
        return "high"
    if effort == "low":
        return "low"
    return "medium"


def resolve_litellm_model(model: str, api_protocol: str | None) -> str:
    """Build a litellm model identifier from a raw model name and explicit protocol.

    When ``api_protocol`` is set, it is prepended as the litellm routing
    prefix (e.g. ``openai/claude-sonnet-4.6``).  When ``api_protocol`` is
    ``None``, ``model`` is passed through unchanged (backward-compatible).
    """
    if api_protocol is None:
        return model
    if api_protocol not in VALID_API_PROTOCOLS:
        raise ValueError(
            f"Invalid api_protocol '{api_protocol}'. "
            f"Valid values: {', '.join(VALID_API_PROTOCOLS)}"
        )
    return f"{api_protocol}/{model}"


def to_chat_completions_model(model: str) -> str:
    """Remap a litellm ``openai/`` model to the generic chat-completions provider.

    litellm routes the ``openai/`` provider through the OpenAI **Responses API**
    (``POST /v1/responses``).  Many OpenAI-*compatible* gateways (DeepSeek,
    TokenRouter, OpenRouter, vLLM, …) implement only ``/v1/chat/completions``
    and return 500 ``not implemented`` for ``/v1/responses``.  ``hosted_vllm`` is
    litellm's generic OpenAI-compatible provider that always uses
    ``/v1/chat/completions``, so we remap the prefix.  Non-``openai/`` models
    (e.g. already ``anthropic/``) pass through unchanged.
    """
    prefix = "openai/"
    if model.startswith(prefix):
        return "hosted_vllm/" + model[len(prefix):]
    return model


def _anthropic_tool_choice_to_openai(tool_choice: Any) -> Any:
    """Translate an Anthropic ``tool_choice`` to its OpenAI equivalent.

    The chat/completions endpoint rejects Anthropic's object form outright
    (SGLang answers 400 ``ToolChoice.function: Field required``), so a request
    forwarded there has to be converted rather than passed through.  Anthropic
    ``any`` means "call one of the tools", which is OpenAI ``required`` — not
    ``auto``, which lets the model answer without calling anything and is a
    materially different instruction.  Mirrors litellm's own
    ``translate_anthropic_tool_choice_to_openai``.
    """
    if not isinstance(tool_choice, dict):
        return tool_choice
    kind = tool_choice.get("type")
    if kind == "any":
        return "required"
    if kind == "auto":
        return "auto"
    if kind == "none":
        return "none"
    if kind == "tool" and tool_choice.get("name"):
        return {"type": "function", "function": {"name": str(tool_choice["name"])}}
    return "auto"


def _upstream_timeout_seconds(real_streaming: bool) -> float:
    """Read timeout for the upstream call, in seconds.

    litellm falls back to 600s and httpx applies the value per read operation,
    so the same number means opposite things on the two paths.  While a
    response is streamed every chunk resets the clock, making it an idle
    timeout that a long generation never trips.  On a non-streamed call the
    whole body is a single read, so it caps total generation time instead — and
    a max-effort reasoning turn runs well past 600s.  That is why non-streamed
    calls time out while streamed ones of the same length succeed, so the two
    paths get separate budgets: tight on a stream to surface a stalled
    upstream quickly, generous on a blocking call to fit a full output budget.
    """
    if real_streaming:
        name, default = "ASIBENCH_STREAM_IDLE_TIMEOUT_SECONDS", 600.0
    else:
        name, default = "ASIBENCH_BLOCKING_TIMEOUT_SECONDS", 3600.0
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _stream_retry_attempts() -> int:
    """Total attempts for a streamed upstream call, retries included.

    A stream can die mid-response (observed as httpcore ReadError "[Errno 9]
    Bad file descriptor").  Retrying here is worth doing because the CLI's own
    retry drops to a non-streamed request, which then has to fit the whole
    generation inside one read and reliably times out on a long reasoning turn.
    Only replays that emitted nothing are eligible; see ``_handle_streaming``.
    """
    raw = os.getenv("ASIBENCH_STREAM_RETRY_ATTEMPTS", "").strip()
    try:
        value = int(raw)
    except ValueError:
        return 3
    return value if value >= 1 else 3


class _LiteLLMProxyHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler that bridges Anthropic Messages API to litellm."""

    # Set by LiteLLMProxy before serving
    litellm_model: str
    litellm_api_base: str | None
    litellm_api_key: str | None
    routing_key: str | None = None
    supports_image_input: bool = False
    # request_id -> routing_key, for requests whose handler thread never
    # returned.  ``LiteLLMProxy.stop`` reports the leftovers, which is the only
    # way a hung thread (killed silently as a daemon at process exit) shows up.
    _inflight: dict[str, str] = {}

    @staticmethod
    def _real_streaming_enabled() -> bool:
        """Return whether stream requests should be forwarded upstream as streams."""
        value = os.getenv("ASIBENCH_REAL_STREAMING_PROXY", "1").strip().lower()
        return value not in {"0", "false", "no", "off"}

    def do_POST(self) -> None:
        if "/v1/messages" not in self.path:
            self._send_error(404, f"Not found: {self.path}")
            return

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._send_error(400, "Empty request body")
            return

        try:
            body = json.loads(self.rfile.read(content_length))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            self._send_error(400, f"Invalid JSON: {e}")
            return

        body = prepare_model_input_for_endpoint(
            body,
            supports_image_input=self.supports_image_input,
        )
        is_stream = body.get("stream", False)
        real_streaming = is_stream and self._real_streaming_enabled()
        request_id = self.headers.get("x-request-id") or f"proxy-{uuid.uuid4().hex[:12]}"

        kwargs = self._build_litellm_kwargs(body)
        kwargs["timeout"] = _upstream_timeout_seconds(real_streaming)
        type(self)._inflight[request_id] = self.routing_key or ""
        logger.warning(
            "litellm proxy request_start id=%s stream=%s real_stream=%s model=%s "
            "max_tokens=%s timeout=%ss routing_key=%s",
            request_id, is_stream, real_streaming, self.litellm_model,
            kwargs.get("max_tokens"), kwargs["timeout"], self.routing_key or "",
        )
        if real_streaming and "tool_choice" in kwargs:
            kwargs["tool_choice"] = _anthropic_tool_choice_to_openai(kwargs["tool_choice"])
        if real_streaming:
            # The OpenAI completion endpoint requires OpenAI message roles;
            # translate Anthropic tool_result blocks before forwarding.
            kwargs["messages"] = _anthropic_messages_to_openai_standard(body)

        attempts = _stream_retry_attempts() if real_streaming else 1
        for attempt in range(1, attempts + 1):
            try:
                if real_streaming:
                    kwargs["stream"] = True
                    import litellm
                    response = litellm.completion(**kwargs)
                elif is_stream:
                    # Compatibility mode: retain the original non-streaming
                    # Anthropic call and synthesize SSE locally below.
                    kwargs["stream"] = False
                    response = self._call_litellm_anthropic(kwargs)
                else:
                    response = self._call_litellm_anthropic(kwargs)
            except Exception as e:
                logger.exception("litellm proxy: upstream call failed")
                if real_streaming and attempt < attempts:
                    logger.warning("litellm proxy stream_retry id=%s attempt=%d/%d reason=call_failed",
                                   request_id, attempt, attempts)
                    continue
                type(self)._inflight.pop(request_id, None)
                self._send_anthropic_error(502, f"Upstream error: {type(e).__name__}: {e}")
                return

            if not real_streaming:
                break

            outcome = self._handle_streaming(response, request_id=request_id)
            # Only a stream that died before any SSE reached the client can be
            # replayed; once deltas are out, a second attempt would duplicate
            # content the client has already committed to its transcript.
            if outcome != "failed_before_write" or attempt >= attempts:
                return
            logger.warning("litellm proxy stream_retry id=%s attempt=%d/%d reason=%s",
                           request_id, attempt, attempts, outcome)
            type(self)._inflight[request_id] = self.routing_key or ""

        type(self)._inflight.pop(request_id, None)
        if is_stream:
            self._handle_synthetic_streaming(response)
        else:
            self._handle_non_streaming(response, request_id=request_id)

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok", "proxy": "litellm"}).encode())

    @staticmethod
    def _call_litellm_anthropic(kwargs: dict[str, Any]) -> Any:
        """Call litellm anthropic messages, falling back to async if sync is unimplemented."""
        try:
            from litellm.anthropic_interface.messages import create
            return create(**kwargs)
        except ValueError as exc:
            if "not implemented for sync calls" not in str(exc):
                raise
        from litellm.anthropic_interface.messages import acreate
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(acreate(**kwargs))
        finally:
            loop.close()

    def _build_litellm_kwargs(self, body: dict) -> dict[str, Any]:
        """Extract Anthropic Messages API params and build litellm.messages.create kwargs."""
        # The proxy's job is Anthropic-in → OpenAI-*compatible*-chat-out, so force
        # the chat/completions route (litellm sends ``openai/`` to the Responses
        # API, which OpenAI-compatible gateways usually do not implement).
        kwargs: dict[str, Any] = {
            "model": to_chat_completions_model(self.litellm_model),
            "messages": body.get("messages", []),
            "max_tokens": body.get("max_tokens", 4096),
            "stream": body.get("stream", False),
        }

        if self.litellm_api_base:
            kwargs["api_base"] = self.litellm_api_base
        if self.litellm_api_key:
            kwargs["api_key"] = self.litellm_api_key
        if self.routing_key:
            # Keep every turn from one CLI agent session on the same SGLang
            # worker. The router uses this header only for placement; it is
            # not included in the model prompt.
            kwargs["extra_headers"] = {"X-SMG-Routing-Key": self.routing_key}

        for key in (
            "system", "temperature", "top_p", "top_k",
            "stop_sequences", "metadata", "tools", "tool_choice",
            "thinking",
        ):
            if key in body and body[key] is not None:
                kwargs[key] = body[key]

        # Claude sends Anthropic tool definitions while the SGLang target
        # expects OpenAI chat-completions tools.
        tools = kwargs.get("tools")
        if isinstance(tools, list):
            converted = []
            for tool in tools:
                if not isinstance(tool, dict):
                    continue
                if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
                    converted.append(tool)
                    continue
                converted.append({"type": "function", "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {"type": "object"}),
                }})
            kwargs["tools"] = converted

        return kwargs

    def _handle_non_streaming(self, response: Any, *, request_id: str = "") -> None:
        # The streaming path logs a terminal ``stream_outcome`` line for every
        # request; without the same here, a non-streaming request that dies on
        # the way back to the client leaves no trace in the proxy log at all --
        # the failure is only inferrable from the client's ConnectionResetError.
        outcome = "ok"
        stop_reason = None
        usage = None
        resp_json = b""
        try:
            try:
                if isinstance(response, dict):
                    payload = response
                else:
                    payload = dict(response)
            except (TypeError, ValueError):
                outcome = "serialize_failed"
                resp_json = json.dumps({"error": "Failed to serialize response"}).encode()
            else:
                stop_reason = payload.get("stop_reason")
                usage = payload.get("usage")
                try:
                    resp_json = json.dumps(payload, default=str).encode()
                except (TypeError, ValueError):
                    outcome = "serialize_failed"
                    resp_json = json.dumps({"error": "Failed to serialize response"}).encode()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_json)))
            self.end_headers()
            self.wfile.write(resp_json)
        except (BrokenPipeError, ConnectionResetError):
            outcome = "client_gone"
            raise
        except Exception:
            outcome = "write_failed"
            raise
        finally:
            logger.warning(
                "litellm proxy non_streaming_outcome id=%s outcome=%s bytes=%d "
                "stop_reason=%s usage=%s",
                request_id, outcome, len(resp_json), stop_reason, usage,
            )

    def _handle_synthetic_streaming(self, response: Any) -> None:
        """Convert a non-streaming response into Anthropic SSE events."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        try:
            resp = dict(response) if not isinstance(response, dict) else response

            msg_start = {
                "type": "message_start",
                "message": {
                    "id": resp.get("id", "msg_proxy"),
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": resp.get("model", self.litellm_model),
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": resp.get("usage", {}).get("input_tokens", 0), "output_tokens": 0},
                },
            }
            self._write_sse("message_start", msg_start)

            content = resp.get("content", [])
            if not isinstance(content, list):
                content = [{"type": "text", "text": str(content)}]

            for idx, block in enumerate(content):
                if not isinstance(block, dict):
                    block = {"type": "text", "text": str(block)}
                btype = block.get("type", "text")

                if btype == "tool_use":
                    # Forward the tool call as the model issued it.  A previous
                    # revision rewrote Bash inputs down to ``command`` alone and
                    # dropped the block when that was empty, which silently
                    # discarded the model's ``timeout`` and ``run_in_background``
                    # arguments and hid whole tool calls from the CLI — and only
                    # on this path, so the two transports ran different agents.
                    tool_input = block.get("input", {})
                    tool_name = str(block.get("name", ""))
                    self._write_sse("content_block_start", {
                        "type": "content_block_start",
                        "index": idx,
                        "content_block": {"type": "tool_use", "id": block.get("id", f"toolu_{idx}"), "name": tool_name, "input": {}},
                    })
                    input_json = json.dumps(tool_input, default=str)
                    self._write_sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": {"type": "input_json_delta", "partial_json": input_json},
                    })
                elif btype == "thinking":
                    # Reasoning models (DeepSeek/GLM/Kimi via OpenAI chat) return
                    # reasoning_content, which litellm maps to a ``thinking`` block.
                    # Replay it as Anthropic thinking SSE so the CLI keeps the
                    # model's reasoning instead of silently dropping it.
                    self._write_sse("content_block_start", {
                        "type": "content_block_start",
                        "index": idx,
                        "content_block": {"type": "thinking", "thinking": ""},
                    })
                    thinking = block.get("thinking", "") or ""
                    chunk_size = 100
                    for i in range(0, len(thinking), chunk_size):
                        self._write_sse("content_block_delta", {
                            "type": "content_block_delta",
                            "index": idx,
                            "delta": {"type": "thinking_delta", "thinking": thinking[i:i + chunk_size]},
                        })
                    signature = block.get("signature")
                    if signature:
                        self._write_sse("content_block_delta", {
                            "type": "content_block_delta",
                            "index": idx,
                            "delta": {"type": "signature_delta", "signature": signature},
                        })
                else:
                    self._write_sse("content_block_start", {
                        "type": "content_block_start",
                        "index": idx,
                        "content_block": {"type": "text", "text": ""},
                    })
                    text = block.get("text", "")
                    if text:
                        chunk_size = 100
                        for i in range(0, len(text), chunk_size):
                            self._write_sse("content_block_delta", {
                                "type": "content_block_delta",
                                "index": idx,
                                "delta": {"type": "text_delta", "text": text[i:i + chunk_size]},
                            })

                self._write_sse("content_block_stop", {
                    "type": "content_block_stop",
                    "index": idx,
                })

            usage = resp.get("usage", {})
            self._write_sse("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": resp.get("stop_reason", "end_turn"), "stop_sequence": None},
                "usage": {"output_tokens": usage.get("output_tokens", 0)},
            })

            self._write_sse("message_stop", {"type": "message_stop"})
            # Keep the HTTP connection reusable for Claude's follow-up
            # requests (session title and tool-result turns).
            self.close_connection = True

        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            logger.exception("litellm proxy: synthetic streaming error")

    def _handle_streaming(self, response: Any, *, request_id: str = "") -> str:
        """Translate LiteLLM OpenAI chunks to Anthropic SSE immediately.

        Returns an outcome tag.  ``failed_before_write`` means the stream died
        without anything reaching the client, so the caller may safely retry;
        every other failure has already emitted SSE and must not be replayed.
        """
        started = False
        outcome = "unknown"
        index = 0
        active_blocks: dict[int, str] = {}
        tool_block_indices: dict[int, int] = {}
        pending_tools: dict[int, dict[str, Any]] = {}
        reasoning_block_index: int | None = None
        final_stop_reason = "end_turn"
        sse_buffer = b""
        saw_terminal = False
        item_count = 0
        byte_count = 0

        def start_block(block_index: int, block_type: str, block: dict[str, Any]) -> None:
            if block_index in active_blocks:
                return
            active_blocks[block_index] = block_type
            self._write_sse("content_block_start", {
                "type": "content_block_start", "index": block_index,
                "content_block": block,
            })

        def stop_blocks() -> None:
            for block_index in list(active_blocks):
                self._write_sse("content_block_stop", {
                    "type": "content_block_stop", "index": block_index,
                })
                del active_blocks[block_index]

        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            for item in response:
                item_count += 1
                # LiteLLM's Anthropic adapter may already yield wire-format
                # SSE bytes. Forward them unchanged instead of treating them
                # as OpenAI response objects.
                if isinstance(item, (bytes, bytearray)):
                    byte_count += len(item)
                    sse_buffer += bytes(item)
                    if b"\n\n" not in sse_buffer:
                        continue
                    raw, sse_buffer = sse_buffer.split(b"\n\n", 1)
                    raw += b"\n\n"
                    out_lines = []
                    for line in raw.splitlines(keepends=True):
                        if not line.endswith(b"\n"):
                            line += b"\n"
                        # LiteLLM can emit the SSE event label as a separate
                        # frame. Claude's stream parser expects JSON data
                        # frames, so omit orphan labels and retain data.
                        if line.startswith(b"event:"):
                            continue
                        if line.startswith(b"data: "):
                            try:
                                payload = json.loads(line[6:])
                                if payload.get("type") == "message_stop":
                                    saw_terminal = True
                                if payload.get("type") == "message_delta" and (
                                    payload.get("delta", {}).get("stop_reason")
                                ):
                                    saw_terminal = True
                                if payload.get("type") == "content_block_delta":
                                    idx = int(payload.get("index", 0))
                                line = b"data: " + json.dumps(payload, default=str).encode() + b"\n"
                            except (json.JSONDecodeError, UnicodeDecodeError):
                                pass
                        out_lines.append(line)
                    self.wfile.write(b"".join(out_lines))
                    self.wfile.flush()
                    continue
                if isinstance(item, str) and item.startswith("event:"):
                    self.wfile.write(item.encode())
                    self.wfile.flush()
                    continue
                if isinstance(item, dict):
                    chunk = item
                elif hasattr(item, "model_dump"):
                    chunk = item.model_dump()
                elif hasattr(item, "dict"):
                    chunk = item.dict()
                elif hasattr(item, "choices"):
                    chunk = {
                        "id": getattr(item, "id", "msg_proxy"),
                        "model": getattr(item, "model", self.litellm_model),
                        "choices": getattr(item, "choices", []),
                    }
                else:
                    try:
                        chunk = dict(item)
                    except (TypeError, ValueError):
                        logger.warning("litellm proxy: ignoring unsupported stream item %r", item)
                        continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0] or {}
                delta = choice.get("delta") or {}
                if not started:
                    self._write_sse("message_start", {"type": "message_start", "message": {
                        "id": chunk.get("id", "msg_proxy"), "type": "message", "role": "assistant",
                        "content": [], "model": chunk.get("model", self.litellm_model),
                        "stop_reason": None, "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    }})
                    started = True
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                if reasoning:
                    # Preserve reasoning in the same Anthropic shape emitted by
                    # the non-streaming adapter.  Dropping reasoning deltas can
                    # leave Claude with an empty assistant message (and no
                    # tool call), which is serialized as a completed run with
                    # no task outputs.
                    if reasoning_block_index is None:
                        while index in active_blocks:
                            index += 1
                        reasoning_block_index = index
                        start_block(reasoning_block_index, "thinking", {
                            "type": "thinking", "thinking": "",
                        })
                    self._write_sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": reasoning_block_index,
                        "delta": {"type": "thinking_delta", "thinking": str(reasoning)},
                    })
                text = delta.get("content")
                if text:
                    if index not in active_blocks:
                        start_block(index, "text", {"type": "text", "text": ""})
                    self._write_sse("content_block_delta", {"type": "content_block_delta", "index": index,
                        "delta": {"type": "text_delta", "text": text}})
                tool_calls = delta.get("tool_calls") or []
                for tool in tool_calls:
                    source_index = int(tool.get("index", 0) or 0)
                    if source_index in tool_block_indices:
                        block_index = tool_block_indices[source_index]
                    else:
                        while index in active_blocks:
                            index += 1
                        block_index = index
                        tool_block_indices[source_index] = block_index
                        index += 1
                    function = tool.get("function") or {}
                    pending = pending_tools.setdefault(source_index, {
                        "id": tool.get("id"), "name": "", "arguments": "", "started": False,
                    })
                    if tool.get("id"):
                        pending["id"] = tool["id"]
                    if function.get("name"):
                        pending["name"] += str(function["name"])
                    if function.get("arguments"):
                        pending["arguments"] += str(function["arguments"])
                    if not pending["started"] and pending["name"]:
                        pending["started"] = True
                        start_block(block_index, "tool_use", {
                            "type": "tool_use", "id": pending["id"] or f"toolu_{block_index}",
                            "name": pending["name"], "input": {},
                        })
                    if pending["started"] and function.get("arguments"):
                        self._write_sse("content_block_delta", {"type": "content_block_delta", "index": block_index,
                            "delta": {"type": "input_json_delta", "partial_json": str(function["arguments"])}})
                finish = choice.get("finish_reason")
                if finish:
                    saw_terminal = True
                    # Mirror the non-streaming mapping in
                    # ``_openai_response_to_anthropic``. Collapsing ``length``
                    # into ``end_turn`` hides output-budget truncation from the
                    # CLI, which then treats a thinking-only reply as a normal
                    # finished turn and ends the session with no task outputs.
                    if finish == "tool_calls":
                        final_stop_reason = "tool_use"
                    elif finish == "length":
                        final_stop_reason = "max_tokens"
                    elif finish == "content_filter":
                        final_stop_reason = "stop_sequence"
                    else:
                        final_stop_reason = "end_turn"
            if sse_buffer:
                logger.warning(
                    "litellm proxy stream_incomplete_tail id=%s bytes=%d tail_bytes=%d",
                    request_id, byte_count, len(sse_buffer),
                )
            if not saw_terminal:
                logger.error(
                    "litellm proxy stream_incomplete id=%s items=%d bytes=%d started=%s",
                    request_id, item_count, byte_count, started,
                )
                if started:
                    stop_blocks()
                self._write_sse("error", {"type": "error", "error": {
                    "type": "api_error",
                    "message": "Upstream stream ended before a terminal message_stop",
                }})
                self.close_connection = True
                outcome = "incomplete_after_write" if started else "failed_before_write"
                return outcome
            logger.warning(
                "litellm proxy stream_complete id=%s items=%d bytes=%d stop_reason=%s",
                request_id, item_count, byte_count, final_stop_reason,
            )
            if started:
                stop_blocks()
                self._write_sse("message_delta", {"type": "message_delta",
                    "delta": {"stop_reason": final_stop_reason, "stop_sequence": None},
                    "usage": {"output_tokens": 0}})
                self._write_sse("message_stop", {"type": "message_stop"})
                self.close_connection = True
            outcome = "ok"
        except (BrokenPipeError, ConnectionResetError):
            logger.info("litellm proxy: downstream client disconnected during streaming")
            outcome = "client_gone"
        except Exception:
            # A mid-stream upstream failure (seen in the wild as httpcore
            # ReadError "[Errno 9] Bad file descriptor") used to be logged and
            # nothing more, so the client just saw the connection stop with no
            # HTTP status and no SSE error.  Report it the same way an
            # early-terminating stream is reported, so the caller can retry
            # against a real error instead of guessing.
            logger.exception("litellm proxy stream_failed id=%s items=%d bytes=%d started=%s",
                             request_id, item_count, byte_count, started)
            outcome = "failed_after_write" if started else "failed_before_write"
            try:
                if started:
                    stop_blocks()
                self._write_sse("error", {"type": "error", "error": {
                    "type": "api_error",
                    "message": "Upstream stream failed mid-response",
                }})
            except (BrokenPipeError, ConnectionResetError):
                outcome = "client_gone"
            self.close_connection = True
        finally:
            # Without this, a request that never reaches any of the branches
            # above (a handler thread still blocked when the process exits)
            # leaves nothing but its request_start line, which makes the
            # failure impossible to attribute after the fact.
            type(self)._inflight.pop(request_id, None)
            logger.warning("litellm proxy stream_outcome id=%s outcome=%s items=%d",
                           request_id, outcome, item_count)
        return outcome

    def _write_sse(self, event: str, data: dict) -> None:
        line = f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"
        self.wfile.write(line.encode())
        self.wfile.flush()

    def _send_error(self, code: int, message: str) -> None:
        body = json.dumps({"error": message}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_anthropic_error(self, code: int, message: str) -> None:
        """Return error in Anthropic API error format so Claude CLI can parse it."""
        body = json.dumps({
            "type": "error",
            "error": {
                "type": "api_error",
                "message": message,
            },
        }).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        pass


class LiteLLMProxy:
    """Local proxy that translates Anthropic Messages API to any litellm backend.

    Parameters
    ----------
    model : str
        litellm model identifier with provider prefix (e.g. ``openai/deepseek-chat``,
        ``gemini/gemini-pro``, ``ollama/llama3``).
    api_base : str or None
        Target API endpoint URL.
    api_key : str or None
        API key for the target service.
    port : int
        Local port to bind. ``0`` picks a random free port (recommended).
    supports_image_input : bool
        Whether the upstream endpoint accepts image content blocks. Defaults
        to ``False`` so unknown endpoints fail safe; visual endpoints must opt
        in explicitly.
    """

    def __init__(
        self,
        model: str,
        api_base: str | None = None,
        api_key: str | None = None,
        port: int = 0,
        supports_image_input: bool = False,
    ) -> None:
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self._port = port
        self.supports_image_input = supports_image_input
        self._server: http.server.HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._handler_cls: type | None = None

    @property
    def local_url(self) -> str:
        if self._server is None:
            raise RuntimeError("Proxy not started")
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def start(self) -> str:
        """Start the proxy in a daemon thread. Returns the local URL."""
        # Proxy diagnostics need a handler of their own.  ``logger.setLevel``
        # alone does not publish them: the benchmark CLI installs its handler on
        # the root logger at WARNING, and a record still has to clear the
        # handler's level after it clears the logger's.  That is why the
        # per-request INFO diagnostics never reached the pod log, which in turn
        # hid every mid-stream client disconnect.
        proxy_log_level = os.getenv("ASIBENCH_PROXY_LOG_LEVEL", "INFO").upper()
        level = getattr(logging, proxy_log_level, logging.INFO)
        logger.setLevel(level)
        if not any(getattr(h, "_asibench_proxy", False) for h in logger.handlers):
            handler = logging.StreamHandler()
            handler.setLevel(level)
            handler.setFormatter(logging.Formatter("%(message)s"))
            handler._asibench_proxy = True  # type: ignore[attr-defined]
            logger.addHandler(handler)
        # Emit through this handler only, so the root handler cannot duplicate
        # records that do clear its level.
        logger.propagate = False
        routing_key = os.getenv("ASIBENCH_ROUTING_KEY") or uuid.uuid4().hex
        handler_cls = type("Handler", (_LiteLLMProxyHandler,), {
            "litellm_model": self.model,
            "litellm_api_base": self.api_base,
            "litellm_api_key": self.api_key,
            "routing_key": routing_key,
            "supports_image_input": self.supports_image_input,
            # Per-proxy, not inherited: one process runs many proxies at the
            # parallelism the benchmark is launched with, and a shared dict
            # would make each one report the others' requests.
            "_inflight": {},
        })
        self._server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", self._port), handler_cls,
        )
        self._handler_cls = handler_cls
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True,
        )
        self._thread.start()
        url = self.local_url
        logger.info("litellm proxy started: %s → %s (%s)", url, self.api_base, self.model)
        return url

    def stop(self) -> None:
        leftover = dict(getattr(self._handler_cls, "_inflight", {}) or {})
        if leftover:
            # Handler threads are daemons, so anything still blocked here is
            # about to be killed without reaching its own logging.  Name the
            # requests now or they leave only a request_start line behind.
            logger.error("litellm proxy stopped with %d request(s) still in flight: %s",
                         len(leftover), ", ".join(sorted(leftover)))
        if self._server is not None:
            self._server.shutdown()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __del__(self) -> None:
        self.stop()


# ── TokenRouter Gemini OpenAI Chat proxy (Claude Code → chat/completions) ───


class _TokenRouterOpenAIChatProxyHandler(_LiteLLMProxyHandler):
    """Bridge Anthropic Messages from Claude Code to TokenRouter chat/completions.

    TokenRouter's native Anthropic Gemini route currently emits tool_use blocks
    without the Gemini thought signatures needed for the next turn. Its OpenAI
    chat route does return and accept ``thought_signature`` on ``tool_calls``.
    This handler keeps Claude Code's Anthropic interface while using that
    working OpenAI route upstream.
    """

    upstream_base_url: str
    upstream_api_key: str | None
    model: str
    effort: str
    thought_signature_cache: dict[str, str]
    thought_signature_lock: threading.Lock
    tool_id_counter: list[int]

    def do_POST(self) -> None:
        if "/v1/messages" not in self.path:
            self._send_error(404, f"Not found: {self.path}")
            return

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._send_error(400, "Empty request body")
            return

        try:
            body = json.loads(self.rfile.read(content_length))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            self._send_error(400, f"Invalid JSON: {e}")
            return

        body = prepare_model_input_for_endpoint(
            body,
            supports_image_input=self.supports_image_input,
        )
        is_stream = body.get("stream", False)
        payload = self._build_openai_payload(body)
        self._maybe_dump_debug_payload(payload)
        req = urllib.request.Request(
            self._chat_completions_url(),
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.upstream_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=None) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            raw = e.read()
            logger.warning(
                "tokenrouter openai chat proxy: upstream HTTP %s: %s",
                e.code,
                raw.decode("utf-8", errors="ignore")[:1000],
            )
            self._send_anthropic_error(e.code, raw.decode("utf-8", errors="ignore"))
            return
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as e:
            logger.exception("tokenrouter openai chat proxy: upstream call failed")
            self._send_anthropic_error(502, f"Upstream error: {type(e).__name__}: {e}")
            return

        try:
            upstream = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_anthropic_error(502, "Upstream returned invalid JSON")
            return

        anthropic = self._openai_response_to_anthropic(upstream)
        if is_stream:
            self._handle_synthetic_streaming(anthropic)
        else:
            self._send_anthropic_json(anthropic)

    def _build_openai_payload(self, body: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._anthropic_messages_to_openai(body),
            "max_tokens": body.get("max_tokens", 4096),
        }
        payload["reasoning_effort"] = _openai_reasoning_effort_from_claude(self.effort)
        tools = body.get("tools")
        if isinstance(tools, list) and tools:
            payload["tools"] = [self._anthropic_tool_to_openai(tool) for tool in tools]
        tool_choice = body.get("tool_choice")
        if isinstance(tool_choice, dict):
            choice_type = tool_choice.get("type")
            if choice_type == "auto":
                payload["tool_choice"] = "auto"
            elif choice_type == "any":
                payload["tool_choice"] = "required"
            elif choice_type == "tool" and tool_choice.get("name"):
                payload["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tool_choice["name"]},
                }
        for key in ("temperature", "top_p", "stop"):
            if key in body and body[key] is not None:
                payload[key] = body[key]
        if "stop_sequences" in body and body["stop_sequences"] is not None:
            payload["stop"] = body["stop_sequences"]
        return payload

    def _anthropic_messages_to_openai(self, body: dict[str, Any]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        system = self._anthropic_system_to_text(body.get("system"))
        if system:
            messages.append({"role": "system", "content": system})

        for message in body.get("messages", []):
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = message.get("content")
            if role == "assistant":
                messages.append(self._anthropic_assistant_to_openai(content))
            elif role == "user":
                messages.extend(self._anthropic_user_to_openai(content))
        return messages

    def _anthropic_system_to_text(self, system: Any) -> str:
        if isinstance(system, str):
            return system
        if isinstance(system, list):
            return "\n".join(
                str(block.get("text", ""))
                for block in system
                if isinstance(block, dict) and block.get("type") == "text"
            ).strip()
        return ""

    def _anthropic_assistant_to_openai(self, content: Any) -> dict[str, Any]:
        if isinstance(content, str):
            return {"role": "assistant", "content": content}
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        reasoning_parts: list[str] = []
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    text_parts.append(str(block))
                    continue
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(str(block.get("text", "")))
                elif btype == "tool_use":
                    tool_id = str(block.get("id") or self._next_tool_id())
                    tool_call = {
                        "id": tool_id,
                        "type": "function",
                        "function": {
                            "name": str(block.get("name", "")),
                            "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                        },
                    }
                    signature = self._signature_for_tool_block(block, tool_id)
                    if signature:
                        tool_call["thought_signature"] = signature
                    tool_calls.append(tool_call)
                elif btype == "thinking":
                    thinking_text = block.get("thinking", "")
                    if thinking_text:
                        reasoning_parts.append(str(thinking_text))
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": "\n".join(part for part in text_parts if part) or None,
        }
        if reasoning_parts and self._model_supports_reasoning_content():
            msg["reasoning_content"] = "\n\n".join(reasoning_parts)
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return msg

    def _anthropic_user_to_openai(self, content: Any) -> list[dict[str, Any]]:
        if isinstance(content, str):
            return [{"role": "user", "content": content}]
        if not isinstance(content, list):
            return [{"role": "user", "content": str(content)}]

        messages: list[dict[str, Any]] = []
        text_parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                text_parts.append(str(block))
                continue
            btype = block.get("type")
            if btype == "text":
                text_parts.append(str(block.get("text", "")))
            elif btype == "tool_result":
                if text_parts:
                    messages.append({"role": "user", "content": "\n".join(text_parts)})
                    text_parts = []
                messages.append({
                    "role": "tool",
                    "tool_call_id": str(block.get("tool_use_id", "")),
                    "content": self._tool_result_content_to_text(block.get("content")),
                })
        if text_parts:
            messages.append({"role": "user", "content": "\n".join(text_parts)})
        return messages or [{"role": "user", "content": ""}]

    def _tool_result_content_to_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        return str(content)

    def _anthropic_tool_to_openai(self, tool: Any) -> dict[str, Any]:
        if not isinstance(tool, dict):
            return {"type": "function", "function": {"name": "tool", "parameters": {"type": "object"}}}
        schema = strip_gemini_unsupported_schema_keys(tool.get("input_schema") or {"type": "object"})
        return {
            "type": "function",
            "function": {
                "name": tool.get("name", "tool"),
                "description": tool.get("description", ""),
                "parameters": schema,
            },
        }

    def _openai_response_to_anthropic(self, upstream: dict[str, Any]) -> dict[str, Any]:
        choices = upstream.get("choices") or []
        choice = choices[0] if choices else {}
        message = choice.get("message") if isinstance(choice, dict) else {}
        if not isinstance(message, dict):
            message = {}

        content_blocks: list[dict[str, Any]] = []
        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            content_blocks.append({"type": "thinking", "thinking": reasoning})
        text = message.get("content")
        if isinstance(text, str) and text:
            content_blocks.append({"type": "text", "text": text})

        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                block = self._openai_tool_call_to_anthropic(call)
                if block is not None:
                    content_blocks.append(block)

        finish = choice.get("finish_reason")
        stop_reason = "tool_use" if any(b.get("type") == "tool_use" for b in content_blocks) else "end_turn"
        if finish in ("length", "content_filter"):
            stop_reason = "max_tokens" if finish == "length" else "stop_sequence"
        usage = upstream.get("usage") if isinstance(upstream.get("usage"), dict) else {}
        return {
            "id": upstream.get("id", "msg_tokenrouter_proxy"),
            "type": "message",
            "role": "assistant",
            "content": content_blocks or [{"type": "text", "text": ""}],
            "model": upstream.get("model", self.model),
            "stop_reason": stop_reason,
            "usage": {
                "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens", 0)),
                "output_tokens": usage.get("completion_tokens", usage.get("output_tokens", 0)),
            },
        }

    def _openai_tool_call_to_anthropic(self, call: Any) -> dict[str, Any] | None:
        if not isinstance(call, dict):
            return None
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        raw_args = function.get("arguments") or "{}"
        try:
            tool_input = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            tool_input = {"input": raw_args}
        tool_id = self._next_tool_id()
        signature = _get_signature_value(call)
        if signature:
            with self.thought_signature_lock:
                self.thought_signature_cache[tool_id] = signature
        return {
            "type": "tool_use",
            "id": tool_id,
            "name": str(function.get("name", "")),
            "input": tool_input,
        }

    def _model_supports_reasoning_content(self) -> bool:
        return "deepseek" in self.model.lower()

    def _signature_for_tool_block(self, block: dict[str, Any], tool_id: str) -> str | None:
        signature = _get_signature_value(block)
        if signature:
            return signature
        with self.thought_signature_lock:
            return self.thought_signature_cache.get(tool_id)

    def _next_tool_id(self) -> str:
        with self.thought_signature_lock:
            self.tool_id_counter[0] += 1
            return f"toolu_gemini_{self.tool_id_counter[0]:06d}"

    def _chat_completions_url(self) -> str:
        base = self.upstream_base_url.rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    def _maybe_dump_debug_payload(self, body: dict[str, Any]) -> None:
        debug_dir = os.environ.get("AI4SCI_DEBUG_API_PROXY_DIR")
        if not debug_dir:
            return
        try:
            path = Path(debug_dir)
            path.mkdir(parents=True, exist_ok=True)
            stem = f"tokenrouter_openai_chat_{time.time_ns()}.json"
            (path / stem).write_text(
                json.dumps(body, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            logger.exception("tokenrouter openai chat proxy: failed to dump debug payload")

    def _send_anthropic_json(self, body: dict[str, Any]) -> None:
        data = json.dumps(body, default=str).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        pass


class TokenRouterOpenAIChatProxy:
    """Local proxy that maps Anthropic Messages to TokenRouter OpenAI chat."""

    def __init__(
        self,
        model: str,
        api_base: str | None = None,
        api_key: str | None = None,
        effort: str = "medium",
        port: int = 0,
        supports_image_input: bool = False,
    ) -> None:
        if api_base is None:
            raise ValueError("api_base is required")
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.effort = effort
        self._port = port
        self.supports_image_input = supports_image_input
        self._server: http.server.HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._thought_signature_cache: dict[str, str] = {}
        self._thought_signature_lock = threading.Lock()
        self._tool_id_counter = [0]

    @property
    def local_url(self) -> str:
        if self._server is None:
            raise RuntimeError("Proxy not started")
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def start(self) -> str:
        handler = type("Handler", (_TokenRouterOpenAIChatProxyHandler,), {
            "litellm_model": self.model,
            "upstream_base_url": self.api_base,
            "upstream_api_key": self.api_key,
            "model": self.model,
            "effort": self.effort,
            "supports_image_input": self.supports_image_input,
            "thought_signature_cache": self._thought_signature_cache,
            "thought_signature_lock": self._thought_signature_lock,
            "tool_id_counter": self._tool_id_counter,
        })
        self._server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", self._port), handler,
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True,
        )
        self._thread.start()
        url = self.local_url
        logger.info("tokenrouter openai chat proxy started: %s → %s (%s)", url, self.api_base, self.model)
        return url

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __del__(self) -> None:
        self.stop()


# ── Anthropic passthrough/rewrite proxy (for TokenRouter native endpoint) ────


class _AnthropicRewriteProxyHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler that rewrites selected Anthropic payload fields, then forwards."""

    upstream_base_url: str
    upstream_api_key: str | None
    model: str
    effort: str
    gemini_thought_signature_cache: dict[str, str]
    gemini_thought_signature_lock: threading.Lock
    gemini_tool_id_counter: list[int]
    supports_image_input: bool = False

    def do_POST(self) -> None:
        if "/v1/messages" not in self.path:
            self._send_error(404, f"Not found: {self.path}")
            return

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._send_error(400, "Empty request body")
            return

        try:
            body = json.loads(self.rfile.read(content_length))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            self._send_error(400, f"Invalid JSON: {e}")
            return

        body = prepare_model_input_for_endpoint(
            body,
            supports_image_input=self.supports_image_input,
        )
        rewritten = rewrite_anthropic_payload_for_tokenrouter(
            body,
            model=self.model,
            effort=self.effort,
            gemini_thought_signature_cache=self.gemini_thought_signature_cache,
        )
        self._maybe_dump_debug_payload(rewritten)
        payload = json.dumps(rewritten).encode()

        headers = {
            "Content-Type": "application/json",
            "Accept": self.headers.get("Accept", "application/json"),
        }
        anthropic_version = self.headers.get("anthropic-version")
        if anthropic_version:
            headers["anthropic-version"] = anthropic_version
        beta = self.headers.get("anthropic-beta")
        if beta:
            headers["anthropic-beta"] = beta
        api_key = self.upstream_api_key or self.headers.get("x-api-key")
        if api_key:
            headers["x-api-key"] = api_key
            headers["Authorization"] = f"Bearer {api_key}"

        req = urllib.request.Request(
            self._messages_url(),
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=None) as resp:
                self.send_response(resp.status)
                self._copy_response_headers(resp.headers)
                self.end_headers()
                self._copy_body(resp)
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self._copy_response_headers(e.headers)
            self.end_headers()
            self.wfile.write(e.read())
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            logger.exception("anthropic rewrite proxy: upstream call failed")
            self._send_anthropic_error(502, f"Upstream error: {type(e).__name__}: {e}")

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok", "proxy": "anthropic-rewrite"}).encode())

    def _messages_url(self) -> str:
        base = self.upstream_base_url.rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/messages"
        return f"{base}/v1/messages"

    def _maybe_dump_debug_payload(self, body: dict[str, Any]) -> None:
        debug_dir = os.environ.get("AI4SCI_DEBUG_API_PROXY_DIR")
        if not debug_dir:
            return
        try:
            path = Path(debug_dir)
            path.mkdir(parents=True, exist_ok=True)
            stem = f"anthropic_rewrite_{time.time_ns()}.json"
            (path / stem).write_text(
                json.dumps(body, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            logger.exception("anthropic rewrite proxy: failed to dump debug payload")

    def _copy_response_headers(self, headers: Any) -> None:
        skip = {"connection", "content-length", "transfer-encoding"}
        for key, value in headers.items():
            if key.lower() not in skip:
                self.send_header(key, value)

    def _copy_body(self, resp: Any) -> None:
        if "gemini" not in self.model.lower():
            while True:
                chunk = resp.read(64 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
            return

        processed_chunks: list[bytes] = []
        buffer = b""
        saw_stream = False
        while True:
            chunk = resp.read(64 * 1024)
            if not chunk:
                break
            buffer += chunk
            if not saw_stream:
                stripped = buffer.lstrip()
                if stripped.startswith((b"event:", b"data:", b":")):
                    saw_stream = True
                else:
                    continue
            while saw_stream and b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line += b"\n"
                if saw_stream:
                    out = _rewrite_gemini_response_sse_line(line, self._next_gemini_tool_id)
                    processed_chunks.append(out)
                    self.wfile.write(out)
                    self.wfile.flush()

        if buffer:
            if saw_stream:
                out = _rewrite_gemini_response_sse_line(buffer, self._next_gemini_tool_id)
            else:
                out = _rewrite_gemini_response_tool_use_ids(
                    buffer,
                    self._next_gemini_tool_id,
                )
            processed_chunks.append(out)
            self.wfile.write(out)
            self.wfile.flush()

        if processed_chunks:
            _cache_gemini_tool_thought_signatures(
                b"".join(processed_chunks),
                self.gemini_thought_signature_cache,
                self.gemini_thought_signature_lock,
            )

    def _next_gemini_tool_id(self) -> str:
        with self.gemini_thought_signature_lock:
            self.gemini_tool_id_counter[0] += 1
            return f"toolu_gemini_{self.gemini_tool_id_counter[0]:06d}"

    def _send_error(self, code: int, message: str) -> None:
        body = json.dumps({"error": message}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_anthropic_error(self, code: int, message: str) -> None:
        body = json.dumps({
            "type": "error",
            "error": {
                "type": "api_error",
                "message": message,
            },
        }).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        pass


class AnthropicRewriteProxy:
    """Local proxy for Anthropic-compatible endpoints requiring light rewrites."""

    def __init__(
        self,
        model: str,
        api_base: str,
        api_key: str | None = None,
        effort: str = "medium",
        port: int = 0,
        supports_image_input: bool = False,
    ) -> None:
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.effort = effort
        self._port = port
        self.supports_image_input = supports_image_input
        self._server: http.server.HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._gemini_thought_signature_cache: dict[str, str] = {}
        self._gemini_thought_signature_lock = threading.Lock()
        self._gemini_tool_id_counter = [0]

    @property
    def local_url(self) -> str:
        if self._server is None:
            raise RuntimeError("Proxy not started")
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def start(self) -> str:
        handler = type("Handler", (_AnthropicRewriteProxyHandler,), {
            "upstream_base_url": self.api_base,
            "upstream_api_key": self.api_key,
            "model": self.model,
            "effort": self.effort,
            "supports_image_input": self.supports_image_input,
            "gemini_thought_signature_cache": self._gemini_thought_signature_cache,
            "gemini_thought_signature_lock": self._gemini_thought_signature_lock,
            "gemini_tool_id_counter": self._gemini_tool_id_counter,
        })
        self._server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", self._port), handler,
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True,
        )
        self._thread.start()
        url = self.local_url
        logger.info("anthropic rewrite proxy started: %s → %s (%s)", url, self.api_base, self.model)
        return url

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __del__(self) -> None:
        self.stop()


# ── OpenAI Chat Completions proxy (for Codex CLI) ─────────────


class _LiteLLMOpenAIProxyHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler that bridges OpenAI Chat Completions API to litellm."""

    litellm_model: str
    litellm_api_base: str | None
    litellm_api_key: str | None
    translate_responses: bool = False
    supports_image_input: bool = False

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._send_error(400, "Empty request body")
            return

        raw_body = self.rfile.read(content_length)

        # Responses API: translate via litellm.responses() when opted-in
        # (so upstreams that don't implement /v1/responses still work); else
        # forward raw to upstream with only model/key rewrite.
        if "responses" in self.path:
            if self.translate_responses:
                self._handle_responses_translated(raw_body)
            else:
                self._handle_responses_passthrough(raw_body)
            return

        if "chat/completions" not in self.path:
            self._send_error(404, f"Not found: {self.path}")
            return

        try:
            body = json.loads(raw_body)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            self._send_error(400, f"Invalid JSON: {e}")
            return

        body = self._prepare_model_input(body)
        is_stream = body.get("stream", False)
        kwargs = self._build_litellm_kwargs(body)
        # Always call upstream non-streaming to avoid compatibility issues
        # with reasoning models and provider-specific streaming formats.
        # Synthesize SSE events from the complete response if client wants streaming.
        kwargs["stream"] = False

        try:
            import litellm
            response = litellm.completion(**kwargs)
        except Exception as e:
            logger.exception("litellm openai proxy: upstream call failed")
            self._send_openai_error(502, f"Upstream error: {type(e).__name__}: {e}")
            return

        if is_stream:
            self._handle_synthetic_openai_streaming(response)
        else:
            self._handle_non_streaming(response)

    def _handle_responses_translated(self, raw_body: bytes) -> None:
        """Translate Responses API via ``litellm.responses()``.

        Used when the upstream doesn't implement ``/v1/responses`` directly
        (e.g. MiMo Code's Anthropic-shape endpoint only exposes
        ``/v1/messages``). ``litellm.responses()`` handles the wire
        translation based on the model's provider prefix, so this works
        for both ``openai/...`` and ``anthropic/...`` backends.
        """
        try:
            body = json.loads(raw_body)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            self._send_error(400, f"Invalid JSON: {e}")
            return

        body = self._prepare_model_input(body)
        is_stream = bool(body.get("stream"))

        # Build litellm.responses kwargs.
        kwargs: dict[str, Any] = {"model": self.litellm_model}
        for key in (
            "input", "instructions", "max_output_tokens", "reasoning",
            "tools", "tool_choice", "temperature", "top_p",
            "parallel_tool_calls", "previous_response_id", "metadata",
            "text", "truncation", "store", "user",
        ):
            if key in body and body[key] is not None:
                kwargs[key] = body[key]

        if self.litellm_api_base:
            kwargs["api_base"] = self.litellm_api_base
        if self.litellm_api_key:
            kwargs["api_key"] = self.litellm_api_key

        # Call upstream non-streaming; synthesize SSE if the client asked for it.
        # Streaming is deliberately dropped here because upstream stream formats
        # differ across providers and can conflict with reasoning models.
        kwargs["stream"] = False

        try:
            import litellm
            # drop unsupported params (e.g. reasoning_effort for Anthropic)
            # instead of raising — MiMo / Codex CLIs pass Responses-shaped
            # params that don't universally map.
            litellm.drop_params = True
            response = litellm.responses(**kwargs)
        except Exception as e:
            logger.exception("responses translation: litellm.responses failed")
            self._send_openai_error(502, f"Upstream error: {type(e).__name__}: {e}")
            return

        try:
            resp_dict = response.model_dump() if hasattr(response, "model_dump") else dict(response)
        except Exception:
            try:
                resp_dict = json.loads(json.dumps(response, default=str))
            except Exception:
                resp_dict = {"error": "Failed to serialize response"}

        if is_stream:
            self._handle_synthetic_responses_streaming(resp_dict)
        else:
            resp_json = json.dumps(resp_dict, default=str).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_json)))
            self.end_headers()
            try:
                self.wfile.write(resp_json)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def _handle_synthetic_responses_streaming(self, resp_dict: dict) -> None:
        """Emit a minimal Responses API SSE sequence from a completed response.

        We synthesize the events the OpenCode / MiMo family of CLIs actually
        consume: ``response.created`` → per-output-item added/done →
        ``response.completed``. Text output items also stream a single
        ``response.output_text.delta`` with the full text and a matching
        ``.done`` event so the CLI's incremental UI still updates.

        Reasoning items are skipped in the streamed sequence — some CLIs
        treat a stream that starts with a reasoning delta as an empty
        assistant turn and never look at the subsequent message item.
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def emit(event_type: str, payload: dict) -> None:
            payload = {"type": event_type, **payload}
            frame = f"event: {event_type}\ndata: {json.dumps(payload, default=str)}\n\n"
            try:
                self.wfile.write(frame.encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                raise

        # Rewrite item IDs so message ids get msg_... prefix that Responses
        # consumers expect (litellm hands back chatcmpl-... for both the
        # response and its message item, which confuses OpenCode-family CLIs).
        response_id = resp_dict.get("id") or "resp_synth"
        if not response_id.startswith("resp_"):
            response_id = "resp_" + response_id.replace("chatcmpl-", "")
        resp_dict = dict(resp_dict)
        resp_dict["id"] = response_id

        rewritten_output: list[dict] = []
        for i, item in enumerate(resp_dict.get("output") or []):
            item = dict(item)
            if item.get("type") == "message" and not str(item.get("id", "")).startswith("msg_"):
                item["id"] = f"msg_{response_id}_{i}"
            rewritten_output.append(item)
        resp_dict["output"] = rewritten_output

        try:
            emit("response.created", {"response": resp_dict})
            emit("response.in_progress", {"response": resp_dict})

            for idx, item in enumerate(resp_dict["output"]):
                itype = item.get("type")
                if itype == "reasoning":
                    # Skip reasoning entirely in the streamed sequence.
                    continue

                emit("response.output_item.added", {"output_index": idx, "item": item})

                if itype == "message":
                    for cidx, part in enumerate(item.get("content") or []):
                        emit("response.content_part.added", {
                            "output_index": idx, "content_index": cidx,
                            "item_id": item.get("id"), "part": part,
                        })
                        if part.get("type") in ("output_text", "text"):
                            text = part.get("text", "") or ""
                            emit("response.output_text.delta", {
                                "output_index": idx, "content_index": cidx,
                                "item_id": item.get("id"), "delta": text,
                            })
                            emit("response.output_text.done", {
                                "output_index": idx, "content_index": cidx,
                                "item_id": item.get("id"), "text": text,
                            })
                        emit("response.content_part.done", {
                            "output_index": idx, "content_index": cidx,
                            "item_id": item.get("id"), "part": part,
                        })
                elif itype in ("function_call", "custom_tool_call"):
                    args = item.get("arguments", "") or ""
                    if args:
                        emit("response.function_call_arguments.delta", {
                            "output_index": idx, "item_id": item.get("id"),
                            "delta": args,
                        })
                        emit("response.function_call_arguments.done", {
                            "output_index": idx, "item_id": item.get("id"),
                            "arguments": args,
                        })

                emit("response.output_item.done", {"output_index": idx, "item": item})

            emit("response.completed", {"response": resp_dict})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            logger.exception("synthetic responses streaming failed")

    def _handle_responses_passthrough(self, raw_body: bytes) -> None:
        """Forward Responses API requests to upstream, rewriting model name and auth."""
        import http.client
        import ssl
        from urllib.parse import urlparse

        try:
            body = json.loads(raw_body)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            self._send_error(400, f"Invalid JSON: {e}")
            return

        body = self._prepare_model_input(body)
        real_model = self.litellm_model
        for prefix in ("openai/", "hosted_vllm/"):
            if real_model.startswith(prefix):
                real_model = real_model[len(prefix):]
                break
        body["model"] = real_model

        upstream_base = (self.litellm_api_base or "").rstrip("/")
        if upstream_base.endswith("/v1"):
            upstream_url = f"{upstream_base}/responses"
        else:
            upstream_url = f"{upstream_base}/v1/responses"

        api_key = self.litellm_api_key or ""
        req_body = json.dumps(body).encode()
        parsed = urlparse(upstream_url)

        try:
            if parsed.scheme == "https":
                ctx = ssl.create_default_context()
                conn = http.client.HTTPSConnection(parsed.hostname, parsed.port or 443,
                                                    context=ctx, timeout=600)
            else:
                conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80,
                                                   timeout=600)

            conn.request("POST", parsed.path, body=req_body, headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "Accept": "text/event-stream" if body.get("stream") else "application/json",
            })
            resp = conn.getresponse()

            if resp.status != 200:
                error_body = resp.read().decode(errors="replace")
                logger.error("responses passthrough: %s %s", resp.status, error_body[:500])
                self._send_openai_error(resp.status, f"Upstream error: {error_body[:500]}")
                conn.close()
                return

            if body.get("stream"):
                # Read entire SSE response, then forward.
                # Use socket-level timeout to avoid blocking forever.
                conn.sock.settimeout(30)
                chunks = []
                try:
                    while True:
                        data = resp.read(8192)
                        if not data:
                            break
                        chunks.append(data)
                except Exception:
                    pass
                full_body = b"".join(chunks)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(full_body)))
                self.end_headers()
                self.wfile.write(full_body)
                self.wfile.flush()
            else:
                resp_body = resp.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)

            conn.close()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            logger.exception("responses passthrough failed")
            self._send_openai_error(502, f"Upstream error: {type(e).__name__}: {e}")

    def _prepare_model_input(self, body: dict[str, Any]) -> dict[str, Any]:
        """Apply endpoint capability filters before any upstream request."""
        return prepare_model_input_for_endpoint(
            body,
            supports_image_input=self.supports_image_input,
        )

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok", "proxy": "litellm-openai"}).encode())

    def _build_litellm_kwargs(self, body: dict) -> dict[str, Any]:
        """Extract OpenAI Chat Completions params and build litellm.completion kwargs."""
        kwargs: dict[str, Any] = {
            "model": self.litellm_model,
            "messages": body.get("messages", []),
            "stream": body.get("stream", False),
        }

        if self.litellm_api_base:
            kwargs["api_base"] = self.litellm_api_base
        if self.litellm_api_key:
            kwargs["api_key"] = self.litellm_api_key

        for key in (
            "temperature", "top_p", "max_tokens", "max_completion_tokens",
            "stop", "tools", "tool_choice", "response_format",
            "frequency_penalty", "presence_penalty", "seed", "n",
            "reasoning_effort",
        ):
            if key in body and body[key] is not None:
                kwargs[key] = body[key]

        return kwargs

    def _handle_non_streaming(self, response: Any) -> None:
        try:
            resp_json = response.model_dump_json().encode()
        except (AttributeError, TypeError):
            try:
                resp_json = json.dumps(response, default=str).encode()
            except (TypeError, ValueError):
                resp_json = json.dumps({"error": "Failed to serialize response"}).encode()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_json)))
        self.end_headers()
        self.wfile.write(resp_json)

    def _handle_synthetic_openai_streaming(self, response: Any) -> None:
        """Convert a non-streaming response into OpenAI SSE chunk events."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        try:
            resp_dict = response.model_dump() if hasattr(response, "model_dump") else response
            for choice in resp_dict.get("choices", []):
                msg = choice.get("message", {})
                chunk = {
                    "id": resp_dict.get("id", "chatcmpl-proxy"),
                    "object": "chat.completion.chunk",
                    "created": resp_dict.get("created", 0),
                    "model": resp_dict.get("model", ""),
                    "choices": [{
                        "index": choice.get("index", 0),
                        "delta": {
                            "role": msg.get("role", "assistant"),
                            "content": msg.get("content") or "",
                        },
                        "finish_reason": None,
                    }],
                }
                if msg.get("tool_calls"):
                    chunk["choices"][0]["delta"]["tool_calls"] = msg["tool_calls"]
                self.wfile.write(f"data: {json.dumps(chunk, default=str)}\n\n".encode())
                self.wfile.flush()

                finish_chunk = {
                    "id": resp_dict.get("id", "chatcmpl-proxy"),
                    "object": "chat.completion.chunk",
                    "created": resp_dict.get("created", 0),
                    "model": resp_dict.get("model", ""),
                    "choices": [{
                        "index": choice.get("index", 0),
                        "delta": {},
                        "finish_reason": choice.get("finish_reason", "stop"),
                    }],
                }
                if resp_dict.get("usage"):
                    finish_chunk["usage"] = resp_dict["usage"]
                self.wfile.write(f"data: {json.dumps(finish_chunk, default=str)}\n\n".encode())
                self.wfile.flush()

            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            logger.exception("litellm openai proxy: synthetic streaming error")

    def _send_error(self, code: int, message: str) -> None:
        body = json.dumps({"error": {"message": message, "type": "server_error"}}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_openai_error(self, code: int, message: str) -> None:
        body = json.dumps({
            "error": {
                "message": message,
                "type": "server_error",
                "code": str(code),
            },
        }).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        pass


class LiteLLMOpenAIProxy:
    """Local proxy that translates OpenAI Chat Completions API to any litellm backend.

    Parameters
    ----------
    model : str
        litellm model identifier with provider prefix (e.g. ``anthropic/claude-opus-4-6``,
        ``gemini/gemini-pro``, ``ollama/llama3``).
    api_base : str or None
        Target API endpoint URL.
    api_key : str or None
        API key for the target service.
    port : int
        Local port to bind. ``0`` picks a random free port (recommended).
    supports_image_input : bool
        Whether the upstream model endpoint accepts image content blocks.
        Defaults to ``False`` so unknown endpoints fail safe; visual endpoints
        must opt in explicitly.
    """

    def __init__(
        self,
        model: str,
        api_base: str | None = None,
        api_key: str | None = None,
        port: int = 0,
        translate_responses: bool = False,
        supports_image_input: bool = False,
    ) -> None:
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self._port = port
        self._translate_responses = translate_responses
        self.supports_image_input = supports_image_input
        self._server: http.server.HTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def local_url(self) -> str:
        if self._server is None:
            raise RuntimeError("Proxy not started")
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def start(self) -> str:
        """Start the proxy in a daemon thread. Returns the local URL."""
        handler = type("Handler", (_LiteLLMOpenAIProxyHandler,), {
            "litellm_model": self.model,
            "litellm_api_base": self.api_base,
            "litellm_api_key": self.api_key,
            "translate_responses": self._translate_responses,
            "supports_image_input": self.supports_image_input,
        })
        self._server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", self._port), handler,
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True,
        )
        self._thread.start()
        url = self.local_url
        logger.info("litellm openai proxy started: %s → %s (%s)", url, self.api_base, self.model)
        return url

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __del__(self) -> None:
        self.stop()
