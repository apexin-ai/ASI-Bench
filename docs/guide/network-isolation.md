# Tool restrictions and network isolation

`tool_mode` selects agent tools. It does not, by itself, make arbitrary shell
commands or MCP servers offline. Record the network policy separately when
comparing results. Historical restricted runs with open networking retain their
original conditions; rescoring cannot remove information used during generation.

## Shared policy for OS agents

For all built-in OS harnesses, `network_policy` in `--agent-config` accepts:

| Policy | Behavior |
| --- | --- |
| `auto` (default) | `model-only` for restricted OS runs; `inherit` for search/unrestricted or other sandboxes |
| `model-only` | Require the OS sandbox, remove all container IP networking, allow only fixed model API routes through a Unix socket relay |
| `inherit` | Retain legacy Docker/API networking, with **no agent egress isolation guarantee** |

For an isolated run, install task dependencies and prepare images/instances
before starting the agent. Configure explicit API-key authentication, for example:

```bash
asibench run \
  --agent claude_code_cli --sandbox os --tool-mode restricted \
  --agent-config '{"model":"claude-sonnet-4-6","api_key_env":"ANTHROPIC_API_KEY","network_policy":"model-only"}' \
  --instances-dir /path/to/materialized-instances \
  --tasks-dir /path/to/tasks --output-dir /path/to/new-run \
  --fail-on-agent-error
```

The same policy works with `run-score` and task author workflows. For Codex:

```bash
asibench run --agent codex_cli --sandbox os --tool-mode restricted \
  --agent-config '{"model":"gpt-5.1-codex","api_key_env":"OPENAI_API_KEY","network_policy":"model-only"}' \
  --instances-dir /path/to/materialized-instances \
  --tasks-dir /path/to/tasks --output-dir /path/to/new-run --fail-on-agent-error
```

| Harness | Model-only transport and requirements |
| --- | --- |
| Claude Code | Anthropic Messages, including existing host-side protocol translation |
| Codex | OpenAI Responses over HTTP SSE, with WebSockets disabled; explicit `provider` + `api_base` selects a native Responses endpoint; the existing no-provider translation proxy remains host-side |
| pi / OpenCode | Explicit Anthropic, OpenAI Chat or OpenAI Responses endpoint; known key-only providers also work |
| Kimi Code | Anthropic, OpenAI Chat (including native `kimi` wire shape), or Responses; generated container-local config |
| MiMo Code | OpenAI Chat, including its existing host-side translation proxy |
| OpenHands | Anthropic or OpenAI Chat; **SDK/tools must be installed in the task image**; model-only does not mount a host virtualenv |
| Antigravity | Selects model-only for restricted OS, but refuses to start: its custom endpoint transport has not been verified |
| Direct LLM | Host model request, then generated code in a networkless, capability-dropped container; no model relay is exposed to the code |
| Hermes / CodeWhale / generic CLI | No supported OS execution implementation; OS requests are rejected, never silently run on the host |

Unknown protocols (including Google/Vertex) and custom CLI homes fail closed in
model-only. Use explicit endpoint/key configuration for supported protocols.
This uniform policy does not claim every provider protocol is implemented.
Use `api_base` / `api_base_env` and `api_protocol` for a compatible third-party
endpoint. Existing host-side protocol translation proxies are reachable through
the relay without exposing host networking to the container.

The container starts with `--network none`, `--cap-drop ALL` and
`no-new-privileges`. Before starting it, the framework inspects these effective
Docker settings and fails if they do not match. Only loopback networking is
available to the agent and its subprocesses. A per-attempt host relay listens
on a Unix socket mounted read-only into that container; a stdlib bridge gives
the CLI a loopback HTTP endpoint. No Docker socket or general host filesystem
is mounted. The real API key stays in the relay, and login/settings files are
not mounted into the isolated agent.

The relay fixes the upstream endpoint and authentication from operator
configuration and accepts JSON POSTs only for the selected model protocol:
Anthropic `/v1/messages` and `/v1/messages/count_tokens` (optionally exactly
`?beta=true`), OpenAI Chat `/v1/chat/completions`, or Responses `/v1/responses`
and `/v1/responses/compact`. Cross-protocol routes and all other queries are
rejected. Responses stored conversations, remote item references, hosted search,
remote MCP and nested hosted tools are rejected. Only local tool namespaces are
allowed; URL/file-ID attachments and unknown content blocks are rejected.
It rejects CONNECT, arbitrary paths/URLs, ambiguous HTTP framing, redirects,
provider-hosted tools (including search/fetch), remote URL/file sources, and
remote MCP declarations. Ordinary local tool definitions and base64 attachments
remain supported. Unknown API request fields fail closed; future CLI protocol
changes may require a reviewed allowlist update.

The configured upstream is a trusted inference service. This mechanism does
not remove knowledge already in the model or prevent the agent from submitting
additional legal model requests. Docker administrators and the host runner
remain trusted and can change container/network configuration; the policy is
not a defense against host compromise or kernel/container escapes.

## Conflicts and supported scope

- `AI4SCI_DOCKER_NETWORK` (including `host` and named offline bridges) conflicts
  with model-only and is rejected. There is no fallback to an open network.
- Host proxy variables are not forwarded; proxy variables from the image are
  cleared before launching the CLI. The host relay uses its fixed endpoint,
  rather than the agent's HTTP proxy settings.
- OAuth/login credential refresh is not implemented by this relay. Configure
  an API key or explicitly choose `inherit` with its weaker network semantics.
- `runtime.services`, service MCP, and arbitrary additional host mounts are
  rejected under model-only. Offline instance-local stdio MCP shares the same
  networkless container and can be used if all required dependencies are in the
  image. A network-capable MCP service is not silently treated as isolated.
- Model-only requires a local Linux Docker daemon with Unix bind mounts.
  Non-OS sandboxes retain their existing policies. A harness without a supported
  model transport fails before container startup, without open-network fallback.
- Search/unrestricted retain legacy networking by default. An explicit
  `network_policy: inherit` on restricted also permits legacy networking and
  must not be compared as an isolated run.
- Input pulling, image/dependency preparation, host model requests, and Judge
  scoring are outside the agent container's network boundary. Prepare offline
  images in advance; runtime package downloads cannot use the public internet.

## Audit records and validation

Run metadata records requested and selected policy independently of tool mode.
Each OS `agent_output.network_runtime`, also copied to result
`provenance.network`, includes policy version, effective policy, enforcement
status, Docker startup check, route allowlist, an upstream identifier hash,
and forwarded/denied request counts. Rejected startup records
`enforced: false`; a selected policy alone is not evidence of successful startup.
The audit contains no API keys, request bodies or full upstream URLs. IP
requests are blocked by the kernel network namespace; relay counters count
relay requests, not all attempted external IP connections.

Tests use a controlled external canary, first prove it is reachable from a
normal bridge container, and then require that isolated containers cannot
reach it. They also check fixed model requests, local computation, IPv4/IPv6,
Python HTTP/DNS, subprocesses, inherited proxies, concurrency and rejected
relay bypasses. Set `ASIBENCH_NETWORK_DOCKER=1` to run these actual Docker
probes. With `ASIBENCH_NETWORK_AGENT_IMAGE` pointing to a prepared image that
contains Claude and curl, an additional test drives the real CLI's Bash tool
via a local Messages protocol fixture. This tests transport and enforcement,
not the correctness or availability of a live commercial model service.

The multi-harness regression suite is `tests/test_unified_network_policy.py`.
With `ASIBENCH_NETWORK_CODEX_IMAGE`, actual Codex is driven through a local
Responses fixture, both with and without offline stdio MCP. With
`ASIBENCH_NETWORK_NATIVE_IMAGE`, actual pi/OpenCode are driven through Chat
fixtures. These tests require `ASIBENCH_NETWORK_DOCKER=1`; they assert local
output creation and no external canary requests. They are transport/enforcement
tests, not live paid-model certification. Kimi/MiMo/OpenHands have configuration
and protocol coverage; no full live harness certification is claimed for them.

Codex provider configuration follows the [official OpenAI configuration reference](https://developers.openai.com/codex/config-reference/)
(`base_url`, `env_key`, `wire_api`, `supports_websockets` and provider capabilities).
