# MCP in the OS sandbox

Claude Code and Codex support explicit MCP configurations with `--sandbox os`.
MCP does **not** enable built-in web search: the default remains restricted;
choose `--tool-mode search` to additionally enable built-in search/fetch.
Claude's MCP `ToolSearch` remains available in restricted mode.

## Instance-local stdio

Install the server and its locked dependencies in the task runtime image
(`runtime.dockerfile` or the existing task image mechanism). The framework
never installs third-party servers automatically, mounts host virtualenvs,
rewrites host binary paths, or mounts a Docker socket. Paths in the config
refer to the **container**, not the host:

```json
{
  "mcpServers": {
    "science": {
      "command": "/opt/science/bin/python",
      "args": ["-m", "science_mcp.server"],
      "cwd": "/workspace",
      "env": {"OMP_NUM_THREADS": "1"},
      "sandbox": {
        "deployment": "instance",
        "startup_timeout_seconds": 30
      }
    }
  }
}
```

`command` must be an absolute image path under `/opt/`, `/usr/` or `/bin/`.
`cwd` defaults to `/workspace`; image directories under `/opt/` and `/usr/`
are also supported. Args/env are passed literally, without shell expansion.
Use an image entry point/script if the backend needs shell setup.

The launcher gives each server a fresh HOME/TMPDIR and a minimal environment,
plus explicitly configured `env` values. It does not inherit agent API keys,
proxy settings or config variables. For credentials, use
`"env_from": {"SCIENCE_API_KEY": "MY_SCIENCE_KEY"}` inside `sandbox`;
values are resolved from the runner environment into a private temporary
file, mounted read-only and removed after the run. Missing references fail
before execution. Supply appropriately scoped/short-lived credentials yourself;
the framework does not issue or renew tokens. Keep ordinary `env` for
non-secret settings. No credential values enter command-line flags or provenance.

Scientific backends that JIT-compile shared libraries need an executable cache
outside `/tmp` (the OS sandbox mounts `/tmp` with `noexec`). For the pinned
FEniCSx image, explicitly set `XDG_CACHE_HOME=/home/agent/.cache` in the MCP
server's `env`, plus the image's DOLFINx `PYTHONPATH`, `LD_LIBRARY_PATH`,
`PETSC_DIR`/`PETSC_ARCH` and `SLEPC_DIR`; Dockerfile `ENV` alone is not
inherited by the stdio launcher. Ensure the cache is writable by the runtime
UID. A custom Codex image must also pre-create `/home/agent/.codex` owned by
that UID, so mounting a read-only config does not create a root-owned parent.

**Trust boundary:** instance-local servers share the agent's container, user,
filesystem and network policy. A clean environment is not a separate security
boundary: the server can access files available to that container user, including
agent auth files. Use a sidecar for filesystem/credential/network separation.
There is no independent domain egress whitelist for instance-local servers in
this version. Disabling web search is not a network firewall.

## Internal service (sidecar)

Use the existing `runtime.services` declaration in task metadata:

```yaml
runtime:
  services:
    - name: science
      image: your-registry/science-mcp@sha256:<digest>
      port: 8765
      startup_timeout_seconds: 60
```

The image must implement a Docker `HEALTHCHECK`, listen on `0.0.0.0:8765`,
and expose streamable HTTP MCP. Use an immutable image digest in production.
Reference it from the operator's MCP config:

```json
{
  "mcpServers": {
    "science": {
      "url": "http://science:8765/mcp",
      "sandbox": {"deployment": "service", "service": "science"}
    }
  }
}
```

The hostname and explicit port must match a declared task service. Arbitrary
remote URLs, host networking endpoints, embedded URL credentials, legacy SSE
transport and HTTP authentication headers are rejected in this mode. This
release supports unauthenticated MCP within the private per-instance network;
service credential injection/remote authenticated MCP require further work.
The sidecar has no agent auth mounts or workspace mount, publishes no host
ports, and has only an internal network, so it cannot access external APIs.
The agent keeps its existing model API network policy. Domain-whitelisted
external sidecars, GUI/commercial software and remote MCP are not implemented.
Sidecars require a connectable bridge network. Existing agent API proxy
configurations that force `AI4SCI_DOCKER_NETWORK=host` cannot be combined
with services; use a bridge-reachable direct API endpoint instead.
Service resource limits and cleanup follow the existing `runtime.services`
mechanism (currently at most two services per instance).

## Run and author verification

```sh
asibench mcp check --config sandbox-mcp.json
asibench run --agent codex_cli --agent-config '{"model":"your-model"}' \
  --mcp-config sandbox-mcp.json --sandbox os \
  --instances-dir instances/ --tasks domain.task --output-dir out/
# Add --tool-mode search only if built-in web tools are also allowed.

asibench difficulty-check --task domain.task --sandbox os \
  --agent codex_cli --agent-config '{"model":"your-model"}' \
  --mcp-config sandbox-mcp.json
```

Difficulty-check remains restricted, including when MCP is enabled. The
per-agent `mcp_config` field is also accepted through `--agent-config`.
`run-score` forwards `--mcp-config` to its OS runs. `mcp check` validates the
schema; image binaries and MCP readiness are checked inside the actual run.
Existing host-only configs continue to work with `--sandbox none`; they need
explicit deployment declarations and image preparation before use with OS.
Configs with sandbox declarations are rejected by host execution.

Before the agent starts, a container-local preflight checks `initialize` and
`tools/list` for every configured server. Stdio probe processes and children
are terminated, then the CLI starts a fresh server for the actual session.
Servers must tolerate separate readiness sessions. HTTP readiness is checked
after Docker HEALTHCHECK, through the same private network. A startup failure
returns `ASIBENCH_MCP_STARTUP_FAILED: <name>` and exits nonzero without running
the agent. There is no fallback to host commands, web tools or another server.
Actual tool-call timeouts and post-start reconnect behavior are controlled by
the agent CLI; Codex servers are marked required. The overall Docker timeout
bounds the run, and the existing finally cleanup removes agent/service
containers, temporary networks and staging even on failure.

Results record MCP deployment/transport, a configuration fingerprint,
environment variable names and actual instance/service image identities.
Actual tool names and returned values remain in the ordinary trajectory.
Readiness proves protocol availability, not scientific correctness or actual
agent use: retain L1 reference tests, L2 trajectory verification and task scoring.
