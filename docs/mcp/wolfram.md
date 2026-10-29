# Mathematica / Wolfram MCP

Catalog id and server name: **`wolfram`**. This adapter runs the official
Wolfram `AgentTools` MCP and adds native Wolfram Language jobs, cancellation,
Notebook workflows, and file transfer. Arbitrary WL code is supported; the
adapter does not restrict computation to a few mathematical templates.

**A valid Wolfram Engine / Mathematica license is required.** Engine
computation, a usable Mathematica notebook front end, and online Wolfram
services have separate prerequisites. No real licensed computation or Codex
scientific acceptance has been performed for this integration yet. Protocol,
process lifecycle, image/preflight and missing-license tests are separate
from the opt-in scientific tests below.

## Build and authorization

From the source repository:

```sh
docker build -t asibench-wolfram:test scripts/mcp/wolfram
# Optional preinstalled Codex image (ordinary task-image CLI overlay also works):
docker build -t asibench-wolfram-codex:test \
  -f scripts/mcp/wolfram/Dockerfile.codex scripts/mcp/wolfram
```

`upstream.lock.json` pins the published linux/amd64 image by digest: **Engine
14.3 + AgentTools 2.2.0**, source revision `72691d4`. It deliberately does not
combine the published image with newer source-only AgentTools 2.2.21 (WL
15.0+). Python dependencies have a complete SHA-256 lock. Engine, paclets,
search data and gateway are installed at image-build time. **Codex does not
install Mathematica when it reads MCP config.**

For licensed private installations, the gateway also runs on the host:
install `requirements.lock` into a Python 3.12 venv, provide `WOLFRAMSCRIPT`
as the actual executable, install compatible official AgentTools and start
`server.py`. `start.wls` loads `/opt/AgentTools` when present, otherwise uses
the installation's paclet search path. No user client config is modified.
Host configs must omit `sandbox`; image paths are not host paths.

The operator supplies one of:

- **`WOLFRAM_LICENSE_B64`**: base64-encoded `mathpass` content valid for the
  actual container. The gateway writes private license files in its fresh
  HOME, removes the base64 value from child environments and never prints it.
  A node-locked host license is not assumed to work in a new container.
- **`WOLFRAMSCRIPT_ENTITLEMENTID`**: an applicable Service Credits entitlement,
  resolved at runtime. Activation may need network access. It works only where
  the existing instance network policy allows that access; it cannot activate
  through an offline internal sidecar. Running kernels may consume credits.

Get credentials from your approved secret mechanism; do not paste them in a
JSON/YAML task, command argument, Docker build argument, or Git. The free
Developer license must not be assumed to cover organizational evaluation,
production serving or redistribution; use applicable Wolfram terms/entitlements.
The MIT license for AgentTools does not license the proprietary engine.

Build-time paclet/data directories are seeded into the stdio server's fresh
HOME using read-only file links and writable directories. The gateway does not
reuse a developer's HOME, license or cloud login in the image configurations.
`WOLFRAM_STATE_ROOT=/home/agent/.cache/wolfram` creates a private per-server HOME
and TMPDIR outside the OS sandbox's noexec `/tmp`; it is deleted on shutdown.
This also permits executable compiler caches when the required compiler is
installed. The stdio JSON explicitly supplies it because the launcher does not
inherit Dockerfile ENV. A runtime UID must be able to write this state root and
`/workspace`. Host users should supply a private HOME/state root explicitly.

## Author a task: instance stdio

1. Set `runtime.image: asibench-wolfram:test` in `task_meta.yaml`, or use your
   registry's immutable image digest. A task-specific image can extend it.
2. Declare the required MCP tools and outputs using the normal
   [MCP task authoring guide](../guide/authoring-mcp-task.md). Describe the
   scientific computation and deliverables, not the framework's smoke tests.
3. The operator copies `scripts/mcp/wolfram/mcp.os.json` and adds the applicable
   runtime credential reference to `mcpServers.wolfram.sandbox`, for example:

   ```json
   "env_from": {"WOLFRAM_LICENSE_B64": "ASIBENCH_WOLFRAM_LICENSE_B64"}
   ```

   The right side is the **runner environment variable name**, not its value.
   For entitlement licensing, use a matching mapping for
   `WOLFRAMSCRIPT_ENTITLEMENTID`. Author-owned scientific dependencies/data
   belong in the image or task inputs, not in the credential mapping.
4. Run normally, adding the explicit MCP configuration:

   ```sh
   asibench mcp check --config /path/to/wolfram.os.json
   asibench run --sandbox os --agent codex_cli \
     --agent-config '{"model":"your-model"}' \
     --mcp-config /path/to/wolfram.os.json \
     --instances-dir instances/ --tasks domain.task --output-dir out/
   ```

For long Codex calls, add `"tool_timeout_sec": 240` to the **server entry**
(next to `command`, outside `sandbox`). This is a Codex-only field; omit it for
Claude. For long computations prefer native asynchronous jobs and short polls.
MCP does not automatically enable `--tool-mode search`.

ASI-Bench first performs a separate initialize/tools/list preflight and then
starts the actual CLI server session. A license must accommodate these starts;
no preflight session state is available to the agent. Missing installation or
license fails startup, without substituting another mathematical system.

## Internal HTTP sidecar

Merge `scripts/mcp/wolfram/service.yaml` into task metadata and use
`mcp.service.json` for the operator MCP config. The service image exposes
Streamable HTTP at `http://wolfram:8765/mcp` and has a real MCP healthcheck.
Its gateway maintains **one official upstream per task**, so healthcheck
connections do not each create a new licensed kernel.

Service options now include:

```yaml
runtime:
  services:
    - name: wolfram
      image: asibench-wolfram:test
      port: 8765
      command: [/opt/venv/bin/python, /opt/wolfram-mcp/server.py, --transport, streamable-http]
      startup_timeout_seconds: 300
      memory_mb: 8192
      cpus: 2
      env_from:
        WOLFRAM_LICENSE_B64: ASIBENCH_WOLFRAM_LICENSE_B64
```

`env_from` resolves operator environment names via a temporary 0600 Docker env
file, deleted after container creation even on failure. Values never go into
argv, task metadata, agent environment or result provenance. Docker
administrators can inspect container environments; this does not protect
credentials from the Docker daemon/operator. Newline/NUL values are rejected;
encode license files as base64. There is no arbitrary host license mount.

Services remain on a private **offline** network, without agent auth or
workspace mounts, Docker socket, published ports or remote MCP authentication.
This example therefore requires an offline-valid license; online activation,
external license servers and cloud APIs need a separate deployment design.
Resource bounds are memory 256–65536 MiB and CPU 0.25–16, with defaults 1024 MiB
and 1 CPU; PID limit remains 512. Configure concurrency to fit host resources
and license kernel limits. Native jobs allow up to two concurrent kernels in
addition to the official evaluator's kernels.

## Tools and results

All tools actually exposed by the selected official preset retain their names,
schemas and results. The default `WolframLanguage` preset includes the
Evaluator, Context, ReadNotebook, WriteNotebook, SymbolDefinition, CodeInspector
and TestReport tools. Optional upstream tools depend on its installed version.
Tool results preserve text, images, structured content, embedded resources,
metadata and `isError`; prompts and read-only resource primitives are forwarded.
The gateway declares a workspace MCP root. MCP Apps UI, resource subscriptions,
sampling, elicitation and completion are not advertised as gateway features.
No external AI or UI capability is silently enabled.

For interactive calculations use `WolframLanguageEvaluator(code=...,
timeConstraint=..., session=...)`; reuse the returned opaque session ID for
persistent definitions. Official local evaluation defaults to `Session`, not a
claim of an independent security sandbox. Official context tools may access
online services or optional LLMKit features; preinstalled search data does not
make every context operation offline. For reproducible offline work, use
explicit WL scripts and fixed inputs.

Gateway tools:

| Tool | Contract |
|---|---|
| `wolfram_info` | Official server identity, pinned versions and execution limits; no license value |
| `wolfram_capabilities` | Native job producing real version/system/paclet metadata |
| `wolfram_restart` | Restart official server/kernels after failure; session survival is not guaranteed |
| `wolfram_submit` | Code **or** uploaded script in a fresh native kernel; stages selected inputs; async job ID |
| `wolfram_jobs`, `wolfram_status`, `wolfram_wait` | Job lifecycle; wait at most 60 seconds |
| `wolfram_cancel` | Kill native job process group, including children |
| `wolfram_result` | Status, exact-expression preview and artifact names |
| `wolfram_upload` | No-overwrite task input upload, UTF-8/base64, at most 16 MiB |
| `wolfram_files`, `wolfram_read` | Paginated file listing and byte-range download with chunk SHA-256 |
| `wolfram_notebook` | Frontend evaluate/save or render a `.nb`; requires a usable licensed front end |

Native jobs and official interactive sessions are **independent**. Native jobs
run in `wolfram-jobs/<job_id>/` and do not restore session definitions. Job
records belong to this gateway lifetime; files remain accessible until the
workspace/container is removed. A restart of the *official* upstream leaves
native jobs running.

Native job inputs are at most 1 MiB of code and 16 MiB of staged files. The
kernel has an internal computation limit (1–3600 seconds), plus an external
startup/evaluation/export deadline of that limit + 30 seconds. Individual
output files are limited to 128 MiB; the container/operator must enforce
aggregate disk/resource limits for untrusted workloads. Diagnostics go to
`stdout.log` / `stderr.log`; gateway stdout stays MCP-only.

`result.inputform.txt` and `result.wxf` preserve symbols, exact rationals,
complex values and high precision. JSON contains only a bounded preview and
status, so values are not silently converted to Python/JSON floats. In native
execution `$Failed`/Failure, `$Aborted`, timeouts and nonzero process exits are
not success. Messages may coexist with a valid result and remain in logs;
an unevaluated expression is preserved, **not certified as a solved problem**.
Do not hide messages or truncated output when assessing scientific validity.

Example agent workflow:

```text
wolfram_upload(name="input.csv", data="1,2\n3,4\n")
wolfram_submit(code='data=Import["input.csv","CSV"]; Export["copy.csv",data,"CSV"]; 1/3',
               input_files=["input.csv"])
wolfram_wait(job_id=..., wait_seconds=30)
wolfram_result(job_id=...)
wolfram_read(job_id=..., name="result.inputform.txt")
wolfram_read(job_id=..., name="copy.csv", encoding="base64")
```

Sidecar file paths are not agent-local paths. Download and save all required
outputs before task teardown; chunks use byte offsets, so base64 is safest for
lossless transfer across UTF-8 boundaries or binary files.

## Notebook and full-language scope

The official Notebook tools provide file workflows. `wolfram_notebook` adds
real `UsingFrontEnd` / NotebookEvaluate+NotebookSave or Rasterize operations;
absence of a usable front end must fail rather than return placeholder results.
A plain Engine image contains Player components and is **not a claim that the
full Mathematica authoring front end is available/licensed**. A private image
with the appropriate installed frontend, shared libraries and display support
is required for those operations. Set `WOLFRAMSCRIPT` to that image's executable
and install the matching pinned official paclet. Real frontend acceptance is
opt-in and has not been executed for this integration.

Arbitrary native WL can access all functions/packages that the installed,
licensed product provides, including imports/exports, plotting, PDE solving
and user packages. GPU, cluster kernels, GUI Dynamic/Manipulate interaction,
MCP Apps display and online services are not bundled by this headless adapter.
A task needing those features must declare and validate its additional runtime.
Native scripts have container privileges, including process/file/network
operations; file-API path validation is **not a Wolfram code sandbox**.
Instance-local MCP shares the agent's user and files; use a sidecar when its
filesystem/auth separation is required.

## Verification

```sh
# Protocol/lifecycle tests: use a venv with the locked MCP SDK + framework test deps.
python -m pytest -q tests/mcp_e2e/test_wolfram_adapter.py \
  tests/test_service_runtime_credentials.py

# Actual built-image + fresh-HOME preflight and missing-license behavior.
# Upstream fixture is protocol-only; this is not mathematical acceptance.
ASIBENCH_WOLFRAM_IMAGE_TEST=1 python -m pytest -q \
  tests/mcp_e2e/test_wolfram_adapter.py

# Scientific acceptance, AFTER supplying a valid license in the environment.
# WOLFRAM_LICENSE_B64 must work offline for the HTTP test.
ASIBENCH_WOLFRAM_DOCKER=1 python -m pytest -q \
  tests/mcp_e2e/test_wolfram_adapter.py -k licensed
```

Licensed acceptance includes exact integration, native rational preservation,
CSV/PNG generation, real numerical/symbolic calculations, high precision,
interactive state, and framework-managed HTTP sidecar input/output transfer.
With the explicit opt-in, invalid licenses fail the test. Separately use the
normal `asibench run` command above with a real Codex model, inspect the saved
MCP trajectory, independently check submitted artifacts, and score the task.
A protocol fixture or model-free container test cannot replace that evidence.
