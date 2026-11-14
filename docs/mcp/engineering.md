# Chip design and engineering MCP

Six repository-owned adapters expose native open-source engine jobs through the
same ten MCP tools. They run in an instance or internal HTTP sidecar. They do
not need a paid engine license or third-party MCP installation. Review each
engine's redistribution license before distributing images; PDKs, model
libraries, and third-party materials have separate terms.

| Catalog/server ID | Native input | Engine packaged in tested image | Scope |
| --- | --- | --- | --- |
| `ngspice_native` | SPICE netlist/control blocks | ngspice 44.2 | Batch circuit simulation and raw numerical exports |
| `iverilog_native` | Verilog/SystemVerilog | Icarus 12.0 | Compile and run HDL testbenches using `vvp` |
| `verilator_native` | Verilog/SystemVerilog | Verilator 5.032 | Generate/build/run a native testbench with `--binary --timing` |
| `opensta_native` | Tcl plus staged Liberty/netlist/SDC | OpenSTA distro `0~20191111gitc018cb2` | Native static timing scripts; bring compatible libraries |
| `calculix_native` | INP plus staged include files | CalculiX CCX 2.17 | Native finite-element input decks and solver outputs |
| `openmodelica_native` | MOS scripting plus staged `.mo` files | OpenModelica 1.27.1 | Native model compilation/simulation and result export |

These IDs are separate from the older third-party catalog templates
`ngspice`, `icarus_verilog`, `verilator`, `opensta`, `calculix`, and
`openmodelica`. Existing template commands are preserved. The new adapters
call native tools directly; they do not wrap or redistribute those MCP
repositories. OpenROAD remains a separate integration. No tool here supplies
a complete chip design/signoff flow, a PDK, or a generic prevalidated CAE model.

## Build before running tasks

From the repository root:

```bash
docker build -t asibench-engineering-eda:20261114 \
  -f scripts/mcp/engineering/Dockerfile.eda scripts/mcp/engineering
docker build -t asibench-engineering-cae:20261114 \
  -f scripts/mcp/engineering/Dockerfile.cae scripts/mcp/engineering
```

The EDA image contains the first four engines; CAE contains the last two.
Images target Linux amd64 and run as UID/GID 1000. Base images are pinned by
digest; native distro packages are pinned explicitly and the full installed
package inventory is `/opt/engine-packages.txt`. MCP Python dependencies use
SHA-256 locks. System transitive packages still come from distro repositories;
archive the built image digest for a fixed benchmark runtime. Images are local
builds, not published registry images. Python/engine installs happen at build
time, never during agent startup. Missing engines fail server startup.

## Instance stdio

For ngspice, use this task runtime Dockerfile:

```dockerfile
FROM asibench-engineering-eda:20261114
```

Reference it via `runtime.dockerfile` in task metadata. For CalculiX or
OpenModelica, use the CAE image. The framework adds the selected agent CLI.
Use the backend's supplied MCP config:

```bash
asibench run ... --agent codex_cli --sandbox os \
  --mcp-config scripts/mcp/engineering/ngspice.os.json
```

Replace `ngspice` with the backend from the table (use `iverilog` for Icarus).
Each file registers the corresponding `_native` server. Multiple instance
servers can be combined into one `mcpServers` object if the task image
contains all required engines. Configs do not install engines.

Instance stdio works under restricted model-only networking; MCP does not
require web search. Compilation and simulation run in writable `/workspace`
job directories, not the isolated HOME or noexec `/tmp`.

## Internal HTTP sidecar

Merge `<backend>.service.yaml` into the task's `runtime.services` and pass
`<backend>.service.json` via `--mcp-config`. For example, ngspice's service is
`ngspice-native`, URL `http://ngspice-native:8765/mcp`. The service declaration
includes the backend command and image. Do not use the image's default CMD
for a different backend. The image HEALTHCHECK initializes MCP and checks
`get_info` and required tools. Services publish no host ports.

Current restricted model-only networking rejects runtime sidecars; explicitly
select `--agent-config '{"model":"your-model","network_policy":"inherit"}'`
when using service deployment, following [network isolation](../guide/network-isolation.md).
The service network remains internal/offline. This network policy is distinct
from enabling web search. Services have no agent workspace mount: transfer
inputs and outputs through MCP, and save outputs before task teardown removes
the service. Choose instance deployment when the task requires model-only
networking.

## MCP tools and native contract

| Tool | Purpose |
| --- | --- |
| `get_info` | Real native engine version and limits |
| `upload_file` | Upload UTF-8/base64 input, no overwrite, up to 16 MiB |
| `submit_script` | Submit inline `code` or uploaded `script_name`, explicit `input_files`, timeout and HDL `top` |
| `list_jobs` / `get_status` | Query this server lifetime's job records |
| `wait_job` | Poll up to 60 seconds without blocking other HTTP clients |
| `cancel_job` | Terminate the native process group |
| `get_result` | State, native stage exit codes, and artifact inventory |
| `list_files` / `read_file` | Paginated paths and bounded text/base64 byte downloads |

Example ngspice call sequence:

1. `submit_script(code="Voltage divider\nV1 in 0 10\nR1 in out 1000\nR2 out 0 1000\n.control\nop\nwrdata divider.txt v(out)\n.endc\n.end\n")`
2. `wait_job(job_id=..., wait_seconds=30)` and require `state="completed"`.
3. `get_result(job_id=...)`; inspect native stage results.
4. `read_file(job_id=..., name="divider.txt")`; save the returned data in the
   task's declared outputs. Here the reference voltage is 5 V.

For uploaded Liberty, HDL sources, `.include` decks or `.mo` files, list their
workspace-relative names in `input_files`. HDL uploads ending in `.v`/`.sv`
are compiled alongside the inline source; `top` defaults to `top`. Upload
header/library files with their correct relative paths. The primary inline
file is named `input.cir`, `input.sv`, `input.tcl`, `input.inp` or `input.mos`.
Do not stage a file under one of these reserved names. Native jobs are fresh
processes with no implicit persistent design/model state. OpenModelica's
standard library and optional packages must be prepared in the image if the
model requires them; no automatic package downloads are performed.

States are `running`, `completed`, `failed`, `timeout`, and `cancelled`.
At most two jobs run concurrently; timeouts are 1–3600 seconds, covering
compilation and execution together. `stage-N.stdout.log`/`stderr.log` retain
native output; `stdout.log` is a bounded summary and `result.json` holds stage
exit codes. Large output reads must use pagination; base64 preserves bytes.
Jobs belong to the current server process and cannot be resumed after restart.

Nonzero native exit codes fail jobs. The launcher additionally checks common
zero-exit error reports for ngspice/OpenSTA/OpenModelica and CCX completion.
`completed` means the native command finished, **not** scientific convergence,
test coverage, or timing/design validity. Native scripts can catch/suppress
errors themselves. The task scorer must independently check expected artifacts,
units, solver convergence, constraints, HDL assertions and numerical references.

Native Tcl, SPICE controls, HDL/DPI, INP and MOS programs execute with server
container privileges. Path validation on transfer APIs is not a code sandbox.
Use per-instance OS isolation, resource limits, and no host credentials or Docker
socket. Instance servers share the agent's container boundary; sidecars provide
a separate filesystem boundary. The default output file-size limit is 128 MiB
per file on Linux; it is not a total disk quota. Bound task workloads and preserve
needed results before cleanup.

## Verification

```bash
uv run --frozen pytest -q tests/mcp_e2e/test_engineering_adapters.py
ASIBENCH_ENGINEERING_DOCKER=1 uv run --frozen pytest -q \
  tests/mcp_e2e/test_engineering_adapters.py
```

Offline tests cover catalog/config staging, transfer paths, overwrite rejection,
timeout and cancellation. Docker tests call real native engines in a
network-disabled container with a fresh HOME and compare against independent
closed-form references: 5 V divider, all 16 two-bit sums, 0.1 ns buffer delay,
0.004 length-unit elastic displacement, and `exp(-1)` decay. Invalid native
inputs must fail without killing the server. Framework tests run both stdio and
HTTP via actual OSSandbox preflight/config injection and verify cleanup and
sidecar workspace separation, using a protocol client stand-in for the agent.
They are not live model/Codex CLI acceptance and do not certify all engine
features, real PDKs, arbitrary materials or general scientific accuracy.
