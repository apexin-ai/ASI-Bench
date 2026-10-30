# SageMath and SymPy MCP

These repository-owned servers expose native mathematics programs, asynchronous
jobs and artifact transfer. They require **no engine activation or Wolfram
license**. They do not implement Wolfram Language, Mathematica notebooks or a
compatibility layer for the Wolfram MCP.

| Server ID | Installed mathematics | Program syntax |
| --- | --- | --- |
| `sagemath` | Official SageMath 10.7 and its bundled components | Python by default; `syntax: "sage"` enables Sage preparsing |
| `sympy` | SymPy 1.14.0, mpmath 1.3.0, NumPy 2.2.6, SciPy 1.15.3, Matplotlib 3.10.3 | Python |

Both can use their installed backend's full API through scripts. Optional
packages, external datasets, GUI notebooks, distributed/GPU execution and
uninstalled Sage optional components are not implied. Call `get_info` for the
actual backend identity; a missing backend fails startup without a fallback.

## Install before running a task

From the repository root:

```sh
docker build -t asibench-sagemath:test -f scripts/mcp/math/Dockerfile.sagemath scripts/mcp/math
docker build -t asibench-sympy:test -f scripts/mcp/math/Dockerfile.sympy scripts/mcp/math
```

The Sage base image is pinned to an amd64 manifest digest in
`scripts/mcp/math/upstream.lock.json`. Python requirements have exact versions
and SHA-256 hashes. Sage's own Python environment is separate from the MCP
venv; upgrading MCP dependencies does not replace Sage's mathematics packages.
The engine and adapter are installed at **image build time**, not by the agent.
The Sage image adds the system CA bundle and an explicit `SSL_CERT_FILE`; its
upstream image omits roots and Python does not otherwise load the bundle.
Distribute immutable image digests for reproducible tasks.

The framework can add the selected agent CLI using its ordinary task-image
overlay. An optional pinned Codex image is also available:

```sh
docker build -t asibench-sympy-codex:test -f scripts/mcp/math/Dockerfile.codex scripts/mcp/math
docker build -t asibench-sagemath-codex:test --build-arg MATH_BASE_IMAGE=asibench-sagemath:test \
  -f scripts/mcp/math/Dockerfile.codex scripts/mcp/math
```

SageMath is GPL-licensed and SymPy is BSD-licensed; bundled libraries have
their own licenses. Preserve notices and meet applicable source-distribution
requirements when distributing images. Free engine use does not remove
software-license obligations or compute costs.

## Configure a task

Follow [the MCP authoring guide](../guide/authoring-mcp-task.md) for normal task
files, prompts, independent scoring and runtime declarations. There is no
generic `mcp: sagemath` task field.

For **instance stdio**, set `runtime.image: asibench-sagemath:test` (or
`asibench-sympy:test`) in `task_meta.yaml`, and pass the corresponding config:

```sh
asibench mcp catalog
asibench mcp check --config scripts/mcp/math/sagemath.os.json
asibench run --sandbox os --agent codex_cli \
  --agent-config '{"model":"your-model"}' \
  --mcp-config scripts/mcp/math/sagemath.os.json \
  --tasks domain.my_task --instances-dir instances/ --output-dir out/
```

For SymPy, use `sympy.os.json` and its image. No license environment variables
are needed. `mcp check` is a static check; real startup and computation must
also be tested. The OS preflight imports the backend and checks MCP tools
before starting the agent. It is a separate session from the agent's server.
The explicit interpreter/cache settings in the templates are necessary:
the launcher does not inherit image ENV and its initial HOME is on noexec
`/tmp`. Each server gets private executable cache state under
`MATH_STATE_ROOT`, removed on normal shutdown.

For **internal HTTP**, merge `sagemath.service.yaml` or `sympy.service.yaml`
from `scripts/mcp/math/` into `task_meta.yaml` and pass the matching
`*.service.json`. To expose both servers, merge `both.service.yaml` and use
`both.service.json`. The agent image only needs its CLI and ordinary task
dependencies; the two scientific images run as separate services.

Sidecars have a private offline network and no shared agent workspace. Use
`upload_file`, `input_files`, and `read_file` to transfer all data and save
outputs before task teardown. Service names, port 8765 and allowed HTTP hosts
must match the provided templates. No host ports or remote authentication are
configured. HTTP healthchecks perform MCP initialization and backend identity
checks. Memory/CPU are bounded by the framework; increase task budgets for
large mathematics workloads.

For a host installation, use a private Python venv with the relevant lockfile,
set `MATH_PYTHON` to the real backend interpreter, `MATH_MCP_WORKSPACE` to a
private writable directory and `MATH_STATE_ROOT` to a private cache directory.
Run `server.py --backend sagemath` or `--backend sympy`; host MCP configs omit
`sandbox`. Sage must be installed separately; pip-installing the MCP venv
does not install Sage.

## Tools and exact results

Both servers expose the same 11 tools:

| Tools | Purpose |
| --- | --- |
| `get_info` | Actual backend/version, syntax, process limits |
| `evaluate` | Short program, wait up to 60 seconds, return status and result |
| `submit_script` | Submit code or uploaded script, explicitly stage input files |
| `list_jobs`, `get_status`, `wait_job`, `cancel_job` | Asynchronous lifecycle and process-group cancellation |
| `get_result` | State, expression preview and artifact inventory |
| `upload_file`, `list_files`, `read_file` | Bounded input upload and paginated text/binary download |

Each job starts a **fresh process**. Its namespace includes the backend's
top-level names, plus `sp` for SymPy or `sage` for SageMath. The final expression
is returned; a script without a final expression can assign `result`.
Variables do not persist across calls. Upload/stage files to carry state.

Example `evaluate` arguments for Sage:

```json
{"code":"R.<x> = PolynomialRing(QQ)\n((x^2-1).factor(), 1/3)","syntax":"sage"}
```

Example for SymPy:

```json
{"code":"x=symbols('x')\n(integrate(x**4,(x,0,1)), Rational(1,3))"}
```

Python `1/3` is a float; use `Rational(1,3)` in SymPy or `QQ(1)/3` in Sage
Python mode. Sage preparsing handles exact `1/3` and Sage's `^` exponent syntax;
ordinary Python uses `**`. Returned expressions are not converted to floats.
`result.repr.txt`, `result.exact.txt` (SymPy srepr or Sage repr) and
`result.latex.txt` preserve successful exports; repr is not a universal
round-trip serialization format. Previews are limited to 2048 characters and
flag truncation; download full files when necessary. Scripts can explicitly
save native objects, CSV, JSON, images or other artifacts with the backend API.
Do not unpickle untrusted artifacts.

Programs have a 1 MiB source limit, 16 MiB total staged input, at most 64 input
files, two concurrent jobs, and a 1–3600 second deadline including startup.
Each output file is limited to 128 MiB on Linux via prlimit; total disk use
still depends on the container's storage budget. Uploads are at most 16 MiB;
downloads use byte offsets and base64 for lossless binary transfer. Managed
filenames and symlink traversal are rejected by file APIs.

`failed`, `timeout` and `cancelled` are not successful calculations. Inspect
`stderr.log`/`stdout.log` and report failures honestly. For long work, submit
and poll rather than relying on a long synchronous call; a still-running
`evaluate` response contains a job ID for later retrieval. Codex's optional
`tool_timeout_sec` belongs on the server entry, outside `sandbox`. MCP does
not automatically enable built-in web search.

**Native code has server-container privileges.** These file checks are not a
Python/Sage sandbox. Use the OS container boundary and trusted deployment
configuration. Jobs should not modify managed files. Independent mathematical
validation is still required; a completed process alone is not a correct answer.

## Verify

The offline suite checks deployment contracts and file/lifecycle boundaries.
Installing the SymPy lock and pytest also enables real local SymPy tests.
After building both images, run:

```sh
ASIBENCH_MATH_DOCKER=1 pytest -q tests/mcp_e2e/test_math_adapters.py
```

Docker tests check both real engines, finite-field/polynomial/matrix/group
calculations, rational results, PNG/binary transfer, failures and cancellation.
They also run real OSSandbox preflight/config injection for stdio and HTTP;
those protocol-client tests are separate from live model validation and full
task generation/run/scoring.
