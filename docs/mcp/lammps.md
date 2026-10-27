# LAMMPS MCP

This repository-owned adapter exposes the installed LAMMPS engine through
18 MCP tools. Native input scripts provide access to the engine's compiled
styles and packages; convenience tools cover common periodic atomic systems.
It supports instance-local stdio and internal HTTP sidecars with the same tools.

## Tools and scope

| Purpose | Tools |
|---|---|
| Inspect engine and potentials | `get_info`, `get_engine_help`, `list_potentials` |
| Prepare and execute | `write_input`, `submit_script` |
| Manage jobs | `list_jobs`, `get_status`, `wait_job`, `cancel_job` |
| Retrieve results | `get_result`, `list_files`, `read_output`, `read_log`, `read_trajectory` |
| Restart | `restart` (`info`, `data`, `dump`, `continue`) |
| Common calculations | `single_point`, `minimize`, `run_md` |

Native scripts can build structures, read data, select force fields, compute
properties, relax cells, integrate dynamics, and write trajectories/restarts.
They are not restricted to the three convenience models. Use `get_info` for
actual installed packages, `get_engine_help` for compiled styles, and
`list_potentials` (prefix/offset/limit) for bundled files and SHA-256 hashes.
A potential must match its material, atom types and pair style. Custom inputs,
potentials and include files can be uploaded or staged from the workspace.

The pinned CPU distribution is LAMMPS **22 Jul 2025 Update 4**, Python package
`2025.7.22.4.0`, with MPICH **4.3.2**, ASE **3.29.0**, MCP **1.30.0**.
It includes MANYBODY, KSPACE, MOLECULE, MEAM, REAXFF and other packages reported
by the engine. Package availability does not validate every force field or
simulation workflow. GPU, multi-node MPI, Slurm and optional packages absent
from this build are not provided. MPI is explicit, 1–4 ranks, with one OpenMP
thread each; numerical checks cover serial and two ranks on Linux/amd64.

## Install once, then run a task

Build from a source worktree:

```sh
docker build -t asibench-lammps:test scripts/mcp/lammps
# Optional Codex task image with pinned CLI and writable agent home:
docker build -t asibench-lammps-codex:test \
  -f scripts/mcp/lammps/Dockerfile.codex scripts/mcp/lammps
```

The Dockerfile installs the full SHA-256-locked Python dependencies, LAMMPS,
MPI and native runtime libraries at build time. `/opt/venv/lib` is explicitly
on the dynamic library path, so both `lmp` and the Python shared library work.
It checks actual engine startup, not just imports. Use immutable image digests
for shared deployments. The framework package alone does not install this
source-tree adapter or its scientific runtime.

For an instance-local Codex task, set `task_meta.yaml`:

```yaml
runtime:
  image: asibench-lammps-codex:test
```

Use the checked-in operator config and normal run command:

```sh
asibench mcp catalog
asibench mcp check --config scripts/mcp/lammps/mcp.os.json
asibench run --agent codex_cli --agent-config '{"model":"your-model"}' \
  --sandbox os --mcp-config scripts/mcp/lammps/mcp.os.json \
  --instances-dir instances/ --tasks domain.task --output-dir out/
```

`asibench mcp init --servers lammps --output mcp.json` generates the instance
config too. Commands refer to the server **inside the image** and explicitly
select `sandbox.deployment: instance`. The framework initializes and lists MCP
tools during preflight, then injects the config into the agent's isolated home.
The agent calls tools; it does not install LAMMPS. Configure model authentication
through the normal agent setup; web search is independent of MCP.

Declare the server name `lammps` and required tools in the task MCP contract as
explained in [the task-author guide](../guide/authoring-mcp-task.md). Have the
agent retain required scientific artifacts in the task's declared output paths.
For Claude, add its CLI to the scientific image using the normal agent overlay.

## Execution and data contract

Every execution returns `job_id` immediately. A typical native sequence is:

1. Upload required files with `write_input`, or use existing relative workspace
   paths. Uploads accept UTF-8 or base64, do not overwrite, and reject symlinks.
2. Call `submit_script(script=..., input_files=[...])`. Files are copied into a
   fresh `lammps-runs/<job_id>/` directory, retaining their relative names.
   For example, `input_files=["data/system.data"]` works with
   `read_data data/system.data` in the script. Fixed server filenames such as
   `in.lammps` and `log.lammps` are reserved.
3. Call `wait_job` (0–60 seconds per call) or `get_status`, then `get_result`.
4. Use `list_files`, `read_log`, `read_trajectory` or paginated `read_output`.
   Instance jobs' files are directly in `/workspace/lammps-runs/<job_id>/`;
   copy declared outputs to the task's required paths. HTTP clients retrieve
   file bytes over MCP and save them in the agent workspace.

States are `running`, `completed`, `failed`, `timeout`, `cancelled`,
`output_limit`, or `interrupted`. `completed` means process exit 0, not physical
convergence. `read_log` / `get_result` report the real minimizer stop reason;
force/energy tolerance counts as convergence, iteration exhaustion does not.
Native units and thermo normalization follow the script. The adapter never
labels arbitrary native output as eV. Logs parse standard numeric tables headed
by `Step`; YAML/custom logging can instead be retrieved with `read_output`.

`read_output` offsets and limits count bytes; use base64 for lossless binary
or non-UTF-8 transfer. It returns `next_offset` and `eof`. `read_trajectory`
reads custom text dumps, retaining atom columns and box flags/bounds without
assuming units or converting scaled/triclinic coordinates. Other dump formats
remain accessible as files. Input parsing is bounded to 16 MiB. Avoid huge tool
responses: retrieve selected frames/atoms and save native artifacts.

`restart` copies a binary restart into a fresh job. `action=continue` prepends
`read_restart` to explicit `continuation_script`; supply additional potentials
through `input_files`. Some styles do not persist coefficients/fixes, which
must be restored in those commands. Restart files are engine/build dependent.
Native `read_restart` also accepts uploaded base64 restart files.

### Convenience calculations

`single_point`, `minimize` and `run_md` accept `input_name`, `model`,
`timeout_seconds` and `mpi_ranks`; they return a job handle for the same polling
and result tools. Input is the first XYZ/extxyz/CIF frame or atomic-style
LAMMPS `.data` with inferable element masses, at most 4096 atoms. These helpers
require a fully periodic nondegenerate 3D cell; use native scripts for other
boundaries, charged/molecular atom styles or custom forces.

| Model | Definition |
|---|---|
| `lj` | `lj/cut`, common epsilon/sigma for all types, energy shifted at cutoff. Defaults 1 eV, 1 A, 3 A; parameters explicit. |
| `cu_eam` | Cu only, `eam`, bundled `Cu_u3.eam` |
| `si_tersoff` | Si only, `tersoff`, bundled `Si.tersoff` |

The two file-based models' hashes and source headers are recorded in
`potentials.json`. Convenience outputs use `units metal`: A, eV, eV/A, ps, K,
bar; thermo quantities are extensive (`thermo_modify norm no`). Final dumps
contain sorted IDs, types, positions and forces at high numeric precision.
`workflow.json` records the type-to-element mapping, potential hash and model.
Outputs include thermo log, trajectory, final data and binary restart.

`minimize` uses fixed-cell conjugate gradient, zero energy tolerance and
`min_modify norm max`; `fmax` targets the maximum per-atom force norm. It
reports the actual stop criterion even when `steps` is exhausted.
`run_md` supports NVE, NVT and NPT, up to 100000 steps. It generates fresh
seeded Maxwell velocities; `timestep` is in ps. NVT uses damping 100 timesteps;
NPT uses isotropic 0 bar pressure with damping 1000 timesteps and requires an
orthorhombic cell. Use native scripts for other controls or true continuation.

## Internal HTTP sidecar

Use a normal agent image and this task runtime service:

```yaml
runtime:
  image: your-agent-image
  services:
    - name: lammps
      image: asibench-lammps:test
      port: 8765
      command: [/opt/venv/bin/python, /opt/lammps-mcp/server.py, --transport, streamable-http]
      startup_timeout_seconds: 120
```

Run with `--sandbox os --mcp-config scripts/mcp/lammps/mcp.service.json`.
The config uses `http://lammps:8765/mcp` and explicit service deployment.
The image's Docker healthcheck performs MCP initialize/tools/list. The framework
creates an internal per-instance network, checks health, performs preflight,
and cleans up the service. No host ports or Docker socket are needed.

The service **does not share the agent workspace**. Upload via `write_input`,
then fetch outputs via `read_output` (base64 for restart/binary files). Job files
are ephemeral when the sidecar is removed, so fetch required outputs before
completion. This endpoint has no authentication and is for the framework's
private task network, not a publicly exposed server.

## Resource and trust boundaries

The supervisor permits two concurrent jobs, 1–600 seconds per job, 1–4 MPI
ranks, 16 MiB total staged input, and a monitored 128 MiB / 2048-entry output
budget. OS file-size limits also apply to child processes. Timeout/cancel and
server shutdown terminate the job process group. Recorded unfinished jobs on
server restart become `interrupted`; they are not certified as successful and
persisted PIDs are never used to kill processes.

Native LAMMPS scripts can invoke shell, include files, plugins and other
capabilities compiled into the engine. File API path checks are not a sandbox
for script execution. Jobs share their server container UID, filesystem and
network. Processes deliberately escaping their process group, shell-generated
output outside the run directory and arbitrary native allocation require the
outer Docker boundary and its CPU/memory/PID/network/mount limits. Instance
servers also share the agent trust boundary; use a sidecar to separate them
from agent credentials. This adapter is not a multi-tenant remote job service.

LAMMPS's normal distribution is GPLv2; MPICH, ASE and other dependencies retain
their own licenses. Keep original potential headers and distribution notices.
When distributing the image, provide corresponding LAMMPS source for the pinned
release and review the included potential files' original terms. The adapter
invokes `lmp` as a subprocess and does not vendor third-party MCP code.

## Verification

```sh
ASIBENCH_LAMMPS_DOCKER=1 uv run --frozen pytest -q \
  tests/mcp_e2e/test_lammps_adapter.py
```

Real Docker tests exercise all 18 tools, analytic LJ and independent ASE
energy/forces/optimization references, independently constructed native-library
EAM/Tersoff references, NVE drift, NVT/NPT, two-rank MPI, restart actions,
negative inputs, failure/timeout/cancellation/output limits, clean JSON-RPC
stdout, and HTTP upload/compute/binary download without a shared workspace.
Offline tests cover catalog/config, thermo parsing and interrupted-job recovery.
Without the opt-in flag, Docker tests explicitly skip. Protocol success alone
is not scientific validation, and direct tests do not prove agent tool use.
