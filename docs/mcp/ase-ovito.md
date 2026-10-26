# ASE and OVITO MCP adapters

These repository-owned, CPU adapters expose ordinary `tools/call` interfaces
for Claude/Codex. ASE builds and computes structures; OVITO analyzes or renders
the same workspace files. They do not require a GUI application or model download.

| Server | Tools |
|---|---|
| `ase` | `get_info`, `build_bulk`, `read_structure`, `repeat_structure`, `single_point`, `optimize_structure` |
| `ovito` | `get_info`, `inspect_structure`, `identify_structure` (PTM), `compute_rdf`, `dislocation_analysis` (DXA), `render_snapshot` (Tachyon PNG) |

## Install and configure

From a source checkout, build the shared scientific runtime:

```sh
docker build -t asibench-ase-ovito:test scripts/mcp/atomistic
```

This installs ASE 3.29.0, OVITO 3.17.1 and MCP 1.30.0 from a fully pinned,
SHA-256-checked Python lock. Linux/amd64 is the tested platform. The image
also installs native Qt/OpenGL/Vulkan dependencies and Mesa's software driver.
OVITO uses `QT_QPA_PLATFORM=ovitoheadless`; Qt's generic `offscreen` plugin
cannot initialize the Vulkan backend required by this OVITO version.
No GPU, display server or OVITO Pro installation is needed for these tools.

For Codex, build the optional task image, which adds pinned Node/Codex and a
writable agent home:

```sh
docker build -t asibench-ase-ovito-codex:test \
  -f scripts/mcp/atomistic/Dockerfile.codex scripts/mcp/atomistic
```

Set the task's `task_meta.yaml` runtime:

```yaml
runtime:
  image: asibench-ase-ovito-codex:test
```

Use `scripts/mcp/atomistic/mcp.os.json` as the operator MCP configuration;
remove either server entry if the task needs only one. Its commands refer to
files **inside the image**, with explicit `sandbox.deployment=instance`:

```sh
asibench mcp check --config scripts/mcp/atomistic/mcp.os.json
asibench run --agent codex_cli --agent-config '{"model":"your-model"}' \
  --sandbox os --mcp-config scripts/mcp/atomistic/mcp.os.json \
  --instances-dir instances/ --tasks domain.task --output-dir out/
```

Configure model API authentication through the normal agent setup. Built-in
web search is independent of MCP. A Claude deployment needs its own CLI layer
in the scientific image; the optional Dockerfile above installs only Codex.
Use immutable image digests for shared/production deployments.

`asibench mcp catalog` lists the new adapter as **`ase_native`** and **`ovito`**.
`asibench mcp init --servers ase_native,ovito --output mcp.json` produces
the same image commands; its ASE server name is `ase_native`. The checked-in
config uses the shorter server name `ase`, so tools appear as
`mcp__ase__build_bulk`, etc. Server names in config and task requirements must
match. The older catalog key `ase` and E2E id `atomictoolkit` remain the
third-party AtomicToolkit integration with their existing tool contract.

These adapters are maintained directly in `scripts/mcp/atomistic/`, rather
than cloned by `scripts/mcp/e2e/setup.py`. Their version is the ASI-Bench
revision plus Python lock and image identity; no dummy upstream clone or
entry in the third-party E2E manifest is needed. A PyPI framework install
alone does not install these source-tree adapters or scientific dependencies.

For host execution, install the lock into an isolated Python 3.12 environment
and the same native runtime libraries. Copy the Python files together, then
set each config command/args to that environment and source directory,
`ATOMISTIC_MCP_WORKSPACE` to a dedicated workspace, and remove `sandbox`.
Do not point an OS configuration at a host virtualenv.

## Task inputs and outputs

A typical chain is `ase.build_bulk` → `ase.single_point` →
`ovito.identify_structure` / `compute_rdf` → `render_snapshot`.
The structure filenames are shared through `/workspace`. Inputs may use safe
relative subpaths such as `data/structure.extxyz`; outputs use simple filenames.

- Input formats: XYZ/extxyz/CIF, first frame only. Multi-frame trajectory,
  LAMMPS/GROMACS native formats and arbitrary modifiers are outside this adapter.
- Bulk builders: conventional cubic fcc/bcc/diamond, explicit lattice constant
  in angstrom and three repeat counts. Max 4096 atoms, 16 MiB input files.
- Calculators: **EMT** or **Lennard-Jones**, max 512 atoms. EMT supports only its
  built-in element set and is not a general DFT or ML potential. LJ uses
  `epsilon=1 eV`, `sigma=1 A`, `rc=3 A` with ASE's energy shift at the cutoff.
  No implicit VASP, LAMMPS, Quantum ESPRESSO or external executable is launched.
- Optimization: position-only BFGS with fixed cell, at most 200 steps.
  `converged: false` means the requested force threshold was not reached;
  it must not be treated as successful physical convergence.
- PTM returns per-atom structure ids, counts and RMSD. RDF returns total `g(r)`,
  coordination and radius bins; it requires full 3D periodicity and a cutoff
  no larger than half the shortest cell height. DXA returns segment counts,
  line lengths and Burgers vectors for fcc/bcc/diamond assumptions.
- Lengths use angstrom for ASE output. OVITO follows the input's coordinate
  units; it does not infer or convert units. Energies are eV, forces eV/A.
  Snapshot output is a bounded 64–1024 pixel PNG, with size and SHA-256 returned.
- Analysis tools write JSON as well as returning its data; rendering writes PNG.
  Invalid requests raise MCP tool errors. Symlink outputs and workspace escapes
  are rejected. These checks do not create a separate security boundary:
  instance servers share the agent container and workspace.

Keep numerical tolerances, expected structures and scoring in the task's own
contract. Successful tool execution is not certification of an empirical
potential, DXA's validity for a particular material, or a rendered image's
scientific interpretation. OVITO's Python distribution identifies as MIT,
ASE as LGPL-2.1-or-later; Qt and other bundled dependencies retain their own
licenses. OVITO desktop/Pro packaging is separate.

## Verify deployment

```sh
ASIBENCH_ATOMISTIC_DOCKER=1 uv run --frozen pytest -q \
  tests/mcp_e2e/test_atomistic_adapters.py
```

Real Docker tests run all tools without network, check pure JSON-RPC stdout,
compare energies/forces and optimization with a separate ASE process, compare
RDF with an independent periodic pair histogram, check perfect FCC PTM/DXA,
and parse the rendered PNG. Without the flag, scientific tests skip explicitly;
offline configuration/path checks still run. Direct tests do not by themselves
prove an agent used MCP; retain trajectory evidence for agent validation.

Validation on 2026-10-26 also exercised all twelve tools through an actual
CodexCLIAdapter + OSSandbox run (gpt-6.1-sol, Codex 0.162.1), with independently
checked numerical results and files. This validates that deployment chain,
not full generate/run/score CLI orchestration or arbitrary scientific inputs.
For large structures, prefer retaining scientific files rather than copying
entire tool envelopes into shell command arguments; those arguments have
OS size limits.
