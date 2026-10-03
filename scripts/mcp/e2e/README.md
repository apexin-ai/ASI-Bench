# MCP E2E: install and smoke test

Installs catalog MCP servers at pinned upstream revisions and checks real
`tools/call` results against independently computed references, with no agent
involved. Upstream code is cloned, never vendored; upstream licenses apply.

| Level | Meaning |
|---|---|
| L0 | `initialize` + `tools/list` work and match `manifest.json` |
| L1 | real `tools/call` results match a reference computed by the smoke script, outside the server process |
| L2 | an agent in `asibench run` calls the tool and its answer comes from the tool — see [`examples/mcp-e2e-tasks`](../../../examples/mcp-e2e-tasks/README.md) |

## Files

| File | Purpose |
|---|---|
| `manifest.json` | per server: repository, 40-char revision, Python, install mode, launch command/env (`{checkout}` placeholder), expected tools, optional shared `checkout` |
| `setup.py` | clone, pin, build `<root>/<checkout or id>/.venv`, write `<root>/<id>.mcp.json` (stdlib only) |
| `smoke.py` | L0/L1 smoke CLI: `smoke.py <id> --config …`, JSON report |
| `e2e_smoke/` | `runner.py` (the shared run and generic checks), `client.py` (stdio client recording non-JSON stdout), `helpers.py`, and `servers/<id>.py` per server (references + server-specific checks, declared as `SMOKE`) |
| `verify_run.py` | L2 verifier CLI for agent runs; implementation in `e2e_verify/` (spec, extractors, values, evidence, checks) |
| `locks/` | committed conda `@EXPLICIT` locks (psi4, quantum_espresso) |

Install modes:

| Mode | Use when | Manifest fields |
|---|---|---|
| `uv-sync-frozen` | upstream has a `uv.lock` | optional `uv_sync_args` |
| `uv-pip-pinned` | no lockfile | `requirements` (exact `==` pins), `exclude_newer` |
| `conda-explicit` | conda-only packages | `conda` (channel, exact specs); lock per platform in `locks/`, regenerate with `setup.py <id> --lock` |

Servers with prebuilt native code also declare `host_requirements`; `setup.py`
checks them before cloning and never installs system packages.

Ids backed by the same upstream repository — one repository exposing several
tool faces, such as the ToolUniverse SMCP servers — share one checkout and one
`.venv` through the optional `checkout` key: `setup.py` installs into
`<root>/<checkout>` instead of `<root>/<id>`, and still writes one
`<root>/<id>.mcp.json` per id whose MCP server name is the id, so tool names
(`mcp__<id>__<tool>`) do not change. Every id of a group must declare the same
repository, revision, python, install mode and install fields, and the group
name must not be another manifest id unless that id joins the group with the
same key. Running `setup.py` for the second id of a group installs into the
shared directory again: a fast no-op for `uv-sync-frozen`, a full rebuild of
`.venv` for the other modes.

The manifest is strict: an entry holds only the common keys plus its install
mode's fields; a misspelt key or another mode's field is an error. A new mode is
one `Installer` subclass registered in `INSTALLERS` (fields + validators,
`install`, optional `lock`); a new host check is one probe in `HOST_PROBES`.

## Run

Linux, as an unprivileged E2E user:

```sh
cd ~/ASI-Bench
python3 scripts/mcp/e2e/setup.py <id> --root ~/mcp
~/mcp/<id>/.venv/bin/python scripts/mcp/e2e/smoke.py <id> \
  --config ~/mcp/<id>.mcp.json --report ~/mcp/<id>-smoke-report.json
uv run asibench mcp check --config ~/mcp/<id>.mcp.json
```

For a server with a shared `checkout` the interpreter is
`~/mcp/<checkout>/.venv/bin/python` (`arxiv` → `~/mcp/tooluniverse`); the
config path stays `~/mcp/<id>.mcp.json`.

Run the smoke with the server's own venv: the reference needs the same
scientific library. The server is started from a temporary cwd/HOME/TMPDIR
with a minimal environment and no credentials. FAIL means a wrong result and
exits non-zero; WARN means an upstream defect that does not make correct use
wrong. Every server gets the shared checks of `e2e_smoke/runner.py` (pinned
revision, config equals what `setup.py` renders, handshake and `tools/list`,
unknown tool is an error, server alive, stdout is pure JSON-RPC, cwd
untouched); its own checks are in the `e2e_smoke/servers/<id>.py` docstring
and messages.

A new server's smoke is one `e2e_smoke/servers/<id>.py` defining
`SMOKE = Smoke(server=…, run_l1=…)`, plus optional hooks (`prepare` for extra
L0 checks or reference objects, `after` for short-lived probe servers via
`session.spawn`, `extra_env`, `pass_proxies`, `expected_cwd_files`,
`add_arguments`, `report_fields`). Reuse `Caller.json`, `check_rejected`
(isError PASS / in-band WARN / accepted FAIL) and `helpers.py` instead of
re-implementing them.

## Servers

| id | Upstream | Install | Needs | Last smoke (PASS / WARN / FAIL) |
|---|---|---|---|---|
| `pyscf` | `lixin19/mcp2pyscf` | uv-sync-frozen, Py 3.13 | — | 21 / 8 / 0, amd64, 2026-10-02 (aarch64 PASS 2026-09-29) |
| `arxiv` | `mims-harvard/ToolUniverse` (SMCP) | uv-sync-frozen, Py 3.12, checkout `tooluniverse` | network to arxiv.org | 16 / 8 / 0, amd64, 2026-10-02 (pre-shared-checkout layout; aarch64 PASS 2026-09-30) |
| `jsbsim` | `flyintothesky/jsbsim-mcp` | uv-pip-pinned, Py 3.12 | — | 17 / 17 / 0, amd64, 2026-10-02 (aarch64 PASS 2026-10-02, old layout) |
| `s4` | `prof-davifr/mcp-s4-rcwa` | uv-pip-pinned, Py 3.12 | x86-64 (AVX2/FMA/BMI2), `libblas3 liblapack3` | 41 / 7 / 0, amd64, 2026-10-02 |
| `psi4` | `Keith9922/chemaster` (`calc_psi4`) | conda-explicit | `micromamba` on `PATH` | 14 / 11 / 0, amd64, 2026-10-02 (aarch64 PASS 2026-10-02, old layout) |
| `rdkit` | `tandemai-inc/rdkit-mcp-server` (catalog `rdkit_tandem`) | uv-pip-pinned, Py 3.12 | — | 125 / 28 / 0, amd64, 2026-10-03 (aarch64 identical 2026-10-02) |
| `build123d` | `pzfreo/build123d-mcp` | uv-sync-frozen, Py 3.12 | — | 89 / 15 / 0, amd64 and aarch64, 2026-10-03 |
| `quantum_espresso` | `frimpsjoek/qe-mcp` | conda-explicit (`qe=7.5`) | `micromamba` on `PATH` | L0 only: 16 / 0 / 0, amd64 and aarch64, 2026-10-03 |

`quantum_espresso` is at L0: the environment checks and `qe_status` run, the
numerical references for the other 18 tools are not written yet.

Install the prerequisites before running `setup.py`:

```sh
sudo apt-get install libblas3 liblapack3            # s4, admin, once
mkdir -p ~/.local/bin && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
  | tar -xj -C ~/.local bin/micromamba               # psi4, quantum_espresso
```

## Notes for task authors

These are the things that affect fake tasks and agent runs. The full list of
known defects is in each smoke script.

**pyscf**

- PySCF, geomeTRIC and upstream `print` write to stdout, i.e. into the stdio
  transport. Claude Code and Codex tolerate this; stricter clients may not.
- The server embeds molecules with unseeded RDKit. Only rigid molecules give
  reproducible geometries (energies within ~1e-7 Ha); flexible ones such as
  ethanol vary by ~1e-3 Ha.
- `generate_pyscf_geom_input` → `pyscf_rhf_energy` sometimes fails with
  `PointGroupSymmetryError` for high-symmetry molecules such as benzene.
- `run_bond_stretch_calculation_mcp` ignores `basis` (always STO-3G).
  `visualize_molecule_3d_mcp` writes HTML into the server cwd and never
  returns it, so it cannot be used in L2.

**arxiv**

- The launch env turns off ToolUniverse's result cache. With the cache on,
  repeated calls never reach arXiv and results leak between runs.
- Never start the server with a `.tooluniverse/` directory in its cwd
  (e.g. cwd = `$HOME`): it then exposes all 2718 tools and ignores
  `--include-tools`.
- Results are live data. Tasks must use closed date windows, put OR groups in
  parentheses (otherwise the date clause binds only to the last term), and
  must not score abstracts.

**jsbsim**

- The launch env sets `JBSIM_ROOT` (upstream's spelling) and `JSBSIM_DEBUG=0`.
  Without the latter, each `create_session` prints ~1100 lines to stdout.
- Always pass all seven initial conditions, then the properties, then step.
  Without initial conditions the state becomes NaN; keys left out are reset
  to 0.
- The c172x engine starts with `propulsion/set-running = -1`.
- Read the state with `get_property`: 20 of the 33 `get_telemetry` fields read
  properties that do not exist and always return 0.
- Do not use `trim` (it does not trim) or `execute_script` (it can crash the
  server). Idle sessions are closed after 300 s.

**s4**

- The repository ships a prebuilt `libS4.so`, hence the host requirements
  above. On other platforms you can only use a locally built library in a
  scratch checkout; `setup.py` refuses that checkout and the smoke run is
  marked as not representative.
- List layers substrate first and incidence medium last.
  `incidence_layer`/`substrate_layer` must name exactly those two layers.
  Anything else gives plausible but meaningless numbers.
- A layer material missing from `materials` silently becomes vacuum.
- `n_harmonics` counts 2D harmonics (51 keeps only orders ±4 of a 1D grating),
  and TM converges slowly.
- `include_plot` defaults to true and returns a PNG on every call; tasks should
  ask for `false`.

**psi4**

- This is ChemMaster's server, not one from the psi4 project. psi4 is imported
  lazily, so a server without psi4 passes L0 and then fails every call.
- On small hosts, pass small `memory_gb`/`n_threads` (the default is 4 GB).
- Trustworthy: energies, optimised geometries, frequencies at minima, TDDFT
  excitations.
- Broken:
  - Imaginary frequencies lose their sign (`n_imaginary` stays 0).
  - Several fields are always `null`.
  - IR intensities are 0.
  - `temperature_K` is ignored.
  - `optimize_excited_state` returns the ground-state minimum.

**rdkit**

- This is tandemai's server; catalog id `rdkit_tandem`. The catalog's `rdkit`
  entry is a different server (two ToolUniverse RDKit tools).
- Upstream has no lockfile and allows any `mcp>=1.23`; with mcp 2 the server
  does not start. The manifest pins `mcp==1.30.0`. The default transport is
  SSE, so the launch passes `--transport stdio`.
- Molecules move between tools as base64 pickles (`p_mol`/`pmol`, hundreds of
  characters). The agent has to copy them exactly. The server loads them with
  `pickle.loads`, so it trusts the client completely.
- Trustworthy (identical to RDKit): 45 SMILES descriptor tools,
  `compute_descriptors`, scaffolds, `FragmentMol`, Tanimoto, `EmbedMolecule`
  and `EmbedMultipleConfs` with `params.randomSeed`, 2D coordinates, SDF/PDB
  I/O, PNG images.
- Broken: `CalcFractionCSP3`, `CalcPBF`, `GetUSR`, `GetUSRScore` (no usable
  signature) and `CalcOxidationNumbers` always fail. `GetSubstructMatch` only
  works when the match has exactly one atom. The `Set*Prop` tools and
  `UpdatePropertyCache` have no effect (properties are lost in the pickle).
  `MolsMatrixToGridImage` crashes with `useSVG`, `returnPNG` or highlights.
- Pitfalls:
  - Without `randomSeed`, embedding is not reproducible. Explicit `[H]` atoms
    are dropped and no tool adds hydrogens.
  - `compute_descriptors` silently drops invalid SMILES. `batch_map` returns
    results in completion order, `fail_fast` included.
  - The default SDF/PDB file name is the SMILES, which breaks on `/`.
    `filename` may contain `../`. Default PNG names have one-second
    resolution.
  - Unknown arguments are ignored.

**build123d**

- The session is stateful: `execute` keeps a build123d namespace and `show()`
  names, so a task can chain build → measure → export. `reset()` and
  `execute_file()` clear it; `destroy_session` is refused in stdio mode.
- `execute` needs an explicit `from build123d import *`, and its failures come
  back **in band** (`isError: false`, body starting `Error: …`). `validate` and
  `restore_snapshot` answer the same way for unknown names.
- `validate` returns PASS with `n_solids: 2` for a part a bore cut in two — the
  disjoint bodies are only a warning, so a task gate must check `n_solids`.
- Writes are limited to the server's cwd (the checkout, as `setup.py` renders
  the config) and `TMPDIR`. A task must export under `/tmp` with an
  instance-unique name and copy the file into the submission directory.
- `export` without `object_name` loses the `show()` label; STL import yields a
  shell with `volume: 0`; `design_audit` reports failed ±ε rebuilds as
  `coupling`, never `brittle`.
- The 2D drawing tools are deprecated at this revision (#465) and the suite
  needs `draft_preset()`, not the `Draft` class their docstrings mention — do
  not build a task on them.
- No display, Xvfb, network or CAD application is needed: `render_view` and
  `health_check` pass headless.

**quantum_espresso**

- Real DFT with real binaries: the conda environment provides `pw.x`,
  `bands.x`, `dos.x` and `projwfc.x` (`qe=7.5`, openmpi build). The launch env
  pins `QE_RUNNER=local`, `QE_USE_DOCKER=false` and `QE_NPROCS=1`, so
  executables are run directly, never through `mpirun`, Docker or Globus. A
  singleton `pw.x` needs no `OMPI_MCA_*` settings and writes nothing to stderr
  (measured on amd64 and aarch64).
- The two platform locks pin `qe` 7.5 in different conda-forge builds
  (`h19104ac_2` on linux-64, `hc91ee90_1` on linux-aarch64), yet a Si SCF
  (2 atoms, 30/120 Ry, 4×4×4, cold smearing) agreed **bit for bit** across
  them: `-15.75077338 Ry`, Fermi `6.5233 eV`, 3 s single-threaded. Tolerances
  can be tight; the drift to watch is the QE version, not the platform.
- `mcp` must stay below 2: this revision imports `mcp.server.fastmcp`, which
  mcp 2.x replaced with `mcp.server.mcpserver.MCPServer`. The manifest pins
  `mcp=1.28.1`. `spglib` is declared by upstream but never imported.
- The 219 SG15 ONCV `.upf` files (69 elements) are vendored in the pinned
  revision, so `scripts/download_pseudos.py` is never run and no calculation
  needs the network. The two Materials Project tools are the only ones that go
  online, and they need `MP_API_KEY`; do not describe the server as fully
  offline.
- Which pseudopotential file an element gets is **not** reproducible across
  hosts: `SG15Library._scan_library` iterates `glob("*.upf")` and lets a later
  non-`_FR` file overwrite an earlier one without sorting or comparing
  versions, so for Si (1.0, 1.1, 1.2 are all shipped) the pick follows the
  host's directory order. The *element set* is stable. A task must read the
  actual pick from `qe_list_pseudopotentials` → `details.<El>.filename` (a bare
  file name, so path scrubbing leaves it intact) and generate its ground truth
  on the host that runs the agent, or score only quantities that do not depend
  on the pseudopotential version. Measured on two hosts whose directory orders
  differ completely: Si got `1.2` (the newest) on both, O and Fe got `1.0` (not
  the newest) on both, and Ag diverged — `1.2` on amd64, `1.0` on aarch64. So
  a Si task happens to be reproducible; that is luck, not a guarantee.
- `qe_read_bands(output_dir)` and `qe_read_dos(output_dir)` want a **file**
  path (`bands.dat.gnu`, the dos `.dat`), not a directory, despite the
  parameter name; a directory gives `File not found`.
- `bands.x` and `dos.x` can only be reached through the workflow tools:
  `server.py` imports `postprocessing.run_bands/run_dos/run_pdos` but never
  registers them, so no tool can produce a PDOS and `qe_read_pdos` can only
  read a file from elsewhere.
- Semiconductors are forced to `occupations='smearing'` with cold smearing and
  `degauss=0.02`, so a band gap is derived from the `bands.dat` eigenvalues and
  the smeared SCF Fermi energy. Physically crude, but deterministic — and a
  good fingerprint, since another code will not reproduce it.
- Defaults worth knowing: cutoffs come from an SG15 hint table (Si 30/120 Ry),
  the automatic k grid is `round(40/|a_i|)` snapped to odd numbers (Si diamond
  → 11×11×11), `nbnd = 8·natoms`, and `workflow_dos` runs its NSCF step on
  **twice** the SCF grid — pass `kpoints` explicitly or it becomes very
  expensive.
- `QE_WORKDIR` is deliberately left unset, so work directories are
  `<cwd>/qe_calculations`: a temporary directory under the smoke, the checkout
  (gitignored upstream) under an agent run. Each SCF copies its `.upf` files
  there, so clean `<checkout>/qe_calculations` between runs — and nothing else,
  the checkout must stay clean for `setup.py`.
- `qe_get_job_status` only means something for the Globus runner; with
  `QE_RUNNER=local` it always answers `not found in registry`.
