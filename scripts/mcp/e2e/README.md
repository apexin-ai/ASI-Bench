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
| `measure_gpaw_table.py` | gpaw only: measures the `MEASURED` ground-truth table of `mcp_e2e.gpaw_mos2_bandgap` against the pinned server (~1 h; re-run after a revision or lock bump) |
| `measure_qe_table.py` | quantum_espresso only: measures the `MEASURED` table of `mcp_e2e.qe_si_bandstructure` (27 chains, ~3 min aarch64; `--ecut` per part + `--merge`; `--check` compares a fresh measurement with the committed table) |
| `locks/` | committed conda `@EXPLICIT` locks (psi4, gpaw, quantum_espresso) |

Install modes:

| Mode | Use when | Manifest fields |
|---|---|---|
| `uv-sync-frozen` | upstream has a `uv.lock` | optional `uv_sync_args` |
| `uv-pip-pinned` | no lockfile | `requirements` (exact `==` pins), `exclude_newer` |
| `conda-explicit` | conda-only packages | `conda` (channel, exact specs); lock per platform in `locks/`, regenerate with `setup.py <id> --lock` |
| `npm-ci` | Node server with an upstream `package-lock.json` | `npm` (`workdir` holding the lockfile, `scripts` run after `npm ci`, e.g. the TypeScript build); `launch.command` is the built `#!` script inside `workdir`, made executable by `setup.py`; `.venv` is an empty venv of `python` that only runs `smoke.py` |

Servers with prebuilt native code or host binaries also declare
`host_requirements` (`machine`, `cpu_flags`, `shared_libraries`,
`executables`); `setup.py` checks them before cloning and never installs system
packages. `executables` are looked up in `/usr/bin:/bin` only — the PATH tail a
smoke-test server gets after its launch directory — so a tool found only in
`~/.nvm` or `/usr/local/bin` is reported missing instead of failing later.
`npm-ci` runs `npm` with that same PATH (and without `npm_config_*` /
`NODE_OPTIONS`), so native addons are compiled for the node that runs them.

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
`~/mcp/<checkout>/.venv/bin/python` (`arxiv`, `alphafold_db` and `ncbi` →
`~/mcp/tooluniverse`); the config path stays `~/mcp/<id>.mcp.json`. The second
and third id of a group need no clone and no new `.venv`, only their own config:

```sh
python3 scripts/mcp/e2e/setup.py alphafold_db --root ~/mcp
~/mcp/tooluniverse/.venv/bin/python scripts/mcp/e2e/smoke.py alphafold_db \
  --config ~/mcp/alphafold_db.mcp.json --report ~/mcp/alphafold_db-smoke-report.json
```

Run the smoke with the server's own venv: the reference needs the same
scientific library. For a non-Python server (`npm-ci`) that venv is empty and
the reference comes from closed-form values or from running the host backend
binary (e.g. `/usr/bin/openroad`) directly in a separate process, not through
the server. The server is started from a temporary cwd/HOME/TMPDIR
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
| `arxiv` | `mims-harvard/ToolUniverse` (SMCP) | uv-sync-frozen, Py 3.12, checkout `tooluniverse` | network to arxiv.org | 16 / 8 / 0, amd64, 2026-10-03 (unchanged by the move to the shared checkout; aarch64 PASS 2026-09-30) |
| `alphafold_db` | `mims-harvard/ToolUniverse` (SMCP) | uv-sync-frozen, Py 3.12, checkout `tooluniverse` | network to alphafold.ebi.ac.uk | 16 / 7 / 0, amd64, 2026-10-03 |
| `ncbi` | `mims-harvard/ToolUniverse` (SMCP) | uv-sync-frozen, Py 3.12, checkout `tooluniverse` | network to eutils.ncbi.nlm.nih.gov and api.ncbi.nlm.nih.gov | 13 / 5 / 0, amd64, 2026-10-03 |
| `jsbsim` | `flyintothesky/jsbsim-mcp` | uv-pip-pinned, Py 3.12 | — | 17 / 17 / 0, amd64, 2026-10-02 (aarch64 PASS 2026-10-02, old layout) |
| `s4` | `prof-davifr/mcp-s4-rcwa` | uv-pip-pinned, Py 3.12 | x86-64 (AVX2/FMA/BMI2), `libblas3 liblapack3` | 41 / 7 / 0, amd64, 2026-10-02 |
| `psi4` | `Keith9922/chemaster` (`calc_psi4`) | conda-explicit | `micromamba` on `PATH` | 14 / 11 / 0, amd64, 2026-10-02 (aarch64 PASS 2026-10-02, old layout) |
| `rdkit` | `tandemai-inc/rdkit-mcp-server` (catalog `rdkit_tandem`) | uv-pip-pinned, Py 3.12 | — | 125 / 28 / 0, amd64, 2026-10-03 (aarch64 identical 2026-10-02) |
| `build123d` | `pzfreo/build123d-mcp` | uv-sync-frozen, Py 3.12 | — | 89 / 15 / 0, amd64 and aarch64, 2026-10-03 |
| `gpaw` | `Crystalhihihi/matmcp` | conda-explicit | `micromamba` on `PATH` | 38 / 13 / 0, amd64, 2026-10-03 (24 min single-threaded) |
| `quantum_espresso` | `frimpsjoek/qe-mcp` | conda-explicit (`qe=7.5`) | `micromamba` on `PATH` | 44 / 18 / 0, amd64 and aarch64, 2026-10-07 (~2 min / 62 s single-threaded) |
| `openroad` | `The-OpenROAD-Project/OpenROAD-MCP` (TypeScript, tag v1.1.0) | npm-ci, `typescript/`, Py 3.12 venv for the smoke only | `node` 22+, `npm`, `openroad`, `make`, `g++`, `python3` in `/usr/bin` or `/bin` | 103 / 4 / 0, amd64, 2026-10-07 (openroad `v2.0-17598-ga008522d8`; no sentinel-echo race in three runs) |

Install the prerequisites before running `setup.py`:

```sh
sudo apt-get install libblas3 liblapack3            # s4, admin, once
mkdir -p ~/.local/bin && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
  | tar -xj -C ~/.local bin/micromamba               # psi4, gpaw, quantum_espresso
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

**alphafold_db**

- Same checkout, pin and cwd rules as `arxiv` (one `tooluniverse` checkout
  serves all three ids).
- Live data has no date window, so stability comes from an independent query of
  the same endpoint plus recomputation: the model entry is compared field by
  field with a raw query made by the smoke (internal consistency alone would let
  a stale payload pass), `sequenceChecksum` is `md5(uniprotSequence)`,
  `globalMetricValue` and the
  `fractionPlddt*` bins can be recomputed from the model `pdbUrl` (CA
  B-factors), and MUTAGEN values are per-position means of the AlphaMissense
  `-aa-substitutions.csv` the annotation links to. The file carries two
  decimals, so a recomputed mean needs a ~0.02 tolerance.
- Tasks must not score pLDDT, any `*Url`, `latestVersion` or
  `modelCreatedDate`: AlphaFold DB is at model v6 and these change with every
  model version. The AlphaMissense scores are a fixed 2023 release and are safe.
- `alphafold_get_prediction` returns **one model per isoform** (P04637: 9),
  each with its own isoform accession (`P04637-2`) and `entryId`
  (`AF-P04637-2-F1`). Select by `entryId`, never by position.
- `alphafold_get_summary.structures` mixes in predicted complexes (AF3-style
  HETERODIMERs) and grows upstream: for P69905 it holds 16 entries and
  `AF-P69905-F1` is at index 8. Select by `model_identifier`.
- `alphafold_get_annotations` returns AlphaMissense scores (`description:
  "AM score"`), not the "experimental mutagenesis data mapped from UniProt"
  its description claims, and the description's "returns empty for every
  accession" note is stale.
- Only *declared* parameters reach the tool: FastMCP rejects an undeclared
  argument, so the config's `auto_query_params` silently overwriting a caller's
  `type` (a real defect of the Python API) cannot be triggered over MCP —
  `type=NONSENSE` is a validation error. `sequence_checksum` is declared, so it
  does get through and is forwarded as a query parameter the API ignores.
- Every failure is in-band: an entry name (`HBA_HUMAN`) or a malformed
  accession is HTTP 400 upstream, which the tool reports as
  `{"status": "error"}` with `isError: false`.

**ncbi**

- Same checkout, pin and cwd rules as `arxiv`.
- Thin `esearch`/`esummary` wrappers, so every payload is nested under
  `data` → `esearchresult` / `result`.
- Address proteins by **GI number**: a GI pins one record version, so `slen`,
  `accessionversion` and `title` do not drift; an accession follows the latest
  version.
- `genomicinfo[].chrstart/chrstop` are **0-based**, and `chrstart > chrstop` is
  how a minus-strand gene is expressed. `efetch rettype=gene_table` reports the
  same locus 1-based with the strand spelled out (+1 on both ends). Never fetch
  `efetch db=gene retmode=xml` for a reference: 34 MB for TP53.
- Tasks must not score the gene `summary` text, `exoncount`, `geneweight`,
  `createdate`/`updatedate`, or counts for free-text queries; `idlist` for a
  `SYM[Symbol] AND organism[Organism]` query, `maplocation` and `chraccver`
  are stable.
- Every failure is in-band with HTTP 200, in three different shapes: a
  top-level `error` (id that is not a uid), a per-uid `error` (unknown uid),
  and an empty `uids` list with no message at all. `retmax=0` is accepted and
  returns an empty `idlist` next to a non-zero `count`.
- The wrapper sends neither `tool=` nor `email=` and injects no `api_key`, so
  only the shared 3 requests/s budget is available; the smoke serialises its
  calls and its own reference requests.

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

**gpaw**

- This is the matmcp server (catalog `gpaw`); GPAW comes from conda-forge, so
  the install is `conda-explicit` and the smoke must run with the server's own
  prefix. No network, no PAW data download (`gpaw-data` ships the setups) and no
  `GPAW_SETUP_PATH`.
- Only one structure is reachable without a Materials Project key: the built-in
  2H-MoS2 monolayer. With `use_builtin=true` the `query` is ignored (it only
  names the run directory), and every other query needs `MP_API_KEY`, so tasks
  can vary parameters but not the material. `mp-api` is nevertheless a hard
  dependency: `fetch_structure` imports `MPRester` before the built-in branch.
- Each tool works on a run directory, `runs/<run_id>/`, under `MATMCP_REPO`
  (the checkout, as `setup.py` renders the config; upstream gitignores `runs/`).
  The `run_id` is the server's own timestamp + uuid4, which chains the tools
  together: a task should require it to be passed on verbatim.
- The order of calls matters. `calc_band_dos` only reports
  `params_verified: true` when a `check_convergence` gate has passed at or below
  its own `ecut`/`kpts_density`, so the gate has to run first. The reply then
  carries a `verification_note` that `summary.json` does not: an agent copying
  the file loses the warning.
- `gs.gpw` is one shared mutable restart file per run, written by every SCF
  (`check_convergence` included) and read by the band structure: a task must fix
  the call order and must not mix an old `summary.json` with a newer `gs.gpw`.
- `check_convergence` sweeps a hard-coded range (ecut 300-800 eV, density
  10-45) regardless of the run's parameters, takes about a minute, and only
  `tol_mev_per_atom` moves the k-grid recommendation (`tol_gap_ev` is not on the
  MCP surface). Densities that realize the same grid are evaluated once.
- `run_verified_workflow` runs the whole chain as Python functions, so it
  produces every artefact from a single tools/call and leaves no evidence of the
  individual steps: a task about the chain has to forbid it. Its convergence
  gate is real — bad parameters give `rejected_unconverged` with no gap and no
  figures (and a Chinese rejection notice).
- `calc_band_dos` prints ~123 non-JSON lines to stdout (GPAW's own SCF table:
  `band_eigs` reopens the restart file without `txt=`). Claude Code and Codex
  tolerate it, stricter clients may not.
- All errors come back in band (`isError: false`, `{"ok": false, "error": ...}`),
  including path traversal, unknown runs, `engine="qe"` (no pseudopotentials)
  and unknown engines. The `traceback_tail` contains host paths.
- Keep `OMP_NUM_THREADS=1` (the launch env does): the thread count moves the
  last digits of the energies. Across hosts the same conda lock agrees to
  ~1e-10 eV on energies and the gap and ~1e-8 eV on the Fermi level (aarch64 vs
  amd64), but values drift by a few meV between GPAW releases, so ground truth
  has to pin the lock. The FastMCP banner on stderr cannot be switched off when
  the server is started as `python -m matmcp.server`; it does not touch the
  transport.
- Budget time, not just correctness: the whole smoke is 24 min of
  single-threaded plane-wave DFT (amd64, 2026-10-03), one `run_verified_workflow`
  call alone being several minutes. The smoke declares `call_timeout` (2 h)
  because the client default of 300 s turned a correct workflow call into a
  FAIL, prints each step before it starts and records `seconds_by_step` in the
  report — use those numbers for an L2 task's `--timeout`.
- The built-in path needs no network and no credentials, but one check
  deliberately asks for a Materials Project structure without a key and expects
  it to fail (HTTP 401, or a connection error on an offline host): that is the
  only call that reaches out.
- A task cannot recompute the DFT: GPAW exists only in this conda prefix and a
  task runtime may not build one. `mcp_e2e.gpaw_mos2_bandgap` therefore carries
  a measured table, produced by `measure_gpaw_table.py` (one chain per grid
  point; eight chains took 63 min on amd64), and derives everything else — k-grid, convergence recommendation, `params_verified`
  and the `verify_run` verdict — with the same pure functions this smoke
  validates. Only `(ecut, kpts_density)` needs measuring; `tol_mev_per_atom` just
  picks a row of the fixed sweep and `gap_tol_ev` a branch of the verifier.
- `verify_run`'s `structure_drift` check is **not reproducible**. Upstream takes
  `norm(a1.positions - a0.positions).max()` over the two CIFs (`verify.py:153`)
  with no minimum-image convention, and ASE wraps coordinates when it reads a
  CIF. `a2_x` is negative in this hexagonal cell, so an infinitesimal *positive*
  fractional y becomes an infinitesimal *negative* Cartesian x and wraps to
  `+a`. Mo's y force is zero by symmetry, but floating-point summation leaves a
  denormal residue, so identical parameters on the same host give `y = 0.0` in
  one run and `1.17e-19` in the next: 1e-19 of input, 3.18 A of reported drift,
  and the check flips pass/warn per run (observed 0.020 A in 14 runs,
  3.180 A in the 15th). A task must keep this check out of its ground truth, and
  out of anything derived from it — verdicts are fail > warn > pass, so only a
  verdict that some *failing* check already decides is derivable. The L1 smoke
  cannot catch this: its own `drift` reimplements the same naive formula, so it
  agrees with the server either way.
- Relaxation below ecut 400 eV stops on the force threshold rather than at a
  minimum: at 350 eV BFGS halts after 2 steps with fmax 0.0495 against the 0.05
  criterion and reports an *indirect* Gamma->K gap, while 400-500 eV converge in
  3 steps at ~0.0065 eV/A and give the direct K->K gap. A task must not build
  instances on that corner.

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
  offline. Without the key both answer in band before any request
  (`MP_API_KEY not set` / `Materials Project API key not found. ...`).
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
  a Si task happens to be reproducible; that is luck, not a guarantee. A full
  scan gave 43 of the 69 elements an older file on the aarch64 VM and 33 on
  AWS amd64 (2026-10-07; Si got `1.2` on both).
  The smoke re-runs the scan in directory order (it must match the server),
  reports the stale elements as the D1 WARN and runs its DFT references with
  the file the server picked.
- `qe_read_bands(output_dir)` and `qe_read_dos(output_dir)` want a **file**
  path (`bands.dat.gnu`, the dos `.dat`), not a directory, despite the
  parameter name; a directory gives `[Errno 21] Is a directory: ...` (D2).
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
  → 11×11×11), `nbnd = 8·natoms`, the band path has `npoints_band` = 100
  points, and `workflow_dos` runs its NSCF step (tetrahedra) on **twice an
  explicit** SCF grid (`4,4,4` → 8×8×8); with `kpoints="auto"` both steps use
  the automatic grid (11×11×11 for Si).
- `qe_get_kpath` fails for **every** structure (D10): it sorts the special
  points by their coordinate arrays and gets `The truth value of an array with
  more than one element is ambiguous`. `qe_workflow_bandstructure` builds its
  path without that sort and works; a task must not depend on `qe_get_kpath`.
- `qe_run_relax` and `qe_run_vc_relax` report the energy, Fermi level, forces
  and stress of the **first** SCF step (D11: the parser takes the first `!`
  line), not of the relaxed structure — for the displaced Si cell
  `-15.74891110 Ry` instead of `Final energy = -15.7507748647 Ry`, and a
  vc-relax reports the input cell's `-15.75077338 Ry` while the cell relaxes to
  a = 5.4887 Å. Only the `.out` file in `output_dir` holds the result;
  `qe_workflow_relax_and_scf`'s `relaxation.energy_eV` has the same defect.
- `qe_workflow_relax_and_scf` runs its final SCF (`conv_thr` 1e-8) on the
  **input** geometry (D12; upstream says so in a comment), so its
  `total_energy` is the unrelaxed one — the relaxation is wasted.
- `forces_eV_per_angstrom` has 7×nat rows (D13): `verbosity='high'` prints six
  contribution blocks after the total forces and the parser keeps them all.
  Only the first nat rows are forces.
- Budget: every DFT tool on this Si workload (2 atoms, 30/120 Ry, 4×4×4, 40 band
  points) takes 2–8 s single-threaded on the aarch64 VM (scf 2.0, relax 2.6,
  vc-relax 3.5, band structure 7.2, DOS 8.0, relax+SCF 4.6) and about twice
  that on AWS amd64 (3.2, 6.1, 6.1, 14.3, 14.6, 9.9); the whole smoke,
  references included, is one to two minutes. Every energy the smoke prints —
  each relaxation step, the final vc-relax SCF `-15.75176482 Ry`, the tight
  SCFs — was identical on both platforms. `seconds_by_step` in the report has
  the numbers of the host it ran on.
- `QE_WORKDIR` is deliberately left unset, so work directories are
  `<cwd>/qe_calculations`: a temporary directory under the smoke, the checkout
  (gitignored upstream) under an agent run. Each SCF copies its `.upf` files
  there, so clean `<checkout>/qe_calculations` between runs — and nothing else,
  the checkout must stay clean for `setup.py`.
- `qe_get_job_status` only means something for the Globus runner; with
  `QE_RUNNER=local` it always answers `not found in registry`.
- The density cutoff is not on the MCP surface: `ecutrho` stays at the hint
  table's value (Si 120 Ry) whatever `ecutwfc` a tool is given, so above 30 Ry
  the dual drops below 4. `mcp_e2e.qe_si_bandstructure` keeps `ecutwfc` ≤ 30.
- A task cannot recompute the DFT (pw.x exists only in this conda prefix), so
  `mcp_e2e.qe_si_bandstructure` carries a table measured by
  `measure_qe_table.py`: 27 band-structure chains over (ecutwfc, grid,
  npoints_band), 180 s on the aarch64 VM, re-measured identical. The grid comes
  from the instance's k-spacing by the documented rule and is reconciled with
  `qe_suggest_kpoints`. Run `measure_qe_table.py --check` on the host that runs
  the agent before an L2 round: it proves the committed table (and the Si pick)
  is that host's own answer.
- `qe_list_files` and `qe_read_bands` take host paths as `output_dir`; the
  trajectory keeps that argument (`KEY_ARG_NAMES`), so the verifier can follow
  the chain workflow → listing → band file through scrubbed logs.

**openroad**

- The first Node server: `setup.py` runs `npm ci` on upstream's
  `typescript/package-lock.json` and `npm run build`; node-pty has no Linux
  prebuilds and compiles there (hence `make`, `g++`, `python3`). Pin the tag,
  not `main` (daily dependabot merges; #203 changed result fields).
- The Precision Innovations `.deb` does not install on Ubuntu 26.04 (`libpython3.10`
  has no candidate). The AWS host runs the 2024-12-14 build
  (`v2.0-17598-ga008522d8`) unpacked under `/opt` behind a `/usr/bin/openroad`
  wrapper that sets `LD_LIBRARY_PATH` (uv's CPython 3.10 `libpython`, a
  `libtclreadline-2.3.8.so` link to 2.4.0). tclreadline fails to start, so
  the REPL is plain Tcl and the prompt is `%`.
- Without a Liberty file STA's distance unit is 1 m and `report_design_area`
  says `0 u^2`; `set_cmd_units -distance um` (exec: `set_*`) fixes it.
- Session results are the PTY text: the echoed command (sometimes behind a
  stale `% `), the output, a bare `%`. `error` is a regex guess over that text
  (`invalid command name`, `[ERROR XXX-nnnn]`, `Error:` ...), never `isError`.
  Blocked commands never reach openroad and take no command number.
- Completion is a sentinel `puts "[join {ORMCP DONE <nonce>} -]"` written right
  after the command; lines holding the marker are dropped. When the command is
  still printing, the sentinel's PTY echo interleaves with the output: lines
  are lost or carry echo pieces (`...4000put`). The smoke compares every
  session command with a direct `openroad -exit` run and reports exactly that
  signature as WARN, any other difference as FAIL. Frequent with the fast Tcl
  stand-in; a task must not depend on one exact output line.
- The allowlist checks statement verbs and `[bracketed]` verbs only: the
  read-only query tool runs `exec` inside a `dict for` body, and the exec tool
  allows `exec` itself, so neither is a sandbox. The launch allowlist compares
  the executable's basename (`/any/path/openroad` runs). A call without
  `session_id` starts a new session and leaves it running.
- History is sorted by a millisecond timestamp, so commands in the same
  millisecond come back oldest-first; `total_commands` is the number returned.
- `ORFS_FLOW_PATH` is deliberately unset (default `$HOME/OpenROAD-flow-scripts/flow`):
  the smoke builds a fake flow tree under its temporary HOME (platform
  `l1pdk`, design `l1design`, images, metrics with a repeated key, tagged logs,
  `rules-base.json`, a stub `Makefile`), so the five ORFS tools are exercised
  without ORFS or yosys. A run's metrics count only if written after it
  started; cancel and timeout kill the run's whole process group. Which
  process notices first varies: make either dies of the SIGTERM (`signal`) or
  reaps its killed recipe and exits 2 (`exit_code`, no signal).
