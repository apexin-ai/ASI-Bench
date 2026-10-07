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
| `locks/` | committed conda `@EXPLICIT` locks (psi4, gpaw) |

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
`~/mcp/<checkout>/.venv/bin/python` (`arxiv`, `alphafold_db` and `ncbi` →
`~/mcp/tooluniverse`); the config path stays `~/mcp/<id>.mcp.json`. The second
and third id of a group need no clone and no new `.venv`, only their own config:

```sh
python3 scripts/mcp/e2e/setup.py alphafold_db --root ~/mcp
~/mcp/tooluniverse/.venv/bin/python scripts/mcp/e2e/smoke.py alphafold_db \
  --config ~/mcp/alphafold_db.mcp.json --report ~/mcp/alphafold_db-smoke-report.json
```

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
| `arxiv` | `mims-harvard/ToolUniverse` (SMCP) | uv-sync-frozen, Py 3.12, checkout `tooluniverse` | network to arxiv.org | 16 / 8 / 0, amd64, 2026-10-03 (unchanged by the move to the shared checkout; aarch64 PASS 2026-09-30) |
| `alphafold_db` | `mims-harvard/ToolUniverse` (SMCP) | uv-sync-frozen, Py 3.12, checkout `tooluniverse` | network to alphafold.ebi.ac.uk | 16 / 7 / 0, amd64, 2026-10-03 |
| `ncbi` | `mims-harvard/ToolUniverse` (SMCP) | uv-sync-frozen, Py 3.12, checkout `tooluniverse` | network to eutils.ncbi.nlm.nih.gov and api.ncbi.nlm.nih.gov | 13 / 5 / 0, amd64, 2026-10-03 |
| `jsbsim` | `flyintothesky/jsbsim-mcp` | uv-pip-pinned, Py 3.12 | — | 17 / 17 / 0, amd64, 2026-10-02 (aarch64 PASS 2026-10-02, old layout) |
| `s4` | `prof-davifr/mcp-s4-rcwa` | uv-pip-pinned, Py 3.12 | x86-64 (AVX2/FMA/BMI2), `libblas3 liblapack3` | 41 / 7 / 0, amd64, 2026-10-02 |
| `psi4` | `Keith9922/chemaster` (`calc_psi4`) | conda-explicit | `micromamba` on `PATH` | 14 / 11 / 0, amd64, 2026-10-02 (aarch64 PASS 2026-10-02, old layout) |
| `rdkit` | `tandemai-inc/rdkit-mcp-server` (catalog `rdkit_tandem`) | uv-pip-pinned, Py 3.12 | — | 125 / 28 / 0, amd64, 2026-10-03 (aarch64 identical 2026-10-02) |
| `build123d` | `pzfreo/build123d-mcp` | uv-sync-frozen, Py 3.12 | — | 89 / 15 / 0, amd64 and aarch64, 2026-10-03 |
| `gpaw` | `Crystalhihihi/matmcp` | conda-explicit | `micromamba` on `PATH` | 38 / 13 / 0, amd64, 2026-10-03 (24 min single-threaded) |
| `atomictoolkit` | `XirtamEsrevni/mcp-atomictoolkit` (catalog `ase`; the catalog's `pymatgen` entry is the same server, and `tools/list` has no pymatgen-specific tool) | uv-pip-pinned, Py 3.12 | — | 44 / 31 / 0, aarch64, 2026-10-07 (~5 s) |

Install the prerequisites before running `setup.py`:

```sh
sudo apt-get install libblas3 liblapack3            # s4, admin, once
mkdir -p ~/.local/bin && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
  | tar -xj -C ~/.local bin/micromamba               # psi4, gpaw
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

**atomictoolkit**

- One server behind two catalog entries (`ase`, `pymatgen`): pymatgen only
  supplies the spacegroup strings; all file I/O and numbers are ASE (EMT).
  Upstream's `-e .` cannot be pinned, so the manifest pins the third-party
  dependencies and launches with `PYTHONPATH={checkout}/src`.
- Only 9 of the 18 tools are usable: `list_workspace_capabilities`, `build`,
  `import`, `manipulate`, `analyze_structure`, `write`, `single_point`,
  `estimate_elastic` (all `*_workflow`) and `create_download_artifact`. The four
  deprecated wrappers always fail (`'FunctionTool' object is not callable`).
  The five `taskSupport: required` tools (optimize, MD, relax+MD, trajectory,
  autocorrelation) refuse a plain call; with `params.task` they run and write
  their files, but `tasks/get`/`tasks/result` always answer `No active context
  found.`. So there is no relaxation and no MD: a task must say "unrelaxed".
- Always pass `calculator_name: "emt"` (Al, Cu, Ag, Au, Ni, Pd, Pt only).
  `auto` tries kim, orb and nequix first (kim is listed as available but cannot
  start) and returns three `calculator_fallbacks` messages. Even an explicit
  `emt` falls back to the others on failure.
- `analyze_structure_workflow` coordination numbers keep ASE's default
  neighbour-list skin (fcc Cu: 18, not 12) and its g(r) tends to 2, not 1. They
  are reproducible, so they can be fingerprints of the tool, but never
  "physical" answers. Molecules (no cell) fail analysis in band.
- `estimate_elastic_workflow`'s B is `2a/(9V0)` from a quadratic E(strain) fit
  at the given cell, pressure term included.
- Relative paths resolve against the server cwd — the MCP checkout in agent
  runs — and every in-band error writes `tool_errors/<tool>_<UTC>.log` there. Tasks
  must ask for absolute output paths; before an L2 batch, delete earlier
  products from the checkout by name (not `git clean -fdx`, which removes
  `.venv`).
- Every result echoes absolute paths (`filepath`, `input_filepath`) and an
  `artifacts` list with a fresh `art_<32 hex>` id per file and call: an id proves
  that a call happened, not which earlier file it named. `download_url` is a
  relative HTTP route that stdio cannot serve.
- An error message longer than a path component (e.g. Si with EMT) crashes the
  artifact scan: the structured report is lost and the call becomes a bare
  `[Errno 36] File name too long` protocol error. Other failures are in band.
- Numbers agree bit for bit with ASE called directly on the same file; extxyz
  stores 8 decimals, so ground truth must start from the file content (or allow
  ~1e-13 eV). `estimate_elastic` uses `np.polyfit`: compare with rtol 1e-9.
