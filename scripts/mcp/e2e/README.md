# Opt-in MCP end-to-end (E2E) setup and smoke tests

Installs individual catalog MCP servers at pinned upstream revisions and checks
them with **real `tools/call`**, without an agent in the loop. This is the step
between the protocol-only L0 survey (`initialize` + `tools/list`) and agent E2E
runs (`asibench run --mcp-config ...`). Upstream code is cloned, never vendored;
upstream licenses apply.

| Level | Meaning |
|---|---|
| L0 | `initialize` succeeds; `tools/list` is complete, stable and matches `manifest.json` |
| L1 | Real `tools/call` results are checked against an independent computation done by the smoke script, outside the server process |
| L2 | An agent in `asibench run` calls the tool for a fake task and its answer comes from the tool (`verify_run.py`, see `examples/mcp-e2e-tasks/README.md`) |

## Layout

- `manifest.json` — one entry per server: repository, 40-char revision, Python
  version, install mode, launch command relative to the checkout (path
  arguments and path values in the optional launch `env` use the `{checkout}`
  placeholder, everything else is literal), expected tool names. Install
  `uv-sync-frozen` uses the upstream lockfile (optional extra `uv_sync_args`);
  `uv-pip-pinned` is for upstreams without one and needs `requirements` (exact
  `name==version` pins only) and `exclude_newer` (a UTC timestamp that fixes
  the transitive resolution). `conda-explicit` is for servers that need
  conda-only packages (psi4): `conda` holds the `channel`, exact
  `name=version` `specs` (Python pinned to the manifest version) and one
  committed `@EXPLICIT` lock per conda platform under `locks/` (every package
  URL with its SHA-256; the lock header repeats server, platform, channel and
  specs, and a lock that no longer matches the manifest is refused). Servers
  that vendor prebuilt native code declare `host_requirements` (`machine`,
  `cpu_flags`, loadable `shared_libraries`, plus a human-readable `reason`).
- `setup.py` — stdlib only. Clones into `<root>/<id>` (`--filter=blob:none`),
  detaches at the pinned revision (refuses dirty checkouts or foreign remotes),
  builds `<root>/<id>/.venv` with the manifest's Python (ignoring the caller's
  `UV_PYTHON`) — `uv sync --frozen [uv_sync_args]`, or `uv venv --clear` +
  `uv pip install --exclude-newer <exclude_newer> <pins>`, or for
  `conda-explicit` a fresh `micromamba create --no-rc --prefix <root>/<id>/.venv
  --file locks/<id>-<platform>.txt` (no solver at install time; micromamba
  verifies every SHA-256; package cache in `<root>/.micromamba`; `micromamba`
  must be on `PATH`) — and writes
  `<root>/<id>.mcp.json` for `--mcp-config` (including the manifest's launch
  `env` with `{checkout}` resolved). `host_requirements` are checked before
  anything is cloned; system packages are never installed (that needs an
  administrator). `setup.py <id> --lock` (maintainers, needs network)
  re-solves `conda.specs` with `micromamba create --dry-run --json` for every
  locked platform (`CONDA_OVERRIDE_GLIBC=2.28`, conda-forge's Linux baseline)
  and rewrites the locks.
- `stdio_client.py` — stdlib-only JSON-RPC stdio client. It reads raw server
  stdout so non-JSON lines are **recorded**, not swallowed by an SDK.
- `smoke_<id>.py` — per-server L0/L1 checks; writes a JSON report with versions,
  revision, every check and the server's stderr tail. `smoke_common.py` holds
  the shared report and `tools/call` wrapper.
- `verify_run.py` — L2 verifier for agent runs of `examples/mcp-e2e-tasks`:
  reads Claude Code stream-json or Codex `exec --json` JSONL (or the
  trajectory) and the persisted outputs
  and checks connection, every required tool call and its result (including
  image results), chained inputs between calls, answer provenance and bypass.

Verified 2026-09-29: the original pyscf smoke (`pyscf_rhf_energy` +
`generate_pyscf_geom_input`) passed on AWS Linux amd64 (Ubuntu 26.04) and Linux
aarch64. The all-tools smoke below passed on both with identical values
(19 PASS, 8 WARN, 0 FAIL, ~4 s).

The arxiv smoke passed on 2026-09-30 on Linux aarch64 and AWS Linux amd64
(16 PASS, 8 WARN, 0 FAIL, ~85 s including the 3 s arXiv request spacing). The
smoke process may print a harmless pydub "Couldn't find ffmpeg" warning when it
imports MarkItDown for the reference.

The jsbsim smoke passed on 2026-10-02 on Linux aarch64 (five consecutive runs)
and AWS Linux amd64 (17 PASS, 17 WARN, 0 FAIL, ~2 s each); altitude and
airspeed after the 10 s run are bit-identical on both architectures, attitude
angles differ by < 1e-12 deg. Where `execute_script` segfaults varies between
runs (during the call, the next `step` or `close_session`).

The s4 smoke passed on AWS Linux amd64 with the pinned upstream `libS4.so` on
2026-10-02 (41 PASS, 7 WARN, 0 FAIL, ~10 s). On Linux aarch64 it passed
against a locally built `libS4.so` with the same values (40 PASS, 8 WARN: the
extra one is "not the upstream binary", four consecutive runs).

The psi4 smoke passed on Linux aarch64 on 2026-10-02 (14 PASS, 11 WARN,
0 FAIL, ~20 s, four consecutive runs); AWS amd64 pending.

## Run (Linux, as the unprivileged E2E user)

```sh
cd ~/ASI-Bench
python3 scripts/mcp/e2e/setup.py pyscf --root ~/mcp
~/mcp/pyscf/.venv/bin/python scripts/mcp/e2e/smoke_pyscf.py \
  --config ~/mcp/pyscf.mcp.json --report ~/mcp/pyscf-smoke-report.json
uv run asibench mcp check --config ~/mcp/pyscf.mcp.json

python3 scripts/mcp/e2e/setup.py arxiv --root ~/mcp
~/mcp/arxiv/.venv/bin/python scripts/mcp/e2e/smoke_arxiv.py \
  --config ~/mcp/arxiv.mcp.json --report ~/mcp/arxiv-smoke-report.json
uv run asibench mcp check --config ~/mcp/arxiv.mcp.json

python3 scripts/mcp/e2e/setup.py jsbsim --root ~/mcp
~/mcp/jsbsim/.venv/bin/python scripts/mcp/e2e/smoke_jsbsim.py \
  --config ~/mcp/jsbsim.mcp.json --report ~/mcp/jsbsim-smoke-report.json
uv run asibench mcp check --config ~/mcp/jsbsim.mcp.json

# s4 needs x86-64 and the system BLAS/LAPACK (admin, once: apt-get install libblas3 liblapack3)
python3 scripts/mcp/e2e/setup.py s4 --root ~/mcp
~/mcp/s4/.venv/bin/python scripts/mcp/e2e/smoke_s4.py \
  --config ~/mcp/s4.mcp.json --report ~/mcp/s4-smoke-report.json
uv run asibench mcp check --config ~/mcp/s4.mcp.json

# psi4 is conda-only: needs the standalone micromamba binary on PATH, e.g.
#   mkdir -p ~/.local/bin && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
#     | tar -xj -C ~/.local bin/micromamba
python3 scripts/mcp/e2e/setup.py psi4 --root ~/mcp
~/mcp/psi4/.venv/bin/python scripts/mcp/e2e/smoke_psi4.py \
  --config ~/mcp/psi4.mcp.json --report ~/mcp/psi4-smoke-report.json
uv run asibench mcp check --config ~/mcp/psi4.mcp.json
```

The smoke script must run with the server's own virtualenv so that its
reference calculation can import the same scientific library. The server is
launched from a temporary cwd/HOME (separate directories for arxiv; jsbsim,
s4 and psi4 also get a temporary `TMPDIR`) with a
minimal environment plus the config's `env`, and no operator credentials. Exit code is non-zero if any check FAILs; WARNs do not fail.

## Server notes

### pyscf (`lixin19/mcp2pyscf`)

Upstream requires Python ≥ 3.13 (`.python-version` 3.13); the lockfile pins
PySCF 2.9.0, RDKit 2025.3.3, geomeTRIC 1.1, MCP SDK 1.9.4. The smoke calls all
seven tools. References are computed in the smoke process with the same
libraries but independently of the server code:

| Tool | L1 check (FAIL if wrong) | Known WARN on the pinned revision |
|---|---|---|
| `pyscf_rhf_energy` | H₂/STO-3G, H₂O/STO-3G, H₂O/6-31G vs PySCF RHF (`symmetry=True`), 1e-7 Ha | one stdout line per call |
| `generate_pyscf_geom_input` | H₂O, NH₃, HCN: atom order and sorted interatomic distances (≤1e-3 Å) vs seeded RDKit ETKDG + UFF, RHF/STO-3G energy at both geometries (≤1e-5 Ha) | invalid SMILES returns `"Error: ..."` with `isError=false` |
| → `pyscf_rhf_energy` chain | benzene ×8: geometry from the tool passed to `pyscf_rhf_energy`; successful energies must agree | intermittent `PointGroupSymmetryError` (about half of the calls; can be 0/8 by chance) |
| `run_bond_stretch_calculation_mcp` | H₂O O–H 0.8–1.2 Å ×5 and HCN C–N 1.0–1.3 Å ×4 vs the same stretch of a seeded UFF geometry, RHF/STO-3G, 1e-5 Ha | `basis` is ignored (always STO-3G); errors returned in-band as `{"error": ...}` |
| `optimize_molecule_mcp` | NH₃, H₂CO: sorted distances (≤2e-3 Å) and RHF/STO-3G energy at the returned geometry (≤1e-6 Ha) vs PySCF + geomeTRIC from a seeded UFF start | returns only an XYZ block (documented `optimized_energy` missing); invalid SMILES in-band; prints the geometry to stdout |
| `scan_pes_rhf` | 9 H₂/STO-3G energies (0.7–1.5 Å, hard-coded upstream) vs PySCF, 1e-7 Ha | ~19 stdout lines per call |
| `plot_energy_scan_image_mcp` | one `image/png` block, valid PNG (1000×600); mismatched input lengths must give `isError` | — |
| `visualize_molecule_3d_mcp` | HTML with the 3Dmol viewer and the geometry is written to the server cwd | result text is a hard-coded developer path; the HTML is never returned |

Why the geometry checks are loose: RDKit embedding in the server is unseeded,
so conformers differ between calls. For the rigid molecules above the UFF
minimum is unique up to rotation, translation and H permutation (measured over
repeated embeddings: distances within 4e-6 Å, RHF/STO-3G energies within
3e-7 Ha), so the smoke compares rotation- and permutation-invariant quantities.
Flexible molecules (e.g. ethanol, 7e-4 Ha spread) are deliberately not used.

Implications for agent runs:

- PySCF, geomeTRIC and upstream `print` calls write to **stdout**, i.e. into the
  stdio transport. Claude Code 2.1.284 tolerated this in the
  `mcp_e2e.pyscf_rhf_energy` runs, and so did codex-cli 0.159.2 (single energy,
  bond scan and plot calls, default MCP timeouts); stricter clients may not.
- `generate_pyscf_geom_input` → `pyscf_rhf_energy` is fragile for
  high-symmetry molecules (benzene, sometimes methane/ethane); fake tasks must
  not depend on that chain for such molecules.
- `visualize_molecule_3d_mcp` writes `optimized_geom_3d.html` into the server
  cwd, which in `--mcp-config` runs is the MCP checkout (untracked file; it
  does not block `setup.py`, which only refuses modified tracked files). It is
  not usable as an agent-verifiable tool and has no L2 task.

L2 coverage: `pyscf_rhf_energy` (`mcp_e2e.pyscf_rhf_energy`) and
`run_bond_stretch_calculation_mcp` → `plot_energy_scan_image_mcp`
(`mcp_e2e.pyscf_bond_stretch`). The remaining tools are L1 only by design: the
agent-level path (connection, calls, results, chaining, images) is already
exercised by these two tasks.

### arxiv (`mims-harvard/ToolUniverse`, SMCP)

ToolUniverse 1.5.0 at the pinned revision; `requires-python >= 3.10`, pinned to
3.12 here. `uv_sync_args: ["--no-dev"]` keeps the dev group (pytest plus the
`embedding` extra) out of the venv; the runtime still resolves ~160 packages
(FastMCP 3.4.5, MCP SDK 1.29.0, MarkItDown 0.1.7, pdfminer.six, pdfplumber;
~700 MB). The server is `tooluniverse-smcp-stdio --no-search --include-tools
ArXiv_search_papers ArXiv_get_pdf_snippets`, as in the catalog entry. Both
tools and the smoke's references need network access to `export.arxiv.org`
and `arxiv.org`; no key is needed.

Launch `env` in the manifest, and why:

- `TOOLUNIVERSE_CACHE_ENABLED=false`, `TOOLUNIVERSE_CACHE_PERSIST=false`: by
  default ToolUniverse caches every tool result forever in memory and in
  `~/.tooluniverse/cache.sqlite`, so a repeated call would not reach arXiv and
  one run's results could leak into the next. Persistence also creates
  `~/.tooluniverse`, which triggers the workspace defect below whenever the
  server's cwd is `$HOME`.
- `FASTMCP_SHOW_SERVER_BANNER=false`, `FASTMCP_CHECK_FOR_UPDATES=off`: FastMCP 3
  ignores the catalog's `FASTMCP_NO_BANNER` and otherwise queries PyPI for
  updates at every start (stderr only; not a stdout problem).
- `HF_HUB_OFFLINE=1`, `DO_NOT_TRACK=1`: kept from the catalog.
- No `TOOLUNIVERSE_HOME` and no `--workspace`: both create the workspace
  directory, which re-enables the defect below.

References are fetched in the smoke process, independently of the server code:
search results from a raw arXiv API query written out in the smoke, PDF text
from the same PDF version downloaded by the smoke and converted with the same
MarkItDown library. All searches use closed historical date windows
(2011-03), so the result sets are stable.

| Tool | L1 check (FAIL if wrong) | Known WARN on the pinned revision |
|---|---|---|
| `ArXiv_search_papers` | five queries (field prefix + date + `submittedDate` asc; unquoted keywords are ANDed; `cat:` + `limit` + desc; parenthesised OR + date; multi-word `au:` is quoted): IDs, versions, order, titles, abstracts, authors, dates and primary category identical to the raw API; dates inside the range; requested order | an unparenthesised `OR` query with a date range returns papers from any date (the date clause is appended as `... OR ti:b AND submittedDate:[...]`, so it binds to the last term only); empty query and invalid `sort_by` are errors in-band with `isError=false` |
| → `ArXiv_get_pdf_snippets` chain | the first paper of the first search (1103.0212v1) by `arxiv_id`, and again by `pdf_url`: `pdf_url`, snippet list (term, text) and count identical to snippets cut from the reference text with the documented window/limit rules; `max_total_chars` cap respected | `truncated` stays `false` when the cap drops matches (it is only set when the total reaches the cap exactly); old-style IDs whose archive contains a `v` (`solv-int/9901001`) become `sol.pdf` because the version is stripped with `split("v")`; the tool always downloads the latest version; missing paper and empty `terms` are in-band errors |
| server | unknown tool is an error; stdout is pure JSON-RPC; cwd left untouched | with a `./.tooluniverse` directory in the server cwd, ToolUniverse seeds and loads its default `profile.yaml` (all tools), `--include-tools` is ignored and **2718 tools** are exposed (probed with a second server; skip with `--skip-workspace-probe`) |

Implications for agent runs:

- The generated config's `cwd` is the checkout, which has no `.tooluniverse`;
  do not create one there, and do not run the server with cwd = `$HOME`.
  `mcp_connected` in `verify_run.py` would still pass with 2718 tools, so check
  the offered-tool count in the Claude init event if in doubt.
- Search results and snippets are live network data. Fake tasks must use closed
  date windows, parenthesise OR groups, pin a single-version paper, and must
  not score abstracts or `updated` (they change when authors post a new
  version).
- `--mcp-config` forces search mode, so Claude Code has `WebSearch`/`WebFetch`
  and Codex keeps `web_search`; the host shell has network access too. The L2
  task's `e2e_check.json` therefore flags web tools and direct arXiv access.

L2 coverage: `ArXiv_search_papers` → `ArXiv_get_pdf_snippets`
(`mcp_e2e.arxiv_search_snippets`, both tools, chained).

### jsbsim (`flyintothesky/jsbsim-mcp`)

FastMCP (MCP SDK 1.x) stdio server `run_stdio.py` over the JSBSim 1.3.1 Python
module; the repository bundles a full JSBSim source tree as `jsbsim_data/`
(aircraft, engines, systems, scripts). Upstream has no `pyproject.toml` or
lockfile and its `requirements.txt` uses ranges plus dashboard-only packages,
so the manifest uses `uv-pip-pinned`: Python 3.12, `jsbsim==1.3.1`,
`mcp==1.30.0`, `pydantic==2.13.5`, `exclude_newer` 2026-10-01 (fastapi,
uvicorn extras and websockets serve only the web dashboard; the stdio path does
not import them). `jsbsim` has wheels for Linux amd64 and aarch64. No network
or credentials are needed.

Launch `env` in the manifest, and why:

- `JBSIM_ROOT={checkout}/jsbsim_data` (upstream's spelling): otherwise the data
  root is searched from the server's cwd upwards and falls back to `.`, so
  `create_session` fails when a client does not honour the config's `cwd`.
- `JSBSIM_DEBUG=0`: read by JSBSim's `FGFDMExec`; without it every
  `create_session` writes ~1100 lines (banner and vehicle configuration) to
  stdout, i.e. into the JSON-RPC stream. Error messages (e.g. an unknown
  aircraft, 2 lines) still go to stdout.

Both are probed with two short-lived extra servers (`--skip-env-probes` skips
them).

References are computed in the smoke process with the same `jsbsim` module but
not with the server code: a separate `FGFDMExec` is driven with JSBSim's own
property names (`ic/h-sl-ft`, `ic/vc-kts` with an exact ft/s→kt conversion,
`attitude/theta-deg`, ...). Scenario: c172x, all seven initial conditions
(4000 ft, 37°N 122°W, 168.78 ft/s calibrated, heading 90°, pitch 2°, roll 0°),
then `propulsion/set-running=-1`, mixture 1.0, throttle 0.8, then 10 s at the
default dt of 1/60 s. JSBSim is deterministic: repeated runs are bit-identical.

| Tool | L1 check (FAIL if wrong) | Known WARN on the pinned revision |
|---|---|---|
| `list_aircraft` | equals the `aircraft/<name>/<name>.xml` directories under `JBSIM_ROOT` (60), count matches | — |
| `create_session` | c172x, default dt 1/60 s, t=0, root = `JBSIM_ROOT`; second session with `initial_conditions` reaches a bit-identical state (also covers chunked `step`); unknown aircraft is `isError` | without initial conditions `run_ic()` is never called and the state is invalid after stepping (h=NaN); an unknown `initial_conditions` key is silently ignored |
| `set_initial_conditions` | all seven keys read back via `get_property` (`airspeed_fps` is a calibrated airspeed: written to `ic/vc-kts`) | unspecified keys are reset to 0 (`x or 0.0`), not kept |
| `set_property` / `get_property` | writes read back; after 10 s six state properties at full precision vs the reference (altitude within 2e-3 ft; the server's 0.592484 kt/fps factor alone moves it by 3e-4 ft) | unknown paths read as `value: 0.0, present: true` |
| `step` | 600 frames, t = 10 s | `seconds=0` still integrates one frame |
| `get_telemetry` | the 13 fields whose properties exist equal the reference to printed precision | **20 of 33 fields read properties that do not exist** and are always 0/false: lat/lon, AGL, pitch/roll/heading, lift/drag/side, n1/rpm/running, nz, wind, gear WOW/compression; `cl` is CL² |
| `trim` | — (not a JSBSim trim) | returns `ok: true` but the aircraft is not trimmed (max \|u̇,ẇ,q̇\| ≈ 12 vs 1e-4 after JSBSim's own `do_trim(0)` from the same state, which the 1.3.1 module does export); the loop chases the non-existent `attitude/pitch-deg`, forces throttle 0.7, advances the simulation ~2 s; `mode` is ignored (`longitudinal`, `none` and an invalid mode give identical results) |
| `execute_script` | runs in a separate server process | the `<run>…</run>` literal the description suggests is rejected by JSBSim, yet `ok: true` (return value ignored) and the temp file is never deleted; a stock script (`scripts/c1722.xml`) is loaded on top of the live model: ~640 stdout lines despite `JSBSIM_DEBUG=0`, then the server **segfaults** (exit −11) during the call, the next `step` or `close_session` |
| `close_session` | closes; the id is unknown afterwards; closing twice gives `ok: false` | — |
| server | unknown tool is an error; cwd left untouched; process alive after the main run | unknown session ids give in-band `{"ok": false, "error": "unknown-session"}` with `isError=false` |

Implications for agent runs:

- Always pass all seven initial conditions, then the property settings, then
  step; never rely on `trim` or `execute_script` (the latter can kill the
  server, losing every session).
- Read attitude, position and engine state with `get_property` and JSBSim's
  real names (`attitude/theta-deg`, `attitude/psi-deg`, `position/lat-geod-deg`,
  `propulsion/engine[0]/engine-rpm`), not from `get_telemetry`.
- c172x starts with the engine off; `propulsion/set-running=-1` starts it
  (`propulsion/engine[0]/set-running=1` does not).
- Sessions idle for 300 s are closed by a background thread (not configurable).

L2 coverage: `create_session` → `set_initial_conditions` → `set_property` ×3 →
`step` → `get_property`/`get_telemetry` on one session
(`mcp_e2e.jsbsim_engine_run`). `list_aircraft`, `close_session`, `trim` and
`execute_script` are L1 only; the last two are unusable on the pinned revision.

### s4 (`prof-davifr/mcp-s4-rcwa`)

FastMCP (MCP SDK 1.x) stdio server over S4, the Stanford Stratified Structure
Solver (RCWA). Upstream drives a **prebuilt `libS4.so` committed to the
repository** through its own ctypes wrapper (the `S4` Python binding is
Python-2 only). That binary is x86-64 only, was built with GCC 13.3 on Ubuntu
24.04 with `-march=native` (it uses AVX2, FMA and BMI2, no AVX-512), links the
system `libblas.so.3`/`liblapack.so.3` and needs glibc ≥ 2.35; the S4 source
revision it was built from is not recorded (S4's last upstream commit is from
2018). Hence the manifest's `host_requirements`; on Debian/Ubuntu an
administrator installs `libblas3 liblapack3` once. The smoke reports the
binary's SHA-256 and WARNs when it is not the upstream one.

Upstream has a `pyproject.toml` (hatchling) but no lockfile, so the manifest
uses `uv-pip-pinned` (Python 3.12, `matplotlib==3.11.2`, `mcp==1.30.0`,
`numpy==2.5.3`, `pydantic==2.13.5`, `exclude_newer` 2026-10-01) and runs the
package from the checkout instead of building it: `python -m
mcp_s4_rcwa.server` with launch `env` `PYTHONPATH={checkout}/src`. No network
or credentials are needed.

For development on other platforms (e.g. Linux aarch64) S4 can be built from
<https://github.com/victorliu/S4> (`make build/libS4.a` with
`-DHAVE_BLAS -DHAVE_LAPACK`, then link it as a shared library against the
system BLAS/LAPACK) and copied over the vendored binary in a scratch checkout;
`setup.py` refuses such a modified checkout, and the smoke marks the run as
not representative.

References are computed in the smoke process with numpy only, sharing nothing
with S4 or the server: a transfer-matrix method (TMM) for unpatterned stacks,
and a 1D RCWA (enhanced transmittance matrix) for lamellar gratings. S4's
default formulation uses Laurent's rule for both polarisations and keeps the
reciprocal-lattice vectors inside a circle; for a complete shell of the
square lattice (21, 49, 81 harmonics → orders ±2, ±4, ±5 on the x axis) and a
grating uniform along y, the 1D RCWA with Laurent's rule and the same orders
must agree to rounding. Grating wavelength grids avoid Rayleigh anomalies.

| Tool | L1 check (FAIL if wrong) | Known WARN on the pinned revision |
|---|---|---|
| `check_engine_sanity` | default, `n_harmonics` 1 and 51: `ok`, R/T vs the exact Fresnel value for air/n=3.47 (1e-9) | — (`expected_R` 0.3055 is a rounded value; the tool's own tolerance is 0.01) |
| `simulate_stack_spectrum` — planar stacks | upstream README example; quarter-wave mirror (glass \| (HL)×4 \| air) in TE and TM at 0/30/60°; absorbing film (n+ik) in TE/TM at 0/45°: R, T, A vs TMM ≤ 1e-9 (observed ~1e-15); eps notation equals n/k; wavelength grid is `linspace(start, stop, points)` | — |
| `simulate_stack_spectrum` — gratings | TE/TM at normal incidence (diffracting below the period), TE 20°, TM 20°, TM 10° with an absorbing ridge: R, T, A vs the 1D RCWA with matching truncation ≤ 1e-9 (observed ≤ 2e-14); a 2D rectangle rotated by 90° with TE↔TM gives the same spectrum; shifting the pattern centre changes nothing; identical calls are bit-identical | — |
| convergence (informational) | vs the converged 1D RCWA (Li's rule, ±60 orders) at 51 and 201 harmonics: TE 7e-5 / 1e-5, TM 2e-3 / 1e-3; WARN only above 5e-3 at 201 | S4's default TM formulation converges slowly; results must be checked against `n_harmonics`, as the tool description says |
| validation | `wavelength_points < 2`, `stop <= start`, unknown `incidence_layer`, polarisation `XY`, unknown pattern material: `isError` | the unknown pattern material is reported only as `'nope'` (a bare `KeyError`) |
| PNG plot | `include_plot` (default true) adds one valid `image/png` (720×480); `false` returns the same spectrum as text only | — |
| defect probes | — | swapped `incidence_layer`/`substrate_layer` returns R=0, T>1, A<0; an inner layer as `incidence_layer` returns a wavelength-independent, meaningless spectrum (S4 always illuminates from the **last** layer; the names only select where fluxes are read); a layer material missing from `materials` silently becomes vacuum (material id −1); `theta_deg` 90 or 120 returns R=T=0, **A=1**; negative thickness and duplicate layer names are accepted |
| server | unknown tool is an error; stdout is pure JSON-RPC; cwd left untouched; process alive after all calls | — |

Implications for agent runs:

- Layers must be ordered substrate first, incidence medium last, and
  `incidence_layer`/`substrate_layer` must name exactly those two; nothing
  else is validated, and wrong choices give plausible-looking numbers.
- Every material named by a layer must be in `materials` (a typo makes the
  layer disappear without an error).
- `n_harmonics` counts 2D harmonics of a square lattice, so 51 keeps only
  orders ±4 for a 1D grating; TM gratings need many harmonics to converge.
- With the default `include_plot=true` every call also returns a PNG, which
  costs context; fake tasks should ask for `include_plot=false` unless the
  plot is the point.

L2 coverage: `simulate_stack_spectrum` of a TM grating with a given harmonic
count, answers taken from the returned spectrum (`mcp_e2e.s4_grating_spectrum`);
`check_engine_sanity` is judged when an agent calls it, otherwise L1 only.

### psi4 (`Keith9922/chemaster`, `calc_psi4`)

The catalog's psi4 entry is not a psi4-project server but the `calc_psi4`
FastMCP server of ChemMaster, a computational-chemistry agent with a dozen
MCP servers. psi4 is imported lazily inside each tool, so a server without
psi4 passes L0 (`initialize`, `tools/list`) and fails every call with
`No module named 'psi4'`. psi4 is conda-only (no PyPI wheel; ChemMaster is not
on PyPI either), hence install mode `conda-explicit`: Python 3.12.14, psi4 1.11,
dftd3-python 1.6.0 (needed by the default method `B3LYP-D3(BJ)`), mcp 1.28.1
(the newest 1.x on conda-forge), pint 0.26.1 and scipy 1.18.1 from
conda-forge, locked for linux-64 (143 packages; psi4 pulls MKL/einsums) and
linux-aarch64 (129). The server only needs those packages, so the checkout is
not installed: launch `python -m chemaster.mcp.calc_psi4.server` with `env`
`PYTHONPATH={checkout}` and `OMP_NUM_THREADS=1`. No network or credentials.

References are computed in the smoke process by psi4 itself with the settings
the tools document (density-fitted SCF, C1, RHF/UHF by multiplicity) but
through a different path — wavefunction objects, `psi4.variable` and
`tdscf_excitations` instead of the server's output-log parsers. During
probing the HF/STO-3G water energy also matched PySCF with density fitting
(`def2-universal-jkfit`) to 1e-10 Eh. Every call passes `memory_gb=1`,
`n_threads=1` (the default 4 GB exceeds small VMs).

| Tool | L1 check (FAIL if wrong) | Known WARN on the pinned revision |
|---|---|---|
| `single_point` | water HF/STO-3G, `B3LYP-D3(BJ)`/def2-SVP, MP2/cc-pVDZ and OH UHF/6-31G doublet energies vs psi4 (≤ 2e-7 Eh, observed ≤ 5e-9) | `n_basis_functions`, `n_iterations`, `homo_lumo_gap`, `dipole` are always `null`: they come from `psi4.core.get_active_wavefunction()`, which psi4 1.11 does not have (upstream's tests monkeypatch it), and the `AttributeError` is swallowed; a doublet closed-shell water is rejected in-band (`ok:false`, `INVALID_MULTIPLICITY`, `isError:false`) |
| `optimize` | water HF/STO-3G (tight) from a distorted start: energy ≤ 1e-6 Eh and interatomic distances ≤ 1e-3 Å vs an independent psi4 optimisation, `converged` | `n_iterations` is always 0 (same cause) |
| `frequency` — minimum | at the server's own minimum: signed frequencies (≤ 0.5 cm⁻¹, observed 5e-5), `n_imaginary` 0, ZPE and thermal E/H/G corrections (≤ 2e-6 Eh) vs psi4 | IR intensities are all zero (log-parser fallback) although psi4 gives 7.2/44.3/30.1 km/mol; `temperature_K` (and `pressure_atm`) are only echoed — the thermochemistry stays at 298.15 K |
| `frequency` — saddle | planar NH₃ (D3h, first-order saddle, optimised in symmetry by the reference): PASS only if the imaginary mode is returned negative with `n_imaginary` 1 | the 1081.6i cm⁻¹ umbrella mode is returned as **+1081.6** with `n_imaginary` 0, no `IMAGINARY_FREQUENCY` warning, and a ZPE that counts it (0.0400 vs psi4 0.0376 Eh): the log parser drops the `i`; any other mismatch is FAIL |
| `tddft` | water B3LYP/def2-SVP, TDA singlets+triplets and full TDDFT singlets: excitation energies (≤ 2e-3 eV) and oscillator strengths vs `tdscf_excitations`, ground-state energy, `delta_E_ST_eV` = E(T1) − E(S1) as documented | `n_states` is documented "per spin manifold" but psi4 splits it: `n_states=4` with triplets gives 2 singlets + 2 triplets |
| `optimize_excited_state` | PASS only if `final_total_energy` equals E(S1) at the returned geometry and the S1 gradient there (central differences) is below 2e-3 Eh/Å | water S1 B3LYP/STO-3G returns the **ground-state minimum** (energy equal to an independent S0 optimisation; psi4's finite-difference TDSCF optimisation ignores `FOLLOW_ROOT`), and `excitation_energy_at_opt` is the starting geometry's value (11.06 vs 10.23 eV); recognised exactly → WARN, otherwise FAIL |
| server | unknown tool is an error; process alive after all calls | psi4 prints `Memory set to …`, `Threads set to …`, `Optimizer: Optimization complete!` on stdout (non-JSON lines); psi4 writes `timer.dat` into the server cwd |

Implications for agent runs:

- Only energies, optimised geometries, frequencies at minima and TDDFT
  excitations are trustworthy; treat `null` fields as missing, never compute
  "imaginary frequency" conclusions from `n_imaginary`, and do not use
  `optimize_excited_state` for excited-state geometries.
- Agents should pass small `memory_gb` / `n_threads` on small hosts and expect
  psi4 logs (`meta.output_path`) under the server's `TMPDIR`.
