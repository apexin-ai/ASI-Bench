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
| `manifest.json` | per server: repository, 40-char revision, Python, install mode, launch command/env (`{checkout}` placeholder), expected tools |
| `setup.py` | clone, pin, build `<root>/<id>/.venv`, write `<root>/<id>.mcp.json` (stdlib only) |
| `smoke_<id>.py` | L0/L1 checks for one server, JSON report; `smoke_common.py` and `stdio_client.py` are shared |
| `verify_run.py` | L2 verifier CLI for agent runs; implementation in `e2e_verify/` (spec, extractors, values, evidence, checks) |
| `locks/` | committed conda `@EXPLICIT` locks (psi4) |

Install modes:

| Mode | Use when | Manifest fields |
|---|---|---|
| `uv-sync-frozen` | upstream has a `uv.lock` | optional `uv_sync_args` |
| `uv-pip-pinned` | no lockfile | `requirements` (exact `==` pins), `exclude_newer` |
| `conda-explicit` | conda-only packages | `conda` (channel, exact specs); lock per platform in `locks/`, regenerate with `setup.py <id> --lock` |

Servers with prebuilt native code also declare `host_requirements`; `setup.py`
checks them before cloning and never installs system packages.

The manifest is strict: an entry holds only the common keys plus its install
mode's fields; a misspelt key or another mode's field is an error. A new mode is
one `Installer` subclass registered in `INSTALLERS` (fields + validators,
`install`, optional `lock`); a new host check is one probe in `HOST_PROBES`.

## Run

Linux, as an unprivileged E2E user:

```sh
cd ~/ASI-Bench
python3 scripts/mcp/e2e/setup.py <id> --root ~/mcp
~/mcp/<id>/.venv/bin/python scripts/mcp/e2e/smoke_<id>.py \
  --config ~/mcp/<id>.mcp.json --report ~/mcp/<id>-smoke-report.json
uv run asibench mcp check --config ~/mcp/<id>.mcp.json
```

Run the smoke script with the server's own venv: the reference needs the same
scientific library. The server is started from a temporary cwd/HOME with a
minimal environment and no credentials. FAIL means a wrong result and exits
non-zero; WARN means an upstream defect that does not make correct use wrong.
What each check does is in the `smoke_<id>.py` docstring and its messages.

## Servers

| id | Upstream | Install | Needs | Last smoke (PASS / WARN / FAIL) |
|---|---|---|---|---|
| `pyscf` | `lixin19/mcp2pyscf` | uv-sync-frozen, Py 3.13 | — | 19 / 8 / 0, amd64 + aarch64, 2026-09-29 |
| `arxiv` | `mims-harvard/ToolUniverse` (SMCP) | uv-sync-frozen, Py 3.12 | network to arxiv.org | 16 / 8 / 0, amd64 + aarch64, 2026-09-30 |
| `jsbsim` | `flyintothesky/jsbsim-mcp` | uv-pip-pinned, Py 3.12 | — | 17 / 17 / 0, amd64 + aarch64, 2026-10-02 |
| `s4` | `prof-davifr/mcp-s4-rcwa` | uv-pip-pinned, Py 3.12 | x86-64 (AVX2/FMA/BMI2), `libblas3 liblapack3` | 41 / 7 / 0, amd64, 2026-10-02 |
| `psi4` | `Keith9922/chemaster` (`calc_psi4`) | conda-explicit | `micromamba` on `PATH` | 14 / 11 / 0, amd64 + aarch64, 2026-10-02 |

Install the prerequisites before running `setup.py`:

```sh
sudo apt-get install libblas3 liblapack3            # s4, admin, once
mkdir -p ~/.local/bin && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
  | tar -xj -C ~/.local bin/micromamba               # psi4
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
