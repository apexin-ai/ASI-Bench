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
  version, launch command relative to the checkout, expected tool names.
- `setup.py` — stdlib only. Clones into `<root>/<id>`, detaches at the pinned
  revision (refuses dirty checkouts or foreign remotes), runs
  `uv sync --frozen --python <manifest python>` (ignoring the caller's
  `UV_PYTHON`), and writes `<root>/<id>.mcp.json` for `--mcp-config`.
- `stdio_client.py` — stdlib-only JSON-RPC stdio client. It reads raw server
  stdout so non-JSON lines are **recorded**, not swallowed by an SDK.
- `smoke_<id>.py` — per-server L0/L1 checks; writes a JSON report with versions,
  revision, every check and the server's stderr tail.
- `verify_run.py` — L2 verifier for agent runs of `examples/mcp-e2e-tasks`:
  reads Claude Code stream-json or Codex `exec --json` JSONL (or the
  trajectory) and the persisted outputs
  and checks connection, every required tool call and its result (including
  image results), chained inputs between calls, answer provenance and bypass.

Verified 2026-09-29: the original pyscf smoke (`pyscf_rhf_energy` +
`generate_pyscf_geom_input`) passed on AWS Linux amd64 (Ubuntu 26.04) and Linux
aarch64. The all-tools smoke below passed on both with identical values
(19 PASS, 8 WARN, 0 FAIL, ~4 s).

## Run (Linux, as the unprivileged E2E user)

```sh
cd ~/ASI-Bench
python3 scripts/mcp/e2e/setup.py pyscf --root ~/mcp
~/mcp/pyscf/.venv/bin/python scripts/mcp/e2e/smoke_pyscf.py \
  --config ~/mcp/pyscf.mcp.json --report ~/mcp/pyscf-smoke-report.json
uv run asibench mcp check --config ~/mcp/pyscf.mcp.json
```

The smoke script must run with the server's own virtualenv so that its
reference calculation can import the same scientific library. The server is
launched from a temporary cwd/HOME with a minimal environment and no operator
credentials. Exit code is non-zero if any check FAILs; WARNs do not fail.

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
