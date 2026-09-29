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
  reads Claude Code stream-json (or the trajectory) and the persisted outputs
  and checks connection, tool call, tool result, answer provenance and bypass.

Verified 2026-09-29 on AWS Linux amd64 (Ubuntu 26.04) and Linux aarch64: pyscf
smoke all PASS with the stdout WARN below.

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

- Upstream requires Python ≥ 3.13 (`.python-version` 3.13); the lockfile pins
  PySCF 2.9.0, RDKit 2025.3.3, geomeTRIC 1.1, MCP SDK 1.9.4.
- L1 checks `pyscf_rhf_energy` for H₂/STO-3G, H₂O/STO-3G and H₂O/6-31G against
  PySCF RHF with identical settings (`symmetry=True`), tolerance 1e-7 Ha, and
  checks that `generate_pyscf_geom_input("O")` returns O,H,H coordinates that
  PySCF can parse. RDKit embedding is unseeded (random conformer), so geometry
  values are not compared.
- **Known WARN:** PySCF logs `converged SCF energy = ...` to **stdout** at its
  default verbosity, so every RHF call injects a non-JSON line into the stdio
  transport. Whether a given agent client tolerates this must be verified in
  the agent E2E run.
- Not covered by this smoke: `scan_pes_rhf` (fixed H₂ demo that also `print`s
  to stdout), bond-stretch scans, geometry optimisation, plotting and
  `visualize_molecule_3d_mcp` (writes into cwd and reports a hard-coded
  developer path).
