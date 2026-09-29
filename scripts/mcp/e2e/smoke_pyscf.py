"""Direct (agent-free) E2E smoke test for the pinned mcp2pyscf server.

Run with the server's own virtualenv so PySCF is importable for the
independent reference calculation::

    ~/mcp/pyscf/.venv/bin/python scripts/mcp/e2e/smoke_pyscf.py --config ~/mcp/pyscf.mcp.json

Levels reported:
  L0  initialize + tools/list (stable, matches manifest)
  L1  real tools/call with results checked against PySCF computed here,
      outside the server process

The server runs from a temporary cwd/HOME with a minimal environment and no
operator credentials. Non-JSON lines on the server's stdout are reported as a
WARN: they corrupt the stdio transport for strict clients.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata as md
import json
import math
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from stdio_client import MCPError, StdioMCP, text_of  # noqa: E402

H2 = "H 0 0 0; H 0 0 0.74"
H2O = "O 0.000000 0.000000 0.117790; H 0.000000 0.755453 -0.471161; H 0.000000 -0.755453 -0.471161"
ENERGY_TOL = 1e-7  # Hartree; same code path and machine, should agree far tighter


def reference_rhf(atom: str, basis: str) -> float:
    """Independent RHF energy, mirroring the server's settings (symmetry=True)."""
    from pyscf import gto, scf
    mol = gto.M(atom=atom, basis=basis, symmetry=True, verbose=0)
    mf = scf.HF(mol)
    mf.verbose = 0
    energy = mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"reference SCF did not converge for {atom!r}")
    return float(energy)


class Report:
    def __init__(self) -> None:
        self.checks: list[dict] = []

    def add(self, level: str, name: str, status: str, detail: str = "", **data) -> None:
        self.checks.append({"level": level, "name": name, "status": status, "detail": detail, **data})
        print(f"[{status:<4}] {level} {name}" + (f" — {detail}" if detail else ""), flush=True)

    @property
    def failed(self) -> bool:
        return any(c["status"] == "FAIL" for c in self.checks)


def versions() -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ("pyscf", "rdkit", "geometric", "mcp", "numpy"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    return out


def check_energy(client: StdioMCP, report: Report, label: str, atom: str, basis: str) -> None:
    name = f"pyscf_rhf_energy[{label}/{basis}]"
    try:
        resp = client.call_tool("pyscf_rhf_energy", {"atom": atom, "basis": basis})
    except MCPError as exc:
        report.add("L1", name, "FAIL", str(exc))
        return
    if "error" in resp or resp["result"].get("isError"):
        report.add("L1", name, "FAIL", f"tool error: {resp.get('error') or text_of(resp['result'])[:300]}")
        return
    raw = text_of(resp["result"]).strip()
    try:
        got = float(raw)
    except ValueError:
        report.add("L1", name, "FAIL", f"non-numeric result: {raw[:200]!r}")
        return
    ref = reference_rhf(atom, basis)
    diff = abs(got - ref)
    ok = math.isfinite(got) and diff <= ENERGY_TOL
    report.add("L1", name, "PASS" if ok else "FAIL",
               f"server={got:.10f} reference={ref:.10f} |diff|={diff:.2e}",
               server_value=got, reference_value=ref, abs_diff=diff)


def check_geometry(client: StdioMCP, report: Report) -> None:
    name = "generate_pyscf_geom_input[O]"
    try:
        resp = client.call_tool("generate_pyscf_geom_input", {"smiles_string": "O"})
    except MCPError as exc:
        report.add("L1", name, "FAIL", str(exc))
        return
    if "error" in resp or resp["result"].get("isError"):
        report.add("L1", name, "FAIL", f"tool error: {resp.get('error') or text_of(resp['result'])[:300]}")
        return
    text = text_of(resp["result"]).strip()
    symbols = sorted(line.split()[0] for line in text.replace(";", "\n").splitlines() if line.strip())
    if symbols != ["H", "H", "O"]:
        report.add("L1", name, "FAIL", f"unexpected atoms {symbols}: {text[:200]!r}")
        return
    # Chain check: the returned string must be directly usable as PySCF input.
    try:
        from pyscf import gto
        gto.M(atom=text, basis="sto-3g", verbose=0)
    except Exception as exc:  # noqa: BLE001
        report.add("L1", name, "FAIL", f"output not parseable by PySCF: {exc}")
        return
    report.add("L1", name, "PASS", "3 atoms (O,H,H), parseable by PySCF; conformer is RDKit-random, not value-checked",
               output=text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="generated <root>/pyscf.mcp.json")
    parser.add_argument("--server", default="pyscf")
    parser.add_argument("--report", default="pyscf-smoke-report.json")
    args = parser.parse_args(argv)

    config = json.loads(Path(args.config).expanduser().read_text(encoding="utf-8"))
    server = config["mcpServers"][args.server]
    manifest = {e["id"]: e for e in json.loads((HERE / "manifest.json").read_text())["servers"]}
    entry = manifest[args.server]
    checkout = Path(server["cwd"])
    try:
        revision = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None

    report = Report()
    if revision != entry["revision"]:
        report.add("L0", "pinned revision", "FAIL", f"checkout at {revision}, manifest pins {entry['revision']}")

    with tempfile.TemporaryDirectory(prefix="mcp-e2e-pyscf-") as tmp:
        tmp_path = Path(tmp)
        venv_bin = str(Path(server["command"]).parent)
        env = {"HOME": str(tmp_path), "PATH": f"{venv_bin}:/usr/bin:/bin",
               "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1", "LANG": "C.UTF-8"}
        stderr_path = tmp_path / "server.stderr.log"
        client = StdioMCP(server["command"], server["args"], cwd=tmp_path, env=env, stderr_path=stderr_path)
        try:
            try:
                info = client.initialize()
                report.add("L0", "initialize", "PASS",
                           f"server={info.get('serverInfo')} protocol={info.get('protocolVersion')}")
                tools = client.list_tools()
                names = sorted(t["name"] for t in tools)
                expected = sorted(entry["expected_tools"])
                report.add("L0", "tools/list", "PASS" if names == expected else "FAIL",
                           f"{len(names)} tools" + ("" if names == expected else f"; expected {expected}, got {names}"))
                again = sorted(t["name"] for t in client.list_tools())
                report.add("L0", "tools/list stable", "PASS" if again == names else "FAIL")
            except MCPError as exc:
                report.add("L0", "handshake", "FAIL", str(exc))
            else:
                check_energy(client, report, "H2@0.74", H2, "sto-3g")
                check_energy(client, report, "H2O", H2O, "sto-3g")
                check_energy(client, report, "H2O", H2O, "6-31g")
                check_geometry(client, report)
                try:
                    bad = client.call_tool("__nonexistent__", {})
                    ok = "error" in bad or bad.get("result", {}).get("isError") is True
                    report.add("L1", "unknown tool is an error", "PASS" if ok else "FAIL",
                               "" if ok else f"got success: {bad}")
                except MCPError as exc:
                    report.add("L1", "unknown tool is an error", "FAIL", str(exc))
            alive = client.proc.poll() is None
            report.add("L1", "server alive after calls", "PASS" if alive else "FAIL",
                       "" if alive else f"exit code {client.proc.returncode}")
        finally:
            client.close()
            stderr_tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]

        polluted = client.non_json_stdout
        report.add("L1", "stdout is pure JSON-RPC", "WARN" if polluted else "PASS",
                   f"{len(polluted)} non-JSON line(s), e.g. {polluted[:3]}" if polluted else "",
                   non_json_stdout=polluted[:50])

    document = {
        "server": args.server,
        "repository": entry["repository"],
        "revision": revision,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "environment": versions(),
        "result": "FAIL" if report.failed else "PASS",
        "checks": report.checks,
        "server_stderr_tail": stderr_tail,
    }
    Path(args.report).write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n{document['result']}  report -> {Path(args.report).resolve()}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
