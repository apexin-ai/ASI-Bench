"""Direct (agent-free) E2E smoke test for the pinned mcp2pyscf server.

Run with the server's own virtualenv so PySCF, RDKit and geomeTRIC are
importable for the independent reference calculations::

    ~/mcp/pyscf/.venv/bin/python scripts/mcp/e2e/smoke_pyscf.py --config ~/mcp/pyscf.mcp.json

Levels reported:
  L0  initialize + tools/list (stable, matches manifest)
  L1  real tools/call of every tool, with results checked against values
      computed here, outside the server process

Upstream defects that do not make a result wrong (in-band errors, ignored
arguments, misleading return text, stdout pollution, intermittent symmetry
errors) are reported as WARN; wrong values are FAIL.

The server runs from a temporary cwd/HOME with a minimal environment and no
operator credentials. Non-JSON lines on the server's stdout are reported as a
WARN, attributed per tool: they corrupt the stdio transport for strict clients.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import contextlib
import datetime as dt
import importlib.metadata as md
import json
import math
import os
import platform
import struct
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from stdio_client import MCPError, StdioMCP, text_of  # noqa: E402

H2 = "H 0 0 0; H 0 0 0.74"
H2O = "O 0.000000 0.000000 0.117790; H 0.000000 0.755453 -0.471161; H 0.000000 -0.755453 -0.471161"
H2O_XYZ = "3\nwater\n" + "\n".join(part.strip() for part in H2O.split(";")) + "\n"

# Tolerances. Identical geometry in → same code path, so plain RHF agrees far
# tighter than ENERGY_TOL. Geometries built by the server come from an
# *unseeded* RDKit conformer followed by UFF; for rigid molecules the UFF
# minimum is unique up to rotation and H permutation (measured: interatomic
# distances within 4e-6 Å, RHF/STO-3G energies within 3e-7 Ha over repeated
# embeddings), so they are compared with looser, rotation-invariant checks.
ENERGY_TOL = 1e-7          # Ha
GEOM_ENERGY_TOL = 1e-5     # Ha, energies at server-built UFF geometries
DIST_TOL = 1e-3            # Å, sorted interatomic distances of UFF minima
OPT_DIST_TOL = 2e-3        # Å, HF/STO-3G minima (geomeTRIC default convergence)
OPT_ENERGY_TOL = 1e-6      # Ha, energy at the optimized geometry
RDKIT_SEED = 20260929

UFF_SMILES = ("O", "N", "C#N")          # rigid molecules only
SYMMETRY_PROBE_SMILES = "c1ccccc1"       # fails intermittently with symmetry=True
SYMMETRY_PROBE_TRIES = 8
# (SMILES, atom1, atom2, start Å, end Å, points); RDKit order: heavy atoms, then H.
BOND_SCANS = (("O", 0, 1, 0.8, 1.2, 5), ("C#N", 0, 1, 1.0, 1.3, 4))
OPT_SMILES = ("N", "O=C")
PES_POINTS = 9                           # scan_pes_rhf: H2, np.arange(0.7, 1.51, 0.1)
PLOT_SIZE = (1000, 600)                  # matplotlib figsize (10, 6) at 100 dpi
VISUALIZE_FILE = "optimized_geom_3d.html"


# --------------------------------------------------------------------------
# Pure helpers (stdlib only; unit-tested offline)
# --------------------------------------------------------------------------

Atoms = list[tuple[str, tuple[float, float, float]]]


def parse_atom_string(text: str) -> Atoms:
    """Parse a PySCF atom string ("O x y z; H x y z" or newline separated)."""
    atoms: Atoms = []
    for chunk in text.replace(";", "\n").splitlines():
        fields = chunk.split()
        if not fields:
            continue
        if len(fields) != 4:
            raise ValueError(f"expected 'symbol x y z', got {chunk.strip()!r}")
        atoms.append((fields[0], (float(fields[1]), float(fields[2]), float(fields[3]))))
    if not atoms:
        raise ValueError("no atoms")
    return atoms


def parse_xyz_block(text: str) -> Atoms:
    """Parse an XYZ block: atom count, comment line, then one atom per line."""
    lines = text.strip("\n").splitlines()
    if len(lines) < 3:
        raise ValueError("XYZ block too short")
    natm = int(lines[0].strip())
    body = [line for line in lines[2:] if line.strip()]
    if len(body) != natm:
        raise ValueError(f"XYZ header says {natm} atoms, found {len(body)}")
    return parse_atom_string("\n".join(body))


def symbols(atoms: Atoms) -> list[str]:
    return [sym for sym, _ in atoms]


def sorted_distances(atoms: Atoms) -> list[float]:
    """All interatomic distances, sorted: invariant to rotation, translation and permutation."""
    out = []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            out.append(math.dist(atoms[i][1], atoms[j][1]))
    return sorted(out)


def max_abs_diff(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return math.inf
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def parse_float_list(text: str) -> list[float]:
    value = json.loads(text)
    if not isinstance(value, list) or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
        raise ValueError(f"expected a JSON list of numbers, got {text[:120]!r}")
    return [float(v) for v in value]


def png_size(data: bytes) -> tuple[int, int]:
    """Width and height from a PNG IHDR chunk."""
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise ValueError("not a PNG")
    return struct.unpack(">II", data[16:24])


def image_blocks(result: dict) -> list[dict]:
    return [b for b in result.get("content", []) if b.get("type") == "image"]


# --------------------------------------------------------------------------
# Independent references (computed in this process, not by the server)
# --------------------------------------------------------------------------

@contextlib.contextmanager
def quiet_fds():
    """Silence fd 1/2 (geomeTRIC and PySCF log to the process streams directly)."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for fd in (*saved, devnull):
            os.close(fd)


def rhf_energy(atom, basis: str, symmetry: bool) -> float:
    from pyscf import gto, scf
    mol = gto.M(atom=atom, basis=basis, symmetry=symmetry, verbose=0)
    mf = scf.HF(mol)
    mf.verbose = 0
    energy = mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"reference SCF did not converge for {atom!r}/{basis}")
    return float(energy)


def reference_rhf(atom: str, basis: str) -> float:
    """Independent RHF energy, mirroring pyscf_rhf_energy (symmetry=True)."""
    return rhf_energy(atom, basis, symmetry=True)


def as_pyscf(atoms: Atoms) -> list:
    return [[sym, *xyz] for sym, xyz in atoms]


def uff_geometry(smiles: str) -> Atoms:
    """Seeded RDKit ETKDG conformer + UFF minimum (the server's recipe, unseeded there)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES {smiles!r}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDG()
    params.randomSeed = RDKIT_SEED
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise RuntimeError(f"RDKit embedding failed for {smiles!r}")
    AllChem.UFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    return [(a.GetSymbol(), tuple(float(v) for v in conf.GetAtomPosition(i))) for i, a in enumerate(mol.GetAtoms())]


def reference_bond_scan(smiles: str, i: int, j: int, start: float, end: float, n: int, basis: str):
    """Move atom j along the i→j vector of the UFF geometry; RHF (no symmetry) at each distance."""
    import numpy as np
    atoms = uff_geometry(smiles)
    lengths = np.linspace(start, end, n).tolist()
    energies = []
    for dist in lengths:
        coords = np.array([xyz for _, xyz in atoms])
        vec = coords[j] - coords[i]
        coords[j] = coords[i] + vec * (dist / np.linalg.norm(vec))
        energies.append(rhf_energy([[sym, *xyz] for (sym, _), xyz in zip(atoms, coords.tolist())], basis, symmetry=False))
    return lengths, energies


def reference_optimize(smiles: str) -> Atoms:
    """HF/STO-3G minimum with geomeTRIC, started from the seeded UFF geometry."""
    from pyscf import gto, scf
    from pyscf.geomopt.geometric_solver import optimize
    atoms = uff_geometry(smiles)
    mol = gto.M(atom=as_pyscf(atoms), basis="sto-3g", verbose=0)
    with quiet_fds():
        mf = scf.RHF(mol).run(verbose=0)
        opt = optimize(mf, maxsteps=100)
    coords = opt.atom_coords(unit="Angstrom")
    return [(opt.atom_symbol(k), tuple(float(v) for v in coords[k])) for k in range(opt.natm)]


def reference_pes_scan() -> list[float]:
    import numpy as np
    return [reference_rhf(f"H 0 0 0; H 0 0 {b}", "sto-3g") for b in np.arange(0.7, 1.51, 0.1)]


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

class Report:
    def __init__(self) -> None:
        self.checks: list[dict] = []

    def add(self, level: str, name: str, status: str, detail: str = "", **data) -> None:
        self.checks.append({"level": level, "name": name, "status": status, "detail": detail, **data})
        print(f"[{status:<4}] {level} {name}" + (f" — {detail}" if detail else ""), flush=True)

    @property
    def failed(self) -> bool:
        return any(c["status"] == "FAIL" for c in self.checks)


class Caller:
    """tools/call wrapper: records FAILs for transport/tool errors and stdout lines per tool."""

    def __init__(self, client: StdioMCP, report: Report) -> None:
        self.client = client
        self.report = report
        self.stdout_by_tool: Counter = Counter()

    def __call__(self, check: str, tool: str, arguments: dict, *, allow_error: bool = False) -> dict | None:
        before = len(self.client.non_json_stdout)
        try:
            resp = self.client.call_tool(tool, arguments)
        except MCPError as exc:
            self.report.add("L1", check, "FAIL", str(exc))
            return None
        finally:
            self.stdout_by_tool[tool] += len(self.client.non_json_stdout) - before
        if "error" in resp:
            self.report.add("L1", check, "FAIL", f"JSON-RPC error: {resp['error']}")
            return None
        result = resp["result"]
        if result.get("isError") and not allow_error:
            self.report.add("L1", check, "FAIL", f"tool error: {text_of(result)[:300]}")
            return None
        return result


def check_energy(call: Caller, report: Report, label: str, atom: str, basis: str) -> None:
    name = f"pyscf_rhf_energy[{label}/{basis}]"
    result = call(name, "pyscf_rhf_energy", {"atom": atom, "basis": basis})
    if result is None:
        return
    raw = text_of(result).strip()
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


def check_in_band_error(call: Caller, report: Report, tool: str, arguments: dict, marker: str) -> None:
    """Invalid input should give isError=true; an error string in a normal result is a WARN."""
    name = f"{tool}[invalid input]"
    result = call(name, tool, arguments, allow_error=True)
    if result is None:
        return
    text = text_of(result)
    if result.get("isError"):
        report.add("L1", name, "PASS", f"isError=true: {text[:120]!r}")
    elif marker.lower() in text.lower():
        report.add("L1", name, "WARN", f"error reported in-band with isError=false: {text[:160]!r}")
    else:
        report.add("L1", name, "FAIL", f"invalid input accepted: {text[:200]!r}")


def check_geometry(call: Caller, report: Report) -> None:
    for smiles in UFF_SMILES:
        name = f"generate_pyscf_geom_input[{smiles}]"
        result = call(name, "generate_pyscf_geom_input", {"smiles_string": smiles})
        if result is None:
            continue
        text = text_of(result).strip()
        try:
            got = parse_atom_string(text)
        except ValueError as exc:
            report.add("L1", name, "FAIL", f"unparseable output ({exc}): {text[:200]!r}")
            continue
        ref = uff_geometry(smiles)
        if symbols(got) != symbols(ref):
            report.add("L1", name, "FAIL", f"atom order {symbols(got)} != reference {symbols(ref)}")
            continue
        d_diff = max_abs_diff(sorted_distances(got), sorted_distances(ref))
        e_got = rhf_energy(as_pyscf(got), "sto-3g", symmetry=False)
        e_ref = rhf_energy(as_pyscf(ref), "sto-3g", symmetry=False)
        e_diff = abs(e_got - e_ref)
        ok = d_diff <= DIST_TOL and e_diff <= GEOM_ENERGY_TOL
        report.add("L1", name, "PASS" if ok else "FAIL",
                   f"atoms={''.join(symbols(got))} max|Δd|={d_diff:.1e} Å |ΔE(RHF/STO-3G)|={e_diff:.1e} Ha "
                   "vs seeded RDKit+UFF",
                   output=text, max_distance_diff=d_diff, energy_diff=e_diff)
    check_in_band_error(call, report, "generate_pyscf_geom_input", {"smiles_string": "not_a_smiles"}, "error")


def check_geometry_to_energy_chain(call: Caller, report: Report) -> None:
    """generate_pyscf_geom_input → pyscf_rhf_energy, the natural agent chain.

    pyscf_rhf_energy builds the molecule with symmetry=True; for high-symmetry
    molecules the UFF geometry is only approximately symmetric and PySCF
    intermittently raises PointGroupSymmetryError.
    """
    smiles = SYMMETRY_PROBE_SMILES
    name = f"generate_pyscf_geom_input→pyscf_rhf_energy[{smiles}]"
    failures, energies = [], []
    for _ in range(SYMMETRY_PROBE_TRIES):
        geom = call(name, "generate_pyscf_geom_input", {"smiles_string": smiles})
        if geom is None:
            return
        result = call(name, "pyscf_rhf_energy", {"atom": text_of(geom).strip(), "basis": "sto-3g"}, allow_error=True)
        if result is None:
            return
        if result.get("isError"):
            failures.append(text_of(result)[:200])
            continue
        try:
            energies.append(float(text_of(result).strip()))
        except ValueError:
            report.add("L1", name, "FAIL", f"non-numeric result: {text_of(result)[:200]!r}")
            return
    ref = rhf_energy(as_pyscf(uff_geometry(smiles)), "sto-3g", symmetry=False)
    worst = max((abs(e - ref) for e in energies), default=0.0)
    if worst > GEOM_ENERGY_TOL:
        report.add("L1", name, "FAIL", f"successful energies deviate from reference {ref:.8f} by {worst:.1e} Ha")
    elif failures:
        report.add("L1", name, "WARN",
                   f"{len(failures)}/{SYMMETRY_PROBE_TRIES} chained calls failed (symmetry=True on an unseeded "
                   f"UFF geometry); successful energies within {worst:.1e} Ha. e.g. {failures[0]!r}",
                   failures=failures)
    else:
        report.add("L1", name, "PASS",
                   f"{SYMMETRY_PROBE_TRIES}/{SYMMETRY_PROBE_TRIES} succeeded within {worst:.1e} Ha "
                   "(failure is intermittent; none observed this run)")


def _scan_result(report: Report, name: str, result: dict) -> dict | None:
    text = text_of(result)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        report.add("L1", name, "FAIL", f"result is not JSON: {text[:200]!r}")
        return None
    if not isinstance(data, dict):
        report.add("L1", name, "FAIL", f"expected a JSON object, got {text[:200]!r}")
        return None
    if data.get("error"):
        report.add("L1", name, "FAIL", f"in-band error: {data['error']}")
        return None
    return data


def check_bond_scan(call: Caller, report: Report) -> None:
    for smiles, i, j, start, end, n in BOND_SCANS:
        name = f"run_bond_stretch_calculation_mcp[{smiles} {i}-{j} {start}-{end}x{n}]"
        args = {"smiles_string": smiles, "atom1_idx": i, "atom2_idx": j,
                "start_dist": start, "end_dist": end, "num_points": n, "basis": "sto-3g"}
        result = call(name, "run_bond_stretch_calculation_mcp", args)
        if result is None:
            continue
        data = _scan_result(report, name, result)
        if data is None:
            continue
        ref_len, ref_e = reference_bond_scan(smiles, i, j, start, end, n, "sto-3g")
        len_diff = max_abs_diff(data.get("bond_lengths", []), ref_len)
        e_diff = max_abs_diff(data.get("energies", []), ref_e)
        ok = len_diff <= 1e-9 and e_diff <= GEOM_ENERGY_TOL
        report.add("L1", name, "PASS" if ok else "FAIL",
                   f"{len(ref_e)} points, max|ΔE|={e_diff:.1e} Ha, max|Δr|={len_diff:.1e} Å",
                   server_value=data, reference_energies=ref_e, max_energy_diff=e_diff)

    # The tool accepts a basis argument; check whether it is honoured.
    name = "run_bond_stretch_calculation_mcp[basis=6-31g]"
    smiles, i, j, start, end, n = "O", 0, 1, 0.9, 1.1, 3
    args = {"smiles_string": smiles, "atom1_idx": i, "atom2_idx": j,
            "start_dist": start, "end_dist": end, "num_points": n, "basis": "6-31g"}
    result = call(name, "run_bond_stretch_calculation_mcp", args)
    data = _scan_result(report, name, result) if result is not None else None
    if data is not None:
        got = data.get("energies", [])
        d_631 = max_abs_diff(got, reference_bond_scan(smiles, i, j, start, end, n, "6-31g")[1])
        d_sto = max_abs_diff(got, reference_bond_scan(smiles, i, j, start, end, n, "sto-3g")[1])
        if d_631 <= GEOM_ENERGY_TOL:
            report.add("L1", name, "PASS", f"6-31G energies (max|ΔE|={d_631:.1e} Ha)")
        elif d_sto <= GEOM_ENERGY_TOL:
            report.add("L1", name, "WARN", "basis argument ignored: returned STO-3G energies "
                       f"(max|ΔE| vs STO-3G {d_sto:.1e}, vs 6-31G {d_631:.1e} Ha)")
        else:
            report.add("L1", name, "FAIL", f"matches neither basis (6-31G {d_631:.1e}, STO-3G {d_sto:.1e} Ha)")

    check_in_band_error(call, report, "run_bond_stretch_calculation_mcp",
                        {"smiles_string": "not_a_smiles", "atom1_idx": 0, "atom2_idx": 1,
                         "start_dist": 0.9, "end_dist": 1.1, "num_points": 3}, "error")


def check_optimize(call: Caller, report: Report) -> None:
    returns_energy = None
    for smiles in OPT_SMILES:
        name = f"optimize_molecule_mcp[{smiles}]"
        result = call(name, "optimize_molecule_mcp", {"smiles_string": smiles})
        if result is None:
            continue
        text = text_of(result)
        try:
            got = parse_xyz_block(text)
        except ValueError as exc:
            report.add("L1", name, "FAIL", f"not an XYZ block ({exc}): {text[:200]!r}")
            continue
        ref = reference_optimize(smiles)
        if symbols(got) != symbols(ref):
            report.add("L1", name, "FAIL", f"atom order {symbols(got)} != reference {symbols(ref)}")
            continue
        d_diff = max_abs_diff(sorted_distances(got), sorted_distances(ref))
        e_got = rhf_energy(as_pyscf(got), "sto-3g", symmetry=False)
        e_ref = rhf_energy(as_pyscf(ref), "sto-3g", symmetry=False)
        e_diff = abs(e_got - e_ref)
        ok = d_diff <= OPT_DIST_TOL and e_diff <= OPT_ENERGY_TOL
        report.add("L1", name, "PASS" if ok else "FAIL",
                   f"max|Δd|={d_diff:.1e} Å, E(RHF/STO-3G) at server geometry={e_got:.8f} "
                   f"reference minimum={e_ref:.8f} |ΔE|={e_diff:.1e} Ha",
                   output=text, max_distance_diff=d_diff, energy=e_got, reference_energy=e_ref)
        returns_energy = "energy" in text.lower()
    if returns_energy is False:
        report.add("L1", "optimize_molecule_mcp[result fields]", "WARN",
                   "returns only an XYZ block; the documented optimized_energy is missing, so agents must "
                   "compute the energy with another call")
    check_in_band_error(call, report, "optimize_molecule_mcp", {"smiles_string": "not_a_smiles"}, "invalid")


def check_scan_pes(call: Caller, report: Report) -> None:
    name = "scan_pes_rhf[H2/STO-3G 0.7-1.5]"
    result = call(name, "scan_pes_rhf", {})
    if result is None:
        return
    text = text_of(result).strip()
    try:
        got = parse_float_list(text)
    except ValueError as exc:
        report.add("L1", name, "FAIL", str(exc))
        return
    ref = reference_pes_scan()
    diff = max_abs_diff(got, ref)
    ok = len(got) == PES_POINTS and diff <= ENERGY_TOL
    report.add("L1", name, "PASS" if ok else "FAIL",
               f"{len(got)} points (expected {PES_POINTS}), max|ΔE|={diff:.1e} Ha; molecule and grid are "
               "hard-coded upstream", server_value=got, reference_value=ref)


def check_plot(call: Caller, report: Report) -> None:
    name = "plot_energy_scan_image_mcp"
    lengths = [0.7, 0.8, 0.9, 1.0]
    energies = [-1.1173, -1.1109, -1.0919, -1.0661]
    result = call(name, "plot_energy_scan_image_mcp", {"bond_lengths": lengths, "energies": energies})
    if result is not None:
        images = image_blocks(result)
        if len(images) != 1:
            report.add("L1", name, "FAIL", f"expected one image block, got {[b.get('type') for b in result.get('content', [])]}")
        elif images[0].get("mimeType") != "image/png":
            report.add("L1", name, "FAIL", f"mimeType {images[0].get('mimeType')!r}, expected image/png")
        else:
            try:
                size = png_size(base64.b64decode(images[0].get("data") or "", validate=True))
            except (ValueError, binascii.Error) as exc:
                report.add("L1", name, "FAIL", f"image data is not a base64 PNG: {exc}")
            else:
                report.add("L1", name, "PASS",
                           f"image/png {size[0]}x{size[1]}" + ("" if size == PLOT_SIZE else f" (expected {PLOT_SIZE})"),
                           image_size=list(size))
    bad = f"{name}[mismatched lengths]"
    result = call(bad, "plot_energy_scan_image_mcp", {"bond_lengths": lengths, "energies": energies[:2]},
                  allow_error=True)
    if result is not None:
        if result.get("isError"):
            report.add("L1", bad, "PASS", f"isError=true: {text_of(result)[:120]!r}")
        else:
            report.add("L1", bad, "WARN", "mismatched input lengths did not produce isError")


def check_visualize(call: Caller, report: Report, server_cwd: Path) -> None:
    name = "visualize_molecule_3d_mcp"
    result = call(name, "visualize_molecule_3d_mcp", {"xyz_string": H2O_XYZ, "title": "smoke"})
    if result is None:
        return
    text = text_of(result)
    html_path = server_cwd / VISUALIZE_FILE
    html = html_path.read_text(encoding="utf-8") if html_path.is_file() else ""
    coords_line = H2O_XYZ.splitlines()[2]
    ok = "3Dmol" in html and coords_line in html
    report.add("L1", f"{name}[HTML written]", "PASS" if ok else "FAIL",
               f"{VISUALIZE_FILE} in server cwd " + ("contains the 3Dmol viewer and the geometry" if ok
                                                     else ("lacks viewer/geometry" if html else "missing")))
    if str(html_path) in text:
        report.add("L1", f"{name}[result locates output]", "PASS", text[:160])
    else:
        report.add("L1", f"{name}[result locates output]", "WARN",
                   "HTML is written into the server cwd (the MCP checkout in agent runs) and never returned; "
                   f"the result text names a different path: {text[:160]!r}")


def versions() -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ("pyscf", "rdkit", "geometric", "matplotlib", "mcp", "numpy"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    return out


def run_l1(client: StdioMCP, report: Report, server_cwd: Path) -> Caller:
    call = Caller(client, report)
    check_energy(call, report, "H2@0.74", H2, "sto-3g")
    check_energy(call, report, "H2O", H2O, "sto-3g")
    check_energy(call, report, "H2O", H2O, "6-31g")
    check_geometry(call, report)
    check_geometry_to_energy_chain(call, report)
    check_bond_scan(call, report)
    check_optimize(call, report)
    check_scan_pes(call, report)
    check_plot(call, report)
    check_visualize(call, report, server_cwd)
    try:
        bad = client.call_tool("__nonexistent__", {})
        ok = "error" in bad or bad.get("result", {}).get("isError") is True
        report.add("L1", "unknown tool is an error", "PASS" if ok else "FAIL",
                   "" if ok else f"got success: {bad}")
    except MCPError as exc:
        report.add("L1", "unknown tool is an error", "FAIL", str(exc))
    return call


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

    call = None
    with tempfile.TemporaryDirectory(prefix="mcp-e2e-pyscf-") as tmp:
        tmp_path = Path(tmp)
        venv_bin = str(Path(server["command"]).parent)
        env = {"HOME": str(tmp_path), "PATH": f"{venv_bin}:/usr/bin:/bin",
               "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1", "LANG": "C.UTF-8", "MPLBACKEND": "Agg"}
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
                call = run_l1(client, report, tmp_path)
            alive = client.proc.poll() is None
            report.add("L1", "server alive after calls", "PASS" if alive else "FAIL",
                       "" if alive else f"exit code {client.proc.returncode}")
        finally:
            client.close()
            stderr_tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]

        polluted = client.non_json_stdout
        per_tool = {k: v for k, v in sorted(call.stdout_by_tool.items()) if v} if call else {}
        report.add("L1", "stdout is pure JSON-RPC", "WARN" if polluted else "PASS",
                   (f"{len(polluted)} non-JSON line(s), per tool {per_tool}, e.g. {polluted[:3]}"
                    if polluted else ""),
                   non_json_stdout=polluted[:50], non_json_stdout_by_tool=per_tool)

    document = {
        "server": args.server,
        "repository": entry["repository"],
        "revision": revision,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "environment": versions(),
        "result": "FAIL" if report.failed else "PASS",
        "summary": dict(Counter(c["status"] for c in report.checks)),
        "checks": report.checks,
        "server_stderr_tail": stderr_tail,
    }
    Path(args.report).write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n{document['result']} {document['summary']}  report -> {Path(args.report).resolve()}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
