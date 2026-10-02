"""Direct (agent-free) E2E smoke test for the pinned mcp2pyscf server.

Run with the server's own virtualenv so PySCF, RDKit and geomeTRIC are
importable for the independent reference calculations::

    ~/mcp/pyscf/.venv/bin/python scripts/mcp/e2e/smoke.py pyscf --config ~/mcp/pyscf.mcp.json

L1: real tools/call of every tool, with results checked against values
computed here, outside the server process (the shared L0/L1 checks are in
``e2e_smoke/runner.py``).

Upstream defects that do not make a result wrong (in-band errors, ignored
arguments, misleading return text, stdout pollution, intermittent symmetry
errors) are reported as WARN; wrong values are FAIL.
"""
from __future__ import annotations

import base64
import binascii
import json
import math
from pathlib import Path

from ..client import text_of
from ..helpers import Atoms, image_blocks, max_abs_diff, png_size, quiet_fds, sorted_distances
from ..runner import Caller, Report, Session, Smoke, check_rejected

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


def parse_float_list(text: str) -> list[float]:
    value = json.loads(text)
    if not isinstance(value, list) or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
        raise ValueError(f"expected a JSON list of numbers, got {text[:120]!r}")
    return [float(v) for v in value]


# --------------------------------------------------------------------------
# Independent references (computed in this process, not by the server)
# --------------------------------------------------------------------------

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
    check_rejected(call, f"{tool}[invalid input]", tool, arguments,
                   in_band=lambda result: repr(text_of(result)) if marker.lower() in text_of(result).lower() else None)


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
    check_rejected(call, bad, "plot_energy_scan_image_mcp", {"bond_lengths": lengths, "energies": energies[:2]},
                   on_accept=lambda result: ("WARN", "mismatched input lengths did not produce isError"))


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


def run_l1(session: Session) -> None:
    call, report = session.call, session.report
    check_energy(call, report, "H2@0.74", H2, "sto-3g")
    check_energy(call, report, "H2O", H2O, "sto-3g")
    check_energy(call, report, "H2O", H2O, "6-31g")
    check_geometry(call, report)
    check_geometry_to_energy_chain(call, report)
    check_bond_scan(call, report)
    check_optimize(call, report)
    check_scan_pes(call, report)
    check_plot(call, report)
    check_visualize(call, report, session.cwd)


SMOKE = Smoke(
    server="pyscf",
    run_l1=run_l1,
    packages=("pyscf", "rdkit", "geometric", "matplotlib", "mcp", "numpy"),
    extra_env={"MPLBACKEND": "Agg"},
    expected_cwd_files=(VISUALIZE_FILE,),   # checked by check_visualize
)
