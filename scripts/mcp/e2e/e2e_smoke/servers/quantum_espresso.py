"""Direct (agent-free) E2E smoke test for the pinned ``frimpsjoek/qe-mcp`` MCP server.

Run with the server's own conda prefix (it provides the Quantum ESPRESSO
binaries and ASE that the references need)::

    ~/mcp/quantum_espresso/.venv/bin/python scripts/mcp/e2e/smoke.py quantum_espresso \
        --config ~/mcp/quantum_espresso.mcp.json

No network is needed: the 219 SG15 ONCV ``.upf`` files are vendored in the
pinned revision, so nothing is downloaded and ``scripts/download_pseudos.py``
is never run. The two Materials Project tools are called without
``MP_API_KEY`` and answer from the missing key alone, before any request.

L0 (on top of the shared checks in ``e2e_smoke/runner.py``: pinned revision,
config equals what ``setup.py`` renders, handshake, the 19 expected tools,
stable ``tools/list``)

* ``pw.x``, ``bands.x``, ``dos.x`` and ``projwfc.x`` exist in the prefix the
  manifest points ``QE_PREFIX`` at, are executable, and each prints the
  QE version pinned by ``conda.specs`` (``qe=7.5``) in its own startup banner,
  measured outside the server with an empty input file;
* the vendored SG15 library holds the expected number of ``.upf`` files, and
  the element index derived from their names (order-independently, see D1)
  is the reference for ``qe_status`` and ``qe_list_pseudopotentials``.

L1 — all 19 tools, on bulk silicon (two atoms, 30/120 Ry, a 4×4×4 grid, a
40-point band path): a few seconds of single-threaded DFT per call.

* Ten tools are pure Python (structures, k-points, the pseudopotential index,
  file readers, ``qe_status``). Their references are our own: the silicon cell
  written out by hand, the k-grid rule re-implemented, the pseudopotential
  directory scanned again, synthetic band/DOS/PDOS files whose contents we
  know, and the server's own output directories listed by us.
* Six tools run ``pw.x``/``bands.x``/``dos.x``. The references run the same
  executables **outside the server**: our own ``pw.x`` input (written from the
  settings the server documents — cold smearing, ``degauss`` 0.02,
  ``conv_thr`` 1e-6, ``mixing_beta`` 0.7, BFGS thresholds, ``nbnd = 8·nat``,
  ``nosym`` on the band path, tetrahedra on the DOS grid — with the
  pseudopotential file the server reports it uses), our own subprocess call,
  our own parsers of the text output. Same binary, same settings, one thread:
  the results agree to the printed digit, so tolerances are tight.
* Three tools depend on the environment: ``qe_get_job_status`` (only
  meaningful for the Globus runner) and the two Materials Project tools
  (need ``MP_API_KEY``). Each must answer with its exact in-band error.

Classification: a wrong number is FAIL. Errors returned in-band
(``isError: false`` with ``success: false``) are WARN, as for every server.
Upstream defects are three-state — the correct answer PASS, an answer that is
exactly the defect WARN, anything else FAIL:

* **D1** ``SG15Library._scan_library`` iterates ``glob("*.upf")`` and lets a
  later non-``_FR`` file overwrite an earlier one, so which version an element
  gets follows the host's directory order. The smoke re-runs that scan in
  directory order (must match the server exactly) and compares the pick with
  the newest version (WARN when they differ, naming the elements); the DFT
  references then use the file the server picked, so the physics comparison
  stays exact on every host.
* **D2** ``qe_read_bands`` / ``qe_read_dos`` name their parameter
  ``output_dir`` but want a file: a directory gives ``[Errno 21] Is a
  directory``.
* **D10** ``qe_get_kpath`` always fails: it sorts the special points by their
  coordinate arrays (``The truth value of an array ... is ambiguous``).
* **D11** ``qe_run_relax`` / ``qe_run_vc_relax`` report the energy, Fermi
  level, forces and stress of the **first** SCF step (``re.search`` finds the
  first ``!``), not of the relaxed structure.
* **D12** ``qe_workflow_relax_and_scf`` runs its final SCF on the **input**
  geometry (upstream comment: "Uses original structure"), so its
  ``total_energy`` is the unrelaxed one, at ``conv_thr`` 1e-8.
* **D13** ``forces_eV_per_angstrom`` holds 7×nat rows: with ``verbosity =
  'high'`` pw.x prints six contribution blocks after the total forces, and the
  parser collects them all. The first nat rows are the forces.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import time
from pathlib import Path

from ..client import text_of
from ..runner import Caller, Session, Smoke, check_rejected

# The QE executables the server can invoke (config.py's exe_map), and the one
# version string all of them must report: the manifest's conda spec qe=<version>.
QE_EXECUTABLES = ("pw.x", "bands.x", "dos.x", "projwfc.x")
# "     Program PWSCF v.7.5 starts on  3Oct2026 at 13:39:10"
BANNER_RE = re.compile(r"^\s*Program\s+(?P<program>\S+)\s+v\.(?P<version>\S+)\s+starts", re.MULTILINE)
BANNER_TIMEOUT = 60.0        # seconds; a banner appears in well under a second

# The vendored SG15 ONCV library of the pinned revision.
PSEUDO_SUBDIR = "pseudopotentials/sg15_oncv"
EXPECTED_UPF_FILES = 219
# SG15Library's own naming pattern (pseudopotentials.py), applied to every file
# instead of to the first one glob happens to return.
UPF_RE = re.compile(r"^([A-Z][a-z]?)_ONCV_PBE.*\.upf$", re.IGNORECASE)
UPF_VERSION_RE = re.compile(r"-(\d+(?:\.\d+)*)\.upf$", re.IGNORECASE)

# --------------------------------------------------------------------------
# The L1 workload
# --------------------------------------------------------------------------

STRUCTURE = "Si"             # the server's built-in: bulk('Si', 'diamond', a=5.43), primitive fcc
SI_A = 5.43                  # Å
FCC_CELL = ((0.0, 2.715, 2.715), (2.715, 0.0, 2.715), (2.715, 2.715, 0.0))
SI_ATOMS = (("Si", (0.0, 0.0, 0.0)), ("Si", (1.3575, 1.3575, 1.3575)))
# The second atom pushed along x (1.3575 -> 1.42 Å), so a relaxation has work to do.
PERTURBED_ATOMS = (("Si", (0.0, 0.0, 0.0)), ("Si", (1.42, 1.3575, 1.3575)))
# Two atoms 0.3 Å apart in a cubic box: qe_validate_structure must refuse it.
OVERLAP_CELL = ((5.43, 0.0, 0.0), (0.0, 5.43, 0.0), (0.0, 0.0, 5.43))
OVERLAP_ATOMS = (("Si", (0.0, 0.0, 0.0)), ("Si", (0.3, 0.0, 0.0)))
NOT_A_STRUCTURE = "definitely not a structure"

# Si's entry in the server's SG15 cutoff-hint table (Ry). ecutrho is not on the
# MCP surface: the server always takes it from the hint table.
SI_ECUTWFC, SI_ECUTRHO = 30, 120
KGRID = (4, 4, 4)
KPOINTS_ARG = "4,4,4"        # passed explicitly to every DFT tool
NPOINTS_BAND = 40            # band path points (server default 100)
DOS_KGRID = tuple(2 * k for k in KGRID)   # workflow_dos: NSCF grid = 2 x an explicit SCF grid
DELTAE = 0.01                # dos.x step (eV), server default
NBND = 8 * len(SI_ATOMS)     # workflow_bandstructure: nbnd = 8 * natoms
TIGHT_CONV_THR = 1e-8        # workflow_relax_and_scf's final SCF
KGRID_FACTORS = {"low": 25, "medium": 40, "high": 60}   # k * |a_i| targets (structures.py)
PREFERRED_KPTS = (1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21)
KSPACING = 0.05

# The server's documented calculation settings (input_generator.py and
# calculations.py defaults), written into our own inputs.
SMEARING = {"occupations": "smearing", "smearing": "cold", "degauss": 0.02}
ELECTRONS = {"conv_thr": 1e-6, "mixing_beta": 0.7, "electron_maxstep": 100}
RELAX_CONTROL = {"nstep": 100, "forc_conv_thr": 1e-3, "etot_conv_thr": 1e-4}
CELL_NAMELIST = {"cell_dynamics": "bfgs", "press_conv_thr": 0.5, "cell_dofree": "all"}

# CODATA 2018, as in QE 7.5's constants.f90.
RY_EV = 13.605693122994
BOHR_A = 0.529177210903
RYBOHR_TO_EVA = RY_EV / BOHR_A

# Tolerances: same binary, same settings, one thread — QE prints energies to
# 1e-8 Ry, the Fermi level to 1e-4 eV, eigenvalues to 1e-3 eV and repeated runs
# agree to the last digit, so these only absorb the unit conversions. The
# defects below move the compared numbers by ~1e-3 Ry or more.
ENERGY_TOL_RY = 1e-6
ENERGY_TOL_EV = ENERGY_TOL_RY * RY_EV
FERMI_TOL = 1e-6             # eV
FORCE_TOL = 1e-5             # eV/Å (the server converts with an older Bohr radius, ~7e-8 relative)
STRESS_TOL = 1e-6            # kbar, printed with two decimals
EIG_TOL = 1e-6               # eV
ARRAY_TOL = 1e-9             # numbers both sides parse from identical text
GEOM_TOL = 1e-9              # Å
CELL_TOL = 1e-8              # Å, final cells printed with nine decimals

CALL_TIMEOUT = 1800.0        # seconds per tools/call; each call here is seconds on amd64 and aarch64
REF_TIMEOUT = 1800.0         # seconds per reference executable

WORKFLOW_ID_RE = {kind: re.compile(rf"{kind}_[0-9a-f]{{8}}")
                  for kind in ("scf", "relax", "vc-relax", "bands", "dos", "relax_scf")}
MISSING_JOB = "scf_00000000"

# Wall-clock seconds per step, written to the report: what an L2 task has to budget.
TIMINGS: dict[str, float] = {}


def progress(what: str) -> None:
    print(f"[ .. ] L1 {what}", flush=True)


def timed(label: str, call, *args, **kwargs):
    started = time.monotonic()
    try:
        return call(*args, **kwargs)
    finally:
        TIMINGS[label] = round(time.monotonic() - started, 2)


# --------------------------------------------------------------------------
# Pure helpers (stdlib only; unit-tested offline)
# --------------------------------------------------------------------------

def spec_version(entry: dict, package: str) -> str | None:
    """The version ``conda.specs`` pins for ``package``, e.g. 'qe=7.5' -> '7.5'."""
    prefix = f"{package}="
    for spec in entry["conda"]["specs"]:
        if spec.startswith(prefix):
            return spec[len(prefix):]
    return None


def banner_version(output: str, program: str) -> str | None:
    """The version from a QE startup banner, if it is the banner of ``program``."""
    match = BANNER_RE.search(output)
    if match is None or match["program"].upper() != program.upper():
        return None
    return match["version"]


def upf_elements(filenames: list[str]) -> list[str]:
    """The element index SG15Library builds, derived order-independently from the file names.

    Which *file* an element maps to depends on the host's directory order
    (D1); which *elements* exist does not.
    """
    return sorted({match.group(1).capitalize() for name in filenames
                   if (match := UPF_RE.match(name))})


def program_name(executable: str) -> str:
    """The banner program name of a QE executable: pw.x -> PWSCF, dos.x -> DOS."""
    stem = executable.removesuffix(".x").upper()
    return "PWSCF" if stem == "PW" else stem


def scan_pick(filenames_in_directory_order: list[str]) -> dict[str, str]:
    """D1 as specified: the file SG15Library indexes per element when its glob yields
    the names in this order (a later non-``_FR`` file replaces an earlier entry; an
    ``_FR`` file only fills an empty slot)."""
    picked: dict[str, str] = {}
    for name in filenames_in_directory_order:
        match = UPF_RE.match(name)
        if match:
            element = match.group(1).capitalize()
            if element not in picked or "_FR" not in name:
                picked[element] = name
    return picked


def upf_version(name: str) -> tuple[int, ...]:
    match = UPF_VERSION_RE.search(name)
    return tuple(int(p) for p in match.group(1).split(".")) if match else ()


def newest_pick(filenames: list[str]) -> dict[str, str]:
    """Per element the newest scalar-relativistic file (an ``_FR`` one only if nothing else
    exists): what an order-independent index would choose."""
    best: dict[str, str] = {}
    for name in sorted(filenames):
        match = UPF_RE.match(name)
        if not match:
            continue
        element = match.group(1).capitalize()
        key = ("_FR" not in name, upf_version(name))
        current = best.get(element)
        if current is None or key > ("_FR" not in current, upf_version(current)):
            best[element] = name
    return best


def snap_odd(n: int) -> int:
    """Nearest value of :data:`PREFERRED_KPTS`, the larger one on a tie."""
    if n <= 1:
        return 1
    if n >= PREFERRED_KPTS[-1]:
        return PREFERRED_KPTS[-1]
    return min(PREFERRED_KPTS, key=lambda p: (abs(n - p), -p))


def kgrid_reference(lengths: list[float], pbc: list[bool], *, density: str = "medium",
                    kspacing: float | None = None) -> list[int]:
    """The documented k-grid rule: ``round(factor/|a_i|)`` (or ``ceil(1/(|a_i|·kspacing))``)
    snapped to an odd number, 1 along non-periodic axes."""
    grid = []
    for length, periodic in zip(lengths, pbc):
        if not periodic:
            grid.append(1)
            continue
        if kspacing is not None:
            n = max(1, math.ceil(1.0 / (length * kspacing)))
        else:
            n = max(1, round(KGRID_FACTORS.get(density, KGRID_FACTORS["medium"]) / length))
        grid.append(snap_odd(n))
    return grid


def vector_lengths(cell) -> list[float]:
    return [math.sqrt(sum(c * c for c in row)) for row in cell]


def cell_volume(cell) -> float:
    (a, b, c), (d, e, f), (g, h, i) = cell
    return abs(a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g))


def min_image_distance(atoms, cell) -> float:
    """Shortest interatomic distance over the 27 neighbouring images (small cells only)."""
    best = math.inf
    images = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
    for a in range(len(atoms)):
        for b in range(a + 1, len(atoms)):
            pa, pb = atoms[a][1], atoms[b][1]
            for i, j, k in images:
                shift = [i * cell[0][x] + j * cell[1][x] + k * cell[2][x] for x in range(3)]
                best = min(best, math.dist(pa, [pb[x] + shift[x] for x in range(3)]))
    return best


def chemical_formula(symbols: list[str]) -> str:
    """ASE's default ('hill' for these elements) formula of a list of symbols."""
    counts: dict[str, int] = {}
    for s in symbols:
        counts[s] = counts.get(s, 0) + 1
    return "".join(f"{s}{n if n > 1 else ''}" for s, n in sorted(counts.items()))


def inline_structure(atoms, cell) -> str:
    """The server's inline format: ``xyz:Si 0 0 0; Si ...|lattice:a1,a2,a3,b1,...``."""
    xyz = "; ".join(f"{s} {' '.join(repr(float(c)) for c in pos)}" for s, pos in atoms)
    return f"xyz:{xyz}|lattice:{','.join(repr(float(c)) for row in cell for c in row)}"


def poscar(atoms, cell, comment: str = "smoke") -> str:
    """A VASP POSCAR (Cartesian, one species block per element in order of appearance)."""
    species: list[str] = []
    for s, _ in atoms:
        if s not in species:
            species.append(s)
    lines = [comment, "1.0", *(" ".join(repr(float(c)) for c in row) for row in cell),
             " ".join(species), " ".join(str(sum(1 for s, _ in atoms if s == sp)) for sp in species),
             "Cartesian"]
    for sp in species:
        lines += [" ".join(repr(float(c)) for c in pos) for s, pos in atoms if s == sp]
    return "\n".join(lines) + "\n"


def fortran(value) -> str:
    if isinstance(value, bool):
        return ".true." if value else ".false."
    if isinstance(value, str):
        return f"'{value}'"
    return repr(value)


def namelist(name: str, entries: dict) -> str:
    return "\n".join([f"&{name}", *(f"  {key} = {fortran(value)}" for key, value in entries.items()), "/"])


def pw_input(calculation: str, atoms, cell, pseudo_file: str, *, prefix: str = "ref",
             occupations: dict | None = None, electrons: dict | None = None, system: dict | None = None,
             kgrid=None, kpoints_crystal=None) -> str:
    """Our own ``pw.x`` input for ``calculation`` (one species, Ångström cell and positions)."""
    control = {"calculation": calculation, "prefix": prefix, "outdir": "./out", "pseudo_dir": "./pseudo",
               "verbosity": "high", "tprnfor": True, "tstress": True}
    if calculation in ("relax", "vc-relax"):
        control.update(RELAX_CONTROL)
    symbols = sorted({s for s, _ in atoms})
    sysnl = {"ibrav": 0, "nat": len(atoms), "ntyp": len(symbols), "ecutwfc": float(SI_ECUTWFC),
             "ecutrho": float(SI_ECUTRHO), **(SMEARING if occupations is None else occupations),
             **(system or {})}
    blocks = [namelist("CONTROL", control), namelist("SYSTEM", sysnl),
              namelist("ELECTRONS", {**ELECTRONS, **(electrons or {})})]
    if calculation in ("relax", "vc-relax"):
        blocks.append(namelist("IONS", {"ion_dynamics": "bfgs"}))
    if calculation == "vc-relax":
        blocks.append(namelist("CELL", CELL_NAMELIST))
    blocks.append("ATOMIC_SPECIES\n" + "\n".join(f"  {s} 1.0 {pseudo_file}" for s in symbols))
    blocks.append("CELL_PARAMETERS angstrom\n" + "\n".join("  " + " ".join(repr(float(c)) for c in row)
                                                           for row in cell))
    blocks.append("ATOMIC_POSITIONS angstrom\n" + "\n".join(
        f"  {s} " + " ".join(repr(float(c)) for c in pos) for s, pos in atoms))
    if kpoints_crystal is not None:
        # %.10f: the precision the server writes its band path with, so both runs
        # see the same k-points (repr would differ in the 11th decimal).
        blocks.append(f"K_POINTS crystal\n  {len(kpoints_crystal)}\n" + "\n".join(
            "  " + " ".join(f"{c:.10f}" for c in k) + " 1.0" for k in kpoints_crystal))
    else:
        blocks.append("K_POINTS automatic\n  " + " ".join(str(k) for k in kgrid) + " 0 0 0")
    return "\n\n".join(blocks) + "\n"


def bands_x_input(prefix: str = "ref") -> str:
    return namelist("BANDS", {"prefix": prefix, "outdir": "./out", "filband": "bands.dat"}) + "\n"


def dos_x_input(prefix: str = "ref") -> str:
    return namelist("DOS", {"prefix": prefix, "outdir": "./out", "fildos": "dos.dat", "deltae": DELTAE}) + "\n"


_NUM = r"(-?\d+\.\d+)"
ENERGY_LINE_RE = re.compile(rf"^!\s+total energy\s+=\s+{_NUM}\s+Ry", re.MULTILINE)
FINAL_LINE_RE = re.compile(rf"^\s*Final (?:energy|enthalpy)\s+=\s+{_NUM}\s+Ry", re.MULTILINE)
FERMI_LINE_RE = re.compile(rf"the Fermi energy is\s+{_NUM}\s+ev")
ITERATIONS_RE = re.compile(r"convergence has been achieved in\s+(\d+)\s+iterations")
TOTAL_FORCE_RE = re.compile(rf"Total force =\s+{_NUM}")
FORCE_ROW_RE = re.compile(rf"atom\s+\d+\s+type\s+\d+\s+force =\s+{_NUM}\s+{_NUM}\s+{_NUM}")
STRESS_HEAD_RE = re.compile(r"total\s+stress\s+\(Ry/bohr\*\*3\)\s+\(kbar\)\s+P=\s*(-?\d+\.\d+)")
FLOATS_RE = re.compile(r"-?\d+\.\d+(?:[eE][-+]?\d+)?")


def parse_pw(text: str, nat: int) -> dict:
    """Every SCF step of a pw.x output: energies (Ry), Fermi levels (eV), iterations,
    force blocks (the totals and all rows printed with them, eV/Å), stress (kbar),
    the final energy/enthalpy and the final cell of a relaxation."""
    force_blocks = []
    for block in re.findall(r"Forces acting on atoms.*?Total force", text, re.DOTALL):
        rows = [[float(x) * RYBOHR_TO_EVA for x in row] for row in FORCE_ROW_RE.findall(block)]
        force_blocks.append({"total": rows[:nat], "all": rows})
    stresses = []
    for match in STRESS_HEAD_RE.finditer(text):
        rows = text[match.end():].splitlines()[1:4]
        stresses.append([[float(x) for x in FLOATS_RE.findall(row)[3:6]] for row in rows])
    final = FINAL_LINE_RE.search(text)
    final_cell = final_positions = None
    tail = text.split("Begin final coordinates", 1)
    if len(tail) == 2:
        block = tail[1].split("End final coordinates", 1)[0]
        cell_block = block.split("CELL_PARAMETERS (angstrom)", 1)
        if len(cell_block) == 2:
            final_cell = [[float(x) for x in FLOATS_RE.findall(row)] for row in cell_block[1].splitlines()[1:4]]
        pos_block = block.split("ATOMIC_POSITIONS (angstrom)", 1)
        if len(pos_block) == 2:
            rows = [row.split() for row in pos_block[1].splitlines()[1:1 + nat]]
            final_positions = [(row[0], tuple(float(x) for x in row[1:4])) for row in rows]
    return {
        "energies_ry": [float(x) for x in ENERGY_LINE_RE.findall(text)],
        "fermi_ev": [float(x) for x in FERMI_LINE_RE.findall(text)],
        "iterations": [int(x) for x in ITERATIONS_RE.findall(text)],
        "total_force_ry_bohr": [float(x) for x in TOTAL_FORCE_RE.findall(text)],
        "forces": force_blocks,
        "stress_kbar": stresses,
        "final_ry": float(final.group(1)) if final else None,
        "final_cell": final_cell,
        "final_positions": final_positions,
        "bfgs_converged": "End of BFGS Geometry Optimization" in text,
        "scf_converged": "convergence has been achieved" in text,
        "job_done": "JOB DONE" in text,
    }


def parse_bands_dat(text: str) -> dict:
    """bands.x ``filband`` (``&plot nbnd=.., nks=.. /`` then per k: 3 coordinates, nbnd energies)."""
    head, _, body = text.partition("/")
    match = re.search(r"nbnd=\s*(\d+),\s*nks=\s*(\d+)", head)
    if match is None:
        raise ValueError("no '&plot nbnd=, nks=' header")
    nbnd, nks = int(match.group(1)), int(match.group(2))
    values = [float(x) for x in body.split()]
    if len(values) != nks * (3 + nbnd):
        raise ValueError(f"{len(values)} numbers for {nks} k-points x (3 + {nbnd})")
    stride = 3 + nbnd
    kpoints = [values[i * stride:i * stride + 3] for i in range(nks)]
    eigenvalues = [values[i * stride + 3:(i + 1) * stride] for i in range(nks)]
    return {"nbnd": nbnd, "nks": nks, "kpoints": kpoints, "eigenvalues": eigenvalues}


def parse_bands_gnu(text: str) -> dict:
    """bands.x ``.gnu`` file: one ``k-distance energy`` block per band, blocks separated by blank lines."""
    bands, current = [], []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            if current:
                bands.append(current)
                current = []
        else:
            current.append((float(parts[0]), float(parts[1])))
    if current:
        bands.append(current)
    return {"k_distances": [k for k, _ in bands[0]] if bands else [],
            "bands": [[e for _, e in band] for band in bands]}


def parse_dos_dat(text: str) -> dict:
    """dos.x output: ``# E (eV) dos(E) Int dos(E) EFermi = x eV`` then three columns."""
    lines = text.splitlines()
    match = re.search(r"EFermi\s*=\s*(-?\d+\.\d+)", lines[0]) if lines else None
    rows = [[float(x) for x in line.split()] for line in lines[1:] if line.strip() and not line.startswith("#")]
    return {"fermi_ev": float(match.group(1)) if match else None,
            "energies": [r[0] for r in rows], "dos": [r[1] for r in rows],
            "integrated": [r[2] for r in rows if len(r) > 2]}


def gap_reference(eigenvalues: list[list[float]], fermi_ev: float) -> dict:
    """The server's documented band-edge rule on per-k eigenvalues: occupied is ``e <= E_F``;
    a gap below 0.01 eV is a metal; direct when the first k of the VBM is the first k of the CBM."""
    occupied = [(e, k) for k, eigs in enumerate(eigenvalues) for e in eigs if e <= fermi_ev]
    empty = [(e, k) for k, eigs in enumerate(eigenvalues) for e in eigs if e > fermi_ev]
    if not occupied or not empty:
        return {"is_metal": True, "band_gap_eV": None, "vbm_eV": None, "cbm_eV": None, "is_direct_gap": False}
    vbm = max(e for e, _ in occupied)
    cbm = min(e for e, _ in empty)
    gap = cbm - vbm
    metal = gap < 0.01
    vbm_k = min(k for e, k in occupied if e == vbm)
    cbm_k = min(k for e, k in empty if e == cbm)
    return {"is_metal": metal, "band_gap_eV": None if metal else gap, "vbm_eV": vbm, "cbm_eV": cbm,
            "is_direct_gap": (not metal) and vbm_k == cbm_k}


def categorize_files(names: list[str]) -> dict[str, list[str]]:
    """qe_list_files' documented categories, for the plain files of one directory."""
    out = {key: [] for key in ("band_files", "dos_files", "pdos_files", "input_files", "output_files",
                               "other_files")}
    for name in names:
        if name.endswith(".gnu"):
            out["band_files"].append(name)
        elif name.startswith("dos") and name.endswith(".dat"):
            out["dos_files"].append(name)
        elif "pdos" in name:
            out["pdos_files"].append(name)
        elif name.endswith(".in"):
            out["input_files"].append(name)
        elif name.endswith(".out"):
            out["output_files"].append(name)
        else:
            out["other_files"].append(name)
    return {key: sorted(value) for key, value in out.items()}


def flatten(value) -> list[float]:
    if isinstance(value, (list, tuple)):
        return [x for item in value for x in flatten(item)]
    return [value]


def max_diff(a, b) -> float:
    """Largest elementwise difference of two (nested) number lists; inf on a shape mismatch."""
    fa, fb = flatten(a), flatten(b)
    if len(fa) != len(fb) or not _same_shape(a, b):
        return math.inf
    if any(not isinstance(x, (int, float)) or isinstance(x, bool) for x in fa + fb):
        return math.inf
    return max((abs(x - y) for x, y in zip(fa, fb)), default=0.0)


def _same_shape(a, b) -> bool:
    if isinstance(a, (list, tuple)) != isinstance(b, (list, tuple)):
        return False
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same_shape(x, y) for x, y in zip(a, b))
    return True


def close(got, want, tol: float) -> bool:
    """Equal within ``tol`` elementwise; dicts key by key (same keys), None only to None."""
    if got is None or want is None:
        return got is want
    if isinstance(want, dict) or isinstance(got, dict):
        return isinstance(got, dict) and isinstance(want, dict) and got.keys() == want.keys() \
            and all(close(got[k], want[k], tol) for k in want)
    return max_diff(got, want) <= tol


def mismatches(server: dict, want: dict, tolerances: dict[str, float]) -> list[str]:
    """Fields of ``server`` that differ from ``want`` (tolerance per key; exact when absent)."""
    problems = []
    for key, value in want.items():
        got = server.get(key)
        tol = tolerances.get(key)
        ok = close(got, value, tol) if tol is not None else got == value
        if not ok:
            shown_got, shown_want = (got, value) if len(str(value)) < 120 else ("<differs>", f"<{key}>")
            problems.append(f"{key}: {shown_got!r} != {shown_want!r}")
    return problems


def payload(result: dict | None):
    """The JSON object of a tools/call result (``structuredContent`` or the text block); None if absent."""
    if not isinstance(result, dict):
        return None
    data = result.get("structuredContent")
    if isinstance(data, dict):
        return data
    try:
        return json.loads(text_of(result))
    except (json.JSONDecodeError, TypeError):
        return None


def error_of(data) -> str | None:
    """The message of an error reported in a normal reply (``success: false``)."""
    if isinstance(data, dict) and data.get("success") is False:
        return str(data.get("error"))
    return None


def in_band_error(result: dict) -> str | None:
    """For :func:`check_rejected`: an in-band error of a raw tools/call result."""
    return error_of(payload(result))


# --------------------------------------------------------------------------
# L0: the locked environment is a working QE installation
# --------------------------------------------------------------------------

def check_qe_binaries(session: Session) -> None:
    """Each declared executable exists under QE_PREFIX and reports the pinned QE version.

    Runs them outside the server, with an empty input file in a scratch
    directory of their own: they then print the startup banner and stop (and
    may drop a ``CRASH`` file, which is why this is not the session cwd).
    """
    report, entry = session.report, session.entry
    pinned = spec_version(entry, "qe")
    prefix = Path(session.server.get("env", {}).get("QE_PREFIX", ""))
    if not prefix.is_dir():
        report.add("L0", "QE_PREFIX", "FAIL", f"launch env QE_PREFIX={str(prefix)!r} is not a directory")
        return
    probe = session.tmp / "qe-version-probe"
    probe.mkdir()
    empty = probe / "empty.in"
    empty.write_text("", encoding="utf-8")
    versions = session.state["qe_versions"] = {}
    for executable in QE_EXECUTABLES:
        name = f"{executable} reports QE {pinned}"
        binary = prefix / executable
        try:
            completed = subprocess.run([str(binary), "-i", str(empty)], cwd=probe, env=session.env,
                                       capture_output=True, text=True, timeout=BANNER_TIMEOUT)
        except (OSError, subprocess.SubprocessError) as exc:
            report.add("L0", name, "FAIL", f"cannot run {binary}: {exc}")
            continue
        finally:
            (probe / "CRASH").unlink(missing_ok=True)   # QE's error marker for the empty input
        version = banner_version(completed.stdout, program_name(executable))
        versions[executable] = version
        if version == pinned:
            report.add("L0", name, "PASS", f"{binary.name} banner: {program_name(executable)} v.{version}")
        else:
            report.add("L0", name, "FAIL",
                       f"banner reports {version!r}, conda.specs pins qe={pinned}; "
                       f"stdout {completed.stdout[:200]!r} stderr {completed.stderr[:200]!r}")


def check_pseudopotential_library(session: Session) -> None:
    """The vendored SG15 library is complete; its element index and its scan order are the L1 references."""
    report = session.report
    pseudo_dir = Path(session.server.get("env", {}).get("QE_PSEUDO_DIR", ""))
    expected_dir = session.checkout / PSEUDO_SUBDIR
    if pseudo_dir != expected_dir:
        report.add("L0", "QE_PSEUDO_DIR", "FAIL",
                   f"launch env points at {str(pseudo_dir)!r}, expected the vendored {str(expected_dir)!r}")
        return
    in_order = [p.name for p in pseudo_dir.glob("*.upf")]   # the host's directory order, as the server sees it
    names = sorted(in_order)
    elements = session.state["elements"] = upf_elements(names)
    session.state["upf_in_order"] = in_order
    status = "PASS" if len(names) == EXPECTED_UPF_FILES else "FAIL"
    report.add("L0", "vendored SG15 ONCV library", status,
               f"{len(names)} .upf files" + ("" if status == "PASS" else f", expected {EXPECTED_UPF_FILES}")
               + f"; {len(elements)} elements indexed by name",
               n_upf_files=len(names), n_elements=len(elements))


def prepare(session: Session) -> None:
    TIMINGS.clear()
    check_qe_binaries(session)
    check_pseudopotential_library(session)


# --------------------------------------------------------------------------
# Independent references: the QE executables, run outside the server
# --------------------------------------------------------------------------

class QERef:
    """Our own runs of pw.x / bands.x / dos.x, each in its own directory under the
    session tmp, with the server's environment (one thread) and the pseudopotential
    file the server reports it uses."""

    def __init__(self, session: Session, pseudo_file: str) -> None:
        self.prefix = Path(session.server["env"]["QE_PREFIX"])
        self.pseudo = session.checkout / PSEUDO_SUBDIR / pseudo_file
        self.pseudo_file = pseudo_file
        self.root = session.tmp / "reference"
        self.root.mkdir(exist_ok=True)
        self.env = session.env
        self._cache: dict[str, dict] = {}

    def pipeline(self, tag: str, steps: list[tuple[str, str, str]]) -> dict:
        """Run ``(name, executable, input)`` steps in one directory; outputs by step name."""
        if tag in self._cache:
            return self._cache[tag]
        work = self.root / tag
        (work / "pseudo").mkdir(parents=True)
        shutil.copy(self.pseudo, work / "pseudo" / self.pseudo_file)
        outputs = {"dir": work}
        progress(f"reference {tag}: {' -> '.join(exe for _, exe, _ in steps)}")
        started = time.monotonic()
        for name, executable, text in steps:
            (work / f"{name}.in").write_text(text, encoding="utf-8")
            completed = subprocess.run([str(self.prefix / executable), "-i", f"{name}.in"], cwd=work,
                                       env=self.env, capture_output=True, text=True, timeout=REF_TIMEOUT)
            (work / f"{name}.out").write_text(completed.stdout, encoding="utf-8")
            if completed.returncode != 0 or "JOB DONE" not in completed.stdout:
                raise RuntimeError(f"reference {tag}/{name}: {executable} exit {completed.returncode}; "
                                   f"stderr {completed.stderr[-300:]!r}")
            outputs[name] = completed.stdout
        TIMINGS[f"reference {tag}"] = round(time.monotonic() - started, 2)
        self._cache[tag] = outputs
        return outputs

    def _pw(self, calculation: str, atoms, cell=FCC_CELL, **kwargs) -> str:
        return pw_input(calculation, atoms, cell, self.pseudo_file, **kwargs)

    def scf(self) -> dict:
        out = self.pipeline("scf", [("scf", "pw.x", self._pw("scf", SI_ATOMS, kgrid=KGRID))])
        return parse_pw(out["scf"], len(SI_ATOMS))

    def relax(self) -> dict:
        out = self.pipeline("relax", [("relax", "pw.x", self._pw("relax", PERTURBED_ATOMS, kgrid=KGRID))])
        return parse_pw(out["relax"], len(PERTURBED_ATOMS))

    def vc_relax(self) -> dict:
        out = self.pipeline("vc-relax", [("vc", "pw.x", self._pw("vc-relax", SI_ATOMS, kgrid=KGRID))])
        return parse_pw(out["vc"], len(SI_ATOMS))

    def tight_scf(self, tag: str, atoms) -> dict:
        text = self._pw("scf", atoms, kgrid=KGRID, electrons={"conv_thr": TIGHT_CONV_THR})
        return parse_pw(self.pipeline(tag, [("scf", "pw.x", text)])["scf"], len(atoms))

    def bands(self, kpoints_crystal) -> dict:
        out = self.pipeline("bands", [
            ("scf", "pw.x", self._pw("scf", SI_ATOMS, kgrid=KGRID)),
            ("nscf", "pw.x", self._pw("bands", SI_ATOMS, kpoints_crystal=kpoints_crystal,
                                      system={"nbnd": NBND, "nosym": True})),
            ("bands", "bands.x", bands_x_input()),
        ])
        work = out["dir"]
        return {"scf": parse_pw(out["scf"], len(SI_ATOMS)),
                "dat": parse_bands_dat((work / "bands.dat").read_text(encoding="utf-8")),
                "gnu": parse_bands_gnu((work / "bands.dat.gnu").read_text(encoding="utf-8"))}

    def dos(self) -> dict:
        out = self.pipeline("dos", [
            ("scf", "pw.x", self._pw("scf", SI_ATOMS, kgrid=KGRID)),
            ("nscf", "pw.x", self._pw("nscf", SI_ATOMS, kgrid=DOS_KGRID, occupations={"occupations": "tetrahedra"})),
            ("dos", "dos.x", dos_x_input()),
        ])
        return {"scf": parse_pw(out["scf"], len(SI_ATOMS)),
                "dat": parse_dos_dat((out["dir"] / "dos.dat").read_text(encoding="utf-8"))}


def band_path(npoints: int) -> tuple[list[list[float]], dict[str, list[float]]]:
    """ASE's band path of the silicon cell (crystal coordinates) and its special points."""
    from ase.cell import Cell
    path = Cell(FCC_CELL).bandpath(npoints=npoints)
    return path.kpts.tolist(), {label: [float(c) for c in k] for label, k in path.special_points.items()}


# --------------------------------------------------------------------------
# L1 helpers
# --------------------------------------------------------------------------

def tool_ok(call: Caller, name: str, tool: str, arguments: dict) -> dict | None:
    """A successful call of a valid request; FAIL (and None) otherwise."""
    progress(f"{name}: calling {tool}")
    data = timed(name, call.json, name, tool, arguments)
    if data is None:
        return None
    if not isinstance(data, dict) or data.get("_isError"):
        call.report.add("L1", name, "FAIL", f"tool error for a valid request: {data!r:.300}")
        return None
    if data.get("success") is False:
        call.report.add("L1", name, "FAIL", f"success=false for a valid request: {data.get('error')!r:.300}")
        return None
    return data


def verdict(report, name: str, problems: list[str], detail: str, **data) -> bool:
    report.add("L1", name, "FAIL" if problems else "PASS", "; ".join(problems)[:600] if problems else detail, **data)
    return not problems


def three_state(report, name: str, correct: list[str], defect: list[str], pass_detail: str,
                warn_detail: str, **data) -> None:
    """PASS when the correct answer matches, WARN when exactly the known defect does, FAIL otherwise."""
    if not correct:
        report.add("L1", name, "PASS", pass_detail, **data)
    elif not defect:
        report.add("L1", name, "WARN", warn_detail, **data)
    else:
        report.add("L1", name, "FAIL", f"neither correct ({'; '.join(correct)[:300]}) nor the known defect "
                                       f"({'; '.join(defect)[:300]})", **data)


def in_workdir(session: Session, output_dir) -> list[str]:
    """The server's work directories are <cwd>/qe_calculations/<id> (QE_WORKDIR unset)."""
    if not isinstance(output_dir, str):
        return [f"output_dir {output_dir!r}"]
    path = Path(output_dir)
    if path.parent.resolve() != (session.cwd / "qe_calculations").resolve() or not path.is_dir():
        return [f"output_dir {output_dir} is not a directory under <cwd>/qe_calculations"]
    return []


def step_values(out: dict, index: int) -> dict:
    """Energy, Fermi level, total forces and stress of one SCF step of a parsed run."""
    return {"total_energy_Ry": out["energies_ry"][index],
            "total_energy_eV": out["energies_ry"][index] * RY_EV,
            "fermi_energy_eV": out["fermi_ev"][index],
            "forces": out["forces"][index]["total"] if out["forces"] else None,
            **({"stress_kbar": out["stress_kbar"][index]} if out["stress_kbar"] else {})}


STEP_TOLERANCES = {"total_energy_Ry": ENERGY_TOL_RY, "total_energy_eV": ENERGY_TOL_EV,
                   "fermi_energy_eV": FERMI_TOL, "forces": FORCE_TOL, "stress_kbar": STRESS_TOL}


def server_step(data: dict, nat: int) -> dict:
    forces = data.get("forces_eV_per_angstrom")
    view = dict(data)
    view["forces"] = forces[:nat] if isinstance(forces, list) else forces
    return view


def check_force_rows(report, name: str, data: dict, out: dict, nat: int) -> None:
    """D13: the forces list should hold nat rows; with every contribution block appended it is the defect."""
    rows = data.get("forces_eV_per_angstrom")
    block = out["forces"][0] if out["forces"] else None
    if block is None or not isinstance(rows, list):
        report.add("L1", name, "FAIL", f"no forces to compare (server {type(rows).__name__}, reference {block!r:.80})")
        return
    three_state(report, name,
                [] if close(rows, block["total"], FORCE_TOL) else [f"{len(rows)} rows are not the {nat} total forces"],
                [] if close(rows, block["all"], FORCE_TOL)
                else [f"nor the {len(block['all'])} rows printed with them"],
                f"{nat} rows, the total forces",
                f"D13: {len(rows)} rows = the {nat} total forces followed by the six contribution blocks "
                f"pw.x prints with verbosity='high'; only the first {nat} are forces")


# --------------------------------------------------------------------------
# L1: pure-Python tools
# --------------------------------------------------------------------------

def check_status(session: Session) -> None:
    """``qe_status``: the server's view of the environment the manifest gives it."""
    call, report = session.call, session.report
    result = call.json("qe_status", "qe_status", {})
    if not isinstance(result, dict) or result.get("_isError"):
        report.add("L1", "qe_status", "FAIL", f"unusable result: {result!r}")
        return
    config, runner = result.get("config", {}), result.get("runner", {})
    pseudos = result.get("pseudopotentials", {})
    env = session.server.get("env", {})

    expected_config = {"use_docker": False, "nprocs": int(env.get("QE_NPROCS", "1")),
                       "runner": env.get("QE_RUNNER"), "pseudo_dir": env.get("QE_PSEUDO_DIR")}
    wrong = {key: config.get(key) for key, want in expected_config.items() if config.get(key) != want}
    report.add("L1", "qe_status[config] echoes the launch env", "PASS" if not wrong else "FAIL",
               f"{expected_config}" if not wrong else f"{wrong} != expected {expected_config}",
               server_value=config)

    local = runner.get("available") is True and runner.get("type") == "LocalQERunner"
    report.add("L1", "qe_status[runner] finds the local QE binaries", "PASS" if local else "FAIL",
               f"{runner.get('type')}, requested {runner.get('requested')!r}" if local
               else f"expected an available LocalQERunner, got {runner!r}",
               server_value=runner)

    elements = session.state.get("elements")
    ok = pseudos.get("available") is True and pseudos.get("library") == "SG15 ONCV" \
        and elements is not None and pseudos.get("n_elements") == len(elements)
    report.add("L1", "qe_status[pseudopotentials] matches the vendored library", "PASS" if ok else "FAIL",
               f"{pseudos.get('n_elements')} elements, as derived from the .upf names" if ok
               else f"{pseudos!r} does not match {len(elements) if elements else '?'} indexed elements",
               server_value=pseudos)


def check_pseudopotentials(session: Session) -> str | None:
    """``qe_list_pseudopotentials``: the element index, the per-element pick (D1), the Si hints.
    Returns the Si file the server uses (the DFT references use the same one)."""
    call, report = session.call, session.report
    data = tool_ok(call, "qe_list_pseudopotentials", "qe_list_pseudopotentials", {})
    in_order = session.state.get("upf_in_order")
    if data is None or in_order is None:
        return None
    details = data.get("details") or {}
    elements = session.state["elements"]
    verdict(report, "qe_list_pseudopotentials[index]",
            mismatches(data, {"library": "SG15 ONCV", "n_elements": len(elements), "elements": elements}, {}),
            f"{len(elements)} elements, sorted")
    picked = {el: (details.get(el) or {}).get("filename") for el in elements}
    emulated = scan_pick(in_order)
    wrong = {el: (picked[el], emulated.get(el)) for el in elements if picked[el] != emulated.get(el)}
    verdict(report, "qe_list_pseudopotentials[files follow the directory scan]",
            [f"{el}: server {got!r}, scan {want!r}" for el, (got, want) in sorted(wrong.items())],
            f"all {len(elements)} picks equal a re-run of the scan in this host's directory order")
    newest = newest_pick(in_order)
    stale = {el: (picked[el], newest[el]) for el in elements if picked[el] != newest.get(el)}
    session.state["pseudo_pick"] = {"Si": picked.get("Si"),
                                    "not_newest": {el: f"{got} (newest {want})" for el, (got, want) in sorted(stale.items())}}
    three_state(report, "qe_list_pseudopotentials[newest version per element]",
                [f"{len(stale)} not newest"] if stale else [],
                [] if not wrong else ["picks do not follow the scan"],
                "every element got its newest file",
                f"D1: {len(stale)} element(s) got an older file in this host's directory order: "
                + ", ".join(f"{el} {got} (newest {want})" for el, (got, want) in sorted(stale.items()))[:400],
                not_newest=sorted(stale))
    si = details.get("Si") or {}
    verdict(report, "qe_list_pseudopotentials[Si cutoff hints]",
            mismatches(si, {"ecutwfc_Ry": SI_ECUTWFC, "ecutrho_Ry": SI_ECUTRHO}, {}),
            f"{si.get('filename')}, {SI_ECUTWFC}/{SI_ECUTRHO} Ry")
    pick = picked.get("Si")
    if pick and (session.checkout / PSEUDO_SUBDIR / pick).is_file():
        return pick
    report.add("L1", "qe_list_pseudopotentials[Si file]", "FAIL", f"Si -> {pick!r} is not a vendored file")
    return None


def structure_expectation(atoms, cell, pbc=(True, True, True)) -> dict:
    return {"symbols": [s for s, _ in atoms], "positions": [list(p) for _, p in atoms],
            "cell": [list(r) for r in cell], "pbc": list(pbc), "formula": chemical_formula([s for s, _ in atoms]),
            "n_atoms": len(atoms), "volume_angstrom3": cell_volume(cell)}


STRUCTURE_TOLERANCES = {"positions": GEOM_TOL, "cell": GEOM_TOL, "volume_angstrom3": 1e-9}


def check_structures(session: Session) -> None:
    """``qe_load_structure`` from the built-in name, the inline format and a POSCAR file."""
    call, report = session.call, session.report
    files = session.tmp / "structures"
    files.mkdir(exist_ok=True)
    vasp = files / "si_displaced.vasp"
    vasp.write_text(poscar(PERTURBED_ATOMS, FCC_CELL), encoding="utf-8")
    cases = [("built-in 'Si'", STRUCTURE, SI_ATOMS, FCC_CELL),
             ("inline xyz|lattice", inline_structure(PERTURBED_ATOMS, FCC_CELL), PERTURBED_ATOMS, FCC_CELL),
             ("POSCAR file path", str(vasp), PERTURBED_ATOMS, FCC_CELL)]
    for label, structure, atoms, cell in cases:
        name = f"qe_load_structure[{label}]"
        data = tool_ok(call, name, "qe_load_structure", {"structure": structure})
        if data is not None:
            want = structure_expectation(atoms, cell)
            verdict(report, name, mismatches(data, want, STRUCTURE_TOLERANCES),
                    f"{want['formula']}, V = {want['volume_angstrom3']:.6f} Å³")
    check_rejected(call, "qe_load_structure[unparseable text]", "qe_load_structure",
                   {"structure": NOT_A_STRUCTURE}, in_band=in_band_error)


def check_validate(session: Session) -> None:
    """``qe_validate_structure``: a sound cell, and one with overlapping atoms."""
    call, report = session.call, session.report
    data = tool_ok(call, "qe_validate_structure[Si]", "qe_validate_structure", {"structure": STRUCTURE})
    if data is not None:
        lengths = vector_lengths(FCC_CELL)
        want = {"valid": True, "formula": "Si2", "n_atoms": 2, "volume_angstrom3": cell_volume(FCC_CELL),
                "errors": [], "warnings": [],
                "recommendations": {"ecutwfc_Ry": SI_ECUTWFC, "ecutrho_Ry": SI_ECUTRHO,
                                    "kpoints": kgrid_reference(lengths, [True] * 3)}}
        verdict(report, "qe_validate_structure[Si]", mismatches(data, want, {"volume_angstrom3": 1e-9}),
                f"valid, nearest neighbour {min_image_distance(SI_ATOMS, FCC_CELL):.4f} Å, "
                f"recommends {want['recommendations']}")
    name = "qe_validate_structure[overlapping atoms]"
    progress(f"{name}: calling qe_validate_structure")
    data = call.json(name, "qe_validate_structure", {"structure": inline_structure(OVERLAP_ATOMS, OVERLAP_CELL)})
    if isinstance(data, dict) and not data.get("_isError"):
        distance = min_image_distance(OVERLAP_ATOMS, OVERLAP_CELL)
        want = {"valid": False, "errors": [f"Atoms too close: minimum distance = {distance:.2f} Å"], "warnings": []}
        verdict(report, name, mismatches(data, want, {}), f"refused: {want['errors'][0]}")


def check_suggest_kpoints(session: Session) -> None:
    """``qe_suggest_kpoints``: the three density presets and an explicit spacing."""
    call, report = session.call, session.report
    lengths = vector_lengths(FCC_CELL)
    cases = [({"density": d}, kgrid_reference(lengths, [True] * 3, density=d), f"density='{d}'")
             for d in ("low", "medium", "high")]
    cases.append(({"kspacing": KSPACING}, kgrid_reference(lengths, [True] * 3, kspacing=KSPACING),
                  f"kspacing={KSPACING} 1/Å"))
    for arguments, grid, method in cases:
        name = f"qe_suggest_kpoints[{method}]"
        data = tool_ok(call, name, "qe_suggest_kpoints", {"structure": STRUCTURE, **arguments})
        if data is not None:
            want = {"kpoints": grid, "total_kpoints_approx": math.prod(grid), "method": method,
                    "cell_lengths_angstrom": [f"{x:.2f}" for x in lengths]}
            verdict(report, name, mismatches(data, want, {}), f"{grid} for |a_i| = {lengths[0]:.4f} Å")


def check_kpath(session: Session) -> None:
    """``qe_get_kpath``: correct would be ASE's band path; D10 makes it fail for every cell."""
    call, report = session.call, session.report
    name = "qe_get_kpath"
    progress(f"{name}: calling qe_get_kpath")
    data = call.json(name, "qe_get_kpath", {"structure": STRUCTURE, "npoints": NPOINTS_BAND})
    if not isinstance(data, dict) or data.get("_isError"):
        report.add("L1", name, "FAIL", f"unusable result: {data!r:.200}")
        return
    kpts, special = band_path(NPOINTS_BAND)
    labels = "-".join(label for label, _ in sorted(special.items(), key=lambda item: item[1]))
    correct = mismatches(data, {"success": True, "n_kpoints": len(kpts), "special_points": special,
                                "path_labels": labels, "kpoints": kpts},
                         {"special_points": ARRAY_TOL, "kpoints": ARRAY_TOL})
    defect = mismatches(data, {"success": False, "error": "The truth value of an array with more than one "
                                                          "element is ambiguous. Use a.any() or a.all()"}, {})
    three_state(report, name, correct, defect, f"{len(kpts)} k-points, {labels}",
                "D10: sorting the special points by their coordinate arrays raises, for every structure: "
                f"{data.get('error')!r}")


SYNTHETIC_GNU = "    0.0000   -5.7007\n    0.1414   -5.5258\n    0.2828   -5.0049\n\n" \
                "    0.0000    6.2771\n    0.1414    5.3349\n    0.2828    3.6470\n\n"
SYNTHETIC_DOS = ("#  E (eV)   dos(E)     Int dos(E) EFermi =    6.716 eV\n"
                 "  -5.701  0.0000E+00  0.0000E+00\n  -5.691  0.1759E-03  0.5863E-06\n"
                 "  -5.681  0.7035E-03  0.4690E-05\n")
SYNTHETIC_PDOS_NAME = "pwscf.pdos_atm#1(Si)_wfc#2(p)"
SYNTHETIC_PDOS = ("# E (eV)  ldos(E)  pdos(E)   pdos(E)   pdos(E)\n"
                  " -5.701  0.100E-02  0.300E-03  0.300E-03  0.400E-03\n"
                  " -5.691  0.200E-02  0.600E-03  0.700E-03  0.700E-03\n")


def check_readers_synthetic(session: Session) -> None:
    """``qe_read_bands`` / ``qe_read_dos`` / ``qe_read_pdos`` on files whose contents we wrote."""
    call, report = session.call, session.report
    folder = session.tmp / "synthetic"
    folder.mkdir(exist_ok=True)
    (folder / "bands.dat.gnu").write_text(SYNTHETIC_GNU, encoding="utf-8")
    (folder / "dos.dat").write_text(SYNTHETIC_DOS, encoding="utf-8")
    (folder / SYNTHETIC_PDOS_NAME).write_text(SYNTHETIC_PDOS, encoding="utf-8")

    data = tool_ok(call, "qe_read_bands[synthetic]", "qe_read_bands", {"output_dir": str(folder / "bands.dat.gnu")})
    if data is not None:
        gnu = parse_bands_gnu(SYNTHETIC_GNU)
        want = {"n_bands": len(gnu["bands"]), "n_kpoints": len(gnu["k_distances"]),
                "k_distances": gnu["k_distances"], "bands": gnu["bands"], "raw_data": SYNTHETIC_GNU}
        verdict(report, "qe_read_bands[synthetic]", mismatches(data, want, {}), "2 bands x 3 k-points as written")
    data = tool_ok(call, "qe_read_dos[synthetic]", "qe_read_dos", {"output_dir": str(folder / "dos.dat")})
    if data is not None:
        dos = parse_dos_dat(SYNTHETIC_DOS)
        want = {"n_points": len(dos["energies"]), "energies": dos["energies"], "dos": dos["dos"],
                "integrated_dos": dos["integrated"], "fermi_energy": dos["fermi_ev"], "raw_data": SYNTHETIC_DOS}
        verdict(report, "qe_read_dos[synthetic]", mismatches(data, want, {}), "3 points and EFermi as written")
    data = tool_ok(call, "qe_read_pdos[synthetic]", "qe_read_pdos", {"pdos_file": str(folder / SYNTHETIC_PDOS_NAME)})
    if data is not None:
        rows = [[float(x) for x in line.split()] for line in SYNTHETIC_PDOS.splitlines()[1:]]
        want = {"atom_info": SYNTHETIC_PDOS_NAME, "n_points": len(rows), "energies": [r[0] for r in rows],
                "ldos": [r[1] for r in rows], "pdos_columns": [r[2:] for r in rows], "raw_data": SYNTHETIC_PDOS}
        verdict(report, "qe_read_pdos[synthetic]", mismatches(data, want, {}),
                "2 points, ldos + 3 orbital columns as written")
    for tool, arg in (("qe_read_bands", "output_dir"), ("qe_read_dos", "output_dir"), ("qe_read_pdos", "pdos_file")):
        missing = str(folder / "missing.file")
        check_rejected(call, f"{tool}[missing file]", tool, {arg: missing}, in_band=in_band_error)


def check_directory_reader(session: Session, tool: str, folder: str, file_name: str, want: dict) -> None:
    """D2: the parameter is called ``output_dir``; reading the directory itself is the natural call."""
    call, report = session.call, session.report
    name = f"{tool}[a directory, as the parameter name says]"
    progress(f"{name}: calling {tool}")
    data = call.json(name, tool, {"output_dir": folder})
    if not isinstance(data, dict) or data.get("_isError"):
        report.add("L1", name, "FAIL", f"unusable result: {data!r:.200}")
        return
    error = str(data.get("error", ""))
    three_state(report, name, mismatches(data, want, {}),
                [] if data.get("success") is False and error.startswith("[Errno 21] Is a directory") else [error],
                f"read {file_name} from the directory",
                f"D2: the parameter is named output_dir but only a file path works ({file_name}): {error[:120]!r}")


def check_list_files(session: Session, name: str, folder: str) -> None:
    """``qe_list_files`` on a server work directory, against our own listing of it."""
    call, report = session.call, session.report
    data = tool_ok(call, name, "qe_list_files", {"output_dir": folder})
    if data is None:
        return
    path = Path(folder)
    want = {key: [str(path / n) for n in value]
            for key, value in categorize_files([p.name for p in path.iterdir() if p.is_file()]).items()}
    got = {key: sorted(data.get(key) or []) for key in want}
    verdict(report, name, mismatches(got, want, {}) + mismatches(data, {"directory": folder}, {}),
            ", ".join(f"{key} {len(value)}" for key, value in want.items()))


# --------------------------------------------------------------------------
# L1: the DFT tools
# --------------------------------------------------------------------------

def dft_args(structure: str = STRUCTURE, *, ecut: bool = True, **extra) -> dict:
    return {"structure": structure, "kpoints": KPOINTS_ARG, **({"ecutwfc": SI_ECUTWFC} if ecut else {}), **extra}


def check_artifacts(session: Session, name: str, data: dict, kind: str, pick: str) -> list[str]:
    """The run directory: under <cwd>/qe_calculations, named <kind>_<hex8>, with the picked pseudopotential."""
    problems = in_workdir(session, data.get("output_dir"))
    if problems:
        return problems
    folder = Path(data["output_dir"])
    if not WORKFLOW_ID_RE[kind].fullmatch(folder.name):
        problems.append(f"run directory {folder.name!r} is not {kind}_<8 hex>")
    copied = sorted(p.name for p in (folder / "pseudo").glob("*.upf"))
    if copied != [pick]:
        problems.append(f"pseudo/ holds {copied}, the index says {pick}")
    return problems


def check_scf(session: Session, ref: QERef) -> None:
    call, report = session.call, session.report
    name = "qe_run_scf"
    data = tool_ok(call, name, "qe_run_scf", dft_args())
    if data is None:
        return
    out = ref.scf()
    want = {**step_values(out, 0), "converged": True, "n_iterations": out["iterations"][0],
            "parameters_used": {"spin_polarized": False, "smearing": "cold", "degauss": 0.02}}
    verdict(report, name, mismatches(server_step(data, 2), want, STEP_TOLERANCES)
            + check_artifacts(session, name, data, "scf", ref.pseudo_file),
            f"E = {want['total_energy_Ry']:.8f} Ry, E_F = {want['fermi_energy_eV']} eV, "
            f"{want['n_iterations']} iterations, P stress {want.get('stress_kbar', [[None]])[0][0]} kbar",
            server_value={k: data.get(k) for k in ("total_energy_Ry", "fermi_energy_eV", "n_iterations")})
    check_force_rows(report, "qe_run_scf[forces rows]", data, out, len(SI_ATOMS))
    check_rejected(call, "qe_run_scf[unparseable structure]", "qe_run_scf",
                   {"structure": NOT_A_STRUCTURE, "kpoints": KPOINTS_ARG}, in_band=in_band_error)
    check_rejected(call, "qe_run_scf[docker runner without Docker]", "qe_run_scf",
                   dft_args(runner="docker"), in_band=in_band_error)


def check_relaxation(session: Session, ref: QERef, tool: str, kind: str, out: dict, structure: str,
                     nat: int) -> dict | None:
    """``qe_run_relax`` / ``qe_run_vc_relax``: the run itself (artefact) and what the reply reports (D11)."""
    call, report = session.call, session.report
    data = tool_ok(call, tool, tool, dft_args(structure))
    if data is None:
        return None
    first, last = step_values(out, 0), step_values(out, -1)
    verdict(report, f"{tool}[run]",
            mismatches(data, {"relaxation_converged": out["bfgs_converged"], "converged": True}, {})
            + check_artifacts(session, tool, data, kind, ref.pseudo_file),
            f"{len(out['energies_ry'])} SCF steps, BFGS converged {out['bfgs_converged']}")
    # The server's own output file: the same relaxation, step for step.
    try:
        own = parse_pw(Path(data["output_file"]).read_text(encoding="utf-8"), nat)
        problems = mismatches(own, {"energies_ry": out["energies_ry"], "final_ry": out["final_ry"],
                                    "final_cell": out["final_cell"]},
                              {"energies_ry": ENERGY_TOL_RY, "final_ry": ENERGY_TOL_RY, "final_cell": CELL_TOL})
    except (OSError, KeyError, TypeError, ValueError, IndexError) as exc:
        problems = [f"cannot read the server's output file: {exc}"]
    verdict(report, f"{tool}[output file]", problems,
            f"final {out['final_ry']:.10f} Ry" + (f", final cell a1 = {out['final_cell'][0]}" if out["final_cell"] else ""))
    view = server_step(data, nat)
    keys = [k for k in first if k in STEP_TOLERANCES and (k != "stress_kbar" or "stress_kbar" in data)]
    pick = lambda values: {k: values[k] for k in keys if k in values}
    three_state(report, f"{tool}[reported energy, Fermi level, forces]",
                mismatches(view, pick(last), STEP_TOLERANCES), mismatches(view, pick(first), STEP_TOLERANCES),
                f"the relaxed structure: E = {last['total_energy_Ry']:.8f} Ry",
                f"D11: the reply describes the first SCF step (E = {first['total_energy_Ry']:.8f} Ry, "
                f"E_F = {first['fermi_energy_eV']} eV), not the relaxed structure "
                f"(E = {last['total_energy_Ry']:.8f} Ry, final {out['final_ry']:.10f} Ry)",
                server_value={k: data.get(k) for k in ("total_energy_Ry", "fermi_energy_eV")})
    check_force_rows(report, f"{tool}[forces rows]", data, out, nat)
    return data


def check_bandstructure(session: Session, ref: QERef) -> None:
    call, report = session.call, session.report
    name = "qe_workflow_bandstructure"
    data = tool_ok(call, name, "qe_workflow_bandstructure", dft_args(npoints_band=NPOINTS_BAND))
    if data is None:
        return
    kpts, special = band_path(NPOINTS_BAND)
    result = ref.bands(kpts)
    scf, dat = result["scf"], result["dat"]
    gap = gap_reference(dat["eigenvalues"], scf["fermi_ev"][0])
    want = {"total_energy_eV": scf["energies_ry"][0] * RY_EV, "fermi_energy_eV": scf["fermi_ev"][0],
            "n_bands": NBND, "n_kpoints": NPOINTS_BAND, "eigenvalues_eV": dat["eigenvalues"],
            "kpoints": dat["kpoints"], "high_symmetry_points": special, **gap}
    problems = mismatches(data, want, {"total_energy_eV": ENERGY_TOL_EV, "fermi_energy_eV": FERMI_TOL,
                                       "eigenvalues_eV": EIG_TOL, "kpoints": ARRAY_TOL,
                                       "high_symmetry_points": ARRAY_TOL, "band_gap_eV": EIG_TOL,
                                       "vbm_eV": EIG_TOL, "cbm_eV": EIG_TOL})
    if dat["nbnd"] != NBND or dat["nks"] != NPOINTS_BAND:
        problems.append(f"reference bands.dat has {dat['nbnd']} bands x {dat['nks']} k")
    problems += check_artifacts(session, name, data, "bands", ref.pseudo_file)
    folder = data.get("output_dir")
    if data.get("workflow_id") != (Path(folder).name if isinstance(folder, str) else None):
        problems.append(f"workflow_id {data.get('workflow_id')!r} is not the directory name")
    if data.get("bands_file") != (str(Path(folder) / "bands.dat") if isinstance(folder, str) else None):
        problems.append(f"bands_file {data.get('bands_file')!r}")
    session.state["bands_workflow_id"] = data.get("workflow_id")
    verdict(report, name, problems,
            f"{data.get('workflow_id')}: gap {gap['band_gap_eV']:.3f} eV ({'direct' if gap['is_direct_gap'] else 'indirect'}), "
            f"VBM {gap['vbm_eV']} / CBM {gap['cbm_eV']} eV, {NBND} bands x {NPOINTS_BAND} k",
            server_value={k: data.get(k) for k in ("workflow_id", "band_gap_eV", "is_direct_gap", "fermi_energy_eV")})
    if problems and any(p.startswith("output_dir") for p in problems):
        return
    folder = data["output_dir"]
    check_list_files(session, "qe_list_files[band structure run]", folder)
    server_gnu = Path(folder) / "bands.dat.gnu"
    gnu = parse_bands_gnu(server_gnu.read_text(encoding="utf-8")) if server_gnu.is_file() else {"bands": []}
    mine = result["gnu"]
    read = tool_ok(call, "qe_read_bands[the run's bands.dat.gnu]", "qe_read_bands", {"output_dir": str(server_gnu)})
    if read is not None:
        verdict(report, "qe_read_bands[the run's bands.dat.gnu]",
                mismatches(read, {"n_bands": len(mine["bands"]), "n_kpoints": len(mine["k_distances"]),
                                  "k_distances": mine["k_distances"], "bands": mine["bands"]},
                           {"k_distances": ARRAY_TOL, "bands": ARRAY_TOL}),
                f"{len(mine['bands'])} bands x {len(mine['k_distances'])} k, equal to the reference run's file")
    check_directory_reader(session, "qe_read_bands", folder, "bands.dat.gnu",
                           {"success": True, "n_bands": len(gnu["bands"]), "bands": gnu["bands"]})


def check_dos(session: Session, ref: QERef) -> None:
    call, report = session.call, session.report
    name = "qe_workflow_dos"
    data = tool_ok(call, name, "qe_workflow_dos", dft_args())
    if data is None:
        return
    result = ref.dos()
    scf, dos = result["scf"], result["dat"]
    want = {"total_energy_eV": scf["energies_ry"][0] * RY_EV, "fermi_energy_eV": scf["fermi_ev"][0],
            "energies_eV": dos["energies"], "dos": dos["dos"], "integrated_dos": dos["integrated"],
            "dos_fermi_eV": dos["fermi_ev"], "n_points": len(dos["energies"]),
            "energy_range_eV": [min(dos["energies"]), max(dos["energies"])]}
    problems = mismatches(data, want, {"total_energy_eV": ENERGY_TOL_EV, "fermi_energy_eV": FERMI_TOL,
                                       "energies_eV": ARRAY_TOL, "dos": ARRAY_TOL, "integrated_dos": ARRAY_TOL,
                                       "dos_fermi_eV": ARRAY_TOL, "energy_range_eV": ARRAY_TOL})
    problems += check_artifacts(session, name, data, "dos", ref.pseudo_file)
    folder = data.get("output_dir")
    if isinstance(folder, str) and data.get("dos_file") != str(Path(folder) / "dos.dat"):
        problems.append(f"dos_file {data.get('dos_file')!r}")
    if isinstance(folder, str) and (Path(folder) / "nscf.in").is_file():
        nscf_in = (Path(folder) / "nscf.in").read_text(encoding="utf-8")
        grid = " ".join(str(k) for k in DOS_KGRID)
        if f"{grid} 0 0 0" not in nscf_in or "tetrahedra" not in nscf_in:
            problems.append(f"nscf.in is not a {grid} tetrahedra grid")
    verdict(report, name, problems,
            f"{want['n_points']} points over {want['energy_range_eV']} eV, dos.x EFermi {dos['fermi_ev']} eV "
            f"(NSCF {'x'.join(map(str, DOS_KGRID))}, tetrahedra)")
    if problems and any(p.startswith("output_dir") for p in problems):
        return
    read = tool_ok(call, "qe_read_dos[the run's dos.dat]", "qe_read_dos", {"output_dir": data["dos_file"]})
    if read is not None:
        verdict(report, "qe_read_dos[the run's dos.dat]",
                mismatches(read, {"n_points": len(dos["energies"]), "energies": dos["energies"], "dos": dos["dos"],
                                  "integrated_dos": dos["integrated"], "fermi_energy": dos["fermi_ev"]},
                           {"energies": ARRAY_TOL, "dos": ARRAY_TOL, "integrated_dos": ARRAY_TOL,
                            "fermi_energy": ARRAY_TOL}),
                "equal to the reference run's dos.dat")
    check_directory_reader(session, "qe_read_dos", folder, "dos.dat", {"success": True, "n_points": len(dos["energies"])})


def check_relax_and_scf(session: Session, ref: QERef, relax: dict) -> None:
    call, report = session.call, session.report
    name = "qe_workflow_relax_and_scf"
    data = tool_ok(call, name, "qe_workflow_relax_and_scf",
                   dft_args(inline_structure(PERTURBED_ATOMS, FCC_CELL), ecut=False))
    if data is None:
        return
    unrelaxed = ref.tight_scf("scf-unrelaxed", PERTURBED_ATOMS)
    problems = check_artifacts(session, name, data, "relax_scf", ref.pseudo_file)
    first = step_values(relax, 0)
    problems += mismatches(data.get("relaxation") or {},
                           {"success": True, "converged": relax["bfgs_converged"]}, {})
    verdict(report, f"{name}[relaxation]", problems,
            f"BFGS converged {relax['bfgs_converged']} on the cutoffs of the hint table ({SI_ECUTWFC}/{SI_ECUTRHO} Ry)")
    reported = (data.get("relaxation") or {}).get("energy_eV")
    three_state(report, f"{name}[relaxation energy]",
                [] if close(reported, relax["final_ry"] * RY_EV, ENERGY_TOL_EV) else [f"{reported}"],
                [] if close(reported, first["total_energy_eV"], ENERGY_TOL_EV) else [f"{reported}"],
                f"{reported} eV, the relaxed energy",
                f"D11: relaxation.energy_eV {reported} is the first SCF step, not the final {relax['final_ry'] * RY_EV:.6f} eV")
    if relax["final_positions"] is None:
        report.add("L1", f"{name}[final SCF]", "FAIL", "the reference relaxation printed no final coordinates")
        return
    relaxed = ref.tight_scf("scf-relaxed", relax["final_positions"])
    view = {k: data.get(k) for k in ("total_energy_Ry", "total_energy_eV", "fermi_energy_eV")}
    keys = ("total_energy_Ry", "total_energy_eV", "fermi_energy_eV")
    want_relaxed = {k: step_values(relaxed, 0)[k] for k in keys}
    want_unrelaxed = {k: step_values(unrelaxed, 0)[k] for k in keys}
    three_state(report, f"{name}[final SCF]",
                mismatches(view, want_relaxed, STEP_TOLERANCES), mismatches(view, want_unrelaxed, STEP_TOLERANCES),
                f"SCF of the relaxed structure: {want_relaxed['total_energy_Ry']:.8f} Ry",
                f"D12: the final SCF is of the input geometry ({want_unrelaxed['total_energy_Ry']:.8f} Ry at "
                f"conv_thr {TIGHT_CONV_THR}), not of the relaxed one ({want_relaxed['total_energy_Ry']:.8f} Ry)",
                server_value=view)


# --------------------------------------------------------------------------
# L1: environment-dependent tools
# --------------------------------------------------------------------------

MP_KEY_ERRORS = {
    "qe_search_materials_project": "MP_API_KEY not set",
    "qe_get_mp_structure": "Materials Project API key not found. Set MP_API_KEY environment variable "
                           "or use: export MP_API_KEY='your_key'",
}


def job_status_error(workdir: Path):
    """``qe_get_job_status`` with the local runner: the exact 'not found in registry' error."""
    prefix = f"Job '{MISSING_JOB}' not found in registry at "

    def match(result: dict) -> str | None:
        message = in_band_error(result) or ""
        if message.startswith(prefix) and message.endswith(".") \
                and Path(message[len(prefix):-1]).resolve() == workdir.resolve():
            return f"local runner, no async jobs: {message}"
        return None
    return match


def exact_error(expected: str):
    """An ``in_band`` hook that only accepts this exact error message."""
    return lambda result: (message if (message := in_band_error(result)) == expected else None)


def check_environment_tools(session: Session) -> None:
    """The three tools that cannot work in this environment answer with their documented errors.
    The smoke's server environment carries no ``MP_API_KEY``, so no request is made."""
    call = session.call
    check_rejected(call, "qe_get_job_status[local runner]", "qe_get_job_status", {"job_id": MISSING_JOB},
                   in_band=job_status_error(session.cwd / "qe_calculations"))
    check_rejected(call, "qe_search_materials_project[no MP_API_KEY]", "qe_search_materials_project",
                   {"query": "Si", "num_results": 1}, in_band=exact_error(MP_KEY_ERRORS["qe_search_materials_project"]))
    check_rejected(call, "qe_get_mp_structure[no MP_API_KEY]", "qe_get_mp_structure", {"mp_id": "mp-149"},
                   in_band=exact_error(MP_KEY_ERRORS["qe_get_mp_structure"]))


# --------------------------------------------------------------------------

def run_l1(session: Session) -> None:
    report = session.report
    check_status(session)
    pick = check_pseudopotentials(session)
    check_structures(session)
    check_validate(session)
    check_suggest_kpoints(session)
    check_kpath(session)
    check_readers_synthetic(session)
    check_environment_tools(session)
    if pick is None:
        report.add("L1", "DFT tools", "FAIL", "no usable Si pseudopotential pick; the DFT references need it")
        return
    ref = QERef(session, pick)
    try:
        check_scf(session, ref)
        relax = ref.relax()
        check_relaxation(session, ref, "qe_run_relax", "relax", relax,
                         inline_structure(PERTURBED_ATOMS, FCC_CELL), len(PERTURBED_ATOMS))
        check_relaxation(session, ref, "qe_run_vc_relax", "vc-relax", ref.vc_relax(), STRUCTURE, len(SI_ATOMS))
        check_bandstructure(session, ref)
        check_dos(session, ref)
        check_relax_and_scf(session, ref, relax)
    except (RuntimeError, OSError, subprocess.SubprocessError, ValueError) as exc:
        report.add("L1", "DFT references", "FAIL", f"a reference run failed: {exc}")


def report_fields(session: Session) -> dict:
    return {"qe_versions": session.state.get("qe_versions"),
            "sg15_elements": session.state.get("elements"),
            "sg15_pick": session.state.get("pseudo_pick"),
            "workload": {"structure": STRUCTURE, "ecutwfc_ry": SI_ECUTWFC, "ecutrho_ry": SI_ECUTRHO,
                         "kpoints": list(KGRID), "npoints_band": NPOINTS_BAND, "dos_kpoints": list(DOS_KGRID)},
            "seconds_by_step": dict(TIMINGS)}


SMOKE = Smoke(
    server="quantum_espresso",
    run_l1=run_l1,
    packages=("mcp", "ase", "numpy", "spglib", "pyyaml", "python-dotenv"),
    call_timeout=CALL_TIMEOUT,
    prepare=prepare,
    # QE_WORKDIR is deliberately unset in the manifest, so the server's work
    # directory is <cwd>/qe_calculations: the smoke's temporary cwd here (and
    # the checkout in an agent run, where upstream's .gitignore covers it).
    expected_cwd_files=("qe_calculations",),
    report_fields=report_fields,
)
