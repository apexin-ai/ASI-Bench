"""Direct (agent-free) E2E smoke test for the pinned mcp-atomictoolkit (ASE + pymatgen) tools.

Run with the server's own virtualenv (it provides ASE and spglib for the references)::

    ~/mcp/atomictoolkit/.venv/bin/python scripts/mcp/e2e/smoke.py atomictoolkit \\
        --config ~/mcp/atomictoolkit.mcp.json

No network, no key, no GPU. The catalog's ``ase`` and ``pymatgen`` entries are
this one server (pymatgen only supplies the spacegroup strings of
``get_structure_info``). Every number the server returns comes from ASE's EMT
potential or ASE geometry, so the references are computed here, in this
process, with ASE and spglib called directly and never with server code:

    build / import / write   the ASE builders, readers and our own numpy
                             geometry, compared with the file the tool wrote
                             (symbols, cell, positions, pbc) and with spglib's
                             spacegroup at pymatgen's default tolerances
    manipulate               all eight operations re-done in numpy
    single_point             ``EMT()`` energy / forces / stress on the file
    estimate_elastic         five EMT energies and a closed-form quadratic
                             fit (not ``np.polyfit``), B = 2a / (9 V0)
    analyze_structure        g(r) from ``neighbor_list`` distances with the
                             textbook normalisation, coordination numbers from
                             ``NeighborList`` with skin 0
    create_download_artifact the in-memory artifact index (ids, URLs, previews)

Of the 18 tools in ``tools/list`` only 9 can be used. The other 9 are
covered as recognised defects, at the pinned revision exactly:

* D1 the four deprecated wrappers (``build_structure``, ``read_structure_file``,
  ``write_structure_file``, ``optimize_with_mlip``) await a FastMCP
  ``FunctionTool`` and always fail with ``'FunctionTool' object is not callable``;
* D2 the five ``TaskConfig(mode="required")`` tools (optimize, MD, relax+MD,
  trajectory analysis, autocorrelation) refuse a plain ``tools/call``; a
  task-augmented call (``params.task``, probed in a separate server after the
  main run) is accepted and its side effects happen, but ``tasks/get`` and
  ``tasks/result`` always answer ``No active context found.`` — the result can
  never be fetched. D8: the task path's BFGS log goes to stdout.

Further recognised defects: D3 an error message longer than a path component
crashes the artifact scan (``[Errno 36] File name too long``) and the
structured error report is lost; D4 coordination numbers use ASE's default
neighbour-list skin (0.3 Å), so fcc Cu reports 18 instead of 12; D5 a
cell-less molecule from the server's own builder fails analysis; D6 ``kim`` is
listed as available but cannot be initialised, so every ``calculator_name="auto"``
call carries three ``calculator_fallbacks``; D9 g(r) is normalised by N/2
instead of N, i.e. it tends to 2, not 1. Each is three-state: the correct
behaviour PASS, exactly the defect WARN, anything else FAIL.

Every other in-band error (``{"status": "error"}`` with isError=false) is a
WARN, as everywhere in this bundle; so are ``create_download_artifact``
accepting a missing or non-artifact file silently, and its relative
``download_url``, which only the HTTP app serves.

Not defects but worth knowing (detail text only): ``estimate_elastic``'s B
is the curvature of E(strain) at the given cell, including the pressure term
when the cell is not at equilibrium; relative paths and every in-band error's
``tool_errors/<tool>_<UTC>.log`` land in the server cwd, which is the MCP
checkout in agent runs.
"""
from __future__ import annotations

import io
import json
import math
import re
import time
from pathlib import Path
from typing import Callable

import numpy as np

from ..client import MCPError, StdioMCP, text_of
from ..helpers import png_size
from ..runner import Caller, Report, Session, Smoke, check_rejected

# --------------------------------------------------------------------------
# Inputs, defect markers, tolerances
# --------------------------------------------------------------------------

CU_A = 3.61                   # fcc Cu lattice constant used throughout (Å)
EMT_ELEMENTS = ["Ag", "Al", "Au", "Cu", "Ni", "Pd", "Pt"]
AUTO_ORDER = ["kim", "orb", "nequix", "emt"]
INTEGRATORS = ["velocityverlet", "nve", "langevin", "nvt-langevin", "nvt", "nvt-berendsen", "npt",
               "npt-berendsen"]
STRUCTURE_TYPES = ["bulk", "surface", "molecule", "supercell", "amorphous", "liquid", "bicrystal",
                   "polycrystal"]
OPERATIONS = ["rotate", "translate", "strain", "supercell", "wrap", "vacancy", "substitute", "interstitial"]
ART_ID_RE = re.compile(r"art_[0-9a-f]{32}")
STRUCTURE_SUFFIXES = (".xyz", ".extxyz", ".cif", ".vasp", ".poscar")

DEPRECATED = ("build_structure", "read_structure_file", "write_structure_file", "optimize_with_mlip")
TASK_REQUIRED = ("analyze_trajectory_workflow", "autocorrelation_workflow", "optimize_structure_workflow",
                 "relax_and_md_workflow", "run_md_workflow")
FUNCTION_TOOL = "'FunctionTool' object is not callable"
NO_CONTEXT = "No active context found."
NAME_TOO_LONG = "[Errno 36] File name too long"
ZERO_LATTICE = "You have 0 lattice vectors: volume not defined"
TASK_META = "modelcontextprotocol.io/task"
BFGS_LINE = re.compile(r"\s*Step\s+Time\s+Energy\s+fmax\s*|BFGS:\s+\d+\s+\d\d:\d\d:\d\d\s+\S+\s+\S+")
KIM_DEFAULT_MODEL = "LJ_ElliottAkerson_2015_Universal__MO_959249795837_003"

# pymatgen's SpacegroupAnalyzer defaults, used for the spglib reference.
SYMPREC, ANGLE_TOLERANCE = 0.01, 5.0
COVALENT_FACTOR = 1.2         # analyze_structure_workflow default coordination_factor
UPSTREAM_SKIN = 0.3           # ASE NeighborList default skin, which the server leaves in place (D4)

# Same library on the same host: the server and this process agree bit for bit
# (survey: every value identical). The tolerances only leave room for a BLAS
# with a different summation order.
ENERGY_TOL = 1e-10            # eV
FORCE_TOL = 1e-10             # eV/Å
STRESS_TOL = 1e-12            # eV/Å^3
POS_TOL = 1e-7                # Å: extxyz stores 8 decimals
CELL_TOL = 1e-7               # Å
B_RTOL = 1e-9                 # bulk modulus, np.polyfit (LAPACK) vs our closed form
RDF_TOL = 1e-9                # g(r), relative to max(1, |g|)
TASK_WAIT = 60.0              # s for a task-augmented call's side effect to appear


# --------------------------------------------------------------------------
# Pure helpers (numpy only; unit-tested offline)
# --------------------------------------------------------------------------

def payload(result: dict) -> dict:
    """The business dict of a tools/call result. Every tool declares a bare ``object``
    outputSchema, so ``structuredContent`` is the dict itself (no ``result`` wrapper)."""
    data = result.get("structuredContent")
    if not isinstance(data, dict):
        data = json.loads(text_of(result))
    if not isinstance(data, dict):
        raise ValueError("result is not a JSON object")
    return data


def in_band_error(result: dict) -> str | None:
    """``{"status": "error", "error": {...}}`` returned with isError=false (``_tool_error_response``)."""
    try:
        data = payload(result)
    except (ValueError, json.JSONDecodeError):
        return None
    if data.get("status") != "error":
        return None
    error = data.get("error") or {}
    return f"{error.get('type')}: {error.get('message')}"


def without_artifacts(data: dict) -> dict:
    return {k: v for k, v in data.items() if k not in ("artifacts", "artifact_notes")}


def artifact_problems(data: dict, expected: dict[str, Path]) -> list[str]:
    """Problems with the ``artifacts`` list of a result, given label -> file for every
    existing file path the result carries (``with_downloadable_artifacts``): one entry per
    label with a fresh ``art_<32 hex>`` id, the resolved file and ``/artifacts/<id>/<name>``,
    plus a ``<label>_preview_html`` entry for structure files."""
    want: dict[str, Path] = {}
    for label, path in expected.items():
        path = Path(path).resolve()
        want[label] = path
        if path.suffix.lower() in STRUCTURE_SUFFIXES:
            want[f"{label}_preview_html"] = path.with_suffix(path.suffix + ".preview.html")
    artifacts = data.get("artifacts", [])
    if not want:
        return [f"unexpected artifacts {artifacts}"] if artifacts else []
    if not isinstance(artifacts, list):
        return [f"artifacts is {type(artifacts).__name__}"]
    problems = []
    got = {a.get("label"): a for a in artifacts if isinstance(a, dict)}
    if sorted(got) != sorted(want) or len(got) != len(artifacts):
        problems.append(f"artifact labels {sorted(a.get('label') for a in artifacts)}, expected {sorted(want)}")
    ids = [a.get("id") for a in artifacts if isinstance(a, dict)]
    if len(set(ids)) != len(ids):
        problems.append("artifact ids are not unique")
    for label, path in want.items():
        art = got.get(label)
        if art is None:
            continue
        aid = art.get("id")
        if not isinstance(aid, str) or not ART_ID_RE.fullmatch(aid):
            problems.append(f"{label}: id {aid!r}")
        if art.get("filepath") != str(path):
            problems.append(f"{label}: filepath {art.get('filepath')!r}, expected {str(path)!r}")
        elif not path.is_file():
            problems.append(f"{label}: {path.name} does not exist")
        if art.get("download_url") != f"/artifacts/{aid}/{path.name}":
            problems.append(f"{label}: download_url {art.get('download_url')!r}")
    return problems


def crystal_system(number: int) -> str:
    """Crystal system of an international spacegroup number (as pymatgen names it)."""
    for upper, name in ((2, "triclinic"), (15, "monoclinic"), (74, "orthorhombic"), (142, "tetragonal"),
                        (167, "trigonal"), (194, "hexagonal"), (230, "cubic")):
        if number <= upper:
            return name
    raise ValueError(f"spacegroup number {number}")


def quadratic_curvature(strains: list[float], energies: list[float]) -> float:
    """Least-squares a of E = a s^2 + b s + c on a grid symmetric about 0.

    On a symmetric grid (sum s = sum s^3 = 0) the s term decouples, so a is the
    slope of E against u = s^2: a = sum (u - <u>) E / sum (u - <u>)^2. Closed form,
    deliberately not ``np.polyfit``."""
    s = [float(x) for x in strains]
    if len(s) < 3 or abs(sum(s)) > 1e-15 or abs(sum(x ** 3 for x in s)) > 1e-15:
        raise ValueError("strain grid must be symmetric about 0 with at least 3 points")
    u = [x * x for x in s]
    mean = sum(u) / len(u)
    return sum((ui - mean) * e for ui, e in zip(u, energies)) / sum((ui - mean) ** 2 for ui in u)


def elastic_strains(strain_max: float) -> list[float]:
    return [-strain_max, -0.5 * strain_max, 0.0, 0.5 * strain_max, strain_max]


def rdf_reference(distances, n_atoms: int, volume: float, r_max: float, bins: int):
    """Textbook g(r): ``distances`` lists every ordered pair (i, j), i != j, within r_max
    (both directions, as ``neighbor_list('d')``), so g = hist / (N rho 4 pi r^2 dr) -> 1."""
    hist, edges = np.histogram(np.asarray(distances, dtype=float), bins=bins, range=(0.0, r_max))
    r = 0.5 * (edges[1:] + edges[:-1])
    dr = edges[1] - edges[0]
    rho = n_atoms / volume
    return r, hist / (4.0 * np.pi * r ** 2 * dr * rho * n_atoms)


def relative_error(got, want) -> float:
    """max |got - want| / max(1, |want|) over equal-shape arrays; inf on a shape mismatch."""
    try:
        g, w = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    except (TypeError, ValueError):
        return math.inf
    if g.shape != w.shape:
        return math.inf
    if g.size == 0:
        return 0.0
    return float(np.max(np.abs(g - w) / np.maximum(1.0, np.abs(w))))


def abs_error(got, want) -> float:
    try:
        g, w = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    except (TypeError, ValueError):
        return math.inf
    if g.shape != w.shape:
        return math.inf
    return float(np.max(np.abs(g - w))) if g.size else 0.0


def read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    rows = [line.split(",") for line in path.read_text(encoding="utf-8").splitlines() if line]
    return rows[0], rows[1:]


def classify(result: dict | None, *, correct: Callable[[dict], str | None],
             defect: Callable[[dict], str | None]) -> tuple[str, str]:
    """Three-state probe: ``correct(result)`` returns a detail if the tool behaved correctly
    (PASS), ``defect(result)`` one if the result is exactly the recognised defect (WARN);
    anything else is FAIL."""
    if result is None:
        return "FAIL", "no result"
    detail = correct(result)
    if detail is not None:
        return "PASS", detail
    detail = defect(result)
    if detail is not None:
        return "WARN", detail
    return "FAIL", (f"neither correct nor the recognised defect at the pinned revision: "
                    f"isError={result.get('isError')} {text_of(result)[:240]!r}")


def is_error_text(result: dict, text: str) -> bool:
    return result.get("isError") is True and text_of(result) == text


def fallback_prefixes(fallbacks) -> list[str]:
    return [f.split(":", 1)[0] for f in fallbacks] if isinstance(fallbacks, list) else []


# --------------------------------------------------------------------------
# References (ASE / spglib, imported lazily: the offline tests have neither)
# --------------------------------------------------------------------------

class Ref:
    """ASE and spglib, called directly in this process."""

    def __init__(self) -> None:
        import ase
        import ase.build
        import ase.io
        import spglib
        from ase import units
        from ase.calculators.emt import EMT
        from ase.data import covalent_radii
        from ase.neighborlist import NeighborList, neighbor_list
        self.ase, self.build, self.io, self.spglib = ase, ase.build, ase.io, spglib
        self.GPa, self.EMT, self.radii = units.GPa, EMT, covalent_radii
        self.NeighborList, self.neighbor_list = NeighborList, neighbor_list

    def read(self, path: Path):
        path = Path(path)
        return self.io.read(path, format=path.suffix[1:])

    def bulk(self, symbol: str, kind: str, a: float, cubic: bool = True, size=None):
        atoms = self.build.bulk(symbol, kind, a=a, cubic=cubic)
        return atoms.repeat(size) if size else atoms

    def point(self, atoms) -> dict:
        """EMT single point on a copy (energy, forces, stress if fully periodic)."""
        atoms = atoms.copy()
        atoms.calc = self.EMT()
        return {"energy": float(atoms.get_potential_energy()), "forces": atoms.get_forces().tolist(),
                "stress": atoms.get_stress().tolist() if all(atoms.pbc) else None}

    def symmetry(self, atoms) -> dict:
        """spacegroup / crystal system / point group, or all None without a full periodic cell."""
        if not all(atoms.pbc) or abs(atoms.cell.volume) < 1e-12:
            return {"spacegroup": None, "crystal_system": None, "point_group": None}
        cell = (atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers)
        data = self.spglib.get_symmetry_dataset(cell, symprec=SYMPREC, angle_tolerance=ANGLE_TOLERANCE)
        return {"spacegroup": data.international, "crystal_system": crystal_system(data.number),
                "point_group": data.pointgroup}

    def coordination(self, atoms, skin: float) -> list[int]:
        cutoffs = [self.radii[n] * COVALENT_FACTOR for n in atoms.numbers]
        nl = self.NeighborList(cutoffs, skin=skin, self_interaction=False, bothways=True)
        nl.update(atoms)
        return [len(nl.get_neighbors(i)[0]) for i in range(len(atoms))]

    def rdf(self, atoms, r_max: float, bins: int):
        return rdf_reference(self.neighbor_list("d", atoms, cutoff=r_max), len(atoms), atoms.get_volume(),
                             r_max, bins)

    def elastic(self, atoms, strain_max: float) -> dict:
        samples = []
        for s in elastic_strains(strain_max):
            strained = atoms.copy()
            strained.set_cell(atoms.cell.array * (1.0 + s), scale_atoms=True)
            strained.calc = self.EMT()
            samples.append({"strain": s, "energy_eV": float(strained.get_potential_energy()),
                            "volume_A3": float(strained.get_volume())})
        curvature = quadratic_curvature([r["strain"] for r in samples], [r["energy_eV"] for r in samples])
        volume0 = samples[2]["volume_A3"]
        return {"samples": samples, "volume_A3": volume0,
                "bulk_modulus_GPa": 2.0 * curvature / (9.0 * volume0) / self.GPa}

    def text(self, atoms, fmt: str) -> str:
        if fmt == "cif":                  # ASE's CIF writer wants a binary stream
            raw = io.BytesIO()
            self.io.write(raw, atoms, format=fmt)
            return raw.getvalue().decode("utf-8")
        buffer = io.StringIO()
        self.io.write(buffer, atoms, format=fmt)
        return buffer.getvalue()

    def kim_available(self) -> tuple[bool, str]:
        """Whether a KIM calculator can really be constructed here (what ``auto`` tries first)."""
        try:
            from ase.calculators.kim.kim import KIM
            KIM(KIM_DEFAULT_MODEL)
        except Exception as exc:          # noqa: BLE001 - any failure means unavailable
            return False, f"{type(exc).__name__}: {exc}"[:160]
        return True, ""


def structure_problems(ref: Ref, path: Path, want) -> list[str]:
    """Differences between the structure file ``path`` and the reference Atoms ``want``."""
    try:
        got = ref.read(path)
    except Exception as exc:              # noqa: BLE001 - an unreadable file is the finding
        return [f"{Path(path).name} unreadable: {exc}"]
    problems = []
    if got.get_chemical_symbols() != want.get_chemical_symbols():
        problems.append(f"symbols {got.get_chemical_formula()} vs {want.get_chemical_formula()}")
    if list(got.pbc) != list(want.pbc):
        problems.append(f"pbc {got.pbc.tolist()} vs {want.pbc.tolist()}")
    if abs_error(got.cell.array, want.cell.array) > CELL_TOL:
        problems.append(f"cell {got.cell.array.tolist()} vs {want.cell.array.tolist()}")
    err = abs_error(got.positions, want.positions)
    if err > POS_TOL:
        problems.append(f"positions differ by {err:.1e} Å")
    return problems


def info_problems(ref: Ref, data: dict, want, *, keys=("formula", "num_atoms", "cell")) -> list[str]:
    """The metadata a build/manipulate/import result echoes, against the reference Atoms."""
    expected = {"formula": want.get_chemical_formula(), "num_atoms": len(want), "cell": want.cell.array.tolist()}
    problems = []
    for key in keys:
        value = data.get(key)
        bad = abs_error(value, expected[key]) > CELL_TOL if key == "cell" else value != expected[key]
        if bad:
            problems.append(f"{key} {value!r}, expected {expected[key]!r}")
    return problems


# --------------------------------------------------------------------------
# Tool-call helpers
# --------------------------------------------------------------------------

def ok(call: Caller, name: str, tool: str, arguments: dict) -> dict | None:
    """A business payload; FAIL (and None) on isError, an in-band error or a non-object."""
    result = call(name, tool, arguments)
    if result is None:
        return None
    message = in_band_error(result)
    if message:
        call.report.add("L1", name, "FAIL", f"in-band error: {message[:300]}")
        return None
    try:
        return payload(result)
    except (ValueError, json.JSONDecodeError) as exc:
        call.report.add("L1", name, "FAIL", f"{exc}: {text_of(result)[:200]!r}")
        return None


def verdict(report: Report, name: str, problems: list[str], detail: str, **data) -> None:
    report.add("L1", name, "FAIL" if problems else "PASS", "; ".join(problems) if problems else detail, **data)


class Ctx:
    """What the checks share: the caller, the report, the references and a scratch dir."""

    def __init__(self, session: Session, ref: Ref) -> None:
        self.session, self.call, self.report, self.ref = session, session.call, session.report, ref
        self.work = session.tmp / "work"
        self.work.mkdir(exist_ok=True)
        self.files: dict[str, Path] = {}     # name -> structure file written by a tool

    def path(self, name: str) -> Path:
        return self.work / name


# --------------------------------------------------------------------------
# Checks: the nine usable tools
# --------------------------------------------------------------------------

def check_capabilities(ctx: Ctx) -> None:
    name = "list_workspace_capabilities_workflow"
    data = ok(ctx.call, name, name, {})
    if data is None:
        return
    calculators = data.get("calculators") or {}
    expected = {"default_calculator": "auto", "auto_order": AUTO_ORDER, "integrators": INTEGRATORS,
                "structure_types": STRUCTURE_TYPES, "manipulate_operations": OPERATIONS}
    problems = [f"{k} {data.get(k)!r}, expected {v!r}" for k, v in expected.items() if data.get(k) != v]
    emt = calculators.get("emt") or {}
    if emt.get("available") is not True or emt.get("supported_elements") != EMT_ELEMENTS:
        problems.append(f"emt {emt}")
    for key in ("orb", "nequix"):     # not installed by the manifest pins
        entry = calculators.get(key) or {}
        if entry.get("available") is not False or "No module named" not in str(entry.get("error")):
            problems.append(f"{key} {entry}")
    verdict(ctx.report, f"{name} (static lists, emt, orb/nequix unavailable)", problems,
            "auto order kim>orb>nequix>emt; EMT elements " + ",".join(EMT_ELEMENTS))
    kim = calculators.get("kim") or {}
    really, why = ctx.ref.kim_available()
    status, detail = classify({"isError": False, "content": []},
                              correct=lambda _: f"kim available={kim.get('available')} matches a real KIM "
                                                f"construction" if kim.get("available") is really else None,
                              defect=lambda _: (f"D6: kim reported {{'available': true}} but KIM() cannot be "
                                                f"constructed ({why}); the probe only imports the ASE class, so "
                                                "every calculator_name='auto' call tries kim, orb and nequix "
                                                "before emt") if kim == {"available": True} and not really else None)
    ctx.report.add("L1", f"{name}[kim availability]", status, detail)


def check_build(ctx: Ctx) -> None:
    ref, report = ctx.ref, ctx.report
    cases = (
        ("fcc Cu", "cu.extxyz", {"formula": "Cu", "crystal_system": "fcc", "lattice_constant": CU_A},
         ref.bulk("Cu", "fcc", CU_A)),
        ("bcc Fe", "fe.extxyz", {"formula": "Fe", "crystal_system": "bcc", "lattice_constant": 2.87},
         ref.bulk("Fe", "bcc", 2.87)),
        ("hcp Mg (non-cubic default)", "mg.extxyz",
         {"formula": "Mg", "crystal_system": "hcp", "lattice_constant": 3.21},
         ref.bulk("Mg", "hcp", 3.21, cubic=False)),
        ("diamond Si, cif", "si.cif", {"formula": "Si", "crystal_system": "diamond", "lattice_constant": 5.43},
         ref.bulk("Si", "diamond", 5.43)),
        ("supercell type", "ni_sc.extxyz",
         {"formula": "Ni", "structure_type": "supercell", "crystal_system": "fcc", "lattice_constant": 3.52,
          "builder_kwargs": {"size": [1, 2, 3]}}, ref.bulk("Ni", "fcc", 3.52, size=(1, 2, 3))),
        ("molecule H2O", "h2o.extxyz", {"formula": "H2O", "structure_type": "molecule"}, None),
    )
    for label, filename, arguments, want in cases:
        if want is None:
            want = ref.build.molecule("H2O")
            want.pbc = True               # the tool's default pbc, on a cell-less molecule
        path = ctx.path(filename)
        name = f"build_structure_workflow[{label}]"
        data = ok(ctx.call, name, "build_structure_workflow", {**arguments, "output_filepath": str(path)})
        if data is None:
            continue
        problems = info_problems(ref, data, want) + structure_problems(ref, path, want)
        sym = ref.symmetry(want)
        if data.get("symmetry") != sym:
            problems.append(f"symmetry {data.get('symmetry')}, spglib {sym}")
        if data.get("filepath") != str(path) or data.get("format") != path.suffix[1:]:
            problems.append(f"filepath/format {data.get('filepath')!r} {data.get('format')!r}")
        problems += artifact_problems(data, {"filepath": path})
        verdict(report, name, problems, f"{data.get('formula')}, {sym['spacegroup']}, file and artifacts match")
        ctx.files[filename] = path


def check_import(ctx: Ctx) -> None:
    ref = ctx.ref
    dimer = ref.ase.Atoms("Cu2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 2.4]])
    al = ref.bulk("Al", "fcc", 4.05)
    cases = (("xyz Cu2 dimer", "xyz", ref.text(dimer, "xyz"), "cu2.extxyz", dimer),
             ("cif Al", "cif", ref.text(al, "cif"), "al.extxyz", al),
             ("poscar Cu", "vasp", ref.text(ref.bulk("Cu", "fcc", CU_A), "vasp"), "cu_poscar.extxyz",
              ref.bulk("Cu", "fcc", CU_A)))
    for label, fmt, contents, filename, want in cases:
        path = ctx.path(filename)
        name = f"import_structure_workflow[{label}]"
        data = ok(ctx.call, name, "import_structure_workflow",
                  {"contents": contents, "input_format": fmt, "output_filepath": str(path)})
        if data is None:
            continue
        expected = ref.io.read(io.StringIO(contents), format=fmt)   # what the text means
        problems = info_problems(ref, data, expected) + structure_problems(ref, path, expected)
        if fmt == "cif":
            problems += [f"cif cell lengths {expected.cell.lengths().tolist()}"] \
                if abs_error(expected.cell.lengths(), al.cell.lengths()) > CELL_TOL else []
        if data.get("status") != "success" or data.get("input_format") != fmt:
            problems.append(f"status/input_format {data.get('status')!r} {data.get('input_format')!r}")
        problems += artifact_problems(data, {"filepath": path})
        verdict(ctx.report, name, problems, f"{data.get('formula')}, pbc {expected.pbc.tolist()}")
        ctx.files[filename] = path


def check_write(ctx: Ctx) -> None:
    ref = ctx.ref
    want = ref.bulk("Pt", "fcc", 3.92)
    for fmt, filename in ((None, "pt.extxyz"), ("cif", "pt_written.cif")):
        path = ctx.path(filename)
        name = f"write_structure_workflow[{fmt or 'from suffix'}]"
        arguments = {"positions": want.positions.tolist(), "symbols": want.get_chemical_symbols(),
                     "cell": want.cell.array.tolist(), "filepath": str(path), **({"format": fmt} if fmt else {})}
        data = ok(ctx.call, name, "write_structure_workflow", arguments)
        if data is None:
            continue
        problems = structure_problems(ref, path, want)
        if without_artifacts(data) != {"status": "success", "filepath": str(path), "format": path.suffix[1:]}:
            problems.append(f"payload {without_artifacts(data)}")
        problems += artifact_problems(data, {"filepath": path})
        verdict(ctx.report, name, problems, "file round-trips (pbc forced true)")
        ctx.files[filename] = path


def _rotate_z(atoms, angle: float):
    out = atoms.copy()
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    centre = atoms.positions.mean(axis=0)
    out.positions = (atoms.positions - centre) @ np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]) + centre
    return out


def _translate(atoms, vector):
    out = atoms.copy()
    out.positions = atoms.positions + np.asarray(vector, dtype=float)
    return out


def _wrap(atoms):
    """Fractional coordinates into [0, 1); the inputs below keep clear of the faces."""
    out = atoms.copy()
    frac = np.linalg.solve(atoms.cell.array.T, atoms.positions.T).T % 1.0
    out.positions = frac @ atoms.cell.array
    return out


def _strain(atoms, strain: float):
    out = atoms.copy()
    frac = np.linalg.solve(atoms.cell.array.T, atoms.positions.T).T % 1.0
    out.cell = atoms.cell.array * (1.0 + strain)
    out.positions = frac @ out.cell.array
    return out


def _vacancy(atoms, index: int):
    out = atoms.copy()
    del out[index]
    return out


def _substitute(atoms, index: int, symbol: str):
    out = atoms.copy()
    symbols = out.get_chemical_symbols()
    symbols[index] = symbol
    out.set_chemical_symbols(symbols)
    return out


def _interstitial(atoms, symbol: str, position):
    out = atoms.copy()
    out.append(symbol)
    out.positions[-1] = position
    return out


SHIFT = [0.6 * CU_A, 0.3 * CU_A, 0.85 * CU_A]    # fractional 0.6/1.1, 0.3/0.8, 0.85/1.35: clear of the faces


def check_manipulate(ctx: Ctx) -> None:
    ref = ctx.ref
    if "cu.extxyz" not in ctx.files:
        ctx.report.add("L1", "manipulate_structure_workflow", "FAIL", "no fcc Cu file from build_structure_workflow")
        return
    octa = [CU_A / 2, CU_A / 2, CU_A / 2]
    # (label, input file, operation, kwargs, output file, reference from the input Atoms)
    cases = (
        ("supercell 2x2x2", "cu.extxyz", "supercell", {"size": [2, 2, 2]}, "cu32.extxyz",
         lambda a: a.repeat((2, 2, 2))),
        ("translate", "cu.extxyz", "translate", {"vector": SHIFT}, "cu_shifted.extxyz", lambda a: _translate(a, SHIFT)),
        ("wrap", "cu_shifted.extxyz", "wrap", {}, "cu_wrapped.extxyz", _wrap),
        ("strain 3%", "cu_shifted.extxyz", "strain", {"strain": 0.03}, "cu_strained.extxyz",
         lambda a: _strain(a, 0.03)),
        ("rotate 90 deg about z (COP)", "cu.extxyz", "rotate", {"angle": 90, "axis": "z"}, "cu_rotated.extxyz",
         lambda a: _rotate_z(a, 90.0)),
        ("vacancy index 5", "cu32.extxyz", "vacancy", {"index": 5}, "cu31.extxyz", lambda a: _vacancy(a, 5)),
        ("substitute index 0 -> Ni", "cu32.extxyz", "substitute", {"index": 0, "symbol": "Ni"}, "cu31ni.extxyz",
         lambda a: _substitute(a, 0, "Ni")),
        ("interstitial H at the octahedral site", "cu.extxyz", "interstitial", {"symbol": "H", "position": octa},
         "cu4h.extxyz", lambda a: _interstitial(a, "H", octa)),
    )
    for label, source, operation, kwargs, filename, make in cases:
        if source not in ctx.files:
            ctx.report.add("L1", f"manipulate_structure_workflow[{label}]", "FAIL", f"no input {source}")
            continue
        src, path = ctx.files[source], ctx.path(filename)
        name = f"manipulate_structure_workflow[{label}]"
        data = ok(ctx.call, name, "manipulate_structure_workflow",
                  {"input_filepath": str(src), "operation": operation, "operation_kwargs": kwargs,
                   "output_filepath": str(path)})
        if data is None:
            continue
        want = make(ref.read(src))
        problems = info_problems(ref, data, want) + structure_problems(ref, path, want)
        echo = {"status": data.get("status"), "operation": data.get("operation"),
                "input_filepath": data.get("input_filepath"), "filepath": data.get("filepath")}
        if echo != {"status": "success", "operation": operation, "input_filepath": str(src), "filepath": str(path)}:
            problems.append(f"echo {echo}")
        problems += artifact_problems(data, {"input_filepath": src, "filepath": path})
        verdict(ctx.report, name, problems, f"{data.get('formula')}: file matches the numpy reference")
        ctx.files[filename] = path


def check_single_point(ctx: Ctx) -> None:
    ref, report = ctx.ref, ctx.report
    for filename in ("cu.extxyz", "cu32.extxyz", "cu31.extxyz", "cu31ni.extxyz", "cu2.extxyz", "al.extxyz",
                     "pt.extxyz"):
        if filename not in ctx.files:
            report.add("L1", f"single_point_workflow[{filename}]", "FAIL", "input file missing")
            continue
        path = ctx.files[filename]
        name = f"single_point_workflow[{filename}, emt]"
        data = ok(ctx.call, name, "single_point_workflow", {"input_filepath": str(path), "calculator_name": "emt"})
        if data is None:
            continue
        want = ref.point(ref.read(path))
        problems = []
        if abs(data.get("energy", math.inf) - want["energy"]) > ENERGY_TOL:
            problems.append(f"energy {data.get('energy')} vs EMT {want['energy']}")
        if abs_error(data.get("forces"), want["forces"]) > FORCE_TOL:
            problems.append("forces differ")
        if want["stress"] is None:
            if data.get("stress") is not None:
                problems.append(f"stress {data.get('stress')} for a non-periodic structure")
        elif abs_error(data.get("stress"), want["stress"]) > STRESS_TOL:
            problems.append(f"stress {data.get('stress')} vs {want['stress']}")
        meta = {k: data.get(k) for k in ("input_filepath", "calculator_requested", "calculator_used",
                                         "calculator_fallbacks")}
        if meta != {"input_filepath": str(path), "calculator_requested": "emt", "calculator_used": "emt",
                    "calculator_fallbacks": []}:
            problems.append(f"metadata {meta}")
        problems += artifact_problems(data, {"input_filepath": path})
        verdict(report, name, problems, f"E = {want['energy']!r} eV"
                + ("" if want["stress"] is not None else ", stress null (non-periodic)"), energy=data.get("energy"))
        ctx.session.state.setdefault("energies", {})[filename] = data.get("energy")
    if "cu32.extxyz" in ctx.files:
        check_repeatability(ctx, ctx.files["cu32.extxyz"])
        check_auto_calculator(ctx, ctx.files["cu32.extxyz"])


def check_repeatability(ctx: Ctx, path: Path) -> None:
    arguments = {"input_filepath": str(path), "calculator_name": "emt"}
    first = ok(ctx.call, "single_point_workflow[repeat 1]", "single_point_workflow", arguments)
    second = ok(ctx.call, "single_point_workflow[repeat 2]", "single_point_workflow", arguments)
    if first is None or second is None:
        return
    same = {k: first.get(k) for k in ("energy", "forces", "stress")} == \
        {k: second.get(k) for k in ("energy", "forces", "stress")}
    ids = [{a.get("id") for a in d.get("artifacts", [])} for d in (first, second)]
    fresh = bool(ids[0]) and not ids[0] & ids[1]
    ctx.report.add("L1", "single_point_workflow repeatability and fresh artifact ids",
                   "PASS" if same and fresh else "FAIL",
                   f"bit-identical numbers: {same}; artifact ids new on every call: {fresh} "
                   "(an id proves that one call happened, not which file it named before)")


def check_auto_calculator(ctx: Ctx, path: Path) -> None:
    name = "single_point_workflow[calculator_name=auto]"
    result = ctx.call(name, "single_point_workflow", {"input_filepath": str(path)}, allow_error=True)
    want = ctx.ref.point(ctx.ref.read(path))

    def numbers_ok(data: dict) -> bool:
        return abs(data.get("energy", math.inf) - want["energy"]) <= ENERGY_TOL

    def correct(r: dict) -> str | None:
        if r.get("isError") or in_band_error(r):
            return None
        data = payload(r)
        if data.get("calculator_used") == "emt" and data.get("calculator_fallbacks") == [] and numbers_ok(data):
            return "auto chose emt without fallbacks"
        return None

    def defect(r: dict) -> str | None:
        if r.get("isError") or in_band_error(r):
            return None
        data = payload(r)
        if data.get("calculator_used") == "emt" and fallback_prefixes(data.get("calculator_fallbacks")) == \
                ["kim", "orb", "nequix"] and numbers_ok(data):
            return ("D6: correct EMT numbers, but calculator_fallbacks carries three failure messages "
                    "(kim, orb, nequix) on every auto call; tasks must pass calculator_name='emt'")
        return None

    status, detail = classify(result, correct=correct, defect=defect)
    ctx.report.add("L1", name, status, detail)


def check_elastic(ctx: Ctx) -> None:
    ref = ctx.ref
    for filename, strain_max in (("cu.extxyz", 0.02), ("cu.extxyz", 0.01), ("cu_strained.extxyz", 0.02)):
        name = f"estimate_elastic_workflow[{filename}, strain_max={strain_max}]"
        if filename not in ctx.files:
            ctx.report.add("L1", name, "FAIL", "input file missing")
            continue
        path = ctx.files[filename]
        data = ok(ctx.call, name, "estimate_elastic_workflow",
                  {"input_filepath": str(path), "calculator_name": "emt", "strain_max": strain_max})
        if data is None:
            continue
        atoms = ref.read(path)
        want = ref.elastic(atoms, strain_max)
        problems = []
        samples = data.get("samples")
        if not isinstance(samples, list) or len(samples) != 5:
            problems.append(f"samples {samples!r}")
        else:
            if [s.get("strain") for s in samples] != [s["strain"] for s in want["samples"]]:
                problems.append(f"strains {[s.get('strain') for s in samples]}")
            if abs_error([s.get("energy_eV") for s in samples], [s["energy_eV"] for s in want["samples"]]) \
                    > ENERGY_TOL:
                problems.append("sample energies differ from EMT")
            if relative_error([s.get("volume_A3") for s in samples], [s["volume_A3"] for s in want["samples"]]) \
                    > 1e-12:
                problems.append("sample volumes differ")
        b, b_ref = data.get("bulk_modulus_GPa"), want["bulk_modulus_GPa"]
        if not isinstance(b, float) or abs(b - b_ref) > B_RTOL * abs(b_ref):
            problems.append(f"bulk_modulus_GPa {b} vs closed-form fit {b_ref}")
        meta = {k: data.get(k) for k in ("input_filepath", "calculator_used", "calculator_fallbacks", "method")}
        if meta != {"input_filepath": str(path), "calculator_used": "emt", "calculator_fallbacks": [],
                    "method": "isotropic E(strain) quadratic fit"}:
            problems.append(f"metadata {meta}")
        pressure = -sum(ref.point(atoms)["stress"][:3]) / 3 / ref.GPa if all(atoms.pbc) else None
        verdict(ctx.report, name, problems,
                f"B = {b_ref:.10g} GPa (2a/9V0 at the given cell; the cell's own pressure is {pressure:.3g} GPa, "
                "which this fit does not remove)", bulk_modulus_GPa=b)


def check_analyze(ctx: Ctx) -> None:
    ref, report = ctx.ref, ctx.report
    if "cu32.extxyz" not in ctx.files:
        report.add("L1", "analyze_structure_workflow", "FAIL", "no Cu32 file")
        return
    path, out = ctx.files["cu32.extxyz"], ctx.path("analysis_cu32")
    name = "analyze_structure_workflow[Cu32]"
    rdf_max, bins = 8.0, 160
    data = ok(ctx.call, name, "analyze_structure_workflow",
              {"filepath": str(path), "output_dir": str(out), "rdf_max": rdf_max, "rdf_bins": bins})
    if data is None:
        return
    atoms = ref.read(path)
    analysis = data.get("analysis") or {}
    summary, outputs = analysis.get("summary") or {}, analysis.get("outputs") or {}
    sym = ref.symmetry(atoms)
    problems = []
    info = data.get("info") or {}
    expected_info = {"formula": "Cu32", "num_atoms": 32, "pbc": [True, True, True], **sym}
    problems += [f"info.{k} {info.get(k)!r}" for k, v in expected_info.items() if info.get(k) != v]
    if abs(info.get("volume", math.inf) - atoms.get_volume()) > 1e-9:
        problems.append(f"info.volume {info.get('volume')}")
    if data.get("symbols") != atoms.get_chemical_symbols():
        problems.append("symbols")
    files = {"summary_json": out / "structure_summary.json", "rdf_csv": out / "rdf.csv",
             "coordination_csv": out / "coordination.csv", "rdf_plot_png": out / "rdf.png",
             "coordination_plot_png": out / "coordination_hist.png"}
    if outputs != {k: str(v) for k, v in files.items()}:
        problems.append(f"outputs {outputs}")
    for key, file in files.items():
        if not file.is_file():
            problems.append(f"{file.name} missing")
        elif file.suffix == ".png":
            try:
                width, height = png_size(file.read_bytes())
                if width < 100 or height < 100:
                    problems.append(f"{file.name} is {width}x{height}")
            except ValueError:
                problems.append(f"{file.name} is not a PNG")
    if files["summary_json"].is_file() and json.loads(files["summary_json"].read_text()) != summary:
        problems.append("structure_summary.json differs from the returned summary")
    problems += artifact_problems(data, {"filepath": path, "input_filepath": path, **files})
    verdict(report, name, problems, f"info {sym['spacegroup']}, outputs, PNGs, summary file and artifacts")
    if files["coordination_csv"].is_file():
        check_coordination(ctx, atoms, summary, files["coordination_csv"])
    if files["rdf_csv"].is_file():
        check_rdf(ctx, atoms, files["rdf_csv"], rdf_max, bins)
    check_molecule_analysis(ctx)


def check_coordination(ctx: Ctx, atoms, summary: dict, csv_path: Path) -> None:
    correct_cn, defect_cn = ctx.ref.coordination(atoms, 0.0), ctx.ref.coordination(atoms, UPSTREAM_SKIN)
    header, rows = read_csv(csv_path)
    per_atom = [int(r[2]) for r in rows] if header == ["atom_index", "symbol", "coordination"] else None
    coordination = summary.get("coordination") or {}
    average = coordination.get("average")

    def matches(values: list[int]) -> bool:
        mean = float(np.mean(values))
        return per_atom == values and average == mean and coordination.get("by_element") == {"Cu": mean}

    status, detail = classify(
        {"isError": False, "content": []},
        correct=lambda _: f"average {average} (r < 1.2 (r_i + r_j))" if matches(correct_cn) else None,
        defect=lambda _: (f"D4: average {average}, per atom and by element; the physical answer is "
                          f"{float(np.mean(correct_cn))} — the neighbour list keeps ASE's default "
                          f"skin {UPSTREAM_SKIN} Å, so the cutoff is 1.2 (r_i + r_j) + 0.6 Å and the second "
                          "shell (a = 3.61 Å) is counted") if matches(defect_cn) else None)
    ctx.report.add("L1", "analyze_structure_workflow[coordination numbers]", status, detail,
                   coordination_average=average)


def check_rdf(ctx: Ctx, atoms, csv_path: Path, r_max: float, bins: int) -> None:
    header, rows = read_csv(csv_path)
    r_ref, g_ref = ctx.ref.rdf(atoms, r_max, bins)
    try:
        r_got = [float(row[0]) for row in rows]
        g_got = [float(row[1]) for row in rows]
    except (IndexError, ValueError):
        ctx.report.add("L1", "analyze_structure_workflow[g(r)]", "FAIL", f"rdf.csv unparsable, header {header}")
        return
    grid_ok = header == ["r", "g_r"] and abs_error(r_got, r_ref) <= 1e-12
    edge = 0.75 * r_max
    tail = float(np.mean(g_ref[r_ref > edge]))

    def same(scale: float) -> bool:
        return grid_ok and relative_error(g_got, scale * g_ref) <= RDF_TOL

    status, detail = classify(
        {"isError": False, "content": []},
        correct=lambda _: (f"g(r) matches the textbook normalisation (mean {tail:.3f} for r > {edge:g} Å)"
                           if same(1.0) else None),
        defect=lambda _: (f"D9: g(r) is exactly twice the textbook value (mean {2 * tail:.3f} instead of "
                          f"{tail:.3f} for r > {edge:g} Å): neighbor_list counts each pair in both directions "
                          "but the normalisation divides by N/2") if same(2.0) else None)
    ctx.report.add("L1", "analyze_structure_workflow[g(r)]", status, detail)


def check_molecule_analysis(ctx: Ctx) -> None:
    name = "analyze_structure_workflow[cell-less molecule H2O]"
    if "h2o.extxyz" not in ctx.files:
        ctx.report.add("L1", name, "FAIL", "no H2O file from build_structure_workflow")
        return
    result = ctx.call(name, "analyze_structure_workflow",
                      {"filepath": str(ctx.files["h2o.extxyz"]), "output_dir": str(ctx.path("analysis_h2o"))},
                      allow_error=True)

    def correct(r: dict) -> str | None:
        if r.get("isError") or in_band_error(r):
            return None
        info = payload(r).get("info") or {}
        return "analysed a molecule without a cell" if info.get("num_atoms") == 3 else None

    status, detail = classify(result, correct=correct, defect=lambda r: (
        f"D5: the server's own molecule builder output fails analysis in-band ({ZERO_LATTICE}); "
        "only get_structure_info tolerates a cell-less structure")
        if in_band_error(r) == f"ValueError: {ZERO_LATTICE}" else None)
    ctx.report.add("L1", name, status, detail)


def check_download_artifact(ctx: Ctx) -> None:
    tool = "create_download_artifact"
    if "cu.extxyz" in ctx.files:
        path = ctx.files["cu.extxyz"]
        data = ok(ctx.call, f"{tool}[existing structure file]", tool, {"filepath": str(path)})
        if data is not None:
            problems = ([] if without_artifacts(data) == {"filepath": str(path)} else [f"payload {data}"])
            problems += artifact_problems(data, {"filepath": path})
            verdict(ctx.report, f"{tool}[existing structure file]", problems,
                    "art_ id, /artifacts/<id>/<name> URL and an HTML preview")
            ctx.report.add("L1", f"{tool}[download_url over stdio]", "WARN",
                           "download_url is a relative /artifacts/<id>/<name> route that only the HTTP app "
                           "serves; over stdio there is no way to fetch it (the id is still a per-call nonce)")
    missing, other = ctx.path("does-not-exist.extxyz"), ctx.path("notes.md")
    other.write_text("not an artifact type\n", encoding="utf-8")
    for label, path in (("missing file", missing), ("suffix not allowed (.md)", other)):
        check_rejected(ctx.call, f"{tool}[{label}]", tool, {"filepath": str(path)}, in_band=in_band_error,
                       on_accept=lambda r, p=path: ("WARN", "accepted silently: returns {'filepath': ...} with no "
                                                    "artifact and no error") if _bare_echo(r, p) else
                       ("FAIL", f"unexpected result {text_of(r)[:200]!r}"))


def _bare_echo(result: dict, path: Path) -> bool:
    try:
        return payload(result) == {"filepath": str(path)}
    except (ValueError, json.JSONDecodeError):
        return False


def check_errors(ctx: Ctx) -> None:
    call, files = ctx.call, ctx.files
    cu = str(files.get("cu.extxyz", ctx.path("cu.extxyz")))
    cases = (
        ("single_point_workflow", "missing file", {"input_filepath": str(ctx.path("nope.extxyz")),
                                                   "calculator_name": "emt"}),
        ("analyze_structure_workflow", "missing file", {"filepath": str(ctx.path("nope.extxyz")),
                                                        "output_dir": str(ctx.path("analysis_nope"))}),
        ("manipulate_structure_workflow", "unknown operation", {"input_filepath": cu, "operation": "explode",
                                                                "output_filepath": str(ctx.path("x.extxyz"))}),
        ("manipulate_structure_workflow", "vacancy index out of range",
         {"input_filepath": cu, "operation": "vacancy", "operation_kwargs": {"index": 4},
          "output_filepath": str(ctx.path("x.extxyz"))}),
        ("manipulate_structure_workflow", "substitute without symbol",
         {"input_filepath": cu, "operation": "substitute", "output_filepath": str(ctx.path("x.extxyz"))}),
        ("build_structure_workflow", "unknown structure_type",
         {"formula": "Cu", "structure_type": "quasicrystal", "output_filepath": str(ctx.path("x.extxyz"))}),
    )
    for tool, label, arguments in cases:
        check_rejected(call, f"{tool}[{label}]", tool, arguments, in_band=in_band_error)
    if ctx.path("x.extxyz").exists():
        ctx.report.add("L1", "rejected calls wrote no output", "FAIL", "x.extxyz was written")
    check_unsupported_element(ctx)


def check_unsupported_element(ctx: Ctx) -> None:
    name = "single_point_workflow[Si with calculator_name=emt]"
    if "si.cif" not in ctx.files:
        ctx.report.add("L1", name, "FAIL", "no Si file from build_structure_workflow")
        return
    result = ctx.call(name, "single_point_workflow", {"input_filepath": str(ctx.files["si.cif"]),
                                                      "calculator_name": "emt"}, allow_error=True)
    prefix = f"Error calling tool 'single_point_workflow': {NAME_TOO_LONG}: "

    def correct(r: dict) -> str | None:
        message = in_band_error(r) or (text_of(r) if r.get("isError") and NAME_TOO_LONG not in text_of(r) else None)
        return f"rejected: {message[:160]}" if message and "EMT does not support species ['Si']" in message else None

    def defect(r: dict) -> str | None:
        text = text_of(r)
        if r.get("isError") is True and text.startswith(prefix) and "EMT does not support species ['Si']" in text:
            return ("D3: the in-band error report is lost — with_downloadable_artifacts calls Path(v).exists() on "
                    "every string, the long message raises ENAMETOOLONG and the call degrades to a bare "
                    "protocol error; also, calculator_name='emt' is not binding: on failure the server tries "
                    "kim, orb and nequix too (attempted: emt, kim, orb, nequix)"
                    if "attempted: emt, kim, orb, nequix" in text else None)
        return None

    status, detail = classify(result, correct=correct, defect=defect)
    ctx.report.add("L1", name, status, detail)


def check_relative_path(ctx: Ctx) -> None:
    """Documents where relative paths land: the server cwd (the MCP checkout in agent runs)."""
    name = "build_structure_workflow[relative output_filepath]"
    data = ok(ctx.call, name, "build_structure_workflow",
              {"formula": "Cu", "lattice_constant": CU_A, "output_filepath": "relative.extxyz"})
    if data is None:
        return
    expected = ctx.session.cwd.resolve() / "relative.extxyz"
    verdict(ctx.report, name, [] if Path(data.get("filepath", "")).resolve() == expected and expected.is_file()
            else [f"filepath {data.get('filepath')!r}, expected {str(expected)!r}"],
            "resolved against the server cwd — in agent runs that is the MCP checkout, so tasks must ask "
            "for absolute output paths")


def check_tool_error_logs(ctx: Ctx) -> None:
    logs = sorted(p.name for p in (ctx.session.cwd / "tool_errors").glob("*.log"))
    ctx.report.add("L1", "in-band errors write tool_errors/ into the server cwd", "WARN" if logs else "PASS",
                   f"{len(logs)} report file(s) such as {logs[:2]} (one per tool and UTC second; in agent runs "
                   "they accumulate in the MCP checkout)" if logs else "no error report files")


# --------------------------------------------------------------------------
# Checks: the nine unusable tools
# --------------------------------------------------------------------------

def check_deprecated(ctx: Ctx) -> None:
    ref, files = ctx.ref, ctx.files
    cu_ref = ref.bulk("Cu", "fcc", CU_A)
    cases = {
        "build_structure": ({"formula": "Cu", "lattice_constant": CU_A,
                             "output_filepath": str(ctx.path("deprecated_cu.extxyz"))},
                            lambda d: not structure_problems(ref, ctx.path("deprecated_cu.extxyz"), cu_ref)),
        "read_structure_file": ({"filepath": str(files.get("cu.extxyz", ctx.path("cu.extxyz")))},
                                lambda d: (d.get("info") or {}).get("formula") == "Cu4"),
        "write_structure_file": ({"positions": cu_ref.positions.tolist(), "symbols": cu_ref.get_chemical_symbols(),
                                  "cell": cu_ref.cell.array.tolist(),
                                  "filepath": str(ctx.path("deprecated_written.extxyz"))},
                                 lambda d: not structure_problems(ref, ctx.path("deprecated_written.extxyz"), cu_ref)),
        # It forwards to the task-required optimize_structure_workflow: no plain result to check.
        "optimize_with_mlip": ({"input_filepath": str(files.get("cu.extxyz", ctx.path("cu.extxyz"))),
                                "calculator_name": "emt", "output_filepath": str(ctx.path("deprecated_opt.extxyz"))},
                               lambda d: False),
    }
    for tool, (arguments, works) in cases.items():
        name = f"{tool}[deprecated wrapper]"
        result = ctx.call(name, tool, arguments, allow_error=True)

        def correct(r: dict, works=works) -> str | None:
            if r.get("isError") or in_band_error(r):
                return None
            return "works and matches the reference" if works(payload(r)) else None

        status, detail = classify(result, correct=correct, defect=lambda r, tool=tool: (
            "D1: always fails — it awaits the FunctionTool object that @mcp.tool returned for the workflow"
            if is_error_text(r, f"Error calling tool '{tool}': {FUNCTION_TOOL}") else None))
        ctx.report.add("L1", name, status, detail)


def check_task_required(ctx: Ctx, tools: dict[str, dict]) -> None:
    support = {name: (tool.get("execution") or {}).get("taskSupport") for name, tool in tools.items()}
    required = sorted(n for n, s in support.items() if s == "required")
    ctx.report.add("L1", "tools/list execution.taskSupport", "PASS" if required == list(TASK_REQUIRED) else "FAIL",
                   f"required: {required}; everything else 'optional' (also the 4 deprecated wrappers); "
                   "initialize advertises capabilities.tasks")
    cu = str(ctx.files.get("cu.extxyz", ctx.path("cu.extxyz")))
    trajectory = ctx.session.state.get("trajectory_file")
    arguments = {
        "optimize_structure_workflow": {"input_filepath": cu, "calculator_name": "emt",
                                        "output_filepath": str(ctx.path("plain_opt.extxyz"))},
        "run_md_workflow": {"input_filepath": cu, "calculator_name": "emt", "steps": 2,
                            "output_trajectory_filepath": str(ctx.path("plain_md.extxyz")),
                            "log_filepath": str(ctx.path("plain_md.log")),
                            "summary_filepath": str(ctx.path("plain_md.txt"))},
        "relax_and_md_workflow": {"input_filepath": cu, "calculator_name": "emt", "steps": 2,
                                  "integrator": "velocityverlet",
                                  "optimized_filepath": str(ctx.path("plain_relaxed.extxyz")),
                                  "output_trajectory_filepath": str(ctx.path("plain_rmd.extxyz")),
                                  "log_filepath": str(ctx.path("plain_rmd.log")),
                                  "summary_filepath": str(ctx.path("plain_rmd.txt"))},
        "analyze_trajectory_workflow": {"filepath": str(trajectory), "output_dir": str(ctx.path("plain_traj"))},
        "autocorrelation_workflow": {"filepath": str(trajectory), "output_dir": str(ctx.path("plain_vacf"))},
    }
    for tool in TASK_REQUIRED:
        name = f"{tool}[plain tools/call]"
        result = ctx.call(name, tool, arguments[tool], allow_error=True)
        status, detail = classify(result, correct=lambda r: None, defect=lambda r, tool=tool: (
            "D2: refused without task augmentation, which Claude and Codex do not send (task path probed "
            "separately after the main run)"
            if is_error_text(r, f"Tool '{tool}' requires task-augmented execution") else None))
        ctx.report.add("L1", name, status, detail)
    leaked = sorted(p.name for p in ctx.work.glob("plain_*"))
    if leaked:
        ctx.report.add("L1", "refused task tools had no side effects", "FAIL", f"wrote {leaked}")


def write_trajectory(ref: Ref, path: Path, frames: int = 4) -> None:
    """A short Cu4 trajectory with velocities (extxyz), for the trajectory tools."""
    base = ref.bulk("Cu", "fcc", CU_A)
    images = []
    for i in range(frames):
        atoms = base.copy()
        atoms.positions = atoms.positions + 0.01 * i * np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
        atoms.set_velocities(0.001 * (i + 1) * np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1], [-1, 1, -1]]))
        images.append(atoms)
    ref.io.write(path, images, format="extxyz")


def check_coverage(call: Caller, report: Report, expected: list[str]) -> None:
    missing = sorted(set(expected) - set(call.stdout_by_tool))
    report.add("L1", "every listed tool was called", "FAIL" if missing else "PASS",
               f"missing {missing}" if missing else f"{len(expected)} tools")


def run_l1(session: Session) -> None:
    ref = session.state["ref"]
    ctx = Ctx(session, ref)
    tools = {t["name"]: t for t in session.client.list_tools()}
    trajectory = ctx.path("cu4_traj.extxyz")
    write_trajectory(ref, trajectory)
    session.state["trajectory_file"] = trajectory
    check_capabilities(ctx)
    check_build(ctx)
    check_import(ctx)
    check_write(ctx)
    check_manipulate(ctx)
    check_single_point(ctx)
    check_elastic(ctx)
    check_analyze(ctx)
    check_download_artifact(ctx)
    check_errors(ctx)
    check_relative_path(ctx)
    check_deprecated(ctx)
    check_task_required(ctx, tools)
    check_tool_error_logs(ctx)
    check_coverage(session.call, session.report, sorted(session.entry["expected_tools"]))


# --------------------------------------------------------------------------
# After the main run: the task-augmented path, in its own server
# --------------------------------------------------------------------------

def _wait_for(paths: list[Path], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all(p.exists() and (p.is_file() and p.stat().st_size > 0 or p.is_dir() and any(p.iterdir()))
               for p in paths):
            return True
        time.sleep(0.2)
    return False


def task_call(client: StdioMCP, tool: str, arguments: dict) -> tuple[str | None, str]:
    """tools/call with ``params.task``: (taskId, detail)."""
    response = client.request("tools/call", {"name": tool, "arguments": arguments, "task": {"ttl": 60000}})
    meta = ((response.get("result") or {}).get("_meta") or {}).get(TASK_META) or {}
    if "error" in response or response["result"].get("content") != [] or meta.get("status") != "working":
        return None, f"not accepted as a task: {json.dumps(response)[:240]}"
    return meta.get("taskId"), "accepted"


def classify_task_result(get: dict, result: dict) -> tuple[str, str]:
    """D2: tasks/get and tasks/result both fail with "No active context found."."""
    def message(r: dict) -> str | None:
        return (r.get("error") or {}).get("message")
    if message(get) == NO_CONTEXT and message(result) == NO_CONTEXT:
        return "WARN", (f"D2: the side effect happened, but tasks/get and tasks/result both answer {NO_CONTEXT!r} "
                        "(the patched handlers call get_context() outside a request), so the result can never "
                        "be fetched")
    return "FAIL", (f"D2 no longer reproduces: tasks/get {json.dumps(get)[:160]}, "
                    f"tasks/result {json.dumps(result)[:160]}")


def classify_task_stdout(lines: list[str]) -> tuple[str, str]:
    if not lines:
        return "PASS", "no non-JSON stdout on the task path"
    if all(BFGS_LINE.fullmatch(line) for line in lines):
        return "WARN", (f"D8: {len(lines)} BFGS log line(s) on stdout — the optimizer keeps ASE's default "
                        "logfile '-', which corrupts the stdio transport; only reachable on the task path")
    return "WARN", f"{len(lines)} non-JSON stdout line(s) on the task path, e.g. {lines[:3]}"


def after(session: Session) -> None:
    """Task-augmented calls of the five task-required tools in a fresh server."""
    if "ref" not in session.state or session.call is None:
        return
    report, ref = session.report, session.state["ref"]
    work = session.tmp / "task-work"
    work.mkdir()
    trajectory = work / "traj.extxyz"
    write_trajectory(ref, trajectory)
    cu = work / "cu.extxyz"
    ref.io.write(cu, ref.bulk("Cu", "fcc", CU_A), format="extxyz")
    plan = (
        ("optimize_structure_workflow", {"input_filepath": str(cu), "calculator_name": "emt",
                                         "output_filepath": str(work / "opt.extxyz")}, [work / "opt.extxyz"]),
        ("run_md_workflow", {"input_filepath": str(cu), "calculator_name": "emt", "steps": 3,
                             "output_trajectory_filepath": str(work / "md.extxyz"),
                             "log_filepath": str(work / "md.log"),
                             "summary_filepath": str(work / "md.txt")}, [work / "md.extxyz", work / "md.txt"]),
        ("relax_and_md_workflow", {"input_filepath": str(cu), "calculator_name": "emt", "steps": 3,
                                   "integrator": "velocityverlet", "optimized_filepath": str(work / "relaxed.extxyz"),
                                   "output_trajectory_filepath": str(work / "rmd.extxyz"),
                                   "log_filepath": str(work / "rmd.log"), "summary_filepath": str(work / "rmd.txt")},
         [work / "relaxed.extxyz", work / "rmd.extxyz"]),
        ("analyze_trajectory_workflow", {"filepath": str(trajectory), "output_dir": str(work / "traj")},
         [work / "traj"]),
        ("autocorrelation_workflow", {"filepath": str(trajectory), "output_dir": str(work / "vacf")}, [work / "vacf"]),
    )
    with session.spawn("tasks") as client:
        try:
            client.initialize()
            for tool, arguments, effects in plan:
                name = f"{tool}[task-augmented call]"
                task_id, detail = task_call(client, tool, arguments)
                if task_id is None:
                    report.add("L1", name, "FAIL", detail)
                    continue
                if not _wait_for(effects, TASK_WAIT):
                    report.add("L1", name, "FAIL", f"accepted, but {[p.name for p in effects]} did not appear "
                                                   f"within {TASK_WAIT:g} s")
                    continue
                status, detail = classify_task_result(client.request("tasks/get", {"taskId": task_id}),
                                                      client.request("tasks/result", {"taskId": task_id}))
                report.add("L1", name, status, detail + f" ({', '.join(p.name for p in effects)} written)")
        except MCPError as exc:
            report.add("L1", "task-augmented probe", "FAIL", str(exc))
        status, detail = classify_task_stdout(client.non_json_stdout)
        report.add("L1", "task path stdout", status, detail, non_json_stdout=client.non_json_stdout[:20])


def prepare(session: Session) -> None:
    session.state["ref"] = Ref()


SMOKE = Smoke(
    server="atomictoolkit",
    run_l1=run_l1,
    packages=("ase", "spglib", "numpy", "pymatgen", "fastmcp", "mcp", "pydocket"),
    expected_cwd_files=("relative.extxyz", "relative.extxyz.preview.html", "tool_errors"),
    prepare=prepare,
    after=after,
)
