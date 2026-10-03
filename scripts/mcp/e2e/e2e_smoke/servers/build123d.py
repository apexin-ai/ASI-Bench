"""Direct (agent-free) E2E smoke test for the pinned ``pzfreo/build123d-mcp`` MCP server.

Run with the server's own venv (it provides nothing the references need — they are
closed-form — but keeps the reported environment honest)::

    ~/mcp/build123d/.venv/bin/python scripts/mcp/e2e/smoke.py build123d \\
        --config ~/mcp/build123d.mcp.json

No network, no display and no CAD application are needed.

References are computed **here, in this process, in closed form** — never by
build123d, OCP or any mesh library. The fixtures are deliberately chosen so that
every scored quantity has an exact analytic expression:

* ``PLATE`` — a box with a central through hole and four filleted vertical edges:
  volume, surface area, centre of mass and all six volume-inertia components are
  elementary integrals (box − disk − four corner slivers, extruded).
* ``FLANGE`` — a disc with a coaxial bore and a bolt circle: volume is a sum of
  cylinders and the hole table is the construction itself.
* ``FEATURES`` — a plate carrying a boss, a bored boss, a counterbored hole and a
  countersunk hole, so each recogniser has a known answer.
* an open shell and a split solid with known defect counts and coordinates.

Exported meshes are parsed with a stdlib binary-STL reader (divergence-theorem
volume, exact edge-pairing watertightness check); STEP is checked as text
(ISO-10303 envelope, schema, session label), because a correct pure-Python STEP
geometry reader is not realistic.

L1 (the shared L0/L1 checks live in ``e2e_smoke/runner.py``): real tools/call of
all 42 advertised tools, grouped as

      core        execute / measure / validate / inspect_part / cross_sections /
                  session_state / script / resolve
      io          export (step, stl, 3mf, svg, dxf) / import_cad_file / render_view
      recognisers find_holes / find_hole_patterns / find_bosses /
                  find_bored_bosses / find_countersinks / recognise_features
      diagnostics analyze_printability / locate_gate_defects / repair_hints /
                  repair_advice / last_error
      gate        validate on a split solid, compare (shape / fit / align)
      sandbox     blocked builtins and imports, write/read root escapes,
                  unknown objects and snapshots
      session     save_snapshot / restore_snapshot / compare(snapshot) / reset /
                  destroy_session / list_sessions / health_check / version /
                  workflow_hints / install_skill
      audit       design_audit, in a freshly reset session
      file        execute_file provenance and atomic rollback
      drawing     prepare_drawing / crop_drawing / inspect_drawing /
                  lint_drawing / render_drawing / save_drawing_annotations /
                  view_axes / suggest_view_layout
      plus a coverage check (every listed tool called).

The session is stateful, so the order in :func:`run_l1` matters: ``reset`` and
``execute_file`` clear it, and ``design_audit`` parses the *whole* assembled
program, so it only has a well-defined parameter set right after a reset.

Classification: a wrong number from a correctly used tool is FAIL. Everything
below is WARN — correct-but-surprising upstream behaviour that a task author has
to design around:

* ``execute`` reports failures **in band**: ``isError: false`` with a body that
  starts ``Error: …`` plus a ``failure_class``/``suggested_fix`` JSON tail. So do
  ``validate`` and ``restore_snapshot`` for an unknown object/name. A caller that
  only looks at the protocol flag reads a failed build as success.
* ``execute`` needs an explicit ``from build123d import *``; without it even
  ``Box`` is a ``NameError``. Registered ``show()`` names are **not** variables in
  the namespace.
* ``validate`` answers **PASS with ``n_solids: 2``** for a part a bore cut into
  two pieces — the disjoint bodies are only a ``warnings`` entry. A task gate must
  check ``n_solids`` itself.
* ``design_audit`` never calls a parameter ``brittle`` when a ±ε rebuild *fails*
  because a dependent fillet no longer fits: the verdict is ``coupling`` and
  ``brittle: 0``. Read ``needs_review``, as upstream's own note says.
* two of the server's own docstring examples do not work: ``resolve``'s
  ``.faces().filter_by(Axis.Z).last()`` (``ShapeList.last`` is a property, so the
  call raises) and ``inspect_drawing``'s ``from build123d_drafting import Draft``
  (the pinned helper release exports ``draft_preset()``, not ``Draft``).
* ``export`` without ``object_name`` writes ``PRODUCT('COMPOUND','COMPOUND')``:
  only ``export(object_name=…)`` carries the ``show()`` label into the STEP.
* ``import_cad_file`` of an STL yields a **shell** (``volume: 0``), not a solid.
* ``cross_sections(num_slices=0)`` is accepted and sampled anyway.
* ``save_drawing_annotations('x.svg')`` writes ``x.dims.json``, not the documented
  ``x.svg.dims.json``.
* ``destroy_session`` is refused in stdio mode and only points at ``reset()``.
* the 2D drawing suite (``inspect_drawing``, ``lint_drawing``, ``render_drawing``,
  ``save_drawing_annotations``, ``view_axes``, ``suggest_view_layout``) is
  deprecated at this revision (upstream #465, moving to ``draftwright``): the
  tools still work but prefix their answer with a deprecation notice.
* ``prepare_drawing`` strips long straight rules (page borders, title-block
  lines) before grouping ink, so box-shaped line art yields *no* regions; the
  fixture here is three filled discs.
* writes are confined to the server's cwd and ``TMPDIR``, and ``install_skill``
  writes ``.claude/`` or ``AGENTS.md`` into that cwd. A task cannot ask for an
  export straight into an arbitrary submission directory.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

from ..client import text_of
from ..helpers import image_blocks, png_size
from ..runner import Caller, Report, Session, Smoke, check_rejected

# --------------------------------------------------------------------------
# Tolerances
# --------------------------------------------------------------------------

ABS_TOL = 5e-4          # the server rounds its JSON to 4 decimals
REL_TOL = 1e-9          # closed form vs OCC on the same analytic solid
MESH_REL_TOL = 2e-3     # tessellated volume vs exact volume (chords cut corners)
MESH_ABS_TOL = 1e-6     # vertex matching for the watertightness check


# --------------------------------------------------------------------------
# Closed-form reference solids (stdlib only, unit-tested offline)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Plate:
    """Box ``x``×``y``×``z`` centred on the origin, a central through hole of
    diameter ``hole_d`` and the four vertical edges filleted with radius
    ``fillet_r``. Every quantity below is an elementary integral."""

    x: float
    y: float
    z: float
    hole_d: float
    fillet_r: float

    @property
    def code(self) -> str:
        return (
            "from build123d import *\n"
            f"plate_x = {self.x}\n"
            f"plate_y = {self.y}\n"
            f"plate_z = {self.z}\n"
            f"hole_d = {self.hole_d}\n"
            f"fillet_r = {self.fillet_r}\n"
            "with BuildPart() as plate:\n"
            "    Box(plate_x, plate_y, plate_z)\n"
            "    Cylinder(hole_d / 2, plate_z, mode=Mode.SUBTRACT)\n"
            "    fillet(plate.edges().filter_by(Axis.Z), fillet_r)\n"
            "show(plate.part, 'plate')\n"
        )

    # -- cross-section moments ------------------------------------------- #
    def _section(self) -> tuple[float, float, float]:
        """``(area, ∫y²dA, ∫x²dA)`` of the cross-section about its centroid."""
        area = self.x * self.y
        jx = self.x * self.y**3 / 12
        jy = self.x**3 * self.y / 12
        r = self.hole_d / 2
        area -= math.pi * r * r
        jx -= math.pi * r**4 / 4
        jy -= math.pi * r**4 / 4
        f = self.fillet_r
        if f:
            # one corner sliver: the square [cx, cx+f]×[cy, cy+f] minus the
            # quarter disc centred at (cx, cy); four of them by symmetry.
            cx, cy = self.x / 2 - f, self.y / 2 - f
            sq_a = f * f
            sq_jy = ((cx + f) ** 3 - cx**3) / 3 * f
            sq_jx = ((cy + f) ** 3 - cy**3) / 3 * f
            q_a = math.pi * f * f / 4
            # polar about (cx, cy), θ ∈ [0, π/2]: ∫cos dθ = 1, ∫cos²dθ = π/4
            q_jy = cx**2 * q_a + 2 * cx * f**3 / 3 + f**4 / 4 * math.pi / 4
            q_jx = cy**2 * q_a + 2 * cy * f**3 / 3 + f**4 / 4 * math.pi / 4
            area -= 4 * (sq_a - q_a)
            jx -= 4 * (sq_jx - q_jx)
            jy -= 4 * (sq_jy - q_jy)
        return area, jx, jy

    @property
    def section_area(self) -> float:
        return self._section()[0]

    @property
    def volume(self) -> float:
        return self.section_area * self.z

    @property
    def area(self) -> float:
        """Two faces + the outer wall (straights + four quarter arcs) + the bore."""
        straight = 2 * (self.x - 2 * self.fillet_r) + 2 * (self.y - 2 * self.fillet_r)
        perimeter = straight + 2 * math.pi * self.fillet_r
        return 2 * self.section_area + self.z * (perimeter + math.pi * self.hole_d)

    @property
    def inertia(self) -> dict[str, float]:
        area, jx, jy = self._section()
        return {
            "Ixx": self.z * jx + area * self.z**3 / 12,
            "Iyy": self.z * jy + area * self.z**3 / 12,
            "Izz": self.z * (jx + jy),
            "Ixy": 0.0,
            "Ixz": 0.0,
            "Iyz": 0.0,
        }

    @property
    def bbox(self) -> dict[str, float]:
        return {"xsize": self.x, "ysize": self.y, "zsize": self.z}

    @property
    def faces(self) -> int:
        return 2 + 4 + 4 + 1   # top/bottom, four straight walls, four fillets, bore


PLATE = Plate(x=60.0, y=40.0, z=10.0, hole_d=12.0, fillet_r=5.0)

# A disc with a coaxial bore and a bolt circle: volume is a sum of cylinders and
# the hole table is the construction.
FLANGE_R, FLANGE_T, FLANGE_BORE_D, BOLT_D, BOLT_N, BCD = 35.0, 8.0, 20.0, 6.0, 6, 50.0
FLANGE_CODE = (
    "from build123d import *\n"
    f"flange_r = {FLANGE_R}\n"
    f"flange_t = {FLANGE_T}\n"
    f"bore_d = {FLANGE_BORE_D}\n"
    f"bolt_d = {BOLT_D}\n"
    f"bcd = {BCD}\n"
    "with BuildPart() as flange:\n"
    "    Cylinder(flange_r, flange_t)\n"
    "    Cylinder(bore_d / 2, flange_t, mode=Mode.SUBTRACT)\n"
    f"    with PolarLocations(bcd / 2, {BOLT_N}):\n"
    "        Cylinder(bolt_d / 2, flange_t, mode=Mode.SUBTRACT)\n"
    "show(flange.part, 'flange')\n"
)
FLANGE_VOLUME = FLANGE_T * math.pi * (
    FLANGE_R**2 - (FLANGE_BORE_D / 2) ** 2 - BOLT_N * (BOLT_D / 2) ** 2
)
BOLT_LOCATIONS = [
    (
        round(BCD / 2 * math.cos(2 * math.pi * i / BOLT_N), 4),
        round(BCD / 2 * math.sin(2 * math.pi * i / BOLT_N), 4),
    )
    for i in range(BOLT_N)
]

# Documented material presets (g/cm³) used to check that mass is just
# volume × density, not a second geometry computation.
STEEL_DENSITY = 7.85


# --------------------------------------------------------------------------
# Mesh and STEP readers (stdlib only)
# --------------------------------------------------------------------------

def parse_binary_stl(data: bytes) -> list[tuple[tuple[float, float, float], ...]]:
    """Triangles of a binary STL. Raises ``ValueError`` on a malformed file."""
    if len(data) < 84:
        raise ValueError(f"too short for a binary STL ({len(data)} bytes)")
    if data[:5] == b"solid" and b"facet normal" in data[:512]:
        raise ValueError("ASCII STL, expected binary")
    count = struct.unpack("<I", data[80:84])[0]
    if len(data) != 84 + 50 * count:
        raise ValueError(f"declares {count} triangles but is {len(data)} bytes")
    triangles = []
    for i in range(count):
        base = 84 + 50 * i
        values = struct.unpack("<12f", data[base:base + 48])
        triangles.append(tuple(tuple(values[3 * k:3 * k + 3]) for k in range(1, 4)))
    return triangles


def mesh_volume(triangles) -> float:
    """Enclosed volume by the divergence theorem (signed tetrahedra)."""
    total = 0.0
    for a, b, c in triangles:
        total += (
            a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
        )
    return abs(total) / 6.0


def mesh_bbox(triangles) -> tuple[float, float, float]:
    xs = [v[0] for t in triangles for v in t]
    ys = [v[1] for t in triangles for v in t]
    zs = [v[2] for t in triangles for v in t]
    return max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)


def mesh_open_edges(triangles) -> int:
    """Undirected edges not shared by exactly two triangles (0 = watertight)."""
    counts: dict[tuple, int] = {}
    quantum = MESH_ABS_TOL

    def key(v):
        return tuple(round(c / quantum) for c in v)

    for tri in triangles:
        keys = [key(v) for v in tri]
        for i in range(3):
            edge = tuple(sorted((keys[i], keys[(i + 1) % 3])))
            counts[edge] = counts.get(edge, 0) + 1
    return sum(1 for n in counts.values() if n != 2)


def step_problems(text: str, label: str) -> list[str]:
    """Structural complaints about a STEP file (text level only)."""
    problems = []
    if not text.startswith("ISO-10303-21;"):
        problems.append("missing ISO-10303-21 header")
    if not text.rstrip().endswith("END-ISO-10303-21;"):
        problems.append("missing END-ISO-10303-21 terminator")
    if "FILE_SCHEMA" not in text:
        problems.append("no FILE_SCHEMA")
    if "MANIFOLD_SOLID_BREP" not in text and "ADVANCED_BREP_SHAPE_REPRESENTATION" not in text:
        problems.append("no solid B-rep entity")
    if text.count("CARTESIAN_POINT") < 8:
        problems.append(f"only {text.count('CARTESIAN_POINT')} CARTESIAN_POINTs")
    if f"'{label}'" not in text:
        problems.append(f"session label {label!r} not carried into the file")
    return problems


# --------------------------------------------------------------------------
# Small check helpers
# --------------------------------------------------------------------------

def close(got, want, *, abs_tol: float = ABS_TOL, rel_tol: float = REL_TOL) -> bool:
    if not isinstance(got, (int, float)) or isinstance(got, bool):
        return False
    return abs(got - want) <= max(abs_tol, rel_tol * abs(want))


def compare_values(report: Report, name: str, got: dict, want: dict, *,
                   abs_tol: float = ABS_TOL) -> None:
    """Report one check per group of named scalars."""
    bad = {k: (got.get(k), v) for k, v in want.items() if not close(got.get(k), v, abs_tol=abs_tol)}
    report.add("L1", name, "FAIL" if bad else "PASS",
               "; ".join(f"{k}: {g!r} (expected {w!r})" for k, (g, w) in bad.items()) if bad
               else ", ".join(f"{k}={want[k]:.4f}" for k in sorted(want)))


def in_band_error(result: dict) -> str | None:
    """``execute``/``validate`` style error carried in a normal result."""
    text = text_of(result)
    if text.lstrip().startswith("Error:"):
        return text.strip().splitlines()[0]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict) and data.get("error"):
        return str(data["error"])[:200]
    return None


def executed(call: Caller, name: str, code: str) -> str | None:
    """``execute`` + the in-band failure check; returns the body text."""
    result = call(name, "execute", {"code": code})
    if result is None:
        return None
    text = text_of(result)
    if text.lstrip().startswith("Error:"):
        call.report.add("L1", name, "FAIL", f"execute failed in band: {text.strip()[:240]}")
        return None
    return text


def payload(call: Caller, name: str, tool: str, arguments: dict):
    """A tool whose body is one JSON document (possibly after a text preamble)."""
    result = call(name, tool, arguments)
    if result is None:
        return None
    text = text_of(result)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = min((i for i in (text.find("{"), text.find("[")) if i >= 0), default=-1)
    if start >= 0:
        try:
            return json.loads(text[start:])
        except json.JSONDecodeError:
            pass
    call.report.add("L1", name, "FAIL", f"not JSON: {text[:200]!r}")
    return None


DEPRECATION = "DEPRECATED"


def note_deprecated(report: Report, name: str, text: str) -> None:
    if DEPRECATION in text or "deprecated" in text.lower():
        report.add("L1", f"{name} is deprecated upstream", "WARN",
                   "answer carries the #465 deprecation notice (moving to draftwright)")


# --------------------------------------------------------------------------
# Core geometry: build, measure, validate, inspect, introspect
# --------------------------------------------------------------------------

def check_core(session: Session) -> None:
    call, report = session.call, session.report

    text = executed(call, "execute[plate]", PLATE.code)
    if text is not None:
        ok = "Registered 'plate'" in text and f"faces={PLATE.faces}" in text
        report.add("L1", "execute registers the named part", "PASS" if ok else "FAIL",
                   text.splitlines()[0] if text else "")

    measured = payload(call, "measure[plate]", "measure", {"object_name": "plate"})
    if measured:
        compare_values(report, "measure volume and area", measured,
                       {"volume": PLATE.volume, "area": PLATE.area})
        compare_values(report, "measure bbox", measured.get("bbox", {}), PLATE.bbox)
        compare_values(report, "measure centre of mass", measured.get("center_of_mass", {}),
                       {"x": 0.0, "y": 0.0, "z": 0.0})
        compare_values(report, "measure inertia tensor", measured.get("inertia", {}),
                       PLATE.inertia, abs_tol=0.05)
        topology = measured.get("topology", {})
        want_topology = {"faces": PLATE.faces, "edges": 27, "vertices": 18}
        report.add("L1", "measure topology", "PASS" if topology == want_topology else "FAIL",
                   f"{topology} (expected {want_topology})")
        check_face_inventory(report, measured.get("face_inventory", []))

    mass = payload(call, "measure[steel]", "measure", {"object_name": "plate", "material": "steel"})
    if mass:
        compare_values(report, "measure mass from the steel preset", mass,
                       {"mass_g": PLATE.volume * STEEL_DENSITY / 1000.0}, abs_tol=5e-3)

    gate = payload(call, "validate[plate]", "validate", {"object_name": "plate"})
    if gate:
        want = {"passes_gate": True, "n_solids": 1, "watertight_manifold": True,
                "brep_valid": True, "open_edges": 0, "nonmanifold_edges": 0, "reasons": []}
        bad = {k: gate.get(k) for k, v in want.items() if gate.get(k) != v}
        ok_volume = close(gate.get("volume"), PLATE.volume)
        report.add("L1", "validate passes the gate", "FAIL" if bad or not ok_volume else "PASS",
                   f"unexpected {bad}; volume={gate.get('volume')}" if bad or not ok_volume else "")

    part = payload(call, "inspect_part[plate]", "inspect_part", {"object_name": "plate"})
    if part:
        compare_values(report, "inspect_part bbox", part.get("bbox", {}),
                       {"x": PLATE.x, "y": PLATE.y, "z": PLATE.z})
        holes = part.get("holes", {})
        groups = holes.get("groups") or [{}]
        ok = (holes.get("count") == 1 and close(groups[0].get("diameter"), PLATE.hole_d)
              and close(groups[0].get("depth"), PLATE.z) and groups[0].get("bottom") == "through")
        report.add("L1", "inspect_part hole table", "PASS" if ok else "FAIL", f"{holes}")
        sections = part.get("sections", {})
        ok = sections.get("constant_section") is True and close(sections.get("variation_ratio"), 0.0)
        report.add("L1", "inspect_part constant section", "PASS" if ok else "FAIL",
                   f"variation_ratio={sections.get('variation_ratio')}, "
                   f"constant_section={sections.get('constant_section')}")

    slices = payload(call, "cross_sections[plate]", "cross_sections",
                     {"object_name": "plate", "axis": "Z", "num_slices": 7})
    if isinstance(slices, list):
        areas = [s.get("area") for s in slices]
        ok = len(areas) == 7 and all(close(a, PLATE.section_area) for a in areas)
        report.add("L1", "cross_sections areas", "PASS" if ok else "FAIL",
                   f"{len(areas)} slices, areas {areas[:3]} (expected {PLATE.section_area:.4f})")

    state = payload(call, "session_state", "session_state", {})
    if state:
        variables = state.get("variables", {})
        want = {"plate_x": PLATE.x, "plate_y": PLATE.y, "plate_z": PLATE.z,
                "hole_d": PLATE.hole_d, "fillet_r": PLATE.fillet_r}
        got = {k: variables.get(k, {}).get("value") for k in want}
        objects = state.get("objects", {})
        ok = got == want and "plate" in objects and close(objects["plate"]["volume"], PLATE.volume)
        report.add("L1", "session_state parameters and objects", "PASS" if ok else "FAIL",
                   f"variables={got}, objects={sorted(objects)}")

    assembled = payload(call, "script", "script", {})
    if assembled:
        ok = assembled.get("script", "").strip() == PLATE.code.strip() and assembled.get("blocks") == 1
        report.add("L1", "script returns the submitted source", "PASS" if ok else "FAIL",
                   "" if ok else f"blocks={assembled.get('blocks')}, "
                                 f"{assembled.get('script', '')[:160]!r}")

    face = payload(call, "resolve[top face]", "resolve",
                   {"object_name": "plate", "selector": ".faces().sort_by(Axis.Z).last",
                    "label": "top"})
    if face:
        centre = face.get("center") or []
        ok = (face.get("geom_type") == "PLANE" and face.get("type") == "Face"
              and close(face.get("area"), PLATE.section_area)
              and len(centre) == 3 and close(centre[2], PLATE.z / 2))
        report.add("L1", "resolve top face", "PASS" if ok else "FAIL",
                   f"geom_type={face.get('geom_type')}, area={face.get('area')}, center={centre}")
    bore = payload(call, "resolve[bore]", "resolve",
                   {"object_name": "plate", "selector": ".faces().filter_by(GeomType.CYLINDER)"})
    if bore:
        radii = sorted(round(e.get("area", 0.0), 4) for e in bore.get("entities", []))
        want = sorted([round(math.pi * PLATE.hole_d * PLATE.z, 4)]
                      + [round(math.pi * PLATE.fillet_r / 2 * PLATE.z, 4)] * 4)
        ok = bore.get("count") == 5 and radii == want
        report.add("L1", "resolve lists every cylindrical face", "PASS" if ok else "FAIL",
                   f"count={bore.get('count')}, areas={radii} (expected {want})")
    # The resolve docstring advertises '.faces().filter_by(Axis.Z).last()', but
    # build123d's ShapeList.last is a property: the documented form raises.
    check_rejected(call, "resolve[documented .last() example]", "resolve",
                   {"object_name": "plate", "selector": ".faces().filter_by(Axis.Z).last()"},
                   in_band=in_band_error)


def check_face_inventory(report: Report, inventory: list[dict]) -> None:
    """Every analytic face of the plate, by type, area and count."""
    got = sorted(
        (entry.get("type"), round(entry.get("area", 0.0), 4), entry.get("count", 1),
         round(entry.get("diameter"), 4) if entry.get("diameter") is not None else None)
        for entry in inventory
    )
    f, z = PLATE.fillet_r, PLATE.z
    want = sorted([
        ("Cylinder", round(math.pi * PLATE.hole_d * z, 4), 1, PLATE.hole_d),
        ("Cylinder", round(math.pi * f / 2 * z, 4), 4, 2 * f),
        ("Plane", round(PLATE.section_area, 4), 2, None),
        ("Plane", round((PLATE.x - 2 * f) * z, 4), 2, None),
        ("Plane", round((PLATE.y - 2 * f) * z, 4), 2, None),
    ])
    report.add("L1", "measure face inventory", "PASS" if got == want else "FAIL",
               f"{got}" if got != want else f"{len(inventory)} entries")


# --------------------------------------------------------------------------
# File I/O: export, import, render
# --------------------------------------------------------------------------

def check_io(session: Session) -> None:
    call, report = session.call, session.report
    out = session.scratch

    stem = out / "plate"
    result = call("export[step,stl]", "export",
                  {"filename": str(stem), "format": "step,stl", "object_name": "plate"})
    if result is not None:
        text = text_of(result)
        step, stl = stem.with_suffix(".step"), stem.with_suffix(".stl")
        report.add("L1", "export writes both files", "PASS" if step.is_file() and stl.is_file()
                   else "FAIL", f"{text[:160]}")
        if step.is_file():
            problems = step_problems(step.read_text(encoding="utf-8", errors="replace"), "plate")
            report.add("L1", "exported STEP is well formed", "FAIL" if problems else "PASS",
                       "; ".join(problems))
        if stl.is_file():
            check_mesh(report, "STL", stl.read_bytes())

    anonymous = out / "current.step"
    if call("export[current shape]", "export", {"filename": str(anonymous)}) is not None:
        text = anonymous.read_text(encoding="utf-8", errors="replace") if anonymous.is_file() else ""
        if "PRODUCT('COMPOUND'" in text and "'plate'" not in text:
            report.add("L1", "export without object_name loses the session label", "WARN",
                       "the current-shape export writes PRODUCT('COMPOUND','COMPOUND'); only "
                       "export(object_name=…) carries the show() name into the STEP")
        else:
            report.add("L1", "export without object_name loses the session label", "PASS",
                       "label preserved for the current shape too")

    three_mf = out / "plate.3mf"
    if call("export[3mf]", "export", {"filename": str(three_mf), "format": "3mf"}) is not None:
        ok = three_mf.is_file() and three_mf.read_bytes()[:2] == b"PK"
        report.add("L1", "export 3mf is a zip container", "PASS" if ok else "FAIL",
                   f"{three_mf.stat().st_size if three_mf.is_file() else 'missing'} bytes")

    reimported = payload(call, "import_cad_file[step]", "import_cad_file",
                         {"path": str(out / "plate.step"), "name": "from_step"})
    if reimported:
        ok = (close(reimported.get("volume"), PLATE.volume) and reimported.get("solids") == 1
              and reimported.get("faces") == PLATE.faces)
        report.add("L1", "STEP round trip keeps the solid", "PASS" if ok else "FAIL",
                   f"volume={reimported.get('volume')}, solids={reimported.get('solids')}, "
                   f"faces={reimported.get('faces')}")

    as_stl = payload(call, "import_cad_file[stl]", "import_cad_file",
                     {"path": str(out / "plate.stl"), "name": "from_stl"})
    if as_stl:
        if as_stl.get("solids") == 0 and close(as_stl.get("volume"), 0.0):
            report.add("L1", "STL import yields a shell, not a solid", "WARN",
                       "documented upstream behaviour: volume=0, solids=0 — an imported mesh "
                       "cannot be measured or exported as a solid without sewing it first")
        else:
            report.add("L1", "STL import yields a shell, not a solid", "FAIL",
                       f"expected the documented shell, got volume={as_stl.get('volume')}, "
                       f"solids={as_stl.get('solids')}")

    png = out / "plate-iso.png"
    rendered = call("render_view[iso]", "render_view",
                    {"objects": "plate", "direction": "iso", "quality": "preview",
                     "save_to": str(png)})
    if rendered is not None:
        images = image_blocks(rendered)
        try:
            width, height = png_size(png.read_bytes()) if png.is_file() else (0, 0)
        except ValueError as exc:
            width, height = 0, 0
            report.add("L1", "render_view writes a PNG", "FAIL", str(exc))
        ok = bool(images) and width > 64 and height > 64
        report.add("L1", "render_view returns and writes an image", "PASS" if ok else "FAIL",
                   f"{len(images)} image block(s), file {width}x{height} px — headless, no Xvfb")


def check_mesh(report: Report, label: str, data: bytes) -> None:
    try:
        triangles = parse_binary_stl(data)
    except ValueError as exc:
        report.add("L1", f"exported {label} parses", "FAIL", str(exc))
        return
    volume = mesh_volume(triangles)
    deviation = (volume - PLATE.volume) / PLATE.volume
    report.add("L1", f"exported {label} volume matches the exact solid",
               "PASS" if abs(deviation) <= MESH_REL_TOL else "FAIL",
               f"{len(triangles)} triangles, mesh {volume:.4f} vs exact {PLATE.volume:.4f} "
               f"({deviation:+.2e} relative; the tessellated bore over-fills, the fillets "
               f"under-fill)")
    open_edges = mesh_open_edges(triangles)
    report.add("L1", f"exported {label} is watertight", "PASS" if open_edges == 0 else "FAIL",
               f"{open_edges} unpaired edge(s)")
    box = mesh_bbox(triangles)
    want = (PLATE.x, PLATE.y, PLATE.z)
    ok = all(close(g, w, abs_tol=1e-6) for g, w in zip(box, want))
    report.add("L1", f"exported {label} bounding box", "PASS" if ok else "FAIL",
               f"{tuple(round(v, 6) for v in box)} (expected {want})")


# --------------------------------------------------------------------------
# execute_file: provenance and atomic promotion
# --------------------------------------------------------------------------

FILE_MODEL = (
    "from build123d import *\n"
    "\n"
    "box_x = 30.0\n"
    "box_y = 20.0\n"
    "box_z = 4.0\n"
    "result = Box(box_x, box_y, box_z)\n"
)
FILE_MODEL_VOLUME = 30.0 * 20.0 * 4.0
BROKEN_MODEL = "from build123d import *\nresult = Box(1, 1,\n"


def check_execute_file(session: Session) -> None:
    call, report = session.call, session.report
    source = session.scratch / "model.py"
    source.write_text(FILE_MODEL, encoding="utf-8")
    digest = hashlib.sha256(FILE_MODEL.encode("utf-8")).hexdigest()

    promoted = payload(call, "execute_file[model.py]", "execute_file",
                       {"path": str(source), "result_name": "result", "snapshot": "fromfile"})
    if promoted:
        ok = (promoted.get("ok") is True and promoted.get("source_sha256") == digest
              and promoted.get("result_name") == "result")
        report.add("L1", "execute_file reports the source SHA-256", "PASS" if ok else "FAIL",
                   f"ok={promoted.get('ok')}, sha256={promoted.get('source_sha256')} "
                   f"(expected {digest})")
    measured = payload(call, "measure[from file]", "measure", {})
    if measured:
        compare_values(report, "execute_file promotes the file's result", measured,
                       {"volume": FILE_MODEL_VOLUME})

    broken = session.scratch / "broken.py"
    broken.write_text(BROKEN_MODEL, encoding="utf-8")
    rolled = payload(call, "execute_file[syntax error]", "execute_file", {"path": str(broken)})
    if rolled:
        ok = rolled.get("ok") is False and rolled.get("rolled_back") is True
        report.add("L1", "execute_file rolls back a broken source", "PASS" if ok else "FAIL",
                   f"ok={rolled.get('ok')}, rolled_back={rolled.get('rolled_back')}, "
                   f"error={str(rolled.get('error'))[:120]}")
    after = payload(call, "measure[after rollback]", "measure", {})
    if after:
        compare_values(report, "the previous model survives the rollback", after,
                       {"volume": FILE_MODEL_VOLUME})


# --------------------------------------------------------------------------
# Feature recognisers
# --------------------------------------------------------------------------

FEAT = {
    "plate_x": 80.0, "plate_y": 60.0, "plate_z": 10.0,
    "boss_d": 16.0, "boss_h": 6.0, "bore_d": 8.0,
    "cb_drill_d": 6.0, "cb_head_d": 12.0, "cb_depth": 3.0,
    "cs_drill_d": 5.0, "cs_head_d": 10.0, "cs_angle": 82.0,
}
FEATURES_CODE = (
    "from build123d import *\n"
    + "".join(f"{k} = {v}\n" for k, v in FEAT.items() if k != "cs_angle")
    + "with BuildPart() as feat:\n"
      "    Box(plate_x, plate_y, plate_z)\n"
      "    with Locations((-25, 0, plate_z / 2)):\n"
      "        Cylinder(boss_d / 2, boss_h, align=(Align.CENTER, Align.CENTER, Align.MIN))\n"
      "    with Locations((25, 0, plate_z / 2)):\n"
      "        Cylinder(boss_d / 2, boss_h, align=(Align.CENTER, Align.CENTER, Align.MIN))\n"
      "    with Locations((25, 0, plate_z / 2 + boss_h)):\n"
      "        Hole(bore_d / 2)\n"
      "    with Locations((0, 20, plate_z / 2)):\n"
      "        CounterBoreHole(cb_drill_d / 2, cb_head_d / 2, cb_depth)\n"
      "    with Locations((0, -20, plate_z / 2)):\n"
      "        CounterSinkHole(cs_drill_d / 2, cs_head_d / 2)\n"
      "show(feat.part, 'features')\n"
)
# Countersink cone depth from the included angle, and the resulting volume.
CS_CONE_DEPTH = ((FEAT["cs_head_d"] - FEAT["cs_drill_d"]) / 2
                 / math.tan(math.radians(FEAT["cs_angle"] / 2)))


def features_volume() -> float:
    f = FEAT
    box = f["plate_x"] * f["plate_y"] * f["plate_z"]
    bosses = 2 * math.pi * (f["boss_d"] / 2) ** 2 * f["boss_h"]
    bore = math.pi * (f["bore_d"] / 2) ** 2 * (f["plate_z"] + f["boss_h"])
    counterbore = (math.pi * (f["cb_drill_d"] / 2) ** 2 * (f["plate_z"] - f["cb_depth"])
                   + math.pi * (f["cb_head_d"] / 2) ** 2 * f["cb_depth"])
    r, big = f["cs_drill_d"] / 2, f["cs_head_d"] / 2
    countersink = (math.pi * r**2 * (f["plate_z"] - CS_CONE_DEPTH)
                   + math.pi * CS_CONE_DEPTH / 3 * (big**2 + big * r + r**2))
    return box + bosses - bore - counterbore - countersink


FEATURES_VOLUME = features_volume()


def at(records: list[dict], key: str, x: float, y: float) -> dict:
    """The record whose ``key`` location starts with (x, y), or ``{}``."""
    for record in records:
        location = record.get(key) or []
        if len(location) >= 2 and close(location[0], x, abs_tol=1e-3) \
                and close(location[1], y, abs_tol=1e-3):
            return record
    return {}


def check_recognisers(session: Session) -> None:
    call, report = session.call, session.report
    f = FEAT

    if executed(call, "execute[features]", FEATURES_CODE) is not None:
        measured = payload(call, "measure[features]", "measure", {"object_name": "features"})
        if measured:
            compare_values(report, "features volume (box + bosses − bore − cbore − csink)",
                           measured, {"volume": FEATURES_VOLUME}, abs_tol=1e-3)

    holes = payload(call, "find_holes[features]", "find_holes", {"object_name": "features"})
    if holes:
        records = holes.get("holes", [])
        counterbored = at(records, "location", 0.0, 20.0)
        bored = at(records, "location", 25.0, 0.0)
        countersunk = at(records, "location", 0.0, -20.0)
        checks = {
            "count": holes.get("count") == 3,
            "counterbore": (close(counterbored.get("diameter"), f["cb_drill_d"])
                            and close(counterbored.get("depth"), f["plate_z"] - f["cb_depth"])
                            and counterbored.get("bottom") == "through"
                            and close((counterbored.get("cbore") or {}).get("diameter"),
                                      f["cb_head_d"])
                            and close((counterbored.get("cbore") or {}).get("depth"),
                                      f["cb_depth"])),
            "through bore": (close(bored.get("diameter"), f["bore_d"])
                             and close(bored.get("depth"), f["plate_z"] + f["boss_h"])),
            "countersink drill": (close(countersunk.get("diameter"), f["cs_drill_d"])
                                  and close(countersunk.get("depth"),
                                            f["plate_z"] - CS_CONE_DEPTH, abs_tol=5e-3)),
        }
        bad = sorted(k for k, ok in checks.items() if not ok)
        report.add("L1", "find_holes table", "FAIL" if bad else "PASS",
                   f"wrong: {bad}; {records}" if bad else "3 holes, counterbore and drill depths")

    bosses = payload(call, "find_bosses[features]", "find_bosses", {"object_name": "features"})
    if bosses:
        top = f["plate_z"] / 2 + f["boss_h"]
        ok = bosses.get("count") == 2 and all(
            close(at(bosses.get("bosses", []), "location", x, 0.0).get("diameter"), f["boss_d"])
            and close(at(bosses.get("bosses", []), "location", x, 0.0).get("height"), f["boss_h"])
            and close((at(bosses.get("bosses", []), "location", x, 0.0).get("location") or [0, 0, 0])[2], top)
            for x in (-25.0, 25.0))
        report.add("L1", "find_bosses table", "PASS" if ok else "FAIL", f"{bosses}")

    bored = payload(call, "find_bored_bosses[features]", "find_bored_bosses",
                    {"object_name": "features"})
    if bored:
        candidate = (bored.get("candidates") or [{}])[0]
        ok = (bored.get("count") == 1
              and close(candidate.get("bore_diameter"), f["bore_d"])
              and close(candidate.get("bore_depth"), f["plate_z"] + f["boss_h"])
              and candidate.get("bottom") == "through")
        report.add("L1", "find_bored_bosses picks the bored boss only", "PASS" if ok else "FAIL",
                   f"count={bored.get('count')}, bore={candidate.get('bore_diameter')}, "
                   f"depth={candidate.get('bore_depth')}")

    sinks = payload(call, "find_countersinks[features]", "find_countersinks",
                    {"object_name": "features"})
    if sinks:
        sink = (sinks.get("countersinks") or [{}])[0]
        ok = (sinks.get("count") == 1
              and close(sink.get("major_diameter"), f["cs_head_d"])
              and close(sink.get("drill_diameter"), f["cs_drill_d"])
              and close(sink.get("included_angle"), f["cs_angle"])
              and close(sink.get("depth"), CS_CONE_DEPTH, abs_tol=5e-3))
        report.add("L1", "find_countersinks matches the cone geometry", "PASS" if ok else "FAIL",
                   f"{sink} (cone depth {CS_CONE_DEPTH:.4f})")

    inventory = payload(call, "recognise_features[features]", "recognise_features",
                        {"object_name": "features"})
    if inventory:
        got = inventory.get("inventory", {})
        want = {"holes": 3, "bosses": 2, "countersinks": 1, "hole_patterns": 0, "cylinders": 2}
        bad = {k: got.get(k) for k, v in want.items() if got.get(k) != v}
        report.add("L1", "recognise_features inventory", "FAIL" if bad else "PASS",
                   f"unexpected {bad}" if bad else ", ".join(f"{k}={v}" for k, v in want.items()))

    if executed(call, "execute[flange]", FLANGE_CODE) is not None:
        measured = payload(call, "measure[flange]", "measure", {"object_name": "flange"})
        if measured:
            compare_values(report, "flange volume (disc − bore − bolt holes)", measured,
                           {"volume": FLANGE_VOLUME}, abs_tol=1e-3)

    patterns = payload(call, "find_hole_patterns[flange]", "find_hole_patterns",
                       {"object_name": "flange"})
    if patterns:
        pattern = (patterns.get("patterns") or [{}])[0]
        holes = pattern.get("holes", [])
        places = sorted((round(h["location"][0], 4), round(h["location"][1], 4)) for h in holes)
        ok = (patterns.get("count") == 1 and len(holes) == BOLT_N
              and places == sorted(BOLT_LOCATIONS)
              and all(close(h.get("diameter"), BOLT_D) for h in holes))
        report.add("L1", "find_hole_patterns bolt circle", "PASS" if ok else "FAIL",
                   f"{len(holes)} holes at {places} (expected {sorted(BOLT_LOCATIONS)})")


# --------------------------------------------------------------------------
# Diagnostics: printability, design audit, comparisons, gate defects
# --------------------------------------------------------------------------

THIN_CODE = """from build123d import *
with BuildPart() as thin:
    Box(40, 40, 2)
    with Locations((0, 0, 1)):
        Box(40, 0.3, 12, align=(Align.CENTER, Align.CENTER, Align.MIN))
    with Locations((0, 15, 13)):
        Box(30, 8, 1, align=(Align.CENTER, Align.CENTER, Align.MIN))
show(thin.part, 'thin')
"""
ROOF_AREA = 30.0 * 8.0        # the flat ceiling that needs support
THIN_WALL = 0.3               # mm, below 2 perimeters at a 0.4 mm nozzle
THIN_BBOX = (40.0, 40.0, 15.0)

COUPLING_CODE = """from build123d import *
plate_l = 40.0
plate_w = 10.0
fillet_r = 4.8
with BuildPart() as coupled:
    Box(plate_l, plate_w, 6)
    fillet(coupled.edges().filter_by(Axis.Z), fillet_r)
show(coupled.part, 'coupled')
"""
SPLIT_CODE = """from build123d import *
with BuildPart() as two:
    Box(40, 10, 6)
    Cylinder(5.5, 6, mode=Mode.SUBTRACT)
show(two.part, 'split')
"""
SPLIT_VOLUME = 40.0 * 10.0 * 6.0 - (
    # the bore is wider than the plate: two circular segments stay outside it
    6.0 * (math.pi * 5.5**2 - 2 * (5.5**2 * math.acos(5.0 / 5.5) - 5.0 * math.sqrt(5.5**2 - 5.0**2)))
)
FIT_CODE = """from build123d import *
shaft_d = 9.9
bore_d = 10.0
shaft = Cylinder(shaft_d / 2, 30)
block = Box(30, 30, 20) - Cylinder(bore_d / 2, 20)
show(shaft, 'shaft')
show(block, 'block')
"""
SHAFT_VOLUME = math.pi * (9.9 / 2) ** 2 * 30
BLOCK_VOLUME = 30.0 * 30.0 * 20.0 - math.pi * (10.0 / 2) ** 2 * 20.0
FIT_CLEARANCE = (10.0 - 9.9) / 2
ALIGN_DELTA = 30.0 / 2 - 20.0 / 2      # shaft top − block top, flush on Z

SHELL_CODE = """from build123d import *
cube = Box(10, 10, 10)
show(cube.faces().sort_by(Axis.Z).first, 'openshell')
"""
SHELL_EDGES = {(-5.0, 0.0, -5.0), (5.0, 0.0, -5.0), (0.0, -5.0, -5.0), (0.0, 5.0, -5.0)}
BAD_CODE = "from build123d import *\nx = 1\nnosuchname + 1\n"


def check_diagnostics(session: Session) -> None:
    call, report = session.call, session.report

    if executed(call, "execute[thin]", THIN_CODE) is not None:
        printability = payload(call, "analyze_printability[thin]", "analyze_printability",
                               {"object_name": "thin"})
        if printability:
            findings = {entry.get("kind"): entry for entry in printability.get("findings", [])}
            overhang = findings.get("overhang", {})
            thin_wall = findings.get("thin_wall", {})
            ok = (set(findings) == {"overhang", "thin_wall"}
                  and close(overhang.get("area"), ROOF_AREA)
                  and f"{THIN_WALL:.2f} mm" in str(thin_wall.get("message")))
            report.add("L1", "analyze_printability finds the ceiling and the thin wall",
                       "PASS" if ok else "FAIL",
                       f"kinds={sorted(findings)}, overhang area={overhang.get('area')} "
                       f"(expected {ROOF_AREA}), {str(thin_wall.get('message'))[:80]}")
        bed = payload(call, "analyze_printability[bed fit]", "analyze_printability",
                      {"object_name": "thin", "build_volume": "20 20 20"})
        if bed:
            fits = [e for e in bed.get("findings", []) if e.get("kind") == "bed_fit"]
            size = "x".join(f"{v:.1f}" for v in THIN_BBOX)
            ok = (len(fits) == 1 and fits[0].get("severity") == "error"
                  and size in str(fits[0].get("message")))
            report.add("L1", "analyze_printability rejects an oversized build volume",
                       "PASS" if ok else "FAIL", f"{[e.get('message') for e in fits]}")

def check_design_audit(session: Session) -> None:
    """design_audit parses the *whole* assembled session program, so it only has a
    well-defined parameter set in a freshly reset session."""
    call, report = session.call, session.report
    call("reset[before the audit]", "reset", {})
    if executed(call, "execute[coupled]", COUPLING_CODE) is not None:
        audit = payload(call, "design_audit[coupled]", "design_audit",
                        {"epsilon": 0.1, "max_params": 4})
        if audit:
            verdicts = {entry["name"]: entry for entry in audit.get("audit", [])}
            want = {"plate_l": "robust", "plate_w": "coupling", "fillet_r": "coupling"}
            got = {name: entry.get("verdict") for name, entry in verdicts.items()}
            summary = audit.get("summary", {})
            ok = (got == want and summary.get("robust") == 1 and summary.get("coupling") == 2
                  and summary.get("needs_review") == 2
                  and audit.get("baseline", {}).get("passes_gate") is True)
            report.add("L1", "design_audit classifies the fillet coupling",
                       "PASS" if ok else "FAIL",
                       f"verdicts={got} (expected {want}), summary={summary}")
            failed = [name for name, entry in verdicts.items()
                      if any(p.get("rebuilt") is False for p in entry.get("perturbations", []))]
            if sorted(failed) == ["fillet_r", "plate_w"] and summary.get("brittle") == 0:
                report.add("L1", "design_audit reports brittle=0 despite failed rebuilds", "WARN",
                           "±10% on plate_w and fillet_r cannot rebuild (the fillet no longer "
                           "fits), yet both are brittle=false with verdict 'coupling'; only the "
                           "needs_review count surfaces them")

def check_gate_and_comparisons(session: Session) -> None:
    call, report = session.call, session.report
    if executed(call, "execute[split]", SPLIT_CODE) is not None:
        gate = payload(call, "validate[split]", "validate", {"object_name": "split"})
        if gate:
            if (gate.get("passes_gate") is True and gate.get("n_solids") == 2
                    and gate.get("warnings")):
                report.add("L1", "the gate passes a part cut into two solids", "WARN",
                           "validate() answers PASS with n_solids=2 and only a warning — a task "
                           "gate must check n_solids itself")
            else:
                report.add("L1", "the gate passes a part cut into two solids", "FAIL",
                           f"expected the documented warning, got passes_gate="
                           f"{gate.get('passes_gate')}, n_solids={gate.get('n_solids')}, "
                           f"warnings={gate.get('warnings')}")
            compare_values(report, "split part volume", gate, {"volume": SPLIT_VOLUME},
                           abs_tol=1e-3)

    if executed(call, "execute[fit]", FIT_CODE) is not None:
        fit = payload(call, "compare[fit]", "compare",
                      {"a": "shaft", "b": "block", "kind": "fit", "format": "json"})
        if fit:
            ok = (close(fit.get("clearance"), FIT_CLEARANCE) and fit.get("status") == "apart"
                  and close(fit.get("intersection_volume"), 0.0))
            report.add("L1", "compare fit clearance", "PASS" if ok else "FAIL",
                       f"clearance={fit.get('clearance')} (expected {FIT_CLEARANCE}), "
                       f"status={fit.get('status')}")
        align = payload(call, "compare[align]", "compare",
                        {"a": "shaft", "b": "block", "kind": "align", "axis": "Z",
                         "mode": "flush", "format": "json"})
        if align:
            ok = close(align.get("delta"), ALIGN_DELTA)
            report.add("L1", "compare align delta", "PASS" if ok else "FAIL",
                       f"delta={align.get('delta')} (expected {ALIGN_DELTA})")
        shape = payload(call, "compare[shape]", "compare",
                        {"a": "shaft", "b": "block", "kind": "shape", "format": "json"})
        if shape:
            delta = (shape.get("delta") or {}).get("volume")
            ok = (close(shape.get("a", {}).get("volume"), SHAFT_VOLUME)
                  and close(shape.get("b", {}).get("volume"), BLOCK_VOLUME)
                  and close(delta, BLOCK_VOLUME - SHAFT_VOLUME, abs_tol=1e-3))
            report.add("L1", "compare shape volumes and delta", "PASS" if ok else "FAIL",
                       f"a={shape.get('a', {}).get('volume')}, b={shape.get('b', {}).get('volume')},"
                       f" delta={delta} (expected {BLOCK_VOLUME - SHAFT_VOLUME:.4f} = b − a)")

    if executed(call, "execute[open shell]", SHELL_CODE) is not None:
        gate = payload(call, "validate[open shell]", "validate", {"object_name": "openshell"})
        if gate:
            want = {"passes_gate": False, "n_solids": 0, "open_edges": 4, "mesh_open_edges": 4,
                    "watertight_manifold": False}
            bad = {k: gate.get(k) for k, v in want.items() if gate.get(k) != v}
            report.add("L1", "validate fails an open shell", "FAIL" if bad else "PASS",
                       f"unexpected {bad}" if bad else "4 open edges, no solid")
        defects = payload(call, "locate_gate_defects[open shell]", "locate_gate_defects",
                          {"object_name": "openshell"})
        if defects:
            kinds: dict[str, int] = {}
            for defect in defects.get("defects", []):
                kinds[defect.get("kind")] = kinds.get(defect.get("kind"), 0) + 1
            places = {tuple(round(v, 4) for v in d["where"])
                      for d in defects.get("defects", []) if d.get("kind") == "open_edge"}
            ok = (defects.get("count") == 8
                  and kinds == {"open_edge": 4, "mesh_open_edge": 4}
                  and places == SHELL_EDGES)
            report.add("L1", "locate_gate_defects pinpoints the four open edges",
                       "PASS" if ok else "FAIL",
                       f"count={defects.get('count')}, kinds={kinds}, midpoints={sorted(places)}")

    hints = payload(call, "repair_hints[open edges]", "repair_hints",
                    {"error_text": "4 open edge(s) — not watertight (open shell or unsewn faces)"})
    if hints:
        text = " ".join(hints.get("hints", []))
        ok = bool(hints.get("hints")) and "locate_gate_defects" in text and "sew" in text.lower()
        report.add("L1", "repair_hints points at the open-edge ladder", "PASS" if ok else "FAIL",
                   text[:160])

    advice = payload(call, "repair_advice[open shell]", "repair_advice",
                     {"error_text": "4 open edge(s) — not watertight",
                      "goal": "sew the shell into a solid"})
    if advice:
        ids = advice.get("matched_recipe_ids", [])
        ok = "open_shell_or_disjoint_edit_rebuild" in ids and bool(advice.get("recipes"))
        report.add("L1", "repair_advice matches the open-shell recipe", "PASS" if ok else "FAIL",
                   f"{ids}")

    call("execute[deliberate NameError]", "execute", {"code": BAD_CODE})
    failure = payload(call, "last_error", "last_error", {})
    if failure:
        ok = (failure.get("type") == "NameError" and failure.get("line") == 3
              and "nosuchname" in str(failure.get("message"))
              and "nosuchname" in str(failure.get("excerpt", "")))
        report.add("L1", "last_error locates the failing line", "PASS" if ok else "FAIL",
                   f"type={failure.get('type')}, line={failure.get('line')}")


# --------------------------------------------------------------------------
# Sandbox and path policy
# --------------------------------------------------------------------------

def check_sandbox_and_paths(session: Session) -> None:
    call, report = session.call, session.report

    allowed = executed(call, "execute[allowed imports]",
                       "from build123d import *\n"
                       "import math, json\n"
                       "import numpy as np\n"
                       "print(json.dumps({'r': round(float(np.hypot(3.0, 4.0)), 6),"
                       " 'tau': round(math.tau, 6)}))\n")
    if allowed is not None:
        ok = '"r": 5.0' in allowed and '"tau": 6.283185' in allowed
        report.add("L1", "execute allows math, json and numpy", "PASS" if ok else "FAIL",
                   allowed.strip().splitlines()[0][:120] if allowed.strip() else "")

    check_rejected(call, "execute[open() is blocked]", "execute",
                   {"code": "open('/etc/passwd').read()"}, in_band=in_band_error)
    check_rejected(call, "execute[subprocess is blocked]", "execute",
                   {"code": "import subprocess\nsubprocess.run(['id'])"}, in_band=in_band_error)
    check_rejected(call, "execute[os is blocked]", "execute",
                   {"code": "import os\nprint(os.listdir('/'))"}, in_band=in_band_error)
    check_rejected(call, "export[outside the write roots]", "export",
                   {"filename": "/etc/b123d-escape.step", "format": "step",
                    "object_name": "features"})
    check_rejected(call, "export[path traversal]", "export",
                   {"filename": "../../etc/b123d-escape.step", "format": "step",
                    "object_name": "features"})
    check_rejected(call, "export[unknown format]", "export",
                   {"filename": str(session.scratch / "x.foo"), "format": "foo",
                    "object_name": "features"})
    check_rejected(call, "import_cad_file[outside the read roots]", "import_cad_file",
                   {"path": "/etc/passwd"})
    check_rejected(call, "execute_file[outside the read roots]", "execute_file",
                   {"path": "/etc/hostname"})
    check_rejected(call, "measure[unknown object]", "measure", {"object_name": "nope"})
    check_rejected(call, "validate[unknown object]", "validate", {"object_name": "nope"},
                   in_band=in_band_error)
    check_rejected(call, "restore_snapshot[unknown name]", "restore_snapshot", {"name": "nope"},
                   in_band=in_band_error)
    check_rejected(call, "cross_sections[zero slices]", "cross_sections",
                   {"object_name": "features", "num_slices": 0}, in_band=in_band_error,
                   on_accept=lambda result: (
                       "WARN",
                       "num_slices=0 is accepted and silently sampled anyway: "
                       f"{text_of(result)[:100]!r}"))


# --------------------------------------------------------------------------
# Session management
# --------------------------------------------------------------------------

def check_session_tools(session: Session) -> None:
    call, report = session.call, session.report
    cwd = session.cwd

    before = payload(call, "measure[before snapshot]", "measure", {"object_name": "features"})
    saved = call("save_snapshot", "save_snapshot", {"name": "before"})
    if saved is not None:
        report.add("L1", "save_snapshot captures the named geometry",
                   "PASS" if "before" in text_of(saved) else "FAIL", text_of(saved)[:120])
    executed(call, "execute[extra object]",
             "from build123d import *\nshow(Box(3, 3, 3), 'extra')\n")
    diff = call("compare[snapshot]", "compare", {"a": "before", "kind": "snapshot"})
    if diff is not None:
        text = text_of(diff)
        ok = "extra" in text and "added" in text
        report.add("L1", "compare snapshot reports the added object", "PASS" if ok else "FAIL",
                   text.replace("\n", " ")[:160])
    restored = call("restore_snapshot", "restore_snapshot", {"name": "before"})
    if restored is not None:
        state = payload(call, "session_state[after restore]", "session_state", {})
        objects = (state or {}).get("objects", {})
        ok = "extra" not in objects and "features" in objects and "before" in (state or {}).get("snapshots", [])
        report.add("L1", "restore_snapshot drops later objects", "PASS" if ok else "FAIL",
                   f"objects={sorted(objects)}, snapshots={(state or {}).get('snapshots')}")
        if before:
            again = payload(call, "measure[after restore]", "measure", {"object_name": "features"})
            if again:
                compare_values(report, "restored geometry is unchanged", again,
                               {"volume": before.get("volume", 0.0)}, abs_tol=1e-6)

    health = payload(call, "health_check", "health_check", {})
    if health:
        caps = {k: v for k, v in health.items() if isinstance(v, dict)}
        ok = health.get("ok") is True and all(c.get("ok") for c in caps.values()) and len(caps) == 4
        report.add("L1", "health_check covers PNG, SVG, STEP and STL", "PASS" if ok else "FAIL",
                   f"{ {k: v.get('ok') for k, v in caps.items()} }")

    versions = call("version", "version", {})
    if versions is not None:
        text = text_of(versions)
        expected = [f"build123d-mcp: {_installed('build123d-mcp')}",
                    f"build123d: {_installed('build123d')}"]
        missing = [line for line in expected if line not in text]
        report.add("L1", "version matches the installed packages", "FAIL" if missing else "PASS",
                   f"missing {missing}; got {text.replace(chr(10), ' ')[:160]}" if missing
                   else "; ".join(expected))

    guide = call("workflow_hints", "workflow_hints", {})
    if guide is not None:
        text = text_of(guide)
        ok = "session_state()" in text and "health_check()" in text and len(text) > 500
        report.add("L1", "workflow_hints returns the guide", "PASS" if ok else "FAIL",
                   f"{len(text)} chars")

    sessions = payload(call, "list_sessions", "list_sessions", {})
    if sessions:
        ok = sessions.get("sessions") == 1 and sessions.get("mode") == "single-session"
        report.add("L1", "list_sessions reports the single stdio session",
                   "PASS" if ok else "FAIL", f"{sessions}")

    destroyed = call("destroy_session", "destroy_session", {}, allow_error=True)
    if destroyed is not None:
        text = text_of(destroyed)
        if "not enabled" in text and "reset()" in text:
            report.add("L1", "destroy_session is refused without per-handle sessions", "WARN",
                       "stdio runs one CAD session, so destroy_session only points at reset() — "
                       "it cannot be used to isolate work inside a run")
        else:
            report.add("L1", "destroy_session is refused without per-handle sessions", "FAIL",
                       text[:160])

    skill = call("install_skill[claude]", "install_skill",
                 {"target": "claude", "skill": "modeling"})
    if skill is not None:
        installed = cwd / ".claude" / "skills" / "b123d-modeling" / "SKILL.md"
        ok = installed.is_file() and installed.stat().st_size > 200
        report.add("L1", "install_skill writes into the server's cwd", "PASS" if ok else "FAIL",
                   f"{text_of(skill)[:120]}")
        again = call("install_skill[already installed]", "install_skill",
                     {"target": "claude", "skill": "modeling"}, allow_error=True)
        if again is not None:
            text = text_of(again)
            ok = "Already installed" in text and "force=True" in text
            report.add("L1", "install_skill refuses to overwrite without force",
                       "PASS" if ok else "FAIL", text[:120])
        agents = call("install_skill[agents-md]", "install_skill",
                      {"target": "agents-md", "skill": "repair"})
        if agents is not None:
            path = cwd / "AGENTS.md"
            ok = path.is_file() and "b123d" in path.read_text(encoding="utf-8", errors="replace")
            report.add("L1", "install_skill writes AGENTS.md for Codex", "PASS" if ok else "FAIL",
                       text_of(agents)[:120])

    reset = call("reset", "reset", {})
    if reset is not None:
        state = payload(call, "session_state[after reset]", "session_state", {})
        ok = (state or {}).get("current_shape") is None and not (state or {}).get("objects")
        report.add("L1", "reset empties the session", "PASS" if ok else "FAIL", f"{state}")


def _installed(package: str) -> str | None:
    import importlib.metadata as md
    try:
        return md.version(package)
    except md.PackageNotFoundError:
        return None


# --------------------------------------------------------------------------
# 2D drawing suite (deprecated upstream at this revision)
# --------------------------------------------------------------------------

SKETCH_CODE = """from build123d import *
with BuildSketch() as sk:
    Rectangle(40, 20)
    Circle(5, mode=Mode.SUBTRACT)
show(sk.sketch, 'face2d')
"""
SKETCH_W, SKETCH_H, SVG_MARGIN = 40.0, 20.0, 5.2
ANNOTATED_CODE = """from build123d import *
from build123d_drafting import draft_preset, Dimension
with BuildSketch() as sk:
    Rectangle(40, 20)
show(sk.sketch, 'plan')
draft = draft_preset(font_size=2.5, decimal_precision=1)
dim = Dimension((-20, -10), (20, -10), 'below', 6.0, draft)
annotate(dim, 'width_dim')
"""
LAYOUT_CODE = """from build123d import *
show(Box(40, 20, 6), 'layoutbox')
"""
DISCS = ((80, 60, 30), (280, 70, 32), (160, 220, 34))   # centre x, centre y, radius


def write_png(path: Path, width: int, height: int, discs) -> None:
    """A minimal RGB PNG with filled discs — stdlib only, so the regions the
    server must find are known exactly."""
    import zlib
    rows = []
    for y in range(height):
        row = bytearray(b"\xff" * (3 * width))
        for cx, cy, r in discs:
            dy = y - cy
            if abs(dy) <= r:
                half = int(math.sqrt(r * r - dy * dy))
                row[3 * max(0, cx - half):3 * min(width, cx + half)] = \
                    b"\x00" * (3 * (min(width, cx + half) - max(0, cx - half)))
        rows.append(bytes(row))
    raw = b"".join(b"\x00" + row for row in rows)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    path.write_bytes(b"\x89PNG\r\n\x1a\n"
                     + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def svg_page(text: str) -> tuple[float, float, int, int]:
    """Page size, path count and native text count, parsed here."""
    import xml.etree.ElementTree as ET
    root = ET.fromstring(text)
    width = float(root.get("width", "0").removesuffix("mm"))
    height = float(root.get("height", "0").removesuffix("mm"))
    paths = sum(1 for _ in root.iter("{http://www.w3.org/2000/svg}path"))
    texts = sum(1 for _ in root.iter("{http://www.w3.org/2000/svg}text"))
    return width, height, paths, texts


def dxf_entities(text: str) -> dict[str, int]:
    import re
    found = re.findall(r"\n\s*0\r?\n\s*(LINE|CIRCLE|ARC|SPLINE|ELLIPSE|LWPOLYLINE|POLYLINE)\r?\n",
                       text)
    return {name: found.count(name) for name in sorted(set(found))}


def check_drawing(session: Session) -> None:
    call, report = session.call, session.report
    out = session.scratch
    deprecated: list[str] = []

    def note(tool: str, result: dict | None) -> None:
        if result is not None and DEPRECATION in text_of(result):
            deprecated.append(tool)

    if executed(call, "execute[2D sketch]", SKETCH_CODE) is not None:
        stem = out / "face2d"
        if call("export[svg,dxf]", "export",
                {"filename": str(stem), "format": "svg,dxf", "object_name": "face2d"}) is not None:
            svg, dxf = stem.with_suffix(".svg"), stem.with_suffix(".dxf")
            if svg.is_file() and dxf.is_file():
                width, height, paths, texts = svg_page(svg.read_text(encoding="utf-8"))
                ok = (close(width, SKETCH_W + 2 * SVG_MARGIN)
                      and close(height, SKETCH_H + 2 * SVG_MARGIN) and paths == 1 and texts == 0)
                report.add("L1", "exported SVG page and geometry", "PASS" if ok else "FAIL",
                           f"{width}×{height} mm, {paths} path(s), {texts} native <text>")
                entities = dxf_entities(dxf.read_text(encoding="utf-8", errors="replace"))
                want = {"CIRCLE": 1, "LINE": 4}
                report.add("L1", "exported DXF entities", "PASS" if entities == want else "FAIL",
                           f"{entities} (expected {want})")

                drawing = payload(call, "inspect_drawing[svg]", "inspect_drawing",
                                  {"svg_path": str(svg)})
                note("inspect_drawing", {"content": [{"type": "text", "text": json.dumps(drawing)}]})
                if drawing:
                    page = drawing.get("page", {})
                    ok = (close(page.get("width"), width) and close(page.get("height"), height)
                          and drawing.get("counts", {}).get("path") == paths
                          and drawing.get("text") == [])
                    report.add("L1", "inspect_drawing SVG mode matches our own parse",
                               "PASS" if ok else "FAIL", f"page={page}, counts={drawing.get('counts')}")
                lint = payload(call, "lint_drawing[svg]", "lint_drawing", {"svg_path": str(svg)})
                if lint is not None:
                    ok = lint.get("violations") == []
                    report.add("L1", "lint_drawing finds no export pathology",
                               "PASS" if ok else "FAIL", f"{lint.get('violations')}")
                png = out / "face2d.png"
                raster = call("render_drawing[svg]", "render_drawing",
                              {"svg_path": str(svg), "width": 600, "save_to": str(png)})
                note("render_drawing", raster)
                if raster is not None and png.is_file():
                    got = png_size(png.read_bytes())
                    expected_h = round(600 * height / width)
                    ok = got[0] == 600 and abs(got[1] - expected_h) <= 3 and image_blocks(raster)
                    report.add("L1", "render_drawing rasterises the SVG", "PASS" if ok else "FAIL",
                               f"{got[0]}×{got[1]} px (expected 600×{expected_h})")

    # The inspect_drawing docstring's own example (`from build123d_drafting import
    # Draft`) does not exist in the pinned helper release; draft_preset() does.
    check_rejected(call, "execute[documented Draft import]", "execute",
                   {"code": "from build123d_drafting import Draft\n"
                            "draft = Draft(font_size=2.5, decimal_precision=1)\n"},
                   in_band=in_band_error)

    if executed(call, "execute[annotated drawing]", ANNOTATED_CODE) is not None:
        drawing = payload(call, "inspect_drawing[session]", "inspect_drawing", {})
        if drawing:
            objects = drawing.get("objects", {})
            plan = objects.get("plan", {}).get("bbox", {})
            annotation = objects.get("width_dim", {}).get("annotation") or {}
            ok = (close(plan.get("width"), SKETCH_W) and close(plan.get("height"), SKETCH_H)
                  and annotation.get("type") == "Dimension"
                  and close(annotation.get("measured_length"), SKETCH_W)
                  and annotation.get("label_str") == f"{SKETCH_W:.1f}")
            report.add("L1", "inspect_drawing session mode carries the dimension metadata",
                       "PASS" if ok else "FAIL",
                       f"plan={plan}, measured={annotation.get('measured_length')}, "
                       f"label={annotation.get('label_str')!r}")
        lint = payload(call, "lint_drawing[session]", "lint_drawing", {})
        if lint is not None:
            report.add("L1", "lint_drawing session mode is clean for one dimension",
                       "PASS" if lint.get("violations") == [] else "FAIL",
                       f"{lint.get('violations')}")
        stem = out / "annotated"
        if call("export[annotated svg]", "export",
                {"filename": str(stem), "format": "svg", "object_name": "plan"}) is not None:
            svg = stem.with_suffix(".svg")
            sidecar = call("save_drawing_annotations", "save_drawing_annotations",
                           {"svg_path": str(svg)})
            note("save_drawing_annotations", sidecar)
            if sidecar is not None:
                documented = Path(str(svg) + ".dims.json")
                actual = stem.with_suffix(".dims.json")
                if actual.is_file() and not documented.is_file():
                    data = json.loads(actual.read_text(encoding="utf-8"))
                    values = json.dumps(data)
                    ok = f"{SKETCH_W}" in values
                    report.add("L1", "save_drawing_annotations writes <stem>.dims.json",
                               "WARN" if ok else "FAIL",
                               f"sidecar is {actual.name}, not the documented "
                               f"'<svg_path>.dims.json' ({documented.name}); "
                               f"content {values[:120]}")
                else:
                    report.add("L1", "save_drawing_annotations writes <stem>.dims.json",
                               "PASS" if documented.is_file() else "FAIL",
                               f"{text_of(sidecar)[-120:]}")

    axes = payload(call, "view_axes[bottom]", "view_axes", {"viewport_origin": [0, 0, -100]})
    if axes:
        want = {"world_X": ["page_X", -1.0], "world_Y": ["page_Y", 1.0], "world_Z": ["depth", 0.0]}
        got = {k: axes.get(k) for k in want}
        report.add("L1", "view_axes maps a bottom view", "PASS" if got == want else "FAIL",
                   f"{got} (expected {want})")
    axes = payload(call, "view_axes[front]", "view_axes",
                   {"viewport_origin": [0, -100, 0], "viewport_up": [0, 0, 1]})
    if axes:
        want = {"world_X": ["page_X", 1.0], "world_Y": ["depth", 0.0], "world_Z": ["page_Y", 1.0]}
        got = {k: axes.get(k) for k in want}
        report.add("L1", "view_axes maps a front view", "PASS" if got == want else "FAIL",
                   f"{got} (expected {want})")

    if executed(call, "execute[layout box]", LAYOUT_CODE) is not None:
        layout = payload(call, "suggest_view_layout[layoutbox]", "suggest_view_layout",
                         {"object_name": "layoutbox", "page_w": 297.0, "page_h": 210.0})
        if layout:
            views = layout.get("views", {})
            want = {"plan": (20.0, 10.0), "front": (20.0, 3.0), "side": (10.0, 3.0)}
            got = {name: (views.get(name, {}).get("half_w"), views.get(name, {}).get("half_h"))
                   for name in want}
            ok = all(close(got[name][0], want[name][0]) and close(got[name][1], want[name][1])
                     for name in want) and not layout.get("warnings")
            report.add("L1", "suggest_view_layout sizes the views from the bbox",
                       "PASS" if ok else "FAIL", f"{got} (expected {want})")

    drawing_png = out / "sheet.png"
    write_png(drawing_png, 400, 300, DISCS)
    regions = payload(call, "prepare_drawing[discs]", "prepare_drawing",
                      {"image_path": str(drawing_png), "output_dir": str(out / "regions"),
                       "max_regions": 6, "padding": 4})
    if regions:
        found = regions.get("regions", [])
        boxes = sorted(tuple(r["bbox_px"]) for r in found)
        want = sorted((max(0, cx - r - 4), max(0, cy - r - 4), min(400, cx + r + 4),
                       min(300, cy + r + 4)) for cx, cy, r in DISCS)
        crops = all(Path(r["crop"]).is_file() for r in found)
        ok = (regions.get("image_size_px") == [400, 300] and len(found) == len(DISCS)
              and all(all(abs(g - w) <= 2 for g, w in zip(box, target))
                      for box, target in zip(boxes, want)) and crops
              and Path(regions.get("overview", "")).is_file())
        report.add("L1", "prepare_drawing finds the three ink regions", "PASS" if ok else "FAIL",
                   f"{len(found)} region(s) {boxes} (expected {want})")

    crop = payload(call, "crop_drawing[exact]", "crop_drawing",
                   {"image_path": str(drawing_png), "bbox_px": [10, 10, 110, 60],
                    "output_path": str(out / "crop.png"), "scale": 2.0})
    if crop:
        saved = Path(crop.get("crop", ""))
        size = png_size(saved.read_bytes()) if saved.is_file() else (0, 0)
        mapping = (crop.get("coordinate_mapping") or {}).get("crop_to_source")
        ok = (crop.get("crop_size_px") == [200, 100] and size == (200, 100)
              and mapping == [[0.5, 0.0, 10], [0.0, 0.5, 10], [0.0, 0.0, 1.0]])
        report.add("L1", "crop_drawing scales and maps exactly", "PASS" if ok else "FAIL",
                   f"crop {size} px, mapping {mapping}")

    if deprecated:
        report.add("L1", "the 2D drawing tools are deprecated upstream", "WARN",
                   f"{sorted(set(deprecated))} answer with the #465 notice (off by default in "
                   f"0.4.0, removed in 0.5.0; the suite moves to draftwright)")


def check_coverage(call: Caller, report: Report, expected: list[str]) -> None:
    missing = sorted(set(expected) - set(call.stdout_by_tool))
    report.add("L1", "every listed tool was called", "FAIL" if missing else "PASS",
               f"missing {missing}" if missing else f"{len(expected)} tools")


def run_l1(session: Session) -> None:
    # Order matters: the session is stateful, reset() and execute_file() clear it.
    check_core(session)
    check_io(session)
    check_recognisers(session)
    check_diagnostics(session)
    check_gate_and_comparisons(session)
    check_sandbox_and_paths(session)
    check_session_tools(session)
    check_design_audit(session)
    check_execute_file(session)
    check_drawing(session)
    check_coverage(session.call, session.report, session.entry["expected_tools"])


SMOKE = Smoke(
    server="build123d",
    run_l1=run_l1,
    packages=("mcp", "build123d", "build123d-mcp", "cadquery-ocp-novtk", "vtk",
              "b123d-recognisers", "build123d-drafting-helpers", "augura", "scipy"),
    expected_cwd_files=(".claude", "AGENTS.md"),
)
