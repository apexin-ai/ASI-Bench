"""e2e_smoke/servers/build123d.py without build123d: closed-form references, mesh/STEP/SVG/DXF
readers, in-band error handling and the stateful check groups against stub responses."""
import json
import math
import struct

import pytest

from . import support
from .support import StubClient, report_statuses, rpc_json, rpc_text, runner

b = support.smoke_module("build123d")

# Values measured from the pinned server on 2026-10-03 (aarch64, build123d 0.11.0,
# cadquery-ocp-novtk 7.9.3.1). They are hard-coded here so a change in the closed
# forms shows up offline, without OCC.
PLATE_VOLUME = 22654.4248
PLATE_AREA = 6822.0353
PLATE_INERTIA = {"Ixx": 3301898.1052, "Iyy": 7199390.7038, "Izz": 10123715.0622}
FEATURES_VOLUME = 48799.6421
FLANGE_VOLUME = 26917.1659
SPLIT_VOLUME = 1848.3046


# --------------------------------------------------------------------------
# Closed-form solids
# --------------------------------------------------------------------------

def test_plate_closed_forms_match_the_server():
    assert b.PLATE.volume == pytest.approx(PLATE_VOLUME, abs=5e-5)
    assert b.PLATE.area == pytest.approx(PLATE_AREA, abs=5e-5)
    for name, value in PLATE_INERTIA.items():
        assert b.PLATE.inertia[name] == pytest.approx(value, abs=5e-4)
    assert b.PLATE.inertia["Ixy"] == b.PLATE.inertia["Ixz"] == b.PLATE.inertia["Iyz"] == 0.0
    assert b.PLATE.faces == 11
    assert b.PLATE.bbox == {"xsize": 60.0, "ysize": 40.0, "zsize": 10.0}


def test_plate_degenerates_to_a_plain_box():
    box = b.Plate(x=20.0, y=10.0, z=4.0, hole_d=0.0, fillet_r=0.0)
    assert box.volume == pytest.approx(800.0, abs=1e-12)
    assert box.area == pytest.approx(2 * 200.0 + 4.0 * 2 * (20.0 + 10.0), abs=1e-12)
    # a prism's inertia about its own centroid
    assert box.inertia["Izz"] == pytest.approx(800.0 * (20.0**2 + 10.0**2) / 12, abs=1e-9)
    assert box.inertia["Ixx"] == pytest.approx(800.0 * (10.0**2 + 4.0**2) / 12, abs=1e-9)


def test_plate_fillets_and_bore_only_remove_material():
    plain = b.Plate(x=60.0, y=40.0, z=10.0, hole_d=0.0, fillet_r=0.0)
    filleted = b.Plate(x=60.0, y=40.0, z=10.0, hole_d=0.0, fillet_r=5.0)
    assert filleted.volume < plain.volume < 60 * 40 * 10 + 1e-9
    # four corners, each a square minus a quarter disc
    removed = 4 * (25.0 - math.pi * 25.0 / 4) * 10.0
    assert plain.volume - filleted.volume == pytest.approx(removed, abs=1e-9)
    assert b.PLATE.volume == pytest.approx(filleted.volume - math.pi * 36.0 * 10.0, abs=1e-9)


def test_feature_and_flange_references_match_the_server():
    assert b.FEATURES_VOLUME == pytest.approx(FEATURES_VOLUME, abs=1e-3)
    assert b.FLANGE_VOLUME == pytest.approx(FLANGE_VOLUME, abs=1e-3)
    assert b.SPLIT_VOLUME == pytest.approx(SPLIT_VOLUME, abs=1e-3)
    assert b.CS_CONE_DEPTH == pytest.approx(2.8759, abs=5e-5)
    assert len(b.BOLT_LOCATIONS) == b.BOLT_N == 6
    assert (b.BCD / 2, 0.0) == b.BOLT_LOCATIONS[0]
    for x, y in b.BOLT_LOCATIONS:
        assert math.hypot(x, y) == pytest.approx(b.BCD / 2, abs=1e-3)
    assert b.SHAFT_VOLUME < b.BLOCK_VOLUME and b.FIT_CLEARANCE == pytest.approx(0.05, abs=1e-12)


def test_the_fixture_code_is_what_the_closed_forms_describe():
    for value in (60.0, 40.0, 10.0, 12.0, 5.0):
        assert f"= {value}" in b.PLATE.code
    assert "from build123d import *" in b.PLATE.code and "show(plate.part, 'plate')" in b.PLATE.code
    assert "PolarLocations(bcd / 2, 6)" in b.FLANGE_CODE
    assert "CounterSinkHole" in b.FEATURES_CODE and "CounterBoreHole" in b.FEATURES_CODE


# --------------------------------------------------------------------------
# Mesh, STEP, SVG, DXF and PNG readers
# --------------------------------------------------------------------------

def binary_stl(triangles) -> bytes:
    out = bytearray(b"\0" * 80 + struct.pack("<I", len(triangles)))
    for tri in triangles:
        out += struct.pack("<3f", 0.0, 0.0, 0.0)
        for vertex in tri:
            out += struct.pack("<3f", *vertex)
        out += b"\0\0"
    return bytes(out)


def cube_mesh(side=2.0):
    """A closed, outward-oriented triangle mesh of a cube centred on the origin."""
    h = side / 2
    corners = [(x, y, z) for x in (-h, h) for y in (-h, h) for z in (-h, h)]

    def quad(a, c, d, e):
        return [(corners[a], corners[c], corners[d]), (corners[a], corners[d], corners[e])]

    faces = []
    faces += quad(0, 1, 3, 2)      # x = -h
    faces += quad(4, 6, 7, 5)      # x = +h
    faces += quad(0, 4, 5, 1)      # y = -h
    faces += quad(2, 3, 7, 6)      # y = +h
    faces += quad(0, 2, 6, 4)      # z = -h
    faces += quad(1, 5, 7, 3)      # z = +h
    return faces


def test_parse_binary_stl_round_trip_and_rejections():
    triangles = cube_mesh()
    data = binary_stl(triangles)
    assert b.parse_binary_stl(data) == pytest.approx(triangles)
    with pytest.raises(ValueError, match="too short"):
        b.parse_binary_stl(b"\0" * 20)
    with pytest.raises(ValueError, match="declares"):
        b.parse_binary_stl(data[:-10])
    ascii_stl = b"solid x\n facet normal 0 0 1\n" + b" " * 200
    with pytest.raises(ValueError, match="ASCII"):
        b.parse_binary_stl(ascii_stl)


def test_mesh_volume_bbox_and_watertightness():
    triangles = cube_mesh(2.0)
    assert b.mesh_volume(triangles) == pytest.approx(8.0, abs=1e-9)
    assert b.mesh_bbox(triangles) == pytest.approx((2.0, 2.0, 2.0))
    assert b.mesh_open_edges(triangles) == 0
    # dropping one triangle leaves three unpaired edges
    assert b.mesh_open_edges(triangles[:-1]) == 3
    # a reversed orientation must not change the reported volume
    flipped = [(c, bb, a) for a, bb, c in triangles]
    assert b.mesh_volume(flipped) == pytest.approx(8.0, abs=1e-9)


MINIMAL_STEP = (
    "ISO-10303-21;\nHEADER;\nFILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\nENDSEC;\nDATA;\n"
    "#1=PRODUCT('plate','plate','',(#2));\n#3=MANIFOLD_SOLID_BREP('',#4);\n"
    + "".join(f"#{i}=CARTESIAN_POINT('',(0.,0.,0.));\n" for i in range(10, 20))
    + "ENDSEC;\nEND-ISO-10303-21;\n"
)


def test_step_problems_accepts_a_well_formed_file_and_names_every_defect():
    assert b.step_problems(MINIMAL_STEP, "plate") == []
    assert b.step_problems(MINIMAL_STEP, "other") == ["session label 'other' not carried into the file"]
    assert "missing ISO-10303-21 header" in b.step_problems(MINIMAL_STEP[13:], "plate")
    truncated = MINIMAL_STEP.replace("END-ISO-10303-21;", "")
    assert "missing END-ISO-10303-21 terminator" in b.step_problems(truncated, "plate")
    no_schema = MINIMAL_STEP.replace("FILE_SCHEMA", "FILE_NAME")
    assert "no FILE_SCHEMA" in b.step_problems(no_schema, "plate")
    no_brep = MINIMAL_STEP.replace("MANIFOLD_SOLID_BREP", "GEOMETRIC_CURVE_SET")
    assert "no solid B-rep entity" in b.step_problems(no_brep, "plate")
    thin = MINIMAL_STEP.replace("CARTESIAN_POINT", "POINT_X", 5)
    assert any("CARTESIAN_POINT" in problem for problem in b.step_problems(thin, "plate"))


SVG = (
    '<svg width="50.4mm" height="30.4mm" viewBox="-25.2 -15.2 50.4 30.4" '
    'xmlns="http://www.w3.org/2000/svg"><g><path d="M0,0"/></g>'
    '<text x="1" y="2">40.0</text></svg>'
)


def test_svg_and_dxf_parsers():
    assert b.svg_page(SVG) == (50.4, 30.4, 1, 1)
    dxf = ("\n  0\nSECTION\n  2\nENTITIES\n"
           "  0\nLINE\n  8\nface2d\n 10\n0.0\n"
           "  0\nLINE\n  8\nface2d\n 10\n1.0\n"
           "  0\nCIRCLE\n  8\nface2d\n 40\n5.0\n"
           "  0\nENDSEC\n")
    assert b.dxf_entities(dxf) == {"CIRCLE": 1, "LINE": 2}
    assert b.dxf_entities("no entities here") == {}


def test_write_png_places_the_discs_where_the_check_expects_them(tmp_path):
    path = tmp_path / "sheet.png"
    b.write_png(path, 400, 300, b.DISCS)
    data = path.read_bytes()
    assert struct.unpack(">II", data[16:24]) == (400, 300)
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and data[-8:-4] == b"IEND"
    import zlib
    idat = data[data.index(b"IDAT") + 4:]
    raw = zlib.decompressobj().decompress(idat)
    rows = [raw[i * (1 + 3 * 400) + 1:(i + 1) * (1 + 3 * 400)] for i in range(300)]
    for cx, cy, r in b.DISCS:
        assert rows[cy][3 * cx:3 * cx + 3] == b"\x00\x00\x00"          # centre is ink
        assert rows[cy][3 * (cx + r + 2):3 * (cx + r + 2) + 3] == b"\xff\xff\xff"


# --------------------------------------------------------------------------
# Comparison and in-band error helpers
# --------------------------------------------------------------------------

def test_close_and_compare_values():
    assert b.close(1.00004, 1.0) and not b.close(1.001, 1.0)
    assert b.close(1e9 + 0.4, 1e9)          # absolute tolerance is the 4-decimal rounding
    assert not b.close(None, 1.0) and not b.close(True, 1.0) and not b.close("1.0", 1.0)
    report = runner.Report()
    b.compare_values(report, "ok", {"a": 1.0, "b": 2.0}, {"a": 1.0})
    b.compare_values(report, "bad", {"a": 1.5}, {"a": 1.0})
    assert report_statuses(report) == {"ok": "PASS", "bad": "FAIL"}
    assert "expected 1.0" in report.checks[-1]["detail"]


def test_in_band_error_recognises_both_shapes():
    assert b.in_band_error(rpc_text("Error: NameError: x\n\nHint: …")["result"]).startswith("Error: NameError")
    assert b.in_band_error(rpc_json({"error": "Unknown object 'nope'"})["result"]) == "Unknown object 'nope'"
    assert b.in_band_error(rpc_text("Registered 'plate': volume=1")["result"]) is None
    assert b.in_band_error(rpc_json({"volume": 1.0})["result"]) is None


def test_executed_fails_on_an_in_band_execute_error():
    report = runner.Report()
    call = runner.Caller(StubClient({"execute": rpc_text("Error: SecurityError: nope")}), report)
    assert b.executed(call, "build", "code") is None
    assert report_statuses(report) == {"build": "FAIL"}
    report = runner.Report()
    call = runner.Caller(StubClient({"execute": rpc_text("Registered 'plate': volume=1")}), report)
    assert b.executed(call, "build", "code") == "Registered 'plate': volume=1"
    assert report.checks == []


def test_payload_tolerates_a_text_preamble_and_fails_on_prose():
    report = runner.Report()
    call = runner.Caller(
        StubClient({"validate": rpc_text('Validity gate: PASS\n{"passes_gate": true}'),
                    "version": rpc_text("build123d-mcp: 0.3.85")}), report)
    assert b.payload(call, "gate", "validate", {}) == {"passes_gate": True}
    assert b.payload(call, "prose", "version", {}) is None
    assert report_statuses(report) == {"prose": "FAIL"}


def test_at_finds_a_record_by_its_xy_location():
    records = [{"location": [25.0, 0.0, 11.0], "diameter": 8.0},
               {"location": [0.0, -20.0, 5.0], "diameter": 5.0}]
    assert b.at(records, "location", 25.0, 0.0)["diameter"] == 8.0
    assert b.at(records, "location", 0.0, -20.0)["diameter"] == 5.0
    assert b.at(records, "location", 1.0, 1.0) == {}
    assert b.at([{"location": [1.0]}], "location", 1.0, 0.0) == {}


def test_face_inventory_requires_every_analytic_face():
    z, f, hole = b.PLATE.z, b.PLATE.fillet_r, b.PLATE.hole_d
    good = [
        {"type": "Cylinder", "area": round(math.pi * hole * z, 4), "diameter": hole},
        {"type": "Cylinder", "area": round(math.pi * f / 2 * z, 4), "diameter": 2 * f, "count": 4},
        {"type": "Plane", "area": round(b.PLATE.section_area, 4), "count": 2},
        {"type": "Plane", "area": round((b.PLATE.x - 2 * f) * z, 4), "count": 2},
        {"type": "Plane", "area": round((b.PLATE.y - 2 * f) * z, 4), "count": 2},
    ]
    report = runner.Report()
    b.check_face_inventory(report, good)
    b.check_face_inventory(report, good[:-1])
    b.check_face_inventory(report, good[:1] + [dict(good[1], count=3)] + good[2:])
    assert [c["status"] for c in report.checks] == ["PASS", "FAIL", "FAIL"]


def test_check_mesh_classifies_volume_watertightness_and_bbox():
    report = runner.Report()
    b.check_mesh(report, "STL", b"too short")
    assert report_statuses(report)["exported STL parses"] == "FAIL"

    # a box with the plate's bounding box: watertight and right bbox, wrong volume
    triangles = [tuple((x * 30.0, y * 20.0, z * 5.0) for x, y, z in tri) for tri in cube_mesh(2.0)]
    report = runner.Report()
    b.check_mesh(report, "STL", binary_stl(triangles))
    statuses = report_statuses(report)
    assert statuses["exported STL is watertight"] == "PASS"
    assert statuses["exported STL volume matches the exact solid"] == "FAIL"   # a box is not the plate
    assert statuses["exported STL bounding box"] == "PASS"


# --------------------------------------------------------------------------
# Check groups against stub responses
# --------------------------------------------------------------------------

def session_for(responses, tmp_path, entry=None):
    report = runner.Report()
    client = StubClient(responses)
    session = runner.Session(
        smoke=b.SMOKE, args=None, entry=entry or {"expected_tools": []},
        server={"command": "/x/.venv/bin/build123d-mcp", "args": []},
        checkout=tmp_path / "checkout", report=report, tmp=tmp_path,
        home=tmp_path, cwd=tmp_path / "cwd", scratch=tmp_path / "scratch", env={},
        client=client,
    )
    session.cwd.mkdir(exist_ok=True)
    session.scratch.mkdir(exist_ok=True)
    session.call = runner.Caller(client, report)
    return session


def test_check_core_passes_on_the_server_answers(tmp_path):
    plate = b.PLATE
    measured = {
        "volume": round(plate.volume, 4), "area": round(plate.area, 4),
        "topology": {"faces": 11, "edges": 27, "vertices": 18},
        "bbox": {"xsize": 60.0, "ysize": 40.0, "zsize": 10.0},
        "center_of_mass": {"x": 0.0, "y": -0.0, "z": 0.0},
        "inertia": {k: round(v, 4) for k, v in plate.inertia.items()},
        "face_inventory": [
            {"type": "Cylinder", "area": round(math.pi * plate.hole_d * plate.z, 4),
             "diameter": plate.hole_d},
            {"type": "Cylinder", "area": round(math.pi * plate.fillet_r / 2 * plate.z, 4),
             "diameter": 2 * plate.fillet_r, "count": 4},
            {"type": "Plane", "area": round(plate.section_area, 4), "count": 2},
            {"type": "Plane", "area": round((plate.x - 2 * plate.fillet_r) * plate.z, 4), "count": 2},
            {"type": "Plane", "area": round((plate.y - 2 * plate.fillet_r) * plate.z, 4), "count": 2},
        ],
    }
    mass = dict(measured, mass_g=round(plate.volume * b.STEEL_DENSITY / 1000, 4))
    responses = {
        "execute": rpc_text("Registered 'plate': volume=2.265e+04 mm³, faces=11"),
        "measure": lambda args: rpc_json(mass if args.get("material") else measured),
        "validate": rpc_text("Validity gate: PASS\n" + json.dumps({
            "passes_gate": True, "n_solids": 1, "watertight_manifold": True, "brep_valid": True,
            "open_edges": 0, "nonmanifold_edges": 0, "reasons": [],
            "volume": round(plate.volume, 4)})),
        "inspect_part": rpc_json({
            "bbox": {"x": 60.0, "y": 40.0, "z": 10.0},
            "holes": {"count": 1, "groups": [{"diameter": 12.0, "depth": 10.0,
                                              "bottom": "through", "count": 1}]},
            "sections": {"constant_section": True, "variation_ratio": 0.0}}),
        "cross_sections": rpc_json([{"position": p, "area": round(plate.section_area, 4)}
                                    for p in range(7)]),
        "session_state": rpc_json({
            "objects": {"plate": {"volume": round(plate.volume, 4)}},
            "variables": {"plate_x": {"value": 60.0}, "plate_y": {"value": 40.0},
                          "plate_z": {"value": 10.0}, "hole_d": {"value": 12.0},
                          "fillet_r": {"value": 5.0}}}),
        "script": rpc_json({"script": plate.code, "blocks": 1}),
        "resolve": lambda args: rpc_json(
            {"type": "Face", "geom_type": "PLANE", "area": round(plate.section_area, 6),
             "center": [0.0, 0.0, 5.0]} if ".sort_by" in args["selector"]
            else {"count": 5, "entities": [
                {"area": round(math.pi * plate.hole_d * plate.z, 4)},
                *[{"area": round(math.pi * plate.fillet_r / 2 * plate.z, 4)} for _ in range(4)]]}
            if "filter_by(GeomType" in args["selector"]
            else {"error": "Selector evaluation failed: 'Face' object is not callable"}),
    }
    session = session_for(responses, tmp_path)
    b.check_core(session)
    statuses = report_statuses(session.report)
    assert "FAIL" not in statuses.values(), [c for c in session.report.checks if c["status"] == "FAIL"]
    assert statuses["resolve[documented .last() example]"] == "WARN"


def test_check_core_reports_a_wrong_volume(tmp_path):
    responses = {
        "execute": rpc_text("Registered 'plate': volume=1 mm³, faces=11"),
        "measure": rpc_json({"volume": 22654.0, "area": round(b.PLATE.area, 4)}),
    }
    session = session_for(responses, tmp_path)
    try:
        b.check_core(session)
    except KeyError:
        pass        # the stub stops after the first few tools
    statuses = report_statuses(session.report)
    assert statuses["measure volume and area"] == "FAIL"


def test_check_coverage_lists_the_tools_never_called(tmp_path):
    session = session_for({"version": rpc_text("x")}, tmp_path,
                          entry={"expected_tools": ["version", "measure"]})
    session.call("v", "version", {})
    b.check_coverage(session.call, session.report, ["version", "measure"])
    check = session.report.checks[-1]
    assert check["status"] == "FAIL" and "measure" in check["detail"]
    assert set(session.call.stdout_by_tool) == {"version"}


def test_smoke_declares_the_artefacts_install_skill_writes():
    assert b.SMOKE.server == "build123d"
    assert set(b.SMOKE.expected_cwd_files) == {".claude", "AGENTS.md"}
    assert "build123d" in b.SMOKE.packages and "vtk" in b.SMOKE.packages


def test_every_documented_defect_is_a_warn_check():
    """The docstring's defect list and the WARN checks must not drift apart."""
    source = support.BUNDLE.joinpath("e2e_smoke/servers/build123d.py").read_text()
    for marker in ('"WARN"', "in_band=in_band_error", "note_deprecated"):
        assert marker in source
    for subject in ("n_solids", "brittle=0", "COMPOUND", "shell, not a solid",
                    "dims.json", "destroy_session", "deprecated"):
        assert subject in source
