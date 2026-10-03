"""build123d_plate_measure (model a bolted cover plate, gate it, measure it, export STEP+STL):
generator, scorers and verifier scenarios. No MCP server and no build123d at import time."""
import json
import math
import struct

import pytest

from . import support
from .support import claude, codex, jsonl

TASK = support.Task("mcp_e2e.build123d_plate_measure")
TASK_DIR, TASK_ID, INSTANCE_ID = TASK.dir, TASK.task_id, TASK.instance_id
SERVER = "build123d"
TOOLS = support.setup.load_manifest()[SERVER]["expected_tools"]

generate_gt = TASK.module("generate_gt")
scorer = TASK.module("custom_scorer")
verify = support.verify
_status = support.statuses

# Seed 31415: motor_mount, 90 x 54 x 12 mm, r10 corners, 16 mm bore, 8 x M8 on a 34 mm
# bolt circle, titanium. The numbers are what the pinned server answered on 2026-10-03
# (4 decimals), and they equal the closed form of generate_gt to <= 5e-5.
CASE = {
    "part": "motor_mount", "plate_x": 90.0, "plate_y": 54.0, "plate_z": 12.0,
    "corner_radius": 10.0, "bore_d": 16.0, "bolt_count": 8, "bolt_d": 8.0,
    "material": "titanium", "bolt_circle_d": 34.0,
    "export_stem": "/tmp/asibench-b123d-31415-motor_mount",
    "step_file": "part.step", "stl_file": "part.stl",
}
MEASURED = {"volume": 50051.6817, "area": 14607.8581, "mass_g": 221.7289, "izz": 219304.1548}
REFERENCE = {
    **CASE,
    "volume_mm3": MEASURED["volume"], "surface_area_mm2": MEASURED["area"],
    "mass_g": MEASURED["mass_g"], "izz_g_mm2": MEASURED["izz"],
    "reimported_volume_mm3": MEASURED["volume"],
    "bbox_mm": [90.0, 54.0, 12.0], "hole_count": 9, "bolt_pattern_count": 1,
    "bolt_hole_diameter_mm": 8.0, "face_count": 19, "n_solids": 1, "passes_gate": True,
    "density_g_cm3": 4.43,
    "export_files": ["asibench-b123d-31415-motor_mount.step",
                     "asibench-b123d-31415-motor_mount.stl"],
}
STEP_PATH = CASE["export_stem"] + ".step"
STL_PATH = CASE["export_stem"] + ".stl"


# --------------------------------------------------------------------------
# Submission artefacts
# --------------------------------------------------------------------------

def _answer(**over):
    data = {"part": CASE["part"], "volume_mm3": MEASURED["volume"],
            "surface_area_mm2": MEASURED["area"], "mass_g": MEASURED["mass_g"],
            "izz_g_mm2": MEASURED["izz"], "reimported_volume_mm3": MEASURED["volume"],
            "bbox_mm": [90.0, 54.0, 12.0], "hole_count": 9,
            "bolt_hole_diameter_mm": 8.0, "n_solids": 1, "passes_gate": True}
    data.update(over)
    return data


def _box_mesh(x=90.0, y=54.0, z=12.0, offset=(0.0, 0.0, 0.0)):
    """A closed, outward-oriented triangle mesh of a box: the shape of a tessellated plate
    without its curved faces (volume x*y*z)."""
    ox, oy, oz = offset
    corners = [(ox + sx * x / 2, oy + sy * y / 2, oz + sz * z / 2)
               for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]

    def quad(a, b, c, d):
        return [(corners[a], corners[b], corners[c]), (corners[a], corners[c], corners[d])]

    return (quad(0, 1, 3, 2) + quad(4, 6, 7, 5) + quad(0, 4, 5, 1)
            + quad(2, 3, 7, 6) + quad(0, 2, 6, 4) + quad(1, 5, 7, 3))


def _stl(triangles) -> bytes:
    """A binary STL of the given triangles, as the server writes them."""
    out = bytearray(b"\0" * 80 + struct.pack("<I", len(triangles)))
    for triangle in triangles:
        out += struct.pack("<3f", 0.0, 0.0, 0.0)
        for vertex in triangle:
            out += struct.pack("<3f", *vertex)
        out += b"\0\0"
    return bytes(out)


def _frame_mesh(x=90.0, y=54.0, z=12.0, volume=None):
    """A box x*y*z with one rectangular through hole: watertight, one component, and with a
    volume we can dial in — the topology of a tessellated plate, without 2000 facets."""
    hole_x = 26.0
    hole_y = (x * y * z - (volume if volume is not None else x * y * z / 2)) / z / hole_x
    outer = [(-x / 2, -y / 2), (x / 2, -y / 2), (x / 2, y / 2), (-x / 2, y / 2)]
    inner = [(-hole_x / 2, -hole_y / 2), (hole_x / 2, -hole_y / 2),
             (hole_x / 2, hole_y / 2), (-hole_x / 2, hole_y / 2)]
    triangles = []
    for height, up in ((z / 2, True), (-z / 2, False)):           # the two ring faces
        for i in range(4):
            j = (i + 1) % 4
            a, b = (*outer[i], height), (*outer[j], height)
            c, d = (*inner[j], height), (*inner[i], height)
            triangles += [(a, b, c), (a, c, d)] if up else [(a, c, b), (a, d, c)]
    for i in range(4):                                            # the four walls, twice
        j = (i + 1) % 4
        for ring, wind in ((outer, False), (inner, True)):
            a, b = (*ring[i], z / 2), (*ring[j], z / 2)
            c, d = (*ring[j], -z / 2), (*ring[i], -z / 2)
            triangles += [(a, b, c), (a, c, d)] if wind else [(a, c, b), (a, d, c)]
    return triangles


def _plate_stl(volume=None) -> bytes:
    """The submitted mesh: the plate's bounding box and a volume inside the tessellation band."""
    return _stl(_frame_mesh(volume=MEASURED["volume"] * 1.00003 if volume is None else volume))


def _step(label=CASE["part"], *, body=True, header=True, terminator=True, schema=True,
          points=12) -> str:
    lines = ["ISO-10303-21;" if header else "ISO-1-21;", "HEADER;"]
    if schema:
        lines.append("FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 }'));")
    lines += ["ENDSEC;", "DATA;", f"#7 = PRODUCT('{label}','{label}','',(#8));"]
    if body:
        lines.append("#9 = MANIFOLD_SOLID_BREP('',#10);")
    lines += [f"#{20 + i} = CARTESIAN_POINT('',(0.,0.,0.));" for i in range(points)]
    lines += ["ENDSEC;"]
    if terminator:
        lines.append("END-ISO-10303-21;")
    return "\n".join(lines) + "\n"


def _files(step=None, stl=None) -> dict:
    return {CASE["step_file"]: step if step is not None else _step(),
            CASE["stl_file"]: stl if stl is not None else _plate_stl()}


def _dirs(tmp_path, prediction, reference=REFERENCE, *, step=None, stl=None):
    pred, ref = support.score_dirs(tmp_path, prediction, reference)
    (pred / CASE["step_file"]).write_text(step if step is not None else _step())
    (pred / CASE["stl_file"]).write_bytes(stl if stl is not None else _plate_stl())
    return pred, ref


def _gate(pred, ref) -> float:
    item = TASK.eval_config()["gates"][0]
    from ai4sci_bench.core.scorer import get_scorer
    return get_scorer(item["scorer"]).score(pred, ref, {**item["config"], "weight": 1.0}).score


# --------------------------------------------------------------------------
# Agent logs
# --------------------------------------------------------------------------

def _b1_calls(*, export_text=None, path=STEP_PATH, measured=None, holes=9, pattern_d=8.0):
    """The B1 tool calls of this instance, as the pinned server answered them."""
    measured = {**MEASURED, **(measured or {})}
    measure_payload = {
        "volume": measured["volume"], "area": measured["area"],
        "topology": {"faces": 19, "edges": 57, "vertices": 38},
        "bbox": {"xsize": 90.0, "ysize": 54.0, "zsize": 12.0,
                 "center": {"x": 0.0, "y": 0.0, "z": 0.0}},
        "center_of_mass": {"x": 0.0, "y": -0.0, "z": 0.0},
        "inertia": {"Ixx": 59281.1245, "Iyy": 165344.5250, "Izz": measured["izz"],
                    "Ixy": -0.0, "Ixz": 0.0, "Iyz": -0.0},
        "inertia_units": "g·mm²", "mass_g": measured["mass_g"],
    }
    if export_text is None:
        export_text = (f"Exported to:\n{STEP_PATH}\n{STL_PATH}\n"
                       "volume 5.005e+04 mm³, bbox 90×54×12 mm, 19 faces")
    return [
        ("measure", {"object_name": CASE["part"], "material": CASE["material"]}, measure_payload),
        ("find_holes", {"object_name": CASE["part"]},
         {"count": holes, "holes": [{"diameter": 16.0, "depth": 12.0, "bottom": "through",
                                     "location": [0.0, 0.0, 6.0]}]}),
        ("find_hole_patterns", {"object_name": CASE["part"]},
         {"count": 1, "patterns": [{"holes": [{"diameter": pattern_d, "depth": 12.0,
                                               "location": [17.0, 0.0, 6.0]}]}]}),
        ("export", {"filename": CASE["export_stem"], "format": "step,stl",
                    "object_name": CASE["part"]}, export_text),
        ("import_cad_file", {"path": path, "name": "written_step"},
         {"volume": measured["volume"], "solids": 1, "faces": 19, "imported": "written_step",
          "format": "step", "path": path}),
    ]


def _stream(calls=None, extra=(), *, build=True, gate=True):
    """Claude Code stream-json: every tool answers with one text block, which is what the
    server's string-returning tools look like once FastMCP's wrapper is unwrapped."""
    calls = _b1_calls() if calls is None else calls
    events = [claude.init(SERVER, ["Bash", "Read", "Write", "WebFetch", "WebSearch",
                                   *(f"mcp__{SERVER}__{t}" for t in TOOLS)])]
    if build:
        events += claude.call("b0", f"mcp__{SERVER}__execute", {"code": "from build123d import *"},
                              f"Registered '{CASE['part']}': volume=5.005e+04 mm³, faces=19")
    if gate:
        events += claude.call("b1", f"mcp__{SERVER}__validate", {"object_name": CASE["part"]},
                              "Validity gate: PASS\n"
                              + json.dumps({"passes_gate": True, "n_solids": 1,
                                            "volume": MEASURED["volume"], "brep_valid": True}))
    for index, (name, args) in enumerate(extra):
        events += claude.call(f"x{index}", name, args, "ok")
    for index, (tool, args, payload) in enumerate(calls):
        shown = payload if isinstance(payload, str) else json.dumps({"result": json.dumps(payload)})
        events += claude.call(f"c{index}", f"mcp__{SERVER}__{tool}", args, shown)
    events += claude.call("cp", "Bash",
                          {"command": f"cp {STEP_PATH} part.step && cp {STL_PATH} part.stl"}, "")
    events.append(claude.result(len(calls) + 3))
    return jsonl(events)


def _codex_stream(calls=None):
    calls = _b1_calls() if calls is None else calls
    events = list(codex.START)
    for index, (tool, args, payload) in enumerate(calls):
        text = payload if isinstance(payload, str) else json.dumps(payload, indent=2)
        events += codex.mcp(f"m{index}", SERVER, tool, args, text)
    events += codex.shell("s0", f"cp {STEP_PATH} part.step && cp {STL_PATH} part.stl")
    events.append(codex.done())
    return jsonl(events)


def _run(tmp_path, stream=None, answer=None, *, harness="claude", files=None, extra=(), **kwargs):
    """One run result as ``asibench run`` writes it, through verify_one."""
    stream = _stream(extra=extra) if stream is None else stream
    return TASK.verify(tmp_path, stream, reference=REFERENCE,
                       answer=_answer() if answer is None else answer, harness=harness,
                       files=_files() if files is None else files, **kwargs)


# --------------------------------------------------------------------------
# Generator
# --------------------------------------------------------------------------

def test_cases_are_deterministic_and_varied():
    assert generate_gt.build_case(31415) == generate_gt.build_case(31415) == CASE
    cases = [generate_gt.build_case(seed) for seed in range(120)]
    assert {c["part"] for c in cases} == set(generate_gt.PARTS)
    assert {c["material"] for c in cases} == set(generate_gt.MATERIALS)
    assert len({(c["plate_x"], c["plate_y"], c["plate_z"]) for c in cases}) > 40
    for case in cases:
        bolt_radius, bolt = case["bolt_circle_d"] / 2, case["bolt_d"] / 2
        assert bolt_radius - bolt - case["bore_d"] / 2 >= generate_gt.BORE_CLEARANCE - 1e-9
        assert (min(case["plate_x"], case["plate_y"]) / 2 - bolt_radius - bolt
                >= generate_gt.EDGE_CLEARANCE - 1e-9)
        assert (case["bolt_circle_d"] * math.sin(math.pi / case["bolt_count"])
                >= case["bolt_d"] + generate_gt.BOLT_GAP - 1e-9)
        assert case["corner_radius"] <= min(case["plate_x"], case["plate_y"]) / 4
        assert case["export_stem"].startswith("/tmp/asibench-b123d-")


def test_reference_matches_the_recorded_closed_form(tmp_path):
    """The closed form is stdlib only, so this needs neither build123d nor OpenCascade."""
    generate_gt.generate(tmp_path, {"seed": 31415})
    ref = json.loads((tmp_path / "reference/reference.json").read_text())
    for key in ("volume_mm3", "surface_area_mm2", "mass_g", "izz_g_mm2",
                "reimported_volume_mm3"):
        assert ref[key] == pytest.approx(REFERENCE[key], abs=5e-5), key
    for key in ("hole_count", "bolt_pattern_count", "face_count", "n_solids", "bbox_mm",
                "export_files", "bolt_hole_diameter_mm", "passes_gate", "part"):
        assert ref[key] == REFERENCE[key], key
    assert ref["center_of_mass_mm"] == [0.0, 0.0, 0.0]
    assert json.loads((tmp_path / "data/plate.json").read_text()) == CASE
    assert len(ref["bolt_positions_mm"]) == CASE["bolt_count"]
    for level in ("b1", "b2", "b3", "b4"):
        text = (tmp_path / f"prompt_{level}.md").read_text()
        assert "{{" not in text
        assert (CASE["part"] in text) and ("90.0 × 54.0 × 12.0" in text) == (level != "b4")


def test_reference_scales_with_the_material_density():
    case = {**CASE, "material": "aluminum"}
    light = generate_gt.reference(case)
    heavy = generate_gt.reference(CASE)
    ratio = generate_gt.DENSITIES["titanium"] / generate_gt.DENSITIES["aluminum"]
    assert heavy["mass_g"] / light["mass_g"] == pytest.approx(ratio, rel=1e-12)
    assert heavy["izz_g_mm2"] / light["izz_g_mm2"] == pytest.approx(ratio, rel=1e-12)
    assert light["volume_mm3"] == heavy["volume_mm3"]
    # Ixx + Iyy = Izz for a prism only via the section moments, so check the ordering instead
    assert heavy["ixx_g_mm2"] < heavy["iyy_g_mm2"] < heavy["izz_g_mm2"]


def test_reference_carries_every_key_the_checks_use():
    spec = json.loads((TASK_DIR / "e2e_check.json").read_text())
    keys = {cs["result"]["reference_key"] for cs in spec["calls"] if "result" in cs}
    keys |= {ref for cs in spec["calls"] for ref in cs.get("inputs_from_reference", {}).values()}
    keys |= {answer["reference_key"] for answer in spec["answers"]}
    assert keys <= set(REFERENCE)
    assert spec["server_tools"] == TOOLS
    assert {cs["tool"] for cs in spec["calls"]} <= set(TOOLS)
    linked = {cs["name"]: cs["inputs_from_call"] for cs in spec["calls"] if cs.get("inputs_from_call")}
    assert linked["reimport"] == {"call": "write", "extract": "exported_files",
                                  "args": ["path"], "match": "member"}


def test_prompts_name_the_server_only_at_b1_b2_and_keep_the_rules():
    for level in ("b1", "b2", "b3", "b4"):
        text = (TASK_DIR / f"prompt_{level}.md").read_text()
        assert ("`build123d` MCP server" in text) == (level in ("b1", "b2"))
        assert ("`find_hole_patterns`" in text) == (level == "b1")
        assert "no `import build123d`" in text and "Do not modify `data/plate.json`" in text
        assert "full precision" in text and "copied unchanged" in text
        # the verifier requires import_cad_file, so every level must ask for the read-back
        assert "reimported_volume_mm3" in text
        assert ("read the file back" in text or "reading it back" in text
                or "importing it back" in text or "read the written STEP back" in text)
        assert "mcp__" not in text and "Claude" not in text and "Codex" not in text
    assert "```python" in (TASK_DIR / "prompt_b1.md").read_text()


def test_every_required_tool_is_asked_for_at_every_level():
    """A required call the prompt never asks for fails an honest run: the first AWS B4 run
    did everything right and still failed `tool_called` on import_cad_file."""
    spec = json.loads((TASK_DIR / "e2e_check.json").read_text())
    required = {cs["tool"] for cs in spec["calls"] if not cs.get("optional")}
    asked = {"measure": ("measure", "mass properties"), "find_holes": ("hole", "recognis"),
             "find_hole_patterns": ("bolt circle", "bolt-circle"), "export": ("export",),
             "import_cad_file": ("read the file back", "reading it back", "importing it back",
                                 "read the written STEP back")}
    assert required <= set(asked)
    for level in ("b1", "b2", "b3", "b4"):
        text = (TASK_DIR / f"prompt_{level}.md").read_text().lower()
        for tool in required:
            assert any(word.lower() in text for word in asked[tool]), (level, tool)


def test_task_meta_needs_no_cad_runtime():
    import yaml
    meta = yaml.safe_load((TASK_DIR / "task_meta.yaml").read_text())
    assert meta["status"] == "test" and meta["difficulty"]["requires_network"] is False
    assert meta["runtime"]["packages"] == []
    assert [f["name"] for f in meta["output"]["files"]] == ["result.json", "part.step", "part.stl"]


# --------------------------------------------------------------------------
# Readers in the scorer
# --------------------------------------------------------------------------

def test_stl_reader_measures_volume_bbox_and_shells():
    triangles = scorer.parse_binary_stl(_stl(_box_mesh(2.0, 2.0, 2.0)))
    assert scorer.mesh_volume(triangles) == pytest.approx(8.0, abs=1e-9)
    assert scorer.mesh_bbox(triangles) == pytest.approx([2.0, 2.0, 2.0])
    assert scorer.mesh_open_edges(triangles) == 0 and scorer.mesh_components(triangles) == 1
    assert scorer.mesh_open_edges(triangles[:-1]) == 3
    two = _box_mesh(2.0, 2.0, 2.0) + _box_mesh(2.0, 2.0, 2.0, offset=(10.0, 0.0, 0.0))
    assert scorer.mesh_components(scorer.parse_binary_stl(_stl(two))) == 2
    flipped = [(c, b, a) for a, b, c in triangles]
    assert scorer.mesh_volume(flipped) == pytest.approx(8.0, abs=1e-9)


def test_stl_reader_rejects_what_the_server_never_writes():
    good = _stl(_box_mesh())
    for data, match in ((b"\0" * 20, "too short"), (good[:-10], "declares"),
                        (b"\0" * 80 + struct.pack("<I", 0), "no triangles"),
                        (b"solid x\nfacet normal 0 0 1\n" + b" " * 200, "ASCII")):
        with pytest.raises(scorer._PredictionError, match=match):
            scorer.parse_binary_stl(data)
    nan = bytearray(good)
    nan[84 + 12:84 + 16] = struct.pack("<f", float("nan"))
    with pytest.raises(scorer._PredictionError, match="non-finite"):
        scorer.parse_binary_stl(bytes(nan))


def test_step_reader_checks_structure_and_label():
    assert scorer.step_problems(_step()) == []
    assert scorer.step_labels(_step()) == [CASE["part"]]
    assert scorer.step_labels(_step(label="COMPOUND")) == ["COMPOUND"]
    assert "no ISO-10303-21 header" in scorer.step_problems(_step(header=False))
    assert "no END-ISO-10303-21 terminator" in scorer.step_problems(_step(terminator=False))
    assert "no FILE_SCHEMA" in scorer.step_problems(_step(schema=False))
    assert "no solid B-rep entity" in scorer.step_problems(_step(body=False))
    assert any("CARTESIAN_POINT" in p for p in scorer.step_problems(_step(points=2)))


def test_credit_is_log_linear_between_the_bounds():
    assert scorer.credit(0.0, 1e-6, 1e-2) == 1.0
    assert scorer.credit(1e-6, 1e-6, 1e-2) == 1.0
    assert scorer.credit(1e-2, 1e-6, 1e-2) == 0.0
    assert scorer.credit(1e-4, 1e-6, 1e-2) == pytest.approx(0.5)
    assert scorer.credit(math.inf, 1e-6, 1e-2) == 0.0


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def test_a_genuine_submission_scores_full(tmp_path):
    pred, ref = _dirs(tmp_path, _answer())
    assert _gate(pred, ref) == 1.0
    assert TASK.total(pred, ref) == pytest.approx(100.0)


def test_a_genuine_run_passes_every_check(tmp_path):
    row = _run(tmp_path)
    assert row["verdict"] == "PASS", row["checks"]
    assert _status(row) == {name: "PASS" for name in verify.CHECK_ORDER}
    assert row["tool_call_counts"][f"mcp__{SERVER}__export"] == 1


def test_a_genuine_codex_run_passes_every_check(tmp_path):
    row = _run(tmp_path, _codex_stream(), harness="codex")
    assert row["verdict"] == "PASS", row["checks"]
    assert _status(row)["tool_chain"] == "PASS"


def test_submission_failures_score_zero(tmp_path):
    # no result.json at all: the hard gate zeroes the result, and every scorer that
    # reads result.json scores zero (the two artefact-only scorers still see the files)
    pred, ref = _dirs(tmp_path / "a", None)
    assert _gate(pred, ref) == 0.0
    assert TASK.total(pred, ref) == pytest.approx(40.0)
    # result.json is not JSON
    pred, ref = _dirs(tmp_path / "b", "not json")
    assert _gate(pred, ref) == 0.0
    assert TASK.total(pred, ref) == pytest.approx(40.0)
    # a missing export
    pred, ref = support.score_dirs(tmp_path / "c", _answer(), REFERENCE)
    assert _gate(pred, ref) == 0.0
    # the mesh is not one watertight solid
    open_mesh = _stl(_box_mesh()[:-1])
    pred, ref = _dirs(tmp_path / "d", _answer(), stl=open_mesh)
    assert _gate(pred, ref) == 0.0
    two_bodies = _stl(_box_mesh() + _box_mesh(offset=(200.0, 0.0, 0.0)))
    pred, ref = _dirs(tmp_path / "e", _answer(), stl=two_bodies)
    assert _gate(pred, ref) == 0.0
    # values recomputed by hand (1% off) lose the measurement credit but keep the rest
    wrong = _answer(volume_mm3=MEASURED["volume"] * 1.01, surface_area_mm2=MEASURED["area"] * 1.01,
                    mass_g=MEASURED["mass_g"] * 1.01, izz_g_mm2=MEASURED["izz"] * 1.01,
                    reimported_volume_mm3=MEASURED["volume"] * 1.01)
    pred, ref = _dirs(tmp_path / "f", wrong)
    assert _gate(pred, ref) == 1.0
    assert TASK.total(pred, ref) == pytest.approx(60.0)
    # a hole table read off the input file instead of the recogniser
    pred, ref = _dirs(tmp_path / "g", _answer(hole_count=8, bolt_hole_diameter_mm=6.0))
    assert TASK.total(pred, ref) == pytest.approx(100.0 - 2 * 20 / 5)
    # the STEP written without the object name carries no part label
    pred, ref = _dirs(tmp_path / "h", _answer(), step=_step(label="COMPOUND"))
    assert TASK.total(pred, ref) == pytest.approx(90.0)
    # a split part: the gate fails and the features lose n_solids
    pred, ref = _dirs(tmp_path / "i", _answer(n_solids=2, passes_gate=False), stl=two_bodies)
    assert _gate(pred, ref) == 0.0
    assert TASK.total(pred, ref) == pytest.approx(100.0 - 2 * 20 / 5 - 30.0)


def test_mesh_scorer_accepts_the_tessellation_band_only(tmp_path):
    item = next(i for i in TASK.eval_config()["scoring"] if i["scorer"] == "build123d_e2e_mesh")
    from ai4sci_bench.core.scorer import get_scorer
    config = {**item["config"], "weight": item["weight"]}

    def score(volume):
        pred, ref = _dirs(tmp_path / f"v{volume:.0f}", _answer(), stl=_plate_stl(volume))
        return get_scorer(item["scorer"]).score(pred, ref, config)

    # the real server's STL sits 3e-5 to 1.2e-4 above the exact solid (it depends on
    # whether validate() already tessellated the shape), so both are full credit
    assert score(MEASURED["volume"] * 1.00003).details["credit_fractions"]["volume"] == 1.0
    assert score(MEASURED["volume"] * 1.000115).details["credit_fractions"]["volume"] == 1.0
    # a 10% smaller mesh (a different part, or one without its holes) is zero
    assert score(MEASURED["volume"] * 0.90).details["credit_fractions"]["volume"] == 0.0


def test_evaluator_failures_are_flagged(tmp_path):
    from ai4sci_bench.core.scorer import get_scorer
    for item in TASK.eval_config()["scoring"]:
        pred, ref = _dirs(tmp_path / item["scorer"], _answer(), reference=None)
        detail = get_scorer(item["scorer"]).score(pred, ref, {**item["config"], "weight": item["weight"]})
        assert detail.details.get("scorer_internal_error") is True, item["scorer"]
        assert detail.details["failure_kind"] == "missing_evaluator_input"
        assert detail.score == 0.0
    # a reference without geometry is an evaluator failure too, not a zero score
    pred, ref = _dirs(tmp_path / "empty", _answer(), reference={**REFERENCE, "volume_mm3": 0.0})
    item = TASK.eval_config()["scoring"][0]
    detail = get_scorer(item["scorer"]).score(pred, ref, {**item["config"], "weight": item["weight"]})
    assert detail.details.get("scorer_internal_error") is True
    # the gate, in contrast, never reads the reference: a submission problem stays a zero
    pred, ref = _dirs(tmp_path / "gate", None, reference=None)
    assert _gate(pred, ref) == 0.0


# --------------------------------------------------------------------------
# Verifier scenarios
# --------------------------------------------------------------------------

def test_values_not_from_the_tools_fail_answer_from_tool(tmp_path):
    row = _run(tmp_path, answer=_answer(volume_mm3=MEASURED["volume"] + 2e-4))
    assert _status(row)["answer_from_tool"] == "FAIL"
    assert "volume_mm3" in row["checks"]["answer_from_tool"]["detail"]


def test_a_missing_export_call_fails_tool_called(tmp_path):
    calls = [call for call in _b1_calls() if call[0] != "export"]
    row = _run(tmp_path, _stream(calls))
    assert _status(row)["tool_called"] == "FAIL"
    assert row["failure"] == "tool_called"


def test_importing_a_file_the_server_never_wrote_fails_the_chain(tmp_path):
    calls = [(tool, args, payload) if tool != "import_cad_file"
             else (tool, {**args, "path": "/tmp/handmade.step"}, payload)
             for tool, args, payload in _b1_calls()]
    row = _run(tmp_path, _stream(calls))
    assert _status(row)["tool_chain"] == "FAIL"
    assert "path" in row["checks"]["tool_chain"]["detail"]


def test_exporting_somewhere_else_warns_about_the_inputs(tmp_path):
    calls = [(tool, args, payload) if tool != "export"
             else (tool, {**args, "filename": "/tmp/other-stem"}, payload)
             for tool, args, payload in _b1_calls()]
    row = _run(tmp_path, _stream(calls))
    assert _status(row)["tool_correct"] == "WARN"
    assert "inputs differ from reference" in row["checks"]["tool_correct"]["detail"]


def test_a_local_cad_install_is_a_bypass(tmp_path):
    for index, command in enumerate(("pip install build123d cadquery-ocp",
                                     "uv pip install --python 3.12 build123d",
                                     "/opt/mcp/build123d/.venv/bin/python -c 'print(1)'",
                                     "openscad -o part.stl part.scad")):
        row = _run(tmp_path / f"cmd{index}", extra=[("Bash", {"command": command})])
        assert _status(row)["no_bypass"] == "FAIL", command
    row = _run(tmp_path / "import", files={**_files(), "model.py": "from build123d import *\n"})
    assert _status(row)["no_bypass"] == "FAIL"


def test_a_cad_web_lookup_is_suspicious(tmp_path):
    row = _run(tmp_path, extra=[("WebSearch", {"query": "build123d fillet volume"})])
    assert _status(row)["no_bypass"] == "WARN"
    assert row["verdict"] == "PASS"


