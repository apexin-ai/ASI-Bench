"""e2e_smoke/servers/atomictoolkit.py without ASE: payload/in-band split, artifact
lists, the closed-form elastic fit, the g(r) normalisation, and the three-state
defect checks (D1-D9) against stubbed servers and stubbed references."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from . import support
from .support import StubClient, rpc_json, rpc_text

atk = support.smoke_module("atomictoolkit")


def make_ctx(tmp_path, responses, ref=None, files=None):
    report = atk.Report()
    call = atk.Caller(StubClient(responses), report)
    cwd = tmp_path / "cwd"
    cwd.mkdir(exist_ok=True)
    session = SimpleNamespace(call=call, report=report, tmp=tmp_path, cwd=cwd, state={})
    ctx = atk.Ctx(session, ref or SimpleNamespace())
    ctx.files.update(files or {})
    return ctx


def statuses(ctx):
    return support.report_statuses(ctx.report)


def artifact(label, path, aid="art_" + "a" * 32):
    return {"label": label, "id": aid, "filepath": str(path), "download_url": f"/artifacts/{aid}/{path.name}"}


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------

def test_payload_prefers_structured_content_and_in_band_errors_are_split():
    result = {"content": [{"type": "text", "text": '{"a": 2}'}], "structuredContent": {"a": 1}}
    assert atk.payload(result) == {"a": 1}
    assert atk.payload(rpc_json({"a": 2})["result"]) == {"a": 2}
    error = rpc_json({"status": "error", "error": {"type": "ValueError", "message": "boom"}})["result"]
    assert atk.in_band_error(error) == "ValueError: boom"
    assert atk.in_band_error(rpc_json({"status": "success"})["result"]) is None
    assert atk.in_band_error(rpc_text("not json")["result"]) is None
    with pytest.raises(ValueError):
        atk.payload(rpc_json([1, 2])["result"])


def test_artifact_problems_accepts_the_server_shape_and_names_each_deviation(tmp_path):
    cu = tmp_path / "cu.extxyz"
    cu.write_text("x")
    preview = tmp_path / "cu.extxyz.preview.html"
    preview.write_text("x")
    csv = tmp_path / "rdf.csv"
    csv.write_text("x")
    good = {"artifacts": [artifact("filepath", cu, "art_" + "1" * 32),
                          artifact("filepath_preview_html", preview, "art_" + "2" * 32),
                          artifact("rdf_csv", csv, "art_" + "3" * 32)]}
    assert atk.artifact_problems(good, {"filepath": cu, "rdf_csv": csv}) == []
    # A table gets no preview; a missing preview, a reused id and a bad URL are each reported.
    assert atk.artifact_problems({"artifacts": good["artifacts"][:1] + good["artifacts"][2:]},
                                 {"filepath": cu, "rdf_csv": csv})
    reused = {"artifacts": [dict(a, id="art_" + "1" * 32) for a in good["artifacts"]]}
    assert any("unique" in p for p in atk.artifact_problems(reused, {"filepath": cu, "rdf_csv": csv}))
    bad_url = {"artifacts": [dict(good["artifacts"][0], download_url="http://x/cu.extxyz")] + good["artifacts"][1:]}
    assert any("download_url" in p for p in atk.artifact_problems(bad_url, {"filepath": cu, "rdf_csv": csv}))
    assert any("id" in p for p in atk.artifact_problems(
        {"artifacts": [dict(good["artifacts"][0], id="art_xyz")] + good["artifacts"][1:]},
        {"filepath": cu, "rdf_csv": csv}))
    assert atk.artifact_problems({}, {}) == []
    assert atk.artifact_problems(good, {}) != []


@pytest.mark.parametrize("number,name", [(1, "triclinic"), (15, "monoclinic"), (62, "orthorhombic"),
                                         (139, "tetragonal"), (166, "trigonal"), (194, "hexagonal"),
                                         (225, "cubic"), (230, "cubic")])
def test_crystal_system(number, name):
    assert atk.crystal_system(number) == name


def test_quadratic_curvature_matches_polyfit_on_a_symmetric_grid():
    strains = atk.elastic_strains(0.02)
    energies = [3.0 * s * s - 0.7 * s + 0.1 + 5.0 * s ** 4 for s in strains]
    assert atk.quadratic_curvature(strains, energies) == pytest.approx(np.polyfit(strains, energies, 2)[0],
                                                                        rel=1e-12)
    assert atk.quadratic_curvature(strains, [2.0 * s * s for s in strains]) == pytest.approx(2.0, rel=1e-12)
    with pytest.raises(ValueError):
        atk.quadratic_curvature([0.0, 0.01, 0.02], [0.0, 1.0, 4.0])


def test_rdf_reference_is_one_for_an_ideal_gas_and_counts_each_ordered_pair():
    rng = np.random.default_rng(7)
    n, box, r_max = 400, 20.0, 6.0
    pos = rng.random((n, 3)) * box
    delta = pos[:, None, :] - pos[None, :, :]
    delta -= box * np.round(delta / box)
    d = np.linalg.norm(delta, axis=-1)[~np.eye(n, dtype=bool)]
    r, g = atk.rdf_reference(d[d < r_max], n, box ** 3, r_max, 30)
    assert float(np.mean(g[r > 2.0])) == pytest.approx(1.0, abs=0.05)
    # one pair at 1.0 in both directions -> one bin
    r, g = atk.rdf_reference([1.01, 1.01], 2, 1000.0, 2.0, 2)
    assert g[0] == 0 and g[1] == pytest.approx(2 / (4 * np.pi * r[1] ** 2 * 1.0 * (2 / 1000.0) * 2))


def test_error_helpers():
    assert atk.relative_error([1.0, 200.0], [1.0, 200.0 * (1 + 1e-10)]) <= 1.01e-10
    assert atk.relative_error([1.0], [1.0, 2.0]) == float("inf")
    assert atk.abs_error(None, [1.0]) == float("inf")
    assert atk.fallback_prefixes(["kim: x", "orb: No module", "nequix: y"]) == ["kim", "orb", "nequix"]
    assert atk.fallback_prefixes(None) == []


def test_classify_three_states():
    yes, no = (lambda r: "ok"), (lambda r: None)
    assert atk.classify({}, correct=yes, defect=yes)[0] == "PASS"
    assert atk.classify({}, correct=no, defect=yes)[0] == "WARN"
    assert atk.classify({}, correct=no, defect=no)[0] == "FAIL"
    assert atk.classify(None, correct=yes, defect=yes)[0] == "FAIL"


def test_task_result_and_task_stdout_classification():
    no_ctx = {"error": {"code": 0, "message": atk.NO_CONTEXT}}
    assert atk.classify_task_result(no_ctx, no_ctx)[0] == "WARN"
    assert atk.classify_task_result(no_ctx, {"result": {"content": []}})[0] == "FAIL"
    assert atk.classify_task_stdout([]) == ("PASS", "no non-JSON stdout on the task path")
    bfgs = ["      Step     Time          Energy          fmax", "BFGS:    0 09:29:18       -0.022726        0.000000"]
    status, detail = atk.classify_task_stdout(bfgs)
    assert status == "WARN" and detail.startswith("D8")
    status, detail = atk.classify_task_stdout(bfgs + ["something else"])
    assert status == "WARN" and not detail.startswith("D8")


def test_task_call_requires_the_created_task_shape():
    class Client:
        def __init__(self, response):
            self.response, self.sent = response, None

        def request(self, method, params, timeout=120.0):
            self.sent = (method, params)
            return self.response

    accepted = {"result": {"content": [], "_meta": {atk.TASK_META: {"taskId": "t-1", "status": "working"}}}}
    client = Client(accepted)
    assert atk.task_call(client, "run_md_workflow", {"steps": 3}) == ("t-1", "accepted")
    assert client.sent == ("tools/call", {"name": "run_md_workflow", "arguments": {"steps": 3},
                                          "task": {"ttl": 60000}})
    refused = {"result": {"content": [{"type": "text", "text": "requires task-augmented execution"}],
                          "isError": True}}
    assert atk.task_call(Client(refused), "run_md_workflow", {})[0] is None


# --------------------------------------------------------------------------
# Three-state checks against stubbed servers
# --------------------------------------------------------------------------

def test_deprecated_wrappers_warn_only_on_the_exact_functiontool_error(tmp_path):
    responses = {tool: rpc_text(f"Error calling tool '{tool}': {atk.FUNCTION_TOOL}", is_error=True)
                 for tool in atk.DEPRECATED}
    responses["optimize_with_mlip"] = rpc_text("Error calling tool 'optimize_with_mlip': boom", is_error=True)
    ref = SimpleNamespace(bulk=lambda *a, **k: SimpleNamespace(positions=np.zeros((4, 3)), cell=SimpleNamespace(
        array=np.eye(3)), get_chemical_symbols=lambda: ["Cu"] * 4))
    ctx = make_ctx(tmp_path, responses, ref=ref)
    atk.check_deprecated(ctx)
    assert statuses(ctx) == {"build_structure[deprecated wrapper]": "WARN",
                             "read_structure_file[deprecated wrapper]": "WARN",
                             "write_structure_file[deprecated wrapper]": "WARN",
                             "optimize_with_mlip[deprecated wrapper]": "FAIL"}


def test_task_required_tools_warn_on_the_exact_refusal_and_check_tools_list(tmp_path):
    responses = {tool: rpc_text(f"Tool '{tool}' requires task-augmented execution", is_error=True)
                 for tool in atk.TASK_REQUIRED}
    responses["run_md_workflow"] = rpc_json({"status": "success"})          # behaviour changed
    tools = {name: {"execution": {"taskSupport": "required" if name in atk.TASK_REQUIRED else "optional"}}
             for name in (*atk.TASK_REQUIRED, *atk.DEPRECATED, "single_point_workflow")}
    ctx = make_ctx(tmp_path, responses)
    ctx.session.state["trajectory_file"] = tmp_path / "traj.extxyz"
    atk.check_task_required(ctx, tools)
    got = statuses(ctx)
    assert got.pop("tools/list execution.taskSupport") == "PASS"
    assert got.pop("run_md_workflow[plain tools/call]") == "FAIL"
    assert set(got.values()) == {"WARN"} and len(got) == 4
    tools["single_point_workflow"]["execution"]["taskSupport"] = "required"
    ctx = make_ctx(tmp_path, responses)
    ctx.session.state["trajectory_file"] = tmp_path / "traj.extxyz"
    atk.check_task_required(ctx, tools)
    assert statuses(ctx)["tools/list execution.taskSupport"] == "FAIL"


SI_MESSAGE = ("Failed to initialize any MLIP calculator (attempted: emt, kim, orb, nequix). Details: emt: EMT does "
              "not support species ['Si']. Supported elements: [...]")


@pytest.mark.parametrize("response,expected", [
    (rpc_text(f"Error calling tool 'single_point_workflow': {atk.NAME_TOO_LONG}: \"{SI_MESSAGE}\"", True), "WARN"),
    (rpc_json({"status": "error", "error": {"type": "RuntimeError", "message": SI_MESSAGE}}), "PASS"),
    (rpc_text(f"Error calling tool 'single_point_workflow': {SI_MESSAGE}", True), "PASS"),
    (rpc_json({"energy": -1.0, "calculator_used": "kim"}), "FAIL"),
    (rpc_text("Error calling tool 'single_point_workflow': something else", True), "FAIL"),
])
def test_unsupported_element_d3(tmp_path, response, expected):
    ctx = make_ctx(tmp_path, {"single_point_workflow": response}, files={"si.cif": tmp_path / "si.cif"})
    atk.check_unsupported_element(ctx)
    assert statuses(ctx) == {"single_point_workflow[Si with calculator_name=emt]": expected}


def fake_point_ref(energy):
    return SimpleNamespace(read=lambda path: None, point=lambda atoms: {"energy": energy, "forces": [], "stress": None})


@pytest.mark.parametrize("data,expected", [
    ({"energy": -0.5, "calculator_used": "emt", "calculator_fallbacks": []}, "PASS"),
    ({"energy": -0.5, "calculator_used": "emt", "calculator_fallbacks": ["kim: a", "orb: b", "nequix: c"]}, "WARN"),
    ({"energy": -0.4, "calculator_used": "emt", "calculator_fallbacks": ["kim: a", "orb: b", "nequix: c"]}, "FAIL"),
    ({"energy": -0.5, "calculator_used": "orb", "calculator_fallbacks": ["kim: a"]}, "FAIL"),
])
def test_auto_calculator_d6(tmp_path, data, expected):
    ctx = make_ctx(tmp_path, {"single_point_workflow": rpc_json(data)}, ref=fake_point_ref(-0.5))
    atk.check_auto_calculator(ctx, tmp_path / "cu32.extxyz")
    assert statuses(ctx) == {"single_point_workflow[calculator_name=auto]": expected}


def coordination_ctx(tmp_path, per_atom, average):
    csv = tmp_path / "coordination.csv"
    csv.write_text("atom_index,symbol,coordination\n" + "".join(f"{i},Cu,{c}\n" for i, c in enumerate(per_atom)))
    ref = SimpleNamespace(coordination=lambda atoms, skin: [12, 12] if skin == 0.0 else [18, 18])
    ctx = make_ctx(tmp_path, {}, ref=ref)
    summary = {"coordination": {"average": average, "by_element": {"Cu": average}}}
    atk.check_coordination(ctx, None, summary, csv)
    return statuses(ctx)["analyze_structure_workflow[coordination numbers]"]


def test_coordination_d4(tmp_path):
    assert coordination_ctx(tmp_path, [12, 12], 12.0) == "PASS"
    assert coordination_ctx(tmp_path, [18, 18], 18.0) == "WARN"
    assert coordination_ctx(tmp_path, [18, 18], 12.0) == "FAIL"       # summary and csv disagree
    assert coordination_ctx(tmp_path, [14, 14], 14.0) == "FAIL"


def rdf_status(tmp_path, scale, header="r,g_r"):
    r = np.array([0.5, 1.5, 2.5])
    g = np.array([0.0, 1.2, 0.9])
    csv = tmp_path / "rdf.csv"
    csv.write_text(header + "\n" + "".join(f"{float(a)!r},{float(b)!r}\n" for a, b in zip(r, scale * g)))
    ctx = make_ctx(tmp_path, {}, ref=SimpleNamespace(rdf=lambda atoms, r_max, bins: (r, g)))
    atk.check_rdf(ctx, None, csv, 3.0, 3)
    return statuses(ctx)["analyze_structure_workflow[g(r)]"]


def test_rdf_d9(tmp_path):
    assert rdf_status(tmp_path, 1.0) == "PASS"
    assert rdf_status(tmp_path, 2.0) == "WARN"
    assert rdf_status(tmp_path, 1.5) == "FAIL"
    assert rdf_status(tmp_path, 1.0, header="radius,g") == "FAIL"


@pytest.mark.parametrize("response,expected", [
    (rpc_json({"status": "error", "error": {"type": "ValueError", "message": atk.ZERO_LATTICE}}), "WARN"),
    (rpc_json({"info": {"num_atoms": 3}, "analysis": {}}), "PASS"),
    (rpc_json({"status": "error", "error": {"type": "ValueError", "message": "other"}}), "FAIL"),
    (rpc_text("boom", True), "FAIL"),
])
def test_molecule_analysis_d5(tmp_path, response, expected):
    ctx = make_ctx(tmp_path, {"analyze_structure_workflow": response}, files={"h2o.extxyz": tmp_path / "h2o.extxyz"})
    atk.check_molecule_analysis(ctx)
    assert statuses(ctx) == {"analyze_structure_workflow[cell-less molecule H2O]": expected}


def test_kim_availability_d6(tmp_path):
    def run(kim, really):
        caps = {"default_calculator": "auto", "auto_order": atk.AUTO_ORDER, "integrators": atk.INTEGRATORS,
                "structure_types": atk.STRUCTURE_TYPES, "manipulate_operations": atk.OPERATIONS,
                "calculators": {"kim": kim, "orb": {"available": False, "error": "No module named 'orb_models'"},
                                "nequix": {"available": False, "error": "No module named 'nequix'"},
                                "emt": {"available": True, "supported_elements": atk.EMT_ELEMENTS}}}
        ref = SimpleNamespace(kim_available=lambda: (really, "" if really else "ModuleNotFoundError: kimpy"))
        ctx = make_ctx(tmp_path, {"list_workspace_capabilities_workflow": rpc_json(caps)}, ref=ref)
        atk.check_capabilities(ctx)
        return statuses(ctx)

    got = run({"available": True}, False)
    assert list(got.values()) == ["PASS", "WARN"]
    assert list(run({"available": False, "error": "x"}, False).values()) == ["PASS", "PASS"]
    assert list(run({"available": False}, True).values()) == ["PASS", "FAIL"]


def test_download_artifact_probes(tmp_path):
    cu = tmp_path / "cu.extxyz"
    cu.write_text("x")
    preview = tmp_path / "cu.extxyz.preview.html"
    preview.write_text("x")

    def respond(arguments):
        path = arguments["filepath"]
        if path == str(cu):
            return rpc_json({"filepath": path, "artifacts": [artifact("filepath", cu, "art_" + "1" * 32),
                                                             artifact("filepath_preview_html", preview,
                                                                      "art_" + "2" * 32)]})
        return rpc_json({"filepath": path})

    ctx = make_ctx(tmp_path, {"create_download_artifact": respond}, files={"cu.extxyz": cu})
    ctx.work = tmp_path
    atk.check_download_artifact(ctx)
    assert statuses(ctx) == {"create_download_artifact[existing structure file]": "PASS",
                             "create_download_artifact[download_url over stdio]": "WARN",
                             "create_download_artifact[missing file]": "WARN",
                             "create_download_artifact[suffix not allowed (.md)]": "WARN"}


def test_smoke_declaration_matches_the_manifest():
    entry = support.setup.load_manifest()["atomictoolkit"]
    assert atk.SMOKE.server == "atomictoolkit" and atk.SMOKE.prepare and atk.SMOKE.after
    assert entry["catalog_id"] == "ase" and "checkout" not in entry
    assert set(atk.DEPRECATED) | set(atk.TASK_REQUIRED) < set(entry["expected_tools"])
    assert len(entry["expected_tools"]) == 18
    assert "tool_errors" in atk.SMOKE.expected_cwd_files
    pins = dict(r.split("==") for r in entry["requirements"])
    assert pins["ase"] == "3.29.0" and not any(r.startswith("-e") for r in entry["requirements"])
    assert entry["launch"]["env"]["PYTHONPATH"] == "{checkout}/src"
    json.dumps(entry)
