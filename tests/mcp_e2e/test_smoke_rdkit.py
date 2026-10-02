"""e2e_smoke/servers/rdkit.py without RDKit: result decoding, comparison and three-state
classification helpers, and the checks that need no reference library, against stub servers."""
import json

import pytest

from . import support
from .support import StubClient, report_statuses, rpc_json, rpc_text, setup

rk = support.smoke_module("rdkit")


def rpc_structured(value, *, wrap=True, text=None):
    """A FastMCP tools/call response with structuredContent ({"result": value} for plain returns)."""
    structured = {"result": value} if wrap else value
    return {"result": {"content": [{"type": "text", "text": text if text is not None else json.dumps(value)}],
                       "structuredContent": structured, "isError": False}}


def run_check(check, responses, *args):
    report = rk.Report()
    call = rk.Caller(StubClient(responses), report)
    check(call, report, *args)
    return report, call


def test_value_of_prefers_structured_content_and_unwraps_result():
    # FastMCP returns a list as one text block per element; only structuredContent keeps the list
    crippen = rpc_structured([1.3101, 44.71], text="1.310144.71")["result"]
    assert rk.value_of(crippen) == [1.3101, 44.71]
    model = rpc_structured({"conf_id": 0, "mol": "abc"}, wrap=False)["result"]
    assert rk.value_of(model) == {"conf_id": 0, "mol": "abc"}
    assert rk.value_of(rpc_json({"a": 1})["result"]) == {"a": 1}
    assert rk.value_of(rpc_text("C9H8O4")["result"]) == "C9H8O4"
    assert rk.value_of(None) is None


def test_deviation():
    assert rk.deviation(46.069, 46.069) == 0.0
    assert rk.deviation(100.0 + 1e-10, 100.0) == pytest.approx(1e-12)
    assert rk.deviation(1e-13, 0.0) == pytest.approx(1e-13)             # absolute below 1
    assert rk.deviation([[1.0, 2.0], [3]], [[1.0, 2.0], [3]]) == 0.0
    assert rk.deviation([1.0], [1.0, 2.0]) == float("inf")
    assert rk.deviation("C9H8O4", "C9H8O4") == 0.0 and rk.deviation("C9H8O4", "C9H8O3") == float("inf")
    assert rk.deviation(True, 1) == float("inf") and rk.deviation(True, True) == 0.0
    assert rk.deviation(float("nan"), float("nan")) == 0.0
    assert rk.deviation(None, 1.0) == float("inf")


def test_classify_is_three_state():
    def correct(r):
        return "ok" if rk.value_of(r) == 1 else None

    def defect(r):
        return "known" if rk.is_error(r, "Tuple should have") else None

    assert rk.classify(rpc_structured(1)["result"], correct=correct, defect=defect) == ("PASS", "ok")
    known = rpc_text("validation error: Tuple should have at most 1 item", is_error=True)["result"]
    assert rk.classify(known, correct=correct, defect=defect) == ("WARN", "known")
    other = rpc_text("something else", is_error=True)["result"]
    assert rk.classify(other, correct=correct, defect=defect)[0] == "FAIL"
    assert rk.classify(None, correct=correct, defect=defect) == ("FAIL", "no result")
    assert not rk.is_error(rpc_text("Tuple should have")["result"], "Tuple")   # isError=false never matches


def test_inside(tmp_path):
    root = tmp_path / "files" / "out"
    assert rk.inside(root / "a.sdf", root)
    assert not rk.inside(root / ".." / "outside.sdf", root)
    assert not rk.inside("/etc/passwd", root)


def test_batch_outputs_unpacks_the_inner_call_tool_return():
    payload = {"results": [
        {"ok": True, "input": {"smiles": "C"}, "output": [[{"type": "text", "text": "16.043"}], {"result": 16.043}]},
        {"ok": False, "input": {"smiles": "x"}, "output": None, "error": "Invalid SMILES string"},
        {"ok": True, "input": None, "output": [[], {"conf_id": 0}]},
    ]}
    assert rk.batch_outputs(payload) == [(True, {"smiles": "C"}, 16.043, None),
                                         (False, {"smiles": "x"}, None, "Invalid SMILES string"),
                                         (True, None, {"conf_id": 0}, None)]


def test_fail_fast_outcome():
    bad = (False, {"smiles": "X"}, None, "Invalid SMILES string")
    good = (True, {"smiles": "C"}, 16.043, None)
    assert rk.fail_fast_outcome([bad], "X")[0] == "PASS"
    assert rk.fail_fast_outcome([good, bad], "X")[0] == "WARN"
    assert rk.fail_fast_outcome([bad, good], "X")[0] == "FAIL"         # did not stop
    assert rk.fail_fast_outcome([good], "X")[0] == "FAIL"              # error lost
    assert rk.fail_fast_outcome([], "X")[0] == "FAIL"


def test_args_kwargs_outcome():
    required = ["args", "kwargs"]
    known = rpc_text("2 validation errors for CalcPBFArguments\nargs\n  Field required [type=missing]",
                     is_error=True)["result"]
    assert rk.args_kwargs_outcome("CalcPBF", known, required, "PBF", 0.1)[0] == "WARN"
    status, detail = rk.args_kwargs_outcome("CalcFractionCSP3", known, required,
                                            "Calculate the number of heavy atoms", 0.1)
    assert status == "WARN" and "CalcNumHeavyAtoms' docstring" in detail
    fixed = rpc_structured(1 / 9)["result"]
    assert rk.args_kwargs_outcome("CalcFractionCSP3", fixed, ["smiles"], "", 1 / 9)[0] == "PASS"
    assert rk.args_kwargs_outcome("CalcFractionCSP3", rpc_structured(0.5)["result"], ["smiles"], "", 1 / 9)[0] == "FAIL"
    assert rk.args_kwargs_outcome("GetUSR", rpc_structured([1.0])["result"], ["smiles"], "", 0.1)[0] == "FAIL"


def test_property_outcome():
    assert rk.property_outcome({"label": "aspirin"}, "label", "aspirin", True)[0] == "PASS"
    assert rk.property_outcome({"id": 7}, "id", 7, True)[0] == "PASS"
    assert rk.property_outcome({"flag": 1}, "flag", True, True)[0] == "FAIL"        # wrong type
    status, detail = rk.property_outcome({}, "cache", 1, True, "; renamed")
    assert status == "WARN" and detail.endswith("; renamed")
    assert rk.property_outcome({}, "label", "x", False)[0] == "FAIL"                # molecule changed too
    assert rk.property_outcome({"other": 1}, "label", "x", True)[0] == "FAIL"


def test_default_image_names():
    assert rk.default_image_names(["mol_20261002_121737.png"] * 2)[0] == "WARN"
    assert rk.default_image_names(["mol_20261002_121737.png", "mol_20261002_121738.png"])[0] == "WARN"
    assert rk.default_image_names(["mol_1.png", "mol_2.png"])[0] == "PASS"


def test_schema_defaults_reads_pydantic_defs():
    tool = {"inputSchema": {"$defs": {"EmbedParameters": {"properties": {
        "randomSeed": {"default": -1, "type": "integer"}, "coordMap": {"type": "object"},
        "onlyHeavyAtomsForRMS": {"default": False}}}}}}
    assert rk.schema_defaults(tool, "EmbedParameters") == {"randomSeed": -1, "onlyHeavyAtomsForRMS": False}
    assert rk.schema_defaults({}, "EmbedParameters") == {}


def test_tool_tables_cover_the_manifest_without_overlap():
    tools = set(setup.load_manifest()["rdkit"]["expected_tools"])
    groups = [set(rk.DESCRIPTORS_MODULE), set(rk.RDMOLDESCRIPTORS), set(rk.OPTION_TOOLS), set(rk.ARGS_KWARGS_TOOLS),
              {t for t, _, _ in rk.PROPERTY_TOOLS}]
    assert all(g <= tools for g in groups)
    assert sum(map(len, groups)) == len(set().union(*groups))
    assert {t for t, *_ in rk.ANCHORS} <= tools
    assert len(rk.DESCRIPTOR_NAMES) == len(set(rk.DESCRIPTOR_NAMES)) == 36


def test_anchors_are_hand_values():
    anchors = {(tool, smiles): expected for tool, smiles, expected, _, _ in rk.ANCHORS}
    assert anchors[("MolWt", "CCO")] == pytest.approx(46.069, abs=1e-12)
    assert anchors[("ExactMolWt", "CCO")] == pytest.approx(46.041864813, abs=1e-8)
    assert anchors[("CalcTPSA", rk.ASPIRIN)] == pytest.approx(63.60, abs=1e-12)
    assert anchors[("NumValenceElectrons", rk.MOLECULES["caffeine"])] == 74


def test_check_anchors_compares_with_tolerance_or_exactly():
    values = {(t, s): e for t, s, e, _, _ in rk.ANCHORS}

    def answer(args, tool):
        return rpc_structured(values[(tool, args["smiles"])])

    responses = {t: (lambda args, t=t: answer(args, t)) for t, *_ in rk.ANCHORS}
    report, _ = run_check(rk.check_anchors, responses)
    assert set(report_statuses(report).values()) == {"PASS"}
    responses["CalcMolFormula"] = rpc_structured("C9H8O3")
    responses["MolWt"] = rpc_structured(46.07)
    statuses = report_statuses(run_check(rk.check_anchors, responses)[0])
    assert statuses["MolWt(CCO) = 46.069 (C2H6O from conventional atomic weights)"] == "FAIL"
    assert [s for n, s in statuses.items() if n.startswith("CalcMolFormula")] == ["FAIL"] * 3


class FakeRef:
    """Answers descriptor() from a dict; enough for check_descriptors."""

    def __init__(self, table):
        self.table = table

    def descriptor(self, tool, smiles, **options):
        return self.table(tool, smiles, options)


def test_check_descriptors_passes_matching_values_and_fails_a_wrong_option():
    def table(tool, smiles, options):
        return len(smiles) + (0.5 if options.get("includeHs") is False or options.get("strict") is False else 0.0)

    def server(tool, *, ignore_option=False):
        def answer(args):
            options = {} if ignore_option else {k: v for k, v in args.items() if k != "smiles"}
            return rpc_structured(table(tool, args["smiles"], options))
        return answer

    tools = list(rk.DESCRIPTORS_MODULE + rk.RDMOLDESCRIPTORS) + list(rk.OPTION_TOOLS)
    report, call = run_check(rk.check_descriptors, {t: server(t) for t in tools}, FakeRef(table))
    assert set(report_statuses(report).values()) == {"PASS"}
    assert set(call.stdout_by_tool) == set(tools)
    responses = {t: server(t, ignore_option=t == "CalcLabuteASA") for t in tools}
    statuses = report_statuses(run_check(rk.check_descriptors, responses, FakeRef(table))[0])
    assert statuses["CalcLabuteASA vs RDKit (default and includeHs=True/False)"] == "FAIL"
    assert statuses["CalcTPSA vs RDKit (6 molecules)"] == "PASS"


def test_check_oxidation_numbers():
    known = rpc_text("1 validation error for CalcOxidationNumbersOutput\nresult\n  Input should be a valid number",
                     is_error=True)
    assert set(report_statuses(run_check(rk.check_oxidation_numbers, {"CalcOxidationNumbers": known})[0]).values()) \
        == {"WARN"}
    fixed = rpc_structured([-3, -1, -2])
    assert set(report_statuses(run_check(rk.check_oxidation_numbers, {"CalcOxidationNumbers": fixed})[0]).values()) \
        == {"PASS"}
    wrong = rpc_structured(0.0)
    assert set(report_statuses(run_check(rk.check_oxidation_numbers, {"CalcOxidationNumbers": wrong})[0]).values()) \
        == {"FAIL"}


def test_check_invalid_inputs_warns_on_ignored_arguments():
    def answer(args):
        if args.get("smiles", "CCO") == rk.INVALID_SMILES or "smiles" not in args and "smiles1" not in args \
                and "smarts" not in args:
            return rpc_text("Invalid SMILES string", is_error=True)
        if args.get("smarts") == "[C" or args.get("pattern") == "[C" or args.get("smiles2") == rk.INVALID_SMILES:
            return rpc_text("invalid", is_error=True)
        return rpc_structured(46.069)

    tools = ("MolWt", "CalcMolFormula", "CalcTPSA", "smiles_to_mol", "MurckoScaffoldSmilesFromSmiles",
             "MakeScaffoldGeneric", "FragmentMol", "TanimotoSimilarity", "smarts_to_mol")
    statuses = report_statuses(run_check(rk.check_invalid_inputs, {t: answer for t in tools})[0])
    assert statuses.pop("MolWt[unknown extra argument]") == "WARN"
    assert set(statuses.values()) == {"PASS"}


def test_check_coverage():
    report = rk.Report()
    call = rk.Caller(StubClient({"a": rpc_structured(1), "b": rpc_text("x", is_error=True)}), report)
    call("a", "a", {})
    call("b", "b", {}, allow_error=True)
    rk.check_coverage(call, report, ["a", "b"])
    rk.check_coverage(call, report, ["a", "b", "c"])
    assert [c["status"] for c in report.checks] == ["PASS", "FAIL"]
    assert "['c']" in report.checks[-1]["detail"]


def test_smoke_declaration():
    assert rk.SMOKE.server == "rdkit" and "rdkit" in rk.SMOKE.packages
    assert rk.SMOKE.expected_cwd_files == ()         # every file tool gets an explicit file_dir
