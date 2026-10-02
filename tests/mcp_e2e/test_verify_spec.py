"""Strict e2e_check.json parsing (e2e_verify/spec.py): unknown, inapplicable and invalid keys,
dangling references, schema-1 normalisation, requirement grouping."""
import json
import re

import pytest

from . import support
from .support import E2E_TASKS

verify = support.verify


def _valid_spec():
    """A minimal valid schema-2 spec dict for mutation tests."""
    return {
        "schema_version": 2, "server": "demo",
        "reference_file": "reference.json", "prediction_file": "result.json",
        "calls": [{"name": "c", "tool": "do_thing",
                   "inputs_from_reference": {"x": "x_ref"},
                   "result": {"format": "number", "reference_key": "y_ref", "abs_tol": 1e-6}}],
        "answers": [{"prediction_key": "y", "from_call": "c",
                     "result_key": None, "reference_key": "y_ref", "abs_tol": 1e-6}],
        "bypass_patterns": [r"import\s+demo"],
    }


def test_valid_spec_fixture_parses():
    assert verify.parse_spec(_valid_spec()).server == "demo"


@pytest.mark.parametrize("mutate,match", [
    (lambda s: s.update(extra_top=1), "top-level"),
    (lambda s: s["calls"][0].update(toool="x"), r"calls\[0\]"),
    (lambda s: s["calls"][0]["result"].update(abs_toll=1e-6), "result"),
    (lambda s: s["answers"][0].update(prediciton_key="y"), r"answers\[0\]"),
])
def test_unknown_spec_keys_are_rejected(mutate, match):
    spec = _valid_spec()
    mutate(spec)
    with pytest.raises(verify.SpecError, match=match):
        verify.parse_spec(spec)


def test_unknown_keys_in_chain_source_and_select_are_rejected():
    chain = _valid_spec()
    chain["calls"].append({"name": "d", "tool": "t2",
                           "inputs_from_call": {"call": "c", "map": {"a": "b"}, "typo": 1},
                           "result": {"format": "number", "reference_key": "z", "abs_tol": 0}})
    with pytest.raises(verify.SpecError, match="inputs_from_call"):
        verify.parse_spec(chain)

    src = _valid_spec()
    src["answers"][0] = {"prediction_key": "y", "reference_key": "y_ref", "abs_tol": 0,
                         "from_calls": [{"call": "c", "result_key": "r", "selct": {}}]}
    with pytest.raises(verify.SpecError, match="from_calls"):
        verify.parse_spec(src)

    sel = _valid_spec()
    sel["answers"][0] = {"prediction_key": "y", "reference_key": "y_ref", "abs_tol": 0,
                         "from_calls": [{"call": "c", "select": {"reduce": "max", "bogus": 1}}]}
    with pytest.raises(verify.SpecError, match="select"):
        verify.parse_spec(sel)


@pytest.mark.parametrize("mutate,match", [
    (lambda s: s["calls"][0]["result"].update(format="csv"), "format"),
    (lambda s: s["calls"][0]["result"].update(match="nope"), "match"),
    (lambda s: s["calls"][0]["result"].update(format="json", key=None, extract="unknown_x"), "extract"),
    (lambda s: s["calls"][0]["result"].update(format="json", key=None, extract=None), "requires"),
])
def test_invalid_result_enums_and_requirements(mutate, match):
    spec = _valid_spec()
    mutate(spec)
    with pytest.raises(verify.SpecError, match=match):
        verify.parse_spec(spec)


def test_invalid_chain_compare_enum_is_rejected():
    spec = _valid_spec()
    spec["calls"].append({"name": "d", "tool": "t2",
                          "inputs_from_call": {"call": "c", "map": {"a": "b"}, "compare": "fuzzy"},
                          "result": {"format": "number", "reference_key": "z", "abs_tol": 0}})
    with pytest.raises(verify.SpecError, match="compare"):
        verify.parse_spec(spec)


@pytest.mark.parametrize("mutate,match", [
    (lambda s: s["answers"][0].update(from_call="ghost"), "from_call"),
    (lambda s: s["calls"].append({"name": "d", "tool": "t", "inputs_from_call": {"call": "ghost", "map": {"a": "b"}},
                                  "result": {"format": "number", "reference_key": "z", "abs_tol": 0}}),
     "inputs_from_call"),
    (lambda s: (s["calls"].append({"name": "opt", "tool": "o", "optional": True,
                                   "result": {"format": "number", "reference_key": "z", "abs_tol": 0}}),
                s["calls"].append({"name": "d", "tool": "t", "inputs_from_call": {"call": "opt", "map": {"a": "b"}},
                                   "result": {"format": "number", "reference_key": "z", "abs_tol": 0}})),
     "optional"),
])
def test_dangling_and_optional_cross_references_are_rejected(mutate, match):
    spec = _valid_spec()
    mutate(spec)
    with pytest.raises(verify.SpecError, match=match):
        verify.parse_spec(spec)


def test_invalid_bypass_regex_is_rejected():
    spec = _valid_spec()
    spec["bypass_patterns"] = ["[invalid"]
    with pytest.raises(verify.SpecError, match="regex"):
        verify.parse_spec(spec)
    spec = _valid_spec()
    spec["suspicious_patterns"] = ["(unclosed"]
    with pytest.raises(verify.SpecError, match="regex"):
        verify.parse_spec(spec)


def test_schema_1_spec_parses_through_normalisation():
    raw = json.loads((E2E_TASKS / "mcp_e2e/pyscf_rhf_energy/e2e_check.json").read_text())
    assert raw["schema_version"] == 1
    spec = verify.parse_spec(raw)
    assert [c.tool for c in spec.calls] == ["pyscf_rhf_energy"]
    assert [src.call for src in spec.answers[0].sources] == ["pyscf_rhf_energy"]


@pytest.mark.parametrize("mutate,match", [
    (lambda s: s["calls"][0]["result"].update(key="y"), "do not apply to format 'number'"),
    (lambda s: s["calls"][0]["result"].update(format="image"), "do not apply to format 'image'"),
    (lambda s: s["calls"][0]["result"].update(format="json", key="v", select={"reduce": "max", "argmax_of": "w"}),
     "exactly one of"),
    (lambda s: s["calls"][0].update(group="g"), "only applies to optional calls"),
    (lambda s: s["calls"].append(dict(s["calls"][0])), "duplicate call name"),
    (lambda s: s["calls"].append({"name": "d", "tool": "t", "inputs_from_call": {"call": "c", "map": {"a": "b"},
                                                                                 "abs_tol": 1e-3}}),
     "abs_tol only applies to compare 'geometry'"),
    (lambda s: s["calls"].append({"name": "d", "tool": "t", "inputs_from_call": {"call": "c",
                                                                                 "extract": "arxiv_ids"}}),
     "needs 'args'"),
    (lambda s: s["answers"][0].update(from_calls=[{"call": "c"}]), "exactly one of from_call / from_calls"),
    (lambda s: s["answers"][0].update(match="member"), "only applies to extracted answers"),
    (lambda s: s["answers"][0].update(abs_tol=-1), "non-negative"),
    (lambda s: s.update(bypass_tools={"WebFetch": "(bad"}), "regex"),
])
def test_spec_rejects_keys_that_would_be_ignored(mutate, match):
    spec = _valid_spec()
    mutate(spec)
    with pytest.raises(verify.SpecError, match=re.escape(match)):
        verify.parse_spec(spec)


def test_schema_1_specs_are_normalised():
    spec = json.loads((E2E_TASKS / "mcp_e2e/pyscf_rhf_energy/e2e_check.json").read_text())
    norm = verify.normalize_spec(spec)
    assert [c["tool"] for c in norm["calls"]] == ["pyscf_rhf_energy"]
    assert norm["answers"][0]["from_call"] == "pyscf_rhf_energy"
    assert verify.normalize_spec(norm) is norm


def test_requirements_group_optional_calls():
    CallSpec = verify.spec.CallSpec
    spec = verify.spec.Spec(2, "s", "r.json", "p.json", answers=(), calls=(
        CallSpec("a", "A"), CallSpec("b", "B", optional=True, group="g"),
        CallSpec("c", "C", optional=True, group="g"), CallSpec("d", "D", optional=True)))
    assert [[cs.name for cs in req] for req in spec.requirements()] == [["a"], ["b", "c"]]


def test_select_narrows_a_call_result_like_an_answer_source():
    raw = _valid_spec()
    raw["calls"][0]["result"] = {"format": "json", "key": "R", "select": {"argmax_of": "wavelength"},
                                 "reference_key": "y_ref", "abs_tol": 1e-9}
    cs = verify.parse_spec(raw).calls[0]
    call = verify.evidence.ToolCall(result_text=json.dumps({"wavelength": [1.0, 1.2, 1.1], "R": [0.1, 0.4, 0.2]}),
                                    is_error=False, input={"x": 3})
    assert verify.checks.judge_call(call, cs, {"y_ref": 0.4, "x_ref": 3})["result_ok"] is True
    assert verify.checks.judge_call(call, cs, {"y_ref": 0.2, "x_ref": 3})["result_ok"] is False
