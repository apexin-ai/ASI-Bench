"""Verifier value reading and comparison (e2e_verify/values.py): selectors, structured-output
unwrapping, chained-input and geometry comparators."""
import json

from . import support

verify = support.verify


def _call(payload):
    return verify.evidence.ToolCall(result_text=json.dumps(payload), is_error=False)


def _pick(call, key, ref, **select):
    selector = verify.spec.Selector(key=key, select=verify.spec.SelectSpec(**select) if select else None)
    return verify.values.read(call, selector, ref)


def test_source_value_select_modes():
    call = _call({"wavelength": [1.0, 1.1, 1.2], "R": [0.1, 0.3, 0.2], "T": [0.9, 0.7, 0.8]})
    ref = {"lam": 1.1, "far": 1.15}
    assert _pick(call, "R", ref, reduce="max") == 0.3
    assert _pick(call, "T", ref, reduce="min") == 0.7
    assert _pick(call, "wavelength", ref, argmax_of="R") == 1.1
    assert _pick(call, "T", ref, argmin_of="R") == 0.9
    assert _pick(call, "T", ref, where_key="wavelength", equals_reference_key="lam") == 0.7
    assert _pick(call, "T", ref, where_key="wavelength", equals_reference_key="far") is None
    assert _pick(call, "A", ref, reduce="max") is None
    assert _pick(call, "R", ref, argmax_of="missing") is None
    assert _pick(call, "R", ref) == [0.1, 0.3, 0.2]
    assert _pick(_call({"R": 0.5}), "R", ref, reduce="max") is None
    assert _pick(_call({"R": 0.5}), "R", ref) == 0.5
    assert _pick(verify.evidence.ToolCall(result_text="not json"), "R", ref, reduce="max") is None


def test_unwrap_structured_output():
    payload = {"R": 0.3, "ok": True}
    assert verify.values.unwrap_structured({"result": json.dumps(payload)}) == payload
    blocks = [{"type": "text", "text": json.dumps(payload), "annotations": None}, {"type": "image", "data": "x"}]
    assert verify.values.unwrap_structured({"result": blocks}) == payload
    assert verify.values.unwrap_structured({"result": 1.5}) == 1.5
    assert verify.values.unwrap_structured({"result": "not json"}) == "not json"
    assert verify.values.unwrap_structured({"result": [{"url": "a"}]}) == [{"url": "a"}]   # not content blocks
    assert verify.values.unwrap_structured({"result": 1, "other": 2}) == {"result": 1, "other": 2}
    call = verify.evidence.ToolCall(result_text=json.dumps({"result": "0.25"}))
    assert verify.values.read(call, verify.spec.Selector()) == 0.25


def test_same_link_compares_strings_exactly_and_numbers_to_print_precision():
    assert verify.values.same_link("b03a88ab5beb", "b03a88ab5beb")
    assert not verify.values.same_link("b03a88ab5beb", "b03a88ab5bec")
    assert verify.values.same_link("123456789012", 123456789012)
    assert verify.values.same_link([1.0, 2.0], [1.0, 2.0 + 1e-13])
    assert not verify.values.same_link(None, None) and not verify.values.same_link("", "")
    assert not verify.values.same_link({"a": 1}, {"a": 1})


def test_same_geometry():
    plain = "O 0 0 0.1\nH 0 0.75 -0.47\nH 0 -0.75 -0.47"
    assert verify.values.same_geometry("0 1\n" + plain + "\nsymmetry c1\n", "3\nwater\n" + plain, 1e-4)
    assert verify.values.same_geometry(plain.replace("0.75", "0.75004"), plain, 1e-4)
    assert not verify.values.same_geometry(plain.replace("0.75", "0.7502"), plain, 1e-4)
    assert not verify.values.same_geometry(plain.replace("O", "S", 1), plain, 1e-4)
    assert not verify.values.same_geometry("\n".join(plain.splitlines()[:2]), plain, 1e-4)
    assert not verify.values.same_geometry("no atoms here", plain, 1e-4) and not verify.values.same_geometry(None, plain, 1e-4)
    link = verify.spec.Binding(("g",), verify.spec.Selector(key="g", raw=True))
    assert verify.values.link_comparator(link)("abc", "abc") and not verify.values.link_comparator(link)("0.1", "0.2")
    geo = verify.spec.Binding(("g",), verify.spec.Selector(key="g", raw=True), "geometry", 1e-4)
    assert verify.values.link_comparator(geo)(plain.replace("0.75", "0.75004"), plain)
