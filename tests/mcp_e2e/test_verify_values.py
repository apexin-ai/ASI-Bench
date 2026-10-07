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


def _persisted(text, tmp_path):
    """``text`` through the orchestrator's real persistence sanitizer (as a JSONL string value)."""
    line = support.persist_like_run(json.dumps({"type": "assistant", "v": text}) + "\n", tmp_path=tmp_path)[0]
    return json.loads(line)["v"]


def test_scrub_host_paths_equals_the_persistence_sanitizer(tmp_path):
    import base64
    import random
    rng = random.Random(7)
    samples = ["gASV+/AAAA/BBBB", "x /usr/bin/python3 -c 1", "see https://example.org/a/b and /c/d", "a/b/c",
               "//AA/BB", "C(=O)O", "<workspace>/out/conformer.sdf", "/tmp/x"]
    samples += [base64.b64encode(rng.randbytes(rng.randrange(200, 800))).decode() for _ in range(200)]
    for text in samples:
        assert verify.values.scrub_host_paths(text) == _persisted(text, tmp_path), text[:60]
    assert sum(verify.values.SCRUBBED in verify.values.scrub_host_paths(s) for s in samples) > 10


def test_same_text_tolerates_only_the_persistence_scrubbing():
    pickle_b64 = "gASVnwEAAAAAAACMEXJka2l0+/Q2hlbS5yZGNoZW0/lIwDTW9slJOUQnMBAADvvq3e"
    scrubbed = verify.values.scrub_host_paths(pickle_b64)
    assert verify.values.SCRUBBED in scrubbed
    assert verify.values.same_text(pickle_b64, pickle_b64) and verify.values.same_text(scrubbed, pickle_b64)
    assert verify.values.same_text(f" {pickle_b64}\n", pickle_b64)
    assert not verify.values.same_text(scrubbed.replace("gASV", "gASW"), pickle_b64)    # the visible rest must match
    assert not verify.values.same_text(pickle_b64, scrubbed)                            # only the given side is persisted
    # limitation: a value scrubbed entirely matches any path-like value, so tasks never chain bare host paths
    assert verify.values.same_text("<abs_path>", "/some/other/path")
    assert not verify.values.same_text("", "") and not verify.values.same_text(None, "x")
    assert verify.values.same_link(scrubbed, pickle_b64) and verify.values.same_input(scrubbed, pickle_b64)
    assert verify.values.matches(scrubbed, pickle_b64, "equal")


def test_whole_result_selector_and_text_extractors():
    pickle_b64 = "gASVdQAAAAAAAACMEXJka2l0LkNoZW0ucmRjaGVt"
    shown_structured = verify.evidence.ToolCall(result_text=json.dumps({"result": pickle_b64}))   # Claude
    shown_text = verify.evidence.ToolCall(result_text=pickle_b64 + "\n")                          # Codex text block
    whole = verify.spec.Selector(raw=True)
    assert verify.values.read(shown_structured, whole) == pickle_b64 == verify.values.read(shown_text, whole)
    text = verify.spec.Selector(extract="text")
    assert verify.values.read(shown_structured, text) == pickle_b64 == verify.values.read(shown_text, text)
    embedded = verify.evidence.ToolCall(result_text=json.dumps({"conf_id": 0, "mol": pickle_b64}))
    assert verify.values.read(embedded, verify.spec.Selector(extract="rdkit_mol")) == pickle_b64
    assert verify.values.read(embedded, verify.spec.Selector(extract="text")) is None
    assert verify.values.read(shown_text, verify.spec.Selector(extract="rdkit_mol")) is None
    name = verify.spec.Selector(extract="file_name")
    for shown in ("/tmp/ws/conformer.sdf", "<workspace>/conformer.sdf", json.dumps({"result": "/a/b/conformer.sdf"})):
        assert verify.values.read(verify.evidence.ToolCall(result_text=shown), name) == "conformer.sdf"
    assert verify.values.read(verify.evidence.ToolCall(result_text=None), name) is None


def test_json_scalars_reads_records_keyed_by_the_requested_identifier():
    """E-utilities keys a record by the uid that was asked for, so no static dotted path
    reaches ``slen``; the extractor collects the scalar leaves instead."""
    esummary = _call({"status": "success", "url": "https://eutils.ncbi.nlm.nih.gov/x",
                      "data": {"result": {"uids": ["4557819"],
                                          "4557819": {"accessionversion": "NP_000268.1", "slen": 452,
                                                      "taxid": 9606, "moltype": "aa",
                                                      "subtype": None, "replaced": False}}}})
    scalars = verify.spec.Selector(extract="json_scalars")
    values = verify.values.read(esummary, scalars)
    assert values == sorted({"4557819", "9606", "452", "NP_000268.1", "aa", "success",
                             "https://eutils.ncbi.nlm.nih.gov/x"})
    assert None not in values                          # null and False are not answerable values
    canon = verify.extractors.canon_scalar
    for answer, found in ((452, True), ("452", True), ("NP_000268.1", True), ("452.0", True),
                          (9606, True), ("NP_000268.2", False), (453, False), (True, False)):
        assert verify.values.matches(values, canon(answer), "member") is found, answer
    # the whole record, not just the asked-for uid: a value nested in a list is reachable too
    gene = _call({"status": "success",
                  "data": {"result": {"uids": ["5053"],
                                      "5053": {"nomenclaturesymbol": "PAH", "maplocation": "12q23.2",
                                               "genomicinfo": [{"chraccver": "NC_000012.12",
                                                                "chrstart": 102958440}],
                                               "organism": {"taxid": 9606}}}}})
    found = verify.values.read(gene, scalars)
    assert all(canon(v) in found for v in ("PAH", "12q23.2", "NC_000012.12", 9606, 102958440))
    assert verify.values.read(_call({"empty": {}, "nothing": [None, True]}), scalars) is None
    assert verify.values.read(verify.evidence.ToolCall(result_text=None), scalars) is None


def test_canon_scalar_pairs_numbers_with_their_string_spelling():
    canon = verify.extractors.canon_scalar
    assert canon(644) == canon("644") == canon(" 644 ") == canon(644.0) == canon("0644") == "644"
    assert canon(0.457) == canon("0.4570") == canon("4.57e-1") == repr(0.457)
    assert canon("NP_000268.1") == "NP_000268.1" and canon("12q23.2") == "12q23.2"
    assert canon("PAH") != canon("pah")                                  # identifiers stay case-sensitive
    assert canon(None) is None and canon(True) is None and canon("") is None and canon("  ") is None
    assert canon(float("inf")) == "inf" and canon("nan") == "nan"         # kept as text, never equal numerically
    assert canon([3, "1", 1.0, None, True]) == ["1", "3"]                 # sorted, unique, nulls dropped
    assert canon([]) is None and canon([None]) is None
    # A big integer must not round-trip through float, and float()'s literal tolerance must not
    # make a grouped number equal to a plain one.
    assert canon(9007199254740993) == "9007199254740993" and canon("1_000") == "1_000" != canon(1000)
    # result.json and tool arguments are agent-controlled: a nested list or an object reads as
    # "no value" instead of raising (a raise here would abort the whole verify_run report).
    assert canon([[5053]]) is None and canon({"a": 1}) is None and canon([{"a": 1}, "ok"]) == ["ok"]
