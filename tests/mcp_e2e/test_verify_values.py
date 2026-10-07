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


def test_listed_files_extractor_reads_any_file_listing():
    """A listing result is compared by file name, not by the host directory it names."""
    read = lambda data: verify.values.read(                                   # noqa: E731
        verify.evidence.ToolCall(result_text=json.dumps(data)),
        verify.spec.Selector(extract="listed_files"))
    want = ["bands.png", "dos.png"]
    assert read({"ok": True, "artifacts": [{"path": "runs/x/dos.png", "bytes": 1},
                                           {"path": "runs/x/bands.png", "bytes": 2}]}) == want
    assert read({"files": ["/abs/dos.png", "/abs/bands.png"]}) == want
    assert read(["runs/x/bands.png", "runs/x/dos.png"]) == want
    assert read({"entries": [{"name": "bands.png"}, {"filename": "dos.png"}]}) == want
    # The reference side canonicalises the same way, so a spec lists bare names.
    assert verify.extractors.EXTRACTORS["listed_files"].canon(want) == want
    # Shapes it must refuse rather than guess at
    for bad in ({"artifacts": [], "files": []}, {"artifacts": [{"bytes": 1}]},
                {"artifacts": [1, 2]}, {"nothing": ["a.png"]}, {}, "a.png"):
        assert read(bad) is None, bad


def test_named_statuses_extractor_reads_a_report_of_named_checks():
    read = lambda data: verify.values.read(                                   # noqa: E731
        verify.evidence.ToolCall(result_text=json.dumps(data)),
        verify.spec.Selector(extract="named_statuses"))
    report = {"verdict": "pass_with_warnings",
              "checks": [{"check": "convergence_gate", "status": "pass"},
                         {"check": "band_gap_vs_mp", "status": "warn"}]}
    want = ["convergence_gate:pass", "band_gap_vs_mp:warn"]
    assert read(report) == want
    assert read({"checks": [{"name": "a", "status": "PASS"}]}) == ["a:pass"]
    assert verify.extractors.EXTRACTORS["named_statuses"].canon(want) == want
    # Order is part of the report, so a reordered list is a different value.
    assert read(report) != list(reversed(want))
    for bad in ({"checks": []}, {"checks": [{"check": "a"}]}, {"checks": [{"status": "pass"}]},
                {"checks": "pass"}, {"verdict": "pass"}, ["a:pass"]):
        assert read(bad) is None, bad


def test_superset_match_accepts_a_longer_listing():
    """A file listing carries entries a task does not pin (GPAW's per-calculation logs),
    so the required ones only have to be present."""
    required = ["bands.png", "summary.json"]
    listing = ["bands.png", "scf_300.txt", "summary.json"]
    assert verify.values.matches(listing, required, "superset")
    assert not verify.values.matches(listing, [*required, "missing.json"], "superset")
    assert not verify.values.matches(required, listing, "superset")      # direction matters
    # equal still means equal, and the dict-only subset mode is untouched
    assert not verify.values.matches(listing, required, "equal")
    assert verify.values.matches({"a": 1}, {"a": 1, "b": 2}, "subset")
    assert not verify.values.matches(listing, required, "subset")
    for empty in ([], None):
        assert not verify.values.matches(listing, empty, "superset")
