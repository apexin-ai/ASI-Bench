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


def test_categorized_files_reads_every_category_of_a_listing():
    """qe-mcp's qe_list_files groups paths under ``*_files`` keys; names compare by file
    name (categorized_files), a chained argument by exact path (categorized_paths)."""
    listing = {"success": True, "directory": "/r/bands_1",
               "band_files": ["/r/bands_1/bands.dat.gnu"], "dos_files": [],
               "input_files": ["/r/bands_1/scf.in", "/r/bands_1/bands.in"],
               "output_files": ["/r/bands_1/scf.out"], "other_files": ["/r/bands_1/bands.dat"]}
    call = verify.evidence.ToolCall(result_text=json.dumps(listing))
    names = verify.values.read(call, verify.spec.Selector(extract="categorized_files"))
    assert names == ["bands.dat", "bands.dat.gnu", "bands.in", "scf.in", "scf.out"]
    paths = verify.values.read(call, verify.spec.Selector(extract="categorized_paths"))
    assert "/r/bands_1/bands.dat.gnu" in paths and "/r/bands_1" not in paths
    canon = verify.extractors.EXTRACTORS["categorized_paths"].canon
    assert canon(" /r/bands_1/bands.dat.gnu ") == "/r/bands_1/bands.dat.gnu"
    for bad in ({"directory": "/r"}, {"band_files": [1]}, {"band_files": [""]}, ["a"], "a"):
        read = verify.values.read(verify.evidence.ToolCall(result_text=json.dumps(bad)),
                                  verify.spec.Selector(extract="categorized_files"))
        assert read is None, bad


def test_kpoint_grid_canonicalises_every_spelling():
    canon = verify.extractors.EXTRACTORS["kpoint_grid"].canon
    for spelling in ([7, 7, 7], [7.0, 7, "7"], "7,7,7", "7 7 7", " 7, 7, 7 ", "7x7x7", "7"):
        assert canon(spelling) == "7,7,7", spelling
    assert canon([4, 4, 1]) == "4,4,1"
    for bad in ("auto", "gamma", "4 4 4 0 0 0", [7, 7], [0, 1, 1], [1.5, 1, 1], [True, 1, 1],
                None, {"kpoints": [7, 7, 7]}, "", float("nan")):
        assert canon(bad) is None, bad
    call = verify.evidence.ToolCall(result_text=json.dumps({"kpoints": [5, 5, 5], "method": "x"}))
    assert verify.values.read(call, verify.spec.Selector(extract="kpoint_grid")) == "5,5,5"


def test_output_file_reads_the_written_file_by_name():
    """mcp-atomictoolkit echoes the file it read as ``input_filepath`` and the file it wrote
    as ``filepath`` (both also in ``artifacts``); only the written one is the output, and a
    path argument scrubbed in the persisted log still compares by its file name."""
    result = {"status": "success", "operation": "vacancy", "input_filepath": "/w/cu32.extxyz",
              "filepath": "/w/cu31.extxyz", "num_atoms": 31,
              "artifacts": [{"label": "input_filepath", "filepath": "/w/cu32.extxyz"},
                            {"label": "filepath", "filepath": "/w/cu31.extxyz"}]}
    call = verify.evidence.ToolCall(result_text=json.dumps(result))
    assert verify.values.read(call, verify.spec.Selector(extract="output_file")) == "cu31.extxyz"
    canon = verify.extractors.EXTRACTORS["output_file"].canon
    assert canon("<workspace>/cu31.extxyz") == "cu31.extxyz"
    assert canon("<abs_path>") == "<abs_path>"
    for bad in ({"input_filepath": "/w/cu32.extxyz"}, {"filepath": ""}, {"filepath": 3}, ["/w/a.extxyz"],
                "/w/a.extxyz"):
        read = verify.values.read(verify.evidence.ToolCall(result_text=json.dumps(bad)),
                                  verify.spec.Selector(extract="output_file"))
        assert read is None, bad


def test_a_scrubbed_link_input_is_lost_only_if_its_canonical_form_is():
    """A file-name link survives ``<workspace>/`` scrubbing (its mismatch is real); a bare
    ``<abs_path>``, or a scrubbed argument of a verbatim link, leaves nothing to compare."""
    Binding, Selector = verify.spec.Binding, verify.spec.Selector
    by_name = Binding(("input_filepath",), Selector(extract="output_file"), "member")
    verbatim = Binding(("input_filepath",), Selector(key="filepath", raw=True))
    ToolCall = verify.evidence.ToolCall
    lost = verify.values.link_input_lost
    assert not lost(ToolCall(input={"input_filepath": "<workspace>/cu31.extxyz"}), by_name)
    assert not lost(ToolCall(input={"input_filepath": "/w/cu31.extxyz"}), by_name)
    assert lost(ToolCall(input={"input_filepath": "<abs_path>"}), by_name)
    assert lost(ToolCall(input={"input_filepath": "<workspace>/cu31.extxyz"}), verbatim)
    assert not lost(ToolCall(input={"input_filepath": "/w/cu31.extxyz"}), verbatim)
    assert lost(ToolCall(input=None), by_name)


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


# Session results as OpenROAD-MCP v1.1.0 returned them on the AWS host (openroad v2.0-17598).
_READ_DEF = ("read_def /home/e2e/or-pre/tiny_31415.def\n[INFO ODB-0127] Reading DEF file: /home/e2e/or-pre/tiny_31415.def\n"
             "[INFO ODB-0128] Design: tiny\n[INFO ODB-0130]     Created 2 pins.\n"
             "[INFO ODB-0131]     Created 3 components and 6 component-terminals.\n"
             "[INFO ODB-0133]     Created 4 nets and 6 connections.\n"
             "[INFO ODB-0134] Finished DEF file: /home/e2e/or-pre/tiny_31415.def\n%")


def _session(output, error=None):
    return _call({"output": output, "session_id": "pre1", "timestamp": "2026-10-07T11:30:10.804Z",
                  "execution_time": 0.003, "command_count": 3, "buffer_size": 131072, "truncated": False,
                  "error": error})


def test_openroad_output_reads_facts_and_printed_numbers():
    """Labelled facts from the read_lef / read_def INFO lines and report_design_area, plus every
    printed number; message ids (``ODB-0131``), ``u^2``, paths and the echoed command do not count."""
    read = lambda call: verify.values.read(call, verify.spec.Selector(extract="openroad_output"))
    got = read(_session(_READ_DEF))
    assert {"pins=2", "components=3", "component_terminals=6", "nets=4", "connections=6"} <= set(got)
    assert {"2", "3", "4", "6"} <= set(got)
    assert not {"0127", "127", "131", "31415", "-0131"} & set(got)       # ids and path digits
    lef = read(_session("read_lef /w/tiny.lef\n[INFO ODB-0227] LEF file: /w/tiny.lef, created 1 layers, "
                        "2 library cells\n%"))
    assert {"layers=1", "library_cells=2"} <= set(lef)
    area = read(_session("% report_design_area\nDesign area 10 u^2 37% utilization.\n%"))
    assert set(area) == {"design_area_um2=10", "utilization_percent=37", "10", "37"}     # canonical: sorted, unique
    assert {"design_area_um2=6", "utilization_percent=30"} <= \
        set(read(_session("report_design_area\nDesign area 6.0 u^2 30% utilization.")))
    # bare numbers of a user Tcl line, after a stale prompt; the echo's literal 1000 is dropped
    assert read(_session("puts \"HPWL_TOTAL [expr {9400 * 1000 / 1000}]\"\n% HPWL_TOTAL 9400\n%")) == ["9400"]
    assert read(_session("x\nDIE 0 0 12000 6000\n%")) == ["0", "12000", "6000"]
    assert set(read(_session("x\n12.0, (6.5); -3 nan inf 1_0 0x10 v2.0-17598"))) == {"12", "6.5", "-3"}
    # nothing printed, a flagged command, or not a session result
    for bad in (_session("set_cmd_units -distance um\n%"),
                _session("read_lef /x.lef\n[ERROR ORD-0001] /x.lef does not exist.\nORD-0001",
                         error="OpenROAD ORD-0001: /x.lef does not exist."),
                _call({"session_id": "a", "command_count": 0}), _call(["x\n3"]),
                verify.evidence.ToolCall(result_text=None)):
        assert read(bad) is None
    canon = verify.extractors.EXTRACTORS["openroad_output"].canon
    assert canon(37) == canon("37") == canon(37.0) == "37"
    assert verify.values.matches(read(_session(_READ_DEF)), canon(["components=3", "nets=4"]), "superset")
    assert not verify.values.matches(read(_session(_READ_DEF)), canon(["components=4"]), "superset")
    assert verify.values.matches(read(_session(_READ_DEF)), canon(4), "member")
