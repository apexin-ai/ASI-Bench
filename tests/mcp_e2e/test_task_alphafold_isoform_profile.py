"""alphafold_isoform_profile (all three tools on one accession: isoform models -> summary -> per-residue
AlphaMissense annotation; entry ids fixed by the input file; answers read with json_scalars and a
numeric dotted path): generator helpers and assertions, scorers, verifier scenarios. No network."""
import hashlib
import json

import pytest

from . import support
from .support import claude, codex, jsonl

TASK = support.Task("mcp_e2e.alphafold_isoform_profile")
TASK_DIR, TASK_ID, INSTANCE_ID = TASK.dir, TASK.task_id, TASK.instance_id

# seed31415 case (BAX, Q07812), as generated 2026-10-07 against model v6
ACCESSION = "Q07812"
LENGTHS = {"AF-Q07812-F1": 192, "AF-Q07812-2-F1": 218, "AF-Q07812-3-F1": 41, "AF-Q07812-4-F1": 143,
           "AF-Q07812-5-F1": 164, "AF-Q07812-6-F1": 114, "AF-Q07812-7-F1": 173, "AF-Q07812-8-F1": 179}
RANKING = ["AF-Q07812-2-F1", "AF-Q07812-F1", "AF-Q07812-8-F1", "AF-Q07812-7-F1", "AF-Q07812-5-F1",
           "AF-Q07812-4-F1", "AF-Q07812-6-F1", "AF-Q07812-3-F1"]
MAX_RESIDUE, MAX_SCORE, N_ABOVE = 108, 0.999, 61
REFERENCE = {"accession": ACCESSION, "entry_ids": sorted(LENGTHS), "entry_lengths": LENGTHS,
             "entry_ids_by_length_desc": RANKING, "longest_entry_id": RANKING[0],
             "canonical_entry_id": "AF-Q07812-F1", "canonical_length": 192, "am_threshold": 0.9,
             "max_am_residue": MAX_RESIDUE, "max_am_score": MAX_SCORE, "max_am_score_text": "0.999",
             "n_am_above_threshold": N_ABOVE}
PREDICTION = "mcp__alphafold_db__alphafold_get_prediction"
SUMMARY = "mcp__alphafold_db__alphafold_get_summary"
ANNOTATIONS = "mcp__alphafold_db__alphafold_get_annotations"
ALL = ("afdb_e2e_schema", "afdb_e2e_isoforms", "afdb_e2e_canonical_length", "afdb_e2e_am_profile")
FILES = "https://alphafold.ebi.ac.uk/files"

generate_gt = TASK.module("generate_gt")
scorer = TASK.module("custom_scorer")
verify = support.verify
_status = support.statuses


def _md5(text):
    return hashlib.md5(text.encode()).hexdigest()


def _model(entry_id, length):
    """One prediction entry with the fields the live API returns (measured 2026-10-07), the
    four length-like fields consistent, as they are for every model of the curated cases."""
    isoform = entry_id[3:-3]                             # AF-Q07812-2-F1 -> Q07812-2
    sequence = ("MDGSGEQPRGGGPTSSEQIMKTGALLLQGFIQDRAGRMGGEAPELALDPVPQDASTKKLSECLKRIGDELDSNMELQRMIAA"
                "VDTDSPREVFFRVAADMFSDGNFNWGRVVALFYFASKLVLKALCTKVPELIRTIMGWTLDFLRERLLGWIQDQGGWDGLLS"
                "YFGTPTWQTVTIFVAGVLTASLTIWKKMG" * 3)[:length]
    return {"entryId": entry_id, "uniprotAccession": isoform, "uniprotId": "BAX_HUMAN", "gene": "BAX",
            "taxId": 9606, "uniprotSequence": sequence, "sequence": sequence,
            "sequenceChecksum": _md5(sequence), "uniprotStart": 1, "uniprotEnd": length,
            "sequenceStart": 1, "sequenceEnd": length, "latestVersion": 6,
            "modelCreatedDate": "2025-08-01T00:00:00Z", "globalMetricValue": 75.2,
            "pdbUrl": f"{FILES}/{entry_id}-model_v6.pdb",
            "amAnnotationsUrl": f"{FILES}/{entry_id}-aa-substitutions.csv"}


def _scores(n=192, top=MAX_RESIDUE, top_text="0.999", above=N_ABOVE):
    """Region values: ``above`` residues over 0.9 (one of them the top), the rest at 0.4552."""
    values = {p: "0.4552" for p in range(1, n + 1)}
    for p in [p for p in range(1, n + 1) if p != top][:above - 1]:
        values[p] = "0.9213"
    values[top] = top_text
    return values


def _wrap(data):
    """A payload as AlphaFoldRESTTool returns it."""
    return json.dumps({"status": "success", "data": data,
                       "metadata": {"count": 1, "endpoint": "x", "query": {}}})


def _prediction_result(lengths=LENGTHS):
    return _wrap([_model(entry_id, length) for entry_id, length in lengths.items()])


def _summary_result(length=192):
    return _wrap({"uniprot_entry": {"ac": ACCESSION, "id": "BAX_HUMAN", "uniprot_checksum": "c" * 32,
                                    "sequence_length": length, "segment_start": 1, "segment_end": length},
                  "structures": [{"summary": {"model_identifier": "AF-Q07812-F1", "coverage": 1.0,
                                              "confidence_avg_local_score": 75.2, "uniprot_start": 1,
                                              "uniprot_end": length}}]})


def _annotation_result(values=None):
    values = values or _scores()
    regions = [{"start": p, "end": p, "annotation_value": v, "unit": None} for p, v in sorted(values.items())]
    return _wrap({"accession": ACCESSION, "id": "AF-Q07812-F1", "sequence": "M" * len(regions),
                  "annotation": [{"type": "MUTAGEN", "description": "AM score", "source_name": "AFDB",
                                  "source_url": f"{FILES}/AF-Q07812-F1-aa-substitutions.csv",
                                  "evidence": "COMPUTATIONAL/PREDICTED", "residues": sorted(values),
                                  "regions": regions}]})


def _answer(**overrides):
    return {"entry_ids_by_length_desc": list(RANKING), "longest_entry_id": RANKING[0], "canonical_length": 192,
            "max_am_residue": MAX_RESIDUE, "max_am_score": MAX_SCORE, "n_am_above_threshold": N_ABOVE,
            **overrides}


def _dirs(tmp_path, prediction, reference=REFERENCE):
    return support.score_dirs(tmp_path, prediction, reference)


def _score(name, pred, ref, weight=1.0):
    from ai4sci_bench.core.scorer import get_scorer
    return get_scorer(name).score(pred, ref, {"weight": weight})


def _stream(args=None, prediction_result=None, summary_result=None, annotation_result=None, tools=None,
            extra=(), skip=()):
    """Claude stream-json for the three calls; extra = [(tool name, input)] before them."""
    args = args or {"qualifier": ACCESSION}
    events = [claude.init("alphafold_db", tools or ["Bash", "Read", "Write", "WebFetch", "WebSearch",
                                                    PREDICTION, SUMMARY, ANNOTATIONS])]
    for k, (name, arguments) in enumerate(extra):
        events += claude.call(f"x{k}", name, arguments, "ok")
    for cid, tool, result in (("p1", PREDICTION, prediction_result or _prediction_result()),
                              ("s1", SUMMARY, summary_result or _summary_result()),
                              ("a1", ANNOTATIONS, annotation_result or _annotation_result())):
        if tool not in skip:
            events += claude.call(cid, tool, args, result)
    events.append(claude.result(5))
    return jsonl(events)


def _codex(extra_items=()):
    events = [*codex.START, *(codex.item(item) for item in extra_items)]
    for item_id, tool, result in (("m1", "alphafold_get_prediction", _prediction_result()),
                                  ("m2", "alphafold_get_summary", _summary_result()),
                                  ("m3", "alphafold_get_annotations", _annotation_result())):
        events += codex.mcp(item_id, "alphafold_db", tool, {"qualifier": ACCESSION}, result)
    events.append(codex.done())
    return jsonl(events)


def _run(tmp_path, stream, answer=None, codex=False, files=None):
    return TASK.verify(tmp_path, stream, reference=REFERENCE, answer=answer or _answer(),
                       harness="codex" if codex else "claude", files=files)


# --------------------------------------------------------------------------
# Generator: pure helpers
# --------------------------------------------------------------------------

def test_cases_are_deterministic_and_cover_all_curated_cases():
    assert generate_gt.build_case(31415) == generate_gt.build_case(31415)
    case = generate_gt.build_case(31415)
    assert case["accession"] == ACCESSION and case["protein"] == "BAX" and case["am_threshold"] == 0.9
    assert {generate_gt.build_case(s)["accession"] for s in range(100)} == {c["accession"] for c in generate_gt.CASES}


def test_curated_cases_are_unversioned_accessions_with_a_topic():
    for case in generate_gt.CASES:
        assert len(case["accession"]) == 6 and case["accession"].isalnum() and "-" not in case["accession"]
        assert case["protein"] in case["topic"]


def test_isoform_accessions_and_ranking():
    assert generate_gt.isoform_of("Q07812", "Q07812") and generate_gt.isoform_of("Q07812-2", "Q07812")
    assert not generate_gt.isoform_of("Q07813", "Q07812") and not generate_gt.isoform_of("Q07812-x", "Q07812")
    assert not generate_gt.isoform_of("", "Q07812")
    assert generate_gt.rank_by_length(LENGTHS) == RANKING


def test_region_scores_keep_the_written_value_and_refuse_multi_residue_regions():
    block = {"regions": [{"start": 1, "end": 1, "annotation_value": "0.999"},
                         {"start": 2, "end": 2, "annotation_value": "1.0"}]}
    assert generate_gt.region_scores(block) == {1: ("0.999", 0.999), 2: ("1.0", 1.0)}
    with pytest.raises(RuntimeError, match="multi-residue"):
        generate_gt.region_scores({"regions": [{"start": 1, "end": 3, "annotation_value": "0.5"}]})


def test_am_means_average_the_19_substitutions_per_position():
    csv = ("protein_variant,am_pathogenicity,am_class\nM1A,0.2,Ben\nM1C,0.4,Amb\nD2A,0.9,Path\n"
           "K108A,0.99,Path\nK108W,0.98,Path\nG1480R,0.5,Amb\n")
    assert generate_gt.am_means(csv) == pytest.approx({1: 0.3, 2: 0.9, 108: 0.985, 1480: 0.5})
    assert generate_gt.am_means("variant,score\nM1A,0.2\n") == {}


def test_profile_scores_need_a_unique_top_and_no_value_at_the_threshold():
    """A discrete answer must not sit on a decision boundary: a tied top has no unique residue,
    and a score exactly at the threshold makes 'above' and 'at least' differ."""
    scores = {1: ("0.5", 0.5), 2: ("0.95", 0.95), 3: ("0.9001", 0.9001)}
    assert generate_gt.profile_scores(scores, 0.9) == {"max_am_residue": 2, "max_am_score": 0.95,
                                                       "max_am_score_text": "0.95", "n_am_above_threshold": 2}
    with pytest.raises(RuntimeError, match="shared by residues"):
        generate_gt.profile_scores({**scores, 4: ("0.95", 0.95)}, 0.9)
    with pytest.raises(RuntimeError, match="exactly 0.9"):
        generate_gt.profile_scores({**scores, 4: ("0.9", 0.9)}, 0.9)


def test_payloads_near_the_server_truncation_are_refused(monkeypatch):
    monkeypatch.setattr(generate_gt, "http_get", lambda url: json.dumps([{"x": "y" * 70_000}]))
    with pytest.raises(RuntimeError, match="truncation"):
        generate_gt.api_json("prediction/X")
    monkeypatch.setattr(generate_gt, "http_get", lambda url: json.dumps([{"x": "y"}]))
    assert generate_gt.api_json("prediction/X") == [{"x": "y"}]


# --------------------------------------------------------------------------
# Generator: the reference, built offline from fixtures of the raw endpoints
# --------------------------------------------------------------------------
# The generator talks to the AlphaFold DB API directly, so these payloads carry no
# {"status", "data", "metadata"} envelope -- that one belongs to the MCP wrapper.

SMALL = {"AF-Q07812-F1": 5, "AF-Q07812-2-F1": 6, "AF-Q07812-3-F1": 4}
SMALL_VALUES = {1: "0.4552", 2: "0.9213", 3: "0.999", 4: "0.1", 5: "0.95"}


def _csv(values):
    rows = [f"X{p}A,{float(v):.4f},amb" for p, v in sorted(values.items()) for _ in range(19)]
    return "protein_variant,am_pathogenicity,am_class\n" + "\n".join(rows) + "\n"


def _patch_api(monkeypatch, *, lengths=SMALL, values=SMALL_VALUES, models=None, summary=None,
               annotation=None, csv=None):
    models = models if models is not None else [_model(e, n) for e, n in lengths.items()]
    canonical = next((m for m in models if m["entryId"] == "AF-Q07812-F1"), models[0])
    summary = {"uniprot_entry": {"ac": ACCESSION, "uniprot_checksum": canonical["sequenceChecksum"],
                                 "sequence_length": len(canonical["uniprotSequence"])},
               "structures": [], **(summary or {})}
    regions = [{"start": p, "end": p, "annotation_value": v, "unit": None} for p, v in sorted(values.items())]
    annotation = {"accession": ACCESSION, "id": "AF-Q07812-F1", "sequence": canonical["uniprotSequence"],
                  "annotation": [{"type": "MUTAGEN", "source_url": f"{FILES}/AF-Q07812-F1-aa-substitutions.csv",
                                  "regions": regions}], **(annotation or {})}
    documents = {f"prediction/{ACCESSION}": models, f"uniprot/summary/{ACCESSION}.json": summary,
                 f"annotations/{ACCESSION}.json?type=MUTAGEN": annotation}
    monkeypatch.setattr(generate_gt, "api_json", lambda path: documents[path])
    monkeypatch.setattr(generate_gt, "http_get", lambda url: csv if csv is not None else _csv(values))


CASE = {"accession": ACCESSION, "protein": "BAX", "topic": "BAX", "am_threshold": 0.9}


def test_reference_carries_every_key_the_verifier_and_the_scorers_read(monkeypatch):
    _patch_api(monkeypatch)
    reference = generate_gt.build_reference(CASE)
    assert reference["entry_ids"] == sorted(SMALL)
    assert reference["entry_ids_by_length_desc"] == ["AF-Q07812-2-F1", "AF-Q07812-F1", "AF-Q07812-3-F1"]
    assert reference["longest_entry_id"] == "AF-Q07812-2-F1" and reference["canonical_length"] == 5
    assert (reference["max_am_residue"], reference["max_am_score"], reference["n_am_above_threshold"]) == \
        (3, 0.999, 3)
    spec = verify.parse_spec(json.loads((TASK_DIR / "e2e_check.json").read_text()))
    needed = {cs.result.reference_key for cs in spec.calls} | {a.reference_key for a in spec.answers}
    needed |= {key for cs in spec.calls for key in cs.inputs_from_reference.values()}
    assert needed <= set(reference), sorted(needed - set(reference))
    # and the scorers accept it as a reference
    assert scorer._normalize(reference, "reference.json")["longest_entry_id"] == "AF-Q07812-2-F1"


def test_csv_means_are_matched_to_the_4_decimal_rounding(monkeypatch):
    """The annotation carries round(mean, 4); a CSV whose rows average to within 5e-5 of it agrees."""
    rows = [f"X{p}A,{float(v) + (3e-5 if p == 2 else 0):.6f},amb" for p, v in sorted(SMALL_VALUES.items())
            for _ in range(19)]
    _patch_api(monkeypatch, csv="protein_variant,am_pathogenicity,am_class\n" + "\n".join(rows) + "\n")
    assert generate_gt.build_reference(CASE)["max_am_residue"] == 3


def test_generate_writes_the_listed_entries_the_reference_and_the_prompts(monkeypatch, tmp_path):
    _patch_api(monkeypatch)
    monkeypatch.setattr(generate_gt, "build_case", lambda seed: dict(CASE, topic="the apoptosis regulator BAX"))
    meta = generate_gt.generate(tmp_path, {"seed": 31415})
    profile = json.loads((tmp_path / "data/profile.json").read_text())
    reference = json.loads((tmp_path / "reference/reference.json").read_text())
    assert profile["entry_ids"] == reference["entry_ids"] == sorted(SMALL)
    assert profile["accession"] == ACCESSION and profile["am_threshold"] == 0.9
    assert meta["input_files"] == ["profile.json"] and meta["reference_files"] == ["reference.json"]
    for level in ("b1", "b2", "b3", "b4"):
        text = (tmp_path / f"prompt_{level}.md").read_text()
        assert "{{topic}}" not in text and "the apoptosis regulator BAX" in text


def _equal_lengths():
    return {"lengths": {**SMALL, "AF-Q07812-3-F1": 6}}


def _fragment():
    models = [_model(e, n) for e, n in SMALL.items()]
    models[1] = {**models[1], "uniprotStart": 3}
    return {"models": models}


def _bad_checksum():
    models = [_model(e, n) for e, n in SMALL.items()]
    models[2] = {**models[2], "sequenceChecksum": "0" * 32}
    return {"models": models}


def _foreign_model():
    return {"models": [*(_model(e, n) for e, n in SMALL.items()),
                       {**_model("AF-P04637-F1", 3), "uniprotAccession": "P04637"}]}


@pytest.mark.parametrize("mutation,message", [
    (_equal_lengths(), "equal length"),
    ({"lengths": {"AF-Q07812-2-F1": 6, "AF-Q07812-3-F1": 4}}, "lack AF-Q07812-F1"),
    ({"lengths": {"AF-Q07812-F1": 5}}, "lack AF-Q07812-F1 or any isoform"),
    (_fragment(), "length fields disagree"),
    (_bad_checksum(), "sequenceChecksum"),
    (_foreign_model(), "not an isoform model"),
    ({"summary": {"uniprot_entry": {"sequence_length": 6, "uniprot_checksum": "x"}}}, "sequence_length"),
    ({"summary": {"uniprot_entry": {"sequence_length": 5, "uniprot_checksum": "x"}}}, "uniprot_checksum"),
    ({"annotation": {"id": "AF-Q07812-2-F1"}}, "not the canonical"),
    ({"annotation": {"annotation": []}}, "0 MUTAGEN blocks"),
    ({"values": {1: "0.1", 2: "0.2"}}, "expected one per residue"),
    ({"csv": _csv({**SMALL_VALUES, 2: "0.5"})}, "differ from the CSV means"),
    ({"csv": "protein_variant,am_pathogenicity\n"}, "covers 0 positions"),
    ({"values": {**SMALL_VALUES, 5: "0.999"}}, "shared by residues"),
    ({"values": {**SMALL_VALUES, 4: "0.9"}}, "exactly 0.9"),
])
def test_an_upstream_change_raises_instead_of_writing_a_bad_instance(monkeypatch, mutation, message):
    _patch_api(monkeypatch, **mutation)
    with pytest.raises(RuntimeError, match=message):
        generate_gt.build_reference(CASE)


# --------------------------------------------------------------------------
# Prompts and metadata
# --------------------------------------------------------------------------

def test_prompts_name_tools_only_at_b1_and_forbid_other_network_access():
    for level in ("b1", "b2", "b3", "b4"):
        text = (TASK_DIR / f"prompt_{level}.md").read_text()
        assert "result.json" in text and "Do not use web search or web fetch tools" in text
        assert "`curl`/`wget`" in text and "AlphaMissense data" in text and "strictly greater" in text
        assert "{{topic}}" in text and "Leave out models that `entry_ids` does not list" in text
        named = tuple(f"`{tool}`" in text for tool in
                      ("alphafold_get_prediction", "alphafold_get_summary", "alphafold_get_annotations"))
        assert named == ((True,) * 3 if level == "b1" else (False,) * 3), level
        assert ("`alphafold_db` MCP server" in text) == (level in ("b1", "b2"))
        assert "mcp__" not in text and "Claude" not in text and "Codex" not in text


def test_task_is_test_status_and_needs_only_the_standard_library():
    import yaml
    meta = yaml.safe_load((TASK_DIR / "task_meta.yaml").read_text())
    assert meta["status"] == "test" and meta["difficulty"]["requires_network"] is True
    assert meta["runtime"]["packages"] == []
    gates = TASK.eval_config()["gates"]
    assert [g["scorer"] for g in gates] == ["afdb_e2e_schema"] and gates[0]["severity"] == "hard"
    assert sum(item["weight"] for item in TASK.eval_config()["scoring"]) == 100


# --------------------------------------------------------------------------
# Scorers
# --------------------------------------------------------------------------

def test_reference_answer_scores_full_and_numbers_may_be_strings(tmp_path):
    pred, ref = _dirs(tmp_path, _answer())
    assert [_score(n, pred, ref).score for n in ALL] == [1.0] * 4
    assert TASK.total(pred, ref) == 100.0
    spelled = _answer(canonical_length="192", max_am_residue=108.0, max_am_score="0.9990",
                      n_am_above_threshold="61")
    pred, ref = _dirs(tmp_path / "2", spelled)
    assert TASK.total(pred, ref) == 100.0


def test_each_wrong_field_costs_only_its_own_share(tmp_path):
    swapped = [RANKING[1], RANKING[0], *RANKING[2:]]                 # the canonical model is not the longest
    pred, ref = _dirs(tmp_path, _answer(entry_ids_by_length_desc=swapped, longest_entry_id=RANKING[1]))
    assert TASK.total(pred, ref) == pytest.approx(60.0)
    isoforms = _score("afdb_e2e_isoforms", pred, ref, weight=40)
    assert isoforms.details["per_field_correct"] == {"entry_ids_by_length_desc": False, "longest_entry_id": False}
    pred, ref = _dirs(tmp_path / "2", _answer(n_am_above_threshold=62, max_am_score=0.9991))
    profile = _score("afdb_e2e_am_profile", pred, ref, weight=45)
    assert profile.score == pytest.approx(15.0)
    assert profile.details["per_field_correct"] == {"max_am_residue": True, "max_am_score": False,
                                                   "n_am_above_threshold": False}
    # The tool already serves the rounded mean, so a score between two of its values was never
    # reported by it: the scorer must not accept what the verifier rejects.
    pred, ref = _dirs(tmp_path / "3", _answer(max_am_score=0.99904))
    assert _score("afdb_e2e_am_profile", pred, ref).score == pytest.approx(2 / 3)
    row = _run(tmp_path / "4", _stream(), answer=_answer(max_am_score=0.99904))
    assert row["failure"] == "answer_from_tool"


def test_a_ranking_with_an_extra_or_missing_entry_is_wrong(tmp_path):
    for k, ranking in enumerate(([*RANKING, "AF-Q07812-9-F1"], RANKING[:-1])):
        pred, ref = _dirs(tmp_path / str(k), _answer(entry_ids_by_length_desc=ranking))
        assert TASK.total(pred, ref) == pytest.approx(80.0)


def test_entry_ids_are_compared_as_the_tool_spells_them(tmp_path):
    """The scorer must not be more forgiving than the verifier, which compares answers with the
    tool's own spelling."""
    pred, ref = _dirs(tmp_path, _answer(longest_entry_id="af-q07812-2-f1"))
    assert TASK.total(pred, ref) == pytest.approx(80.0)
    row = _run(tmp_path / "2", _stream(), answer=_answer(longest_entry_id="af-q07812-2-f1"))
    assert row["failure"] == "answer_from_tool"


@pytest.mark.parametrize("prediction", [
    None, "not json", [], _answer(entry_ids_by_length_desc=[]), _answer(entry_ids_by_length_desc="AF-Q07812-F1"),
    _answer(entry_ids_by_length_desc=["AF-Q07812-F1", ""]), _answer(longest_entry_id=None),
    _answer(canonical_length=192.5), _answer(max_am_residue=True), _answer(max_am_score="high"),
    _answer(max_am_score=float("nan")), {k: v for k, v in _answer().items() if k != "n_am_above_threshold"},
])
def test_submission_failures_are_valid_zero_scores(tmp_path, prediction):
    pred, ref = _dirs(tmp_path, prediction)
    for name in ALL:
        detail = _score(name, pred, ref)
        assert detail.score == 0.0 and not detail.details.get("scorer_internal_error")


def test_missing_reference_is_an_evaluator_failure(tmp_path):
    pred, ref = _dirs(tmp_path, _answer(), reference=None)
    for name in ALL[1:]:
        assert _score(name, pred, ref).details["scorer_internal_error"] is True
    pred, ref = _dirs(tmp_path / "2", _answer(), reference={"accession": ACCESSION})
    assert _score("afdb_e2e_isoforms", pred, ref).details["failure_kind"] == "missing_evaluator_input"


# --------------------------------------------------------------------------
# Verifier
# --------------------------------------------------------------------------

def test_genuine_three_tool_run_passes(tmp_path):
    row = _run(tmp_path, _stream(extra=[("Bash", {"command": "cat data/profile.json"})]))
    assert row["verdict"] == "PASS", row["checks"]
    assert set(_status(row).values()) == {"PASS"}
    # no call takes another's output, so there is no chain to judge
    assert list(_status(row)) == [c for c in verify.CHECK_ORDER if c != "tool_chain"]
    assert row["tool_sequence"] == ["Bash", PREDICTION, SUMMARY, ANNOTATIONS]


def test_the_accession_alias_is_a_warning_not_a_failure(tmp_path):
    """The tools also accept ``accession`` for ``qualifier``; the result is the same, only the
    input differs from the reference spelling."""
    row = _run(tmp_path, _stream(args={"accession": ACCESSION}))
    assert row["verdict"] == "PASS" and _status(row)["tool_correct"] == "WARN"


def test_a_score_written_with_trailing_zeros_still_comes_from_the_tool(tmp_path):
    row = _run(tmp_path, _stream(), answer=_answer(max_am_score="0.9990", canonical_length="192"))
    assert row["verdict"] == "PASS", row["checks"]


def test_every_tool_is_required(tmp_path):
    for k, tool in enumerate((PREDICTION, SUMMARY, ANNOTATIONS)):
        row = _run(tmp_path / str(k), _stream(skip=(tool,)))
        assert row["failure"] == "tool_called", tool


def test_answer_values_must_come_from_the_tools(tmp_path):
    # the canonical model taken as the longest: a value the tool returned, but not the reference
    row = _run(tmp_path, _stream(), answer=_answer(longest_entry_id="AF-Q07812-F1"))
    assert row["failure"] == "answer_from_tool"
    assert "longest_entry_id: matches reference: False" in row["checks"]["answer_from_tool"]["detail"]
    # the right score, but the annotation the agent saw never carried it (answered from memory)
    row = _run(tmp_path / "2", _stream(annotation_result=_annotation_result(_scores(top_text="0.9987"))))
    assert row["failure"] == "tool_correct"
    assert "max_am_score: matches reference: True, equals a tool-returned value: False" in \
        row["checks"]["answer_from_tool"]["detail"]


def test_in_band_tool_errors_are_not_results(tmp_path):
    empty = json.dumps({"status": "error", "error": "No MUTAGEN annotations available for Q07812",
                        "reason": "empty"})
    row = _run(tmp_path, _stream(annotation_result=empty))
    assert row["failure"] == "tool_correct"
    bad_request = json.dumps({"status": "error", "error": "HTTP 400", "detail": "invalid accession"})
    row = _run(tmp_path / "2", _stream(summary_result=bad_request))
    assert row["failure"] == "tool_correct"


def test_web_fetch_of_alphafold_or_uniprot_is_a_bypass_and_web_search_a_warning(tmp_path):
    for k, url in enumerate((f"{FILES}/AF-Q07812-F1-aa-substitutions.csv",
                             "https://rest.uniprot.org/uniprotkb/Q07812.fasta")):
        row = _run(tmp_path / str(k), _stream(extra=[("WebFetch", {"url": url, "prompt": "?"})]))
        assert _status(row)["no_bypass"] == "FAIL" and row["failure"] == "no_bypass", url
    row = _run(tmp_path / "s", _stream(extra=[("WebSearch", {"query": "BAX isoform lengths"})]))
    assert row["verdict"] == "PASS" and _status(row)["no_bypass"] == "WARN"


@pytest.mark.parametrize("command", [
    f"curl -sO {FILES}/AF-Q07812-F1-aa-substitutions.csv",
    "wget https://alphafold.ebi.ac.uk/api/prediction/Q07812",
    "python3 -c \"import urllib.request; urllib.request.urlopen('https://rest.uniprot.org/uniprotkb/Q07812')\"",
    "python3 - <<'EOF'\nfrom bioservices import UniProt\nEOF",
    "pip install bioservices",
    "uv run --with unipressed python profile.py",
    "zcat AlphaMissense_aa_substitutions.tsv.gz | grep Q07812 > Q07812-aa-substitutions.csv",
])
def test_shell_access_to_alphafold_or_alphamissense_is_a_bypass(tmp_path, command):
    row = _run(tmp_path, _stream(extra=[("Bash", {"command": command})]))
    assert _status(row)["no_bypass"] == "FAIL", command


def test_computing_the_profile_with_python_is_not_a_bypass(tmp_path):
    command = ("python3 -c \"import json; d=json.load(open('ann.json')); "
               "v=[float(r['annotation_value']) for r in d['regions']]; print(max(v), sum(x>0.9 for x in v))\"")
    row = _run(tmp_path, _stream(extra=[("Bash", {"command": command})]))
    assert _status(row)["no_bypass"] == "PASS"


def test_pasting_a_tool_payload_into_a_script_is_not_a_bypass(tmp_path):
    """Counting residues needs code, and a shell cannot see MCP results, so agents paste the
    payload. It carries the server's own URLs (``source_url``, ``amAnnotationsUrl`` ending in
    ``-aa-substitutions.csv``): reading them as a download would accuse a correct agent."""
    payload = json.loads(_annotation_result())["data"]
    heredoc = f"cat > ann.json <<'EOF'\n{json.dumps(payload)}\nEOF\npython3 count.py ann.json"
    row = _run(tmp_path, _stream(extra=[("Bash", {"command": heredoc})]))
    assert _status(row)["no_bypass"] != "FAIL", row["checks"]["no_bypass"]
    models = json.loads(_prediction_result())["data"]
    script = f"import json\nMODELS = {json.dumps(models)}\nprint(sorted(MODELS, key=lambda m: -len(m['sequence'])))\n"
    row = _run(tmp_path / "2", _stream(), files={"rank.py": script})
    assert _status(row)["no_bypass"] != "FAIL", row["checks"]["no_bypass"]
    assert row["verdict"] == "PASS"


def test_reading_an_alphafold_url_with_pandas_is_a_bypass(tmp_path):
    script = f"import pandas as pd\ndf = pd.read_csv('{FILES}/AF-Q07812-F1-aa-substitutions.csv')\n"
    row = _run(tmp_path, _stream(), files={"am.py": script})
    assert _status(row)["no_bypass"] == "FAIL"


def test_per_isoform_prediction_calls_are_a_warning_not_a_failure(tmp_path):
    events = [claude.init("alphafold_db", ["Bash", PREDICTION, SUMMARY, ANNOTATIONS])]
    for k, (entry_id, length) in enumerate(LENGTHS.items()):
        events += claude.call(f"p{k}", PREDICTION, {"qualifier": entry_id[3:-3]},
                              _prediction_result({entry_id: length}))
    events += claude.call("s1", SUMMARY, {"qualifier": ACCESSION}, _summary_result())
    events += claude.call("a1", ANNOTATIONS, {"qualifier": ACCESSION}, _annotation_result())
    events.append(claude.result(11))
    row = _run(tmp_path, jsonl(events))
    assert row["verdict"] == "PASS" and _status(row)["tool_correct"] == "WARN"


def test_structured_content_shown_by_the_client_is_unwrapped(tmp_path):
    def structured(text):
        return json.dumps({"result": text})
    row = _run(tmp_path, _stream(prediction_result=structured(_prediction_result()),
                                 summary_result=structured(_summary_result()),
                                 annotation_result=structured(_annotation_result())))
    assert row["verdict"] == "PASS", row["checks"]


def test_extra_server_tools_warn_about_the_workspace_trap(tmp_path):
    tools = ["Bash", PREDICTION, SUMMARY, ANNOTATIONS, "mcp__alphafold_db__UniProt_get_entry_by_accession"]
    row = _run(tmp_path, _stream(tools=tools))
    assert row["verdict"] == "PASS" and _status(row)["mcp_connected"] == "WARN"
    assert "offered 4 tools, expected 3" in row["checks"]["mcp_connected"]["detail"]


def test_codex_run_passes_and_web_search_items_are_recorded(tmp_path):
    row = _run(tmp_path, _codex(), codex=True)
    assert row["verdict"] == "PASS", row["checks"]
    assert set(_status(row).values()) == {"PASS"}
    web = {"id": "w1", "type": "web_search", "query": "BAX AlphaMissense"}
    row = _run(tmp_path / "2", _codex([web]), codex=True)
    assert row["tool_sequence"][0] == "web_search" and _status(row)["no_bypass"] == "WARN"
