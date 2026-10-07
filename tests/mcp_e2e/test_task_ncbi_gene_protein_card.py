"""ncbi_gene_protein_card (gene search -> gene record of the returned id -> protein record of a GI,
uid-keyed records read with the json_scalars extractor, in-band E-utilities failures, bypass via
Biopython/entrez-direct): generator helpers, scorers, verifier scenarios. No network."""
import json

import pytest

from . import support
from .support import claude, codex, jsonl

TASK = support.Task("mcp_e2e.ncbi_gene_protein_card")
TASK_DIR, TASK_ID, INSTANCE_ID = TASK.dir, TASK.task_id, TASK.instance_id

# seed31415 case (PAH), as generated 2026-10-07
TERM = "PAH[Symbol] AND Homo sapiens[Organism]"
GENE_ID, GI = "5053", "4557819"
REFERENCE = {"symbol": "PAH", "term": TERM, "gene_id": GENE_ID, "gene_ids": [GENE_ID],
             "maplocation": "12q23.2", "chr_accession": "NC_000012.12", "taxid": 9606,
             "protein_accession": "NP_000268", "protein_gi": GI, "protein_gis": [GI],
             "protein_accver": "NP_000268.1", "protein_len": 452}
SEARCH = "mcp__ncbi__NCBIGene_search"
GENE = "mcp__ncbi__NCBIGene_get_summary"
PROTEIN = "mcp__ncbi__NCBIProtein_get_summary"
ALL = ("ncbi_e2e_schema", "ncbi_e2e_gene_card", "ncbi_e2e_protein_card")
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# A minus-strand gene_table (TP53, with the strand marker) and a plus-strand one
# (SOD1, without any marker at all), as efetch returns them.
GENE_TABLE_MINUS = """TP53 tumor protein p53[Homo sapiens]
Gene ID: 7157, updated on 20-Sep-2026

Reference GRCh38.p14 Primary Assembly NC_000017.11  (minus strand) from: 7687490 to: 7668421
mRNA  NM_000546.6, 11 exons,  total annotated spliced exon length: 2591
protein  NP_000537.3 (CCDS11118.1), 11 coding  exons,  annotated AA length: 393
"""
GENE_TABLE_PLUS = """SOD1 superoxide dismutase 1[Homo sapiens]
Gene ID: 6647, updated on 13-Sep-2026

Reference GRCh38.p14 Primary Assembly NC_000021.9  from: 31659693 to: 31668931
mRNA  NM_000454.5, 5 exons,  total annotated spliced exon length: 964
protein  NP_000445.1 (CCDS13614.1), 5 coding  exons,  annotated AA length: 154
"""

generate_gt = TASK.module("generate_gt")
scorer = TASK.module("custom_scorer")
verify = support.verify
_status = support.statuses


def _wrap(payload, endpoint="esummary"):
    """The BaseRESTTool envelope: a successful payload under ``data``, plus the URL used."""
    return json.dumps({"status": "success", "data": payload, "url": f"{EUTILS}/{endpoint}.fcgi"})


def _search_result(idlist=(GENE_ID,), count=None):
    return _wrap({"header": {"type": "esearch", "version": "0.3"},
                  "esearchresult": {"count": str(len(idlist)) if count is None else count,
                                    "retmax": str(len(idlist)), "retstart": "0", "idlist": list(idlist),
                                    "translationset": [],
                                    "querytranslation": '"PAH"[Gene Name] AND "Homo sapiens"[Organism]'}},
                 "esearch")


def _gene_result(gene_id=GENE_ID, **overrides):
    record = {"uid": gene_id, "name": "PAH", "description": "phenylalanine hydroxylase",
              "nomenclaturesymbol": "PAH", "nomenclaturename": "phenylalanine hydroxylase",
              "maplocation": "12q23.2", "otheraliases": "PH, PKU, PKU1", "chromosome": "12",
              "geneweight": 193771, "summary": "The protein encoded by this gene ...",
              "organism": {"scientificname": "Homo sapiens", "commonname": "human", "taxid": 9606},
              "genomicinfo": [{"chrloc": "12", "chraccver": "NC_000012.12", "chrstart": 102958440,
                               "chrstop": 102836888, "exoncount": 13}],
              "mim": ["612349"], "currentid": "", "status": "", **overrides}
    return _wrap({"result": {"uids": [gene_id], gene_id: record}})


def _protein_result(gi=GI, **overrides):
    record = {"uid": gi, "caption": "NP_000268", "title": "phenylalanine-4-hydroxylase [Homo sapiens]",
              "extra": f"gi|{gi}|ref|NP_000268.1|", "accessionversion": "NP_000268.1", "slen": 452,
              "moltype": "aa", "sourcedb": "refseq", "taxid": 9606, "topology": "linear",
              "createdate": "1999/04/01", "updatedate": "2026/09/06", **overrides}
    return _wrap({"result": {"uids": [gi], gi: record}})


def _answer(**overrides):
    return {"gene_id": GENE_ID, "symbol": "PAH", "maplocation": "12q23.2",
            "chr_accession": "NC_000012.12", "protein_accver": "NP_000268.1", "protein_len": 452,
            "taxid": 9606, **overrides}


def _dirs(tmp_path, prediction, reference=REFERENCE):
    return support.score_dirs(tmp_path, prediction, reference)


def _score(name, pred, ref, weight=1.0):
    from ai4sci_bench.core.scorer import get_scorer
    return get_scorer(name).score(pred, ref, {"weight": weight})


def _stream(search_args=None, gene_args=None, protein_args=None, search_result=None, gene_result=None,
            protein_result=None, tools=None, extra=()):
    """Claude stream-json for the three-call chain; extra = [(tool name, input)] before it."""
    events = [claude.init("ncbi", tools or ["Bash", "Read", "Write", "WebFetch", "WebSearch",
                                            SEARCH, GENE, PROTEIN])]
    for k, (name, args) in enumerate(extra):
        events += claude.call(f"x{k}", name, args, "ok")
    events += claude.call("s1", SEARCH, search_args or {"term": TERM}, search_result or _search_result())
    events += claude.call("g1", GENE, gene_args or {"id": GENE_ID}, gene_result or _gene_result())
    events += claude.call("p1", PROTEIN, protein_args or {"id": GI}, protein_result or _protein_result())
    events.append(claude.result(5))
    return jsonl(events)


def _codex(extra_items=()):
    events = [*codex.START, *(codex.item(item) for item in extra_items)]
    events += codex.mcp("m1", "ncbi", "NCBIGene_search", {"term": TERM}, _search_result())
    events += codex.mcp("m2", "ncbi", "NCBIGene_get_summary", {"id": GENE_ID}, _gene_result())
    events += codex.mcp("m3", "ncbi", "NCBIProtein_get_summary", {"id": GI}, _protein_result())
    events.append(codex.done())
    return jsonl(events)


def _run(tmp_path, stream, answer=None, codex=False, files=None):
    return TASK.verify(tmp_path, stream, reference=REFERENCE, answer=answer or _answer(),
                       harness="codex" if codex else "claude", files=files)


# --------------------------------------------------------------------------
# Generator
# --------------------------------------------------------------------------

def test_cases_are_deterministic_and_cover_all_curated_cases():
    assert generate_gt.build_case(31415) == generate_gt.build_case(31415)
    case = generate_gt.build_case(31415)
    assert case["symbol"] == "PAH" and case["term"] == TERM and case["protein"] == "NP_000268"
    assert {generate_gt.build_case(s)["symbol"] for s in range(100)} == {c["symbol"] for c in generate_gt.CASES}


def test_curated_cases_are_single_symbol_queries_with_unversioned_protein_accessions():
    for case in generate_gt.CASES:
        assert case["symbol"].isupper() and case["symbol"].isalnum()
        # the GI is resolved at generation time, so the curated accession must be unversioned:
        # a version suffix would pin a sequence that upstream can revise.
        assert case["protein"].startswith("NP_") and "." not in case["protein"]
        assert case["topic"] and case["symbol"] in case["topic"]
    assert generate_gt.TERM.format(symbol="SOD1") == "SOD1[Symbol] AND Homo sapiens[Organism]"


def test_parse_gene_table_reads_both_strands_and_the_protein_lengths():
    """A plus-strand gene_table carries no strand marker at all, so the orientation comes from
    the coordinates (which carry it either way) and the marker is only cross-checked; requiring
    the marker made every plus-strand gene look like an unparsable reference."""
    minus = generate_gt.parse_gene_table(GENE_TABLE_MINUS)
    assert minus["gene_id"] == "7157" and minus["accession"] == "NC_000017.11"
    assert (minus["strand"], minus["start"], minus["stop"]) == ("minus", 7687490, 7668421)
    assert minus["strand_marker"] == "minus" and minus["protein_lengths"] == {"NP_000537.3": 393}
    plus = generate_gt.parse_gene_table(GENE_TABLE_PLUS)
    assert plus["gene_id"] == "6647" and plus["accession"] == "NC_000021.9"
    assert (plus["strand"], plus["start"], plus["stop"]) == ("plus", 31659693, 31668931)
    assert plus["strand_marker"] is None and plus["protein_lengths"] == {"NP_000445.1": 154}
    assert plus["title"] == "SOD1 superoxide dismutase 1[Homo sapiens]"
    contradictory = generate_gt.parse_gene_table(
        GENE_TABLE_PLUS.replace("NC_000021.9  from", "NC_000021.9  (minus strand) from"))
    assert (contradictory["strand"], contradictory["strand_marker"]) == ("plus", "minus")
    empty = generate_gt.parse_gene_table("Gene ID: 1\n")
    assert empty["accession"] is None and empty["start"] is None and empty["protein_lengths"] == {}


def test_normalise_locus_shifts_the_0_based_coordinates_and_labels_the_strand():
    """esummary is 0-based and expresses the minus strand by start > stop; the gene_table is
    1-based and names the strand, so both ends shift by +1."""
    minus = generate_gt.normalise_locus({"chraccver": "NC_000012.12", "chrstart": 102958440,
                                         "chrstop": 102836888})
    assert minus == {"accession": "NC_000012.12", "strand": "minus", "start": 102958441, "stop": 102836889}
    plus = generate_gt.normalise_locus({"chraccver": "NC_000021.9", "chrstart": 31659692,
                                        "chrstop": 31668930})
    assert plus == {"accession": "NC_000021.9", "strand": "plus", "start": 31659693, "stop": 31668931}
    assert generate_gt.normalise_locus({"chraccver": "NC_1.1"})["start"] is None


def test_fasta_record_counts_residues_of_the_first_record_only():
    text = ">NP_000268.1 phenylalanine-4-hydroxylase [Homo sapiens]\nMSTAVLENPG\nLGRKLSDFGQ\n>NP_X.1 other\nAAAA\n"
    assert generate_gt.fasta_record(text) == ("NP_000268.1 phenylalanine-4-hydroxylase [Homo sapiens]", 20)
    assert generate_gt.fasta_record("") == ("", 0)


# --- the reference, built offline from fixtures of the raw endpoints ---------
# The generator talks to E-utilities directly, so these payloads carry no
# {"status": "success", "data": …} envelope — that one belongs to the MCP wrapper.

PAH_GENE_TABLE = """PAH phenylalanine hydroxylase[Homo sapiens]
Gene ID: 5053, updated on 20-Sep-2026

Reference GRCh38.p14 Primary Assembly NC_000012.12  (minus strand) from: 102958441 to: 102836889
mRNA  NM_000277.3, 13 exons,  total annotated spliced exon length: 2680
protein  NP_000268.1 (CCDS9032.1), 13 coding  exons,  annotated AA length: 452
"""
PAH_FASTA = ">NP_000268.1 phenylalanine-4-hydroxylase [Homo sapiens]\n" + "M" * 452 + "\n"
PAH_GENE_RECORD = {"uid": GENE_ID, "nomenclaturesymbol": "PAH", "maplocation": "12q23.2",
                   "otheraliases": "PH, PKU, PKU1", "summary": "The protein encoded ...",
                   "organism": {"scientificname": "Homo sapiens", "taxid": 9606},
                   "genomicinfo": [{"chrloc": "12", "chraccver": "NC_000012.12", "chrstart": 102958440,
                                    "chrstop": 102836888, "exoncount": 13}], "mim": ["612349"]}
PAH_PROTEIN_RECORD = {"uid": GI, "caption": "NP_000268", "accessionversion": "NP_000268.1",
                      "title": "phenylalanine-4-hydroxylase [Homo sapiens]", "slen": 452,
                      "moltype": "aa", "sourcedb": "refseq", "taxid": 9606}
PAH_DATASETS = {"map_locations": [{"map_type": "Cytogenetic", "map_value": "12q23.2"}],
                "omim_ids": ["612349"], "orientation": "minus"}


def _patch_endpoints(monkeypatch, *, gene_table=PAH_GENE_TABLE, fasta=PAH_FASTA, gene=None, protein=None,
                     search=None, datasets=PAH_DATASETS):
    gene = {**PAH_GENE_RECORD, **(gene or {})}
    protein = {**PAH_PROTEIN_RECORD, **(protein or {})}
    searches = {"gene": {"count": "1", "idlist": [GENE_ID]}, "protein": {"count": "1", "idlist": [GI]},
                **(search or {})}
    records = {"gene": (GENE_ID, gene), "protein": (GI, protein)}

    def eutils_json(endpoint, **params):
        db = params["db"]
        if endpoint == "esearch":
            return {"esearchresult": searches[db]}
        uid, record = records[db]
        return {"result": {"uids": [uid], uid: record}}

    monkeypatch.setattr(generate_gt, "eutils_json", eutils_json)
    monkeypatch.setattr(generate_gt, "eutils_text",
                        lambda endpoint, **params: gene_table if params["db"] == "gene" else fasta)
    monkeypatch.setattr(generate_gt, "datasets_gene", lambda gene_id: datasets)


def test_reference_carries_every_key_the_verifier_and_the_scorers_read(monkeypatch):
    _patch_endpoints(monkeypatch)
    reference = generate_gt.build_reference(generate_gt.build_case(31415))
    for key, value in REFERENCE.items():
        assert reference[key] == value, key
    spec = verify.parse_spec(json.loads((TASK_DIR / "e2e_check.json").read_text()))
    needed = {cs.result.reference_key for cs in spec.calls} | {a.reference_key for a in spec.answers}
    needed |= {key for cs in spec.calls for key in cs.inputs_from_reference.values()}
    assert needed <= set(reference), sorted(needed - set(reference))
    checks = reference["cross_checks"]
    assert checks["gene_table_locus"] == checks["esummary_locus_1based"]
    assert checks["gene_table_annotated_aa_length"] == 452
    assert checks["datasets"]["band_agrees"] and checks["datasets"]["orientation_agrees"]
    assert checks["datasets"]["omim_agrees"]


@pytest.mark.parametrize("mutation,message", [
    ({"search": {"gene": {"count": "2", "idlist": [GENE_ID, "7157"]}}}, "no longer unique"),
    ({"gene": {"nomenclaturesymbol": "PAH1"}}, "nomenclaturesymbol"),
    ({"gene": {"maplocation": ""}}, "maplocation"),
    ({"gene": {"organism": {"taxid": 10090}}}, "organism.taxid"),
    ({"gene": {"genomicinfo": [PAH_GENE_RECORD["genomicinfo"][0],
                               {"chraccver": "NT_187361.1", "chrstart": 1, "chrstop": 2}]}},
     "genomicinfo entries"),
    ({"gene": {"genomicinfo": [{"chraccver": "NC_000012.12", "chrstart": 102836888,
                                "chrstop": 102958440}]}}, "locus"),
    ({"protein": {"accessionversion": "NP_999999.1"}}, "not a version of"),
    ({"protein": {"slen": 451}}, "slen"),
    ({"protein": {"moltype": "nt"}}, "moltype"),
    ({"protein": {"title": "something else"}}, "FASTA header"),
    ({"gene_table": PAH_GENE_TABLE.replace("Gene ID: 5053", "Gene ID: 7157")}, "reports Gene ID"),
    ({"gene_table": PAH_GENE_TABLE.replace("(minus strand)", "(plus strand)")}, "coordinates run"),
    ({"gene_table": PAH_GENE_TABLE.replace("annotated AA length: 452", "annotated AA length: 451")},
     "gene_table annotates"),
    ({"gene_table": "PAH\nGene ID: 5053\n"}, "no reference-assembly locus"),
    ({"gene_table": PAH_GENE_TABLE.replace("protein  NP_000268.1 (CCDS9032.1), 13 coding  exons,  "
                                           "annotated AA length: 452", "")}, "annotated AA length"),
])
def test_an_upstream_change_raises_instead_of_writing_a_bad_instance(monkeypatch, mutation, message):
    _patch_endpoints(monkeypatch, **mutation)
    with pytest.raises(RuntimeError, match=message):
        generate_gt.build_reference(generate_gt.build_case(31415))


def test_an_unavailable_datasets_api_is_recorded_not_fatal(monkeypatch):
    """Datasets is v2alpha and only a soft cross-check: it must never stop generation."""
    _patch_endpoints(monkeypatch)
    monkeypatch.setattr(generate_gt, "datasets_gene",
                        lambda gene_id: (_ for _ in ()).throw(RuntimeError("HTTP 503")))
    reference = generate_gt.build_reference(generate_gt.build_case(31415))
    assert reference["cross_checks"]["datasets"] == {"unavailable": "HTTP 503"}
    assert reference["maplocation"] == "12q23.2"
    _patch_endpoints(monkeypatch, datasets={"map_locations": [], "omim_ids": None})
    assert generate_gt.build_reference(generate_gt.build_case(31415))["maplocation"] == "12q23.2"


def test_a_gene_table_without_a_strand_marker_is_accepted(monkeypatch):
    """The live plus-strand tables have no marker, so its absence must be ordinary."""
    _patch_endpoints(monkeypatch, gene_table=PAH_GENE_TABLE.replace("(minus strand) ", ""))
    reference = generate_gt.build_reference(generate_gt.build_case(31415))
    assert reference["cross_checks"]["gene_table_locus"]["strand"] == "minus"      # from the coordinates
    assert reference["chr_accession"] == "NC_000012.12"


# --------------------------------------------------------------------------
# Prompts and metadata
# --------------------------------------------------------------------------

def test_prompts_name_tools_only_at_b1_and_forbid_other_network_access():
    for level in ("b1", "b2", "b3", "b4"):
        text = (TASK_DIR / f"prompt_{level}.md").read_text()
        assert "result.json" in text and "Do not use web search or web fetch tools" in text
        assert "`curl`/`wget`" in text and "Bio.Entrez" in text and "datasets" in text
        assert "{{topic}}" in text
        named = ("`NCBIGene_search`" in text, "`NCBIGene_get_summary`" in text,
                 "`NCBIProtein_get_summary`" in text)
        assert named == ((True, True, True) if level == "b1" else (False, False, False)), level
        assert ("`ncbi` MCP server" in text) == (level in ("b1", "b2"))
        assert "mcp__" not in text and "Claude" not in text and "Codex" not in text


def test_task_is_test_status_and_needs_only_the_standard_library():
    import yaml
    meta = yaml.safe_load((TASK_DIR / "task_meta.yaml").read_text())
    assert meta["status"] == "test" and meta["difficulty"]["requires_network"] is True
    assert meta["runtime"]["packages"] == []
    gates = TASK.eval_config()["gates"]
    # schema only, like the other fake tasks: a wrong identifier must not zero the instance
    assert [g["scorer"] for g in gates] == ["ncbi_e2e_schema"] and gates[0]["severity"] == "hard"


# --------------------------------------------------------------------------
# Scorers
# --------------------------------------------------------------------------

def test_reference_answer_scores_full_and_numbers_may_be_strings(tmp_path):
    pred, ref = _dirs(tmp_path, _answer())
    assert [_score(n, pred, ref).score for n in ALL] == [1.0] * 3
    assert TASK.total(pred, ref) == 100.0
    spelled = _answer(gene_id=5053, protein_len="452", taxid="9606")
    pred, ref = _dirs(tmp_path / "2", spelled)
    assert TASK.total(pred, ref) == 100.0


def test_identifiers_are_compared_as_the_record_spells_them(tmp_path):
    """The scorer must not be more forgiving than the verifier: ``answer_from_tool`` compares
    the answer with the tool's own spelling (``canon_scalar`` is case-sensitive), so a scorer
    that accepted ``np_000268.1`` would call a submission perfect that the run reports as not
    coming from the tool."""
    recased = _answer(symbol="pah", maplocation="12Q23.2", chr_accession="nc_000012.12",
                      protein_accver="np_000268.1")
    pred, ref = _dirs(tmp_path, recased)
    assert TASK.total(pred, ref) == pytest.approx(44.0)      # gene_id + taxid + protein_len survive
    row = _run(tmp_path / "2", _stream(), answer=recased)
    assert row["failure"] == "answer_from_tool"


def test_each_wrong_field_costs_only_its_own_share(tmp_path):
    pred, ref = _dirs(tmp_path, _answer(maplocation="12q24.1", protein_len=453))
    gene = _score("ncbi_e2e_gene_card", pred, ref, weight=60)
    protein = _score("ncbi_e2e_protein_card", pred, ref, weight=40)
    assert gene.score == pytest.approx(48.0) and not gene.passed
    assert gene.details["per_field_correct"] == {"gene_id": True, "symbol": True, "maplocation": False,
                                                "chr_accession": True, "taxid": True}
    assert protein.score == pytest.approx(20.0)
    # the schema gate still passes: a wrong value is not a malformed submission
    assert _score("ncbi_e2e_schema", pred, ref).passed


def test_the_gate_is_structural_so_an_unversioned_accession_keeps_the_other_credit(tmp_path):
    """The hard gate asks for the seven keys with usable types and nothing else: an accession
    without its version and a nonsensical length are wrong values, and zeroing the instance for
    them would hide what the rest of the card got right."""
    for prediction, total in ((_answer(protein_accver="NP_000268"), 80.0),
                              (_answer(protein_len=0), 80.0),
                              (_answer(gene_id=-1), 88.0)):
        pred, ref = _dirs(tmp_path / str(total) / str(prediction["protein_accver"]), prediction)
        assert _score("ncbi_e2e_schema", pred, ref).passed, prediction
        assert TASK.total(pred, ref) == pytest.approx(total), prediction


@pytest.mark.parametrize("prediction", [None, "not json", [], _answer(gene_id=""), _answer(gene_id="PAH"),
                                        _answer(symbol=""), _answer(protein_len=452.5),
                                        _answer(protein_len=True), _answer(taxid=None),
                                        _answer(maplocation=12), _answer(chr_accession=["NC_000012.12"]),
                                        {k: v for k, v in _answer().items() if k != "taxid"}])
def test_submission_failures_are_valid_zero_scores(tmp_path, prediction):
    pred, ref = _dirs(tmp_path, prediction)
    for name in ALL:
        detail = _score(name, pred, ref)
        assert detail.score == 0.0 and not detail.details.get("scorer_internal_error")


def test_missing_reference_is_an_evaluator_failure(tmp_path):
    pred, ref = _dirs(tmp_path, _answer(), reference=None)
    for name in ALL[1:]:
        assert _score(name, pred, ref).details["scorer_internal_error"] is True
    pred, ref = _dirs(tmp_path / "2", _answer(), reference={"symbol": "PAH"})
    assert _score("ncbi_e2e_gene_card", pred, ref).details["failure_kind"] == "missing_evaluator_input"


# --------------------------------------------------------------------------
# Verifier
# --------------------------------------------------------------------------

def test_genuine_three_call_run_passes(tmp_path):
    row = _run(tmp_path, _stream(extra=[("Bash", {"command": "cat data/card.json"})]))
    assert row["verdict"] == "PASS", row["checks"]
    assert set(_status(row).values()) == {"PASS"}
    assert list(_status(row)) == list(verify.CHECK_ORDER)
    assert row["tool_sequence"] == ["Bash", SEARCH, GENE, PROTEIN]


def test_uid_keyed_records_are_read_without_a_dotted_path(tmp_path):
    """``data.result.<uid>.slen`` cannot be written as a static path, so the answers read the
    scalar leaves of the record instead; a value from the right record is still required."""
    row = _run(tmp_path, _stream())
    detail = row["checks"]["answer_from_tool"]["detail"]
    assert "equals a tool-returned value: False" not in detail
    scalars = verify.spec.Selector(extract="json_scalars")
    protein = verify.evidence.ToolCall(result_text=_protein_result())
    values = verify.values.read(protein, scalars)
    assert "452" in values and "NP_000268.1" in values and "4557819" in values
    # a number the other record carries is not in this one
    assert "102958440" not in values


def test_gene_record_read_for_an_id_the_search_did_not_return_breaks_the_chain(tmp_path):
    row = _run(tmp_path, _stream(search_result=_search_result(["7157"]), gene_args={"id": GENE_ID}))
    assert row["failure"] in ("tool_correct", "tool_chain"), row["checks"]
    assert _status(row)["tool_chain"] == "FAIL"


def test_a_gene_read_by_symbol_without_searching_is_not_chained(tmp_path):
    # NCBIGene_get_summary only takes uids, so a "symbol" argument returns nothing usable
    row = _run(tmp_path, _stream(gene_args={"id": "PAH"}))
    assert _status(row)["tool_chain"] == "FAIL"


def test_answer_values_must_come_from_the_tools(tmp_path):
    # The record of the right uid, but with another gene's band (e.g. the agent kept a value it
    # knew and the tool said something else): the per-call check only pins the record identity,
    # so this has to surface as a provenance failure of the answer.
    row = _run(tmp_path, _stream(gene_result=_gene_result(maplocation="17p13.1")))
    assert row["failure"] == "answer_from_tool"
    assert _status(row)["tool_correct"] == "PASS"
    assert "maplocation: matches reference: True, equals a tool-returned value: False" in \
        row["checks"]["answer_from_tool"]["detail"]
    row = _run(tmp_path / "2", _stream(), answer=_answer(protein_len=451))
    assert row["failure"] == "answer_from_tool"
    assert "matches reference: False" in row["checks"]["answer_from_tool"]["detail"]


def test_in_band_eutils_failures_are_not_results(tmp_path):
    """E-utilities answers HTTP 200 for a bad request; the three in-band failure shapes must all
    read as a failed call, never as a record."""
    wrapper_error = json.dumps({"status": "error", "error": "HTTP 400 from esummary", "status_code": 400})
    top_level = _wrap({"error": "UID=99999999: cannot get document summary"})
    per_uid = _wrap({"result": {"uids": ["99999999"], "99999999": {"uid": "99999999", "error":
                                                                   "cannot get document summary"}}})
    empty = _wrap({"result": {"uids": []}})
    for payload in (wrapper_error, top_level, per_uid, empty):
        row = _run(tmp_path / str(hash(payload)), _stream(gene_result=payload))
        assert row["failure"] == "tool_correct", payload[:80]


def test_nothing_found_search_fails_the_call(tmp_path):
    not_found = _wrap({"esearchresult": {"count": "0", "retmax": "0", "idlist": [],
                                         "errorlist": {"phrasesnotfound": ["ZZZNOTAGENE"]}}}, "esearch")
    row = _run(tmp_path, _stream(search_result=not_found))
    assert _status(row)["tool_correct"] == "FAIL" and _status(row)["tool_chain"] == "FAIL"


def test_web_fetch_of_ncbi_is_a_bypass_and_web_search_a_warning(tmp_path):
    fetch = ("WebFetch", {"url": "https://www.ncbi.nlm.nih.gov/gene/5053", "prompt": "map location?"})
    row = _run(tmp_path, _stream(extra=[fetch]))
    assert _status(row)["no_bypass"] == "FAIL" and row["failure"] == "no_bypass"
    row = _run(tmp_path / "2", _stream(extra=[("WebSearch", {"query": "PAH gene cytogenetic band"})]))
    assert row["verdict"] == "PASS" and _status(row)["no_bypass"] == "WARN"
    # a fetch that has nothing to do with NCBI stays a warning
    row = _run(tmp_path / "3", _stream(extra=[("WebFetch", {"url": "https://docs.python.org/3/library/json.html"})]))
    assert row["verdict"] == "PASS" and _status(row)["no_bypass"] == "WARN"


@pytest.mark.parametrize("command", [
    "curl -s 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=gene&id=5053'",
    "python3 -c \"import urllib.request; urllib.request.urlopen('https://eutils.ncbi.nlm.nih.gov/x')\"",
    "pip install biopython",
    "uv run --with biopython python card.py",
    "python3 - <<'EOF'\nfrom Bio import Entrez\nEOF",
    "esearch -db gene -query 'PAH[Symbol]' | efetch -format docsum",
    "datasets summary gene symbol PAH --taxon human",
])
def test_shell_access_to_ncbi_is_a_bypass(tmp_path, command):
    row = _run(tmp_path, _stream(extra=[("Bash", {"command": command})]))
    assert _status(row)["no_bypass"] == "FAIL", command


def test_writing_the_answer_with_python_is_not_a_bypass(tmp_path):
    command = "python3 -c \"import json; json.dump({'gene_id': '5053'}, open('result.json','w'))\""
    row = _run(tmp_path, _stream(extra=[("Bash", {"command": command})]))
    assert _status(row)["no_bypass"] == "PASS"


def test_extra_server_tools_warn_about_the_workspace_trap(tmp_path):
    tools = ["Bash", SEARCH, GENE, PROTEIN, "mcp__ncbi__UniProt_get_entry_by_accession", "mcp__ncbi__Finish"]
    row = _run(tmp_path, _stream(tools=tools))
    assert row["verdict"] == "PASS" and _status(row)["mcp_connected"] == "WARN"
    assert "offered 5 tools, expected 3" in row["checks"]["mcp_connected"]["detail"]


def test_codex_run_passes_and_web_search_items_are_recorded(tmp_path):
    row = _run(tmp_path, _codex(), codex=True)
    assert row["verdict"] == "PASS", row["checks"]
    assert set(_status(row).values()) == {"PASS"}
    web = {"id": "w1", "type": "web_search", "query": "PAH gene id"}
    row = _run(tmp_path / "2", _codex([web]), codex=True)
    assert row["tool_sequence"][0] == "web_search" and _status(row)["no_bypass"] == "WARN"
