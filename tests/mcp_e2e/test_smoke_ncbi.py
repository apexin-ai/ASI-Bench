"""e2e_smoke/servers/ncbi.py without network: wrapper unwrapping, gene_table parsing, the
0-based -> 1-based locus shift, FASTA counting, and the search/summary checks.

The fixtures are trimmed copies of live payloads (E-utilities, 2026-10-03).
"""

import pytest

from . import support
from .support import StubClient, report_statuses, rpc_json, rpc_text, runner

ncbi = support.smoke_module("ncbi")

ncbi.THROTTLE.gap = 0.0

GENE_TABLE = """TP53 tumor protein p53[Homo sapiens]
Gene ID: 7157, updated on 15-Sep-2026


Reference GRCh38.p14 Primary Assembly NC_000017.11  (minus strand) from: 7687490 to: 7668421
RNA transcript variant 14 NR_176326.1, 10 exons,  total annotated spliced exon length: 2399
"""
PLUS_STRAND_TABLE = """SOD1 superoxide dismutase 1[Homo sapiens]
Gene ID: 6647, updated on 13-Sep-2026

Reference GRCh38.p14 Primary Assembly NC_000021.9  from: 31659693 to: 31668931
"""
GENE_RECORD = {"uid": "7157", "name": "TP53", "description": "tumor protein p53",
               "nomenclaturesymbol": "TP53", "nomenclaturename": "tumor protein p53",
               "maplocation": "17p13.1", "chromosome": "17", "mim": ["191170"],
               "otheraliases": "BCC7, BMFS5, LFS1, P53, TRP53", "geneweight": 846405,
               "summary": "This gene encodes a tumor suppressor protein.",
               "organism": {"scientificname": "Homo sapiens", "commonname": "human", "taxid": 9606},
               "genomicinfo": [{"chrloc": "17", "chraccver": "NC_000017.11", "chrstart": 7687489,
                                "chrstop": 7668420, "exoncount": 13}]}
DATASETS = {"gene_id": 7157, "symbol": "TP53", "orientation": "minus", "omim_ids": ["191170"],
            "synonyms": ["P53", "BCC7", "LFS1", "BMFS5", "TRP53"],
            "map_locations": [{"map_type": "Cytogenetic", "map_value": "17p13.1"}],
            "nomenclature_authority": {"authority": "HGNC", "identifier": "HGNC:11998"},
            "swiss_prot_accessions": ["P04637"]}
PROTEIN_RECORD = {"uid": "119395750", "accessionversion": "NP_006112.3", "caption": "NP_006112",
                  "title": "keratin, type II cytoskeletal 1 [Homo sapiens]", "slen": 644, "moltype": "aa",
                  "sourcedb": "refseq", "taxid": 9606, "createdate": "1999/06/24",
                  "updatedate": "2026/02/02"}
FASTA = ">NP_006112.3 keratin, type II cytoskeletal 1 [Homo sapiens]\n" + "A" * 70 + "\n" + "A" * 574 + "\n"


def wrapped(data):
    """A payload as BaseRESTTool returns it."""
    return {"status": "success", "data": data, "url": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"}


def esearch(idlist, count=None, **extra):
    return wrapped({"header": {"type": "esearch"},
                    "esearchresult": {"count": str(len(idlist)) if count is None else count,
                                      "retmax": str(len(idlist)), "retstart": "0", "idlist": idlist,
                                      "querytranslation": "TP53[Symbol] AND Homo sapiens[Organism]", **extra}})


def esummary(records):
    return wrapped({"header": {"type": "esummary"},
                    "result": {"uids": [r["uid"] for r in records], **{r["uid"]: r for r in records}}})


def session_for(responses, *, checkout=None):
    report = runner.Report()
    client = StubClient(responses)
    session = runner.Session(ncbi.SMOKE, None, {}, {}, checkout, report, None, None, None, None, {})
    session.client, session.call = client, runner.Caller(client, report)
    return session


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------

def test_unwrap_separates_data_from_in_band_errors():
    assert ncbi.unwrap(esearch(["7157"]))[0]["esearchresult"]["idlist"] == ["7157"]
    assert ncbi.unwrap({"status": "error", "error": "NCBI API error", "status_code": 502})[1] == \
        "NCBI API error (HTTP 502)"
    assert ncbi.unwrap({"esearchresult": {}})[1] is None     # a bare payload is not an error


def test_parse_gene_table_reads_id_title_and_a_minus_strand_locus():
    parsed = ncbi.parse_gene_table(GENE_TABLE)
    assert parsed == {"gene_id": "7157", "title": "TP53 tumor protein p53[Homo sapiens]",
                      "accession": "NC_000017.11", "strand": "minus", "start": 7687490, "stop": 7668421,
                      "strand_marker": "minus"}
    assert ncbi.gene_table_symbol(parsed["title"]) == "TP53"
    # A plus-strand gene carries no strand marker at all (measured 2026-10-07 on SOD1, CFTR,
    # HBA1), so the orientation comes from the coordinates and the marker is only a cross-check.
    plus = ncbi.parse_gene_table(PLUS_STRAND_TABLE)
    assert (plus["accession"], plus["strand"], plus["start"], plus["stop"], plus["strand_marker"]) == \
        ("NC_000021.9", "plus", 31659693, 31668931, None)
    spelled_out = ncbi.parse_gene_table(PLUS_STRAND_TABLE.replace("NC_000021.9  from",
                                                                 "NC_000021.9  (plus strand) from"))
    assert spelled_out == {**plus, "strand_marker": "plus"}
    contradictory = ncbi.parse_gene_table(PLUS_STRAND_TABLE.replace("NC_000021.9  from",
                                                                   "NC_000021.9  (minus strand) from"))
    assert (contradictory["strand"], contradictory["strand_marker"]) == ("plus", "minus")
    empty = ncbi.parse_gene_table("")
    assert empty["gene_id"] is None and empty["start"] is None and empty["strand_marker"] is None


def test_normalise_locus_shifts_0_based_coordinates_and_labels_the_strand():
    info = GENE_RECORD["genomicinfo"][0]
    assert ncbi.normalise_locus(info) == {"accession": "NC_000017.11", "strand": "minus",
                                          "start": 7687490, "stop": 7668421}
    plus = ncbi.normalise_locus({"chraccver": "NC_000017.11", "chrstart": 43044294, "chrstop": 43125482})
    assert plus == {"accession": "NC_000017.11", "strand": "plus", "start": 43044295, "stop": 43125483}
    assert ncbi.normalise_locus({"chraccver": "X"})["start"] is None


def test_normalised_locus_matches_the_gene_table_locus():
    assert ncbi.locus_diffs(ncbi.normalise_locus(GENE_RECORD["genomicinfo"][0]),
                            ncbi.parse_gene_table(GENE_TABLE)) == []
    # The classic mistake: comparing the 0-based values directly.
    raw = {"accession": "NC_000017.11", "strand": "minus", "start": 7687489, "stop": 7668420}
    assert len(ncbi.locus_diffs(raw, ncbi.parse_gene_table(GENE_TABLE))) == 2


def test_fasta_record_counts_residues_of_the_first_record():
    assert ncbi.fasta_record(FASTA) == ("NP_006112.3 keratin, type II cytoskeletal 1 [Homo sapiens]", 644)
    assert ncbi.fasta_record(">a\nAA\n>b\nAAAA\n") == ("a", 2)
    assert ncbi.fasta_record("") == ("", 0)


def test_search_not_found_detects_the_in_band_empty_result():
    data = ncbi.unwrap(esearch([], count="0", errorlist={"phrasesnotfound": ["ZZZ[Symbol]"],
                                                        "fieldsnotfound": []}))[0]
    assert "phrasesnotfound=['ZZZ[Symbol]']" in ncbi.search_not_found(data)
    assert ncbi.search_not_found(ncbi.unwrap(esearch(["7157"]))[0]) is None
    assert ncbi.search_not_found({}) is None


@pytest.mark.parametrize("payload,expected", [
    ({"error": "Invalid uid ZZZ at position= 0", "result": {"uids": []}}, "error="),
    ({"result": {"uids": ["99999999"], "99999999": {"uid": "99999999", "error": "cannot get document summary"}}},
     "cannot get document summary"),
    ({"result": {"uids": []}}, "result.uids is empty"),
])
def test_summary_not_found_covers_all_three_in_band_shapes(payload, expected):
    assert expected in ncbi.summary_not_found(payload)


def test_summary_not_found_is_silent_for_a_good_record():
    assert ncbi.summary_not_found(ncbi.unwrap(esummary([GENE_RECORD]))[0]) is None


def test_aliases_of_splits_the_alias_string():
    assert ncbi.aliases_of(GENE_RECORD) == {"BCC7", "BMFS5", "LFS1", "P53", "TRP53"}
    assert ncbi.aliases_of({}) == set()


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def test_symbol_search_passes_against_the_gene_table(monkeypatch):
    monkeypatch.setattr(ncbi, "reference_gene_table", lambda gene_id: ncbi.parse_gene_table(GENE_TABLE))
    session = session_for({ncbi.GENE_SEARCH: esearch_response(["7157"])})
    ncbi.check_symbol_search(session)
    check = session.report.checks[-1]
    assert check["status"] == "PASS" and "7157" in check["detail"]


def esearch_response(idlist, **extra):
    return rpc_json(esearch(idlist, **extra))


@pytest.mark.parametrize("response,expected", [
    (esearch_response(["7157", "22059"]), "idlist"),
    (esearch_response(["7157"], count="2"), "count"),
])
def test_symbol_search_fails_on_a_different_result(monkeypatch, response, expected):
    monkeypatch.setattr(ncbi, "reference_gene_table", lambda gene_id: ncbi.parse_gene_table(GENE_TABLE))
    session = session_for({ncbi.GENE_SEARCH: response})
    ncbi.check_symbol_search(session)
    check = session.report.checks[-1]
    assert check["status"] == "FAIL" and expected in check["detail"]


def test_symbol_search_fails_loudly_when_the_reference_is_wrong(monkeypatch):
    monkeypatch.setattr(ncbi, "reference_gene_table", lambda gene_id: {"gene_id": "1", "title": ""})
    session = session_for({ncbi.GENE_SEARCH: esearch_response(["7157"])})
    ncbi.check_symbol_search(session)
    check = session.report.checks[-1]
    assert check["status"] == "FAIL" and "fix the smoke inputs" in check["detail"]


def test_retmax_bounds_the_list_and_retmax_zero_warns():
    responses = iter([esearch(["1956", "2064", "7157"], count="2272"),
                      esearch([], count="2272")])
    session = session_for({ncbi.GENE_SEARCH: lambda arguments: rpc_json(next(responses))})
    ncbi.check_retmax(session)
    statuses = report_statuses(session.report)
    assert statuses[f"{ncbi.GENE_SEARCH}[retmax={ncbi.FREE_TEXT_RETMAX}]"] == "PASS"
    assert statuses[f"{ncbi.GENE_SEARCH}[retmax=0]"] == "WARN"
    assert session.client.calls[0][1] == {"term": ncbi.FREE_TEXT_TERM, "retmax": ncbi.FREE_TEXT_RETMAX}
    assert session.client.calls[1][1]["retmax"] == 0


def test_retmax_zero_dropped_as_falsy_is_reported():
    """The realistic wrapper defect: a falsy retmax is dropped and the default applies."""
    responses = iter([esearch(["1956", "2064", "7157"], count="2272"),
                      esearch([str(i) for i in range(20)], count="2272")])
    session = session_for({ncbi.GENE_SEARCH: lambda arguments: rpc_json(next(responses))})
    ncbi.check_retmax(session)
    check = session.report.checks[-1]
    assert check["status"] == "WARN" and "did not bound the list" in check["detail"]


def test_retmax_fails_when_the_bound_is_ignored():
    responses = iter([esearch(["1", "2", "3", "4"], count="2272"), esearch([], count="2272")])
    session = session_for({ncbi.GENE_SEARCH: lambda arguments: rpc_json(next(responses))})
    ncbi.check_retmax(session)
    assert session.report.checks[0]["status"] == "FAIL"
    assert "4 ids returned" in session.report.checks[0]["detail"]


def _patch_references(monkeypatch, table=GENE_TABLE, datasets=DATASETS):
    monkeypatch.setattr(ncbi, "reference_gene_table", lambda gene_id: ncbi.parse_gene_table(table))
    monkeypatch.setattr(ncbi, "reference_datasets", lambda gene_id: datasets)

def test_gene_summary_passes_with_the_coordinate_shift(monkeypatch):
    _patch_references(monkeypatch)
    session = session_for({ncbi.GENE_SUMMARY: rpc_json(esummary([GENE_RECORD]))})
    ncbi.check_gene_summary(session)
    check = session.report.checks[-1]
    assert check["status"] == "PASS"
    assert check["normalised_locus"] == {"accession": "NC_000017.11", "strand": "minus",
                                        "start": 7687490, "stop": 7668421}
    assert check["esummary_locus_0based"] == {"chrstart": 7687489, "chrstop": 7668420}
    assert check["reported_only"]["exoncount"] == 13


@pytest.mark.parametrize("mutation,expected", [
    ({"nomenclaturesymbol": "TP63"}, "nomenclaturesymbol"),
    ({"organism": {"taxid": 10090}}, "organism.taxid"),
    ({"genomicinfo": [{"chraccver": "NC_000017.10", "chrstart": 7687489, "chrstop": 7668420}]}, "accession"),
    ({"genomicinfo": [{"chraccver": "NC_000017.11", "chrstart": 7668420, "chrstop": 7687489}]}, "strand"),
    ({"genomicinfo": [{"chraccver": "NC_000017.11", "chrstart": 7687488, "chrstop": 7668420}]}, "start"),
])
def test_gene_summary_fails_on_a_wrong_field(monkeypatch, mutation, expected):
    _patch_references(monkeypatch)
    session = session_for({ncbi.GENE_SUMMARY: rpc_json(esummary([{**GENE_RECORD, **mutation}]))})
    ncbi.check_gene_summary(session)
    check = session.report.checks[0]
    assert check["status"] == "FAIL" and expected in check["detail"]


@pytest.mark.parametrize("mutation", [{"maplocation": "17q21.31"}, {"mim": ["999999"]}])
def test_gene_summary_only_warns_when_the_alpha_datasets_reference_disagrees(monkeypatch, mutation):
    """Datasets is v2alpha: its disagreement is reported, never a FAIL."""
    _patch_references(monkeypatch)
    session = session_for({ncbi.GENE_SUMMARY: rpc_json(esummary([{**GENE_RECORD, **mutation}]))})
    ncbi.check_gene_summary(session)
    assert session.report.checks[0]["status"] == "WARN"


def test_gene_summary_warns_when_the_gene_table_marker_contradicts_its_own_coordinates(monkeypatch):
    """The locus is still read from the coordinates (which carry the orientation on both
    strands), so a contradictory marker is reported instead of mislabelling the strand."""
    _patch_references(monkeypatch, table=GENE_TABLE.replace("(minus strand)", "(plus strand)"))
    session = session_for({ncbi.GENE_SUMMARY: rpc_json(esummary([GENE_RECORD]))})
    ncbi.check_gene_summary(session)
    check = session.report.checks[0]
    assert check["status"] == "WARN" and "(plus strand)" in check["detail"]


def test_gene_summary_blames_its_own_reference_when_the_gene_table_is_unparsable(monkeypatch):
    _patch_references(monkeypatch, table="something upstream changed\n")
    session = session_for({ncbi.GENE_SUMMARY: rpc_json(esummary([GENE_RECORD]))})
    ncbi.check_gene_summary(session)
    statuses = report_statuses(session.report)
    assert statuses[f"{ncbi.GENE_SUMMARY}[7157][reference]"] == "FAIL"
    assert "fix the smoke's parser" in session.report.checks[0]["detail"]
    # the tool itself is not accused: the locus is reported as unchecked
    assert statuses[f"{ncbi.GENE_SUMMARY}[7157]"] == "WARN"


def test_gene_summary_picks_the_reference_assembly_locus(monkeypatch):
    """An added alt-locus entry at index 0 must not read as a wrong locus."""
    _patch_references(monkeypatch)
    alt = {"chraccver": "NW_003315934.1", "chrstart": 100, "chrstop": 50}
    record = {**GENE_RECORD, "genomicinfo": [alt, *GENE_RECORD["genomicinfo"]]}
    session = session_for({ncbi.GENE_SUMMARY: rpc_json(esummary([record]))})
    ncbi.check_gene_summary(session)
    check = session.report.checks[0]
    assert check["status"] == "PASS" and check["normalised_locus"]["accession"] == "NC_000017.11"


def test_gene_summary_warns_but_does_not_fail_when_datasets_is_unavailable(monkeypatch):
    monkeypatch.setattr(ncbi, "reference_gene_table", lambda gene_id: ncbi.parse_gene_table(GENE_TABLE))

    def boom(gene_id):
        raise RuntimeError("GET .../v2alpha/gene/id/7157 failed: HTTP 503")

    monkeypatch.setattr(ncbi, "reference_datasets", boom)
    session = session_for({ncbi.GENE_SUMMARY: rpc_json(esummary([GENE_RECORD]))})
    ncbi.check_gene_summary(session)
    check = session.report.checks[0]
    assert check["status"] == "WARN" and "Datasets v2alpha unavailable" in check["detail"]


def test_gene_summary_warns_on_missing_synonyms_only(monkeypatch):
    _patch_references(monkeypatch, datasets={**DATASETS, "synonyms": ["P53", "NEWALIAS"]})
    session = session_for({ncbi.GENE_SUMMARY: rpc_json(esummary([GENE_RECORD]))})
    ncbi.check_gene_summary(session)
    check = session.report.checks[0]
    assert check["status"] == "WARN" and "NEWALIAS" in check["detail"]


def test_multiple_ids_require_order_and_both_records():
    second = {**GENE_RECORD, "uid": "672", "nomenclaturesymbol": "BRCA1"}
    session = session_for({ncbi.GENE_SUMMARY: rpc_json(esummary([GENE_RECORD, second]))})
    ncbi.check_multiple_ids(session)
    assert session.report.checks[-1]["status"] == "PASS"
    assert session.client.calls[-1][1] == {"id": "7157,672"}

    session = session_for({ncbi.GENE_SUMMARY: rpc_json(esummary([second, GENE_RECORD]))})
    ncbi.check_multiple_ids(session)
    assert session.report.checks[-1]["status"] == "FAIL"


def test_protein_summary_passes_against_the_fasta(monkeypatch):
    monkeypatch.setattr(ncbi, "reference_protein_fasta", lambda gi: ncbi.fasta_record(FASTA))
    session = session_for({ncbi.PROTEIN_SUMMARY: rpc_json(esummary([PROTEIN_RECORD]))})
    ncbi.check_protein_summary(session)
    check = session.report.checks[-1]
    assert check["status"] == "PASS" and "644 aa" in check["detail"]
    assert check["reported_only"] == {"createdate": "1999/06/24", "updatedate": "2026/02/02"}


@pytest.mark.parametrize("mutation,expected", [
    ({"slen": 643}, "slen"),
    ({"title": "something else"}, "FASTA header"),
    ({"moltype": "nt"}, "moltype"),
    ({"sourcedb": "insd"}, "sourcedb"),
])
def test_protein_summary_fails_on_a_wrong_field(monkeypatch, mutation, expected):
    monkeypatch.setattr(ncbi, "reference_protein_fasta", lambda gi: ncbi.fasta_record(FASTA))
    session = session_for({ncbi.PROTEIN_SUMMARY: rpc_json(esummary([{**PROTEIN_RECORD, **mutation}]))})
    ncbi.check_protein_summary(session)
    check = session.report.checks[0]
    assert check["status"] == "FAIL" and expected in check["detail"]


@pytest.mark.parametrize("response,expected,marker", [
    (rpc_text("bad", is_error=True), "PASS", "isError=true"),
    (rpc_json(esearch([], count="0", errorlist={"phrasesnotfound": ["ZZZNOTAGENE[Symbol]"]},
                      warninglist={"outputmessages": ["No items found."]})), "WARN", "in-band"),
    (rpc_json({"status": "error", "error": "NCBI API error", "status_code": 500}), "WARN", "in-band"),
    # zero hits with no error field at all: honest, if indistinguishable -- not a FAIL
    (rpc_json(esearch([], count="0")), "WARN", "reports no hits without an error field"),
    # records for an identifier that cannot exist is the real failure
    (rpc_json(esearch(["7157"])), "FAIL", "records returned"),
])
def test_search_error_path_classification(response, expected, marker):
    session = session_for({ncbi.GENE_SEARCH: response})
    support.runner.check_rejected(session.call, "unknown symbol", ncbi.GENE_SEARCH, {"term": "x"},
                                 in_band=lambda result: ncbi._in_band(result, ncbi.search_not_found),
                                 on_accept=lambda result: ncbi._accepted(result, "esearch"))
    check = session.report.checks[-1]
    assert check["status"] == expected and marker in check["detail"]


@pytest.mark.parametrize("payload,expected", [
    ({"error": "Invalid uid ZZZ at position= 0", "result": {"uids": []}}, "WARN"),
    ({"result": {"uids": ["99999999"], "99999999": {"error": "cannot get document summary"}}}, "WARN"),
    ({"result": {"uids": ["7157"], "7157": GENE_RECORD}}, "FAIL"),
])
def test_summary_error_path_classification(payload, expected):
    session = session_for({ncbi.GENE_SUMMARY: rpc_json(wrapped(payload))})
    support.runner.check_rejected(session.call, "unknown id", ncbi.GENE_SUMMARY, {"id": "x"},
                                 in_band=lambda result: ncbi._in_band(result, ncbi.summary_not_found),
                                 on_accept=lambda result: ncbi._accepted(result, "esummary"))
    assert session.report.checks[-1]["status"] == expected


def test_unwrap_treats_the_runner_is_error_sentinel_as_an_error():
    assert ncbi.unwrap({"_isError": True, "_text": "boom"}) == (None, "boom")
    assert ncbi.unwrap({"_isError": True}) == (None, "isError=true")


def test_identification_check_reads_the_pinned_config(tmp_path):
    data = tmp_path / "src/tooluniverse/data"
    data.mkdir(parents=True)
    config = data / "ncbi_gene_tools.json"
    config.write_text('[{"name": "NCBIGene_search", "fields": {"params": {"db": "gene"}}}]')
    session = session_for({}, checkout=tmp_path)
    ncbi.check_identification(session)
    check = session.report.checks[-1]
    assert check["status"] == "WARN" and check["declared_params"] == {"NCBIGene_search": []}

    config.write_text('[{"name": "NCBIGene_search", "fields": {"params": {"tool": "x", "email": "a@b"}}}]')
    session = session_for({}, checkout=tmp_path)
    ncbi.check_identification(session)
    assert session.report.checks[-1]["status"] == "PASS"

    session = session_for({}, checkout=tmp_path / "nope")
    ncbi.check_identification(session)
    assert session.report.checks[-1]["status"] == "WARN"
    assert "could not read" in session.report.checks[-1]["detail"]


def test_call_data_reports_an_in_band_error_as_a_failure():
    session = session_for({ncbi.GENE_SEARCH: rpc_json({"status": "error", "error": "NCBI API error",
                                                       "status_code": 429})})
    assert ncbi.call_data(session.call, "x", ncbi.GENE_SEARCH, {}) is None
    check = session.report.checks[-1]
    assert check["status"] == "FAIL" and "HTTP 429" in check["detail"]


def test_ncbi_server_env_is_minimal_and_passes_proxies(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy:3128")
    env = runner.server_env({"command": "/x/.venv/bin/tooluniverse-smcp-stdio",
                             "env": {"TOOLUNIVERSE_CACHE_ENABLED": "false"}}, tmp_path, tmp_path / "tmp",
                            pass_proxies=ncbi.SMOKE.pass_proxies)
    assert env["HOME"] == str(tmp_path) and env["PATH"].startswith("/x/.venv/bin:")
    assert env["TOOLUNIVERSE_CACHE_ENABLED"] == "false" and env["HTTPS_PROXY"] == "http://proxy:3128"
    assert "OPENAI_API_KEY" not in env
