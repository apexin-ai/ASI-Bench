"""e2e_smoke/servers/alphafold_db.py without network: wrapper unwrapping, pLDDT recomputation,
AlphaMissense means, and the prediction/summary/annotation checks.

The fixtures are trimmed copies of live payloads (AlphaFold DB model v6, 2026-10-03).
"""

import pytest

from . import support
from .support import StubClient, report_statuses, rpc_json, rpc_text, runner

afdb = support.smoke_module("alphafold_db")

afdb.THROTTLE.gap = 0.0

SEQUENCE = "MVLS"                      # one residue per pLDDT bin, see PDB below
CHECKSUM = afdb.md5_of(SEQUENCE)
# Four residues, one per pLDDT bin: 40.00 / 60.00 / 80.00 / 95.00 -> mean 68.75
PDB = "\n".join([
    "HEADER    FIXTURE",
    "ATOM      1  N   MET A   1      27.409  24.168   3.130  1.00 40.00           N",
    "ATOM      2  CA  MET A   1      26.134  25.195   3.047  1.00 40.00           C",
    "ATOM      3  CA  VAL A   2      24.045  24.746   4.078  1.00 60.00           C",
    "ATOM      4  CA  LEU A   3      23.909  23.342   4.654  1.00 80.00           C",
    "ATOM      5  CA  SER A   4      22.000  22.000   5.000  1.00 95.00           C",
    "TER",
])
ENTRY = {"entryId": "AF-P69905-F1", "uniprotAccession": "P69905", "uniprotSequence": SEQUENCE,
         "sequenceChecksum": CHECKSUM, "uniprotStart": 1, "uniprotEnd": len(SEQUENCE),
         "gene": "HBA1", "taxId": 9606, "uniprotId": "HBA_HUMAN", "globalMetricValue": 68.75,
         "pdbUrl": "https://alphafold.ebi.ac.uk/files/AF-P69905-F1-model_v6.pdb",
         "latestVersion": 6, "modelCreatedDate": "2025-08-01T00:00:00Z",
         "fractionPlddtVeryLow": 0.25, "fractionPlddtLow": 0.25,
         "fractionPlddtConfident": 0.25, "fractionPlddtVeryHigh": 0.25}
AM_CSV = "protein_variant,am_pathogenicity,am_class\nM1A,0.4000,Amb\nM1C,0.5000,Amb\nV2A,0.1000,Ben\n"
ANNOTATION = {"accession": "P69905", "id": "AF-P69905-F1", "sequence": SEQUENCE,
              "annotation": [{"type": "MUTAGEN", "description": "AM score", "source_name": "AFDB",
                              "evidence": "COMPUTATIONAL/PREDICTED",
                              "source_url": "https://alphafold.ebi.ac.uk/files/AF-P69905-F1-aa-substitutions.csv",
                              "regions": [{"start": 1, "end": 1, "annotation_value": "0.45", "unit": None},
                                          {"start": 2, "end": 2, "annotation_value": "0.1", "unit": None}]}]}


def wrapped(data):
    """A payload as AlphaFoldRESTTool returns it."""
    return {"status": "success", "data": data, "metadata": {"count": 1, "endpoint": "x", "query": {}}}


def session_for(module_responses, *, checkout=None, length=len(SEQUENCE)):
    """A Session whose tools/call answers come from ``module_responses``."""
    report = runner.Report()
    client = StubClient(module_responses)
    session = runner.Session(afdb.SMOKE, None, {}, {}, checkout, report,
                             None, None, None, None, {})
    session.client, session.call = client, runner.Caller(client, report)
    afdb.MONOMER_FACTS["sequence_length"] = length
    return session


@pytest.fixture(autouse=True)
def _restore_facts():
    facts = dict(afdb.MONOMER_FACTS)
    yield
    afdb.MONOMER_FACTS.clear()
    afdb.MONOMER_FACTS.update(facts)


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------

def test_unwrap_separates_data_from_in_band_errors():
    assert afdb.unwrap(wrapped([ENTRY])) == ([ENTRY], None)
    assert afdb.unwrap({"status": "error", "error": "Not found", "endpoint": "x"})[1] == "Not found"
    assert afdb.unwrap({"status": "error", "error": "No MUTAGEN annotations available",
                        "reason": "has no annotations"})[1].endswith("(has no annotations)")
    assert afdb.unwrap([ENTRY]) == ([ENTRY], None)        # a bare payload is not an error
    assert afdb.in_band_error(wrapped([ENTRY])) is None


def test_plddt_from_pdb_reads_one_value_per_residue():
    assert afdb.plddt_from_pdb(PDB) == [40.0, 60.0, 80.0, 95.0]
    assert afdb.plddt_from_pdb("") == []


def test_plddt_fractions_bin_at_50_70_90():
    assert afdb.plddt_fractions([40.0, 60.0, 80.0, 95.0]) == {"VeryLow": 0.25, "Low": 0.25,
                                                              "Confident": 0.25, "VeryHigh": 0.25}
    assert afdb.plddt_fractions([70.0, 89.99, 90.0]) == {"VeryLow": 0.0, "Low": 0.0,
                                                         "Confident": 2 / 3, "VeryHigh": 1 / 3}
    assert afdb.plddt_fractions([]) == {}


def test_am_means_averages_each_position():
    assert afdb.am_means(AM_CSV) == {1: 0.45, 2: 0.1}
    assert afdb.am_means("nope,nothing\n1,2\n") == {}


def test_compare_am_accepts_rounded_means_and_reports_differences():
    regions = afdb.region_values(ANNOTATION["annotation"][0])
    assert regions == {1: 0.45, 2: 0.1}
    assert afdb.compare_am(regions, {1: 0.4500002, 2: 0.1}) == []
    assert afdb.compare_am(regions, {1: 0.6, 2: 0.1})[0].startswith("position 1:")
    assert "not in the AlphaMissense CSV" in afdb.compare_am(regions, {1: 0.45})[0]


def test_region_values_ignores_multi_residue_and_unparsable_regions():
    block = {"regions": [{"start": 1, "end": 3, "annotation_value": "0.5"},
                         {"start": 4, "end": 4, "annotation_value": "not a number"},
                         {"start": 5, "end": 5, "annotation_value": "0.25"}]}
    assert afdb.region_values(block) == {5: 0.25}


def test_isoform_of_accepts_only_numeric_isoform_suffixes():
    assert afdb.isoform_of("P04637", "P04637") and afdb.isoform_of("P04637-9", "P04637")
    assert not afdb.isoform_of("P04637-X", "P04637")
    assert not afdb.isoform_of("P69905", "P04637")


def test_structure_summaries_handles_both_nesting_levels():
    nested = {"structures": [{"summary": {"model_identifier": "a"}}, {"model_identifier": "b"}, 7]}
    assert afdb.structure_summaries(nested) == [{"model_identifier": "a"}, {"model_identifier": "b"}]
    assert afdb.structure_summaries([]) == []


def test_declared_fields_collects_array_branch_properties():
    schema = {"oneOf": [{"type": "array", "items": {"type": "object", "properties": {"a": {}, "b": {}}}},
                        {"type": "object", "properties": {"ignored": {}}}]}
    assert afdb.declared_fields(schema) == {"a", "b"}
    assert afdb.declared_fields(None) == set()


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def test_monomer_prediction_passes_and_recomputes_plddt(monkeypatch):
    monkeypatch.setattr(afdb, "reference_text", lambda url: PDB)
    monkeypatch.setattr(afdb, "reference_prediction", lambda acc: {ENTRY["entryId"]: ENTRY})
    session = session_for({afdb.PREDICTION: rpc_json(wrapped([ENTRY]))})
    entry = afdb.check_monomer_prediction(session)
    assert entry["entryId"] == "AF-P69905-F1"
    statuses = report_statuses(session.report)
    assert statuses[f"{afdb.PREDICTION}[{afdb.MONOMER}]"] == "PASS"
    assert statuses[f"{afdb.PREDICTION}[AF-P69905-F1 vs the raw API]"] == "PASS"
    plddt = next(c for c in session.report.checks if "pLDDT" in c["name"])
    assert plddt["status"] == "PASS" and plddt["recomputed_mean"] == 68.75


def test_raw_api_comparison_catches_a_field_the_tool_changed(monkeypatch):
    """A payload that is internally consistent but not what the API returns must FAIL."""
    session = session_for({})
    monkeypatch.setattr(afdb, "reference_prediction", lambda acc: {ENTRY["entryId"]: ENTRY})
    served = {**ENTRY, "gene": "HBA2"}
    afdb.check_against_raw_api(session, served)
    check = session.report.checks[-1]
    assert check["status"] == "FAIL" and "gene:" in check["detail"]

    session = session_for({})
    monkeypatch.setattr(afdb, "reference_prediction", lambda acc: {})
    afdb.check_against_raw_api(session, ENTRY)
    assert session.report.checks[-1]["status"] == "FAIL"
    assert "would prove nothing" in session.report.checks[-1]["detail"]


def test_entry_diffs_ignores_model_version_dependent_fields():
    # a new model version changes these; they must not make a run FAIL
    drifted = {**ENTRY, "pdbUrl": "…-model_v7.pdb", "latestVersion": 7,
               "modelCreatedDate": "2026-09-01T00:00:00Z", "allVersions": [1, 2, 3, 4, 5, 6, 7]}
    assert afdb.entry_diffs(drifted, ENTRY) == []
    assert afdb.entry_diffs({**ENTRY, "uniprotEnd": 99}, ENTRY)[0].startswith("uniprotEnd:")


@pytest.mark.parametrize("mutation,expected", [
    ({"sequenceChecksum": "deadbeef"}, "sequenceChecksum"),
    ({"uniprotEnd": 99}, "span"),
    ({"taxId": 10090}, "taxId"),
    ({"uniprotId": "OTHER_HUMAN"}, "uniprotId"),
])
def test_monomer_prediction_fails_on_inconsistent_entry(monkeypatch, mutation, expected):
    monkeypatch.setattr(afdb, "reference_text", lambda url: PDB)
    monkeypatch.setattr(afdb, "reference_prediction", lambda acc: {})
    session = session_for({afdb.PREDICTION: rpc_json(wrapped([{**ENTRY, **mutation}]))})
    afdb.check_monomer_prediction(session)
    check = session.report.checks[0]
    assert check["status"] == "FAIL" and expected in check["detail"]


def test_plddt_mean_tolerance_and_bin_edge_warning(monkeypatch):
    monkeypatch.setattr(afdb, "reference_text", lambda url: PDB)

    def run(entry):
        session = session_for({})
        afdb.check_plddt(session, entry)
        return session.report.checks[-1]

    assert run({**ENTRY, "globalMetricValue": 68.76})["status"] == "PASS"     # within the 2-decimal bias
    assert run({**ENTRY, "globalMetricValue": 70.0})["status"] == "FAIL"
    # One residue moved across a bin edge: reported as a WARN, never as a FAIL.
    edge = run({**ENTRY, "fractionPlddtConfident": 0.5, "fractionPlddtVeryHigh": 0.0})
    assert edge["status"] == "WARN" and "within one residue" in edge["detail"]
    assert run({**ENTRY, "fractionPlddtVeryHigh": 1.0})["status"] == "FAIL"
    short = run({**ENTRY, "uniprotSequence": SEQUENCE + "AAAA"})
    assert short["status"] == "FAIL" and "residues in the model file" in short["detail"]


def test_isoform_prediction_accepts_isoform_accessions_and_warns_on_a_single_model():
    isoforms = [ENTRY | {"entryId": "AF-P04637-F1", "uniprotAccession": "P04637"},
                ENTRY | {"entryId": "AF-P04637-2-F1", "uniprotAccession": "P04637-2"}]
    session = session_for({afdb.PREDICTION: rpc_json(wrapped(isoforms))})
    afdb.check_isoform_prediction(session)
    assert session.report.checks[0]["status"] == "PASS"
    assert session.report.checks[0]["entry_lengths"] == {"AF-P04637-F1": len(SEQUENCE),
                                                        "AF-P04637-2-F1": len(SEQUENCE)}

    session = session_for({afdb.PREDICTION: rpc_json(wrapped(isoforms[:1]))})
    afdb.check_isoform_prediction(session)
    assert [c["status"] for c in session.report.checks] == ["PASS", "WARN"]

    mismatched = [ENTRY | {"entryId": "AF-P04637-2-F1", "uniprotAccession": "P04637"}]
    session = session_for({afdb.PREDICTION: rpc_json(wrapped(mismatched))})
    afdb.check_isoform_prediction(session)
    assert session.report.checks[0]["status"] == "FAIL"
    assert "entryId does not match" in session.report.checks[0]["detail"]


def test_summary_selects_by_model_identifier_and_warns_about_the_list():
    document = {"uniprot_entry": {"ac": "P69905", "id": "HBA_HUMAN", "uniprot_checksum": CHECKSUM,
                                  "sequence_length": len(SEQUENCE)},
                "structures": [{"summary": {"model_identifier": "AF-0000000203852801", "coverage": 1.0,
                                            "confidence_avg_local_score": 92.8}},
                               {"summary": {"model_identifier": "AF-P69905-F1", "coverage": 1.0,
                                            "confidence_avg_local_score": 68.75}}]}
    session = session_for({afdb.SUMMARY: rpc_json(wrapped(document))})
    afdb.check_summary(session, ENTRY)
    statuses = report_statuses(session.report)
    assert statuses[f"{afdb.SUMMARY}[{afdb.MONOMER}]"] == "PASS"
    assert statuses[f"{afdb.SUMMARY}[{afdb.MONOMER}][structures list]"] == "WARN"
    assert session.report.checks[0]["index_of_entry"] == 1


@pytest.mark.parametrize("mutation,expected", [
    ({"confidence_avg_local_score": 70.0}, "confidence_avg_local_score"),
    ({"coverage": 0.5}, "coverage"),
])
def test_summary_fails_when_it_disagrees_with_the_prediction(mutation, expected):
    document = {"uniprot_entry": {"id": "HBA_HUMAN", "uniprot_checksum": CHECKSUM,
                                 "sequence_length": len(SEQUENCE)},
                "structures": [{"summary": {"model_identifier": "AF-P69905-F1", "coverage": 1.0,
                                            "confidence_avg_local_score": 68.75, **mutation}}]}
    session = session_for({afdb.SUMMARY: rpc_json(wrapped(document))})
    afdb.check_summary(session, ENTRY)
    check = session.report.checks[0]
    assert check["status"] == "FAIL" and expected in check["detail"]


def test_summary_fails_when_the_model_is_absent():
    document = {"uniprot_entry": {}, "structures": [{"summary": {"model_identifier": "AF-00000002"}}]}
    session = session_for({afdb.SUMMARY: rpc_json(wrapped(document))})
    afdb.check_summary(session, ENTRY)
    assert session.report.checks[0]["status"] == "FAIL"
    assert "no AF-P69905-F1" in session.report.checks[0]["detail"]


def test_annotations_compare_against_the_substitutions_csv(monkeypatch):
    monkeypatch.setattr(afdb, "reference_text", lambda url: AM_CSV)
    session = session_for({afdb.ANNOTATIONS: rpc_json(wrapped(ANNOTATION))}, length=len(SEQUENCE))
    afdb.MONOMER_FACTS["sequence_length"] = 2          # the fixture carries two positions
    afdb.check_annotations(session, ENTRY | {"uniprotSequence": SEQUENCE})
    statuses = [c["status"] for c in session.report.checks]
    assert statuses[0] == "PASS" and "AM score" in session.report.checks[-1]["detail"]
    assert session.report.checks[-1]["status"] == "WARN"


def test_annotations_fail_on_a_wrong_value(monkeypatch):
    monkeypatch.setattr(afdb, "reference_text", lambda url: AM_CSV.replace("0.4000", "0.9000"))
    session = session_for({afdb.ANNOTATIONS: rpc_json(wrapped(ANNOTATION))})
    afdb.MONOMER_FACTS["sequence_length"] = 2
    afdb.check_annotations(session, ENTRY)
    assert session.report.checks[0]["status"] == "FAIL"
    assert "position 1" in session.report.checks[0]["detail"]


def test_annotations_empty_state_is_a_warning_but_a_real_failure_is_not():
    empty = {"status": "error", "error": "No MUTAGEN annotations available",
             "reason": "Protein exists in AlphaFold DB but has no MUTAGEN annotations"}
    session = session_for({afdb.ANNOTATIONS: rpc_json(empty)})
    afdb.check_annotations(session, ENTRY)
    assert session.report.checks[-1]["status"] == "WARN"
    assert "the empty state the tool description claims" in session.report.checks[-1]["detail"]

    # an HTTP 500, a 404 or a parse error must not read as the documented empty state
    for payload in ({"status": "error", "error": "AlphaFold EBI API is temporarily unavailable (HTTP 500)."},
                    {"status": "error", "error": "Protein not found in AlphaFold DB"},
                    {"_isError": True, "_text": "transport blew up"}):
        session = session_for({afdb.ANNOTATIONS: rpc_json(payload)})
        afdb.check_annotations(session, ENTRY)
        assert session.report.checks[-1]["status"] == "FAIL"
        assert "the annotation call failed" in session.report.checks[-1]["detail"]


def test_annotations_require_a_mutagen_block():
    other = {"accession": "P69905", "id": "AF-P69905-F1", "sequence": SEQUENCE,
             "annotation": [{"type": "OTHER", "regions": []}]}
    session = session_for({afdb.ANNOTATIONS: rpc_json(wrapped(other))})
    afdb.check_annotations(session, ENTRY)
    assert session.report.checks[-1]["status"] == "FAIL"
    assert "no MUTAGEN block" in session.report.checks[-1]["detail"]


def test_undeclared_type_is_a_pass_when_rejected_and_a_warning_when_overridden():
    """Live, FastMCP rejects the undeclared `type` (PASS); a server that lets it
    through and silently swaps in MUTAGEN is the WARN this check exists for."""
    session = session_for({afdb.ANNOTATIONS: rpc_json(wrapped(ANNOTATION))})
    afdb.check_overridden_type(session)
    assert session.report.checks[-1]["status"] == "WARN"
    assert "overwritten by its auto_query_params" in session.report.checks[-1]["detail"]

    for rejection in (rpc_json({"status": "error", "error": "AlphaFold API returned 422"}),
                      rpc_text("Unexpected keyword argument [type=unexpected_keyword_argument]",
                               is_error=True)):
        session = session_for({afdb.ANNOTATIONS: rejection})
        afdb.check_overridden_type(session)
        assert session.report.checks[-1]["status"] == "PASS"


def test_alias_and_ignored_parameter_checks():
    session = session_for({afdb.PREDICTION: rpc_json(wrapped([ENTRY]))})
    afdb.check_alias(session, ENTRY)
    assert session.report.checks[-1]["status"] == "PASS"
    assert session.client.calls[-1] == (afdb.PREDICTION, {"accession": afdb.MONOMER})

    session = session_for({afdb.PREDICTION: rpc_json(wrapped([{**ENTRY, "sequenceChecksum": "other"}]))})
    afdb.check_alias(session, ENTRY)
    assert session.report.checks[-1]["status"] == "FAIL"

    session = session_for({afdb.PREDICTION: rpc_json(wrapped([ENTRY]))})
    afdb.check_ignored_parameter(session, ENTRY)
    assert session.report.checks[-1]["status"] == "WARN"
    assert "silently ignored" in session.report.checks[-1]["detail"]

    session = session_for({afdb.PREDICTION: rpc_json({"status": "error", "error": "returned 422"})})
    afdb.check_ignored_parameter(session, ENTRY)
    assert session.report.checks[-1]["status"] == "WARN"
    assert "is rejected rather than silently ignored" in session.report.checks[-1]["detail"]

    # a real isError=true result is a rejection too, not an unexpected payload
    session = session_for({afdb.PREDICTION: rpc_text("bad checksum", is_error=True)})
    afdb.check_ignored_parameter(session, ENTRY)
    assert session.report.checks[-1]["status"] == "WARN"

    # succeeding but answering with a different model is a genuine FAIL
    other = {**ENTRY, "entryId": "AF-P04637-F1"}
    session = session_for({afdb.PREDICTION: rpc_json(wrapped([other]))})
    afdb.check_ignored_parameter(session, ENTRY)
    assert session.report.checks[-1]["status"] == "FAIL"


def test_unwrap_treats_the_runner_is_error_sentinel_as_an_error():
    assert afdb.unwrap({"_isError": True, "_text": "boom"}) == (None, "boom")
    assert afdb.unwrap({"_isError": True}) == (None, "isError=true")


def test_declared_schema_check_reads_the_pinned_config(tmp_path):
    data = tmp_path / "src/tooluniverse/data"
    data.mkdir(parents=True)
    config = data / "alphafold_tools.json"
    config.write_text('[{"name": "alphafold_get_prediction", "return_schema": {"oneOf": [{"type": "array",'
                      ' "items": {"type": "object", "properties": {"entryId": {}, "ipTM": {}}}}]}}]')
    session = session_for({}, checkout=tmp_path)
    afdb.check_declared_schema(session, {"entryId": "x", "chainId": "A"})
    check = session.report.checks[-1]
    assert check["status"] == "WARN" and check["declared_absent"] == ["ipTM"]
    assert check["live_undeclared"] == ["chainId"]

    session = session_for({}, checkout=tmp_path)
    afdb.check_declared_schema(session, {"entryId": "x", "ipTM": 1})
    assert session.report.checks[-1]["status"] == "PASS"

    session = session_for({}, checkout=tmp_path / "nope")
    afdb.check_declared_schema(session, {"entryId": "x"})
    assert session.report.checks[-1]["status"] == "WARN"
    assert "could not read" in session.report.checks[-1]["detail"]


@pytest.mark.parametrize("response,expected,marker", [
    (rpc_text("bad", is_error=True), "PASS", "isError=true"),
    (rpc_json({"status": "error", "error": "AlphaFold API returned 400"}), "WARN", "in-band"),
    # upstream gaining support for the identifier is an improvement, not a wrong result
    (rpc_json(wrapped([ENTRY])), "WARN", "description is out of date"),
])
def test_error_paths_classification(response, expected, marker):
    session = session_for({afdb.PREDICTION: response})
    afdb.check_error_paths(session)
    assert {c["status"] for c in session.report.checks} == {expected}
    assert marker in session.report.checks[0]["detail"]


def test_call_data_reports_an_in_band_error_as_a_failure():
    session = session_for({afdb.PREDICTION: rpc_json({"status": "error", "error": "boom"})})
    assert afdb.call_data(session.call, "x", afdb.PREDICTION, {}) is None
    assert session.report.checks[-1]["status"] == "FAIL"
    assert "in-band" in session.report.checks[-1]["detail"]


def test_alphafold_server_env_is_minimal_and_passes_proxies(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy:3128")
    env = runner.server_env({"command": "/x/.venv/bin/tooluniverse-smcp-stdio",
                             "env": {"TOOLUNIVERSE_CACHE_ENABLED": "false"}}, tmp_path, tmp_path / "tmp",
                            pass_proxies=afdb.SMOKE.pass_proxies)
    assert env["HOME"] == str(tmp_path) and env["PATH"].startswith("/x/.venv/bin:")
    assert env["TOOLUNIVERSE_CACHE_ENABLED"] == "false" and env["HTTPS_PROXY"] == "http://proxy:3128"
    assert "OPENAI_API_KEY" not in env
