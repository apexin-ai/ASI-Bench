"""Direct (agent-free) E2E smoke test for the pinned ToolUniverse AlphaFold DB MCP tools.

Run with the shared ToolUniverse virtualenv (the checkout is shared with the
``arxiv`` and ``ncbi`` manifest ids)::

    ~/mcp/tooluniverse/.venv/bin/python scripts/mcp/e2e/smoke.py alphafold_db \\
        --config ~/mcp/alphafold_db.mcp.json

Needs network access to alphafold.ebi.ac.uk (both the server and the references
query the live public API; no key is needed).

There is no date window to pin here, so stability comes from **immutable
identifiers** instead: a UniProt accession plus values recomputed from the very
files the API points at. Every check is computed here, outside the server
process:

* the model entry field by field against a raw query of the same endpoint made
  here (so a whole stale payload cannot pass on internal consistency alone);
* ``md5(uniprotSequence) == sequenceChecksum``;
* ``globalMetricValue`` against the per-residue pLDDT read out of the model
  ``pdbUrl`` (CA B-factors), and the four ``fractionPlddt*`` against the same
  values binned at 50/70/90;
* every ``MUTAGEN`` region value against the mean of that position's 19
  ``am_pathogenicity`` rows in the AlphaMissense ``-aa-substitutions.csv`` the
  annotation itself links to;
* the summary endpoint's ``confidence_avg_local_score`` against the prediction
  endpoint's ``globalMetricValue`` for the same ``AF-<acc>-F1`` model.

Model-version dependent values (which URL, ``latestVersion``,
``modelCreatedDate``) and the growing ``structures`` list are **reported, never
asserted**: AlphaFold DB is at model v6 today and pLDDT changes with every
model version, so only recomputation can stay correct.

Upstream defects that do not make a correctly used tool wrong (errors returned
in-band with ``isError=false``, a declared parameter that is forwarded as an
unsupported query param, a ``return_schema`` and a tool description that no
longer match the live API) are WARN; wrong values for correct inputs are FAIL.
An argument the MCP layer rejects outright, and an identifier upstream starts
supporting, are reported as what they are and never as a wrong result.
"""
from __future__ import annotations

import hashlib
import json
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request

from ..client import text_of
from ..runner import Caller, Session, Smoke, check_rejected

PREDICTION = "alphafold_get_prediction"
SUMMARY = "alphafold_get_summary"
ANNOTATIONS = "alphafold_get_annotations"

API_GAP_S = 0.5                  # be a polite client; EBI publishes no fixed rate limit
USER_AGENT = "asibench-mcp-e2e-smoke/1 (reference client)"

MONOMER = "P69905"               # HBA_HUMAN, 142 aa: one entry, small payloads
ISOFORMS = "P04637"              # TP53: the canonical entry plus 8 isoform models
ENTRY_NAME = "HBA_HUMAN"         # an entry name, not an accession -> HTTP 400 upstream
BAD_ACCESSION = "XXXXXX"
# Identity facts of the accession itself, not of the model: immutable.
MONOMER_FACTS = {"taxId": 9606, "uniprotId": "HBA_HUMAN", "sequence_length": 142}
API_URL = "https://alphafold.ebi.ac.uk/api"
# Compared field by field with an independently fetched copy of the same entry.
# Model-version dependent values (URLs, latestVersion, modelCreatedDate) are left
# out on purpose: they are reported, never asserted.
COMPARED_FIELDS = ("entryId", "uniprotAccession", "uniprotSequence", "sequenceChecksum",
                   "uniprotStart", "uniprotEnd", "gene", "taxId", "uniprotId",
                   "organismScientificName", "globalMetricValue", "fractionPlddtVeryLow",
                   "fractionPlddtLow", "fractionPlddtConfident", "fractionPlddtVeryHigh")

# Per-residue pLDDT is written to the model file with two decimals, so a mean
# recomputed from it differs from the stored full-precision mean by up to ~0.01
# (measured 2026-10-03: P69905 98.06 == 98.06, P04637 75.0501 vs 75.06).
PLDDT_TOL = 0.02
FRACTION_DP = 3                  # fractionPlddt* are rounded to 3 decimals upstream
AM_DP = 4                        # region annotation_value is the mean rounded to 4 decimals
AM_TOL = 5e-5


# --------------------------------------------------------------------------
# Pure helpers (stdlib only; unit-tested offline)
# --------------------------------------------------------------------------

def unwrap(payload):
    """``(data, error)`` for an AlphaFoldRESTTool result.

    The tool wraps a successful payload as ``{"status": "success", "data": ...,
    "metadata": ...}`` and reports every failure in-band as ``{"status":
    "error", "error": ...}`` with ``isError=false``. A real ``isError=true``
    result arrives as the runner's ``{"_isError": True, "_text": ...}`` sentinel
    and must read as an error too, so that a server which starts rejecting bad
    input properly is never mistaken for one returning a wrong payload."""
    if not isinstance(payload, dict):
        return payload, None                      # a bare payload (not this tool's shape)
    if payload.get("_isError"):
        return None, str(payload.get("_text") or "isError=true")
    if payload.get("status") == "success" and "data" in payload:
        return payload["data"], None
    message = payload.get("error") or payload.get("detail")
    if message or payload.get("status") == "error":
        extra = payload.get("reason") or payload.get("detail") or ""
        return None, f"{message}{f' ({extra})' if extra and extra != message else ''}"
    return payload, None


def in_band_error(payload) -> str | None:
    """The tool's message if the payload is an error reported in a normal result."""
    return unwrap(payload)[1]


def md5_of(sequence: str) -> str:
    return hashlib.md5(sequence.encode("utf-8")).hexdigest()


def plddt_from_pdb(text: str) -> list[float]:
    """One pLDDT per residue: the B-factor of each residue's CA atom, in file order.

    AlphaFold writes the residue's pLDDT into the B-factor column of all of its
    atoms, so one atom per residue is enough (and avoids weighting residues by
    their atom count)."""
    return [float(line[60:66]) for line in text.splitlines()
            if line.startswith("ATOM") and line[12:16].strip() == "CA"]


def plddt_fractions(values: list[float]) -> dict[str, float]:
    """The four ``fractionPlddt*`` bins: <50, [50,70), [70,90), >=90."""
    total = len(values)
    if not total:
        return {}
    return {"VeryLow": sum(v < 50 for v in values) / total,
            "Low": sum(50 <= v < 70 for v in values) / total,
            "Confident": sum(70 <= v < 90 for v in values) / total,
            "VeryHigh": sum(v >= 90 for v in values) / total}


def am_means(csv_text: str) -> dict[int, float]:
    """``{position: mean am_pathogenicity}`` from an AlphaMissense substitutions CSV.

    Rows are ``protein_variant,am_pathogenicity,am_class`` with variants like
    ``M1A``; every position carries the 19 non-reference substitutions."""
    position_values: dict[int, list[float]] = {}
    lines = [line for line in csv_text.splitlines() if line.strip()]
    header = lines[0].split(",") if lines else []
    try:
        variant_at, score_at = header.index("protein_variant"), header.index("am_pathogenicity")
    except ValueError:
        return {}
    for line in lines[1:]:
        cells = line.split(",")
        if len(cells) <= max(variant_at, score_at):
            continue
        digits = "".join(c for c in cells[variant_at][1:] if c.isdigit())
        if not digits:
            continue
        try:
            position_values.setdefault(int(digits), []).append(float(cells[score_at]))
        except ValueError:
            continue
    return {position: statistics.fmean(values) for position, values in position_values.items()}


def region_values(block: dict) -> dict[int, float]:
    """``{position: annotation_value}`` for the single-residue regions of one annotation block."""
    out: dict[int, float] = {}
    for region in block.get("regions") or []:
        start, end = region.get("start"), region.get("end")
        if start != end or start is None:
            continue
        try:
            out[int(start)] = float(region.get("annotation_value"))
        except (TypeError, ValueError):
            continue
    return out


def compare_am(regions: dict[int, float], reference: dict[int, float]) -> list[str]:
    """Differences between region values and the recomputed per-position means."""
    problems = []
    missing = sorted(set(regions) - set(reference))
    if missing:
        problems.append(f"{len(missing)} position(s) not in the AlphaMissense CSV: {missing[:5]}")
    for position in sorted(set(regions) & set(reference)):
        got, want = regions[position], reference[position]
        if round(want, AM_DP) != got and abs(want - got) > AM_TOL:
            problems.append(f"position {position}: tool {got} vs recomputed mean {want:.6f}")
    return problems[:5]


def entries_by_id(data) -> dict[str, dict]:
    """``{entryId: entry}`` of a prediction payload (a list of models)."""
    return {e["entryId"]: e for e in data if isinstance(e, dict) and e.get("entryId")} \
        if isinstance(data, list) else {}


def isoform_of(accession: str, base: str) -> bool:
    """``P04637`` or one of its isoform accessions (``P04637-2``)."""
    if accession == base:
        return True
    prefix, _, suffix = (accession or "").partition("-")
    return prefix == base and suffix.isdigit()


def structure_summaries(data) -> list[dict]:
    """The per-model summary dicts of a /uniprot/summary payload, in upstream order."""
    if not isinstance(data, dict):
        return []
    out = []
    for item in data.get("structures") or []:
        summary = item.get("summary") if isinstance(item, dict) and isinstance(item.get("summary"), dict) else item
        if isinstance(summary, dict):
            out.append(summary)
    return out


def declared_fields(return_schema) -> set[str]:
    """Property names of the array-of-object branches of a tool's ``return_schema``."""
    fields: set[str] = set()
    for branch in (return_schema or {}).get("oneOf") or []:
        items = branch.get("items") if isinstance(branch, dict) else None
        if isinstance(items, dict):
            fields |= set((items.get("properties") or {}))
    return fields


# --------------------------------------------------------------------------
# Independent references (the raw AlphaFold API and its own files)
# --------------------------------------------------------------------------

class Throttle:
    """Keeps >= API_GAP_S between AlphaFold requests made by the server *or* by this script."""

    def __init__(self, gap: float = API_GAP_S) -> None:
        self.gap = gap
        self.last = 0.0

    def wait(self) -> None:
        delay = self.last + self.gap - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self.last = time.monotonic()


THROTTLE = Throttle()


def http_get(url: str, timeout: float = 90.0, attempts: int = 3) -> bytes:
    error = None
    for attempt in range(attempts):
        THROTTLE.wait()
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"GET {url} failed: HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            error = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed: {error}")


def reference_text(url: str) -> str:
    return http_get(url).decode("utf-8", "replace")


def reference_prediction(accession: str) -> dict[str, dict]:
    """``{entryId: entry}`` from a raw query of the same endpoint, made here.

    Fetched independently so that a whole payload cannot pass on internal
    consistency alone (a stale model version is self-consistent)."""
    document = json.loads(http_get(f"{API_URL}/prediction/{accession}"))
    return entries_by_id(document)


def entry_diffs(got: dict, reference: dict) -> list[str]:
    """Differences between a model entry and an independently fetched copy of it."""
    return [f"{field}: server {str(got.get(field))[:60]!r} vs reference {str(reference.get(field))[:60]!r}"
            for field in COMPARED_FIELDS if got.get(field) != reference.get(field)]


def call_json(call: Caller, check: str, tool: str, arguments: dict, **kwargs):
    """Throttled tools/call + JSON decode (the server shares our request budget)."""
    THROTTLE.wait()
    return call.json(check, tool, arguments, **kwargs)


def call_data(call: Caller, check: str, tool: str, arguments: dict):
    """``call_json`` + :func:`unwrap`; None (with a FAIL) if the tool reported an error."""
    payload = call_json(call, check, tool, arguments)
    if payload is None:
        return None
    data, error = unwrap(payload)
    if error is not None:
        call.report.add("L1", check, "FAIL", f"tool reported an error in-band: {error[:200]}")
        return None
    return data


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def check_monomer_prediction(session: Session) -> dict | None:
    """The single P69905 model: checksum, declared range, and pLDDT recomputed from the model file."""
    call, report = session.call, session.report
    name = f"{PREDICTION}[{MONOMER}]"
    data = call_data(call, name, PREDICTION, {"qualifier": MONOMER})
    if data is None:
        return None
    entries = entries_by_id(data)
    entry_id = f"AF-{MONOMER}-F1"
    if entry_id not in entries:
        report.add("L1", name, "FAIL", f"no {entry_id} in {sorted(entries)}")
        return None
    entry = entries[entry_id]
    sequence = entry.get("uniprotSequence") or ""
    problems = []
    if len(entries) != 1:
        report.add("L1", f"{name}[model count]", "WARN",
                   f"{len(entries)} models for a single-isoform accession: {sorted(entries)}")
    if md5_of(sequence) != entry.get("sequenceChecksum"):
        problems.append(f"sequenceChecksum {entry.get('sequenceChecksum')} != md5 {md5_of(sequence)}")
    if len(sequence) != MONOMER_FACTS["sequence_length"]:
        problems.append(f"sequence length {len(sequence)} != {MONOMER_FACTS['sequence_length']}")
    span = (entry.get("uniprotEnd") or 0) - (entry.get("uniprotStart") or 0) + 1
    if span != len(sequence):
        problems.append(f"uniprotStart/End span {span} != sequence length {len(sequence)}")
    for key in ("taxId", "uniprotId"):
        if entry.get(key) != MONOMER_FACTS[key]:
            problems.append(f"{key} {entry.get(key)!r} != {MONOMER_FACTS[key]!r}")
    report.add("L1", name, "FAIL" if problems else "PASS",
               "; ".join(problems) if problems else
               f"{entry_id}: {len(sequence)} aa, checksum is md5(sequence), {entry.get('gene')} / tax "
               f"{entry.get('taxId')}",
               model_version=entry.get("latestVersion"), model_created=entry.get("modelCreatedDate"))
    check_against_raw_api(session, entry)
    check_plddt(session, entry)
    return entry


def check_against_raw_api(session: Session, entry: dict) -> None:
    """The model entry field by field against an independently fetched copy."""
    report = session.report
    name = f"{PREDICTION}[{entry['entryId']} vs the raw API]"
    reference = reference_prediction(MONOMER).get(entry["entryId"])
    if reference is None:
        report.add("L1", name, "FAIL",
                   f"{entry['entryId']} is not in a raw query of the same endpoint; the case would prove nothing")
        return
    diffs = entry_diffs(entry, reference)
    report.add("L1", name, "FAIL" if diffs else "PASS",
               "; ".join(diffs[:4]) if diffs else
               f"all {len(COMPARED_FIELDS)} compared fields identical to a raw query made here")


def check_plddt(session: Session, entry: dict) -> None:
    """globalMetricValue and the fractionPlddt* bins against the model file's own B-factors."""
    report = session.report
    name = f"{PREDICTION}[{entry['entryId']} pLDDT from {entry.get('pdbUrl', '').rsplit('/', 1)[-1]}]"
    values = plddt_from_pdb(reference_text(entry["pdbUrl"]))
    sequence = entry.get("uniprotSequence") or ""
    if len(values) != len(sequence):
        report.add("L1", name, "FAIL", f"{len(values)} residues in the model file != {len(sequence)} in the sequence")
        return
    mean = statistics.fmean(values)
    reported = entry.get("globalMetricValue")
    problems = []
    if not isinstance(reported, (int, float)) or abs(mean - reported) > PLDDT_TOL:
        problems.append(f"globalMetricValue {reported} vs recomputed mean {mean:.4f}")
    fractions = plddt_fractions(values)
    off_by_one = 1 / len(values) + 1e-9           # a single residue at a bin edge
    soft = []
    for key, value in fractions.items():
        declared = entry.get(f"fractionPlddt{key}")
        if not isinstance(declared, (int, float)):
            problems.append(f"fractionPlddt{key} is {declared!r}")
        elif round(value, FRACTION_DP) != declared:
            (soft if abs(value - declared) <= off_by_one else problems).append(
                f"fractionPlddt{key} {declared} vs recomputed {value:.6f}")
    if problems:
        report.add("L1", name, "FAIL", "; ".join(problems),
                   recomputed_mean=round(mean, 4), recomputed_fractions=fractions)
    elif soft:
        report.add("L1", name, "WARN",
                   "within one residue of the recomputed bins (the model file carries 2 decimals, so a "
                   f"residue at a bin edge can land either side): {'; '.join(soft)}",
                   recomputed_mean=round(mean, 4), recomputed_fractions=fractions)
    else:
        report.add("L1", name, "PASS",
                   f"{len(values)} residues: mean {mean:.4f} ~= {reported} (tol {PLDDT_TOL}), all four "
                   "fractions match the recomputed bins",
                   recomputed_mean=round(mean, 4), recomputed_fractions=fractions)


def check_isoform_prediction(session: Session) -> None:
    """P04637 returns the canonical model plus isoform models; every one must be self-consistent."""
    call, report = session.call, session.report
    name = f"{PREDICTION}[{ISOFORMS} isoforms]"
    data = call_data(call, name, PREDICTION, {"qualifier": ISOFORMS})
    if data is None:
        return
    entries = entries_by_id(data)
    canonical = f"AF-{ISOFORMS}-F1"
    problems = []
    if canonical not in entries:
        problems.append(f"no {canonical} in {sorted(entries)}")
    lengths = {}
    for entry_id, entry in sorted(entries.items()):
        sequence = entry.get("uniprotSequence") or ""
        accession = entry.get("uniprotAccession") or ""
        lengths[entry_id] = len(sequence)
        if md5_of(sequence) != entry.get("sequenceChecksum"):
            problems.append(f"{entry_id}: sequenceChecksum != md5(uniprotSequence)")
        # An isoform model carries the isoform accession (P04637-2), not the base one.
        if not isoform_of(accession, ISOFORMS):
            problems.append(f"{entry_id}: uniprotAccession {accession!r} is not {ISOFORMS} or an isoform of it")
        elif entry_id != f"AF-{accession}-F1":
            problems.append(f"{entry_id}: entryId does not match uniprotAccession {accession!r}")
    report.add("L1", name, "FAIL" if problems else "PASS",
               "; ".join(problems[:4]) if problems else
               f"{len(entries)} model(s), each checksum is md5 of its own isoform sequence; "
               f"canonical {canonical} is {lengths.get(canonical)} aa",
               entry_lengths=lengths)
    if len(entries) < 2:
        report.add("L1", f"{name}[count]", "WARN",
                   f"only {len(entries)} model(s) for {ISOFORMS} today; isoform coverage grew/shrank upstream, "
                   "so the count is reported, not asserted")


def check_alias(session: Session, entry: dict | None) -> None:
    """The documented ``accession`` alias of ``qualifier`` must reach the same model."""
    call, report = session.call, session.report
    name = f"{PREDICTION}[accession alias]"
    data = call_data(call, name, PREDICTION, {"accession": MONOMER})
    if data is None or entry is None:
        return
    entries = entries_by_id(data)
    alias_entry = entries.get(entry["entryId"])
    if alias_entry is None:
        report.add("L1", name, "FAIL", f"alias call returned {sorted(entries)}, without {entry['entryId']}")
    elif alias_entry.get("sequenceChecksum") != entry.get("sequenceChecksum"):
        report.add("L1", name, "FAIL", "alias call returned a different sequence checksum")
    else:
        report.add("L1", name, "PASS", f"accession= reaches the same {entry['entryId']}")


def check_ignored_parameter(session: Session, entry: dict | None) -> None:
    """``sequence_checksum`` is declared but forwarded as an unsupported query parameter."""
    call, report = session.call, session.report
    name = f"{PREDICTION}[sequence_checksum]"
    payload = call_json(call, name, PREDICTION, {"qualifier": MONOMER, "sequence_checksum": "not-a-crc64"},
                        allow_error=True)
    if payload is None:
        return
    data, error = unwrap(payload)
    if error is not None:
        # Rejecting a checksum that cannot be one is correct behaviour, not a wrong
        # result: report what happened without blaming the server.
        report.add("L1", name, "WARN",
                   f"a bogus `sequence_checksum` is rejected rather than silently ignored: {error[:160]}")
        return
    entries = entries_by_id(data)
    same = entry is not None and entry["entryId"] in entries
    report.add("L1", name, "WARN" if same else "FAIL",
               "a bogus `sequence_checksum` is accepted and silently ignored (it is forwarded as a query "
               "parameter the API does not know), so the parameter cannot be used to validate a sequence"
               if same else f"the call succeeded but returned a different model: {sorted(entries)}")


def check_summary(session: Session, entry: dict | None) -> None:
    """The summary of AF-<acc>-F1, selected by model_identifier rather than by position."""
    call, report = session.call, session.report
    name = f"{SUMMARY}[{MONOMER}]"
    data = call_data(call, name, SUMMARY, {"qualifier": MONOMER})
    if data is None:
        return
    summaries = structure_summaries(data)
    entry_id = f"AF-{MONOMER}-F1"
    positions = [i for i, s in enumerate(summaries) if s.get("model_identifier") == entry_id]
    uniprot = data.get("uniprot_entry") or {} if isinstance(data, dict) else {}
    problems = []
    if not positions:
        report.add("L1", name, "FAIL", f"no {entry_id} among {len(summaries)} structures "
                                       f"({[s.get('model_identifier') for s in summaries[:5]]})")
        return
    model = summaries[positions[0]]
    if uniprot.get("sequence_length") != MONOMER_FACTS["sequence_length"]:
        problems.append(f"uniprot_entry.sequence_length {uniprot.get('sequence_length')!r}")
    if entry is not None and uniprot.get("uniprot_checksum") != md5_of(entry.get("uniprotSequence") or ""):
        problems.append(f"uniprot_checksum {uniprot.get('uniprot_checksum')!r} != md5 of the sequence")
    if uniprot.get("id") != MONOMER_FACTS["uniprotId"]:
        problems.append(f"uniprot_entry.id {uniprot.get('id')!r} != {MONOMER_FACTS['uniprotId']!r}")
    if model.get("coverage") != 1.0:
        problems.append(f"coverage {model.get('coverage')!r} != 1.0 for a full-length model")
    if entry is not None and model.get("confidence_avg_local_score") != entry.get("globalMetricValue"):
        problems.append(f"confidence_avg_local_score {model.get('confidence_avg_local_score')!r} != "
                        f"prediction globalMetricValue {entry.get('globalMetricValue')!r}")
    report.add("L1", name, "FAIL" if problems else "PASS",
               "; ".join(problems[:4]) if problems else
               f"{entry_id} agrees with the prediction endpoint (avg {model.get('confidence_avg_local_score')}, "
               f"coverage {model.get('coverage')}, {uniprot.get('sequence_length')} aa)",
               structures=len(summaries), index_of_entry=positions[0],
               model_identifiers=[s.get("model_identifier") for s in summaries][:20])
    if positions[0] != 0 or len(summaries) > 1:
        report.add("L1", f"{name}[structures list]", "WARN",
                   f"{len(summaries)} structures and {entry_id} is at index {positions[0]}: the list mixes in "
                   "predicted complexes (AF3-style HETERODIMERs) and grows upstream, so a caller must select "
                   "by model_identifier and must not take index 0; count and order are reported, not asserted")


def check_annotations(session: Session, entry: dict | None) -> None:
    """Every MUTAGEN region value against the mean of its AlphaMissense substitution rows."""
    call, report = session.call, session.report
    name = f"{ANNOTATIONS}[{MONOMER}]"
    payload = call_json(call, name, ANNOTATIONS, {"qualifier": MONOMER})
    if payload is None:
        return
    data, error = unwrap(payload)
    if error is not None:
        # The tool has one documented empty state ("No MUTAGEN annotations
        # available", with a reason) and many real failures (HTTP 500, 404, parse
        # errors). Only the former is a WARN; the rest are this run's problem.
        if "no mutagen annotations" in error.lower():
            report.add("L1", name, "WARN",
                       f"no annotations returned, the empty state the tool description claims for every "
                       f"accession: {error[:200]}")
        else:
            report.add("L1", name, "FAIL", f"the annotation call failed: {error[:200]}")
        return
    blocks = [b for b in ((data.get("annotation") or []) if isinstance(data, dict) else [])
              if isinstance(b, dict)]
    if not blocks:
        report.add("L1", name, "FAIL", f"no annotation blocks in {str(data)[:200]}")
        return
    mutagen = [b for b in blocks if b.get("type") == "MUTAGEN"]
    if not mutagen:
        report.add("L1", name, "FAIL",
                   f"no MUTAGEN block among {[b.get('type') for b in blocks]}")
        return
    block = mutagen[0]
    source_url = block.get("source_url") or ""
    regions = region_values(block)
    problems = []
    if entry is not None and data.get("sequence") != entry.get("uniprotSequence"):
        problems.append("annotation sequence differs from the prediction endpoint's uniprotSequence")
    if data.get("id") != f"AF-{MONOMER}-F1":
        problems.append(f"id {data.get('id')!r} != AF-{MONOMER}-F1")
    if len(regions) != MONOMER_FACTS["sequence_length"]:
        problems.append(f"{len(regions)} single-residue regions != {MONOMER_FACTS['sequence_length']} residues")
    if not source_url.endswith("-aa-substitutions.csv"):
        report.add("L1", f"{name}[source]", "WARN",
                   f"source_url is not an AlphaMissense substitutions CSV: {source_url!r}; the values cannot "
                   "be recomputed")
    else:
        reference = am_means(reference_text(source_url))
        if not reference:
            problems.append(f"could not parse {source_url}")
        else:
            problems.extend(compare_am(regions, reference))
    report.add("L1", name, "FAIL" if problems else "PASS",
               "; ".join(problems[:4]) if problems else
               f"{len(regions)} per-residue values equal the mean of each position's 19 am_pathogenicity rows "
               f"in {source_url.rsplit('/', 1)[-1]} (rounded to {AM_DP} decimals)",
               annotation_types=[b.get("type") for b in blocks], description=block.get("description"),
               evidence=block.get("evidence"))
    if block.get("description") == "AM score" or block.get("source_name") == "AFDB":
        report.add("L1", f"{name}[content]", "WARN",
                   "the tool describes MUTAGEN as \"experimental mutagenesis data mapped from UniProt\" and "
                   "its description claims the endpoint returns empty for every accession; live it returns "
                   f"AlphaFold's own computational AlphaMissense scores ({block.get('description')!r}, "
                   f"source {block.get('source_name')!r}, evidence {block.get('evidence')!r})")


def check_overridden_type(session: Session) -> None:
    """An undeclared ``type`` argument: rejected by the MCP layer, or silently overwritten.

    The tool config applies ``auto_query_params`` after the caller's arguments, so a
    Python caller's ``type`` is replaced by ``MUTAGEN`` without a word. Over MCP that
    defect is normally unreachable, because ``type`` is not in the tool's input schema
    and FastMCP rejects undeclared arguments (measured 2026-10-03: a validation
    error). Both outcomes are fine for a caller; only the silent override is a WARN."""
    call, report = session.call, session.report
    name = f"{ANNOTATIONS}[type=NONSENSE]"
    payload = call_json(call, name, ANNOTATIONS, {"qualifier": MONOMER, "type": "NONSENSE"}, allow_error=True)
    if payload is None:
        return
    data, error = unwrap(payload)
    if error is not None:
        report.add("L1", name, "PASS", f"the invalid annotation type is reported: {error[:160]}")
        return
    blocks = (data.get("annotation") or []) if isinstance(data, dict) else []
    types = [b.get("type") for b in blocks]
    report.add("L1", name, "WARN",
               "the caller's `type` reached the tool and was overwritten by its auto_query_params "
               "(which are applied last), so an unsupported annotation type silently returns MUTAGEN "
               f"data instead of an error: {types}")


def check_declared_schema(session: Session, entry: dict | None) -> None:
    """The pinned tool config's ``return_schema`` against the live payload (checkout, offline)."""
    report = session.report
    name = f"{PREDICTION}[return_schema]"
    config = session.checkout / "src/tooluniverse/data/alphafold_tools.json"
    try:
        tools = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report.add("L1", name, "WARN", f"could not read {config.name}: {exc}")
        return
    tools = tools if isinstance(tools, list) else tools.get("tools", [])
    declared_tool = next((t for t in tools if t.get("name") == PREDICTION), None)
    declared = declared_fields((declared_tool or {}).get("return_schema"))
    if entry is None:
        report.add("L1", name, "WARN", "no live model entry to compare the declared schema with")
        return
    if not declared:
        report.add("L1", name, "WARN", f"no array-of-object return_schema for {PREDICTION} in {config.name}")
        return
    live = set(entry)
    missing, extra = sorted(declared - live), sorted(live - declared)
    if not missing and not extra:
        report.add("L1", name, "PASS", f"{len(declared)} declared fields match the live model entry")
        return
    report.add("L1", name, "WARN",
               f"the declared return_schema no longer describes the live payload: {len(missing)} declared "
               f"field(s) absent ({missing[:6]}), {len(extra)} live field(s) undeclared ({extra[:6]}); it also "
               "describes the bare API payload while the tool wraps it in {status, data, metadata}",
               declared_absent=missing, live_undeclared=extra)


def check_error_paths(session: Session) -> None:
    """Invalid identifiers: upstream answers HTTP 400/404, the tool reports them in-band."""
    call = session.call
    for label, arguments in ((f"entry name {ENTRY_NAME}", {"qualifier": ENTRY_NAME}),
                             (f"invalid accession {BAD_ACCESSION}", {"qualifier": BAD_ACCESSION}),
                             ("no qualifier", {})):
        THROTTLE.wait()
        check_rejected(call, f"{PREDICTION}[{label}]", PREDICTION, arguments, in_band=_in_band,
                       on_accept=_accepted)


def _accepted(result: dict) -> tuple:
    """An identifier the tool documents as unusable was answered anyway: a WARN, not a FAIL.

    Upstream gaining support for entry names would be an improvement; only the
    documentation would then be wrong."""
    return ("WARN", "the identifier the tool documents as unusable was accepted and answered; "
                    f"the description is out of date: {text_of(result)[:160]}")


def _in_band(result: dict) -> str | None:
    try:
        return in_band_error(json.loads(text_of(result)))
    except json.JSONDecodeError:
        return None


def run_l1(session: Session) -> None:
    report = session.report
    try:
        entry = check_monomer_prediction(session)
        check_isoform_prediction(session)
        check_alias(session, entry)
        check_ignored_parameter(session, entry)
        check_summary(session, entry)
        check_annotations(session, entry)
        check_overridden_type(session)
        check_declared_schema(session, entry)
        check_error_paths(session)
    except RuntimeError as exc:                   # a reference download failed
        report.add("L1", "reference", "FAIL", str(exc))


SMOKE = Smoke(
    server="alphafold_db",
    run_l1=run_l1,
    packages=("tooluniverse", "mcp", "fastmcp", "requests"),
    pass_proxies=True,
    # No expected_cwd_files: these tools write nothing, and a ToolUniverse
    # workspace appearing in the server cwd (./.tooluniverse, which would seed the
    # default profile and ignore --include-tools) must show up as a leftover. That
    # failure mode is probed once, for the whole ToolUniverse pin, in servers/arxiv.py.
)
