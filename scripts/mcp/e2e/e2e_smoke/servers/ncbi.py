"""Direct (agent-free) E2E smoke test for the pinned ToolUniverse NCBI Gene/Protein MCP tools.

Run with the shared ToolUniverse virtualenv (the checkout is shared with the
``arxiv`` and ``alphafold_db`` manifest ids)::

    ~/mcp/tooluniverse/.venv/bin/python scripts/mcp/e2e/smoke.py ncbi \\
        --config ~/mcp/ncbi.mcp.json

Needs network access to eutils.ncbi.nlm.nih.gov and api.ncbi.nlm.nih.gov (both
the server and the references query the live public APIs; no key is needed).

The three tools are thin E-utilities wrappers (``esearch``/``esummary``), so
there is no date window to pin; stability comes from **immutable identifiers**
and from reference values fetched here, outside the server process, from
*different* NCBI endpoints than the tool uses:

* the gene ID behind a symbol, and the gene's locus, from ``efetch
  rettype=gene_table`` (never ``efetch db=gene retmode=xml``: 34 MB for TP53);
* cytogenetic band, synonyms and OMIM ids from the Datasets API
  (``/datasets/v2alpha/gene/id/<id>``) -- a *soft* reference: ``v2alpha`` may
  change, so a value it does not provide is a WARN, never a FAIL;
* a protein's length and title from ``efetch rettype=fasta``, addressed by **GI
  number** so the record cannot drift to a newer version.

Annotation-dependent values (the gene ``summary`` text, ``exoncount``,
``geneweight``, ``createdate``/``updatedate``, and result counts for
non-symbol queries) are reported, never asserted.

Upstream defects that do not make a correctly used tool wrong (failures
returned in-band with HTTP 200 and ``isError=false``, ``retmax=0`` accepted,
and the E-utilities ``tool=``/``email=`` identification the wrapper never
sends) are WARN; wrong values for correct inputs are FAIL.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from ..client import text_of
from ..runner import Caller, Session, Smoke, check_rejected

GENE_SEARCH = "NCBIGene_search"
GENE_SUMMARY = "NCBIGene_get_summary"
PROTEIN_SUMMARY = "NCBIProtein_get_summary"

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DATASETS = "https://api.ncbi.nlm.nih.gov/datasets/v2alpha"
USER_AGENT = "asibench-mcp-e2e-smoke/1 (reference client)"
# E-utilities allows 3 requests/s without an API key, shared between the server
# and this script's reference requests.
API_GAP_S = 0.4

GENE = ("TP53", "7157")
SECOND_GENE = ("BRCA1", "672")
GENE_TERM = "{symbol}[Symbol] AND Homo sapiens[Organism]"
FREE_TEXT_TERM = "insulin AND Homo sapiens[Organism]"
FREE_TEXT_RETMAX = 3
PROTEIN_GI = "119395750"          # NP_006112.3, keratin type II cytoskeletal 1
PROTEIN_FACTS = {"accessionversion": "NP_006112.3", "caption": "NP_006112", "moltype": "aa",
                 "sourcedb": "refseq", "taxid": 9606}
TAXID = 9606
UNKNOWN_SYMBOL = "ZZZNOTAGENE"
UNKNOWN_GENE_ID = "99999999"


# --------------------------------------------------------------------------
# Pure helpers (stdlib only; unit-tested offline)
# --------------------------------------------------------------------------

def unwrap(payload):
    """``(data, error)`` for a BaseRESTTool result.

    The tool wraps a successful payload as ``{"status": "success", "data": ...,
    "url": ...}``; a transport-level failure becomes ``{"status": "error",
    "error": ..., "status_code": ...}`` with ``isError=false``. A real
    ``isError=true`` result arrives as the runner's ``{"_isError": True,
    "_text": ...}`` sentinel and must read as an error too, so that a server
    which starts rejecting bad input properly is never mistaken for one
    returning a wrong payload."""
    if not isinstance(payload, dict):
        return payload, None
    if payload.get("_isError"):
        return None, str(payload.get("_text") or "isError=true")
    if payload.get("status") == "success" and "data" in payload:
        return payload["data"], None
    message = payload.get("error") or payload.get("detail")
    if message or payload.get("status") == "error":
        code = payload.get("status_code")
        return None, f"{message}{f' (HTTP {code})' if code else ''}"
    return payload, None


def search_result(data) -> dict:
    """The ``esearchresult`` object of an esearch payload."""
    return (data or {}).get("esearchresult", {}) if isinstance(data, dict) else {}


def summary_records(data) -> tuple[list[str], dict]:
    """``(uids, {uid: record})`` of an esummary payload."""
    result = (data or {}).get("result", {}) if isinstance(data, dict) else {}
    uids = [str(uid) for uid in result.get("uids") or []]
    return uids, {uid: result.get(uid) for uid in uids if isinstance(result.get(uid), dict)}


def search_not_found(data) -> str | None:
    """The in-band "nothing found" message of an esearch payload, if any."""
    result = search_result(data)
    if not result:
        return None
    missing = (result.get("errorlist") or {}).get("phrasesnotfound") or []
    messages = (result.get("warninglist") or {}).get("outputmessages") or []
    if missing or (result.get("count") == "0" and messages):
        return f"phrasesnotfound={missing}, outputmessages={messages}"
    return None


def summary_not_found(data) -> str | None:
    """The in-band failure of an esummary payload, if any.

    Three shapes, all HTTP 200: a top-level ``error`` (an id that is not a uid at
    all), a per-uid ``error`` (a well-formed uid with no record), and an empty
    ``uids`` list with no message at all (a syntactically valid but unknown GI)."""
    if not isinstance(data, dict):
        return None
    top = data.get("error")
    if top:
        return f"error={top!r}"
    uids, records = summary_records(data)
    errors = {uid: record["error"] for uid, record in records.items() if record.get("error")}
    if errors:
        return f"result[{sorted(errors)}].error={sorted(errors.values())}"
    if "result" in data and not uids:
        return "result.uids is empty and no error is reported"
    return None


GENE_TABLE_ID = re.compile(r"^Gene ID:\s*(\d+)", re.MULTILINE)
# A plus-strand gene has no strand marker at all (measured 2026-10-07 on CFTR, SOD1
# and HBA1), so the group is optional; the orientation is taken from the coordinates
# either way and the marker, when present, is only cross-checked against them.
GENE_TABLE_LOCUS = re.compile(
    r"^Reference\s+\S+.*?\s(?P<accession>[A-Z]{2}_\d+\.\d+)\s+"
    r"(?:\((?P<strand>plus|minus) strand\)\s+)?from:\s*(?P<start>\d+)\s+to:\s*(?P<stop>\d+)", re.MULTILINE)


def parse_gene_table(text: str) -> dict:
    """Gene ID, title line and the reference-assembly locus of an ``efetch rettype=gene_table``.

    The locus line is 1-based inclusive, counts down on the minus strand and spells the
    strand out only when it is the minus one::

        Reference GRCh38.p14 Primary Assembly NC_000017.11  (minus strand) from: 7687490 to: 7668421
        Reference GRCh38.p14 Primary Assembly NC_000021.9  from: 31659693 to: 31668931

    ``strand`` therefore comes from the coordinates (as it does for the esummary locus),
    and ``strand_marker`` keeps what the line said so a disagreement can be reported.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    gene_id = GENE_TABLE_ID.search(text)
    locus = GENE_TABLE_LOCUS.search(text)
    out = {"gene_id": gene_id.group(1) if gene_id else None,
           "title": lines[0].strip() if lines else "",
           "accession": None, "strand": None, "start": None, "stop": None, "strand_marker": None}
    if locus:
        start, stop = int(locus.group("start")), int(locus.group("stop"))
        out.update(accession=locus.group("accession"), strand="minus" if start > stop else "plus",
                   start=start, stop=stop, strand_marker=locus.group("strand"))
    return out


def gene_table_symbol(title: str) -> str:
    """``TP53`` from the gene_table title line ``TP53 tumor protein p53[Homo sapiens]``."""
    return title.split(" ", 1)[0].strip() if title else ""


def normalise_locus(info: dict) -> dict:
    """An esummary ``genomicinfo`` entry as a 1-based, strand-labelled locus.

    ``chrstart``/``chrstop`` are **0-based** and ``chrstart > chrstop`` is how a
    minus-strand gene is expressed; ``gene_table`` reports the same locus
    1-based with the strand spelled out. Both ends shift by +1."""
    start, stop = info.get("chrstart"), info.get("chrstop")
    if not isinstance(start, int) or not isinstance(stop, int):
        return {"accession": info.get("chraccver"), "strand": None, "start": None, "stop": None}
    return {"accession": info.get("chraccver"),
            "strand": "minus" if start > stop else "plus",
            "start": start + 1, "stop": stop + 1}


def locus_diffs(got: dict, reference: dict) -> list[str]:
    """Differences between a normalised esummary locus and the gene_table locus."""
    return [f"{key}: esummary {got.get(key)!r} vs gene_table {reference.get(key)!r}"
            for key in ("accession", "strand", "start", "stop") if got.get(key) != reference.get(key)]


def fasta_record(text: str) -> tuple[str, int]:
    """``(header without '>', residue count)`` of a single-record FASTA."""
    header, residues = "", 0
    for line in text.splitlines():
        if line.startswith(">"):
            if header:
                break
            header = line[1:].strip()
        else:
            residues += len(line.strip())
    return header, residues


def aliases_of(record: dict) -> set[str]:
    """``otheraliases`` of a gene esummary record as a set."""
    return {part.strip() for part in (record.get("otheraliases") or "").split(",") if part.strip()}


# --------------------------------------------------------------------------
# Independent references (other NCBI endpoints, not the tool's own)
# --------------------------------------------------------------------------

class Throttle:
    """Keeps >= API_GAP_S between NCBI requests made by the server *or* by this script."""

    def __init__(self, gap: float = API_GAP_S) -> None:
        self.gap = gap
        self.last = 0.0

    def wait(self) -> None:
        delay = self.last + self.gap - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self.last = time.monotonic()


THROTTLE = Throttle()


def http_get(url: str, timeout: float = 90.0, attempts: int = 4) -> bytes:
    """GET with backoff; NCBI answers 429 when the shared 3 req/s budget is exceeded."""
    error = None
    for attempt in range(attempts):
        THROTTLE.wait()
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            error = exc
            if exc.code not in (429, 500, 502, 503, 504):
                raise RuntimeError(f"GET {url} failed: HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            error = exc
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed: {error}")


def reference_gene_table(gene_id: str) -> dict:
    """``efetch db=gene rettype=gene_table`` (31 KB for TP53), parsed."""
    params = urllib.parse.urlencode({"db": "gene", "id": gene_id, "rettype": "gene_table",
                                     "retmode": "text"})
    return parse_gene_table(http_get(f"{EUTILS}/efetch.fcgi?{params}").decode("utf-8", "replace"))


def reference_datasets(gene_id: str) -> dict:
    """The Datasets ``v2alpha`` gene report (soft reference: alpha interface)."""
    document = json.loads(http_get(f"{DATASETS}/gene/id/{gene_id}").decode("utf-8", "replace"))
    reports = document.get("reports") or []
    return (reports[0].get("gene") or {}) if reports else {}


def reference_protein_fasta(gi: str) -> tuple[str, int]:
    params = urllib.parse.urlencode({"db": "protein", "id": gi, "rettype": "fasta", "retmode": "text"})
    return fasta_record(http_get(f"{EUTILS}/efetch.fcgi?{params}").decode("utf-8", "replace"))


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

def check_symbol_search(session: Session) -> None:
    """A fielded symbol query must find exactly the gene ID that gene_table reports."""
    call, report = session.call, session.report
    symbol, gene_id = GENE
    name = f"{GENE_SEARCH}[{symbol} fielded]"
    data = call_data(call, name, GENE_SEARCH, {"term": GENE_TERM.format(symbol=symbol)})
    if data is None:
        return
    result = search_result(data)
    table = reference_gene_table(gene_id)
    problems = []
    if table["gene_id"] != gene_id:
        report.add("L1", f"{name}[reference]", "FAIL",
                   f"gene_table for {gene_id} reports Gene ID {table['gene_id']}; fix the smoke inputs")
        return
    if result.get("idlist") != [gene_id]:
        problems.append(f"idlist {result.get('idlist')} != [{gene_id!r}]")
    if result.get("count") != "1":
        problems.append(f"count {result.get('count')!r} != '1' for a single-symbol query")
    if gene_table_symbol(table["title"]) != symbol:
        problems.append(f"gene_table title {table['title']!r} does not start with {symbol}")
    report.add("L1", name, "FAIL" if problems else "PASS",
               "; ".join(problems) if problems else
               f"idlist == [{gene_id!r}], the same gene the gene_table of {gene_id} reports "
               f"({table['title'][:60]!r})",
               querytranslation=result.get("querytranslation"))


def check_retmax(session: Session) -> None:
    """``retmax`` must bound the returned list (the total count is annotation-dependent)."""
    call, report = session.call, session.report
    name = f"{GENE_SEARCH}[retmax={FREE_TEXT_RETMAX}]"
    data = call_data(call, name, GENE_SEARCH, {"term": FREE_TEXT_TERM, "retmax": FREE_TEXT_RETMAX})
    if data is None:
        return
    result = search_result(data)
    idlist = result.get("idlist") or []
    problems = []
    if len(idlist) > FREE_TEXT_RETMAX:
        problems.append(f"{len(idlist)} ids returned for retmax={FREE_TEXT_RETMAX}")
    if result.get("retmax") != str(FREE_TEXT_RETMAX):
        problems.append(f"retmax echoed as {result.get('retmax')!r}")
    if not idlist:
        problems.append("no ids at all for a free-text query")
    if any(not str(uid).isdigit() for uid in idlist):
        problems.append(f"non-numeric gene ids: {idlist}")
    report.add("L1", name, "FAIL" if problems else "PASS",
               "; ".join(problems) if problems else
               f"{len(idlist)} of {result.get('count')} hits returned, all numeric gene ids",
               idlist=idlist, count=result.get("count"))

    name = f"{GENE_SEARCH}[retmax=0]"
    data = call_data(call, name, GENE_SEARCH, {"term": FREE_TEXT_TERM, "retmax": 0})
    if data is None:
        return
    result = search_result(data)
    idlist = result.get("idlist") or []
    if idlist:
        report.add("L1", name, "WARN",
                   f"retmax=0 did not bound the list: {len(idlist)} ids came back, so the falsy value was "
                   "dropped somewhere and esearch applied its own default",
                   idlist=idlist[:5])
    elif (result.get("count") or "0") != "0":
        report.add("L1", name, "WARN",
                   f"retmax=0 is accepted and returns an empty idlist while reporting "
                   f"count={result.get('count')!r}, so a caller cannot tell it from 'nothing found'")
    else:
        report.add("L1", name, "PASS", f"idlist={idlist}, count={result.get('count')!r}")


def check_gene_summary(session: Session) -> None:
    """The gene record: nomenclature, locus (0-based -> 1-based), band, OMIM, aliases."""
    call, report = session.call, session.report
    symbol, gene_id = GENE
    name = f"{GENE_SUMMARY}[{gene_id}]"
    data = call_data(call, name, GENE_SUMMARY, {"id": gene_id})
    if data is None:
        return
    uids, records = summary_records(data)
    if uids != [gene_id] or gene_id not in records:
        report.add("L1", name, "FAIL", f"uids {uids} without a record for {gene_id}")
        return
    record = records[gene_id]
    table = reference_gene_table(gene_id)
    reference_ok = bool(table["title"]) and table["accession"] is not None and table["start"] is not None
    if not reference_ok:
        # Our own reference could not be parsed: say so instead of blaming the tool,
        # and skip every comparison that would rest on it.
        report.add("L1", f"{name}[reference]", "FAIL",
                   f"could not read a symbol and a reference-assembly locus out of the gene_table of "
                   f"{gene_id} (title {table['title'][:60]!r}); fix the smoke's parser")
    problems, soft = [], []
    if record.get("nomenclaturesymbol") != symbol:
        problems.append(f"nomenclaturesymbol {record.get('nomenclaturesymbol')!r} != {symbol!r}")
    if reference_ok and gene_table_symbol(table["title"]) != record.get("nomenclaturesymbol"):
        problems.append(f"gene_table title {table['title'][:40]!r} disagrees with nomenclaturesymbol")
    if (record.get("organism") or {}).get("taxid") != TAXID:
        problems.append(f"organism.taxid {(record.get('organism') or {}).get('taxid')!r} != {TAXID}")
    # Compare the locus of the same reference assembly the gene_table reports, not
    # whatever sits at index 0 (an added alt-locus entry must not read as an error).
    infos = [i for i in (record.get("genomicinfo") or []) if isinstance(i, dict)]
    info = next((i for i in infos if i.get("chraccver") == table["accession"]), infos[0] if infos else {})
    locus = normalise_locus(info)
    if not reference_ok:
        soft.append("gene_table reference unusable, so the locus is unchecked (see the [reference] check)")
    elif len(infos) > 1 and info.get("chraccver") != table["accession"]:
        soft.append(f"no genomicinfo entry for {table['accession']}; compared {info.get('chraccver')!r}")
    else:
        problems.extend(locus_diffs(locus, table))
    if table.get("strand_marker") and table["strand_marker"] != table["strand"]:
        # The orientation is read from the coordinates; a marker that contradicts them is
        # an upstream inconsistency, not a reason to mislabel the locus.
        soft.append(f"gene_table says ({table['strand_marker']} strand) while its own coordinates "
                    f"run {table['start']}..{table['stop']}")
    try:
        gene = reference_datasets(gene_id)
    except RuntimeError as exc:
        gene = {}
        soft.append(f"Datasets v2alpha unavailable: {exc}")
    bands = [m.get("map_value") for m in (gene.get("map_locations") or []) if m.get("map_type") == "Cytogenetic"]
    # Datasets is v2alpha: a disagreement with it is reported, never a FAIL (the
    # hard locus check above already rests on the stable gene_table endpoint).
    if bands and record.get("maplocation") not in bands:
        soft.append(f"maplocation {record.get('maplocation')!r} not in Datasets {bands}")
    elif not bands:
        soft.append(f"Datasets gave no cytogenetic band; maplocation {record.get('maplocation')!r} unchecked")
    if gene.get("orientation") and locus["strand"] and gene["orientation"] != locus["strand"]:
        soft.append(f"strand {locus['strand']!r} != Datasets orientation {gene['orientation']!r}")
    omim = gene.get("omim_ids")
    if omim and record.get("mim") != list(omim):
        soft.append(f"mim {record.get('mim')!r} != Datasets omim_ids {list(omim)!r}")
    synonyms = set(gene.get("synonyms") or [])
    unknown = sorted(synonyms - aliases_of(record))
    if synonyms and unknown:
        soft.append(f"Datasets synonyms missing from otheraliases: {unknown}")
    report.add("L1", name, "FAIL" if problems else ("WARN" if soft else "PASS"),
               "; ".join(problems[:4]) if problems else "; ".join(soft[:3]) if soft else
               f"{symbol}: {record.get('maplocation')} on {locus['accession']} ({locus['strand']} strand) "
               f"{locus['start']}..{locus['stop']}, identical to the gene_table locus after the 0-based -> "
               "1-based shift",
               esummary_locus_0based={"chrstart": info.get("chrstart"), "chrstop": info.get("chrstop")},
               normalised_locus=locus, gene_table_locus=table,
               reported_only={"exoncount": info.get("exoncount"), "geneweight": record.get("geneweight"),
                              "summary_chars": len(record.get("summary") or "")})
    if not (record.get("summary") or "").strip():
        report.add("L1", f"{name}[summary text]", "WARN", "the record carries no summary text")


def check_multiple_ids(session: Session) -> None:
    """A comma-separated ``id`` must return one record per uid, in the requested order."""
    call, report = session.call, session.report
    ids = [GENE[1], SECOND_GENE[1]]
    name = f"{GENE_SUMMARY}[{','.join(ids)}]"
    data = call_data(call, name, GENE_SUMMARY, {"id": ",".join(ids)})
    if data is None:
        return
    uids, records = summary_records(data)
    symbols = {uid: records.get(uid, {}).get("nomenclaturesymbol") for uid in uids}
    expected = {GENE[1]: GENE[0], SECOND_GENE[1]: SECOND_GENE[0]}
    problems = []
    if uids != ids:
        problems.append(f"uids {uids} != {ids}")
    if symbols != expected:
        problems.append(f"symbols {symbols} != {expected}")
    report.add("L1", name, "FAIL" if problems else "PASS",
               "; ".join(problems) if problems else f"both records returned in order: {symbols}")


def check_protein_summary(session: Session) -> None:
    """A protein addressed by GI number, against the FASTA of the same record."""
    call, report = session.call, session.report
    name = f"{PROTEIN_SUMMARY}[GI {PROTEIN_GI}]"
    data = call_data(call, name, PROTEIN_SUMMARY, {"id": PROTEIN_GI})
    if data is None:
        return
    uids, records = summary_records(data)
    if uids != [PROTEIN_GI] or PROTEIN_GI not in records:
        report.add("L1", name, "FAIL", f"uids {uids} without a record for {PROTEIN_GI}")
        return
    record = records[PROTEIN_GI]
    header, residues = reference_protein_fasta(PROTEIN_GI)
    problems = []
    if record.get("slen") != residues:
        problems.append(f"slen {record.get('slen')!r} != {residues} residues in the FASTA")
    expected_header = f"{record.get('accessionversion')} {record.get('title')}"
    if header != expected_header:
        problems.append(f"FASTA header {header!r} != accessionversion + title {expected_header!r}")
    for key, value in PROTEIN_FACTS.items():
        if record.get(key) != value:
            problems.append(f"{key} {record.get(key)!r} != {value!r}")
    report.add("L1", name, "FAIL" if problems else "PASS",
               "; ".join(problems[:4]) if problems else
               f"{record.get('accessionversion')}: {residues} aa, title identical to the FASTA header",
               reported_only={"createdate": record.get("createdate"), "updatedate": record.get("updatedate")})


def check_error_paths(session: Session) -> None:
    """NCBI answers HTTP 200 for both "nothing found" and "bad uid"; the error is in the payload."""
    call = session.call
    THROTTLE.wait()
    check_rejected(call, f"{GENE_SEARCH}[unknown symbol]", GENE_SEARCH,
                   {"term": GENE_TERM.format(symbol=UNKNOWN_SYMBOL)},
                   in_band=lambda result: _in_band(result, search_not_found),
                   on_accept=lambda result: _accepted(result, "esearch"))
    THROTTLE.wait()
    check_rejected(call, f"{GENE_SUMMARY}[unknown gene id]", GENE_SUMMARY, {"id": UNKNOWN_GENE_ID},
                   in_band=lambda result: _in_band(result, summary_not_found),
                   on_accept=lambda result: _accepted(result, "esummary"))
    THROTTLE.wait()
    check_rejected(call, f"{PROTEIN_SUMMARY}[non-numeric id]", PROTEIN_SUMMARY, {"id": UNKNOWN_SYMBOL},
                   in_band=lambda result: _in_band(result, summary_not_found),
                   on_accept=lambda result: _accepted(result, "esummary"))


def _accepted(result: dict, endpoint: str) -> tuple:
    """No error of any shape: either an empty-but-honest result, or real records.

    An ``idlist``/``uids`` that is empty is how E-utilities legitimately says
    "no hits"; reporting that as "invalid input accepted" would blame the server
    for behaving correctly. Only actual records for an identifier that cannot
    exist are a FAIL."""
    try:
        data, _ = unwrap(json.loads(text_of(result)))
    except json.JSONDecodeError:
        return ("FAIL", f"invalid input accepted with an unparsable payload: {text_of(result)[:160]}")
    found = (search_result(data).get("idlist") if endpoint == "esearch" else summary_records(data)[0]) or []
    if found:
        return ("FAIL", f"records returned for an identifier that cannot exist: {list(found)[:5]}")
    return ("WARN", f"{endpoint} reports no hits without an error field, so a caller cannot tell "
                    "'nothing found' from a request it got wrong")


def _in_band(result: dict, detect) -> str | None:
    """``detect(data)`` on the tool's payload: the wrapper's own error, or an in-band NCBI one."""
    try:
        payload = json.loads(text_of(result))
    except json.JSONDecodeError:
        return None
    data, error = unwrap(payload)
    return error or detect(data)


def check_identification(session: Session) -> None:
    """E-utilities asks every client to identify itself with ``tool=`` and ``email=``.

    Read from the pinned tool config in the checkout, so the check states what this
    revision actually sends rather than a remembered claim."""
    report = session.report
    name = f"{GENE_SEARCH}[E-utilities identification]"
    config = session.checkout / "src/tooluniverse/data/ncbi_gene_tools.json"
    try:
        tools = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report.add("L1", name, "WARN", f"could not read {config.name}: {exc}")
        return
    tools = tools if isinstance(tools, list) else tools.get("tools", [])
    sent, authenticated = {}, []
    for tool in tools:
        fields = tool.get("fields") or {}
        params = fields.get("params") or {}
        sent[tool.get("name")] = sorted(key for key in ("tool", "email", "api_key") if key in params)
        if fields.get("auth_param") or fields.get("auth_header"):
            authenticated.append(tool.get("name"))
    missing = {name_: keys for name_, keys in sent.items() if "tool" not in keys or "email" not in keys}
    if not missing:
        report.add("L1", name, "PASS", f"every tool sends tool= and email=: {sent}")
        return
    report.add("L1", name, "WARN",
               "the pinned configs send neither tool= nor email= (and no api_key is injected: "
               f"auth_param/auth_header on {authenticated or 'none'}), which E-utilities policy requires; "
               "without them NCBI cannot warn a heavy caller before blocking it, and the shared 3 req/s "
               "budget is all that is available",
               declared_params=sent)


def run_l1(session: Session) -> None:
    report = session.report
    try:
        check_symbol_search(session)
        check_retmax(session)
        check_gene_summary(session)
        check_multiple_ids(session)
        check_protein_summary(session)
        check_error_paths(session)
        check_identification(session)
    except RuntimeError as exc:                    # a reference request failed
        report.add("L1", "reference", "FAIL", str(exc))


SMOKE = Smoke(
    server="ncbi",
    run_l1=run_l1,
    packages=("tooluniverse", "mcp", "fastmcp", "requests"),
    pass_proxies=True,
    # No expected_cwd_files: these tools write nothing, and a ToolUniverse
    # workspace appearing in the server cwd (./.tooluniverse, which would seed the
    # default profile and ignore --include-tools) must show up as a leftover. That
    # failure mode is probed once, for the whole ToolUniverse pin, in servers/arxiv.py.
)
