"""Instance generator for the NCBI gene → protein card MCP E2E fake task.

A seeded RNG picks one curated case: a human gene symbol and the unversioned
RefSeq accession of one of its proteins. The agent has to find the gene by a
fielded symbol query, read the gene record of the single id the search returns,
read the protein record of the GI named in the input file, and report a small
card of identifiers.

There is no date window to pin here (E-utilities always answers about today's
build), so the protein side rests on an **immutable identifier**: the GI number
this generator resolves is bound to one sequence version, so ``protein_accver``,
``protein_len`` and the protein's taxonomy id cannot drift afterwards.

The gene side cannot be frozen that way and describes the current build: the
cytogenetic band is revised from time to time, the ``NC_…`` accession carries an
assembly version, and a symbol that matches one gene today could match two
later. A stale instance would then blame the agent for an upstream change, so
**generate right before the run** (seconds, 8 requests) rather than reusing an
old instance directory.

The reference is computed here, independently of the MCP server code, from the
raw E-utilities endpoints, and every value that can be cross-checked against a
*different* endpoint is:

* the gene id and the locus against ``efetch rettype=gene_table`` (never
  ``efetch db=gene retmode=xml``: 34 MB for TP53). ``chrstart``/``chrstop`` of
  the gene summary are **0-based**, and ``chrstart > chrstop`` is how a
  minus-strand gene is expressed; the gene_table reports the same locus 1-based
  and only names the strand when it is the minus one.
* the protein length and title against ``efetch rettype=fasta``, and against the
  ``annotated AA length`` the gene_table lists for the same accession.
* the cytogenetic band and the OMIM ids against the Datasets API
  (``/datasets/v2alpha/gene/id/<id>``) -- a *soft* reference: ``v2alpha`` may
  change, so a disagreement is recorded in the reference, never raised.

A curated case that no longer satisfies the assertions (a symbol that became
ambiguous, a gene that gained a second genomic record, a protein whose length
disagrees between the protein and the gene database) raises instead of
producing an instance that would blame the agent for an upstream change.

Needs network access to eutils.ncbi.nlm.nih.gov and api.ncbi.nlm.nih.gov (no
API key). Generate with ``asibench generate --sandbox task``.

Framework call: python generate_gt.py --output-dir <dir> --params '<json>'
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

INPUT_SPEC = [
    {"name": "card.json", "description": "gene symbol, the verbatim Entrez Gene query and the protein GI to read"},
]
OUTPUT_SPEC = [
    {"name": "result.json", "description": "gene id, symbol, cytogenetic band, chromosome accession, "
                                           "protein accession.version, protein length and taxonomy id"},
]
DEFAULT_PARAMS = {"seed": 0}

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DATASETS = "https://api.ncbi.nlm.nih.gov/datasets/v2alpha"
USER_AGENT = "asibench-mcp-e2e-generate/1"
# E-utilities allows 3 requests/s without an API key.
API_GAP_S = 0.4
TAXID = 9606
TERM = "{symbol}[Symbol] AND Homo sapiens[Organism]"

# Curated 2026-10-07: every symbol resolves to exactly one human gene id, the gene
# has a single genomic record on the reference assembly, its gene_table is small
# (0.8-32 KB) and the unversioned protein accession resolves to exactly one GI.
# Both strands are represented (the gene_table names only the minus one).
CASES = [
    {"symbol": "TP53", "protein": "NP_000537", "topic": "the tumour suppressor TP53"},
    {"symbol": "CFTR", "protein": "NP_000483", "topic": "the chloride channel gene CFTR"},
    {"symbol": "SOD1", "protein": "NP_000445", "topic": "the superoxide dismutase gene SOD1"},
    {"symbol": "HBA1", "protein": "NP_000549", "topic": "the alpha-globin gene HBA1"},
    {"symbol": "PAH", "protein": "NP_000268", "topic": "the phenylalanine hydroxylase gene PAH"},
]


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------

GENE_TABLE_ID = re.compile(r"^Gene ID:\s*(\d+)", re.MULTILINE)
# A plus-strand gene has no strand marker at all (measured 2026-10-07: CFTR, SOD1 and
# HBA1 plus; TP53, PAH minus), so the group is optional. The orientation is read from
# the coordinates, which carry it on both strands, and the marker is cross-checked:
#   Reference GRCh38.p14 Primary Assembly NC_000017.11  (minus strand) from: 7687490 to: 7668421
#   Reference GRCh38.p14 Primary Assembly NC_000007.14  from: 117480025 to: 117668665
GENE_TABLE_LOCUS = re.compile(
    r"^Reference\s+\S+.*?\s(?P<accession>[A-Z]{2}_\d+\.\d+)\s+"
    r"(?:\((?P<strand>plus|minus) strand\)\s+)?from:\s*(?P<start>\d+)\s+to:\s*(?P<stop>\d+)", re.MULTILINE)
GENE_TABLE_PROTEIN = re.compile(
    r"^protein\s+(?:\S.*?\s)?(?P<accver>[A-Z]P_\d+\.\d+)[^\n]*?annotated AA length:\s*(?P<length>\d+)",
    re.MULTILINE)


def parse_gene_table(text: str) -> dict:
    """Gene ID, title line, reference-assembly locus (1-based) and protein lengths."""
    lines = [line for line in text.splitlines() if line.strip()]
    gene_id = GENE_TABLE_ID.search(text)
    locus = GENE_TABLE_LOCUS.search(text)
    out = {"gene_id": gene_id.group(1) if gene_id else None,
           "title": lines[0].strip() if lines else "",
           "accession": None, "strand": None, "start": None, "stop": None, "strand_marker": None,
           "protein_lengths": {m.group("accver"): int(m.group("length"))
                               for m in GENE_TABLE_PROTEIN.finditer(text)}}
    if locus:
        start, stop = int(locus.group("start")), int(locus.group("stop"))
        out.update(accession=locus.group("accession"), strand="minus" if start > stop else "plus",
                   start=start, stop=stop, strand_marker=locus.group("strand"))
    return out


def normalise_locus(info: dict) -> dict:
    """An esummary ``genomicinfo`` entry as a 1-based, strand-labelled locus.

    ``chrstart``/``chrstop`` are 0-based and ``chrstart > chrstop`` means the minus
    strand; both ends shift by +1 to reach the gene_table's numbering."""
    start, stop = info.get("chrstart"), info.get("chrstop")
    if not isinstance(start, int) or not isinstance(stop, int):
        return {"accession": info.get("chraccver"), "strand": None, "start": None, "stop": None}
    return {"accession": info.get("chraccver"), "strand": "minus" if start > stop else "plus",
            "start": start + 1, "stop": stop + 1}


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


def build_case(seed: int) -> dict:
    case = dict(CASES[random.Random(seed).randrange(len(CASES))])
    return {**case, "term": TERM.format(symbol=case["symbol"]), "organism": "Homo sapiens"}


# --------------------------------------------------------------------------
# Raw endpoints
# --------------------------------------------------------------------------

_last_request = [0.0]


def http_get(url: str, attempts: int = 5, timeout: float = 90.0) -> bytes:
    """Throttled GET with backoff; NCBI answers 429 above 3 requests/s."""
    error = None
    for attempt in range(attempts):
        delay = _last_request[0] + API_GAP_S - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        _last_request[0] = time.monotonic()
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
    raise RuntimeError(f"GET {url} failed after {attempts} attempts: {error}")


def eutils_json(endpoint: str, **params) -> dict:
    params.setdefault("retmode", "json")
    return json.loads(http_get(f"{EUTILS}/{endpoint}.fcgi?{urllib.parse.urlencode(params)}"))


def eutils_text(endpoint: str, **params) -> str:
    params.setdefault("retmode", "text")
    return http_get(f"{EUTILS}/{endpoint}.fcgi?{urllib.parse.urlencode(params)}").decode("utf-8", "replace")


def one_uid(term: str, db: str) -> str:
    """The single uid an esearch term must resolve to."""
    result = eutils_json("esearch", db=db, term=term).get("esearchresult") or {}
    idlist = [str(uid) for uid in result.get("idlist") or []]
    if result.get("count") != "1" or len(idlist) != 1:
        raise RuntimeError(f"{db} query {term!r} is no longer unique: count={result.get('count')!r} "
                           f"idlist={idlist}")
    return idlist[0]


def summary_record(db: str, uid: str) -> dict:
    """The single esummary record for ``uid`` (the payload keys it by the uid)."""
    result = eutils_json("esummary", db=db, id=uid).get("result") or {}
    uids = [str(value) for value in result.get("uids") or []]
    record = result.get(uid)
    if uids != [uid] or not isinstance(record, dict) or record.get("error"):
        raise RuntimeError(f"{db} esummary for {uid} returned uids={uids} record={str(record)[:160]}")
    return record


def datasets_gene(gene_id: str) -> dict:
    """The Datasets ``v2alpha`` gene report (soft reference: an alpha interface)."""
    document = json.loads(http_get(f"{DATASETS}/gene/id/{gene_id}"))
    reports = document.get("reports") or []
    return (reports[0].get("gene") or {}) if reports else {}


# --------------------------------------------------------------------------
# Reference
# --------------------------------------------------------------------------

def build_reference(case: dict) -> dict:
    """Every value of the card, cross-checked against a second endpoint where one exists."""
    symbol, accession = case["symbol"], case["protein"]
    gene_id = one_uid(case["term"], "gene")
    table = parse_gene_table(eutils_text("efetch", db="gene", id=gene_id, rettype="gene_table"))
    if table["gene_id"] != gene_id:
        raise RuntimeError(f"gene_table of {gene_id} reports Gene ID {table['gene_id']}")
    if table["accession"] is None or table["start"] is None:
        raise RuntimeError(f"no reference-assembly locus in the gene_table of {gene_id} "
                           f"(title {table['title'][:60]!r})")
    if table["strand_marker"] and table["strand_marker"] != table["strand"]:
        raise RuntimeError(f"gene_table of {gene_id} says ({table['strand_marker']} strand) while its "
                           f"coordinates run {table['start']}..{table['stop']}")
    if not table["protein_lengths"]:
        raise RuntimeError(f"no 'annotated AA length' line in the gene_table of {gene_id}; the protein "
                           "cross-check would be silently skipped")

    gene = summary_record("gene", gene_id)
    infos = [info for info in (gene.get("genomicinfo") or []) if isinstance(info, dict)]
    if len(infos) != 1:
        raise RuntimeError(f"{symbol} ({gene_id}) has {len(infos)} genomicinfo entries "
                           f"({[i.get('chraccver') for i in infos]}); the card expects exactly one")
    locus = normalise_locus(infos[0])
    maplocation = (gene.get("maplocation") or "").strip()
    checks = {"gene_table_locus": {k: table[k] for k in ("accession", "strand", "start", "stop")},
              "esummary_locus_0based": {"chrstart": infos[0].get("chrstart"), "chrstop": infos[0].get("chrstop")},
              "esummary_locus_1based": locus}
    if gene.get("nomenclaturesymbol") != symbol:
        raise RuntimeError(f"nomenclaturesymbol {gene.get('nomenclaturesymbol')!r} != {symbol!r}")
    if table["title"].split(" ", 1)[0].strip() != symbol:
        raise RuntimeError(f"gene_table title {table['title'][:60]!r} does not start with {symbol}")
    if any(locus[key] != table[key] for key in ("accession", "strand", "start", "stop")):
        raise RuntimeError(f"gene summary locus {locus} != gene_table locus "
                           f"{ {k: table[k] for k in ('accession', 'strand', 'start', 'stop')} }")
    if not maplocation:
        raise RuntimeError(f"{symbol} ({gene_id}) has no maplocation")
    if (gene.get("organism") or {}).get("taxid") != TAXID:
        raise RuntimeError(f"organism.taxid {(gene.get('organism') or {}).get('taxid')!r} != {TAXID}")

    protein_gi = one_uid(f"{accession}[Accession]", "protein")
    protein = summary_record("protein", protein_gi)
    accver = str(protein.get("accessionversion") or "")
    header, residues = fasta_record(eutils_text("efetch", db="protein", id=protein_gi, rettype="fasta"))
    checks["fasta_header"] = header
    if not accver.startswith(f"{accession}."):
        raise RuntimeError(f"GI {protein_gi} is {accver!r}, not a version of {accession}")
    for key, expected in (("moltype", "aa"), ("sourcedb", "refseq"), ("taxid", TAXID)):
        if protein.get(key) != expected:
            raise RuntimeError(f"protein {accver}: {key} {protein.get(key)!r} != {expected!r}")
    if protein.get("slen") != residues:
        raise RuntimeError(f"protein {accver}: slen {protein.get('slen')!r} != {residues} residues in the FASTA")
    if header != f"{accver} {protein.get('title')}":
        raise RuntimeError(f"FASTA header {header!r} != accessionversion + title")
    annotated = table["protein_lengths"].get(accver)
    checks["gene_table_annotated_aa_length"] = annotated
    if annotated is not None and annotated != residues:
        raise RuntimeError(f"{accver}: gene_table annotates {annotated} aa, the protein record {residues}")

    try:
        report = datasets_gene(gene_id)
    except (RuntimeError, ValueError) as exc:          # an alpha interface: never fatal
        checks["datasets"] = {"unavailable": str(exc)}
        report = {}
    if report:
        bands = [m.get("map_value") for m in (report.get("map_locations") or [])
                 if m.get("map_type") == "Cytogenetic"]
        checks["datasets"] = {"map_locations": bands, "omim_ids": report.get("omim_ids"),
                              "orientation": report.get("orientation"),
                              "band_agrees": maplocation in bands if bands else None,
                              "omim_agrees": (list(report["omim_ids"]) == (gene.get("mim") or [])
                                              if report.get("omim_ids") else None),
                              "orientation_agrees": (report.get("orientation") == locus["strand"]
                                                     if report.get("orientation") else None)}
    return {
        "symbol": symbol,
        "term": case["term"],
        "gene_id": gene_id,
        "gene_ids": [gene_id],
        "maplocation": maplocation,
        "chr_accession": locus["accession"],
        "taxid": TAXID,
        "protein_accession": accession,
        "protein_gi": protein_gi,
        "protein_gis": [protein_gi],
        "protein_accver": accver,
        "protein_len": residues,
        "protein_title": protein.get("title"),
        "gene_table_title": table["title"],
        "cross_checks": checks,
        "reported_only": {"gene_summary_chars": len(gene.get("summary") or ""),
                          "mim": gene.get("mim"), "otheraliases": gene.get("otheraliases"),
                          "createdate": protein.get("createdate"), "updatedate": protein.get("updatedate")},
        "mcp_tools": ["NCBIGene_search", "NCBIGene_get_summary", "NCBIProtein_get_summary"],
    }


def render_prompts(task_dir: Path, output_dir: Path, case: dict) -> None:
    for level in ("b1", "b2", "b3", "b4"):
        text = (task_dir / f"prompt_{level}.md").read_text(encoding="utf-8")
        (output_dir / f"prompt_{level}.md").write_text(text.replace("{{topic}}", case["topic"]),
                                                      encoding="utf-8")


def generate(output_dir: Path, params: dict) -> dict:
    p = {**DEFAULT_PARAMS, **params}
    t0 = time.time()
    output_dir = Path(output_dir)
    data_dir, ref_dir = output_dir / "data", output_dir / "reference"
    data_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)

    case = build_case(int(p["seed"]))
    reference = build_reference(case)
    card = {
        "symbol": case["symbol"],
        "organism": case["organism"],
        "term": case["term"],
        "protein_gi": reference["protein_gi"],
        "notes": {
            "term": "pass this string verbatim as the Entrez Gene query; it resolves to exactly one gene id",
            "gene_record": "the gene has exactly one genomic record; report its chromosome accession.version",
            "maplocation": "the cytogenetic band exactly as the gene record reports it",
            "protein_gi": "read the protein record of this GI number; it is bound to one sequence version",
        },
    }
    (data_dir / "card.json").write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
    (ref_dir / "reference.json").write_text(
        json.dumps({**reference, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                   indent=2) + "\n", encoding="utf-8")

    render_prompts(Path(__file__).resolve().parent, output_dir, case)
    meta = {
        "params_used": p,
        "input_files": [spec["name"] for spec in INPUT_SPEC],
        "reference_files": ["reference.json"],
        "generation_time_seconds": round(time.time() - t0, 2),
    }
    (output_dir / "instance_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--params", type=str, default="{}")
    args = parser.parse_args()
    print(json.dumps(generate(args.output_dir, json.loads(args.params)), indent=2))
