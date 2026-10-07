"""Instance generator for the AlphaFold DB isoform profile MCP E2E fake task.

A seeded RNG picks one curated UniProt accession with several isoform models in
AlphaFold DB. The agent has to read the models of that accession, rank the
entries the input file lists by sequence length, read the canonical length
from the summary endpoint, and profile the per-residue AlphaMissense scores of
the canonical model (the highest-scoring residue and how many residues score
above a threshold).

There is no date window to pin, so the instance is made stable by the input
file instead: it **lists the entry ids** to rank, read here at generation time,
so an isoform model that AlphaFold DB adds later does not change the answer.
Nothing that moves with the model version is used (pLDDT, URLs, version
numbers, the growing ``structures`` list of the summary); the AlphaMissense
scores are a fixed release keyed to the UniProt sequence. A sequence revision
in UniProt would still change a length, so generate right before a run.

The reference is computed here, independently of the MCP server code, from the
raw AlphaFold DB API, and cross-checked where a second source exists:

* every model's ``sequenceChecksum`` is the md5 of its own sequence, its entry
  id names its own isoform accession, and the summary's canonical length and
  checksum equal those of ``AF-<acc>-F1``;
* every per-residue score equals the mean of that position's 19
  ``am_pathogenicity`` rows in the AlphaMissense ``-aa-substitutions.csv`` the
  annotation itself links to (rounded to 4 decimals upstream).

A curated case that no longer allows a unique answer raises instead of
producing an instance: two listed isoforms of equal length (no unique ranking),
a tied highest score (no unique residue), a score exactly at the threshold
(where "above" and "at least" differ), or a payload large enough to come near
the server's 100 000-character truncation or the client's MCP output cap.

Needs network access to alphafold.ebi.ac.uk (no key). Generate with
``asibench generate --sandbox task``.

Framework call: python generate_gt.py --output-dir <dir> --params '<json>'
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

INPUT_SPEC = [
    {"name": "profile.json", "description": "UniProt accession, the AlphaFold DB entry ids to rank and the "
                                            "AlphaMissense score threshold"},
]
OUTPUT_SPEC = [
    {"name": "result.json", "description": "entry ids by length, longest entry, canonical length, highest-scoring "
                                           "residue and its score, residues above the threshold"},
]
DEFAULT_PARAMS = {"seed": 0}

API = "https://alphafold.ebi.ac.uk/api"
USER_AGENT = "asibench-mcp-e2e-generate/1"
API_GAP_S = 0.5                     # EBI publishes no fixed limit; be a polite client
AM_THRESHOLD = 0.9
AM_TOL = 5e-5                       # region values are the CSV means rounded to 4 decimals
# SMCP serialises a result compactly and truncates above 100 000 characters, and
# Claude Code replaces an MCP result above ~25 000 tokens (its default output cap)
# with a file pointer; number-heavy JSON runs ~2-3 characters per token, so stay
# well under both, with room for the tool's {status, data, metadata} wrapper.
MAX_PAYLOAD_CHARS = 40_000

# Curated 2026-10-07 (model v6): several isoform models of pairwise different
# length, a unique highest AlphaMissense score, no score exactly at 0.9, and every
# payload under 35 000 characters. In four of the five the longest model is an
# isoform, not the canonical AF-<acc>-F1.
CASES = [
    {"accession": "P04637", "protein": "TP53", "topic": "the tumour suppressor p53 (TP53)"},
    {"accession": "P42771", "protein": "CDKN2A", "topic": "the cyclin-dependent kinase inhibitor 2A (CDKN2A)"},
    {"accession": "P55957", "protein": "BID", "topic": "the BH3-interacting death agonist (BID)"},
    {"accession": "P63000", "protein": "RAC1", "topic": "the small GTPase Rac1 (RAC1)"},
    {"accession": "Q07812", "protein": "BAX", "topic": "the apoptosis regulator BAX"},
]


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------

def build_case(seed: int) -> dict:
    case = dict(CASES[random.Random(seed).randrange(len(CASES))])
    return {**case, "am_threshold": AM_THRESHOLD}


def md5_of(sequence: str) -> str:
    return hashlib.md5(sequence.encode("utf-8")).hexdigest()


def isoform_of(accession: str, base: str) -> bool:
    """``P04637`` itself or one of its isoform accessions (``P04637-2``)."""
    prefix, sep, suffix = (accession or "").partition("-")
    return prefix == base and (not sep or suffix.isdigit())


def rank_by_length(lengths: dict[str, int]) -> list[str]:
    """Entry ids, longest first. Lengths must be pairwise different (asserted by the
    caller), so no tie-break is needed and none is invented."""
    return sorted(lengths, key=lambda entry_id: -lengths[entry_id])


def region_scores(block: dict) -> dict[int, tuple[str, float]]:
    """``{position: (value as written, value)}`` of the single-residue regions."""
    out = {}
    for region in block.get("regions") or []:
        start, end = region.get("start"), region.get("end")
        if start is None or start != end:
            raise RuntimeError(f"unexpected multi-residue region {region}")
        text = str(region.get("annotation_value"))
        out[int(start)] = (text, float(text))
    return out


def am_means(csv_text: str) -> dict[int, float]:
    """``{position: mean am_pathogenicity}`` of an AlphaMissense substitutions CSV
    (``protein_variant,am_pathogenicity,am_class`` with variants like ``M1A``)."""
    lines = [line for line in csv_text.splitlines() if line.strip()]
    header = lines[0].split(",") if lines else []
    if "protein_variant" not in header or "am_pathogenicity" not in header:
        return {}
    variant_at, score_at = header.index("protein_variant"), header.index("am_pathogenicity")
    values: dict[int, list[float]] = {}
    for line in lines[1:]:
        cells = line.split(",")
        digits = "".join(c for c in cells[variant_at][1:] if c.isdigit())
        values.setdefault(int(digits), []).append(float(cells[score_at]))
    return {position: statistics.fmean(scores) for position, scores in values.items()}


def profile_scores(scores: dict[int, tuple[str, float]], threshold: float) -> dict:
    """Highest-scoring residue, its score and the count strictly above ``threshold``.
    Raises when the answer would not be unique."""
    best = max(value for _, value in scores.values())
    top = sorted(position for position, (_, value) in scores.items() if value == best)
    if len(top) != 1:
        raise RuntimeError(f"highest AlphaMissense score {best} is shared by residues {top}")
    at_threshold = sorted(position for position, (_, value) in scores.items() if value == threshold)
    if at_threshold:
        raise RuntimeError(f"residues {at_threshold} score exactly {threshold}: 'above' would be ambiguous")
    return {"max_am_residue": top[0], "max_am_score": best, "max_am_score_text": scores[top[0]][0],
            "n_am_above_threshold": sum(value > threshold for _, value in scores.values())}


# --------------------------------------------------------------------------
# Raw endpoints
# --------------------------------------------------------------------------

_last_request = [0.0]


def http_get(url: str, attempts: int = 4, timeout: float = 90.0) -> str:
    """Throttled GET with backoff on transient upstream errors (AFDB answers 500/503)."""
    error = None
    for attempt in range(attempts):
        delay = _last_request[0] + API_GAP_S - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        _last_request[0] = time.monotonic()
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            error = exc
            if exc.code not in (429, 500, 502, 503, 504):
                raise RuntimeError(f"GET {url} failed: HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            error = exc
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed after {attempts} attempts: {error}")


def api_json(path: str):
    """A JSON document of the AlphaFold DB API, refused if the tool's answer would come
    near the server's truncation or the client's output cap."""
    payload = json.loads(http_get(f"{API}/{path}"))
    size = len(json.dumps(payload, ensure_ascii=False))
    if size > MAX_PAYLOAD_CHARS:
        raise RuntimeError(f"/{path} is {size} characters, above {MAX_PAYLOAD_CHARS}: too close to "
                           "the server's truncation or the client's output cap; pick a shorter protein")
    return payload


# --------------------------------------------------------------------------
# Reference
# --------------------------------------------------------------------------

def build_reference(case: dict) -> dict:
    accession, threshold = case["accession"], case["am_threshold"]
    canonical_id = f"AF-{accession}-F1"

    models = api_json(f"prediction/{accession}")
    if not isinstance(models, list):
        raise RuntimeError(f"prediction/{accession} is not a list of models")
    lengths, sequences = {}, {}
    for model in models:
        entry_id, isoform = model.get("entryId"), model.get("uniprotAccession") or ""
        sequence = model.get("uniprotSequence") or ""
        if not isoform_of(isoform, accession) or entry_id != f"AF-{isoform}-F1":
            raise RuntimeError(f"model {entry_id!r} of {isoform!r} is not an isoform model of {accession}")
        if md5_of(sequence) != model.get("sequenceChecksum"):
            raise RuntimeError(f"{entry_id}: sequenceChecksum is not the md5 of its sequence")
        # A model carries four length-like fields; an agent may read any of them, so
        # they must agree, or "length" would have more than one answer.
        spans = {"sequence": len(model.get("sequence") or ""),
                 "uniprotStart..uniprotEnd": (model.get("uniprotEnd") or 0) - (model.get("uniprotStart") or 0) + 1,
                 "sequenceStart..sequenceEnd": (model.get("sequenceEnd") or 0) - (model.get("sequenceStart") or 0) + 1}
        if any(span != len(sequence) for span in spans.values()):
            raise RuntimeError(f"{entry_id}: length fields disagree with its {len(sequence)}-residue "
                               f"uniprotSequence: {spans}")
        lengths[entry_id], sequences[entry_id] = len(sequence), sequence
    if canonical_id not in lengths or len(lengths) < 2:
        raise RuntimeError(f"{accession}: models {sorted(lengths)} lack {canonical_id} or any isoform")
    if len(set(lengths.values())) != len(lengths):
        raise RuntimeError(f"{accession}: isoform models of equal length {lengths}; the ranking is not unique")
    ranking = rank_by_length(lengths)

    summary = api_json(f"uniprot/summary/{accession}.json")
    entry = summary.get("uniprot_entry") or {}
    canonical_length = entry.get("sequence_length")
    if canonical_length != lengths[canonical_id]:
        raise RuntimeError(f"summary sequence_length {canonical_length!r} != {lengths[canonical_id]} residues "
                           f"of {canonical_id}")
    if entry.get("uniprot_checksum") != md5_of(sequences[canonical_id]):
        raise RuntimeError(f"summary uniprot_checksum differs from the md5 of {canonical_id}'s sequence")

    annotations = api_json(f"annotations/{accession}.json?type=MUTAGEN")
    if annotations.get("id") != canonical_id or annotations.get("sequence") != sequences[canonical_id]:
        raise RuntimeError(f"annotations are for {annotations.get('id')!r}, not the canonical {canonical_id}")
    blocks = [b for b in annotations.get("annotation") or [] if b.get("type") == "MUTAGEN"]
    if len(blocks) != 1:
        raise RuntimeError(f"{len(blocks)} MUTAGEN blocks in the annotations of {accession}")
    scores = region_scores(blocks[0])
    if sorted(scores) != list(range(1, canonical_length + 1)):
        raise RuntimeError(f"{len(scores)} scored residues, expected one per residue 1..{canonical_length}")
    source = blocks[0].get("source_url") or ""
    means = am_means(http_get(source)) if source.endswith("-aa-substitutions.csv") else {}
    if set(means) != set(scores):
        raise RuntimeError(f"AlphaMissense CSV {source!r} covers {len(means)} positions, the annotation "
                           f"{len(scores)}")
    off = [p for p, (_, value) in scores.items() if round(means[p], 4) != value and abs(means[p] - value) > AM_TOL]
    if off:
        raise RuntimeError(f"annotation values differ from the CSV means at {off[:5]}")
    profile = profile_scores(scores, threshold)

    canonical = next(m for m in models if m.get("entryId") == canonical_id)
    return {
        "accession": accession,
        "protein": case["protein"],
        "entry_ids": sorted(lengths),
        "entry_lengths": lengths,
        "entry_ids_by_length_desc": ranking,
        "longest_entry_id": ranking[0],
        "canonical_entry_id": canonical_id,
        "canonical_length": canonical_length,
        "am_threshold": threshold,
        **profile,
        "cross_checks": {"checksums": "md5 of every model sequence == sequenceChecksum",
                         "summary_checksum": entry.get("uniprot_checksum"),
                         "alphamissense_csv": source, "csv_positions": len(means)},
        "reported_only": {"model_version": canonical.get("latestVersion"),
                          "model_created": canonical.get("modelCreatedDate"),
                          "summary_structures": len(summary.get("structures") or [])},
        "mcp_tools": ["alphafold_get_prediction", "alphafold_get_summary", "alphafold_get_annotations"],
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
    profile = {
        "accession": case["accession"],
        "protein": case["protein"],
        "entry_ids": reference["entry_ids"],
        "am_threshold": case["am_threshold"],
        "notes": {
            "entry_ids": "rank exactly these AlphaFold DB entries; ignore any other model of the accession",
            "length": "the number of residues in an entry's uniprotSequence; the listed lengths all differ",
            "am_threshold": "count residues whose score is strictly greater than this value",
        },
    }
    (data_dir / "profile.json").write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
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
