"""Instance generator for the arXiv search → PDF-snippets MCP E2E fake task.

A seeded RNG picks one curated case: a fielded arXiv query with a closed,
historical submission-date window (so the result set no longer changes), and
two terms to look up in the full text of one of the results. The agent has to
search, pick the paper with the most authors (ties: the earliest in the result
order), fetch snippets for the terms from that paper's PDF and report how many
snippets each term produced.

The reference is computed here, independently of the MCP server code:

- the search with a raw arXiv API request, built from the documented
  semantics of ``ArXiv_search_papers`` (query AND submittedDate range,
  sorted by submission date, ascending);
- the snippet counts by downloading the selected paper's current PDF and
  converting it with MarkItDown, the converter the server uses, pinned to the
  server's lockfile versions in task_meta.yaml. A term's count is its number
  of case-insensitive, non-overlapping occurrences, capped at
  ``max_snippets_per_term``. Counts rather than snippet text are scored
  because the converted text of two-column PDFs is garbled (merged words,
  ``(cid:NN)`` glyphs) and would be fragile to copy.

Needs network access to export.arxiv.org and arxiv.org. Generate with
``asibench generate --sandbox task``.

Framework call: python generate_gt.py --output-dir <dir> --params '<json>'
"""
from __future__ import annotations

import argparse
import json
import random
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

INPUT_SPEC = [
    {"name": "search.json", "description": "arXiv query, date window, sort order, selection rule and snippet terms"},
]
OUTPUT_SPEC = [
    {"name": "result.json", "description": "arxiv_ids, selected_id, snippet_counts"},
]
DEFAULT_PARAMS = {"seed": 0}

API_URL = "https://export.arxiv.org/api/query"
USER_AGENT = "asibench-mcp-e2e-generate/1"
ATOM = "{http://www.w3.org/2005/Atom}"
OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"
LIMIT = 10
WINDOW_CHARS = 100
MAX_SNIPPETS_PER_TERM = 10
SELECTION_RULE = ("the paper with the most authors; if several papers share the largest author count, "
                  "the one that comes first in the search result order")

# Curated 2026-10-02: every window returns 2–6 papers, and both terms occur 2–6
# times in the converted text of the selected paper (checked again at
# generation time; a broken case raises instead of producing a bad instance).
CASES = [
    {"topic": "graphene nanoribbons", "query": "ti:graphene",
     "date_from": "2011-03-01", "date_to": "2011-03-01", "terms": ["ballistic", "dielectric"]},
    {"topic": "exoplanets", "query": "ti:exoplanet",
     "date_from": "2012-06-01", "date_to": "2012-06-15", "terms": ["photometric", "spectroscopy"]},
    {"topic": "dark matter", "query": 'ti:"dark matter" AND cat:astro-ph.CO',
     "date_from": "2010-01-04", "date_to": "2010-01-06", "terms": ["spheroidal", "instrumental"]},
    {"topic": "reinforcement learning", "query": 'ti:"reinforcement learning"',
     "date_from": "2013-01-01", "date_to": "2013-02-28", "terms": ["diameter", "exploration"]},
    {"topic": "gravitational waves", "query": 'ti:"gravitational waves" AND cat:gr-qc',
     "date_from": "2011-03-01", "date_to": "2011-03-07", "terms": ["millisecond", "arecibo"]},
]


def build_case(seed: int) -> dict:
    case = dict(CASES[random.Random(seed).randrange(len(CASES))])
    return {**case, "terms": list(case["terms"]), "sort_by": "submittedDate", "sort_order": "ascending",
            "limit": LIMIT, "selection_rule": SELECTION_RULE,
            "window_chars": WINDOW_CHARS, "max_snippets_per_term": MAX_SNIPPETS_PER_TERM}


def search_query(case: dict) -> str:
    """The query the tool sends: the user query AND a submittedDate range (YYYYMMDDHHMMSS)."""
    start = case["date_from"].replace("-", "") + "000000"
    end = case["date_to"].replace("-", "") + "235959"
    return f"{case['query']} AND submittedDate:[{start} TO {end}]"


def normalize_id(text: str) -> str:
    """'1103.0291' from an abs/pdf URL or an ID with or without version / 'arXiv:' prefix."""
    text = re.sub(r"^https?://(export\.)?arxiv\.org/(abs|pdf)/", "", text.strip())
    text = re.sub(r"\.pdf$", "", text).removeprefix("arXiv:")
    return re.sub(r"v\d+$", "", text)


def parse_feed(xml_text: str) -> tuple[list[dict], int | None]:
    root = ET.fromstring(xml_text)
    total = root.findtext(f"{OPENSEARCH}totalResults")
    papers = []
    for entry in root.findall(f"{ATOM}entry"):
        url = entry.findtext(f"{ATOM}id", "")
        papers.append({"id": normalize_id(url), "version": re.search(r"(v\d+)?$", url).group(1) or "",
                       "title": " ".join(entry.findtext(f"{ATOM}title", "").split()),
                       "num_authors": len(entry.findall(f"{ATOM}author")),
                       "published": entry.findtext(f"{ATOM}published", "")})
    return papers, int(total) if total and total.isdigit() else None


def http_get(url: str, attempts: int = 5) -> bytes:
    error = None
    for attempt in range(attempts):
        time.sleep(3 if attempt == 0 else 5 * attempt)  # arXiv asks for >= 3 s between requests
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=90) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:  # incl. transient 406/429/503
            error = exc
    raise RuntimeError(f"GET {url} failed after {attempts} attempts: {error}")


def reference_search(case: dict) -> list[dict]:
    params = urllib.parse.urlencode({"search_query": search_query(case), "start": 0, "max_results": case["limit"],
                                     "sortBy": case["sort_by"], "sortOrder": case["sort_order"]})
    for _ in range(3):  # the live API occasionally returns a short page
        papers, total = parse_feed(http_get(f"{API_URL}?{params}").decode("utf-8"))
        if papers and total == len(papers):
            return papers
    raise RuntimeError(f"unstable or empty search for {search_query(case)!r}: {len(papers)} of {total}")


def select_paper(papers: list[dict]) -> dict:
    most = max(p["num_authors"] for p in papers)
    return next(p for p in papers if p["num_authors"] == most)


def pdf_text(arxiv_id: str) -> str:
    """Current PDF version converted with MarkItDown, as ArXiv_get_pdf_snippets does."""
    from markitdown import MarkItDown

    data = http_get(f"https://arxiv.org/pdf/{arxiv_id}")
    if not data.startswith(b"%PDF"):
        raise RuntimeError(f"{arxiv_id}: download is not a PDF")
    with tempfile.NamedTemporaryFile(suffix=".pdf") as handle:
        handle.write(data)
        handle.flush()
        return MarkItDown().convert(handle.name).text_content


def snippet_counts(text: str, terms: list[str], cap: int) -> dict[str, int]:
    low = text.lower()
    return {term: min(len(re.findall(re.escape(term.lower()), low)), cap) for term in terms}


def render_prompts(task_dir: Path, output_dir: Path, case: dict) -> None:
    for level in ("b1", "b2", "b3", "b4"):
        text = (task_dir / f"prompt_{level}.md").read_text(encoding="utf-8")
        text = text.replace("{{topic}}", case["topic"])
        (output_dir / f"prompt_{level}.md").write_text(text, encoding="utf-8")


def generate(output_dir: Path, params: dict) -> dict:
    import markitdown
    import pdfminer
    import pdfplumber

    p = {**DEFAULT_PARAMS, **params}
    t0 = time.time()
    output_dir = Path(output_dir)
    data_dir, ref_dir = output_dir / "data", output_dir / "reference"
    data_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)

    case = build_case(int(p["seed"]))
    papers = reference_search(case)
    selected = select_paper(papers)
    counts = snippet_counts(pdf_text(selected["id"]), case["terms"], case["max_snippets_per_term"])
    bad = {t: n for t, n in counts.items() if not 1 <= n < case["max_snippets_per_term"]}
    if bad:
        raise RuntimeError(f"curated case {case['topic']!r} broke: term counts {bad} in {selected['id']}")

    (data_dir / "search.json").write_text(json.dumps(case, indent=2) + "\n", encoding="utf-8")
    reference = {
        **case,
        "search_query": search_query(case),
        "arxiv_ids": [paper["id"] for paper in papers],
        "papers": papers,
        "selected_id": selected["id"],
        "selected_version": selected["version"],
        "selected_num_authors": selected["num_authors"],
        "snippet_counts": counts,
        "markitdown_version": markitdown.__version__,
        "pdfminer_version": pdfminer.__version__,
        "pdfplumber_version": pdfplumber.__version__,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mcp_tools": ["ArXiv_search_papers", "ArXiv_get_pdf_snippets"],
    }
    (ref_dir / "reference.json").write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")

    render_prompts(Path(__file__).resolve().parent, output_dir, case)
    meta = {
        "params_used": p,
        "input_files": [s["name"] for s in INPUT_SPEC],
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
