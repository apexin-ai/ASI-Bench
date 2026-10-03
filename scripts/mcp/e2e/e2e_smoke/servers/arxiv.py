"""Direct (agent-free) E2E smoke test for the pinned ToolUniverse arXiv MCP tools.

Run with the server's own virtualenv so that MarkItDown (the PDF converter the
server uses) is importable for the independent snippet reference::

    ~/mcp/tooluniverse/.venv/bin/python scripts/mcp/e2e/smoke.py arxiv --config ~/mcp/arxiv.mcp.json

Needs network access to export.arxiv.org and arxiv.org (the server and the
references both query the live public arXiv API; no key is needed). Queries
are restricted to closed historical date windows so that the result sets are
stable, and every tool result is compared with a reference fetched here,
outside the server process, at the same time.

L1: real tools/call of both tools; search results are compared field by
field with a raw arXiv API query built here; PDF snippets are compared with
snippets cut from the same PDF version downloaded and converted here (the
shared L0/L1 checks, including the cache-off launch env from the manifest, are
in ``e2e_smoke/runner.py``).

Upstream defects that do not make a correctly used tool wrong (in-band errors,
misleading flags, unparenthesised boolean queries combined with a date range,
old-style IDs, workspace-dependent tool filtering) are WARN; wrong results
for correct inputs are FAIL.
"""
from __future__ import annotations

import argparse
import json
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter

from ..client import MCPError, text_of
from ..runner import Caller, Report, Session, Smoke, check_rejected

SEARCH = "ArXiv_search_papers"
SNIPPETS = "ArXiv_get_pdf_snippets"
API_URL = "https://export.arxiv.org/api/query"
USER_AGENT = "asibench-mcp-e2e-smoke/1 (reference client)"
API_GAP_S = 3.0            # arXiv asks for >= 3 s between API requests
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"
OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"

DAY = "2011-03-01"
MONTH = ("2011-03-01", "2011-03-31")
# (label, tool arguments, reference search_query) — reference queries are
# written out by hand from the tool's documented semantics.
SEARCH_CASES = (
    ("fielded+date, submittedDate asc",
     {"query": "ti:graphene", "limit": 10, "date_from": DAY, "date_to": DAY,
      "sort_by": "submittedDate", "sort_order": "ascending"},
     "ti:graphene AND submittedDate:[20110301000000 TO 20110301235959]"),
    ("keywords are ANDed",
     {"query": "graphene nanoribbons mobility", "limit": 10, "date_from": MONTH[0], "date_to": MONTH[1],
      "sort_by": "submittedDate", "sort_order": "ascending"},
     "all:graphene AND all:nanoribbons AND all:mobility AND submittedDate:[20110301000000 TO 20110331235959]"),
    ("category, limit, submittedDate desc",
     {"query": "cat:hep-th", "limit": 4, "date_from": DAY, "date_to": DAY,
      "sort_by": "submittedDate", "sort_order": "descending"},
     "cat:hep-th AND submittedDate:[20110301000000 TO 20110301235959]"),
    ("parenthesised OR + date",
     {"query": "(ti:graphene OR ti:phonon)", "limit": 10, "date_from": DAY, "date_to": DAY,
      "sort_by": "submittedDate", "sort_order": "ascending"},
     "(ti:graphene OR ti:phonon) AND submittedDate:[20110301000000 TO 20110301235959]"),
    ("multi-word author is quoted",
     {"query": "au:Stefano Baroni", "limit": 5, "sort_by": "submittedDate", "sort_order": "ascending"},
     'au:"Stefano Baroni"'),
)
UNPARENTHESISED_OR = {"query": "ti:graphene OR ti:phonon", "limit": 10, "date_from": DAY, "date_to": DAY,
                      "sort_by": "submittedDate", "sort_order": "ascending"}
SNIPPET_TERMS = ["plateau", "LANDAUER"]            # case-insensitive match
SNIPPET_WINDOW, SNIPPET_MAX = 80, 2
CAP_ARGS = {"terms": ["graphene", "conductance", "ribbon"], "window_chars": 400,
            "max_snippets_per_term": 10, "max_total_chars": 1000}
OLD_STYLE_ID = ("solv-int/9901001", "Camassa")     # archive name contains a "v"
MISSING_ID = "9999.99999"


# --------------------------------------------------------------------------
# Pure helpers (stdlib only; unit-tested offline)
# --------------------------------------------------------------------------

def norm_ws(text: str) -> str:
    return " ".join((text or "").split())


def split_arxiv_id(url_or_id: str) -> tuple[str, str]:
    """('1103.0212', 'v1') from an abs/pdf URL or an ID; version may be ''."""
    text = url_or_id.strip()
    text = re.sub(r"^https?://(export\.)?arxiv\.org/(abs|pdf)/", "", text)
    text = re.sub(r"\.pdf$", "", text).removeprefix("arXiv:")
    match = re.fullmatch(r"(.+?)(v\d+)?", text)
    return match.group(1), match.group(2) or ""


def parse_atom(xml_text: str) -> tuple[list[dict], int | None]:
    """Entries of an arXiv Atom feed as comparable records, plus opensearch totalResults."""
    root = ET.fromstring(xml_text)
    total = root.findtext(f"{OPENSEARCH}totalResults")
    records = []
    for entry in root.findall(f"{ATOM}entry"):
        ident, version = split_arxiv_id(entry.findtext(f"{ATOM}id", ""))
        category = entry.find(f"{ARXIV_NS}primary_category")
        records.append({
            "id": ident, "version": version,
            "title": norm_ws(entry.findtext(f"{ATOM}title", "")),
            "abstract": norm_ws(entry.findtext(f"{ATOM}summary", "")),
            "authors": [norm_ws(a.findtext(f"{ATOM}name", "")) for a in entry.findall(f"{ATOM}author")],
            "published": entry.findtext(f"{ATOM}published", ""),
            "updated": entry.findtext(f"{ATOM}updated", ""),
            "category": category.get("term", "") if category is not None else "",
        })
    return records, int(total) if total and total.isdigit() else None


def tool_records(papers: list[dict]) -> list[dict]:
    """ArXiv_search_papers items in the same comparable form as parse_atom()."""
    out = []
    for paper in papers:
        ident, version = split_arxiv_id(paper.get("url", ""))
        out.append({
            "id": ident, "version": version,
            "title": norm_ws(paper.get("title", "")),
            "abstract": norm_ws(paper.get("abstract", "")),
            "authors": [norm_ws(a) for a in paper.get("authors", [])],
            "published": paper.get("published", ""),
            "updated": paper.get("updated", ""),
            "category": paper.get("category", ""),
        })
    return out


def record_diffs(got: list[dict], ref: list[dict]) -> list[str]:
    """Human-readable differences; empty means identical (order included)."""
    got_ids = [f"{r['id']}{r['version']}" for r in got]
    ref_ids = [f"{r['id']}{r['version']}" for r in ref]
    if got_ids != ref_ids:
        return [f"IDs/order differ: server {got_ids} vs reference {ref_ids}"]
    diffs = []
    for g, r in zip(got, ref):
        for key in ("title", "abstract", "authors", "published", "updated", "category"):
            if g[key] != r[key]:
                diffs.append(f"{r['id']}.{key}: server {str(g[key])[:80]!r} vs reference {str(r[key])[:80]!r}")
    return diffs


def in_window(published: str, date_from: str, date_to: str) -> bool:
    day = published[:10]
    return date_from <= day <= date_to


def expected_snippets(text: str, terms: list[str], window: int, max_per_term: int,
                      max_total: int) -> tuple[list[dict], int]:
    """Snippets as documented by the tool, plus how many matches the size cap dropped.

    Per term (case-insensitive substring), up to ``max_per_term`` matches, each
    ``text[start-window : end+window].strip()``; a snippet that would push the
    running total over ``max_total`` ends that term.
    """
    snippets, total, dropped = [], 0, 0
    low = text.lower()
    for raw in terms:
        term = raw.strip()
        if not term:
            continue
        matches = list(re.finditer(re.escape(term.lower()), low))[:max_per_term]
        for k, m in enumerate(matches):
            snippet = text[max(0, m.start() - window):min(len(text), m.end() + window)].strip()
            if total + len(snippet) > max_total:
                dropped += len(matches) - k
                break
            snippets.append({"term": term, "snippet": snippet})
            total += len(snippet)
    return snippets, dropped


def in_band_error(payload) -> str | None:
    """The tool's error message if the JSON payload is an in-band error object."""
    if isinstance(payload, dict) and (payload.get("status") == "error" or "error" in payload):
        return str(payload.get("error") or payload)
    return None


# --------------------------------------------------------------------------
# Independent references (raw arXiv API / PDF, not the server code)
# --------------------------------------------------------------------------

class Throttle:
    """Keeps >= API_GAP_S between arXiv requests made by the server *or* by this script."""

    def __init__(self, gap: float = API_GAP_S) -> None:
        self.gap = gap
        self.last = 0.0

    def wait(self) -> None:
        delay = self.last + self.gap - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self.last = time.monotonic()


THROTTLE = Throttle()


def http_get(url: str, timeout: float = 60.0, attempts: int = 3) -> bytes:
    error = None
    for attempt in range(attempts):
        THROTTLE.wait()
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            error = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed: {error}")


def reference_search(search_query: str, limit: int, sort_by: str, sort_order: str) -> list[dict]:
    params = urllib.parse.urlencode({"search_query": search_query, "start": 0, "max_results": limit,
                                     "sortBy": sort_by, "sortOrder": sort_order})
    records, _ = parse_atom(http_get(f"{API_URL}?{params}").decode("utf-8"))
    return records


def reference_by_id(arxiv_id: str) -> dict | None:
    records, _ = parse_atom(http_get(f"{API_URL}?id_list={urllib.parse.quote(arxiv_id)}").decode("utf-8"))
    return records[0] if records else None


def reference_pdf_text(arxiv_id: str, version: str) -> str:
    """Download the given PDF version and convert it with MarkItDown, like the server does."""
    from markitdown import MarkItDown
    data = http_get(f"https://arxiv.org/pdf/{arxiv_id}{version}")
    if not data.startswith(b"%PDF"):
        raise RuntimeError(f"{arxiv_id}{version}: download is not a PDF")
    with tempfile.NamedTemporaryFile(suffix=".pdf") as handle:
        handle.write(data)
        handle.flush()
        return MarkItDown().convert(handle.name).text_content


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def call_json(call: Caller, report: Report, check: str, tool: str, arguments: dict):
    """Throttled tools/call + JSON decode; None (with a FAIL) on transport/tool/JSON errors."""
    THROTTLE.wait()
    return call.json(check, tool, arguments)


def check_search(call: Caller, report: Report, label: str, arguments: dict, search_query: str) -> list[dict] | None:
    name = f"{SEARCH}[{label}]"
    payload = call_json(call, report, name, SEARCH, arguments)
    if payload is None:
        return None
    if isinstance(payload, dict) and "data" in payload and payload.get("_truncated"):
        report.add("L1", name, "FAIL", "server truncated the response (>100k chars)")
        return None
    if not isinstance(payload, list):
        report.add("L1", name, "FAIL", f"expected a list of papers, got {str(payload)[:200]}")
        return None
    got = tool_records(payload)
    sort_by, sort_order = arguments.get("sort_by", "relevance"), arguments.get("sort_order", "descending")
    ref = reference_search(search_query, arguments["limit"], sort_by, sort_order)
    diffs = record_diffs(got, ref)
    if diffs:  # one retry: the live API occasionally returns a short page
        ref = reference_search(search_query, arguments["limit"], sort_by, sort_order)
        diffs = record_diffs(got, ref)
    problems = list(diffs)
    if not ref:
        problems.append("reference returned no papers; the case would prove nothing")
    if "date_from" in arguments:
        outside = [r["id"] for r in got if not in_window(r["published"], arguments["date_from"], arguments["date_to"])]
        if outside:
            problems.append(f"published outside the date range: {outside}")
    if sort_by == "submittedDate":
        dates = [r["published"] for r in got]
        if dates != sorted(dates, reverse=(sort_order == "descending")):
            problems.append("results are not in the requested submittedDate order")
    report.add("L1", name, "FAIL" if problems else "PASS",
               "; ".join(problems[:4]) if problems else f"{len(got)} papers identical to the raw API ({search_query})",
               server_ids=[r["id"] + r["version"] for r in got], reference_ids=[r["id"] + r["version"] for r in ref])
    return got


def check_unparenthesised_or(call: Caller, report: Report) -> None:
    name = f"{SEARCH}[OR + date range]"
    payload = call_json(call, report, name, SEARCH, UNPARENTHESISED_OR)
    if payload is None:
        return
    if not isinstance(payload, list):
        report.add("L1", name, "FAIL", f"expected a list of papers, got {str(payload)[:200]}")
        return
    got = tool_records(payload)
    outside = [f"{r['id']}({r['published'][:10]})" for r in got if not in_window(r["published"], DAY, DAY)]
    if outside:
        report.add("L1", name, "WARN",
                   "the date clause is appended without parentheses, so it binds only to the last OR term; "
                   f"{len(outside)}/{len(got)} results lie outside {DAY}: {outside[:4]}. "
                   "Workaround: parenthesise the OR group (checked above).")
    else:
        report.add("L1", name, "PASS", f"all {len(got)} results inside {DAY}")


def check_in_band_error(call: Caller, report: Report, tool: str, label: str, arguments: dict) -> None:
    """Invalid input should give isError=true; an error object in a normal result is a WARN."""
    THROTTLE.wait()
    check_rejected(call, f"{tool}[{label}]", tool, arguments, in_band=_in_band)


def _in_band(result: dict) -> str | None:
    try:
        return in_band_error(json.loads(text_of(result)))
    except json.JSONDecodeError:
        return None


def compare_snippets(report: Report, name: str, payload, expected: list[dict], pdf_url: str) -> bool:
    if not isinstance(payload, dict) or payload.get("status") != "success":
        report.add("L1", name, "FAIL", f"no snippets: {str(payload)[:300]}")
        return False
    problems = []
    if payload.get("pdf_url") != pdf_url:
        problems.append(f"pdf_url {payload.get('pdf_url')!r} != {pdf_url!r}")
    got = [{"term": s.get("term"), "snippet": s.get("snippet")} for s in payload.get("snippets", [])]
    if got != expected:
        problems.append(f"snippets differ from the reference: server {len(got)} vs reference {len(expected)}"
                        + (f"; first server snippet {got[0]['snippet'][:80]!r}" if got else ""))
    if payload.get("snippets_count") != len(payload.get("snippets", [])):
        problems.append("snippets_count does not match the snippet list")
    report.add("L1", name, "FAIL" if problems else "PASS",
               "; ".join(problems) if problems else
               f"{len(got)} snippet(s) identical to the reference ({dict(Counter(s['term'] for s in got))})")
    return not problems


def check_snippets(call: Caller, report: Report, papers: list[dict] | None) -> None:
    if not papers:
        report.add("L1", f"{SNIPPETS}[chained from search]", "FAIL", "no search result to take the paper from")
        return
    paper = papers[0]
    ident, version = paper["id"], paper["version"]
    meta = reference_by_id(ident)
    if meta is None or meta["version"] != version:
        report.add("L1", f"{SNIPPETS}[pinned version]", "WARN",
                   f"{ident}: search gave {version}, latest is {meta and meta['version']}; "
                   "the tool always downloads the latest version")
    text = reference_pdf_text(ident, version)
    for term in SNIPPET_TERMS:
        if term.lower() not in text.lower():
            report.add("L1", f"{SNIPPETS}[reference]", "FAIL", f"term {term!r} not in {ident}; fix the smoke inputs")
            return
    pdf_url = f"https://arxiv.org/pdf/{ident}.pdf"
    expected, _ = expected_snippets(text, SNIPPET_TERMS, SNIPPET_WINDOW, SNIPPET_MAX, 8000)

    name = f"{SNIPPETS}[chained from search: {ident}{version}]"
    payload = call_json(call, report, name, SNIPPETS, {
        "arxiv_id": f"{ident}{version}", "terms": SNIPPET_TERMS,
        "window_chars": SNIPPET_WINDOW, "max_snippets_per_term": SNIPPET_MAX})
    if payload is not None:
        compare_snippets(report, name, payload, expected, pdf_url)

    name = f"{SNIPPETS}[pdf_url]"
    url = f"https://arxiv.org/pdf/{ident}{version}"
    payload = call_json(call, report, name, SNIPPETS, {
        "pdf_url": url, "terms": SNIPPET_TERMS, "window_chars": SNIPPET_WINDOW, "max_snippets_per_term": SNIPPET_MAX})
    if payload is not None:
        compare_snippets(report, name, payload, expected, url)

    name = f"{SNIPPETS}[max_total_chars]"
    payload = call_json(call, report, name, SNIPPETS, {"arxiv_id": ident, **CAP_ARGS})
    if payload is not None:
        expected_cap, dropped = expected_snippets(text, CAP_ARGS["terms"], CAP_ARGS["window_chars"],
                                                  CAP_ARGS["max_snippets_per_term"], CAP_ARGS["max_total_chars"])
        if compare_snippets(report, name, payload, expected_cap, pdf_url):
            total = sum(len(s["snippet"]) for s in payload["snippets"])
            flag = payload.get("truncated")
            if total > CAP_ARGS["max_total_chars"]:
                report.add("L1", f"{name}[cap]", "FAIL", f"{total} chars > cap {CAP_ARGS['max_total_chars']}")
            elif dropped and not flag:
                report.add("L1", f"{name}[truncated flag]", "WARN",
                           f"{dropped} match(es) dropped by the cap but truncated={flag} "
                           "(flag is only set when the total reaches the cap exactly)")
            else:
                report.add("L1", f"{name}[truncated flag]", "PASS", f"truncated={flag}, dropped={dropped}")


def check_old_style_id(call: Caller, report: Report) -> None:
    ident, term = OLD_STYLE_ID
    name = f"{SNIPPETS}[old-style ID {ident}]"
    payload = call_json(call, report, name, SNIPPETS, {"arxiv_id": ident, "terms": [term], "max_snippets_per_term": 1})
    if payload is None:
        return
    if isinstance(payload, dict) and payload.get("status") == "success" and payload.get("snippets"):
        report.add("L1", name, "PASS", f"pdf_url={payload.get('pdf_url')}")
    else:
        report.add("L1", name, "WARN",
                   "version stripping splits the ID at the first 'v', so old-style IDs whose archive "
                   f"contains a 'v' break: {str(payload)[:200]}")


def check_workspace_filter(session: Session) -> None:
    """Start a second server whose cwd holds a ToolUniverse workspace (./.tooluniverse)."""
    if session.args.skip_workspace_probe:
        return
    name, expected = "tool filter with a workspace in cwd", session.entry["expected_tools"]
    try:
        with session.spawn("ws-cwd", prepare_cwd=lambda cwd: (cwd / ".tooluniverse").mkdir()) as client:
            client.initialize()
            count = len(client.list_tools())
    except MCPError as exc:
        session.report.add("L0", name, "FAIL", f"second server failed: {exc}")
        return
    if count == len(expected):
        session.report.add("L0", name, "PASS", f"{count} tools")
    else:
        session.report.add("L0", name, "WARN",
                           f"with ./.tooluniverse in the server cwd the default profile.yaml is seeded and loaded, "
                           f"--include-tools is ignored and {count} tools are exposed; never create a ToolUniverse "
                           "workspace in the MCP cwd (the checkout) and keep HOME != cwd")


def run_l1(session: Session) -> None:
    call, report = session.call, session.report
    try:
        first = None
        for label, arguments, search_query in SEARCH_CASES:
            got = check_search(call, report, label, arguments, search_query)
            if first is None:
                first = got
        check_unparenthesised_or(call, report)
        check_in_band_error(call, report, SEARCH, "empty query", {"query": ""})
        check_in_band_error(call, report, SEARCH, "invalid sort_by", {"query": "ti:graphene", "sort_by": "citations"})
        check_snippets(call, report, first)
        check_old_style_id(call, report)
        check_in_band_error(call, report, SNIPPETS, "empty terms", {"arxiv_id": "1103.0212", "terms": []})
        check_in_band_error(call, report, SNIPPETS, "missing paper", {"arxiv_id": MISSING_ID, "terms": ["x"]})
    except RuntimeError as exc:  # reference download failed
        report.add("L1", "reference", "FAIL", str(exc))


def _arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--skip-workspace-probe", action="store_true",
                        help="do not start the second server that probes workspace-dependent tool filtering")


SMOKE = Smoke(
    server="arxiv",
    run_l1=run_l1,
    packages=("tooluniverse", "mcp", "fastmcp", "markitdown", "pdfminer.six", "pdfplumber", "requests"),
    pass_proxies=True,
    add_arguments=_arguments,
    after=check_workspace_filter,
)
