"""arxiv_search_snippets (search -> snippets, named extractors, member chaining, merged answers,
web-tool bypass, exact server tool set): generator, scorers, verifier scenarios. No network."""
import json

import pytest

from . import support
from .support import claude, codex, jsonl

TASK = support.Task("mcp_e2e.arxiv_search_snippets")
TASK_DIR, TASK_ID, INSTANCE_ID = TASK.dir, TASK.task_id, TASK.instance_id

# seed31415 case (gravitational waves), as generated 2026-10-02
IDS = ["1103.0115", "1103.0346", "1103.0373", "1103.0576", "1103.1301"]
AUTHORS = [8, 10, 2, 19, 2]
VERSIONS = ["v1", "v1", "v1", "v1", "v2"]
COUNTS = {"millisecond": 5, "arecibo": 3}
CASE = {"topic": "gravitational waves", "query": 'ti:"gravitational waves" AND cat:gr-qc',
        "date_from": "2011-03-01", "date_to": "2011-03-07", "terms": ["millisecond", "arecibo"],
        "sort_by": "submittedDate", "sort_order": "ascending", "limit": 10,
        "window_chars": 100, "max_snippets_per_term": 10}
REFERENCE = {**CASE, "arxiv_ids": IDS, "selected_id": "1103.0576", "selected_version": "v1",
             "snippet_counts": COUNTS}
SEARCH_ARGS = {k: CASE[k] for k in ("query", "date_from", "date_to", "sort_by", "sort_order", "limit")}
SNIPPET_ARGS = {"arxiv_id": "1103.0576v1", "terms": ["millisecond", "arecibo"], "window_chars": 100,
                "max_snippets_per_term": 10}
SEARCH = "mcp__arxiv__ArXiv_search_papers"
SNIPPETS = "mcp__arxiv__ArXiv_get_pdf_snippets"
ALL = ("arxiv_e2e_schema", "arxiv_e2e_ids", "arxiv_e2e_selected", "arxiv_e2e_snippet_counts")

generate_gt = TASK.module("generate_gt")
scorer = TASK.module("custom_scorer")
verify = support.verify
_status = support.statuses


def _search_result(ids=IDS):
    papers = [{"title": f"Paper {i}", "abstract": "...", "authors": [f"A{k}" for k in range(n)],
               "published": "2011-03-02T00:00:00Z", "updated": "2011-03-02T00:00:00Z", "category": "gr-qc",
               "url": f"https://arxiv.org/abs/{i}{v}"} for i, n, v in zip(ids, AUTHORS, VERSIONS)]
    return json.dumps(papers, ensure_ascii=False)


def _snippet_result(counts=COUNTS, arxiv_id="1103.0576"):
    snippets = [{"term": t, "snippet": f"... {t} ..."} for t, n in counts.items() for _ in range(n)]
    payload = {"pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf", "snippets": snippets,
               "snippets_count": len(snippets), "truncated": False}
    return json.dumps({"status": "success", **payload, "data": payload})


def _answer(**overrides):
    return {"arxiv_ids": IDS, "selected_id": "1103.0576", "snippet_counts": dict(COUNTS), **overrides}



def _dirs(tmp_path, prediction, reference=REFERENCE):
    return support.score_dirs(tmp_path, prediction, reference)


def _score(name, pred, ref, weight=1.0):
    from ai4sci_bench.core.scorer import get_scorer
    return get_scorer(name).score(pred, ref, {"weight": weight})



def _stream(search_args=SEARCH_ARGS, snippet_calls=None, search_result=None, tools=None, extra=()):
    """Claude stream-json; snippet_calls = [(input, result_text)], extra = [(tool name, input)]."""
    if snippet_calls is None:
        snippet_calls = [(SNIPPET_ARGS, _snippet_result())]
    events = [claude.init("arxiv", tools or ["Bash", "Read", "Write", "WebFetch", "WebSearch", SEARCH, SNIPPETS])]
    for k, (name, args) in enumerate(extra):
        events += claude.call(f"x{k}", name, args, "ok")
    events += claude.call("s1", SEARCH, search_args, search_result or _search_result())
    for k, (args, result) in enumerate(snippet_calls):
        events += claude.call(f"p{k}", SNIPPETS, args, result)
    events.append(claude.result(4))
    return jsonl(events)


def _codex(extra_items=()):
    events = [*codex.START, *(codex.item(item) for item in extra_items)]
    events += codex.mcp("m1", "arxiv", "ArXiv_search_papers", SEARCH_ARGS, _search_result())
    events += codex.mcp("m2", "arxiv", "ArXiv_get_pdf_snippets", SNIPPET_ARGS, _snippet_result())
    events.append(codex.done())
    return jsonl(events)


def _run(tmp_path, stream, answer=None, codex=False, files=None):
    return TASK.verify(tmp_path, stream, reference=REFERENCE, answer=answer or _answer(),
                       harness="codex" if codex else "claude", files=files)


def test_cases_are_deterministic_and_cover_all_curated_cases():
    assert generate_gt.build_case(31415) == generate_gt.build_case(31415)
    assert generate_gt.build_case(31415)["topic"] == "gravitational waves"
    assert {generate_gt.build_case(s)["topic"] for s in range(100)} == {c["topic"] for c in generate_gt.CASES}


def test_curated_cases_are_safe_for_the_tool():
    for case in generate_gt.CASES:
        # the tool appends the date clause without parentheses: no OR allowed
        assert " OR " not in case["query"] and case["date_from"] <= case["date_to"] < "2020"
        assert len(case["terms"]) == 2 and all(t == t.strip() and t for t in case["terms"])
    case = generate_gt.build_case(2)
    assert generate_gt.search_query(case) == "ti:graphene AND submittedDate:[20110301000000 TO 20110301235959]"
    assert case["max_snippets_per_term"] == 10 and case["sort_order"] == "ascending"


def test_selection_rule_takes_first_of_ties():
    papers = [{"id": "a", "num_authors": 2}, {"id": "b", "num_authors": 3}, {"id": "c", "num_authors": 3}]
    assert generate_gt.select_paper(papers)["id"] == "b"


def test_snippet_counts_follow_tool_semantics():
    text = "Millisecond pulsars ... MILLISECOND ... millisecondmillisecond Arecibo"
    assert generate_gt.snippet_counts(text, ["millisecond", "arecibo", "absent"], 3) == \
        {"millisecond": 3, "arecibo": 1, "absent": 0}


def test_parse_feed_and_ids():
    feed = """<feed xmlns="http://www.w3.org/2005/Atom" xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
      <opensearch:totalResults>1</opensearch:totalResults>
      <entry><id>http://arxiv.org/abs/1103.1301v2</id><title>An all-sky
        search</title><published>2011-03-07T00:00:00Z</published>
        <author><name>A</name></author><author><name>B</name></author></entry></feed>"""
    papers, total = generate_gt.parse_feed(feed)
    assert total == 1 and papers == [{"id": "1103.1301", "version": "v2", "title": "An all-sky search",
                                      "num_authors": 2, "published": "2011-03-07T00:00:00Z"}]
    assert generate_gt.normalize_id("solv-int/9901001v2") == "solv-int/9901001"


def test_prompts_name_tools_only_at_b1_and_forbid_other_network_access():
    for level in ("b1", "b2", "b3", "b4"):
        text = (TASK_DIR / f"prompt_{level}.md").read_text()
        assert "result.json" in text and "Do not use web search or web fetch tools" in text
        assert "`curl`/`wget`" in text and "without version suffix" in text
        assert ("{{topic}}" in text) == (level in ("b1", "b2"))
        named = ("`ArXiv_search_papers`" in text, "`ArXiv_get_pdf_snippets`" in text)
        assert named == ((True, True) if level == "b1" else (False, False))
        assert ("`arxiv` MCP server" in text) == (level in ("b1", "b2"))
        assert "mcp__" not in text and "Claude" not in text and "Codex" not in text


def test_task_is_test_status_and_needs_pinned_converter():
    import yaml
    meta = yaml.safe_load((TASK_DIR / "task_meta.yaml").read_text())
    assert meta["status"] == "test" and meta["difficulty"]["requires_network"] is True
    assert "markitdown[pdf]==0.1.7" in meta["runtime"]["packages"]


def test_reference_answer_scores_full_and_ids_are_normalised(tmp_path):
    answer = _answer(arxiv_ids=[f"https://arxiv.org/abs/{i}v1" for i in IDS], selected_id="arXiv:1103.0576v1",
                     snippet_counts={"Millisecond": "5", "arecibo": 3.0})
    pred, ref = _dirs(tmp_path, answer)
    assert [_score(n, pred, ref).score for n in ALL] == [1.0] * 4


def test_wrong_order_selection_and_counts(tmp_path):
    pred, ref = _dirs(tmp_path, _answer(arxiv_ids=IDS[::-1], selected_id="1103.0346",
                                        snippet_counts={"millisecond": 3, "arecibo": 3}))
    assert _score("arxiv_e2e_ids", pred, ref).score == 0.0
    assert _score("arxiv_e2e_selected", pred, ref).score == 0.0
    assert _score("arxiv_e2e_snippet_counts", pred, ref, weight=40).score == 20.0


@pytest.mark.parametrize("prediction", [None, "not json", [], _answer(arxiv_ids=[]), _answer(selected_id=""),
                                        _answer(snippet_counts={"millisecond": -1}),
                                        _answer(snippet_counts={"millisecond": 2.5}),
                                        _answer(snippet_counts={"millisecond": True})])
def test_submission_failures_are_valid_zero_scores(tmp_path, prediction):
    pred, ref = _dirs(tmp_path, prediction)
    for name in ALL:
        detail = _score(name, pred, ref)
        assert detail.score == 0.0 and not detail.details.get("scorer_internal_error")


def test_missing_reference_is_an_evaluator_failure(tmp_path):
    pred, ref = _dirs(tmp_path, _answer(), reference=None)
    for name in ALL[1:]:
        assert _score(name, pred, ref).details["scorer_internal_error"] is True


def test_genuine_search_and_snippets_run_passes(tmp_path):
    row = _run(tmp_path, _stream(extra=[("Bash", {"command": "cat data/search.json"})]))
    assert row["verdict"] == "PASS", row["checks"]
    assert set(_status(row).values()) == {"PASS"}
    assert list(_status(row)) == list(verify.CHECK_ORDER)


def test_snippets_by_pdf_url_and_split_per_term_calls_pass(tmp_path):
    calls = [({"pdf_url": "https://arxiv.org/pdf/1103.0576v1", "terms": ["millisecond"],
               "max_snippets_per_term": 10}, _snippet_result({"millisecond": 5})),
             ({"arxiv_id": "1103.0576", "terms": ["arecibo"], "max_snippets_per_term": 10},
              _snippet_result({"arecibo": 3}))]
    row = _run(tmp_path, _stream(snippet_calls=calls))
    assert row["verdict"] == "PASS", row["checks"]
    # inputs differ from the single reference call (one term each) -> WARN, not FAIL
    assert row["checks"]["tool_correct"]["per_call"]["snippets"]["status"] == "WARN"
    assert _status(row)["tool_chain"] == "PASS" and _status(row)["answer_from_tool"] == "PASS"


def test_snippets_for_a_paper_not_in_the_search_results_break_the_chain(tmp_path):
    row = _run(tmp_path, _stream(snippet_calls=[({**SNIPPET_ARGS, "arxiv_id": "1706.03762"}, _snippet_result())]))
    assert row["failure"] == "tool_chain"


def test_default_cap_gives_wrong_counts(tmp_path):
    capped = {"millisecond": 3, "arecibo": 3}
    row = _run(tmp_path, _stream(snippet_calls=[({**SNIPPET_ARGS, "max_snippets_per_term": 3},
                                                 _snippet_result(capped))]),
               answer=_answer(snippet_counts=capped))
    assert row["failure"] == "tool_correct"
    assert "matches reference: False" in row["checks"]["answer_from_tool"]["detail"]


def test_answer_values_must_come_from_the_tools(tmp_path):
    # correct values, but the search returned something else (e.g. answered from a web search)
    row = _run(tmp_path, _stream(search_result=_search_result(IDS[:4])))
    assert row["failure"] == "tool_correct"
    assert "equals a tool-returned value: False" in row["checks"]["answer_from_tool"]["detail"]
    row = _run(tmp_path / "2", _stream(), answer=_answer(snippet_counts={"millisecond": 5, "arecibo": 2}))
    assert row["failure"] == "answer_from_tool"


def test_web_fetch_of_arxiv_is_a_bypass_and_web_search_a_warning(tmp_path):
    fetch = ("WebFetch", {"url": "https://arxiv.org/abs/1103.0576", "prompt": "authors?"})
    row = _run(tmp_path, _stream(extra=[fetch]))
    assert _status(row)["no_bypass"] == "FAIL" and row["failure"] == "no_bypass"
    row = _run(tmp_path / "2", _stream(extra=[("WebSearch", {"query": "gravitational waves 2011"})]))
    assert row["verdict"] == "PASS" and _status(row)["no_bypass"] == "WARN"


@pytest.mark.parametrize("command", [
    "curl -s 'http://export.arxiv.org/api/query?search_query=ti:graphene'",
    "python3 -c \"import urllib.request; urllib.request.urlopen('https://arxiv.org/pdf/1103.0576')\"",
    "pip install arxiv",
    "uv run --with arxiv python search.py",  # \b before --with never matched
    "python3 - <<'EOF'\nimport feedparser\nEOF",
])
def test_shell_access_to_arxiv_is_a_bypass(tmp_path, command):
    row = _run(tmp_path, _stream(extra=[("Bash", {"command": command})]))
    assert _status(row)["no_bypass"] == "FAIL"


def test_writing_the_answer_with_python_is_not_a_bypass(tmp_path):
    command = "python3 -c \"import json; json.dump({'arxiv_ids': ['1103.0115']}, open('result.json','w'))\""
    row = _run(tmp_path, _stream(extra=[("Bash", {"command": command})]))
    assert _status(row)["no_bypass"] == "PASS"


def test_extra_server_tools_warn_about_the_workspace_trap(tmp_path):
    tools = ["Bash", SEARCH, SNIPPETS, "mcp__arxiv__UniProt_get_entry_by_accession", "mcp__arxiv__Finish"]
    row = _run(tmp_path, _stream(tools=tools))
    assert row["verdict"] == "PASS" and _status(row)["mcp_connected"] == "WARN"
    assert "offered 4 tools, expected 2" in row["checks"]["mcp_connected"]["detail"]


def test_in_band_tool_errors_are_not_results(tmp_path):
    error = json.dumps({"status": "error", "error": "PDF download failed (HTTP 404)"})
    row = _run(tmp_path, _stream(snippet_calls=[(SNIPPET_ARGS, error)]))
    assert row["failure"] == "tool_correct"


def test_codex_run_passes_and_web_search_items_are_recorded(tmp_path):
    row = _run(tmp_path, _codex(), codex=True)
    assert row["verdict"] == "PASS", row["checks"]
    assert set(_status(row).values()) == {"PASS"}
    web = {"id": "w1", "type": "web_search", "query": "arxiv 1103.0576 authors"}
    row = _run(tmp_path / "2", _codex([web]), codex=True)
    assert row["tool_sequence"][0] == "web_search" and _status(row)["no_bypass"] == "WARN"


def test_extractors_canonicalise_ids_and_counts():
    ext, values = verify.extractors, verify.values
    assert ext.canon_ids(["https://arxiv.org/abs/1103.0291v1", "arXiv:1103.0212", "solv-int/9901001v2",
                          "http://arxiv.org/pdf/1206.1177.pdf"]) == \
        ["1103.0291", "1103.0212", "solv-int/9901001", "1206.1177"]
    assert ext.canon_ids([]) is None and ext.canon_ids(3) is None
    assert ext.canon_counts({"Arecibo ": "3"}) == {"arecibo": 3}
    assert ext.canon_counts({"a": 1.5}) is None and ext.canon_counts({}) is None
    assert values.matches({"a": 1}, {"a": 1, "b": 2}, "subset")
    assert not values.matches({"a": 1, "c": 0}, {"a": 1, "b": 2}, "subset")
    assert values.matches(["x", "y"], "y", "member") and not values.matches(None, "y", "member")
    truncated = json.dumps({"data": json.loads(_search_result()), "_truncated": True})
    call = verify.evidence.ToolCall(result_text=truncated)
    assert values.read(call, verify.spec.Selector(extract="arxiv_ids")) == IDS
