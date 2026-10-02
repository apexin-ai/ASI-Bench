"""e2e_smoke/servers/arxiv.py without network: ID/Atom parsing, record diffs, snippet reference, search/snippet checks."""

import pytest

from . import support
from .support import StubClient, rpc_json, rpc_text, runner

arxiv = support.smoke_module("arxiv")


arxiv.THROTTLE.gap = 0.0


ATOM_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <opensearch:totalResults>2</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/abs/1103.0212v1</id>
    <updated>2011-03-01T16:39:23Z</updated><published>2011-03-01T16:39:23Z</published>
    <title>Nonlinear conductance quantization
  in graphene ribbons</title>
    <summary>  We present numerical
 studies.  </summary>
    <author><name>A. Author</name></author><author><name>B. Author</name></author>
    <arxiv:primary_category term="cond-mat.mes-hall"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/solv-int/9901001v2</id>
    <updated>1999-02-01T00:00:00Z</updated><published>1999-01-04T00:00:00Z</published>
    <title>The Camassa-Holm Equation</title><summary>x</summary>
    <author><name>C. Author</name></author>
    <arxiv:primary_category term="solv-int"/>
  </entry>
</feed>"""


def _tool_paper(ident, title="Nonlinear conductance quantization in graphene ribbons", **extra):
    return {"title": title, "abstract": "We present numerical studies.", "authors": ["A. Author", "B. Author"],
            "published": "2011-03-01T16:39:23Z", "updated": "2011-03-01T16:39:23Z",
            "category": "cond-mat.mes-hall", "url": f"https://arxiv.org/abs/{ident}", **extra}


def test_split_arxiv_id_handles_urls_versions_and_old_style_ids():
    assert arxiv.split_arxiv_id("https://arxiv.org/abs/1103.0212v1") == ("1103.0212", "v1")
    assert arxiv.split_arxiv_id("http://arxiv.org/pdf/1103.0212.pdf") == ("1103.0212", "")
    assert arxiv.split_arxiv_id("arXiv:2301.12345v12") == ("2301.12345", "v12")
    assert arxiv.split_arxiv_id("http://arxiv.org/abs/solv-int/9901001v2") == ("solv-int/9901001", "v2")


def test_parse_atom_normalises_whitespace_and_reads_total():
    records, total = arxiv.parse_atom(ATOM_FEED)
    assert total == 2
    assert [(r["id"], r["version"]) for r in records] == [("1103.0212", "v1"), ("solv-int/9901001", "v2")]
    assert records[0]["title"] == "Nonlinear conductance quantization in graphene ribbons"
    assert records[0]["abstract"] == "We present numerical studies."
    assert records[0]["authors"] == ["A. Author", "B. Author"]
    assert records[0]["category"] == "cond-mat.mes-hall"


def test_record_diffs_reports_order_and_field_changes():
    ref, _ = arxiv.parse_atom(ATOM_FEED)
    ref = ref[:1]
    assert arxiv.record_diffs(arxiv.tool_records([_tool_paper("1103.0212v1")]), ref) == []
    assert "IDs/order" in arxiv.record_diffs(arxiv.tool_records([_tool_paper("1103.0212v2")]), ref)[0]
    diffs = arxiv.record_diffs(arxiv.tool_records([_tool_paper("1103.0212v1", title="Other")]), ref)
    assert len(diffs) == 1 and diffs[0].startswith("1103.0212.title")


def test_expected_snippets_windows_limits_and_cap():
    text = "aa Plateau bb plateau cc PLATEAU dd Landauer ee"
    got, dropped = arxiv.expected_snippets(text, ["plateau", "LANDAUER"], 3, 2, 8000)
    assert got == [{"term": "plateau", "snippet": "aa Plateau bb"},
                   {"term": "plateau", "snippet": "bb plateau cc"},
                   {"term": "LANDAUER", "snippet": "dd Landauer ee"}]
    assert dropped == 0
    # 13 + 13 = 26 fits in 27; the next term's 14-char snippet would not -> dropped, term ends
    got, dropped = arxiv.expected_snippets(text, ["plateau", "landauer"], 3, 2, 27)
    assert [s["term"] for s in got] == ["plateau", "plateau"] and dropped == 1


def test_in_band_error_detection():
    assert arxiv.in_band_error({"status": "error", "error": "boom"}) == "boom"
    assert arxiv.in_band_error({"error": "x"}) == "x"
    assert arxiv.in_band_error({"status": "success", "snippets": []}) is None
    assert arxiv.in_band_error([{"title": "t"}]) is None


def test_check_search_compares_with_reference(monkeypatch):
    ref, _ = arxiv.parse_atom(ATOM_FEED)
    monkeypatch.setattr(arxiv, "reference_search", lambda *a: ref[:1])
    label, arguments, query = arxiv.SEARCH_CASES[0]

    def run(papers):
        report = arxiv.Report()
        client = StubClient({arxiv.SEARCH: rpc_json(papers)})
        arxiv.check_search(arxiv.Caller(client, report), report, label, arguments, query)
        return report.checks[-1]

    assert run([_tool_paper("1103.0212v1")])["status"] == "PASS"
    assert run([_tool_paper("1103.0212v1", title="Wrong")])["status"] == "FAIL"
    assert run([])["status"] == "FAIL"
    outside = _tool_paper("1103.0212v1", published="2010-01-01T00:00:00Z")
    assert "outside the date range" in run([outside])["detail"]
    monkeypatch.setattr(arxiv, "reference_search", lambda *a: [])
    assert "reference returned no papers" in run([])["detail"]


def test_check_unparenthesised_or_warns_on_out_of_window_results():
    def run(papers):
        report = arxiv.Report()
        arxiv.check_unparenthesised_or(arxiv.Caller(StubClient({arxiv.SEARCH: rpc_json(papers)}), report), report)
        return report.checks[-1]["status"]

    assert run([_tool_paper("1103.0212v1")]) == "PASS"
    assert run([_tool_paper("0709.2066v1", published="2007-09-13T13:12:33Z")]) == "WARN"


@pytest.mark.parametrize("response,expected", [
    (rpc_text("bad", is_error=True), "PASS"),
    (rpc_json({"status": "error", "error": "`query` parameter is required."}), "WARN"),
    (rpc_json([]), "FAIL"),
])
def test_arxiv_in_band_error_classification(response, expected):
    report = arxiv.Report()
    call = arxiv.Caller(StubClient({arxiv.SEARCH: response}), report)
    arxiv.check_in_band_error(call, report, arxiv.SEARCH, "empty query", {"query": ""})
    assert report.checks[-1]["status"] == expected


def test_compare_snippets_requires_exact_reference_match():
    expected = [{"term": "plateau", "snippet": "aa Plateau bb"}]
    url = "https://arxiv.org/pdf/1103.0212.pdf"
    good = {"status": "success", "pdf_url": url, "snippets": expected, "snippets_count": 1, "truncated": False}
    for payload, ok in ((good, True),
                        ({**good, "pdf_url": "https://arxiv.org/pdf/other.pdf"}, False),
                        ({**good, "snippets": [{"term": "plateau", "snippet": "changed"}]}, False),
                        ({**good, "snippets_count": 2}, False),
                        ({"status": "error", "error": "PDF download failed"}, False)):
        report = arxiv.Report()
        assert arxiv.compare_snippets(report, "s", payload, expected, url) is ok
        assert report.checks[-1]["status"] == ("PASS" if ok else "FAIL")


def test_arxiv_server_env_is_minimal_and_applies_config_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy:3128")
    env = runner.server_env({"command": "/x/.venv/bin/tooluniverse-smcp-stdio",
                             "env": {"TOOLUNIVERSE_CACHE_ENABLED": "false"}}, tmp_path, tmp_path / "tmp",
                            pass_proxies=arxiv.SMOKE.pass_proxies)
    assert env["HOME"] == str(tmp_path) and env["PATH"].startswith("/x/.venv/bin:")
    assert env["TOOLUNIVERSE_CACHE_ENABLED"] == "false" and env["HTTPS_PROXY"] == "http://proxy:3128"
    assert "OPENAI_API_KEY" not in env
