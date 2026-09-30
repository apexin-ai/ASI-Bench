"""Offline checks for the opt-in MCP E2E setup/smoke scripts (no network, no upstream installs)."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from ai4sci_bench.mcp_config import load_mcp_config, load_science_mcp_catalog

BUNDLE = Path(__file__).resolve().parents[1] / "scripts/mcp/e2e"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"mcp_e2e_{name}", BUNDLE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


setup = _load("setup")
sys.path.insert(0, str(BUNDLE))
stdio_client = _load("stdio_client")


def test_manifest_is_pinned_and_catalogued():
    servers = setup.load_manifest()
    catalog = load_science_mcp_catalog()
    assert "pyscf" in servers
    for sid, entry in servers.items():
        assert len(entry["revision"]) == 40
        assert entry["catalog_id"] in catalog
        assert catalog[entry["catalog_id"]]["source"].rstrip("/") == entry["repository"].removesuffix(".git")
        assert (BUNDLE / entry["smoke"]).is_file()
        assert entry["expected_tools"] == sorted(set(entry["expected_tools"]))


def test_rendered_config_is_valid_and_absolute(tmp_path):
    entry = setup.load_manifest()["pyscf"]
    dest = tmp_path / "pyscf"
    path = tmp_path / "pyscf.mcp.json"
    path.write_text(json.dumps(setup.render_config(entry, dest)))
    server = load_mcp_config(path)["pyscf"]
    assert server["command"] == str(dest / ".venv/bin/python")
    assert server["args"] == [str(dest / "main.py")]
    assert "/path/to/" not in path.read_text()


def test_manifest_rejects_absolute_launch(tmp_path):
    document = json.loads((BUNDLE / "manifest.json").read_text())
    document["servers"][0]["launch"]["command"] = "/usr/bin/python3"
    bad = tmp_path / "manifest.json"
    bad.write_text(json.dumps(document))
    with pytest.raises(setup.SetupError, match="relative"):
        setup.load_manifest(bad)


def test_existing_checkout_with_wrong_remote_is_refused(tmp_path):
    entry = setup.load_manifest()["pyscf"]
    dest = tmp_path / "pyscf"
    subprocess.run(["git", "init", "-q", str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "remote", "add", "origin", "https://example.invalid/other.git"], check=True)
    with pytest.raises(setup.SetupError, match="points at"):
        setup.ensure_checkout(entry, dest)


FAKE_SERVER = textwrap.dedent('''
    import json, sys
    for line in sys.stdin:
        msg = json.loads(line)
        if "id" not in msg:
            continue
        method = msg["method"]
        if method == "initialize":
            result = {"protocolVersion": msg["params"]["protocolVersion"], "capabilities": {"tools": {}},
                      "serverInfo": {"name": "fake", "version": "0"}}
        elif method == "tools/list":
            if msg["params"].get("cursor"):
                result = {"tools": [{"name": "b", "inputSchema": {"type": "object"}}]}
            else:
                result = {"tools": [{"name": "a", "inputSchema": {"type": "object"}}], "nextCursor": "p2"}
        elif method == "tools/call":
            print("library chatter on stdout", flush=True)
            if msg["params"]["name"] == "a":
                result = {"content": [{"type": "text", "text": "42"}], "isError": False}
            else:
                result = {"content": [{"type": "text", "text": "unknown"}], "isError": True}
        else:
            result = {}
        print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result}), flush=True)
''')


def test_stdio_client_paginates_and_records_stdout_pollution(tmp_path):
    script = tmp_path / "fake_server.py"
    script.write_text(FAKE_SERVER)
    client = stdio_client.StdioMCP(sys.executable, [str(script)], cwd=tmp_path,
                                   env={"PATH": "/usr/bin:/bin"}, stderr_path=tmp_path / "err.log")
    try:
        assert client.initialize(timeout=30)["serverInfo"]["name"] == "fake"
        assert [t["name"] for t in client.list_tools()] == ["a", "b"]
        ok = client.call_tool("a", {}, timeout=30)
        assert stdio_client.text_of(ok["result"]) == "42"
        assert client.call_tool("zzz", {}, timeout=30)["result"]["isError"] is True
    finally:
        client.close()
    assert client.non_json_stdout == ["library chatter on stdout"] * 2


# --- smoke_pyscf helpers and PySCF-free checks -------------------------------

smoke = _load("smoke_pyscf")


def test_parse_atom_string_accepts_semicolons_and_newlines():
    atoms = smoke.parse_atom_string("O 0 0 0.1;\nH 0 0.7 -0.4; H 0 -0.7 -0.4")
    assert smoke.symbols(atoms) == ["O", "H", "H"]
    assert atoms[1][1] == (0.0, 0.7, -0.4)
    with pytest.raises(ValueError):
        smoke.parse_atom_string("Error: Invalid SMILES string: xx.")
    with pytest.raises(ValueError):
        smoke.parse_atom_string("  ;  ")


def test_parse_xyz_block_checks_atom_count():
    block = "2\nOptimized geometry (HF/STO-3G)\nH 0 0 0\nH 0 0 0.74\n"
    assert smoke.symbols(smoke.parse_xyz_block(block)) == ["H", "H"]
    with pytest.raises(ValueError, match="3 atoms"):
        smoke.parse_xyz_block(block.replace("2\n", "3\n", 1))


def test_sorted_distances_is_rotation_and_permutation_invariant():
    water = smoke.parse_atom_string("O 0 0 0.1178; H 0 0.7555 -0.4712; H 0 -0.7555 -0.4712")
    # 90° rotation about x, translation, and H permutation
    moved = [(s, (x + 1.0, -z, y)) for s, (x, y, z) in (water[0], water[2], water[1])]
    assert smoke.max_abs_diff(smoke.sorted_distances(water), smoke.sorted_distances(moved)) < 1e-12
    assert smoke.max_abs_diff([1.0], [1.0, 2.0]) == float("inf")


def test_parse_float_list_rejects_non_lists():
    assert smoke.parse_float_list("[-1.1, -1.2]") == [-1.1, -1.2]
    for bad in ('{"energies": []}', "[true]", '["1.0"]'):
        with pytest.raises(ValueError):
            smoke.parse_float_list(bad)


def _png(width, height):
    import struct
    import zlib
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + struct.pack(">I", len(ihdr)) + b"IHDR" + ihdr
            + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr)))


def test_png_size():
    assert smoke.png_size(_png(1000, 600)) == (1000, 600)
    with pytest.raises(ValueError):
        smoke.png_size(b"GIF89a" + b"\0" * 20)


class _StubClient:
    """Stands in for StdioMCP: canned tools/call responses, optional stdout chatter."""

    def __init__(self, responses, chatter=0):
        self.responses = responses
        self.chatter = chatter
        self.non_json_stdout = []
        self.calls = []

    def call_tool(self, name, arguments, timeout=300.0):
        self.calls.append((name, arguments))
        self.non_json_stdout.extend(["noise"] * self.chatter)
        response = self.responses[name]
        return response(arguments) if callable(response) else response


def _text(text, is_error=False):
    return {"result": {"content": [{"type": "text", "text": text}], "isError": is_error}}


def _statuses(report):
    return {c["name"]: c["status"] for c in report.checks}


@pytest.mark.parametrize("response,expected", [
    (_text("boom", is_error=True), "PASS"),
    (_text("Error: Invalid SMILES string"), "WARN"),
    (_text("C 0 0 0"), "FAIL"),
])
def test_in_band_error_classification(response, expected):
    report = smoke.Report()
    call = smoke.Caller(_StubClient({"t": response}), report)
    smoke.check_in_band_error(call, report, "t", {}, "error")
    assert _statuses(report) == {"t[invalid input]": expected}


def test_caller_fails_on_tool_error_and_attributes_stdout():
    report = smoke.Report()
    call = smoke.Caller(_StubClient({"t": _text("kaput", is_error=True)}, chatter=2), report)
    assert call("check", "t", {}) is None
    assert _statuses(report) == {"check": "FAIL"}
    assert call.stdout_by_tool == {"t": 2}


def _image(data, mime="image/png"):
    return {"result": {"content": [{"type": "image", "data": data, "mimeType": mime}], "isError": False}}


def test_check_plot_accepts_png_and_rejects_other_payloads():
    import base64
    good = base64.b64encode(_png(1000, 600)).decode()

    def run(first):
        report = smoke.Report()
        client = _StubClient({"plot_energy_scan_image_mcp":
                              lambda args: first if len(args["energies"]) == len(args["bond_lengths"])
                              else _text("validation error", is_error=True)})
        smoke.check_plot(smoke.Caller(client, report), report)
        return _statuses(report)

    assert run(_image(good)) == {"plot_energy_scan_image_mcp": "PASS",
                                 "plot_energy_scan_image_mcp[mismatched lengths]": "PASS"}
    assert run(_image(good, mime="image/jpeg"))["plot_energy_scan_image_mcp"] == "FAIL"
    assert run(_image(base64.b64encode(b"not a png at all, sorry").decode()))["plot_energy_scan_image_mcp"] == "FAIL"
    assert run(_text("no image"))["plot_energy_scan_image_mcp"] == "FAIL"


def test_check_visualize_separates_side_effect_from_returned_path(tmp_path):
    def writes_html(message):
        def respond(args):
            (tmp_path / smoke.VISUALIZE_FILE).write_text(f"<script>3Dmol</script>{args['xyz_string']}")
            return _text(message)
        return respond

    report = smoke.Report()
    client = _StubClient({"visualize_molecule_3d_mcp": writes_html("saved in /Users/someone/x.html")})
    smoke.check_visualize(smoke.Caller(client, report), report, tmp_path)
    assert _statuses(report) == {"visualize_molecule_3d_mcp[HTML written]": "PASS",
                                 "visualize_molecule_3d_mcp[result locates output]": "WARN"}

    report = smoke.Report()
    client = _StubClient({"visualize_molecule_3d_mcp":
                          writes_html(f"saved in {tmp_path / smoke.VISUALIZE_FILE}")})
    smoke.check_visualize(smoke.Caller(client, report), report, tmp_path)
    assert _statuses(report)["visualize_molecule_3d_mcp[result locates output]"] == "PASS"

    report = smoke.Report()
    (tmp_path / smoke.VISUALIZE_FILE).unlink()
    client = _StubClient({"visualize_molecule_3d_mcp": _text("done")})
    smoke.check_visualize(smoke.Caller(client, report), report, tmp_path)
    assert _statuses(report)["visualize_molecule_3d_mcp[HTML written]"] == "FAIL"


@pytest.mark.parametrize("server", sorted(setup.load_manifest()))
def test_smoke_covers_every_manifest_tool(server):
    entry = setup.load_manifest()[server]
    source = (BUNDLE / entry["smoke"]).read_text()
    for tool in entry["expected_tools"]:
        assert f'"{tool}"' in source, f"{entry['smoke']} never calls {tool}"


# --- manifest: env, uv sync flags, {checkout} placeholder --------------------

def test_arxiv_config_keeps_flags_literal_and_carries_env(tmp_path):
    entry = setup.load_manifest()["arxiv"]
    dest = tmp_path / "arxiv"
    path = tmp_path / "arxiv.mcp.json"
    path.write_text(json.dumps(setup.render_config(entry, dest)))
    server = load_mcp_config(path)["arxiv"]
    assert server["command"] == str(dest / ".venv/bin/tooluniverse-smcp-stdio")
    assert server["args"] == ["--no-search", "--include-tools", "ArXiv_search_papers", "ArXiv_get_pdf_snippets"]
    assert server["cwd"] == str(dest)
    # result cache off (real calls every time, no ~/.tooluniverse); no workspace override
    assert server["env"]["TOOLUNIVERSE_CACHE_ENABLED"] == "false"
    assert server["env"]["TOOLUNIVERSE_CACHE_PERSIST"] == "false"
    assert "TOOLUNIVERSE_HOME" not in server["env"]
    assert entry["uv_sync_args"] == ["--no-dev"]


@pytest.mark.parametrize("field,value,match", [
    (("launch", "args"), ["{checkout}main.py"], "placeholder|prefix"),
    (("launch", "env"), {"A": 1}, "strings"),
    (("uv_sync_args",), ["--python", "3.9"], "uv_sync_args"),
    (("uv_sync_args",), ["no-dash"], "uv_sync_args"),
])
def test_manifest_rejects_bad_launch_fields(tmp_path, field, value, match):
    document = json.loads((BUNDLE / "manifest.json").read_text())
    target = document["servers"][0]
    for key in field[:-1]:
        target = target[key]
    target[field[-1]] = value
    bad = tmp_path / "manifest.json"
    bad.write_text(json.dumps(document))
    with pytest.raises(setup.SetupError, match=match):
        setup.load_manifest(bad)


# --- smoke_arxiv helpers and network-free checks -----------------------------

arxiv = _load("smoke_arxiv")
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


def _json(payload, is_error=False):
    return _text(json.dumps(payload), is_error=is_error)


def test_check_search_compares_with_reference(monkeypatch):
    ref, _ = arxiv.parse_atom(ATOM_FEED)
    monkeypatch.setattr(arxiv, "reference_search", lambda *a: ref[:1])
    label, arguments, query = arxiv.SEARCH_CASES[0]

    def run(papers):
        report = arxiv.Report()
        client = _StubClient({arxiv.SEARCH: _json(papers)})
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
        arxiv.check_unparenthesised_or(arxiv.Caller(_StubClient({arxiv.SEARCH: _json(papers)}), report), report)
        return report.checks[-1]["status"]

    assert run([_tool_paper("1103.0212v1")]) == "PASS"
    assert run([_tool_paper("0709.2066v1", published="2007-09-13T13:12:33Z")]) == "WARN"


@pytest.mark.parametrize("response,expected", [
    (_text("bad", is_error=True), "PASS"),
    (_json({"status": "error", "error": "`query` parameter is required."}), "WARN"),
    (_json([]), "FAIL"),
])
def test_arxiv_in_band_error_classification(response, expected):
    report = arxiv.Report()
    call = arxiv.Caller(_StubClient({arxiv.SEARCH: response}), report)
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
    env = arxiv.server_env({"command": "/x/.venv/bin/tooluniverse-smcp-stdio",
                            "env": {"TOOLUNIVERSE_CACHE_ENABLED": "false"}}, tmp_path)
    assert env["HOME"] == str(tmp_path) and env["PATH"].startswith("/x/.venv/bin:")
    assert env["TOOLUNIVERSE_CACHE_ENABLED"] == "false" and env["HTTPS_PROXY"] == "http://proxy:3128"
    assert "OPENAI_API_KEY" not in env
