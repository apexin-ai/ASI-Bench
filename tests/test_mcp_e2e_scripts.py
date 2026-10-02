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


# --- jsbsim: pinned pip install, {checkout} in launch env ---------------------

def test_jsbsim_config_resolves_checkout_in_env(tmp_path):
    entry = setup.load_manifest()["jsbsim"]
    dest = tmp_path / "jsbsim"
    path = tmp_path / "jsbsim.mcp.json"
    path.write_text(json.dumps(setup.render_config(entry, dest)))
    server = load_mcp_config(path)["jsbsim"]
    assert server["command"] == str(dest / ".venv/bin/python")
    assert server["args"] == [str(dest / "run_stdio.py")]
    # JBSIM_ROOT (upstream's spelling) must be absolute: MCP clients may ignore cwd
    assert server["env"] == {"JBSIM_ROOT": str(dest / "jsbsim_data"), "JSBSIM_DEBUG": "0"}
    assert all("==" in r for r in entry["requirements"]) and entry["install"] == "uv-pip-pinned"


def test_pinned_install_builds_fresh_venv_with_exact_pins(tmp_path, monkeypatch):
    entry = setup.load_manifest()["jsbsim"]
    commands = []

    def fake_run(cmd, cwd=None, env=None):
        commands.append(cmd)
        if cmd[0] == "uv":
            assert env is not None and "UV_PYTHON" not in env and "VIRTUAL_ENV" not in env
        return entry["python"] if cmd[-1].startswith("import sys") else ""

    monkeypatch.setenv("UV_PYTHON", "3.9")
    monkeypatch.setattr(setup, "run", fake_run)
    monkeypatch.setattr(setup.shutil, "which", lambda name: "/usr/bin/uv")
    python = setup.build_env(entry, tmp_path)
    assert python == tmp_path / ".venv/bin/python"
    assert commands[0] == ["uv", "venv", "--clear", "--python", "3.12", str(tmp_path / ".venv")]
    assert commands[1] == ["uv", "pip", "install", "--python", str(python), "--exclude-newer",
                           entry["exclude_newer"], *entry["requirements"]]
    assert not any("sync" in c for c in commands)


def _jsbsim_manifest(tmp_path, **changes):
    document = json.loads((BUNDLE / "manifest.json").read_text())
    entry = next(e for e in document["servers"] if e["id"] == "jsbsim")
    for key, value in changes.items():
        if key == "env":
            entry["launch"]["env"] = value
        elif value is None:
            entry.pop(key, None)
        else:
            entry[key] = value
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document))
    return path


@pytest.mark.parametrize("changes,match", [
    ({"requirements": ["jsbsim>=1.3.1"]}, "exact"),
    ({"requirements": []}, "exact"),
    ({"requirements": None}, "exact"),
    ({"exclude_newer": None}, "exclude_newer"),
    ({"exclude_newer": "2026-10-01"}, "exclude_newer"),
    ({"uv_sync_args": ["--no-dev"]}, "uv_sync_args"),
    ({"install": "pip"}, "unsupported install"),
    ({"install": "uv-sync-frozen"}, "only apply"),
    ({"env": {"JBSIM_ROOT": "/abs/jsbsim_data"}}, "absolute"),
    ({"env": {"JBSIM_ROOT": "x{checkout}/jsbsim_data"}}, "prefix"),
])
def test_manifest_rejects_bad_pinned_install(tmp_path, changes, match):
    with pytest.raises(setup.SetupError, match=match):
        setup.load_manifest(_jsbsim_manifest(tmp_path, **changes))


# --- smoke_jsbsim helpers and JSBSim-free checks -----------------------------

jsb = _load("smoke_jsbsim")


def test_frames_for_rounds_and_never_returns_zero():
    assert jsb.frames_for(10.0) == 600
    assert jsb.frames_for(5.0) == 300
    assert jsb.frames_for(0.0) == 1
    assert jsb.frames_for(1.0, 1 / 120) == 120


def test_aircraft_dirs_requires_name_xml(tmp_path):
    for name, files in {"c172x": ["c172x.xml"], "odd": ["other.xml"], "empty": []}.items():
        (tmp_path / "aircraft" / name).mkdir(parents=True)
        for f in files:
            (tmp_path / "aircraft" / name / f).write_text("<fdm_config/>")
    (tmp_path / "aircraft" / "aircraft_template.xml").write_text("x")
    assert jsb.aircraft_dirs(tmp_path) == ["c172x"]


def test_jsbsim_in_band_error_detection():
    assert jsb.in_band_error({"ok": False, "error": "unknown-session"}) == "unknown-session"
    assert jsb.in_band_error({"ok": False, "note": "script not found"}) == "script not found"
    assert jsb.in_band_error({"ok": True, "path": "x", "value": 1.0}) is None
    assert jsb.in_band_error([1, 2]) is None


def _telemetry_reference(**overrides):
    ref = {}
    for path, decimals, real in jsb.TELEMETRY_FIELDS.values():
        ref[path] = None if real else 1.0
        if real:
            ref[real] = 5.0
    ref.update(overrides)
    return ref


def test_compare_telemetry_separates_wrong_values_from_dead_fields():
    ref = _telemetry_reference(**{"position/h-sl-ft": 4115.051741, "velocities/mach": 0.1234567})
    frame = {name: (1.0 if decimals is not None else True) for name, (_, decimals, _) in jsb.TELEMETRY_FIELDS.items()}
    frame.update(alt_ft=4115.05, mach=0.123, pitch_deg=0.0, rpm=0)
    mismatches, dead = jsb.compare_telemetry(frame, ref)
    assert mismatches == []
    assert len(dead) == sum(1 for _, _, real in jsb.TELEMETRY_FIELDS.values() if real)
    assert any(d.startswith("pitch_deg<-attitude/pitch-deg (real attitude/theta-deg=5") for d in dead)

    frame.update(alt_ft=4115.07, mach=float("nan"))
    mismatches, _ = jsb.compare_telemetry(frame, ref)
    assert [m.split("=")[0] for m in mismatches] == ["alt_ft", "mach"]


def test_compare_telemetry_dead_field_without_known_real_property():
    ref = _telemetry_reference(**{"forces/lift-lbs": None})
    frame = {name: 1.0 for name in jsb.TELEMETRY_FIELDS}
    frame["lift_lbs"] = 0.0
    _, dead = jsb.compare_telemetry(frame, ref)
    assert "lift_lbs<-forces/lift-lbs (server 0.0)" in dead


def test_trim_digest_ignores_mode_and_ok():
    a = {"ok": True, "mode": "longitudinal", "throttle": 0.7, "elevator": 0.0}
    b = {"ok": True, "mode": "none", "throttle": 0.7, "elevator": 0.0}
    assert jsb.trim_digest(a) == jsb.trim_digest(b)
    assert jsb.trim_digest(a) != jsb.trim_digest({**b, "throttle": 0.8})


def _json(payload, is_error=False):
    return _text(json.dumps(payload), is_error)


def test_jsbsim_missing_property_and_unknown_session_classification():
    report = jsb.Report()
    client = _StubClient({"get_property": _json({"path": "no/such/property", "value": 0.0, "present": True})})
    jsb.check_missing_property(jsb.Caller(client, report), report, "s1")
    assert _statuses(report) == {"get_property[missing path]": "WARN"}

    report = jsb.Report()
    client = _StubClient({"get_property": _json({"path": "no/such/property", "value": None, "present": False})})
    jsb.check_missing_property(jsb.Caller(client, report), report, "s1")
    assert _statuses(report) == {"get_property[missing path]": "PASS"}

    report = jsb.Report()
    client = _StubClient({
        "create_session": _text("Error executing tool create_session: Failed to load aircraft", is_error=True),
        "step": _json({"ok": False, "error": "unknown-session"}),
        "get_telemetry": _text("unknown session", is_error=True),
        "trim": _json({"ok": True, "mode": "longitudinal"}),
        "__nonexistent__": _text("Unknown tool", is_error=True),
    })
    jsb.check_errors(jsb.Caller(client, report), report, client)
    assert _statuses(report) == {"create_session[unknown aircraft]": "PASS", "step[unknown session]": "WARN",
                                 "get_telemetry[unknown session]": "PASS", "trim[unknown session]": "FAIL",
                                 "unknown tool is an error": "PASS"}


def test_list_aircraft_compared_with_directory_scan(tmp_path):
    for name in ("c172x", "737"):
        (tmp_path / "aircraft" / name).mkdir(parents=True)
        (tmp_path / "aircraft" / name / f"{name}.xml").write_text("x")
    good = _json({"aircraft": ["737", "c172x"], "count": 2})
    report = jsb.Report()
    jsb.check_list_aircraft(jsb.Caller(_StubClient({"list_aircraft": good}), report), report, tmp_path)
    assert _statuses(report) == {"list_aircraft": "PASS"}
    report = jsb.Report()
    bad = _json({"aircraft": ["737"], "count": 2})
    jsb.check_list_aircraft(jsb.Caller(_StubClient({"list_aircraft": bad}), report), report, tmp_path)
    assert _statuses(report) == {"list_aircraft": "FAIL"}


class _DyingClient(_StubClient):
    """Like _StubClient, but the server process dies at `die_at`."""

    class _Proc:
        def poll(self):
            return -11

    def __init__(self, responses, die_at, chatter=0):
        super().__init__(responses, chatter)
        self.die_at = die_at
        self.proc = self._Proc()

    def call_tool(self, name, arguments, timeout=300.0):
        if name == self.die_at:
            raise jsb.MCPError("server closed stdout (exit=-11) while waiting for tools/call")
        return super().call_tool(name, arguments, timeout)


def test_stock_script_probe_turns_crashes_into_warn():
    responses = {"execute_script": _json({"ok": True, "note": "queued c1722.xml"}),
                 "step": _text("Error executing tool step: Trim Failed", is_error=True),
                 "get_property": _text('{"path": "position/h-sl-ft", "value": NaN, "present": true}'),
                 "close_session": _json({"ok": True}), "list_aircraft": _json({"aircraft": []})}
    status, detail = jsb.stock_script_probe(_DyingClient(responses, die_at="close_session", chatter=3), "s", 0)
    assert status == "WARN"
    assert "died during close_session" in detail and "h=nan" in detail and "next step fails" in detail
    status, detail = jsb.stock_script_probe(_DyingClient(responses, die_at="execute_script"), "s", 0)
    assert status == "WARN" and "died during execute_script" in detail
    healthy = dict(responses, step=_json({"frames": 60}),
                   get_property=_json({"path": "position/h-sl-ft", "value": 4000.0, "present": True}))
    assert jsb.stock_script_probe(_DyingClient(healthy, die_at=None), "s", 0)[0] == "PASS"


def test_jsbsim_server_env_is_minimal_and_applies_config_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    env = jsb.server_env({"command": "/m/jsbsim/.venv/bin/python",
                          "env": {"JBSIM_ROOT": "/m/jsbsim/jsbsim_data", "JSBSIM_DEBUG": "0"}},
                         tmp_path / "home", tmp_path / "tmp")
    assert env["TMPDIR"] == str(tmp_path / "tmp") and env["PATH"].startswith("/m/jsbsim/.venv/bin:")
    assert env["JSBSIM_DEBUG"] == "0" and env["JBSIM_ROOT"] == "/m/jsbsim/jsbsim_data"
    assert "ANTHROPIC_API_KEY" not in env
