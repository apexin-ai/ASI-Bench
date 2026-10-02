"""Offline checks for the opt-in MCP E2E setup/smoke scripts (no network, no upstream installs)."""
import importlib.util
import json
import re
import math
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
    sys.modules[spec.name] = module  # dataclasses need the module registered (Python 3.14)
    spec.loader.exec_module(module)
    return module


setup = _load("setup")
sys.path.insert(0, str(BUNDLE))
stdio_client = _load("stdio_client")
verify = _load("verify_run")

E2E_TASKS = Path(__file__).resolve().parents[1] / "examples/mcp-e2e-tasks"
_SPEC_FILES = sorted(E2E_TASKS.glob("mcp_e2e/*/e2e_check.json"))


def _valid_spec():
    """A minimal valid schema-2 spec dict for mutation tests."""
    return {
        "schema_version": 2, "server": "demo",
        "reference_file": "reference.json", "prediction_file": "result.json",
        "calls": [{"name": "c", "tool": "do_thing",
                   "inputs_from_reference": {"x": "x_ref"},
                   "result": {"format": "number", "reference_key": "y_ref", "abs_tol": 1e-6}}],
        "answers": [{"prediction_key": "y", "from_call": "c",
                     "result_key": None, "reference_key": "y_ref", "abs_tol": 1e-6}],
        "bypass_patterns": [r"import\s+demo"],
    }



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


# --- s4: host requirements, PYTHONPATH launch ---------------------------------

def test_s4_config_runs_module_from_checkout_src(tmp_path):
    entry = setup.load_manifest()["s4"]
    dest = tmp_path / "s4"
    path = tmp_path / "s4.mcp.json"
    path.write_text(json.dumps(setup.render_config(entry, dest)))
    server = load_mcp_config(path)["s4"]
    assert server["command"] == str(dest / ".venv/bin/python")
    assert server["args"] == ["-m", "mcp_s4_rcwa.server"]
    assert server["env"] == {"PYTHONPATH": str(dest / "src")}
    assert entry["install"] == "uv-pip-pinned" and all("==" in r for r in entry["requirements"])
    assert entry["host_requirements"]["machine"] == ["x86_64"]


_X86_CPUINFO = "processor\t: 0\nflags\t\t: fpu sse2 avx avx2 bmi1 bmi2 fma\n"


def _loader(available):
    def load(name):
        if name not in available:
            raise OSError(f"{name}: cannot open shared object file")
    return load


def test_check_host_accepts_matching_host(tmp_path):
    entry = setup.load_manifest()["s4"]
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text(_X86_CPUINFO)
    setup.check_host(entry, machine="x86_64", cpuinfo=cpuinfo, loader=_loader({"libblas.so.3", "liblapack.so.3"}))
    setup.check_host(setup.load_manifest()["pyscf"], machine="aarch64", cpuinfo=tmp_path / "none",
                     loader=_loader(set()))   # no host_requirements -> nothing to check


def test_check_host_lists_every_problem(tmp_path):
    entry = setup.load_manifest()["s4"]
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("flags\t: fpu sse2 avx\n")
    with pytest.raises(setup.SetupError) as exc:
        setup.check_host(entry, machine="aarch64", cpuinfo=cpuinfo, loader=_loader({"libblas.so.3"}))
    message = str(exc.value)
    assert "aarch64" in message and "avx2" in message and "fma" in message
    assert "cannot load liblapack.so.3" in message and "cannot load libblas.so.3" not in message
    assert "libblas3 liblapack3" in message          # the reason tells the admin what to install
    with pytest.raises(setup.SetupError, match="CPU lacks"):
        setup.check_host(entry, machine="x86_64", cpuinfo=tmp_path / "missing",
                         loader=_loader({"libblas.so.3", "liblapack.so.3"}))


def test_setup_checks_host_before_cloning(tmp_path, monkeypatch):
    def refuse(entry, **kwargs):
        raise setup.SetupError("host requirements not met")

    monkeypatch.setattr(setup, "check_host", refuse)
    monkeypatch.setattr(setup, "ensure_checkout", lambda *a: pytest.fail("cloned despite a failed host check"))
    assert setup.main(["s4", "--root", str(tmp_path)]) == 1
    assert not (tmp_path / "s4").exists()


@pytest.mark.parametrize("value,match", [
    ({"machine": "x86_64"}, "list"),
    ({"os": ["linux"]}, "may only contain"),
    ({}, "may only contain"),
    ({"cpu_flags": [""]}, "non-empty"),
    ({"machine": ["x86_64"], "reason": 3}, "reason"),
])
def test_manifest_rejects_bad_host_requirements(tmp_path, value, match):
    document = json.loads((BUNDLE / "manifest.json").read_text())
    next(e for e in document["servers"] if e["id"] == "s4")["host_requirements"] = value
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document))
    with pytest.raises(setup.SetupError, match=match):
        setup.load_manifest(path)


# --- smoke_s4: independent TMM / 1D RCWA references and checks ----------------

s4 = _load("smoke_s4")


def test_tmm_matches_closed_forms():
    n = 3.47
    r, t = s4.tmm("TE", [1.0, n], [], 0.5, 0.0)
    assert r == pytest.approx(((n - 1) / (n + 1)) ** 2, abs=1e-14) and r + t == pytest.approx(1.0, abs=1e-14)
    brewster = math.degrees(math.atan(1.5))
    assert s4.tmm("TM", [1.0, 1.5], [], 1.0, brewster)[0] == pytest.approx(0.0, abs=1e-14)
    assert s4.tmm("TE", [1.0, 1.5], [], 1.0, brewster)[0] > 0.1
    # quarter-wave antireflection coating n1 = sqrt(n0 ns): zero reflectance at the design wavelength
    n1 = math.sqrt(1.5)
    assert s4.tmm("TE", [1.0, n1, 1.5], [1.0 / (4 * n1)], 1.0, 0.0)[0] == pytest.approx(0.0, abs=1e-14)
    # quarter-wave mirror air | (L H)^N | ns: each layer maps the admittance Y -> n^2 / Y,
    # so Y = (nL / nH)^(2N) ns and R = ((1 - Y) / (1 + Y))^2 at the design wavelength
    nh, nl, ns, pairs = 2.3, 1.45, 1.5, 4
    indices = [1.0] + [nl, nh] * pairs + [ns]
    thick = [1.55 / (4 * x) for x in indices[1:-1]]
    y = (nl / nh) ** (2 * pairs) * ns
    assert s4.tmm("TE", indices, thick, 1.55, 0.0)[0] == pytest.approx(((1 - y) / (1 + y)) ** 2, abs=1e-12)


def test_tmm_conserves_energy_and_absorbs_with_positive_k():
    for pol in ("TE", "TM"):
        r, t = s4.tmm(pol, [1.0, 2.3, 1.45, 1.5], [0.17, 0.27], 1.3, 50.0)
        assert r + t == pytest.approx(1.0, abs=1e-13)
        r, t = s4.tmm(pol, [1.0, complex(2.0, 0.2), 1.5], [0.05], 0.8, 30.0)
        assert 0.0 < 1.0 - r - t < 1.0
    with pytest.raises(ValueError):
        s4.tmm("TE", [1.0, 1.5], [0.1], 1.0, 0.0)


def test_refractive_index_accepts_both_notations():
    assert s4.refractive_index({"n": 2.0, "k": 0.2}) == pytest.approx(
        s4.refractive_index({"eps_real": 3.96, "eps_imag": 0.8}), abs=1e-14)
    assert s4.refractive_index({"eps_real": 2.25}) == 1.5


def test_rcwa_uniform_limit_equals_tmm():
    for pol in ("TE", "TM"):
        for theta in (0.0, 40.0):
            layers = [(0.25, 2.3 ** 2, 2.3 ** 2, 0.1, 0.0), (0.3, 1.45 ** 2 + 0.1j, 1.45 ** 2 + 0.1j, 0.2, 0.0)]
            got = s4.rcwa_1d(pol, 1.3, theta, 0.6, 1.0, 1.5, layers, orders=5)
            ref = s4.tmm(pol, [1.0, 2.3, (1.45 ** 2 + 0.1j) ** 0.5, 1.5], [0.25, 0.3], 1.3, theta)
            assert got == pytest.approx(ref, abs=1e-12)


def test_rcwa_grating_conserves_energy_and_rules_converge():
    layer = [(0.3, 1.0, 4.0, 0.2, 0.0)]
    for pol in ("TE", "TM"):
        r, t = s4.rcwa_1d(pol, 0.7, 10.0, 0.8, 1.0, 1.5, layer, orders=15)    # diffracting
        assert r + t == pytest.approx(1.0, abs=1e-10)
    assert s4.rcwa_1d("TE", 1.0, 0.0, 0.8, 1.0, 1.5, layer, 8, "li") == \
        s4.rcwa_1d("TE", 1.0, 0.0, 0.8, 1.0, 1.5, layer, 8, "laurent")
    li = s4.rcwa_1d("TM", 1.0, 0.0, 0.8, 1.0, 1.5, layer, 40, "li")
    few = s4.rcwa_1d("TM", 1.0, 0.0, 0.8, 1.0, 1.5, layer, 4, "laurent")
    many = s4.rcwa_1d("TM", 1.0, 0.0, 0.8, 1.0, 1.5, layer, 80, "laurent")
    assert abs(many[0] - li[0]) < abs(few[0] - li[0]) and abs(many[0] - li[0]) < 2e-3
    with pytest.raises(ValueError):
        s4.rcwa_1d("TM", 1.0, 0.0, 0.8, 1.0, 1.5, layer, 4, "lalanne")


@pytest.mark.parametrize("harmonics,orders", [(1, 0), (5, 1), (9, 1), (13, 2), (21, 2), (49, 4), (81, 5),
                                              (11, None), (51, None), (2, None)])
def test_square_shell_orders(harmonics, orders):
    assert s4.square_shell_orders(harmonics) == orders


def test_smoke_grids_avoid_rayleigh_anomalies():
    indices = (1.0, 1.5)
    grids = [(theta, lams) for _, theta, _, lams in s4.GRATING_CASES]
    grids += [(s4.METAL_CASE[1], s4.METAL_CASE[3]), (0.0, s4.OBLIQUE_RANGE), (s4.GEOMETRY_THETA, s4.OBLIQUE_RANGE)]
    for theta, lams in grids:
        anomalies = s4.rayleigh_wavelengths(s4.GRATING_PERIOD, theta, indices)
        for lam in s4.wavelength_grid({"wavelength_start": lams[0], "wavelength_stop": lams[1],
                                       "wavelength_points": 4}):
            assert min(abs(lam - a) for a in anomalies) >= s4.ANOMALY_CLEARANCE, (theta, lam)
    assert 0.8 in [round(x, 12) for x in s4.rayleigh_wavelengths(0.8, 0.0, indices)]
    for _, _, harmonics, _ in s4.GRATING_CASES:
        assert s4.square_shell_orders(harmonics) is not None


def test_spectrum_helpers():
    arguments = {"wavelength_start": 1.0, "wavelength_stop": 2.0, "wavelength_points": 3}
    good = {"wavelength": [1.0, 1.5, 2.0], "R": [0.1] * 3, "T": [0.9] * 3, "A": [0.0] * 3}
    assert s4.spectrum_problems(good, arguments) == []
    assert s4.spectrum_error(good, [(0.1, 0.9)] * 3) == pytest.approx(0.0, abs=1e-15)
    assert s4.spectrum_error(good, [(0.1, 0.8)] * 3) == pytest.approx(0.1)
    assert s4.spectrum_problems({**good, "wavelength": [1.0, 1.4, 2.0]}, arguments)
    assert s4.spectrum_problems({**good, "R": [0.1, float("nan"), 0.1]}, arguments)
    assert s4.spectrum_problems([1, 2], arguments)
    assert s4.unphysical({"R": [0.0], "T": [1.1], "A": [-0.1]}) == ["T in [1.1, 1.1]", "A in [-0.1, -0.1]"]


def _s4_simulator(*, swap_polarisation=False, rule="laurent"):
    """Fake simulate_stack_spectrum that answers from the smoke's own references."""
    def answer(args):
        args = dict(args)
        if swap_polarisation:
            args["polarization"] = {"TE": "TM", "TM": "TE"}[args.get("polarization", "TE")]
        if any(layer.get("pattern") for layer in args["layers"]):
            ref = s4.grating_reference(args, s4.square_shell_orders(args["n_harmonics"]), rule)
        else:
            ref = s4.stack_reference(args)
        spectrum = {"wavelength": s4.wavelength_grid(args), "R": [r for r, _ in ref], "T": [t for _, t in ref],
                    "A": [1 - r - t for r, t in ref]}
        return _json(spectrum)
    return answer


def test_s4_checks_pass_on_correct_spectra_and_catch_polarisation_or_formulation_errors():
    def run(check, **kwargs):
        report = s4.Report()
        check(s4.Caller(_StubClient({"simulate_stack_spectrum": _s4_simulator(**kwargs)}), report), report)
        return report

    assert not run(s4.check_mirror).failed
    assert not run(s4.check_absorber).failed
    assert not run(s4.check_gratings).failed
    swapped = _statuses(run(s4.check_mirror, swap_polarisation=True))
    assert swapped["simulate_stack_spectrum[quarter-wave mirror TM 30 deg vs TMM]"] == "FAIL"
    assert swapped["simulate_stack_spectrum[quarter-wave mirror TM 0 deg vs TMM]"] == "PASS"   # s = p at normal
    li = _statuses(run(s4.check_gratings, rule="li"))
    assert li["simulate_stack_spectrum[grating TM 0 deg, 49 harmonics = orders +-4 vs 1D RCWA]"] == "FAIL"
    assert li["simulate_stack_spectrum[grating TE 0 deg, 49 harmonics = orders +-4 vs 1D RCWA]"] == "PASS"


def test_s4_defect_probe_classification():
    report = s4.Report()
    call = s4.Caller(_StubClient({"simulate_stack_spectrum": _text("rejected", is_error=True)}), report)
    s4._probe(call, report, "x", {}, lambda s: "accepted")
    accepted = {"wavelength": [1.0], "R": [0.0], "T": [1.1], "A": [-0.1]}
    call = s4.Caller(_StubClient({"simulate_stack_spectrum": _json(accepted)}), report)
    s4._probe(call, report, "y", {}, lambda s: f"accepted {s4.unphysical(s)}")
    assert _statuses(report) == {"simulate_stack_spectrum[x]": "PASS", "simulate_stack_spectrum[y]": "WARN"}


def test_s4_sanity_check_uses_exact_fresnel():
    exact = ((3.47 - 1) / 4.47) ** 2
    def answer(r):
        return _json({"R": r, "T": 1 - r, "A": 0.0, "expected_R": 0.3055, "ok": True})
    for value, status in ((exact, "PASS"), (0.3055, "FAIL")):
        report = s4.Report()
        s4.check_sanity(s4.Caller(_StubClient({"check_engine_sanity": answer(value)}), report), report)
        assert set(_statuses(report).values()) == {status}


def test_s4_server_env_is_minimal_and_applies_config_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("PYTHONPATH", "/somewhere/else")
    env = s4.server_env({"command": "/m/s4/.venv/bin/python", "env": {"PYTHONPATH": "/m/s4/src"}},
                        tmp_path / "home", tmp_path / "tmp")
    assert env["PYTHONPATH"] == "/m/s4/src" and env["PATH"].startswith("/m/s4/.venv/bin:")
    assert "OPENAI_API_KEY" not in env


# --- psi4: conda-explicit install mode ---------------------------------------

def _psi4_manifest(tmp_path, mutate):
    document = json.loads((BUNDLE / "manifest.json").read_text())
    mutate(next(e for e in document["servers"] if e["id"] == "psi4"))
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document))
    return path


def test_psi4_config_runs_module_from_checkout(tmp_path):
    entry = setup.load_manifest()["psi4"]
    dest = tmp_path / "psi4"
    path = tmp_path / "psi4.mcp.json"
    path.write_text(json.dumps(setup.render_config(entry, dest)))
    server = load_mcp_config(path)["psi4"]
    assert server["command"] == str(dest / ".venv/bin/python")
    assert server["args"] == ["-m", "chemaster.mcp.calc_psi4.server"]
    assert server["env"] == {"OMP_NUM_THREADS": "1", "PYTHONPATH": str(dest)}   # bare {checkout} allowed
    assert entry["install"] == "conda-explicit" and "requirements" not in entry


@pytest.mark.parametrize("plat", sorted(setup.load_manifest()["psi4"]["conda"]["locks"]))
def test_committed_conda_locks_match_manifest(plat):
    entry = setup.load_manifest()["psi4"]
    lock = setup.read_conda_lock(entry, plat)
    urls = [line for line in lock.read_text().splitlines() if line.startswith("https://")]
    names = {u.rsplit("/", 1)[1].rsplit("-", 2)[0] for u in urls}
    assert {"python", "psi4", "dftd3-python", "mcp", "pint", "scipy"} <= names
    for spec in entry["conda"]["specs"]:
        name, version = spec.split("=")
        assert any(f"/{name}-{version}-" in u for u in urls), spec
    assert len(urls) == len(set(urls))


def test_conda_platform_mapping():
    assert setup.conda_platform("Linux", "x86_64") == "linux-64"
    assert setup.conda_platform("Linux", "aarch64") == "linux-aarch64"
    assert setup.conda_platform("Darwin", "arm64") == "osx-arm64"
    with pytest.raises(setup.SetupError, match="no conda platform"):
        setup.conda_platform("Windows", "AMD64")


def _lock_bundle(tmp_path, entry, plat, lines=None, header=None):
    path = tmp_path / entry["conda"]["locks"][plat]
    path.parent.mkdir(parents=True, exist_ok=True)
    url = (f"https://conda.anaconda.org/conda-forge/{plat}/psi4-1.11-py312_1.conda#sha256:" + "a" * 64)
    body = lines if lines is not None else ["@EXPLICIT", url]
    path.write_text("\n".join([*(header if header is not None else setup.lock_header(entry, plat)), *body]) + "\n")
    return tmp_path


@pytest.mark.parametrize("kind,match", [
    ("stale_specs", "stale or foreign"),
    ("other_platform", "stale or foreign"),
    ("no_explicit", "@EXPLICIT"),
    ("md5_only", "sha256"),
    ("other_channel", "sha256"),
    ("other_subdir", "sha256"),
    ("missing_platform", "no conda lock"),
])
def test_read_conda_lock_rejects_stale_or_foreign_locks(tmp_path, kind, match):
    entry = json.loads(json.dumps(setup.load_manifest()["psi4"]))
    plat = "linux-64"
    header = lines = None
    good = f"https://conda.anaconda.org/conda-forge/{plat}/psi4-1.11-py312_1.conda"
    if kind == "stale_specs":
        header = [h.replace("psi4=1.11", "psi4=1.10") for h in setup.lock_header(entry, plat)]
    elif kind == "other_platform":
        header = setup.lock_header(entry, "linux-aarch64")
    elif kind == "no_explicit":
        lines = [good + "#sha256:" + "a" * 64]
    elif kind == "md5_only":
        lines = ["@EXPLICIT", good + "#" + "b" * 32]
    elif kind == "other_channel":
        lines = ["@EXPLICIT", good.replace("conda-forge", "psi4") + "#sha256:" + "a" * 64]
    elif kind == "other_subdir":
        lines = ["@EXPLICIT", good.replace(plat, "linux-aarch64") + "#sha256:" + "a" * 64]
    bundle = _lock_bundle(tmp_path, entry, plat, lines, header)
    if kind == "missing_platform":
        plat = "osx-arm64"
    with pytest.raises(setup.SetupError, match=match):
        setup.read_conda_lock(entry, plat, bundle)
    if kind == "stale_specs":
        _lock_bundle(tmp_path, entry, plat)
        setup.read_conda_lock(entry, plat, tmp_path)       # noarch/plat sha256 URLs with matching header


@pytest.mark.parametrize("mutate,match", [
    (lambda e: e.pop("conda"), "exactly"),
    (lambda e: e["conda"].update(extra=1), "exactly"),
    (lambda e: e["conda"].update(channel="https://x"), "channel"),
    (lambda e: e["conda"].update(specs=["psi4>=1.11", "python=3.12.14"]), "exact"),
    (lambda e: e["conda"].update(specs=["psi4=1.11"]), "pin python"),
    (lambda e: e["conda"].update(specs=["python=3.11.9", "psi4=1.11"]), "pin python"),
    (lambda e: e["conda"].update(locks={"win-64": "locks/x.txt"}), "platforms"),
    (lambda e: e["conda"].update(locks={"linux-64": "/abs/x.txt"}), "relative"),
    (lambda e: e["conda"].update(locks={"linux-64": "../x.txt"}), "relative"),
    (lambda e: e.update(requirements=["mcp==1.0"]), "only apply"),
    (lambda e: e.update(uv_sync_args=["--no-dev"]), "uv_sync_args"),
    (lambda e: e.update(install="uv-sync-frozen"), r"\[.conda.\] only apply to install conda-explicit"),
    (lambda e: e["launch"]["env"].update(PYTHONPATH="{checkout}x"), "prefix"),
])
def test_manifest_rejects_bad_conda_fields(tmp_path, mutate, match):
    with pytest.raises(setup.SetupError, match=match):
        setup.load_manifest(_psi4_manifest(tmp_path, mutate))


def test_conda_install_creates_fresh_prefix_from_lock(tmp_path, monkeypatch):
    entry = setup.load_manifest()["psi4"]
    dest = tmp_path / "root" / "psi4"
    old = dest / ".venv"
    (old / "conda-meta").mkdir(parents=True)
    (old / "stale").write_text("x")
    commands = []

    def fake_run(cmd, cwd=None, env=None):
        commands.append((cmd, env))
        if cmd[1:2] == ["create"]:
            assert not old.exists(), "previous prefix must be removed first"
            assert env["MAMBA_ROOT_PREFIX"] == str(tmp_path / "root" / ".micromamba")
            assert "CONDA_PREFIX" not in env and "PYTHONPATH" not in env
        return entry["python"] if cmd[-1].startswith("import sys") else ""

    monkeypatch.setenv("CONDA_PREFIX", "/opt/conda")
    monkeypatch.setenv("MAMBA_ROOT_PREFIX", "/elsewhere")
    monkeypatch.setattr(setup, "run", fake_run)
    monkeypatch.setattr(setup.shutil, "which", lambda name: f"/usr/local/bin/{name}" if name == "micromamba" else None)
    monkeypatch.setattr(setup, "conda_platform", lambda *a: "linux-aarch64")
    python = setup.build_env(entry, dest)
    assert python == dest / ".venv/bin/python"
    create = commands[0][0]
    assert create == ["/usr/local/bin/micromamba", "create", "--yes", "--no-rc", "--prefix", str(dest / ".venv"),
                      "--file", str(BUNDLE / "locks/psi4-linux-aarch64.txt")]
    assert not any("uv" in c[0][0] for c in commands)       # uv is not needed for conda-explicit


def test_conda_install_refuses_foreign_prefix_and_missing_micromamba(tmp_path, monkeypatch):
    entry = setup.load_manifest()["psi4"]
    monkeypatch.setattr(setup, "conda_platform", lambda *a: "linux-64")
    monkeypatch.setattr(setup.shutil, "which", lambda name: None)
    with pytest.raises(setup.SetupError, match="micromamba not found"):
        setup.build_env(entry, tmp_path)
    monkeypatch.setattr(setup.shutil, "which", lambda name: "/bin/micromamba")
    (tmp_path / ".venv").mkdir()                  # e.g. a uv venv: never delete it
    with pytest.raises(setup.SetupError, match="not a conda prefix"):
        setup.build_env(entry, tmp_path)
    assert (tmp_path / ".venv").is_dir()
    monkeypatch.setattr(setup, "conda_platform", lambda *a: "osx-arm64")
    with pytest.raises(setup.SetupError, match="no conda lock for platform osx-arm64"):
        setup.build_env(entry, tmp_path)


def test_write_conda_locks_from_dry_run_json(tmp_path, monkeypatch):
    entry = setup.load_manifest()["psi4"]
    seen = []

    def fake_run(cmd, cwd=None, env=None):
        if cmd[-1] == "--version":
            return "2.9.0"
        seen.append((cmd, env))
        plat = cmd[cmd.index("--platform") + 1]
        pkgs = [{"url": f"https://conda.anaconda.org/conda-forge/{sub}/{n}.conda", "sha256": c * 64}
                for n, sub, c in (("zlib-1.3-h0_0", plat, "1"), ("mcp-1.28.1-pyhd8ed1ab_0", "noarch", "2"))]
        return json.dumps({"actions": {"LINK": pkgs}})

    monkeypatch.setattr(setup, "run", fake_run)
    monkeypatch.setattr(setup.shutil, "which", lambda name: "/bin/micromamba")
    written = setup.write_conda_locks(entry, tmp_path / "root", bundle=tmp_path)
    assert sorted(p.name for p in written) == ["psi4-linux-64.txt", "psi4-linux-aarch64.txt"]
    for cmd, env in seen:
        assert {"--dry-run", "--json", "--override-channels", "--no-rc"} <= set(cmd)
        assert cmd[-len(entry["conda"]["specs"]):] == entry["conda"]["specs"]
        assert env["CONDA_OVERRIDE_GLIBC"] == setup.LOCK_GLIBC
    text = (tmp_path / "locks/psi4-linux-64.txt").read_text()
    assert text.splitlines()[-2:] == [
        "https://conda.anaconda.org/conda-forge/linux-64/zlib-1.3-h0_0.conda#sha256:" + "1" * 64,
        "https://conda.anaconda.org/conda-forge/noarch/mcp-1.28.1-pyhd8ed1ab_0.conda#sha256:" + "2" * 64,
    ]
    assert "@EXPLICIT" in text
    assert setup.read_conda_lock(entry, "linux-64", tmp_path)


def test_lock_flag_only_applies_to_conda_servers(tmp_path, monkeypatch):
    monkeypatch.setattr(setup, "ensure_checkout", lambda *a: pytest.fail("--lock must not clone"))
    assert setup.main(["pyscf", "--lock", "--root", str(tmp_path)]) == 1


# --- smoke_psi4: helpers and psi4-free checks --------------------------------

p4 = _load("smoke_psi4")


def test_psi4_parse_geometry_accepts_psi4_xyz_and_bare_lines():
    psi4_out = "0 1\n O    0.0  0.0  0.07\n H    0.0  0.75 -0.56\n H    0.0 -0.75 -0.56\n"
    xyz = "3\nwater\nO 0 0 0.07\nH 0 0.75 -0.56\nH 0 -0.75 -0.56"
    assert p4.parse_geometry(psi4_out) == p4.parse_geometry(xyz)
    assert [s for s, _ in p4.parse_geometry(p4.H2O_START)] == ["O", "H", "H"]
    with pytest.raises(ValueError):
        p4.parse_geometry("0 1\n")


def test_psi4_classify_frequencies():
    ref = [-1081.6, 1866.4, 1866.4, 4023.5, 4363.4, 4363.4]
    assert p4.classify_frequencies(ref, ref, 1)[0] == "PASS"
    dropped_i = [abs(f) for f in ref]
    status, detail = p4.classify_frequencies(dropped_i, ref, 0)
    assert status == "WARN" and "-1081.6" in detail
    assert p4.classify_frequencies(ref, ref, 0)[0] == "FAIL"                 # right values, wrong count
    assert p4.classify_frequencies([f + 5 for f in dropped_i], ref, 0)[0] == "FAIL"
    minimum = [2170.28, 4139.70, 4390.74]
    assert p4.classify_frequencies(minimum, minimum, 0)[0] == "PASS"
    assert p4.classify_frequencies(minimum[:2], minimum, 0)[0] == "FAIL"


def test_psi4_classify_alternative_and_value_of():
    assert p4.classify_alternative(1.0, 1.0, 2.0, 1e-6) == "PASS"
    assert p4.classify_alternative(2.0, 1.0, 2.0, 1e-6) == "WARN"
    assert p4.classify_alternative(3.0, 1.0, 2.0, 1e-6) == "FAIL"
    assert p4.classify_alternative(None, 1.0, 2.0, 1e-6) == "FAIL"
    assert p4.value_of({"value": -1.5, "unit": "Hartree"}) == -1.5
    assert p4.value_of(0.3) == 0.3 and p4.value_of(None) is None and p4.value_of(True) is None


def test_psi4_compare_states():
    server = [{"excitation_energy": {"value": 7.0001, "unit": "eV"}, "oscillator_strength": 0.0157},
              {"excitation_energy": {"value": 9.2, "unit": "eV"}, "oscillator_strength": 0.0}]
    de, df = p4.compare_states(server, [(7.0, 0.0157), (9.2, 0.0)])
    assert de == pytest.approx(1e-4) and df == 0.0
    assert p4.compare_states([{"excitation_energy": None, "oscillator_strength": 0.1}], [(1.0, 0.1)])[0] == math.inf


def _p4json(obj):
    return {"result": {"content": [{"type": "text", "text": json.dumps(obj)}], "structuredContent": obj,
                       "isError": False}}


class _FakeRef:
    """Psi4Ref stand-in with fixed reference numbers."""

    def __init__(self, ground_minimum=-75.3231):
        self.ground_minimum = ground_minimum

    def excited_total(self, atoms, *args):
        return -75.3231, -74.9473

    def excited_gradient_norm(self, *args):
        return 1e-4

    def optimize(self, *args, **kwargs):
        return self.ground_minimum, p4.parse_geometry(_WATER)


_WATER = "O 0 0 0.076\nH 0 0.771 -0.603\nH 0 -0.771 -0.603"


@pytest.mark.parametrize("energy,exc,ground_min,expected", [
    (-74.9473, 10.2264, -75.3231, ("PASS", None)),                        # a real S1 minimum
    (-75.3231, 11.0565, -75.3231, ("WARN", "WARN")),                      # upstream: ground-state minimum
    (-75.3231, 10.2264, -75.0, ("FAIL", None)),                           # E(S0) but not the S0 minimum
    (-70.0, 10.2264, -75.3231, ("FAIL", None)),
])
def test_psi4_excited_state_opt_classification(energy, exc, ground_min, expected):
    result = {"ok": True, "result": {"final_total_energy": {"value": energy, "unit": "Hartree"},
                                     "excitation_energy_at_opt": {"value": exc, "unit": "eV"},
                                     "optimized_geometry_xyz": "0 1\n" + _WATER, "converged": True}}
    report = p4.Report()
    call = p4.Caller(_StubClient({"optimize_excited_state": _p4json(result)}), report)
    p4.check_excited_state_opt(call, report, _FakeRef(ground_min), p4.parse_geometry(_WATER))
    statuses = _statuses(report)
    main = next(v for k, v in statuses.items() if not k.endswith("[excitation_energy_at_opt]"))
    side = next((v for k, v in statuses.items() if k.endswith("[excitation_energy_at_opt]")), None)
    assert (main, side) == expected
    assert call.client.calls[0][1]["memory_gb"] == 1


def test_psi4_tool_payload_fails_on_in_band_error_for_valid_requests():
    report = p4.Report()
    bad = {"ok": False, "error_code": "PSI4_INTERNAL_ERROR", "details": "No module named 'psi4'"}
    call = p4.Caller(_StubClient({"single_point": _p4json(bad)}), report)
    assert p4.tool_payload(call, report, "sp", "single_point", {}) is None
    assert _statuses(report) == {"sp": "FAIL"} and "psi4" in report.checks[0]["detail"]


def test_psi4_server_env_is_minimal_and_applies_config_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    env = p4.server_env({"command": "/m/psi4/.venv/bin/python", "env": {"PYTHONPATH": "/m/psi4"}},
                        tmp_path / "home", tmp_path / "tmp")
    assert env["PYTHONPATH"] == "/m/psi4" and env["TMPDIR"] == str(tmp_path / "tmp")
    assert "ANTHROPIC_API_KEY" not in env


# --- verify_run: strict e2e_check.json spec validation ------------------------

@pytest.mark.parametrize("spec_path", _SPEC_FILES, ids=lambda p: p.parent.name)
def test_all_e2e_check_specs_parse_without_error(spec_path):
    spec = verify.parse_spec(json.loads(spec_path.read_text()), str(spec_path))
    assert spec.server and spec.calls and spec.answers
    assert all(src.call in {cs.name for cs in spec.calls} for a in spec.answers for src in a.sources)


def test_valid_spec_fixture_parses():
    assert verify.parse_spec(_valid_spec()).server == "demo"


@pytest.mark.parametrize("mutate,match", [
    (lambda s: s.update(extra_top=1), "top-level"),
    (lambda s: s["calls"][0].update(toool="x"), r"calls\[0\]"),
    (lambda s: s["calls"][0]["result"].update(abs_toll=1e-6), "result"),
    (lambda s: s["answers"][0].update(prediciton_key="y"), r"answers\[0\]"),
])
def test_unknown_spec_keys_are_rejected(mutate, match):
    spec = _valid_spec()
    mutate(spec)
    with pytest.raises(verify.SpecError, match=match):
        verify.parse_spec(spec)


def test_unknown_keys_in_chain_source_and_select_are_rejected():
    chain = _valid_spec()
    chain["calls"].append({"name": "d", "tool": "t2",
                           "inputs_from_call": {"call": "c", "map": {"a": "b"}, "typo": 1},
                           "result": {"format": "number", "reference_key": "z", "abs_tol": 0}})
    with pytest.raises(verify.SpecError, match="inputs_from_call"):
        verify.parse_spec(chain)

    src = _valid_spec()
    src["answers"][0] = {"prediction_key": "y", "reference_key": "y_ref", "abs_tol": 0,
                         "from_calls": [{"call": "c", "result_key": "r", "selct": {}}]}
    with pytest.raises(verify.SpecError, match="from_calls"):
        verify.parse_spec(src)

    sel = _valid_spec()
    sel["answers"][0] = {"prediction_key": "y", "reference_key": "y_ref", "abs_tol": 0,
                         "from_calls": [{"call": "c", "select": {"reduce": "max", "bogus": 1}}]}
    with pytest.raises(verify.SpecError, match="select"):
        verify.parse_spec(sel)


@pytest.mark.parametrize("mutate,match", [
    (lambda s: s["calls"][0]["result"].update(format="csv"), "format"),
    (lambda s: s["calls"][0]["result"].update(match="nope"), "match"),
    (lambda s: s["calls"][0]["result"].update(format="json", key=None, extract="unknown_x"), "extract"),
    (lambda s: s["calls"][0]["result"].update(format="json", key=None, extract=None), "requires"),
])
def test_invalid_result_enums_and_requirements(mutate, match):
    spec = _valid_spec()
    mutate(spec)
    with pytest.raises(verify.SpecError, match=match):
        verify.parse_spec(spec)


def test_invalid_chain_compare_enum_is_rejected():
    spec = _valid_spec()
    spec["calls"].append({"name": "d", "tool": "t2",
                          "inputs_from_call": {"call": "c", "map": {"a": "b"}, "compare": "fuzzy"},
                          "result": {"format": "number", "reference_key": "z", "abs_tol": 0}})
    with pytest.raises(verify.SpecError, match="compare"):
        verify.parse_spec(spec)


@pytest.mark.parametrize("mutate,match", [
    (lambda s: s["answers"][0].update(from_call="ghost"), "from_call"),
    (lambda s: s["calls"].append({"name": "d", "tool": "t", "inputs_from_call": {"call": "ghost", "map": {"a": "b"}},
                                  "result": {"format": "number", "reference_key": "z", "abs_tol": 0}}),
     "inputs_from_call"),
    (lambda s: (s["calls"].append({"name": "opt", "tool": "o", "optional": True,
                                   "result": {"format": "number", "reference_key": "z", "abs_tol": 0}}),
                s["calls"].append({"name": "d", "tool": "t", "inputs_from_call": {"call": "opt", "map": {"a": "b"}},
                                   "result": {"format": "number", "reference_key": "z", "abs_tol": 0}})),
     "optional"),
])
def test_dangling_and_optional_cross_references_are_rejected(mutate, match):
    spec = _valid_spec()
    mutate(spec)
    with pytest.raises(verify.SpecError, match=match):
        verify.parse_spec(spec)


def test_invalid_bypass_regex_is_rejected():
    spec = _valid_spec()
    spec["bypass_patterns"] = ["[invalid"]
    with pytest.raises(verify.SpecError, match="regex"):
        verify.parse_spec(spec)
    spec = _valid_spec()
    spec["suspicious_patterns"] = ["(unclosed"]
    with pytest.raises(verify.SpecError, match="regex"):
        verify.parse_spec(spec)


def test_schema_1_spec_parses_through_normalisation():
    raw = json.loads((E2E_TASKS / "mcp_e2e/pyscf_rhf_energy/e2e_check.json").read_text())
    assert raw["schema_version"] == 1
    spec = verify.parse_spec(raw)
    assert [c.tool for c in spec.calls] == ["pyscf_rhf_energy"]
    assert [src.call for src in spec.answers[0].sources] == ["pyscf_rhf_energy"]


@pytest.mark.parametrize("mutate,match", [
    (lambda s: s["calls"][0]["result"].update(key="y"), "do not apply to format 'number'"),
    (lambda s: s["calls"][0]["result"].update(format="image"), "do not apply to format 'image'"),
    (lambda s: s["calls"][0]["result"].update(format="json", key="v", select={"reduce": "max", "argmax_of": "w"}),
     "exactly one of"),
    (lambda s: s["calls"][0].update(group="g"), "only applies to optional calls"),
    (lambda s: s["calls"].append(dict(s["calls"][0])), "duplicate call name"),
    (lambda s: s["calls"].append({"name": "d", "tool": "t", "inputs_from_call": {"call": "c", "map": {"a": "b"},
                                                                                 "abs_tol": 1e-3}}),
     "abs_tol only applies to compare 'geometry'"),
    (lambda s: s["calls"].append({"name": "d", "tool": "t", "inputs_from_call": {"call": "c",
                                                                                 "extract": "arxiv_ids"}}),
     "needs 'args'"),
    (lambda s: s["answers"][0].update(from_calls=[{"call": "c"}]), "exactly one of from_call / from_calls"),
    (lambda s: s["answers"][0].update(match="member"), "only applies to extracted answers"),
    (lambda s: s["answers"][0].update(abs_tol=-1), "non-negative"),
    (lambda s: s.update(bypass_tools={"WebFetch": "(bad"}), "regex"),
])
def test_spec_rejects_keys_that_would_be_ignored(mutate, match):
    spec = _valid_spec()
    mutate(spec)
    with pytest.raises(verify.SpecError, match=re.escape(match)):
        verify.parse_spec(spec)


def test_select_narrows_a_call_result_like_an_answer_source():
    raw = _valid_spec()
    raw["calls"][0]["result"] = {"format": "json", "key": "R", "select": {"argmax_of": "wavelength"},
                                 "reference_key": "y_ref", "abs_tol": 1e-9}
    cs = verify.parse_spec(raw).calls[0]
    call = verify.evidence.ToolCall(result_text=json.dumps({"wavelength": [1.0, 1.2, 1.1], "R": [0.1, 0.4, 0.2]}),
                                    is_error=False, input={"x": 3})
    assert verify.checks.judge_call(call, cs, {"y_ref": 0.4, "x_ref": 3})["result_ok"] is True
    assert verify.checks.judge_call(call, cs, {"y_ref": 0.2, "x_ref": 3})["result_ok"] is False


# --- setup.py: installer registry and strict manifest keys ----------------------

def _manifest_with(tmp_path, sid, mutate):
    document = json.loads((BUNDLE / "manifest.json").read_text())
    mutate(next(e for e in document["servers"] if e["id"] == sid))
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document))
    return path


@pytest.mark.parametrize("sid,mutate,match", [
    ("s4", lambda e: e.update(host_requirement=e.pop("host_requirements")), r"unknown manifest key\(s\) \['host_requirement'\]"),
    ("pyscf", lambda e: e.update(uv_sync_arg=["--no-dev"]), "unknown manifest key"),
    ("pyscf", lambda e: e.update(revision="z" * 40), "40-char commit SHA"),
    ("pyscf", lambda e: e.update(python="3"), "3.x version"),
    ("pyscf", lambda e: e.update(expected_tools=["b", "a"]), "sorted"),
    ("pyscf", lambda e: e.pop("smoke"), "smoke"),
    ("pyscf", lambda e: e["launch"].update(cwd="."), "launch must have command"),
    ("jsbsim", lambda e: e.update(conda={}), r"\['conda'\] only apply to install conda-explicit, not uv-pip-pinned"),
])
def test_manifest_rejects_unknown_and_malformed_keys(tmp_path, sid, mutate, match):
    with pytest.raises(setup.SetupError, match=match):
        setup.load_manifest(_manifest_with(tmp_path, sid, mutate))


def test_manifest_document_keys_are_exact(tmp_path):
    document = json.loads((BUNDLE / "manifest.json").read_text())
    document["server"] = []
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document))
    with pytest.raises(setup.SetupError, match="exactly schema_version"):
        setup.load_manifest(path)


def test_installer_fields_are_disjoint_and_cover_the_manifest():
    owners = [key for installer in setup.INSTALLERS.values() for key in installer.fields]
    assert len(owners) == len(set(owners)) and not set(owners) & set(setup.COMMON_KEYS)
    assert setup.INSTALL_MODES == ("uv-sync-frozen", "uv-pip-pinned", "conda-explicit")
    assert {e["install"] for e in setup.load_manifest().values()} == set(setup.INSTALL_MODES)
    assert setup.HOST_REQUIREMENT_KEYS == ("machine", "cpu_flags", "shared_libraries")


def test_a_new_install_mode_is_one_registered_installer(tmp_path, monkeypatch):
    def check_pins(sid, value, entry):
        if value != ["x==1"]:
            raise setup.SetupError(f"{sid}: demo_pins")

    class Demo(setup.Installer):
        mode = "demo"
        fields = {"demo_pins": check_pins}

        def install(self, entry, dest):
            return dest / ".venv/bin/python"

    monkeypatch.setitem(setup.INSTALLERS, "demo", Demo())
    path = _manifest_with(tmp_path, "pyscf", lambda e: e.update(install="demo", demo_pins=["x==1"]))
    entry = setup.load_manifest(path)["pyscf"]
    monkeypatch.setattr(setup, "run", lambda cmd, cwd=None, env=None: entry["python"])
    assert setup.build_env(entry, tmp_path) == tmp_path / ".venv/bin/python"
    with pytest.raises(setup.SetupError, match="demo_pins"):
        setup.load_manifest(_manifest_with(tmp_path, "pyscf", lambda e: e.update(install="demo", demo_pins=[])))
    with pytest.raises(setup.SetupError, match=r"\['demo_pins'\] only apply to install demo"):
        setup.load_manifest(_manifest_with(tmp_path, "pyscf", lambda e: e.update(demo_pins=["x==1"])))
    assert setup.main(["pyscf", "--lock", "--root", str(tmp_path)]) == 1   # no lock() for uv-sync-frozen
