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


def test_smoke_covers_every_manifest_tool():
    source = (BUNDLE / "smoke_pyscf.py").read_text()
    for tool in setup.load_manifest()["pyscf"]["expected_tools"]:
        assert f'"{tool}"' in source, f"smoke_pyscf.py never calls {tool}"
