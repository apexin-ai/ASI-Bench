"""e2e_smoke/servers/pyscf.py without PySCF: parsers, distance/PNG helpers, in-band errors, plot and visualize checks."""

import pytest

from . import support
from .support import StubClient, png_bytes, report_statuses, rpc_image, rpc_text

smoke = support.smoke_module("pyscf")


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


def test_png_size():
    assert smoke.png_size(png_bytes(1000, 600)) == (1000, 600)
    with pytest.raises(ValueError):
        smoke.png_size(b"GIF89a" + b"\0" * 20)


@pytest.mark.parametrize("response,expected", [
    (rpc_text("boom", is_error=True), "PASS"),
    (rpc_text("Error: Invalid SMILES string"), "WARN"),
    (rpc_text("C 0 0 0"), "FAIL"),
])
def test_in_band_error_classification(response, expected):
    report = smoke.Report()
    call = smoke.Caller(StubClient({"t": response}), report)
    smoke.check_in_band_error(call, report, "t", {}, "error")
    assert report_statuses(report) == {"t[invalid input]": expected}


def test_caller_fails_on_tool_error_and_attributes_stdout():
    report = smoke.Report()
    call = smoke.Caller(StubClient({"t": rpc_text("kaput", is_error=True)}, chatter=2), report)
    assert call("check", "t", {}) is None
    assert report_statuses(report) == {"check": "FAIL"}
    assert call.stdout_by_tool == {"t": 2}


def test_check_plot_accepts_png_and_rejects_other_payloads():
    import base64
    good = base64.b64encode(png_bytes(1000, 600)).decode()

    def run(first):
        report = smoke.Report()
        client = StubClient({"plot_energy_scan_image_mcp":
                              lambda args: first if len(args["energies"]) == len(args["bond_lengths"])
                              else rpc_text("validation error", is_error=True)})
        smoke.check_plot(smoke.Caller(client, report), report)
        return report_statuses(report)

    assert run(rpc_image(good)) == {"plot_energy_scan_image_mcp": "PASS",
                                 "plot_energy_scan_image_mcp[mismatched lengths]": "PASS"}
    assert run(rpc_image(good, mime="image/jpeg"))["plot_energy_scan_image_mcp"] == "FAIL"
    assert run(rpc_image(base64.b64encode(b"not a png at all, sorry").decode()))["plot_energy_scan_image_mcp"] == "FAIL"
    assert run(rpc_text("no image"))["plot_energy_scan_image_mcp"] == "FAIL"


def test_check_visualize_separates_side_effect_from_returned_path(tmp_path):
    def writes_html(message):
        def respond(args):
            (tmp_path / smoke.VISUALIZE_FILE).write_text(f"<script>3Dmol</script>{args['xyz_string']}")
            return rpc_text(message)
        return respond

    report = smoke.Report()
    client = StubClient({"visualize_molecule_3d_mcp": writes_html("saved in /Users/someone/x.html")})
    smoke.check_visualize(smoke.Caller(client, report), report, tmp_path)
    assert report_statuses(report) == {"visualize_molecule_3d_mcp[HTML written]": "PASS",
                                 "visualize_molecule_3d_mcp[result locates output]": "WARN"}

    report = smoke.Report()
    client = StubClient({"visualize_molecule_3d_mcp":
                          writes_html(f"saved in {tmp_path / smoke.VISUALIZE_FILE}")})
    smoke.check_visualize(smoke.Caller(client, report), report, tmp_path)
    assert report_statuses(report)["visualize_molecule_3d_mcp[result locates output]"] == "PASS"

    report = smoke.Report()
    (tmp_path / smoke.VISUALIZE_FILE).unlink()
    client = StubClient({"visualize_molecule_3d_mcp": rpc_text("done")})
    smoke.check_visualize(smoke.Caller(client, report), report, tmp_path)
    assert report_statuses(report)["visualize_molecule_3d_mcp[HTML written]"] == "FAIL"
