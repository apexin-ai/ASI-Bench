"""Offline regression checks for the Gmsh integration contract."""
import json

import pytest

from .support import runner, setup, smoke_module, StubClient, rpc_text


def test_gmsh_manifest_and_smoke_match():
    entry = setup.load_manifest()["gmsh"]
    assert entry["install"] == "uv-pip-pinned"
    assert entry["revision"] == "8d0caf2f240f2c1d8ae6a3fcd3c997d835c7f629"
    assert entry["bundle"]["destination"] == "gmsh_server.py"
    assert smoke_module("gmsh").SMOKE.server == "gmsh"


def test_uv_pip_bundle_is_validated_and_copied(tmp_path, monkeypatch):
    entry = setup.load_manifest()["gmsh"]
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    commands = []
    monkeypatch.setattr(setup, "_uv", lambda: None)
    monkeypatch.setattr(setup, "run", lambda cmd, **kwargs: commands.append(cmd))
    setup.INSTALLERS["uv-pip-pinned"].install(entry, checkout)
    assert (checkout / "gmsh_server.py").read_text() == (setup.HERE / "gmsh_server.py").read_text()
    assert commands[0][:4] == ["uv", "venv", "--clear", "--python"]
    assert commands[1][:4] == ["uv", "pip", "install", "--python"]


def test_gmsh_smoke_rejects_in_band_business_error():
    module = smoke_module("gmsh")
    report = runner.Report()
    client = StubClient({name: rpc_text(json.dumps({"error": "gmsh missing"}))
                         for name in module.SMOKE.__dict__.get("tools", [])})
    client.responses.update({"get_info": rpc_text(json.dumps({"error": "gmsh missing"})),
                             "mesh_box": rpc_text(json.dumps({"error": "gmsh missing"}), is_error=True),
                             "inspect_mesh": rpc_text(json.dumps({"error": "gmsh missing"}), is_error=True)})
    session = type("Session", (), {
        "call": runner.Caller(client, report), "report": report,
        "cwd": setup.HERE, "args": None, "smoke": module.SMOKE,
    })()
    module.run_l1(session)
    assert report.failed
