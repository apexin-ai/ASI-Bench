"""Regression checks for material-server deployment and business-error handling."""
import importlib.util
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from .support import BUNDLE, rpc_text, runner, setup, smoke_module, StubClient


def _adapter(monkeypatch):
    sdk = ModuleType("mcp.server.fastmcp")
    sdk.FastMCP = lambda *a: SimpleNamespace(tool=lambda: lambda fn: fn)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", sdk)
    spec = importlib.util.spec_from_file_location("test_mattersim_adapter", BUNDLE / "mattersim_server.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("cuda,expected", [(False, "cpu"), (True, "cuda")])
def test_mattersim_resolves_auto_before_loading(monkeypatch, cuda, expected):
    adapter = _adapter(monkeypatch)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda)))
    monkeypatch.setenv("MATTERSIM_DEVICE", "auto")
    assert adapter._settings(None, None)[1] == expected
    assert adapter._settings(None, "cpu")[1] == "cpu"


def test_mattergen_deployment_includes_source_and_pyg_extensions():
    entry = setup.load_manifest()["mattergen"]
    assert "--extra=cu126" in entry["uv_sync_args"]
    assert entry["bootstrap"]["script"] == "skills/ml-generative-mattergen/scripts/setup_mattergen.py"
    assert entry["launch"]["env"]["MATTERGEN_REPO"] == "{checkout}/mattergen"


def test_pinned_bootstrap_runs_in_the_installed_environment(tmp_path, monkeypatch):
    entry = setup.load_manifest()["mattergen"]
    project = tmp_path / entry["project"]
    project.mkdir(parents=True)
    for name in ("pyproject.toml", "uv.lock"):
        (project / name).touch()
    script = tmp_path / entry["bootstrap"]["script"]
    script.parent.mkdir(parents=True)
    script.touch()
    commands = []
    monkeypatch.setattr(setup, "_uv", lambda: None)
    def fake_run(cmd, **kw):
        commands.append((cmd, kw))
        return "3.10" if "-c" in cmd else ""
    monkeypatch.setattr(setup, "run", fake_run)
    python = setup.build_env(entry, tmp_path)
    assert commands[0][0] == ["uv", "sync", "--frozen", "--no-dev", "--extra=cu126", "--python", "3.10"]
    assert commands[1][0] == [str(python), str(script), "--dir", str(tmp_path / "mattergen")]
    assert commands[1][1]["cwd"] == tmp_path


@pytest.mark.parametrize("bootstrap", [
    {"script": "../escape.py", "args": []},
    {"script": "/absolute.py", "args": []},
    {"script": "setup.py", "args": "/wrong"},
    {"script": "setup.py", "args": ["{checkout}bad"]},
    {"script": "setup.py", "args": [], "typo": True},
])
def test_bootstrap_manifest_is_validated(bootstrap):
    entry = {**setup.load_manifest()["pyscf"], "bootstrap": bootstrap}
    with pytest.raises(setup.SetupError):
        setup.validate_entry(entry)


@pytest.mark.parametrize("server", ["chgnet", "mace"])
def test_failed_model_load_must_fail_smoke(server):
    module = smoke_module(server)
    report = runner.Report()
    responses = {t: rpc_text('{"error":"Model not loaded"}', is_error=True) for t in module.TOOLS}
    responses["get_info"] = rpc_text('{"status":"no model loaded"}')
    responses["load_model"] = rpc_text("Error loading model: missing checkpoint")
    client = StubClient(responses)
    session = SimpleNamespace(call=runner.Caller(client, report), report=report,
                              args=SimpleNamespace(load_model=True, model_name="missing"),
                              smoke=module.SMOKE)
    module.run_l1(session)
    assert report.failed, report.checks


def test_mattergen_in_band_error_must_fail_smoke():
    module = smoke_module("mattergen")
    report = runner.Report()
    client = StubClient({"generate_structures": rpc_text(json.dumps({"error": "missing torch_scatter"}))})
    session = SimpleNamespace(call=runner.Caller(client, report), report=report,
                              cwd=BUNDLE, args=SimpleNamespace(generate=True, num_structures=1,
                                                              chemical_system="Li-Zr-Cl"),
                              smoke=module.SMOKE)
    module.run_l1(session)
    assert report.failed, report.checks
