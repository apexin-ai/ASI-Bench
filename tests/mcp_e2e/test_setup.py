"""scripts/mcp/e2e/setup.py: manifest validation, installer registry, host requirements, rendered
configs, checkouts, conda locks. No network, no upstream installs."""
import json
import subprocess

import pytest

from ai4sci_bench.mcp_config import load_mcp_config, load_science_mcp_catalog

from .support import BUNDLE, setup, smoke_module


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


_X86_CPUINFO = "processor\t: 0\nflags\t\t: fpu sse2 avx avx2 bmi1 bmi2 fma\n"


def _loader(available):
    def load(name):
        if name not in available:
            raise OSError(f"{name}: cannot open shared object file")
    return load


def _psi4_manifest(tmp_path, mutate):
    document = json.loads((BUNDLE / "manifest.json").read_text())
    mutate(next(e for e in document["servers"] if e["id"] == "psi4"))
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document))
    return path


def _lock_bundle(tmp_path, entry, plat, lines=None, header=None):
    path = tmp_path / entry["conda"]["locks"][plat]
    path.parent.mkdir(parents=True, exist_ok=True)
    url = (f"https://conda.anaconda.org/conda-forge/{plat}/psi4-1.11-py312_1.conda#sha256:" + "a" * 64)
    body = lines if lines is not None else ["@EXPLICIT", url]
    path.write_text("\n".join([*(header if header is not None else setup.lock_header(entry, plat)), *body]) + "\n")
    return tmp_path


def _manifest_with(tmp_path, sid, mutate):
    document = json.loads((BUNDLE / "manifest.json").read_text())
    mutate(next(e for e in document["servers"] if e["id"] == sid))
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document))
    return path


def test_manifest_is_pinned_and_catalogued():
    servers = setup.load_manifest()
    catalog = load_science_mcp_catalog()
    assert "pyscf" in servers
    for sid, entry in servers.items():
        assert len(entry["revision"]) == 40
        assert entry["catalog_id"] in catalog
        assert catalog[entry["catalog_id"]]["source"].rstrip("/") == entry["repository"].removesuffix(".git")
        assert (BUNDLE / "e2e_smoke" / "servers" / f"{sid}.py").is_file()
        assert smoke_module(sid).SMOKE.server == sid
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


def test_rdkit_config_runs_script_over_stdio_with_mcp_below_2(tmp_path):
    entry = setup.load_manifest()["rdkit"]
    dest = tmp_path / "rdkit"
    path = tmp_path / "rdkit.mcp.json"
    path.write_text(json.dumps(setup.render_config(entry, dest)))
    server = load_mcp_config(path)["rdkit"]
    assert server["command"] == str(dest / ".venv/bin/python")
    # upstream's default transport is sse; stdio must be explicit
    assert server["args"] == [str(dest / "run_server.py"), "--transport", "stdio"]
    assert "env" not in server
    pins = dict(r.split("==") for r in entry["requirements"])
    # upstream declares mcp>=1.23.0 without an upper bound; mcp 2 removed FastMCP and the server cannot start
    assert int(pins["mcp"].split(".")[0]) == 1 and pins["rdkit"] == "2025.3.1"
    assert entry["catalog_id"] == "rdkit_tandem" and len(entry["expected_tools"]) == 74


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


@pytest.mark.parametrize("sid,mutate,match", [
    ("s4", lambda e: e.update(host_requirement=e.pop("host_requirements")), r"unknown manifest key\(s\) \['host_requirement'\]"),
    ("pyscf", lambda e: e.update(uv_sync_arg=["--no-dev"]), "unknown manifest key"),
    ("pyscf", lambda e: e.update(revision="z" * 40), "40-char commit SHA"),
    ("pyscf", lambda e: e.update(python="3"), "3.x version"),
    ("pyscf", lambda e: e.update(expected_tools=["b", "a"]), "sorted"),
    ("pyscf", lambda e: e.update(smoke="smoke_pyscf.py"), r"unknown manifest key\(s\) \['smoke'\]"),
    ("pyscf", lambda e: e.update(id="py-scf"), "lowercase identifier"),
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
