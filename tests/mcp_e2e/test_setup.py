"""scripts/mcp/e2e/setup.py: manifest validation, installer registry, host requirements, rendered
configs, checkouts, conda locks. No network, no upstream installs."""
import json
import subprocess

import pytest

from ai4sci_bench.mcp_config import load_mcp_config, load_science_mcp_catalog

from .support import BUNDLE, runner, setup, smoke_module


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


def _shared_manifest(tmp_path, mutate):
    """The real manifest with one member of the shared ToolUniverse group edited.

    ``arxiv``, ``alphafold_db`` and ``ncbi`` are three tool faces of one pinned
    checkout, so mutating one of them is how a group is made inconsistent on purpose.
    """
    return _manifest_with(tmp_path, "alphafold_db", mutate)


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
    # one checkout per upstream repository, not per tool face (shared with the other SMCP ids)
    assert entry["checkout"] == "tooluniverse"


def test_shared_checkout_is_one_directory_with_a_config_per_id(tmp_path):
    servers = setup.load_manifest()
    root = tmp_path / "root"
    shared = root / "tooluniverse"
    # three tool faces of the same pinned upstream, one checkout
    for sid in ("arxiv", "alphafold_db", "ncbi"):
        assert setup.checkout_dir(root, servers[sid]) == shared
    assert setup.checkout_dir(root, servers["pyscf"]) == root / "pyscf"      # default: the id
    assert setup.checkout_dir(root, {"id": "pyscf", "checkout": None}) == root / "pyscf"
    first = setup.render_config(servers["arxiv"], shared)["mcpServers"]
    second = setup.render_config(servers["alphafold_db"], shared)["mcpServers"]
    # the server name stays the id, so tool names (mcp__<id>__<tool>) do not change
    assert list(first) == ["arxiv"] and list(second) == ["alphafold_db"]
    assert first["arxiv"]["cwd"] == str(shared) == second["alphafold_db"]["cwd"]
    assert first["arxiv"]["command"] == second["alphafold_db"]["command"] \
        == str(shared / ".venv/bin/tooluniverse-smcp-stdio")
    assert first["arxiv"]["args"] != second["alphafold_db"]["args"]


def test_setup_installs_a_shared_group_into_one_checkout(tmp_path, monkeypatch):
    servers = setup.load_manifest()
    root = (tmp_path / "root").resolve()
    root.mkdir()
    seen = []
    monkeypatch.setattr(setup, "load_manifest", lambda *a, **k: servers)
    monkeypatch.setattr(setup, "check_host", lambda entry, **kwargs: None)
    monkeypatch.setattr(setup, "ensure_checkout", lambda entry, dest: seen.append(dest))
    monkeypatch.setattr(setup, "build_env", lambda entry, dest: dest / ".venv/bin/python")
    assert setup.main(["alphafold_db", "--root", str(root)]) == 0
    assert seen == [root / "tooluniverse"]
    # the config is still per id, and the id is not a directory of its own
    assert json.loads((root / "alphafold_db.mcp.json").read_text())["mcpServers"]["alphafold_db"]["cwd"] \
        == str(root / "tooluniverse")
    assert not (root / "alphafold_db").exists()


@pytest.mark.parametrize("mutate,match", [
    (lambda e: e.update(revision="b" * 40), r"differs on \['revision'\]"),
    (lambda e: e.update(python="3.13"), r"differs on \['python'\]"),
    (lambda e: e.update(uv_sync_args=[]), r"differs on \['uv_sync_args'\]"),
    (lambda e: e.update(repository="https://github.com/other/ToolUniverse.git"), "differs on"),
    (lambda e: e.update(checkout="pyscf"), "collides with manifest id 'pyscf'"),
    (lambda e: e.update(checkout="Tool Universe"), "checkout must be a lowercase identifier"),
    (lambda e: e.update(checkout=""), "checkout must be a lowercase identifier"),
])
def test_manifest_rejects_inconsistent_shared_checkouts(tmp_path, mutate, match):
    with pytest.raises(setup.SetupError, match=match):
        setup.load_manifest(_shared_manifest(tmp_path, mutate))


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


@pytest.mark.parametrize("sid,plat", sorted(
    (sid, plat) for sid, entry in setup.load_manifest().items()
    if entry["install"] == "conda-explicit" for plat in entry["conda"]["locks"]))
def test_committed_conda_locks_match_manifest(sid, plat):
    """Every committed lock pins exactly the versions its manifest specs ask for."""
    entry = setup.load_manifest()[sid]
    lock = setup.read_conda_lock(entry, plat)
    urls = [line for line in lock.read_text().splitlines() if line.startswith("https://")]
    for spec in entry["conda"]["specs"]:
        name, version = spec.split("=")
        assert any(f"/{name}-{version}-" in u for u in urls), f"{sid} {plat}: {spec}"
    assert len(urls) == len(set(urls))


def test_quantum_espresso_config_pins_the_local_runner_and_vendored_pseudos(tmp_path):
    entry = setup.load_manifest()["quantum_espresso"]
    dest = tmp_path / "quantum_espresso"
    path = tmp_path / "quantum_espresso.mcp.json"
    path.write_text(json.dumps(setup.render_config(entry, dest)))
    server = load_mcp_config(path)["quantum_espresso"]
    assert server["command"] == str(dest / ".venv/bin/python")
    assert server["args"] == ["-m", "qe_mcp.server"]
    # No Docker, no Globus, no mpirun, and pw.x resolved from the conda prefix rather
    # than from whatever PATH the caller happens to have.
    assert server["env"] == {
        "OMP_NUM_THREADS": "1",
        "PYTHONPATH": str(dest / "src"),
        "QE_NPROCS": "1",
        "QE_PREFIX": str(dest / ".venv/bin"),
        "QE_PSEUDO_DIR": str(dest / "pseudopotentials/sg15_oncv"),
        "QE_RUNNER": "local",
        "QE_USE_DOCKER": "false",
    }
    # QE_WORKDIR stays unset so the work directory follows the server's cwd
    # (upstream's .gitignore covers <checkout>/qe_calculations).
    assert "QE_WORKDIR" not in server["env"]
    # mcp 2.x removed mcp.server.fastmcp, which this revision imports: the pin is load-bearing.
    assert "mcp=1.28.1" in entry["conda"]["specs"]
    assert len(entry["expected_tools"]) == 19
    assert all(tool.startswith("qe_") for tool in entry["expected_tools"])


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
    assert setup.INSTALL_MODES == ("uv-sync-frozen", "uv-pip-pinned", "conda-explicit", "npm-ci")
    assert {e["install"] for e in setup.load_manifest().values()} == set(setup.INSTALL_MODES)
    assert setup.HOST_REQUIREMENT_KEYS == ("machine", "cpu_flags", "shared_libraries", "executables")


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


# --------------------------------------------------------------------------
# npm-ci (openroad) and the executables host probe
# --------------------------------------------------------------------------

def _npm_checkout(tmp_path, lock=True):
    dest = tmp_path / "openroad"
    package = dest / "typescript"
    package.mkdir(parents=True)
    (package / "package.json").write_text('{"name": "openroad-mcp"}')
    if lock:
        (package / "package-lock.json").write_text('{"lockfileVersion": 3}')
    return dest


def _fake_npm(monkeypatch, entry, *, launcher=b"#!/usr/bin/env node\n", tracked=""):
    """setup.run stand-in: `npm run build` writes the launcher (mode 0644, like tsc)."""
    commands = []

    def fake_run(cmd, cwd=None, env=None):
        commands.append((cmd, cwd, env))
        if cmd[1:3] == ["run", "build"] and launcher is not None:
            out = cwd / "dist" / "main.js"
            out.parent.mkdir(exist_ok=True)
            out.write_bytes(launcher)
            out.chmod(0o644)
        if cmd[0] == "git":
            return tracked
        return entry["python"] if cmd[-1].startswith("import sys") else ""

    def fake_which(name, path=None):
        assert path == setup.HOST_PATH or name == "uv", (name, path)
        return f"/usr/bin/{name}" if path else "/usr/local/bin/uv"

    monkeypatch.setattr(setup, "run", fake_run)
    monkeypatch.setattr(setup.shutil, "which", fake_which)
    return commands


def test_openroad_config_runs_the_built_script_with_the_default_orfs_path(tmp_path):
    entry = setup.load_manifest()["openroad"]
    dest = tmp_path / "openroad"
    path = tmp_path / "openroad.mcp.json"
    path.write_text(json.dumps(setup.render_config(entry, dest)))
    server = load_mcp_config(path)["openroad"]
    # the tsc output, executed through its `#!/usr/bin/env node` line (setup.py adds +x)
    assert server["command"] == str(dest / "typescript/dist/main.js")
    assert server["args"] == ["--transport", "stdio"]
    # pino level names only ("WARNING" crashes the server at start-up); no ORFS_FLOW_PATH, so
    # the smoke's temporary HOME decides where the ORFS tree is
    assert server["env"] == {"LOG_LEVEL": "WARN", "OPENROAD_COMMAND_TIMEOUT": "120"}
    assert entry["install"] == "npm-ci" and entry["npm"] == {"workdir": "typescript", "scripts": ["build"]}
    assert entry["revision"] == "7d2e540f86694beaf7f6dbe975a4e5f308161939"     # tag v1.1.0
    assert {"node", "npm", "openroad", "make", "g++", "python3"} == set(entry["host_requirements"]["executables"])
    assert len(entry["expected_tools"]) == 15


def test_npm_install_runs_ci_and_scripts_with_the_host_path_then_an_empty_venv(tmp_path, monkeypatch):
    entry = setup.load_manifest()["openroad"]
    dest = _npm_checkout(tmp_path)
    monkeypatch.setenv("npm_config_registry", "https://elsewhere.invalid/")
    monkeypatch.setenv("NPM_CONFIG_PREFIX", "/elsewhere")
    monkeypatch.setenv("NODE_OPTIONS", "--require /evil.js")
    monkeypatch.setenv("PATH", "/home/u/.nvm/versions/node/v18/bin:/usr/bin:/bin")
    commands = _fake_npm(monkeypatch, entry)
    python = setup.build_env(entry, dest)
    assert python == dest / ".venv/bin/python"
    package = dest / "typescript"
    npm_calls = [(cmd, cwd, env) for cmd, cwd, env in commands if cmd[0] == "/usr/bin/npm"]
    assert [(cmd, cwd) for cmd, cwd, _ in npm_calls] == [
        (["/usr/bin/npm", "ci", "--no-audit", "--no-fund"], package),
        (["/usr/bin/npm", "run", "build"], package),
    ]
    for _, _, env in npm_calls:
        # the node that compiles node-pty is the node that later runs the server
        assert env["PATH"] == setup.HOST_PATH
        assert not any(k.lower().startswith("npm_config_") for k in env) and "NODE_OPTIONS" not in env
    venv = [cmd for cmd, _, _ in commands if cmd[:2] == ["uv", "venv"]]
    assert venv == [["uv", "venv", "--clear", "--python", "3.12", str(dest / ".venv")]]
    assert (package / "dist/main.js").stat().st_mode & 0o777 == 0o755


@pytest.mark.parametrize("case,match", [
    ("no_lock", "package-lock.json not found"),
    ("no_launcher", "did not produce launch.command"),
    ("no_shebang", "no #! line"),
    ("tracked", "tracked but not executable"),
])
def test_npm_install_refuses_what_it_cannot_run(tmp_path, monkeypatch, case, match):
    entry = setup.load_manifest()["openroad"]
    dest = _npm_checkout(tmp_path, lock=case != "no_lock")
    _fake_npm(monkeypatch, entry,
              launcher=None if case == "no_launcher" else b"console.log(1)\n" if case == "no_shebang"
              else b"#!/usr/bin/env node\n",
              tracked="typescript/dist/main.js" if case == "tracked" else "")
    with pytest.raises(setup.SetupError, match=match):
        setup.build_env(entry, dest)
    if case == "tracked":       # never turned into a mode change of a tracked file
        assert (dest / "typescript/dist/main.js").stat().st_mode & 0o111 == 0


def test_npm_install_needs_npm_in_the_host_path(tmp_path, monkeypatch):
    entry = setup.load_manifest()["openroad"]
    dest = _npm_checkout(tmp_path)
    monkeypatch.setattr(setup, "run", lambda *a, **k: pytest.fail("ran without npm"))
    monkeypatch.setattr(setup.shutil, "which",
                        lambda name, path=None: "/usr/local/bin/uv" if name == "uv" else None)
    with pytest.raises(setup.SetupError, match=r"npm not found in /usr/bin:/bin"):
        setup.build_env(entry, dest)


@pytest.mark.parametrize("mutate,match", [
    (lambda e: e.pop("npm"), "exactly workdir, scripts"),
    (lambda e: e["npm"].update(lockfile="x"), "exactly workdir, scripts"),
    (lambda e: e["npm"].update(workdir="/abs/typescript"), "relative path"),
    (lambda e: e["npm"].update(workdir="../typescript"), "relative path"),
    (lambda e: e["npm"].update(workdir=""), "relative path"),
    (lambda e: e["npm"].update(scripts="build"), "npm script names"),
    (lambda e: e["npm"].update(scripts=["build && curl x"]), "npm script names"),
    (lambda e: e["launch"].update(command="dist/main.js"), "inside npm.workdir"),
    (lambda e: e.update(requirements=["x==1"]), r"\['requirements'\] only apply to install uv-pip-pinned"),
    (lambda e: e.update(install="uv-sync-frozen"), r"\['npm'\] only apply to install npm-ci"),
    (lambda e: e["host_requirements"].update(executables=["/usr/bin/node"]), "bare command names"),
    (lambda e: e["host_requirements"].update(executables=[]), "non-empty"),
])
def test_manifest_rejects_bad_npm_fields(tmp_path, mutate, match):
    with pytest.raises(setup.SetupError, match=match):
        setup.load_manifest(_manifest_with(tmp_path, "openroad", mutate))


def test_npm_workdir_may_be_the_checkout_root(tmp_path):
    def root_package(entry):
        entry["npm"]["workdir"] = "."
        entry["launch"]["command"] = "dist/main.js"
    assert setup.load_manifest(_manifest_with(tmp_path, "openroad", root_package))["openroad"]["npm"]["workdir"] == "."


def test_check_host_finds_executables_in_the_server_path_only(tmp_path, monkeypatch):
    entry = setup.load_manifest()["openroad"]
    on_host = {"node", "npm", "make", "python3"}
    with pytest.raises(setup.SetupError) as exc:
        setup.check_host(entry, which=lambda name: f"/usr/bin/{name}" if name in on_host else None)
    message = str(exc.value)
    assert "['g++', 'openroad'] not found in /usr/bin:/bin" in message and "node-pty" in message
    setup.check_host(entry, which=lambda name: f"/usr/bin/{name}")
    # by default the lookup ignores the caller's PATH (e.g. ~/.nvm, /usr/local/bin)
    seen = []
    monkeypatch.setattr(setup.shutil, "which", lambda name, path=None: seen.append(path) or "/usr/bin/x")
    setup.check_host(entry)
    assert seen and set(seen) == {setup.HOST_PATH}


def test_smoke_server_path_ends_with_the_host_path(tmp_path):
    env = runner.server_env({"command": "/m/openroad/typescript/dist/main.js"}, tmp_path, tmp_path)
    assert env["PATH"] == f"/m/openroad/typescript/dist:{setup.HOST_PATH}"
