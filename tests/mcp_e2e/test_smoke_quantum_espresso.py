"""``e2e_smoke/servers/quantum_espresso.py`` without Quantum ESPRESSO: banner parsing,
the order-independent pseudopotential index, and the environment checks against stub answers.
"""
import pytest

from . import support
from .support import StubClient, report_statuses, rpc_json, rpc_text, runner, setup

qe = support.smoke_module("quantum_espresso")

PW_BANNER = """
     Program PWSCF v.7.5 starts on  3Oct2026 at 13:39:10

     This program is part of the open-source Quantum ESPRESSO suite
     for quantum simulation of materials; please cite
"""
DOS_BANNER = "\n     Program DOS v.7.5 starts on  3Oct2026 at 13:39:13\n"

# The server's own view of the environment the manifest gives it (qe_status), as
# measured against the pinned revision on 2026-10-03.
STATUS = {
    "config": {"use_docker": False, "docker_image": "qe-local", "nprocs": 1,
               "workdir": "/cwd/qe_calculations", "pseudo_dir": "/co/pseudopotentials/sg15_oncv",
               "runner": "local"},
    "runner": {"available": True, "type": "LocalQERunner", "requested": "local"},
    "pseudopotentials": {"available": True, "n_elements": 2, "library": "SG15 ONCV"},
}
ENV = {"QE_NPROCS": "1", "QE_RUNNER": "local", "QE_PSEUDO_DIR": "/co/pseudopotentials/sg15_oncv"}


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------

def test_spec_version_reads_the_manifest_conda_specs():
    entry = setup.load_manifest()["quantum_espresso"]
    assert qe.spec_version(entry, "qe") == "7.5"
    assert qe.spec_version(entry, "mcp") == "1.28.1"
    assert qe.spec_version(entry, "python").startswith("3.12.")
    assert qe.spec_version(entry, "gpaw") is None
    # a prefix of another package's name must not match
    assert qe.spec_version({"conda": {"specs": ["python-dotenv=1.2.4"]}}, "python") is None


def test_banner_version_identifies_the_program():
    assert qe.banner_version(PW_BANNER, "PWSCF") == "7.5"
    assert qe.banner_version(DOS_BANNER, "DOS") == "7.5"
    assert qe.banner_version(PW_BANNER, "DOS") is None            # right banner, wrong program
    assert qe.banner_version("", "PWSCF") is None
    assert qe.banner_version("Program PWSCF v.7.5 stops", "PWSCF") is None
    assert qe.banner_version("     Program pwscf v.7.4.1 starts on x", "PWSCF") == "7.4.1"


def test_program_name_of_each_declared_executable():
    assert [qe.program_name(x) for x in qe.QE_EXECUTABLES] == ["PWSCF", "BANDS", "DOS", "PROJWFC"]


def test_upf_elements_is_independent_of_directory_order():
    """The glob-order defect decides which *file* an element gets, not which elements exist."""
    files = ["Si_ONCV_PBE-1.0.upf", "Si_ONCV_PBE-1.2.upf", "Si_ONCV_PBE_FR-1.1.upf",
             "O_ONCV_PBE-1.0.upf", "Ag_ONCV_PBE-1.2.upf"]
    assert qe.upf_elements(files) == ["Ag", "O", "Si"]
    assert qe.upf_elements(list(reversed(files))) == qe.upf_elements(files)
    # only SG15 ONCV names count, and the symbol is normalised the way SG15Library does
    assert qe.upf_elements(["si_oncv_pbe-1.2.upf"]) == ["Si"]
    assert qe.upf_elements(["Si.pbe-rrkj.UPF", "README.upf", "Si_ONCV_PBE-1.2.upf.bak"]) == []
    assert qe.upf_elements([]) == []


# --------------------------------------------------------------------------
# L0: the locked environment
# --------------------------------------------------------------------------

def session_for(tmp_path, responses=None, env=None, entry=None):
    report = runner.Report()
    client = StubClient(responses or {})
    checkout = tmp_path / "co"
    session = runner.Session(
        smoke=qe.SMOKE, args=None, entry=entry or setup.load_manifest()["quantum_espresso"],
        server={"command": str(checkout / ".venv/bin/python"), "args": [], "env": env or {}},
        checkout=checkout, report=report, tmp=tmp_path, home=tmp_path,
        cwd=tmp_path / "cwd", scratch=tmp_path / "scratch", env={}, client=client,
    )
    session.call = runner.Caller(client, report)
    return session


def pseudo_dir(session, names):
    path = session.checkout / qe.PSEUDO_SUBDIR
    path.mkdir(parents=True)
    for name in names:
        path.joinpath(name).write_text("")
    return path


def test_pseudopotential_library_check_records_the_element_index(tmp_path, monkeypatch):
    names = ["Si_ONCV_PBE-1.0.upf", "Si_ONCV_PBE-1.2.upf", "O_ONCV_PBE-1.0.upf"]
    monkeypatch.setattr(qe, "EXPECTED_UPF_FILES", len(names))
    session = session_for(tmp_path)
    directory = pseudo_dir(session, names)
    session.server["env"] = {"QE_PSEUDO_DIR": str(directory)}
    qe.check_pseudopotential_library(session)
    assert report_statuses(session.report) == {"vendored SG15 ONCV library": "PASS"}
    assert session.state["elements"] == ["O", "Si"]


def test_pseudopotential_library_check_fails_on_a_short_library(tmp_path):
    session = session_for(tmp_path)
    directory = pseudo_dir(session, ["Si_ONCV_PBE-1.2.upf"])
    session.server["env"] = {"QE_PSEUDO_DIR": str(directory)}
    qe.check_pseudopotential_library(session)
    assert report_statuses(session.report) == {"vendored SG15 ONCV library": "FAIL"}
    assert f"expected {qe.EXPECTED_UPF_FILES}" in session.report.checks[0]["detail"]


def test_pseudopotential_library_check_fails_when_the_env_points_elsewhere(tmp_path):
    session = session_for(tmp_path, env={"QE_PSEUDO_DIR": str(tmp_path / "elsewhere")})
    qe.check_pseudopotential_library(session)
    assert report_statuses(session.report) == {"QE_PSEUDO_DIR": "FAIL"}
    assert "elsewhere" in session.report.checks[0]["detail"]
    assert "elements" not in session.state


def test_qe_binaries_check_reads_each_banner(tmp_path, monkeypatch):
    """Four fake executables that print their own banner, as pw.x and friends do."""
    prefix = tmp_path / "co/.venv/bin"
    prefix.mkdir(parents=True)
    for executable in qe.QE_EXECUTABLES:
        script = prefix / executable
        script.write_text(f'#!/bin/sh\necho "     Program {qe.program_name(executable)} '
                          f'v.7.5 starts on 3Oct2026 at 00:00:00"\n')
        script.chmod(0o755)
    session = session_for(tmp_path, env={"QE_PREFIX": str(prefix)})
    qe.check_qe_binaries(session)
    assert set(report_statuses(session.report).values()) == {"PASS"}
    assert session.state["qe_versions"] == {x: "7.5" for x in qe.QE_EXECUTABLES}


@pytest.mark.parametrize("banner,reason", [
    ("     Program PWSCF v.7.4 starts on x", "another QE version"),
    ("", "no banner at all"),
])
def test_qe_binaries_check_fails_on_a_wrong_or_missing_banner(tmp_path, banner, reason):
    prefix = tmp_path / "co/.venv/bin"
    prefix.mkdir(parents=True)
    script = prefix / "pw.x"
    script.write_text(f"#!/bin/sh\necho {banner!r}\n")
    script.chmod(0o755)
    session = session_for(tmp_path, env={"QE_PREFIX": str(prefix)})
    qe.check_qe_binaries(session)
    statuses = report_statuses(session.report)
    assert statuses["pw.x reports QE 7.5"] == "FAIL", reason
    # the other three are not there at all: also FAIL, never silently skipped
    assert set(statuses.values()) == {"FAIL"} and len(statuses) == len(qe.QE_EXECUTABLES)


def test_qe_binaries_check_fails_when_qe_prefix_is_not_a_directory(tmp_path):
    session = session_for(tmp_path, env={"QE_PREFIX": str(tmp_path / "nowhere")})
    qe.check_qe_binaries(session)
    assert report_statuses(session.report) == {"QE_PREFIX": "FAIL"}


# --------------------------------------------------------------------------
# L1: qe_status
# --------------------------------------------------------------------------

def run_status(tmp_path, status, *, elements=("O", "Si")):
    session = session_for(tmp_path, responses={"qe_status": rpc_json(status)}, env=dict(ENV))
    if elements is not None:
        session.state["elements"] = list(elements)
    qe.check_status(session)
    return report_statuses(session.report)


def test_status_check_passes_on_the_servers_answer(tmp_path):
    assert set(run_status(tmp_path, STATUS).values()) == {"PASS"}


@pytest.mark.parametrize("patch,failing", [
    ({"config": {**STATUS["config"], "use_docker": True}}, "qe_status[config] echoes the launch env"),
    ({"config": {**STATUS["config"], "runner": "globus"}}, "qe_status[config] echoes the launch env"),
    ({"config": {**STATUS["config"], "nprocs": 4}}, "qe_status[config] echoes the launch env"),
    ({"config": {**STATUS["config"], "pseudo_dir": "/elsewhere"}},
     "qe_status[config] echoes the launch env"),
    ({"runner": {"available": True, "type": "DockerQERunner", "requested": "docker"}},
     "qe_status[runner] finds the local QE binaries"),
    ({"runner": {"available": False, "error": "Local QE executables not found."}},
     "qe_status[runner] finds the local QE binaries"),
    ({"pseudopotentials": {"available": True, "n_elements": 69, "library": "SG15 ONCV"}},
     "qe_status[pseudopotentials] matches the vendored library"),
    ({"pseudopotentials": {"available": False, "error": "Pseudopotential directory not found"}},
     "qe_status[pseudopotentials] matches the vendored library"),
])
def test_status_check_fails_on_a_misconfigured_environment(tmp_path, patch, failing):
    statuses = run_status(tmp_path, {**STATUS, **patch})
    assert statuses[failing] == "FAIL"
    assert [name for name, status in statuses.items() if status == "FAIL"] == [failing]


def test_status_check_fails_without_a_reference_element_index(tmp_path):
    statuses = run_status(tmp_path, STATUS, elements=None)
    assert statuses["qe_status[pseudopotentials] matches the vendored library"] == "FAIL"


def test_status_check_fails_on_a_tool_error(tmp_path):
    session = session_for(tmp_path, responses={"qe_status": rpc_text("boom", is_error=True)}, env=dict(ENV))
    qe.check_status(session)
    # the Caller already FAILs on isError; check_status must not then read fields off None
    assert set(report_statuses(session.report).values()) == {"FAIL"}


def test_smoke_declares_the_work_directory_it_leaves_in_cwd():
    """QE_WORKDIR is unset, so <cwd>/qe_calculations is expected, not an unexplained leftover."""
    assert qe.SMOKE.expected_cwd_files == ("qe_calculations",)
    assert "QE_WORKDIR" not in setup.load_manifest()["quantum_espresso"]["launch"]["env"]
