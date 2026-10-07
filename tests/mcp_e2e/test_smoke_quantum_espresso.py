"""``e2e_smoke/servers/quantum_espresso.py`` without Quantum ESPRESSO: banner parsing, the
pseudopotential index and its scan order (D1), the k-grid rule, our pw.x inputs and output parsers
(fixtures in pw.x 7.5's own format), and every three-state check against stub server answers.
"""
import math

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


def test_smoke_has_a_call_timeout_and_reports_its_timings():
    assert qe.SMOKE.call_timeout >= 600
    fields = qe.report_fields(session_for_state({"pseudo_pick": {"Si": "Si_ONCV_PBE-1.2.upf"}}))
    assert fields["sg15_pick"]["Si"] == "Si_ONCV_PBE-1.2.upf"
    assert set(fields) >= {"seconds_by_step", "workload", "qe_versions", "sg15_elements"}


def session_for_state(state):
    class _Session:
        pass
    session = _Session()
    session.state = state
    return session


# --------------------------------------------------------------------------
# Pure helpers: D1, the k-grid rule, geometry
# --------------------------------------------------------------------------

def test_scan_pick_follows_the_directory_order_as_sg15library_does():
    """D1: a later non-FR file replaces an earlier one; an FR file only fills an empty slot."""
    order = ["Si_ONCV_PBE-1.2.upf", "Si_ONCV_PBE-1.0.upf", "O_ONCV_PBE_FR-1.0.upf", "O_ONCV_PBE-1.0.upf",
             "Au_ONCV_PBE-1.0.upf", "Au_ONCV_PBE_FR-1.0.upf", "Xe_ONCV_PBE_FR-1.1.upf", "README.txt"]
    assert qe.scan_pick(order) == {"Si": "Si_ONCV_PBE-1.0.upf", "O": "O_ONCV_PBE-1.0.upf",
                                   "Au": "Au_ONCV_PBE-1.0.upf", "Xe": "Xe_ONCV_PBE_FR-1.1.upf"}
    assert qe.scan_pick(list(reversed(order)))["Si"] == "Si_ONCV_PBE-1.2.upf"


def test_newest_pick_is_order_independent_and_prefers_scalar_relativistic():
    names = ["Si_ONCV_PBE-1.0.upf", "Si_ONCV_PBE-1.2.upf", "Si_ONCV_PBE-1.1.upf", "Si_ONCV_PBE_FR-1.3.upf",
             "Xe_ONCV_PBE_FR-1.1.upf", "Xe_ONCV_PBE_FR-1.0.upf"]
    assert qe.newest_pick(names) == {"Si": "Si_ONCV_PBE-1.2.upf", "Xe": "Xe_ONCV_PBE_FR-1.1.upf"}
    assert qe.newest_pick(list(reversed(names))) == qe.newest_pick(names)
    assert qe.upf_version("Si_ONCV_PBE-1.10.upf") > qe.upf_version("Si_ONCV_PBE-1.2.upf")


def test_kgrid_reference_matches_the_documented_rule_for_silicon():
    lengths = qe.vector_lengths(qe.FCC_CELL)            # 3.8396 A
    assert lengths[0] == pytest.approx(5.43 / math.sqrt(2))
    assert qe.kgrid_reference(lengths, [True] * 3) == [11, 11, 11]        # 10.4 -> 10, tie 9/11 -> 11
    assert qe.kgrid_reference(lengths, [True] * 3, density="low") == [7, 7, 7]
    assert qe.kgrid_reference(lengths, [True] * 3, density="high") == [17, 17, 17]
    assert qe.kgrid_reference(lengths, [True] * 3, kspacing=0.05) == [7, 7, 7]   # ceil(5.2) = 6, tie -> 7
    assert qe.kgrid_reference([3.0, 3.0, 20.0], [True, True, False]) == [13, 13, 1]
    assert qe.kgrid_reference(lengths, [True] * 3, density="bogus") == [11, 11, 11]   # falls back to medium


@pytest.mark.parametrize("n,snapped", [(0, 1), (1, 1), (2, 3), (4, 5), (6, 7), (10, 11), (12, 13), (40, 21)])
def test_snap_odd_takes_the_larger_value_on_a_tie(n, snapped):
    assert qe.snap_odd(n) == snapped


def test_geometry_helpers_on_the_silicon_cell():
    assert qe.cell_volume(qe.FCC_CELL) == pytest.approx(5.43 ** 3 / 4, rel=1e-14)
    assert qe.min_image_distance(qe.SI_ATOMS, qe.FCC_CELL) == pytest.approx(5.43 * math.sqrt(3) / 4)
    assert qe.min_image_distance(qe.OVERLAP_ATOMS, qe.OVERLAP_CELL) == pytest.approx(0.3)
    # the nearest image, not the in-cell partner: (0,0,0) and (5.0,0,0) in a 5.43 box are 0.43 apart
    assert qe.min_image_distance((("Si", (0, 0, 0)), ("Si", (5.0, 0, 0))), qe.OVERLAP_CELL) == pytest.approx(0.43)
    assert qe.chemical_formula(["Si", "Si"]) == "Si2"
    assert qe.chemical_formula(["O", "Si", "O"]) == "O2Si"


def test_inline_structure_and_poscar_spell_out_our_cell():
    assert qe.inline_structure(qe.PERTURBED_ATOMS, qe.FCC_CELL) == (
        "xyz:Si 0.0 0.0 0.0; Si 1.42 1.3575 1.3575|lattice:0.0,2.715,2.715,2.715,0.0,2.715,2.715,2.715,0.0")
    text = qe.poscar(qe.PERTURBED_ATOMS, qe.FCC_CELL).splitlines()
    assert text[1:5] == ["1.0", "0.0 2.715 2.715", "2.715 0.0 2.715", "2.715 2.715 0.0"]
    assert text[5:8] == ["Si", "2", "Cartesian"] and text[9] == "1.42 1.3575 1.3575"


# --------------------------------------------------------------------------
# Our pw.x / bands.x / dos.x inputs
# --------------------------------------------------------------------------

def namelist_value(text, key):
    for line in text.splitlines():
        if line.strip().startswith(f"{key} ="):
            return line.split("=", 1)[1].strip()
    return None


def test_pw_input_carries_the_documented_scf_settings():
    text = qe.pw_input("scf", qe.SI_ATOMS, qe.FCC_CELL, "Si_ONCV_PBE-1.2.upf", kgrid=qe.KGRID)
    assert {k: namelist_value(text, k) for k in ("calculation", "occupations", "smearing", "degauss", "conv_thr",
                                                   "mixing_beta", "ecutwfc", "ecutrho", "nat", "ntyp")} == {
        "calculation": "'scf'", "occupations": "'smearing'", "smearing": "'cold'", "degauss": "0.02",
        "conv_thr": "1e-06", "mixing_beta": "0.7", "ecutwfc": "30.0", "ecutrho": "120.0", "nat": "2", "ntyp": "1"}
    assert "&IONS" not in text and "&CELL" not in text and "nbnd" not in text
    assert "  Si 1.0 Si_ONCV_PBE-1.2.upf" in text
    assert "K_POINTS automatic\n  4 4 4 0 0 0" in text
    assert "  Si 1.3575 1.3575 1.3575" in text


def test_pw_input_for_relaxations_bands_and_dos():
    relax = qe.pw_input("relax", qe.PERTURBED_ATOMS, qe.FCC_CELL, "p.upf", kgrid=qe.KGRID)
    assert "&IONS\n  ion_dynamics = 'bfgs'\n/" in relax and "&CELL" not in relax
    assert [namelist_value(relax, k) for k in ("nstep", "forc_conv_thr", "etot_conv_thr")] == ["100", "0.001", "0.0001"]
    vc = qe.pw_input("vc-relax", qe.SI_ATOMS, qe.FCC_CELL, "p.upf", kgrid=qe.KGRID)
    assert [namelist_value(vc, k) for k in ("cell_dynamics", "press_conv_thr", "cell_dofree")] == [
        "'bfgs'", "0.5", "'all'"]
    bands = qe.pw_input("bands", qe.SI_ATOMS, qe.FCC_CELL, "p.upf", kpoints_crystal=[[0, 0, 0], [0.5, 0.25, 0.75]],
                        system={"nbnd": 16, "nosym": True})
    assert namelist_value(bands, "nbnd") == "16" and namelist_value(bands, "nosym") == ".true."
    assert "K_POINTS crystal\n  2\n  0.0000000000 0.0000000000 0.0000000000 1.0\n" \
           "  0.5000000000 0.2500000000 0.7500000000 1.0" in bands
    dos = qe.pw_input("nscf", qe.SI_ATOMS, qe.FCC_CELL, "p.upf", kgrid=qe.DOS_KGRID,
                      occupations={"occupations": "tetrahedra"})
    assert namelist_value(dos, "occupations") == "'tetrahedra'" and namelist_value(dos, "smearing") is None
    assert "8 8 8 0 0 0" in dos
    tight = qe.pw_input("scf", qe.SI_ATOMS, qe.FCC_CELL, "p.upf", kgrid=qe.KGRID, electrons={"conv_thr": 1e-8})
    assert namelist_value(tight, "conv_thr") == "1e-08"
    assert namelist_value(qe.dos_x_input(), "deltae") == "0.01"
    assert namelist_value(qe.bands_x_input(), "filband") == "'bands.dat'"


# --------------------------------------------------------------------------
# Output parsers, on pw.x 7.5's own formats
# --------------------------------------------------------------------------

def pw_step(energy, fermi, totals, contributions, kbar):
    """One SCF step of a pw.x output with verbosity='high' (layout copied from a real relax run)."""
    rows = lambda forces: [f"     atom {i + 1:4d} type  1   force = {x:14.8f}{y:14.8f}{z:14.8f}"
                           for i, (x, y, z) in enumerate(forces)]
    return "\n".join([
        f"     the Fermi energy is {fermi:10.4f} ev", "",
        f"!    total energy              = {energy:17.8f} Ry",
        "     estimated scf accuracy    <       0.00000023 Ry", "",
        "     convergence has been achieved in   4 iterations", "",
        "     Forces acting on atoms (cartesian axes, Ry/au):", "",
        *rows(totals), "     The non-local contrib.  to forces", *rows(contributions), "",
        "     Total force =     0.044989     Total SCF correction =     0.000085", "", "",
        "     Computing stress (Cartesian axis) and pressure", "",
        f"          total   stress  (Ry/bohr**3)                   (kbar)     P=       {kbar[0]:.2f}",
        f"   0.00020714  -0.00000000   0.00000000 {kbar[0]:15.2f} {-0.0:11.2f} {0.0:11.2f}",
        f"   0.00000000   0.00021407  -0.00017812 {0.0:15.2f} {kbar[1]:11.2f} {-26.2:11.2f}",
        f"  -0.00000000  -0.00017812   0.00021407 {-0.0:15.2f} {-26.2:11.2f} {kbar[2]:11.2f}", ""])


RELAX_STEPS = [(-15.74891110, 6.6657, [(0.03181187, 0, 0), (-0.03181187, 0, 0)], (30.47, 31.49, 31.49)),
               (-15.75077486, 6.5238, [(0.00001000, 0, 0), (-0.00001000, 0, 0)], (30.40, 30.41, 30.41))]
RELAX_TAIL = """
     bfgs converged in   2 scf cycles and   1 bfgs steps
     End of BFGS Geometry Optimization

     Final energy             =     -15.7507748647 Ry
Begin final coordinates

ATOMIC_POSITIONS (angstrom)
Si               0.0312184375        0.0000000000        0.0000000000
Si               1.3887815625        1.3575000000        1.3575000000
End final coordinates

     JOB DONE.
"""
VC_TAIL = """
     End of BFGS Geometry Optimization

     Final enthalpy           =     -15.7516641260 Ry
Begin final coordinates
     new unit-cell volume =    278.96934 a.u.^3 (    41.33899 Ang^3 )

CELL_PARAMETERS (angstrom)
   0.000000000   2.744373982   2.744373982
   2.744373982   0.000000000   2.744373982
   2.744373982   2.744373982   0.000000000

ATOMIC_POSITIONS (angstrom)
Si              -0.0000000000        0.0000000000        0.0000000000
Si               1.3721869908        1.3721869908        1.3721869908
End final coordinates
"""


def relax_text(steps=RELAX_STEPS, tail=RELAX_TAIL):
    return "\n".join(pw_step(e, f, t, [(0.04, 0, 0), (-0.04, 0, 0)], k) for e, f, t, k in steps) + tail


def test_parse_pw_reads_every_step_of_a_relaxation():
    out = qe.parse_pw(relax_text(), 2)
    assert out["energies_ry"] == [-15.74891110, -15.75077486]
    assert out["fermi_ev"] == [6.6657, 6.5238] and out["iterations"] == [4, 4]
    assert out["final_ry"] == -15.7507748647 and out["bfgs_converged"] and out["job_done"]
    first = out["forces"][0]
    assert len(first["total"]) == 2 and len(first["all"]) == 4          # totals + one contribution block
    assert first["total"][0][0] == pytest.approx(0.03181187 * 13.605693122994 / 0.529177210903)
    assert out["stress_kbar"][0] == [[30.47, -0.0, 0.0], [0.0, 31.49, -26.2], [-0.0, -26.2, 31.49]]
    assert out["final_positions"] == [("Si", (0.0312184375, 0.0, 0.0)), ("Si", (1.3887815625, 1.3575, 1.3575))]
    assert out["final_cell"] is None                                    # a fixed-cell relaxation prints none


def test_parse_pw_reads_the_final_cell_of_a_vc_relaxation_and_nothing_of_a_bare_scf():
    out = qe.parse_pw(relax_text(tail=VC_TAIL), 2)
    assert out["final_ry"] == -15.7516641260
    assert out["final_cell"][0] == [0.0, 2.744373982, 2.744373982]
    assert out["final_positions"][1] == ("Si", (1.3721869908, 1.3721869908, 1.3721869908))
    scf = qe.parse_pw(pw_step(-15.75077338, 6.5233, [(0, 0, 0), (0, 0, 0)], [], (30.38,) * 3), 2)
    assert scf["final_ry"] is None and scf["final_positions"] is None and not scf["bfgs_converged"]


BANDS_DAT = """ &plot nbnd=  12, nks=     2 /
            0.000000  0.000000  0.000000
   -5.701    6.277    6.277    6.277    8.820    8.820    8.820    9.610   13.980   13.980
   14.209   17.492
           -0.000000  0.141421  0.000000
   -5.526    5.335    5.697    5.697    8.440    9.615    9.615   10.441   12.992   14.174
   14.934   17.797
"""


def test_parse_bands_dat_regroups_ten_values_per_line():
    dat = qe.parse_bands_dat(BANDS_DAT)
    assert (dat["nbnd"], dat["nks"]) == (12, 2)
    assert dat["kpoints"] == [[0.0, 0.0, 0.0], [-0.0, 0.141421, 0.0]]
    assert dat["eigenvalues"][1][:2] == [-5.526, 5.335] and dat["eigenvalues"][1][-1] == 17.797
    with pytest.raises(ValueError):
        qe.parse_bands_dat(BANDS_DAT.replace("17.797", ""))
    with pytest.raises(ValueError):
        qe.parse_bands_dat("no header")


def test_parse_bands_gnu_and_dos_dat():
    gnu = qe.parse_bands_gnu(qe.SYNTHETIC_GNU)
    assert gnu == {"k_distances": [0.0, 0.1414, 0.2828], "bands": [[-5.7007, -5.5258, -5.0049], [6.2771, 5.3349, 3.647]]}
    dos = qe.parse_dos_dat(qe.SYNTHETIC_DOS)
    assert dos["fermi_ev"] == 6.716 and dos["energies"] == [-5.701, -5.691, -5.681]
    assert dos["dos"][1] == 0.1759e-3 and dos["integrated"][2] == 0.4690e-5


@pytest.mark.parametrize("eigs,ef,want", [
    # Si-like: VBM at k0, CBM at k1 -> indirect
    ([[-5.7, 6.277, 8.82], [-5.5, 5.335, 6.828]], 6.5233,
     {"is_metal": False, "vbm_eV": 6.277, "cbm_eV": 6.828, "is_direct_gap": False}),
    # both edges at k1 -> direct; the first k of a repeated edge value decides
    ([[-5.0, 1.0, 9.0], [-5.0, 2.0, 3.0], [-5.0, 2.0, 3.0]], 2.5,
     {"is_metal": False, "vbm_eV": 2.0, "cbm_eV": 3.0, "is_direct_gap": True}),
    # a gap below 0.01 eV is a metal: no gap reported
    ([[1.0, 1.005]], 1.0, {"is_metal": True, "band_gap_eV": None, "is_direct_gap": False}),
    # everything occupied
    ([[1.0, 2.0]], 5.0, {"is_metal": True, "band_gap_eV": None, "vbm_eV": None}),
])
def test_gap_reference_applies_the_documented_band_edge_rule(eigs, ef, want):
    got = qe.gap_reference(eigs, ef)
    assert {k: got[k] for k in want} == want
    if not got["is_metal"]:
        assert got["band_gap_eV"] == pytest.approx(got["cbm_eV"] - got["vbm_eV"])


def test_categorize_files_uses_the_documented_rules():
    names = ["bands.dat.gnu", "bands.dat", "bands.dat.rap", "dos.dat", "pwscf.pdos_tot", "scf.in", "scf.out",
             "manifest.json"]
    assert qe.categorize_files(names) == {
        "band_files": ["bands.dat.gnu"], "dos_files": ["dos.dat"], "pdos_files": ["pwscf.pdos_tot"],
        "input_files": ["scf.in"], "output_files": ["scf.out"],
        "other_files": ["bands.dat", "bands.dat.rap", "manifest.json"]}


def test_close_and_mismatches_compare_nested_values():
    assert qe.close({"G": [0.0, 0.0], "X": [0.5, 0.0]}, {"G": [0.0, 0.0], "X": [0.5, 1e-12]}, 1e-9)
    assert not qe.close({"G": [0.0]}, {"G": [0.0], "X": [0.5]}, 1e-9)       # a missing key
    assert not qe.close([[1.0, 2.0]], [[1.0], [2.0]], 1e-9)                # same numbers, other shape
    assert not qe.close(None, 0.0, 1.0) and qe.close(None, None, 1.0)
    assert qe.max_diff([True], [1]) == math.inf                            # booleans are not numbers
    assert qe.mismatches({"a": 1.0, "b": "x"}, {"a": 1.0 + 1e-7, "b": "x"}, {"a": 1e-6}) == []
    assert qe.mismatches({"a": 1.0}, {"a": 1.1}, {"a": 1e-6}) == ["a: 1.0 != 1.1"]
    assert qe.mismatches({}, {"a": 1}, {}) == ["a: None != 1"]


# --------------------------------------------------------------------------
# L1 checks against stub server answers
# --------------------------------------------------------------------------

def l1_session(tmp_path, responses):
    session = session_for(tmp_path, responses=responses)
    (tmp_path / "cwd" / "qe_calculations").mkdir(parents=True)
    return session


PSEUDO_ORDER = ["Si_ONCV_PBE-1.2.upf", "Si_ONCV_PBE-1.0.upf", "O_ONCV_PBE-1.0.upf"]


def pseudo_reply(picks, si_hints=(30, 120)):
    return {"success": True, "library": "SG15 ONCV", "n_elements": 2, "elements": ["O", "Si"],
            "details": {el: {"filename": f, "ecutwfc_Ry": si_hints[0] if el == "Si" else 60,
                             "ecutrho_Ry": si_hints[1] if el == "Si" else 240} for el, f in picks.items()}}


def run_pseudo_check(tmp_path, picks, order=PSEUDO_ORDER):
    session = l1_session(tmp_path, {"qe_list_pseudopotentials": rpc_json(pseudo_reply(picks))})
    pseudo_dir(session, order)
    session.state.update(upf_in_order=list(order), elements=qe.upf_elements(order))
    return session, qe.check_pseudopotentials(session), report_statuses(session.report)


def test_pseudopotential_pick_that_follows_the_scan_but_is_stale_is_the_d1_warning(tmp_path):
    session, pick, statuses = run_pseudo_check(tmp_path, {"Si": "Si_ONCV_PBE-1.0.upf", "O": "O_ONCV_PBE-1.0.upf"})
    assert statuses == {"qe_list_pseudopotentials[index]": "PASS",
                        "qe_list_pseudopotentials[files follow the directory scan]": "PASS",
                        "qe_list_pseudopotentials[newest version per element]": "WARN",
                        "qe_list_pseudopotentials[Si cutoff hints]": "PASS"}
    assert pick == "Si_ONCV_PBE-1.0.upf"                     # the DFT references use the server's file
    assert session.state["pseudo_pick"]["not_newest"] == {"Si": "Si_ONCV_PBE-1.0.upf (newest Si_ONCV_PBE-1.2.upf)"}


def test_pseudopotential_pick_of_the_newest_files_passes(tmp_path):
    order = list(reversed(PSEUDO_ORDER))
    _, pick, statuses = run_pseudo_check(tmp_path, {"Si": "Si_ONCV_PBE-1.2.upf", "O": "O_ONCV_PBE-1.0.upf"}, order)
    assert set(statuses.values()) == {"PASS"} and pick == "Si_ONCV_PBE-1.2.upf"


def test_pseudopotential_pick_that_does_not_follow_the_scan_fails(tmp_path):
    _, _, statuses = run_pseudo_check(tmp_path, {"Si": "Si_ONCV_PBE-1.2.upf", "O": "O_ONCV_PBE-1.0.upf"})
    assert statuses["qe_list_pseudopotentials[files follow the directory scan]"] == "FAIL"
    assert statuses["qe_list_pseudopotentials[newest version per element]"] == "PASS"


def test_pseudopotential_pick_of_a_file_that_is_not_vendored_stops_the_dft_checks(tmp_path):
    session, pick, statuses = run_pseudo_check(tmp_path, {"Si": "Si_ONCV_PBE-9.9.upf", "O": "O_ONCV_PBE-1.0.upf"})
    assert pick is None and statuses["qe_list_pseudopotentials[Si file]"] == "FAIL"


KPATH_DEFECT = "The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()"
SPECIAL = {"G": [0.0, 0.0, 0.0], "X": [0.5, 0.0, 0.5]}
KPTS = [[0.0, 0.0, 0.0], [0.25, 0.0, 0.25], [0.5, 0.0, 0.5]]


@pytest.mark.parametrize("reply,status", [
    ({"success": False, "error": KPATH_DEFECT}, "WARN"),
    ({"success": False, "error": "something else"}, "FAIL"),
    ({"success": True, "n_kpoints": 3, "special_points": SPECIAL, "path_labels": "G-X", "kpoints": KPTS}, "PASS"),
    ({"success": True, "n_kpoints": 3, "special_points": SPECIAL, "path_labels": "X-G", "kpoints": KPTS}, "FAIL"),
])
def test_kpath_is_three_state_on_d10(tmp_path, monkeypatch, reply, status):
    monkeypatch.setattr(qe, "band_path", lambda npoints: (KPTS, SPECIAL))
    session = l1_session(tmp_path, {"qe_get_kpath": rpc_json(reply)})
    qe.check_kpath(session)
    assert report_statuses(session.report) == {"qe_get_kpath": status}


def test_environment_tools_accept_only_their_exact_errors(tmp_path):
    workdir = tmp_path / "cwd" / "qe_calculations"
    session = l1_session(tmp_path, {
        "qe_get_job_status": rpc_json({"success": False,
                                       "error": f"Job '{qe.MISSING_JOB}' not found in registry at {workdir}."}),
        "qe_search_materials_project": rpc_json({"success": False, "error": "MP_API_KEY not set", "hint": "x"}),
        "qe_get_mp_structure": rpc_json({"success": False, "error": "Failed to fetch from Materials Project: 401"}),
    })
    qe.check_environment_tools(session)
    assert report_statuses(session.report) == {
        "qe_get_job_status[local runner]": "WARN", "qe_search_materials_project[no MP_API_KEY]": "WARN",
        # the key is missing, so an answer from the network is not the documented behaviour
        "qe_get_mp_structure[no MP_API_KEY]": "FAIL"}


def test_job_status_from_another_registry_fails(tmp_path):
    session = l1_session(tmp_path, {"qe_get_job_status": rpc_json(
        {"success": False, "error": f"Job '{qe.MISSING_JOB}' not found in registry at /elsewhere/qe_calculations."})})
    qe.check_rejected(session.call, "j", "qe_get_job_status", {"job_id": qe.MISSING_JOB},
                      in_band=qe.job_status_error(tmp_path / "cwd" / "qe_calculations"))
    assert report_statuses(session.report) == {"j": "FAIL"}


@pytest.mark.parametrize("reply,status", [
    ({"success": False, "error": "[Errno 21] Is a directory: '/cwd/qe_calculations/bands_0123abcd'"}, "WARN"),
    ({"success": False, "error": "File not found: /x"}, "FAIL"),
    ({"success": True, "n_bands": 2, "bands": [[1.0], [2.0]]}, "PASS"),
])
def test_reading_a_directory_is_three_state_on_d2(tmp_path, reply, status):
    session = l1_session(tmp_path, {"qe_read_bands": rpc_json(reply)})
    qe.check_directory_reader(session, "qe_read_bands", "/cwd/qe_calculations/bands_0123abcd", "bands.dat.gnu",
                              {"success": True, "n_bands": 2, "bands": [[1.0], [2.0]]})
    assert list(report_statuses(session.report).values()) == [status]


def test_force_rows_are_three_state_on_d13(tmp_path):
    out = qe.parse_pw(relax_text(), 2)
    block = out["forces"][0]
    for rows, status in ((block["total"], "PASS"), (block["all"], "WARN"), (block["all"][:3], "FAIL")):
        report = runner.Report()
        qe.check_force_rows(report, "f", {"forces_eV_per_angstrom": rows}, out, 2)
        assert report.checks[0]["status"] == status


def test_synthetic_readers_compare_the_whole_reply(tmp_path):
    gnu, dos = qe.parse_bands_gnu(qe.SYNTHETIC_GNU), qe.parse_dos_dat(qe.SYNTHETIC_DOS)
    rows = [[float(x) for x in line.split()] for line in qe.SYNTHETIC_PDOS.splitlines()[1:]]
    missing = rpc_json({"success": False, "error": "File not found: x"})
    replies = {
        "qe_read_bands": lambda a: missing if a["output_dir"].endswith("missing.file") else rpc_json(
            {"success": True, "n_bands": 2, "n_kpoints": 3, "k_distances": gnu["k_distances"],
             "bands": gnu["bands"], "raw_data": qe.SYNTHETIC_GNU}),
        "qe_read_dos": lambda a: missing if a["output_dir"].endswith("missing.file") else rpc_json(
            {"success": True, "n_points": 3, "energies": dos["energies"], "dos": dos["dos"],
             "integrated_dos": dos["integrated"], "fermi_energy": 6.716, "raw_data": qe.SYNTHETIC_DOS}),
        "qe_read_pdos": lambda a: missing if a["pdos_file"].endswith("missing.file") else rpc_json(
            {"success": True, "atom_info": qe.SYNTHETIC_PDOS_NAME, "n_points": 2, "energies": [r[0] for r in rows],
             "ldos": [r[1] for r in rows], "pdos_columns": [r[2:] for r in rows],
             "raw_data": qe.SYNTHETIC_PDOS.replace("0.300E-03  0.300E-03", "0.300E-03  0.301E-03")}),
    }
    session = l1_session(tmp_path, replies)
    qe.check_readers_synthetic(session)
    statuses = report_statuses(session.report)
    assert statuses["qe_read_bands[synthetic]"] == "PASS" and statuses["qe_read_dos[synthetic]"] == "PASS"
    assert statuses["qe_read_pdos[synthetic]"] == "FAIL"                    # raw_data is not what we wrote
    assert {statuses[f"{t}[missing file]"] for t in ("qe_read_bands", "qe_read_dos", "qe_read_pdos")} == {"WARN"}


class FakeRef:
    pseudo_file = "Si_ONCV_PBE-1.2.upf"

    def __init__(self, tight):
        self.tight = tight

    def tight_scf(self, tag, atoms):
        return self.tight[tag]


def run_dir(session, kind, text, *, pick=FakeRef.pseudo_file):
    folder = session.cwd / "qe_calculations" / f"{kind}_0123abcd"
    (folder / "pseudo").mkdir(parents=True)
    (folder / "pseudo" / pick).write_text("")
    (folder / f"{folder.name}.out").write_text(text)
    return folder


def relax_reply(folder, step, *, rows=None):
    energy, fermi, totals, _ = step
    forces = [[x * qe.RYBOHR_TO_EVA for x in row] for row in totals]
    return {"success": True, "converged": True, "relaxation_converged": True, "total_energy_Ry": energy,
            "total_energy_eV": energy * qe.RY_EV, "fermi_energy_eV": fermi,
            "forces_eV_per_angstrom": rows if rows is not None else forces,
            "output_dir": str(folder), "output_file": str(folder / f"{folder.name}.out")}


@pytest.mark.parametrize("step_index,status", [(-1, "PASS"), (0, "WARN")])
def test_relaxation_reply_is_three_state_on_d11(tmp_path, step_index, status):
    session = l1_session(tmp_path, {})
    folder = run_dir(session, "relax", relax_text())
    session.client.responses["qe_run_relax"] = rpc_json(relax_reply(folder, RELAX_STEPS[step_index]))
    out = qe.parse_pw(relax_text(), 2)
    qe.check_relaxation(session, FakeRef({}), "qe_run_relax", "relax", out, "s", 2)
    statuses = report_statuses(session.report)
    assert statuses["qe_run_relax[run]"] == "PASS" and statuses["qe_run_relax[output file]"] == "PASS"
    assert statuses["qe_run_relax[reported energy, Fermi level, forces]"] == status


def test_relaxation_reply_matching_no_step_fails_and_a_different_run_is_caught(tmp_path):
    session = l1_session(tmp_path, {})
    other = relax_text([RELAX_STEPS[0], (-15.76, 6.5, RELAX_STEPS[1][2], (1.0, 1.0, 1.0))])
    folder = run_dir(session, "relax", other)                  # the server's own file disagrees with ours
    session.client.responses["qe_run_relax"] = rpc_json(relax_reply(folder, (-15.7, 6.0, RELAX_STEPS[0][2], None)))
    qe.check_relaxation(session, FakeRef({}), "qe_run_relax", "relax", qe.parse_pw(relax_text(), 2), "s", 2)
    statuses = report_statuses(session.report)
    assert statuses["qe_run_relax[output file]"] == "FAIL"
    assert statuses["qe_run_relax[reported energy, Fermi level, forces]"] == "FAIL"


def test_relaxation_run_outside_the_work_directory_or_with_another_pseudopotential_fails(tmp_path):
    session = l1_session(tmp_path, {})
    folder = run_dir(session, "relax", relax_text(), pick="Si_ONCV_PBE-1.0.upf")
    session.client.responses["qe_run_relax"] = rpc_json(relax_reply(folder, RELAX_STEPS[-1]))
    qe.check_relaxation(session, FakeRef({}), "qe_run_relax", "relax", qe.parse_pw(relax_text(), 2), "s", 2)
    assert "pseudo/ holds ['Si_ONCV_PBE-1.0.upf']" in session.report.checks[0]["detail"]
    assert qe.in_workdir(session, str(tmp_path / "elsewhere")) != []


def scf_parsed(energy, fermi):
    return qe.parse_pw(pw_step(energy, fermi, [(0, 0, 0), (0, 0, 0)], [], (1.0, 1.0, 1.0)), 2)


@pytest.mark.parametrize("final_energy,status", [(-15.75077486, "PASS"), (-15.74891143, "WARN"), (-15.7, "FAIL")])
def test_relax_and_scf_final_energy_is_three_state_on_d12(tmp_path, final_energy, status):
    tight = {"scf-unrelaxed": scf_parsed(-15.74891143, 6.6663), "scf-relaxed": scf_parsed(-15.75077486, 6.5238)}
    fermi = {-15.75077486: 6.5238, -15.74891143: 6.6663}.get(final_energy, 6.0)
    session = l1_session(tmp_path, {})
    folder = run_dir(session, "relax_scf", "")
    session.client.responses["qe_workflow_relax_and_scf"] = rpc_json({
        "workflow_id": folder.name, "output_dir": str(folder), "success": True,
        "relaxation": {"success": True, "converged": True, "energy_eV": RELAX_STEPS[0][0] * qe.RY_EV},
        "total_energy_Ry": final_energy, "total_energy_eV": final_energy * qe.RY_EV, "fermi_energy_eV": fermi})
    qe.check_relax_and_scf(session, FakeRef(tight), qe.parse_pw(relax_text(), 2))
    statuses = report_statuses(session.report)
    assert statuses["qe_workflow_relax_and_scf[relaxation]"] == "PASS"
    assert statuses["qe_workflow_relax_and_scf[relaxation energy]"] == "WARN"      # D11: the first step
    assert statuses["qe_workflow_relax_and_scf[final SCF]"] == status
