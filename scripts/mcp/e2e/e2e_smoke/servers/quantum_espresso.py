"""Direct (agent-free) E2E smoke test for the pinned ``frimpsjoek/qe-mcp`` MCP server.

Run with the server's own conda prefix (it provides the Quantum ESPRESSO
binaries and ASE that the references need)::

    ~/mcp/quantum_espresso/.venv/bin/python scripts/mcp/e2e/smoke.py quantum_espresso \
        --config ~/mcp/quantum_espresso.mcp.json

No network is needed: the 219 SG15 ONCV ``.upf`` files are vendored in the
pinned revision, so nothing is downloaded and ``scripts/download_pseudos.py``
is never run. The two Materials Project tools are the only ones that would go
online, and they are not called here.

**This is the L0 stage of the server.** It establishes that the locked conda
environment really is a working QE installation and that the manifest's launch
environment reaches the server, which is the precondition for every later
numerical check:

L0 (on top of the shared checks in ``e2e_smoke/runner.py``: pinned revision,
config equals what ``setup.py`` renders, handshake, the 19 expected tools,
stable ``tools/list``)

* ``pw.x``, ``bands.x``, ``dos.x`` and ``projwfc.x`` exist in the prefix the
  manifest points ``QE_PREFIX`` at, are executable, and each prints the
  QE version pinned by ``conda.specs`` (``qe=7.5``) in its own startup banner,
  measured outside the server with an empty input file;
* the vendored SG15 library holds the expected number of ``.upf`` files, and
  the element index derived from their names (order-independently, see the
  glob-order defect below) is recorded as the reference for L1.

L1 (minimal for now, one tool)

* ``qe_status`` — the server's own view of the same environment: it echoes the
  launch env (``use_docker`` false, ``nprocs`` 1, runner ``local``, the
  manifest's ``pseudo_dir``), reports a usable ``LocalQERunner`` (which is
  exactly `shutil.which` on ``$QE_PREFIX/pw.x``, i.e. the binaries found
  above), and counts the same elements the reference derived from the
  directory.

The remaining 18 tools — the ten pure-Python ones, the six that really run
``pw.x``/``bands.x``/``dos.x``, and the three environment-dependent ones — are
not covered yet; adding them is the L1 stage, and the facts a reference needs
are in the "Notes for task authors" section of ``scripts/mcp/e2e/README.md``.

Classification: a missing or mismatched binary, or a server that cannot reach
them, is FAIL — it means the locked environment is not a QE installation, so
nothing downstream can be trusted. Upstream defects that do not make correct
use wrong are WARN.

Known upstream defect that matters for the pseudopotential reference:
``SG15Library._scan_library`` iterates ``pseudo_dir.glob("*.upf")`` and lets a
later non-``_FR`` file overwrite an earlier one, without sorting or comparing
versions, so for an element shipped in several revisions (Si has 1.0, 1.1 and
1.2) *which file* the server picks depends on the host's directory order. The
*set of elements* does not, so this reference compares the element index only;
an L1 check of ``qe_list_pseudopotentials`` must read the picked
``details.<El>.filename`` from the server rather than assume the highest
version.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from ..runner import Session, Smoke

# The QE executables the server can invoke (config.py's exe_map), and the one
# version string all of them must report: the manifest's conda spec qe=<version>.
QE_EXECUTABLES = ("pw.x", "bands.x", "dos.x", "projwfc.x")
# "     Program PWSCF v.7.5 starts on  3Oct2026 at 13:39:10"
BANNER_RE = re.compile(r"^\s*Program\s+(?P<program>\S+)\s+v\.(?P<version>\S+)\s+starts", re.MULTILINE)
BANNER_TIMEOUT = 60.0        # seconds; a banner appears in well under a second

# The vendored SG15 ONCV library of the pinned revision.
PSEUDO_SUBDIR = "pseudopotentials/sg15_oncv"
EXPECTED_UPF_FILES = 219
# SG15Library's own naming pattern (pseudopotentials.py), applied to every file
# instead of to the first one glob happens to return.
UPF_RE = re.compile(r"^([A-Z][a-z]?)_ONCV_PBE.*\.upf$", re.IGNORECASE)


# --------------------------------------------------------------------------
# Pure helpers (stdlib only; unit-tested offline)
# --------------------------------------------------------------------------

def spec_version(entry: dict, package: str) -> str | None:
    """The version ``conda.specs`` pins for ``package``, e.g. 'qe=7.5' -> '7.5'."""
    prefix = f"{package}="
    for spec in entry["conda"]["specs"]:
        if spec.startswith(prefix):
            return spec[len(prefix):]
    return None


def banner_version(output: str, program: str) -> str | None:
    """The version from a QE startup banner, if it is the banner of ``program``."""
    match = BANNER_RE.search(output)
    if match is None or match["program"].upper() != program.upper():
        return None
    return match["version"]


def upf_elements(filenames: list[str]) -> list[str]:
    """The element index SG15Library builds, derived order-independently from the file names.

    Which *file* an element maps to depends on the host's directory order
    (see the module docstring); which *elements* exist does not.
    """
    return sorted({match.group(1).capitalize() for name in filenames
                   if (match := UPF_RE.match(name))})


def program_name(executable: str) -> str:
    """The banner program name of a QE executable: pw.x -> PWSCF, dos.x -> DOS."""
    stem = executable.removesuffix(".x").upper()
    return "PWSCF" if stem == "PW" else stem


# --------------------------------------------------------------------------
# L0: the locked environment is a working QE installation
# --------------------------------------------------------------------------

def check_qe_binaries(session: Session) -> None:
    """Each declared executable exists under QE_PREFIX and reports the pinned QE version.

    Runs them outside the server, with an empty input file in a scratch
    directory of their own: they then print the startup banner and stop (and
    may drop a ``CRASH`` file, which is why this is not the session cwd).
    """
    report, entry = session.report, session.entry
    pinned = spec_version(entry, "qe")
    prefix = Path(session.server.get("env", {}).get("QE_PREFIX", ""))
    if not prefix.is_dir():
        report.add("L0", "QE_PREFIX", "FAIL", f"launch env QE_PREFIX={str(prefix)!r} is not a directory")
        return
    probe = session.tmp / "qe-version-probe"
    probe.mkdir()
    empty = probe / "empty.in"
    empty.write_text("", encoding="utf-8")
    versions = session.state["qe_versions"] = {}
    for executable in QE_EXECUTABLES:
        name = f"{executable} reports QE {pinned}"
        binary = prefix / executable
        try:
            completed = subprocess.run([str(binary), "-i", str(empty)], cwd=probe, env=session.env,
                                       capture_output=True, text=True, timeout=BANNER_TIMEOUT)
        except (OSError, subprocess.SubprocessError) as exc:
            report.add("L0", name, "FAIL", f"cannot run {binary}: {exc}")
            continue
        finally:
            (probe / "CRASH").unlink(missing_ok=True)   # QE's error marker for the empty input
        version = banner_version(completed.stdout, program_name(executable))
        versions[executable] = version
        if version == pinned:
            report.add("L0", name, "PASS", f"{binary.name} banner: {program_name(executable)} v.{version}")
        else:
            report.add("L0", name, "FAIL",
                       f"banner reports {version!r}, conda.specs pins qe={pinned}; "
                       f"stdout {completed.stdout[:200]!r} stderr {completed.stderr[:200]!r}")


def check_pseudopotential_library(session: Session) -> None:
    """The vendored SG15 library is complete, and its element index is the L1 reference."""
    report = session.report
    pseudo_dir = Path(session.server.get("env", {}).get("QE_PSEUDO_DIR", ""))
    expected_dir = session.checkout / PSEUDO_SUBDIR
    if pseudo_dir != expected_dir:
        report.add("L0", "QE_PSEUDO_DIR", "FAIL",
                   f"launch env points at {str(pseudo_dir)!r}, expected the vendored {str(expected_dir)!r}")
        return
    names = sorted(p.name for p in pseudo_dir.glob("*.upf"))
    elements = session.state["elements"] = upf_elements(names)
    status = "PASS" if len(names) == EXPECTED_UPF_FILES else "FAIL"
    report.add("L0", "vendored SG15 ONCV library", status,
               f"{len(names)} .upf files" + ("" if status == "PASS" else f", expected {EXPECTED_UPF_FILES}")
               + f"; {len(elements)} elements indexed by name",
               n_upf_files=len(names), n_elements=len(elements))


def prepare(session: Session) -> None:
    check_qe_binaries(session)
    check_pseudopotential_library(session)


# --------------------------------------------------------------------------
# L1
# --------------------------------------------------------------------------

def check_status(session: Session) -> None:
    """``qe_status``: the server's view of the environment the manifest gives it."""
    call, report = session.call, session.report
    result = call.json("qe_status", "qe_status", {})
    if not isinstance(result, dict) or result.get("_isError"):
        report.add("L1", "qe_status", "FAIL", f"unusable result: {result!r}")
        return
    config, runner = result.get("config", {}), result.get("runner", {})
    pseudos = result.get("pseudopotentials", {})
    env = session.server.get("env", {})

    expected_config = {"use_docker": False, "nprocs": int(env.get("QE_NPROCS", "1")),
                       "runner": env.get("QE_RUNNER"), "pseudo_dir": env.get("QE_PSEUDO_DIR")}
    wrong = {key: config.get(key) for key, want in expected_config.items() if config.get(key) != want}
    report.add("L1", "qe_status[config] echoes the launch env", "PASS" if not wrong else "FAIL",
               f"{expected_config}" if not wrong else f"{wrong} != expected {expected_config}",
               server_value=config)

    local = runner.get("available") is True and runner.get("type") == "LocalQERunner"
    report.add("L1", "qe_status[runner] finds the local QE binaries", "PASS" if local else "FAIL",
               f"{runner.get('type')}, requested {runner.get('requested')!r}" if local
               else f"expected an available LocalQERunner, got {runner!r}",
               server_value=runner)

    elements = session.state.get("elements")
    ok = pseudos.get("available") is True and pseudos.get("library") == "SG15 ONCV" \
        and elements is not None and pseudos.get("n_elements") == len(elements)
    report.add("L1", "qe_status[pseudopotentials] matches the vendored library", "PASS" if ok else "FAIL",
               f"{pseudos.get('n_elements')} elements, as derived from the .upf names" if ok
               else f"{pseudos!r} does not match {len(elements) if elements else '?'} indexed elements",
               server_value=pseudos)


def run_l1(session: Session) -> None:
    check_status(session)


SMOKE = Smoke(
    server="quantum_espresso",
    run_l1=run_l1,
    packages=("mcp", "ase", "numpy", "spglib", "pyyaml", "python-dotenv"),
    prepare=prepare,
    # QE_WORKDIR is deliberately unset in the manifest, so the server's work
    # directory is <cwd>/qe_calculations: the smoke's temporary cwd here (and
    # the checkout in an agent run, where upstream's .gitignore covers it).
    expected_cwd_files=("qe_calculations",),
    report_fields=lambda session: {"qe_versions": session.state.get("qe_versions"),
                                   "sg15_elements": session.state.get("elements")},
)
