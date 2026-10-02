"""Direct (agent-free) E2E smoke test for the pinned chemaster ``calc_psi4`` MCP server.

Run with the server's own conda prefix (it provides psi4 for the references)::

    ~/mcp/psi4/.venv/bin/python scripts/mcp/e2e/smoke.py psi4 --config ~/mcp/psi4.mcp.json

No network is needed. References are computed here, in this process (never
in the server), by calling psi4 directly with the same physical settings the
tools document (density-fitted SCF, C1 symmetry, RHF/UHF by multiplicity) but
through a different code path: wavefunction objects, ``psi4.variable`` and
``tdscf_excitations`` instead of the server's output-log parsers.

L1 (the shared L0/L1 checks are in ``e2e_smoke/runner.py``): real tools/call
of all five tools:
      single_point  HF / UHF / DFT-D3(BJ) / MP2 energies and documented fields
      optimize      HF/STO-3G water minimum from a distorted start
      frequency     at that minimum (frequencies, ZPE, thermochemistry, IR),
                    with a non-default temperature, and at a first-order
                    saddle point (planar NH3, one imaginary mode)
      tddft         TDA singlets+triplets and full TDDFT singlets
      optimize_excited_state  checked for consistency with an excited state:
                    returned energy = E(S_n) at the returned geometry, and
                    the S_n gradient vanishes there
      plus in-band validation errors

Classification: wrong numbers for a correctly used tool in its documented
regime (minimum geometries, ground-state properties) are FAIL. Upstream
defects are WARN when the server's output matches a *recognised* defect
exactly — a documented field that is always null/zero, an ignored argument,
an imaginary frequency reported as real, an "excited-state" optimisation
that returns the ground-state minimum — and FAIL when it matches neither the
correct answer nor the recognised defect.
"""
from __future__ import annotations

import contextlib
import json
import math
from pathlib import Path

from ..client import text_of
from ..helpers import Atoms, max_abs_diff, quiet_fds, sorted_distances
from ..runner import Caller, Report, Session, Smoke, check_rejected

# Distorted water (the server docstrings' style of input: bare atom lines, Å).
H2O_START = ("O 0.000000 0.000000 0.117790\n"
             "H 0.000000 0.755453 -0.471161\n"
             "H 0.000000 -0.755453 -0.471161")
OH_RADICAL = "O 0.0 0.0 0.0\nH 0.0 0.0 0.97"
# Planar NH3: D3h stationary point = first-order saddle (umbrella inversion).
NH3_PLANAR_START = "N 0 0 0\nH 1.0 0 0\nH -0.5 0.8660254 0\nH -0.5 -0.8660254 0"

COMMON = {"memory_gb": 1, "n_threads": 1}   # keep psi4 well inside small VMs

# Tolerances. Same program, same settings, different call path: SCF energies
# agree to convergence; the server rounds energies to 1e-8 Eh, log-parsed
# frequencies to 1e-4 cm^-1 and excitation energies / oscillator strengths to
# 1e-4.
ENERGY_TOL = 2e-7          # Eh
OPT_ENERGY_TOL = 1e-6      # Eh, minima from two optimizer runs (gau_tight)
OPT_DIST_TOL = 1e-3        # Å, sorted interatomic distances of the minima
FREQ_TOL = 0.5             # cm^-1 (log-parsed, 4 decimals; geometry from the server's own output)
THERMO_TOL = 2e-6          # Eh, ZPE / thermal corrections
EXC_TOL = 2e-3             # eV
OSC_TOL = 2e-3             # oscillator strength
FIELD_REL_TOL = 1e-3       # relative, for n_basis / gap / dipole when present
GRAD_TOL = 2e-3            # Eh/Å, |grad E(S_n)| at a converged excited-state minimum
FD_STEP = 2e-3             # Å, central differences for the S_n gradient
IMAG_THRESHOLD = -10.0     # cm^-1, the server's documented imaginary cut-off

HARTREE2EV = 27.211386245988   # CODATA 2018 (psi4.constants.hartree2ev); used only to compare eV values
AU2DEBYE = 2.541746473         # psi4.constants.dipmom_au2debye


# --------------------------------------------------------------------------
# Pure helpers (stdlib only; unit-tested offline)
# --------------------------------------------------------------------------

def parse_geometry(text: str) -> Atoms:
    """Atom lines 'Sym x y z'; skips a psi4 'charge multiplicity' header and an XYZ count/comment."""
    atoms: Atoms = []
    for line in text.strip().splitlines():
        fields = line.split()
        if len(fields) != 4:
            continue
        try:
            xyz = tuple(float(v) for v in fields[1:])
        except ValueError:
            continue
        if not fields[0].isalpha():
            continue
        atoms.append((fields[0].capitalize(), xyz))
    if not atoms:
        raise ValueError(f"no atom lines in {text[:120]!r}")
    return atoms


def atom_lines(atoms: Atoms) -> str:
    return "\n".join(f"{sym} {x:.10f} {y:.10f} {z:.10f}" for sym, (x, y, z) in atoms)


def payload(result: dict) -> dict:
    """The server returns one JSON object (as text and structuredContent)."""
    data = result.get("structuredContent")
    if not isinstance(data, dict):
        data = json.loads(text_of(result))
    if not isinstance(data, dict):
        raise ValueError("result is not a JSON object")
    return data


def value_of(field) -> float | None:
    """{'value': x, 'unit': ...} → x; plain numbers pass through; None stays None."""
    if isinstance(field, dict):
        field = field.get("value")
    if isinstance(field, bool) or not isinstance(field, (int, float)):
        return None
    return float(field)


def classify_frequencies(server: list[float], reference: list[float], n_imag_server, tol: float = FREQ_TOL):
    """PASS: signed frequencies match. WARN: match only in magnitude (imaginary sign dropped). FAIL otherwise."""
    ref_n_imag = sum(1 for f in reference if f < IMAG_THRESHOLD)
    signed = max_abs_diff(sorted(server), sorted(reference))
    if signed <= tol and n_imag_server == ref_n_imag:
        return "PASS", f"max|Δν|={signed:.2g} cm^-1, n_imaginary={n_imag_server}"
    magnitude = max_abs_diff(sorted(abs(f) for f in server), sorted(abs(f) for f in reference))
    if magnitude <= tol and ref_n_imag > 0 and n_imag_server == 0 and all(f >= 0 for f in server):
        imag = sorted(f for f in reference if f < IMAG_THRESHOLD)
        return "WARN", (f"imaginary mode(s) {[round(f, 2) for f in imag]} cm^-1 returned as real positive "
                        f"frequencies with n_imaginary=0 and no warning (upstream log parser drops the 'i'); "
                        f"magnitudes match to {magnitude:.2g} cm^-1")
    return "FAIL", (f"frequencies {[round(f, 2) for f in server]} (n_imaginary={n_imag_server}) vs reference "
                    f"{[round(f, 2) for f in reference]}")


def classify_alternative(got: float | None, correct: float, defect: float, tol: float):
    """Tri-state: 'correct' within tol → PASS, the recognised defect within tol → WARN, else FAIL."""
    if got is None:
        return "FAIL"
    if abs(got - correct) <= tol:
        return "PASS"
    if abs(got - defect) <= tol:
        return "WARN"
    return "FAIL"


def compare_states(server: list[dict], reference: list[tuple[float, float]]):
    """Server states [{'excitation_energy': {'value': eV}, 'oscillator_strength': f}] vs reference [(eV, f)]."""
    got = [(value_of(s.get("excitation_energy")), value_of(s.get("oscillator_strength"))) for s in server]
    if any(e is None or f is None for e, f in got):
        return math.inf, math.inf
    n = min(len(got), len(reference))
    de = max((abs(got[k][0] - reference[k][0]) for k in range(n)), default=0.0)
    df = max((abs(got[k][1] - reference[k][1]) for k in range(n)), default=0.0)
    return de, df


# --------------------------------------------------------------------------
# Independent references: psi4 in THIS process
# --------------------------------------------------------------------------

class Psi4Ref:
    """Fresh psi4 state per reference calculation; output goes to a scratch log."""

    def __init__(self, scratch: Path) -> None:
        with quiet_fds():
            import psi4
        self.psi4 = psi4
        self.scratch = scratch
        self.version = psi4.__version__

    @contextlib.contextmanager
    def session(self, geometry: str, charge: int, mult: int, options: dict, *, c1: bool = True):
        psi4 = self.psi4
        with quiet_fds():
            psi4.core.clean()
            psi4.core.clean_options()
            psi4.core.clean_variables()
            psi4.set_memory("1 GB", quiet=True)
            psi4.set_num_threads(1, quiet=True)
            psi4.core.IOManager.shared_object().set_default_path(str(self.scratch))
            psi4.core.set_output_file(str(self.scratch / "reference.log"), False)
            mol = psi4.geometry(f"{charge} {mult}\n{geometry}\n" + ("symmetry c1\n" if c1 else ""))
            psi4.set_options({"scf_type": "df", "reference": "uhf" if mult != 1 else "rhf", **options})
            yield psi4, mol

    def scf(self, geometry: str, method: str, basis: str, charge: int = 0, mult: int = 1) -> dict:
        with self.session(geometry, charge, mult, {"guess": "sad"}) as (psi4, mol):
            energy, wfn = psi4.energy(f"{method}/{basis}", molecule=mol, return_wfn=True)
            scf_wfn = wfn.reference_wavefunction() or wfn     # MP2 wraps the SCF wavefunction
            eps = scf_wfn.epsilon_a().to_array()
            nocc = scf_wfn.nalpha()
            dip = psi4.variable("SCF DIPOLE")
            return {"energy": float(energy), "nbf": int(scf_wfn.basisset().nbf()),
                    "iterations": int(psi4.variable("SCF ITERATIONS")),
                    "gap_eV": float(eps[nocc] - eps[nocc - 1]) * HARTREE2EV,
                    "dipole_debye": math.sqrt(sum(float(c) ** 2 for c in dip)) * AU2DEBYE}

    def optimize(self, geometry: str, method: str, basis: str, g_convergence: str, *, c1: bool = True) -> tuple:
        with self.session(geometry, 0, 1, {"g_convergence": g_convergence, "geom_maxiter": 100},
                          c1=c1) as (psi4, mol):
            energy = psi4.optimize(f"{method}/{basis}", molecule=mol)
            mol.update_geometry()
            return float(energy), parse_geometry(mol.save_string_xyz())

    def frequencies(self, geometry: str, method: str, basis: str, temperature: float = 298.15) -> dict:
        with self.session(geometry, 0, 1, {"t": temperature}) as (psi4, mol):
            _, wfn = psi4.frequencies(f"{method}/{basis}", molecule=mol, return_wfn=True)
            info = wfn.frequency_analysis
            freqs, ir = [], []
            for k, kind in enumerate(info["TRV"].data):
                if kind != "V":
                    continue
                w = complex(info["omega"].data[k])
                freqs.append(-w.imag if abs(w.imag) > abs(w.real) else w.real)
                ir.append(float(info["IR_intensity"].data[k]))
            return {"frequencies": freqs, "ir": ir, "zpe": psi4.variable("ZPVE"),
                    "e_corr": psi4.variable("THERMAL ENERGY CORRECTION"),
                    "h_corr": psi4.variable("ENTHALPY CORRECTION"),
                    "g_corr": psi4.variable("GIBBS FREE ENERGY CORRECTION")}

    def excitations(self, geometry: str, method: str, basis: str, states: int, triplets: bool, tda: bool) -> dict:
        from psi4.driver.procrouting.response.scf_response import tdscf_excitations
        with self.session(geometry, 0, 1, {"save_jk": True}) as (psi4, mol):
            energy, wfn = psi4.energy(f"{method}/{basis}", molecule=mol, return_wfn=True)
            with quiet_fds():
                roots = tdscf_excitations(wfn, states=states, triplets="also" if triplets else "none", tda=tda)
            out = {"energy": float(energy), "singlet": [], "triplet": []}
            for r in roots:
                out[str(r["SPIN"]).lower()].append((float(r["EXCITATION ENERGY"]) * HARTREE2EV,
                                                    float(r["OSCILLATOR STRENGTH (LEN)"])))
            for spin in ("singlet", "triplet"):
                out[spin].sort()
            return out

    def excited_total(self, atoms: Atoms, method: str, basis: str, states: int, root: int, spin: str) -> tuple:
        """(E_ground, E(S_root or T_root)) in Eh with TDA, the server's excited-state settings."""
        exc = self.excitations(atom_lines(atoms), method, basis, states, spin == "triplet", tda=True)
        manifold = exc[spin]
        if len(manifold) < root:
            raise RuntimeError(f"reference found only {len(manifold)} {spin} roots")
        return exc["energy"], exc["energy"] + manifold[root - 1][0] / HARTREE2EV

    def excited_gradient_norm(self, atoms: Atoms, method: str, basis: str, states: int, root: int,
                              spin: str) -> float:
        """|∇E_root| in Eh/Å by central differences (psi4 has no analytic TDDFT gradients)."""
        total = 0.0
        for i, (sym, xyz) in enumerate(atoms):
            for axis in range(3):
                energies = []
                for sign in (1, -1):
                    moved = list(xyz)
                    moved[axis] += sign * FD_STEP
                    displaced = [*atoms[:i], (sym, tuple(moved)), *atoms[i + 1:]]
                    energies.append(self.excited_total(displaced, method, basis, states, root, spin)[1])
                total += ((energies[0] - energies[1]) / (2 * FD_STEP)) ** 2
        return math.sqrt(total)


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def tool_payload(call: Caller, report: Report, name: str, tool: str, arguments: dict) -> dict | None:
    """Successful call returning ok=true; anything else is a FAIL for a valid request."""
    result = call(name, tool, {**arguments, **COMMON})
    if result is None:
        return None
    try:
        data = payload(result)
    except (ValueError, json.JSONDecodeError) as exc:
        report.add("L1", name, "FAIL", f"unparseable result ({exc}): {text_of(result)[:200]!r}")
        return None
    if data.get("ok") is not True:
        report.add("L1", name, "FAIL", f"ok={data.get('ok')!r} for a valid request: "
                   f"{data.get('error_code')} {str(data.get('details'))[:200]}")
        return None
    return data


def check_single_point(call: Caller, report: Report, ref: Psi4Ref) -> None:
    cases = (
        ("H2O HF/STO-3G", H2O_START, "HF", "sto-3g", 0, 1),
        ("H2O B3LYP-D3(BJ)/def2-SVP", H2O_START, "B3LYP-D3(BJ)", "def2-svp", 0, 1),
        ("OH UHF/6-31G doublet", OH_RADICAL, "HF", "6-31g", 0, 2),
        ("H2O MP2/cc-pVDZ", H2O_START, "MP2", "cc-pvdz", 0, 1),
    )
    field_report: dict[str, str] = {}
    for label, geom, method, basis, charge, mult in cases:
        name = f"single_point[{label}]"
        data = tool_payload(call, report, name, "single_point",
                            {"geometry_xyz": geom, "method": method, "basis": basis,
                             "charge": charge, "multiplicity": mult})
        if data is None:
            continue
        result = data.get("result", {})
        reference = ref.scf(geom, method, basis, charge, mult)
        got = value_of(result.get("energy"))
        diff = abs(got - reference["energy"]) if got is not None else math.inf
        report.add("L1", name, "PASS" if diff <= ENERGY_TOL else "FAIL",
                   f"server={got} reference={reference['energy']:.10f} |diff|={diff:.1e} Eh",
                   server_value=got, reference_value=reference["energy"])
        if method == "MP2":
            continue     # gap/dipole/iterations describe the SCF reference; checked on the SCF cases
        for field, ref_value in (("n_basis_functions", reference["nbf"]), ("n_iterations", None),
                                 ("homo_lumo_gap", reference["gap_eV"]), ("dipole", reference["dipole_debye"])):
            value = value_of(result.get(field))
            if value is None:
                field_report.setdefault(field, "null")
            elif ref_value is not None and abs(value - ref_value) > FIELD_REL_TOL * max(1.0, abs(ref_value)):
                report.add("L1", f"{name}[{field}]", "FAIL", f"server={value} reference={ref_value}")
                field_report[field] = "wrong"
            else:
                field_report.setdefault(field, "ok")
    nulls = sorted(f for f, state in field_report.items() if state == "null")
    if nulls:
        report.add("L1", "single_point[documented fields]", "WARN",
                   f"{nulls} always null: the server reads them from psi4.core.get_active_wavefunction(), "
                   "which psi4 1.11 does not have (upstream's own tests monkeypatch it), and swallows the "
                   "AttributeError — only the energy is real", null_fields=nulls)
    elif field_report:
        report.add("L1", "single_point[documented fields]", "PASS", "n_basis/gap/dipole match the reference")

    name = "single_point[invalid multiplicity]"
    check_rejected(call, name, "single_point", {"geometry_xyz": H2O_START, "method": "HF", "basis": "sto-3g",
                                                "multiplicity": 2, **COMMON},
                   in_band=_invalid_multiplicity,
                   on_accept=lambda result: ("FAIL", "closed-shell water accepted as a doublet: "
                                                     f"{text_of(result)[:200]!r}"))


def _invalid_multiplicity(result: dict) -> str | None:
    try:
        data = payload(result)
    except (ValueError, json.JSONDecodeError):
        return None
    if data.get("ok") is False and data.get("error_code") == "INVALID_MULTIPLICITY":
        return "ok=false, INVALID_MULTIPLICITY"
    return None


def check_optimize(call: Caller, report: Report, ref: Psi4Ref) -> Atoms | None:
    """HF/STO-3G water from a distorted start; returns the server's minimum for the frequency checks."""
    name = "optimize[H2O HF/STO-3G tight]"
    data = tool_payload(call, report, name, "optimize",
                        {"geometry_xyz": H2O_START, "method": "HF", "basis": "sto-3g", "convergence": "tight"})
    if data is None:
        return None
    result = data.get("result", {})
    try:
        atoms = parse_geometry(result.get("optimized_geometry_xyz") or "")
    except ValueError as exc:
        report.add("L1", name, "FAIL", f"no optimized geometry: {exc}")
        return None
    ref_energy, ref_atoms = ref.optimize(H2O_START, "HF", "sto-3g", "gau_tight")
    got = value_of(result.get("final_energy"))
    e_diff = abs(got - ref_energy) if got is not None else math.inf
    d_diff = max_abs_diff(sorted_distances(atoms), sorted_distances(ref_atoms))
    ok = e_diff <= OPT_ENERGY_TOL and d_diff <= OPT_DIST_TOL and result.get("converged") is True
    report.add("L1", name, "PASS" if ok else "FAIL",
               f"E={got} reference={ref_energy:.8f} |ΔE|={e_diff:.1e} Eh, max|Δd|={d_diff:.1e} Å, "
               f"converged={result.get('converged')}", server_value=result, reference_energy=ref_energy)
    if result.get("n_iterations") in (0, None):
        report.add("L1", "optimize[n_iterations]", "WARN",
                   f"n_iterations={result.get('n_iterations')!r} although the optimizer took several steps "
                   "(same missing get_active_wavefunction; the field is filled with 0)")
    return atoms if ok else None


def check_frequency(call: Caller, report: Report, ref: Psi4Ref, minimum: Atoms | None) -> None:
    if minimum is None:
        report.add("L1", "frequency[minimum]", "FAIL", "no verified minimum from optimize; skipped")
        return
    geom = atom_lines(minimum)
    base = {"geometry_xyz": geom, "method": "HF", "basis": "sto-3g"}
    name = "frequency[H2O HF/STO-3G minimum]"
    data = tool_payload(call, report, name, "frequency", base)
    reference = ref.frequencies(geom, "HF", "sto-3g")
    if data is not None:
        result = data.get("result", {})
        freqs = [float(f) for f in result.get("frequencies_cm_inv", [])]
        status, detail = classify_frequencies(freqs, reference["frequencies"], result.get("n_imaginary"))
        thermo = result.get("thermal_corrections") or {}
        diffs = {"zpe": abs((value_of(result.get("zpe")) or math.inf) - reference["zpe"])}
        for key in ("e_corr", "h_corr", "g_corr"):
            diffs[key] = abs((value_of(thermo.get(key)) or math.inf) - reference[key])
        worst = max(diffs, key=diffs.get)
        if status == "PASS" and diffs[worst] > THERMO_TOL:
            status = "FAIL"
        report.add("L1", name, status, f"{detail}; ν={[round(f, 2) for f in freqs]}; worst thermo "
                   f"{worst} |Δ|={diffs[worst]:.1e} Eh", server_value=result, reference_value=reference)
        ir = [float(x) for x in result.get("ir_intensities_km_per_mol", [])]
        ref_ir = [x for _, x in sorted(zip(reference["frequencies"], reference["ir"]))]
        got_ir = [x for _, x in sorted(zip(freqs, ir))]
        ir_diff = max_abs_diff(got_ir, ref_ir)
        if ir_diff <= max(0.05, 1e-3 * max(ref_ir, default=0)):
            report.add("L1", "frequency[IR intensities]", "PASS", f"max|ΔI|={ir_diff:.2g} km/mol")
        elif ir and all(x == 0.0 for x in ir) and any(x > 0.05 for x in ref_ir):
            report.add("L1", "frequency[IR intensities]", "WARN",
                       f"all zero, psi4 gives {[round(x, 2) for x in ref_ir]} km/mol (server falls back to the "
                       "log parser, which fills zeros)")
        else:
            report.add("L1", "frequency[IR intensities]", "FAIL", f"{got_ir} vs reference {ref_ir}")

    name = "frequency[temperature_K=500]"
    data = tool_payload(call, report, name, "frequency", {**base, "temperature_K": 500.0})
    if data is not None:
        result = data.get("result", {})
        got = value_of((result.get("thermal_corrections") or {}).get("g_corr"))
        hot = ref.frequencies(geom, "HF", "sto-3g", temperature=500.0)["g_corr"]
        status = classify_alternative(got, hot, reference["g_corr"], THERMO_TOL)
        detail = {"PASS": "G correction at 500 K matches the reference",
                  "WARN": "temperature_K ignored: G correction is the 298.15 K value although the result "
                          f"echoes temperature_K={result.get('temperature_K')}",
                  "FAIL": "G correction matches neither 500 K nor 298.15 K"}[status]
        report.add("L1", name, status, f"{detail} (server={got}, 500 K={hot:.8f}, 298.15 K={reference['g_corr']:.8f})")

    # First-order saddle: planar NH3, symmetric optimisation keeps D3h.
    name = "frequency[NH3 planar saddle HF/STO-3G]"
    _, saddle = ref.optimize(NH3_PLANAR_START, "HF", "sto-3g", "gau_tight", c1=False)
    saddle_geom = atom_lines(saddle)
    saddle_ref = ref.frequencies(saddle_geom, "HF", "sto-3g")
    data = tool_payload(call, report, name, "frequency", {**base, "geometry_xyz": saddle_geom})
    if data is not None:
        result = data.get("result", {})
        freqs = [float(f) for f in result.get("frequencies_cm_inv", [])]
        status, detail = classify_frequencies(freqs, saddle_ref["frequencies"], result.get("n_imaginary"))
        zpe = value_of(result.get("zpe"))
        if status != "FAIL" and zpe is not None and abs(zpe - saddle_ref["zpe"]) > THERMO_TOL:
            detail += f"; ZPE={zpe:.6f} Eh vs psi4 {saddle_ref['zpe']:.6f} (imaginary mode counted as real)"
            if status == "PASS":
                status = "FAIL"
        if not any("IMAGINARY" in str(w).upper() for w in data.get("warnings", [])) and status == "WARN":
            detail += "; warnings=[] (documented IMAGINARY_FREQUENCY warning missing)"
        report.add("L1", name, status, detail, server_value=result, reference_value=saddle_ref)


def check_tddft(call: Caller, report: Report, ref: Psi4Ref, minimum: Atoms | None) -> None:
    geom = atom_lines(minimum) if minimum else H2O_START
    cases = (("TDA singlets+triplets", 4, True, True), ("full TDDFT singlets", 3, False, False))
    for label, n_states, triplets, tda in cases:
        name = f"tddft[H2O B3LYP/def2-SVP {label} n_states={n_states}]"
        data = tool_payload(call, report, name, "tddft",
                            {"geometry_xyz": geom, "method": "B3LYP", "basis": "def2-svp",
                             "n_states": n_states, "triplets": triplets, "tda": tda})
        if data is None:
            continue
        result = data.get("result", {})
        reference = ref.excitations(geom, "B3LYP", "def2-svp", n_states, triplets, tda)
        singlets, trips = result.get("singlets") or [], result.get("triplets") or []
        de_s, df_s = compare_states(singlets, reference["singlet"])
        de_t, df_t = compare_states(trips, reference["triplet"]) if triplets else (0.0, 0.0)
        gs = value_of(result.get("ground_state_energy"))
        gs_diff = abs(gs - reference["energy"]) if gs is not None else math.inf
        counts_match = len(singlets) == len(reference["singlet"]) and len(trips) == len(reference["triplet"])
        ok = counts_match and max(de_s, de_t) <= EXC_TOL and max(df_s, df_t) <= OSC_TOL and gs_diff <= ENERGY_TOL
        report.add("L1", name, "PASS" if ok else "FAIL",
                   f"{len(singlets)} singlets / {len(trips)} triplets (reference {len(reference['singlet'])}/"
                   f"{len(reference['triplet'])}), max|ΔE|={max(de_s, de_t):.1e} eV, max|Δf|={max(df_s, df_t):.1e}, "
                   f"|ΔE_0|={gs_diff:.1e} Eh", server_value=result, reference_value=reference)
        if triplets and singlets and trips:
            dst = value_of(result.get("delta_E_ST_eV"))
            want = value_of(trips[0].get("excitation_energy")) - value_of(singlets[0].get("excitation_energy"))
            if dst is None or abs(dst - want) > 2e-4:
                report.add("L1", f"{name}[delta_E_ST]", "FAIL", f"delta_E_ST_eV={dst}, E(T1)-E(S1)={want:.4f}")
        if triplets and len(singlets) < n_states:
            report.add("L1", "tddft[n_states per spin manifold]", "WARN",
                       f"n_states={n_states} is documented 'per spin manifold' but psi4 splits it between "
                       f"spins: got {len(singlets)} singlets + {len(trips)} triplets")


def check_excited_state_opt(call: Caller, report: Report, ref: Psi4Ref, minimum: Atoms | None) -> None:
    method, basis, n_states, root = "B3LYP", "sto-3g", 2, 1
    start = atom_lines(minimum) if minimum else H2O_START
    name = f"optimize_excited_state[H2O S{root} {method}/{basis}]"
    data = tool_payload(call, report, name, "optimize_excited_state",
                        {"geometry_xyz": start, "method": method, "basis": basis, "n_states": n_states,
                         "target_state": root, "target_spin": "singlet", "convergence": "normal", "max_iter": 50})
    if data is None:
        return
    result = data.get("result", {})
    try:
        atoms = parse_geometry(result.get("optimized_geometry_xyz") or "")
    except ValueError as exc:
        report.add("L1", name, "FAIL", f"no optimized geometry: {exc}")
        return
    got = value_of(result.get("final_total_energy"))
    exc_got = value_of(result.get("excitation_energy_at_opt"))
    e_ground, e_excited = ref.excited_total(atoms, method, basis, n_states, root, "singlet")
    status = classify_alternative(got, e_excited, e_ground, OPT_ENERGY_TOL)
    omega = (e_excited - e_ground) * HARTREE2EV
    facts = (f"final_total_energy={got}; at the returned geometry E(S0)={e_ground:.8f}, "
             f"E(S{root})={e_excited:.8f} Eh; excitation_energy_at_opt={exc_got} eV vs ω={omega:.4f} eV")
    if status == "PASS":
        grad = ref.excited_gradient_norm(atoms, method, basis, n_states, root, "singlet")
        if grad > GRAD_TOL:
            status = "FAIL"
        facts += f"; |∇E(S{root})|={grad:.1e} Eh/Å"
    elif status == "WARN":
        gs_energy, gs_atoms = ref.optimize(start, method, basis, "gau")
        same = abs(got - gs_energy) <= OPT_ENERGY_TOL and \
            max_abs_diff(sorted_distances(atoms), sorted_distances(gs_atoms)) <= OPT_DIST_TOL
        if same:
            facts = ("returns the GROUND-state minimum: final_total_energy equals the independent S0 "
                     f"optimisation ({gs_energy:.8f} Eh) — FOLLOW_ROOT is not honoured by psi4's TDSCF finite-"
                     "difference optimisation; " + facts)
        else:
            status = "FAIL"
            facts = "energy equals E(S0) at the returned geometry, which is not the S0 minimum; " + facts
    report.add("L1", name, status, facts, server_value=result)
    if exc_got is not None and abs(exc_got - omega) > EXC_TOL:
        report.add("L1", f"{name}[excitation_energy_at_opt]", "WARN",
                   f"{exc_got} eV is not the S{root} excitation at the returned geometry ({omega:.4f} eV); "
                   "the server parses the first TDSCF block of the log (the starting geometry)")


def run_l1(session: Session) -> None:
    call, report, ref = session.call, session.report, session.state["ref"]
    check_single_point(call, report, ref)
    minimum = check_optimize(call, report, ref)
    check_frequency(call, report, ref, minimum)
    check_tddft(call, report, ref, minimum)
    check_excited_state_opt(call, report, ref, minimum)


def prepare(session: Session) -> None:
    refdir = session.tmp / "reference"
    refdir.mkdir()
    session.state["ref"] = Psi4Ref(refdir)


SMOKE = Smoke(
    server="psi4",
    run_l1=run_l1,
    packages=("psi4", "mcp", "numpy", "scipy", "pint", "qcelemental", "optking", "dftd3-python"),
    prepare=prepare,
)
