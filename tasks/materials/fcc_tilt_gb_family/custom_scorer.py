"""Scorer for the GB family scaling task.

Per member: the same independent LAMMPS re-evaluation (V1-V5) and numeric
tolerances as materials.fcc_tilt_gb_energy_mcp (copied here, not imported
across task directories). Family-level: the two Pearson correlations must
match both (a) what the submission's own 5 reported points imply (internal
consistency: did the agent actually compute it, not just assert one) and
(b) the independently measured reference (external correctness).
LAMMPS import failure is an evaluator failure (scorer_internal_error), never a zero.
"""
import csv
import json
import math
from pathlib import Path

import numpy as np
from ai4sci_bench.core.scorer import Scorer, register_scorer
from ai4sci_bench.core.types import ScoreDetail

KEYS = ["gb1", "gb2", "gb3", "gb4", "gb5"]
MEMBER_FILES = ["stress_strain.csv", "reference.data", "sheared.data", "top_ids.txt", "bottom_ids.txt"]
NUM_KEYS = ["a0_A", "gamma_GB_J_m2", "tau_peak_GPa", "gamma_first_drop", "n_atoms", "lx_A", "lz_A", "n_nonconverged"]
BAR_TO_GPA = 1e-4
CORR_TOL = 0.2


def number(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"Expected finite number, got {v!r}")
    return float(v)


def pearson(x, y):
    n = len(x); mx = sum(x) / n; my = sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    vx = sum((a - mx) ** 2 for a in x); vy = sum((b - my) ** 2 for b in y)
    if vx <= 0 or vy <= 0:
        raise ValueError("degenerate correlation input")
    return cov / math.sqrt(vx * vy)


def load_submission(pred_dir):
    r = json.loads((pred_dir / "result.json").read_text())
    members = r["members"]
    if not isinstance(members, list) or [m["key"] for m in members] != KEYS:
        raise ValueError("members must be exactly gb1..gb5 in order")
    for m in members:
        number(m["angle_deg"])
        for k in NUM_KEYS:
            number(m[k])
        if m["mode"] not in ("sliding", "migration"):
            raise ValueError("mode must be sliding or migration")
        if any(int(m[k]) != m[k] for k in ("n_atoms", "n_stress_drops", "n_nonconverged")):
            raise ValueError("n_atoms, n_stress_drops and n_nonconverged must be integers")
        missing = [f for f in MEMBER_FILES if not (pred_dir / m["key"] / f).exists()]
        if missing:
            raise ValueError(f"{m['key']}: missing files {missing}")
    number(r["angle_gamma_correlation"]); number(r["angle_tau_correlation"])
    return r


def load_curve(path):
    rows = list(csv.DictReader(Path(path).open()))
    if not rows or any(k not in rows[0] for k in ("step", "gamma", "tau_GPa", "converged")):
        raise ValueError("stress_strain.csv needs columns step,gamma,tau_GPa,converged")
    g = np.array([number(float(x["gamma"])) for x in rows]); t = np.array([number(float(x["tau_GPa"])) for x in rows])
    conv = np.array([str(x["converged"]).strip().lower() in ("1", "true", "t", "yes") for x in rows])
    return g, t, conv


def curve_stats(g, t, rel=0.10, absfloor=0.02):
    a = np.abs(t); rm = np.maximum.accumulate(a)
    drops = [i for i in range(1, len(a)) if a[i] < a[i - 1] - max(rel * rm[i - 1], absfloor)]
    return dict(tau_peak=float(a.max()), n_drops=len(drops), gamma_first=float(abs(g[drops[0]])) if drops else None)


def ids_of(path):
    v = [int(x) for x in Path(path).read_text().split()]
    if len(v) != len(set(v)):
        raise ValueError("duplicate ids")
    return set(v)


class LammpsEval:
    def __init__(self, potential_path, pair_style, element, mass):
        from lammps import lammps
        self.lammps, self.pot, self.ps, self.el, self.mass = lammps, str(potential_path), pair_style, element, mass

    def _open(self, data):
        import shutil, tempfile
        l = self.lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
        self._tmp = tempfile.TemporaryDirectory(prefix="gbfam_")
        staged = Path(self._tmp.name) / "structure.data"
        shutil.copyfile(data, staged)
        pot = Path(self._tmp.name) / Path(self.pot).name
        shutil.copyfile(self.pot, pot)
        self._pot_staged = str(pot)
        l.commands_string(f"units metal\natom_style atomic\nboundary p p p\natom_modify map array\nread_data {staged}")
        nt = l.extract_global("ntypes")
        l.command(f"pair_style {self.ps}")
        if self.ps.strip() == "eam":
            for t in range(1, nt + 1):
                l.command(f"pair_coeff {t} {t} {self._pot_staged}")
        else:
            l.command(f"pair_coeff * * {self._pot_staged} {' '.join([self.el] * nt)}")
        for t in range(1, nt + 1):
            l.command(f"mass {t} {self.mass}")
        l.command("thermo_style custom step pe pxx pyy pzz pxy fmax")
        return l

    def single_point(self, data, cna_cutoff=None):
        l = self._open(data)
        if cna_cutoff:
            l.command(f"compute cna all cna/atom {cna_cutoff:.4f}")
        l.command("run 0")
        nl = l.extract_global("nlocal")
        x = np.array(l.numpy.extract_atom("x")[:nl]); i = np.array(l.numpy.extract_atom("id")[:nl])
        out = dict(natoms=l.get_natoms(), pe=l.get_thermo("pe"), fmax=l.get_thermo("fmax"),
                   pxx=l.get_thermo("pxx"), pzz=l.get_thermo("pzz"), pxy=l.get_thermo("pxy"),
                   vol=l.get_thermo("vol"), lx=l.get_thermo("lx"), lz=l.get_thermo("lz"), x=x, id=i)
        if cna_cutoff:
            out["cna"] = np.array(l.numpy.extract_compute("cna", 1, 1)[:nl])
        l.close()
        return out


def gb_y(sp, window=15.0):
    m = (sp["cna"] != 1) & (np.abs(sp["x"][:, 1]) < window)
    return float(sp["x"][m, 1].mean()) if m.any() else float("nan")


def verify_member(pred_dir, m, ref, potential):
    g, t, conv = load_curve(pred_dir / "stress_strain.csv")
    checks, notes = {}, {}
    ev = LammpsEval(potential, ref["pair_style"], ref["element"], ref["mass_amu"])
    a0 = ref["a0_A"]; cna_cut = a0 * (1 / math.sqrt(2) + 1) / 2
    r0 = ev.single_point(pred_dir / "reference.data", cna_cut)
    r1 = ev.single_point(pred_dir / "sheared.data", cna_cut)
    checks["V1"] = (r0["fmax"] < 1e-4 and abs(r0["pxx"]) * BAR_TO_GPA < 0.05 and abs(r0["pzz"]) * BAR_TO_GPA < 0.05
                    and r0["natoms"] == int(m["n_atoms"]))
    vat = float(np.prod(r0["x"].max(0) - r0["x"].min(0)))
    tau_final = -r1["pxy"] * BAR_TO_GPA * r0["vol"] / vat
    checks["V2"] = abs(tau_final - t[-1]) <= max(0.05 * abs(t[-1]), 0.01)
    dy = gb_y(r1) - gb_y(r0)
    mode = "migration" if abs(dy) > 1.5 else "sliding"
    checks["V3"] = mode == m["mode"]
    top, bot = ids_of(pred_dir / "top_ids.txt"), ids_of(pred_dir / "bottom_ids.txt")
    y = dict(zip(r0["id"].tolist(), r0["x"][:, 1].tolist())); ymax, ymin = max(y.values()), min(y.values())
    n_top_true = sum(v > ymax - 3.0 for v in y.values()); n_bot_true = sum(v < ymin + 3.0 for v in y.values())
    checks["V4"] = (not (top & bot) and abs(len(top) - n_top_true) <= 0.10 * n_top_true
                    and abs(len(bot) - n_bot_true) <= 0.10 * n_bot_true
                    and all(y.get(i, -1e9) > ymax - 3.0 - 0.05 for i in top)
                    and all(y.get(i, 1e9) < ymin + 3.0 + 0.05 for i in bot))
    cs = curve_stats(g, t)
    checks["V5"] = (abs(cs["tau_peak"] - m["tau_peak_GPa"]) <= max(1e-4 * cs["tau_peak"], 1e-5)
                    and cs["n_drops"] == int(m["n_stress_drops"])
                    and cs["gamma_first"] is not None and abs(cs["gamma_first"] - m["gamma_first_drop"]) <= 1e-5
                    and len(g) == 161 and int(m["n_nonconverged"]) == int((~conv).sum()))
    checks = {k: bool(v) for k, v in checks.items()}
    verified = all(checks.values())
    cond = {
        "c1_a0": abs(m["a0_A"] - ref["a0_A"]) <= 0.005,
        "c2_gamma_GB": abs(m["gamma_GB_J_m2"] - ref["gamma_GB_J_m2"]) <= 0.003,
        "c3_tau_peak": abs(m["tau_peak_GPa"] - ref["tau_peak_GPa"]) / ref["tau_peak_GPa"] <= 0.10,
        "c4_gamma_first": abs(m["gamma_first_drop"] - ref["gamma_first_drop"]) <= 0.003,
        "c5_drops": abs(int(m["n_stress_drops"]) - int(ref["n_stress_drops"])) <= 1,
        "c6_mode": m["mode"] == ref["mode"],
        "c7_cell": (abs(m["n_atoms"] - ref["n_atoms"]) <= 0.01 * ref["n_atoms"]
                    and abs(m["lx_A"] - ref["lx_A"]) <= 0.005 * ref["lx_A"] and abs(m["lz_A"] - ref["lz_A"]) <= 0.005 * ref["lz_A"]),
        "c8_converged": int(m["n_nonconverged"]) <= int(ref.get("n_nonconverged", 0)) + 2,
        "c9_angle": abs(m["angle_deg"] - ref["angle_deg"]) <= 0.1,
    }
    cond = {k: bool(v) for k, v in cond.items()}
    # The three scientific answers carry the weight; the six bookkeeping conditions share the rest.
    score = 0.0
    if verified:
        score += 30 * cond["c2_gamma_GB"] + 30 * cond["c3_tau_peak"] + 25 * cond["c6_mode"]
        score += 15 * sum(cond[k] for k in ("c1_a0", "c4_gamma_first", "c5_drops", "c7_cell", "c8_converged", "c9_angle")) / 6
    return dict(score=float(score), verified=verified, checks=checks, conditions=cond)


def compare(pred, ref_members, ref_corr, pred_dir, potential_lookup):
    per_member = []
    for pm, rm in zip(pred["members"], ref_members):
        potential = potential_lookup(rm)
        per_member.append(verify_member(pred_dir / pm["key"], pm, rm, potential))
    member_score = sum(d["score"] for d in per_member) / len(per_member)  # up to 75 of 100 below
    # External correctness: does the submitted correlation match the independently measured one?
    ext_ok = (abs(pred["angle_gamma_correlation"] - ref_corr["angle_gamma_correlation"]) <= CORR_TOL
             and abs(pred["angle_tau_correlation"] - ref_corr["angle_tau_correlation"]) <= CORR_TOL)
    # Internal consistency: does it match what the agent's own 5 reported points imply?
    angles = [m["angle_deg"] for m in pred["members"]]
    own_gamma_corr = pearson(angles, [m["gamma_GB_J_m2"] for m in pred["members"]])
    own_tau_corr = pearson(angles, [m["tau_peak_GPa"] for m in pred["members"]])
    int_ok = (abs(pred["angle_gamma_correlation"] - own_gamma_corr) <= 0.02
             and abs(pred["angle_tau_correlation"] - own_tau_corr) <= 0.02)
    score = 0.75 * member_score + 15 * ext_ok + 10 * int_ok
    return dict(score=float(score), per_member=[{**d, "checks": None, "conditions": d["conditions"]} for d in per_member],
                member_mean=member_score, correlation_external_ok=ext_ok, correlation_internal_ok=int_ok,
                own_angle_gamma_correlation=own_gamma_corr, own_angle_tau_correlation=own_tau_corr,
                submitted_angle_gamma_correlation=pred["angle_gamma_correlation"],
                submitted_angle_tau_correlation=pred["angle_tau_correlation"],
                reference_angle_gamma_correlation=ref_corr["angle_gamma_correlation"],
                reference_angle_tau_correlation=ref_corr["angle_tau_correlation"])


@register_scorer("gb_family_schema")
class Schema(Scorer):
    def score(self, pred_dir: Path, ref_dir: Path, config: dict):
        w = float(config.get("weight", 1))
        try:
            load_submission(pred_dir)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return ScoreDetail(scorer_name="gb_family_schema", score=0, max_score=w, passed=False,
                               details={"error": str(exc)}, message=str(exc))
        return ScoreDetail(scorer_name="gb_family_schema", score=w, max_score=w, passed=True, details={}, message="Valid submission layout")


@register_scorer("gb_family_numeric")
class Numeric(Scorer):
    def score(self, pred_dir: Path, ref_dir: Path, config: dict):
        w = float(config.get("weight", 100))
        try:
            ref = json.loads((ref_dir / config.get("ref_file", "reference.json")).read_text())
            ref_members = ref["members"]

            def potential_lookup(rm):
                for d in (ref_dir, ref_dir / "potentials", pred_dir / "data", pred_dir / "data/potentials"):
                    p = Path(d) / rm.get("potential_file", "Al_zhou.eam.alloy")
                    if p.exists():
                        return p
                raise FileNotFoundError("potential file not found for member")
            import lammps  # noqa: F401
        except (OSError, ValueError, KeyError, StopIteration, ImportError, FileNotFoundError) as exc:
            return ScoreDetail(scorer_name="gb_family_numeric", score=0, max_score=w, passed=False, message=str(exc),
                               details={"scorer_internal_error": True,
                                        "failure_kind": "evaluator_unavailable" if isinstance(exc, ImportError) else "missing_evaluator_input",
                                        "error": str(exc)})
        try:
            # ref_members lack pair_style/element/mass_amu directly; attach the fixed Al potential
            # metadata (hardcoded here, not imported from contract.py: the framework loads this file
            # standalone via spec_from_file_location without adding the task directory to sys.path).
            POTENTIAL = dict(file="Al_zhou.eam.alloy", pair_style="eam/alloy", mass_amu=26.982)
            for rm in ref_members:
                rm.setdefault("pair_style", POTENTIAL["pair_style"]); rm.setdefault("element", "Al"); rm.setdefault("mass_amu", POTENTIAL["mass_amu"])
                rm.setdefault("potential_file", POTENTIAL["file"])
            d = compare(load_submission(pred_dir), ref_members,
                       dict(angle_gamma_correlation=ref["angle_gamma_correlation"], angle_tau_correlation=ref["angle_tau_correlation"]),
                       pred_dir, potential_lookup)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            return ScoreDetail(scorer_name="gb_family_numeric", score=0, max_score=w, passed=False, message=str(exc), details={"error": str(exc)})
        except Exception as exc:
            return ScoreDetail(scorer_name="gb_family_numeric", score=0, max_score=w, passed=False,
                               message=f"LAMMPS could not evaluate a submitted structure: {exc}", details={"error": str(exc)})
        passed = d["score"] >= 99.999
        return ScoreDetail(scorer_name="gb_family_numeric", score=d["score"] * w / 100, max_score=w, passed=passed,
                           details=d, message="Verified per member with independent LAMMPS single points; correlations checked internally and externally")
