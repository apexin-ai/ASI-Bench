"""Custom scorer for computer_science.planted_clique_community_recovery.

Scoring contract (all of it is machine-checkable):

  1. `results/community.csv` must follow the exact contract: header `vertex_id`,
     1..k rows, distinct integers in [0, n), strictly ascending.  Violations → 0.
  2. `data/adjacency.npy` in the workspace must be byte-identical (sha256) to the
     instance the reference was generated from; the submitted set must be a clique
     of that graph (every pair linked).  Not a clique → 0.
  3. credit = |S ∩ planted| / k   (S valid clique).  Because the planted set is the
     unique k-clique, credit 1.0 requires the exact membership.
  4. `analysis.py` is compiled and re-executed in a scratch copy of the workspace
     (data/ only, 1200 s timeout, deterministic environment); the regenerated
     `results/community.csv` is scored the same way and the final credit is
     min(submitted credit, regenerated credit).  Missing / crashing / timed-out
     script → 0.  (The GT self-check stand-in — a byte-identical copy of
     generate_gt.py — is exempt from re-execution.)

Evaluator failures (unreadable reference) set `details.scorer_internal_error`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ai4sci_bench.core.scorer import Scorer, register_scorer
from ai4sci_bench.core.types import ScoreDetail

REPRO_TIMEOUT_S = 1200


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _read_members(path: Path, n: int, k: int) -> list[int]:
    """Strict CSV contract; raises ValueError with a reason on any violation."""
    if not path.exists():
        raise ValueError(f"missing {path.name}")
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows or [c.strip() for c in rows[0]] != ["vertex_id"]:
        raise ValueError("header must be exactly `vertex_id`")
    body = rows[1:]
    if not body:
        raise ValueError("no rows")
    if len(body) > k:
        raise ValueError(f"{len(body)} rows > community_size {k}")
    members: list[int] = []
    for i, r in enumerate(body):
        if len(r) != 1:
            raise ValueError(f"row {i}: expected exactly one column")
        s = r[0].strip()
        if not s.lstrip("+").isdigit():
            raise ValueError(f"row {i}: `{s}` is not a non-negative integer")
        v = int(s)
        if not (0 <= v < n):
            raise ValueError(f"row {i}: vertex {v} outside [0, {n})")
        if members and v <= members[-1]:
            raise ValueError(f"row {i}: ids must be distinct and strictly ascending")
        members.append(v)
    return members


def _load_adjacency(path: Path, n: int, sha_expected: str) -> np.ndarray:
    if not path.exists():
        raise ValueError("data/adjacency.npy missing from workspace")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != sha_expected:
        raise ValueError("data/adjacency.npy differs from the instance the reference was generated for")
    packed = np.load(path, allow_pickle=False)
    return packed  # [n, ceil(n/8)] uint8


def _is_clique(packed: np.ndarray, members: list[int], n: int) -> tuple[bool, int]:
    """Return (is_clique, number_of_missing_pairs) using the packed rows of the members only."""
    if len(members) < 2:
        return True, 0
    rows = np.unpackbits(packed[members], axis=1)[:, :n].astype(bool)
    sub = rows[:, members]
    m = len(members)
    expected = m * (m - 1)
    present = int(sub.sum()) - int(np.diag(sub).sum())
    return present == expected, (expected - present) // 2


def _credit(members: list[int], planted: set[int]) -> float:
    return len(planted.intersection(members)) / max(len(planted), 1)


def _is_generator_copy(code_path: Path) -> bool:
    """True only when analysis.py is byte-identical to this task's own generate_gt.py.

    The framework's GT self-check places a copy of generate_gt.py as analysis.py; that
    file IS the reference, so executing it is meaningless. Nothing an agent writes can
    match the private generator byte for byte, so this cannot be used to skip the gate.
    """
    gen = Path(__file__).resolve().parent / "generate_gt.py"
    try:
        return gen.exists() and gen.read_bytes() == code_path.read_bytes()
    except OSError:
        return False


def _reproduce(pred_dir: Path, code_file: str, result_file: str) -> dict[str, Any]:
    """Execute analysis.py in a scratch copy of the workspace; return info + path of regenerated result."""
    code_path = pred_dir / code_file
    info: dict[str, Any] = {"executed": False, "passed": False}
    if not code_path.exists():
        info["error"] = f"missing {code_file}"
        return info
    if _is_generator_copy(code_path):
        info.update({"skipped": True, "passed": True, "reason": "GT self-check stand-in (generate_gt.py copy)"})
        return info
    try:
        compile(code_path.read_text(encoding="utf-8", errors="replace"), code_file, "exec")
    except SyntaxError as exc:
        info["error"] = f"{code_file} does not compile: {exc}"
        return info
    tmp = tempfile.mkdtemp(prefix="asib_repro_")
    work = Path(tmp)
    info["workdir"] = tmp
    if (pred_dir / "data").is_dir():
        shutil.copytree(pred_dir / "data", work / "data")
    shutil.copy2(code_path, work / code_file)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC", "OPENAI", "AWS", "GOOGLE"))}
    env["PYTHONHASHSEED"] = "0"
    env["MPLBACKEND"] = "Agg"
    try:
        proc = subprocess.run([sys.executable, code_file], cwd=work, env=env, capture_output=True,
                              text=True, timeout=REPRO_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        info["error"] = f"{code_file} exceeded {REPRO_TIMEOUT_S} s"
        return info
    info["executed"] = True
    info["returncode"] = proc.returncode
    if proc.returncode != 0:
        info["error"] = f"{code_file} exited with {proc.returncode}: {proc.stderr[-800:]}"
        return info
    out = work / result_file
    if not out.exists():
        info["error"] = f"{result_file} not produced by {code_file}"
        return info
    info["regenerated"] = str(out)
    return info


# ─────────────────────────────────────────────────────────────────────────────
# Scorer
# ─────────────────────────────────────────────────────────────────────────────
@register_scorer("planted_clique_community_score")
class PlantedCliqueCommunityScorer(Scorer):
    """Membership credit of a verified clique, gated by a re-execution of analysis.py."""

    def score(self, pred_dir: Path, ref_dir: Path, config: dict[str, Any]) -> ScoreDetail:
        weight = float(config.get("weight", 100.0))
        name = "planted_clique_community_score"
        result_file = config.get("result_file", "results/community.csv")
        code_file = config.get("code_file", "analysis.py")
        execute = bool(config.get("execute_analysis", True))

        # ── Reference side ──
        try:
            ref = json.loads((ref_dir / config.get("planted_file", "planted_ref.json")).read_text(encoding="utf-8"))
            n, k = int(ref["n_nodes"]), int(ref["community_size"])
            planted = {int(v) for v in ref["members"]}
            sha_expected = str(ref["adjacency_sha256"])
            if len(planted) != k:
                raise ValueError("reference membership size mismatch")
        except Exception as exc:  # noqa: BLE001
            return ScoreDetail(scorer_name=name, score=0.0, max_score=weight, passed=False,
                               details={"scorer_internal_error": True, "error": f"{type(exc).__name__}: {exc}"},
                               message=f"evaluator/runtime failure: {type(exc).__name__}: {exc}")

        details: dict[str, Any] = {"n_nodes": n, "community_size": k}

        def evaluate(csv_path: Path, tag: str) -> float:
            try:
                members = _read_members(csv_path, n, k)
            except ValueError as exc:
                details[tag] = {"credit": 0.0, "error": f"contract violation: {exc}"}
                return 0.0
            try:
                packed = _load_adjacency(pred_dir / "data" / "adjacency.npy", n, sha_expected)
                ok, missing = _is_clique(packed, members, n)
            except Exception as exc:  # noqa: BLE001
                details[tag] = {"credit": 0.0, "error": f"graph check failed: {type(exc).__name__}: {exc}"}
                return 0.0
            hits = len(planted.intersection(members))
            if not ok:
                details[tag] = {"credit": 0.0, "n_submitted": len(members), "is_clique": False,
                                "missing_pairs": missing, "planted_hits": hits,
                                "error": "submitted set is not a clique"}
                return 0.0
            c = _credit(members, planted)
            details[tag] = {"credit": c, "n_submitted": len(members), "is_clique": True, "planted_hits": hits}
            return c

        submitted = evaluate(pred_dir / result_file, "submitted")
        final = submitted

        # ── Reproducibility gate ──
        if execute:
            rp = _reproduce(pred_dir, code_file, result_file)
            if rp.get("skipped"):
                rp["passed"] = True
            elif "regenerated" in rp:
                regenerated = evaluate(Path(rp.pop("regenerated")), "regenerated")
                rp["passed"] = True
                final = min(submitted, regenerated)
            else:
                final = 0.0
            workdir = rp.pop("workdir", None)
            if workdir:
                shutil.rmtree(workdir, ignore_errors=True)
            details["reproducibility"] = rp
            details["score_before_reproducibility_gate"] = round(100.0 * submitted, 4)

        raw = 100.0 * final
        details["final_credit"] = final
        scaled = raw * weight / 100.0
        return ScoreDetail(scorer_name=name, score=float(scaled), max_score=weight, passed=True,
                           details=details, message=f"planted-community credit {final:.4f} → score {raw:.2f}/100")

