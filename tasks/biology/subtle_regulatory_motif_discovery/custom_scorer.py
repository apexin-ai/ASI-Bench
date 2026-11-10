"""Custom scorer for biology.subtle_regulatory_motif_discovery.

Scoring contract (machine-checkable, identical at every prompt level):

  1. `results/sites.csv`: header exactly `seq_id,start`; exactly n_sequences rows in file
     order `seq_00, seq_01, ...`; `start` a plain non-negative integer in
     [0, sequence_length - motif_length]. `results/motif.txt`: one line, exactly
     motif_length characters from ACGT. Any contract violation of a file zeroes that file's
     component.
  2. The data files in the workspace must be byte-identical (sha256) to the instance.
  3. score = 70 x (fraction of sequences whose reported start equals the true site start)
           + 30 x [reported motif equals the true motif].
     Both quantities are uniquely determined by the public data (see generate_gt.py).
  4. `analysis.py` is compiled and re-executed in a scratch copy of the workspace (data/
     only, 1200 s timeout); the regenerated outputs are scored the same way and the final
     score is min(submitted, regenerated). Missing / crashing / timed-out / non-producing
     script -> 0. (Only a byte-identical copy of this task's generate_gt.py - the framework's
     GT self-check stand-in - is exempt from re-execution.)

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

from ai4sci_bench.core.scorer import Scorer, register_scorer
from ai4sci_bench.core.types import ScoreDetail

REPRO_TIMEOUT_S = 1200
SITE_WEIGHT = 70.0
MOTIF_WEIGHT = 30.0
RESULT_FILES = ("results/sites.csv", "results/motif.txt")


def _read_sites(path: Path, n: int, max_start: int) -> list[int]:
    if not path.exists():
        raise ValueError(f"missing {path.name}")
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.reader(fh) if any(c.strip() for c in r)]
    if not rows or [c.strip() for c in rows[0]] != ["seq_id", "start"]:
        raise ValueError("header must be exactly `seq_id,start`")
    body = rows[1:]
    if len(body) != n:
        raise ValueError(f"expected exactly {n} rows, got {len(body)}")
    starts = []
    for i, r in enumerate(body):
        if len(r) != 2:
            raise ValueError(f"row {i}: expected exactly two columns")
        sid, st = r[0].strip(), r[1].strip()
        if sid != f"seq_{i:02d}":
            raise ValueError(f"row {i}: seq_id must be seq_{i:02d} (file order), got `{sid}`")
        if not st.isdigit():
            raise ValueError(f"row {i}: start `{st}` is not a non-negative integer")
        v = int(st)
        if v > max_start:
            raise ValueError(f"row {i}: start {v} > {max_start}")
        starts.append(v)
    return starts


def _read_motif(path: Path, l: int) -> str:
    if not path.exists():
        raise ValueError(f"missing {path.name}")
    lines = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(lines) != 1:
        raise ValueError("motif file must contain exactly one non-empty line")
    m = lines[0]
    if len(m) != l or any(c not in "ACGT" for c in m):
        raise ValueError(f"motif must be exactly {l} characters from ACGT")
    return m


def _is_generator_copy(code_path: Path) -> bool:
    gen = Path(__file__).resolve().parent / "generate_gt.py"
    try:
        return gen.exists() and gen.read_bytes() == code_path.read_bytes()
    except OSError:
        return False


def _reproduce(pred_dir: Path, code_file: str) -> dict[str, Any]:
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
    missing = [f for f in RESULT_FILES if not (work / f).exists()]
    if missing:
        info["error"] = f"{', '.join(missing)} not produced by {code_file}"
        return info
    info["regenerated_dir"] = str(work)
    return info


@register_scorer("subtle_motif_discovery_score")
class SubtleMotifDiscoveryScorer(Scorer):
    """Site-position and motif credit, gated by data integrity and a re-execution of analysis.py."""

    def score(self, pred_dir: Path, ref_dir: Path, config: dict[str, Any]) -> ScoreDetail:
        weight = float(config.get("weight", 100.0))
        name = "subtle_motif_discovery_score"
        code_file = config.get("code_file", "analysis.py")
        execute = bool(config.get("execute_analysis", True))

        try:
            ref = json.loads((ref_dir / config.get("truth_file", "motif_ref.json")).read_text(encoding="utf-8"))
            n, L, l = int(ref["n_sequences"]), int(ref["sequence_length"]), int(ref["motif_length"])
            true_sites = [int(x) for x in ref["sites"]]
            true_motif = str(ref["motif"])
            sha = dict(ref["data_sha256"])
            if len(true_sites) != n or len(true_motif) != l:
                raise ValueError("reference size mismatch")
        except Exception as exc:  # noqa: BLE001
            return ScoreDetail(scorer_name=name, score=0.0, max_score=weight, passed=False,
                               details={"scorer_internal_error": True, "error": f"{type(exc).__name__}: {exc}"},
                               message=f"evaluator/runtime failure: {type(exc).__name__}: {exc}")

        details: dict[str, Any] = {"n_sequences": n, "motif_length": l}
        data_ok = True
        for fname, h in sha.items():
            fp = pred_dir / "data" / fname
            if not fp.exists() or hashlib.sha256(fp.read_bytes()).hexdigest() != h:
                data_ok = False
                details["data_integrity_error"] = f"data/{fname} missing or differs from the instance"
                break

        def evaluate(root: Path, tag: str) -> float:
            res: dict[str, Any] = {}
            if not data_ok:
                details[tag] = {"points": 0.0, "error": "data integrity check failed"}
                return 0.0
            pts = 0.0
            try:
                starts = _read_sites(root / RESULT_FILES[0], n, L - l)
                hits = sum(int(a == b) for a, b in zip(starts, true_sites))
                res["sites_correct"] = hits
                pts += SITE_WEIGHT * hits / n
            except ValueError as exc:
                res["sites_error"] = f"contract violation: {exc}"
            try:
                m = _read_motif(root / RESULT_FILES[1], l)
                res["motif_exact"] = m == true_motif
                res["motif_hamming"] = sum(a != b for a, b in zip(m, true_motif))
                pts += MOTIF_WEIGHT * (m == true_motif)
            except ValueError as exc:
                res["motif_error"] = f"contract violation: {exc}"
            res["points"] = pts
            details[tag] = res
            return pts

        submitted = evaluate(pred_dir, "submitted")
        final = submitted
        if execute:
            rp = _reproduce(pred_dir, code_file)
            if rp.get("skipped"):
                pass
            elif "regenerated_dir" in rp:
                regenerated = evaluate(Path(rp.pop("regenerated_dir")), "regenerated")
                rp["passed"] = True
                final = min(submitted, regenerated)
            else:
                final = 0.0
            workdir = rp.pop("workdir", None)
            if workdir:
                shutil.rmtree(workdir, ignore_errors=True)
            details["reproducibility"] = rp
            details["score_before_reproducibility_gate"] = round(submitted, 4)

        details["final_points"] = final
        return ScoreDetail(scorer_name=name, score=float(final * weight / 100.0), max_score=weight, passed=True,
                           details=details, message=f"motif discovery score {final:.2f}/100")
