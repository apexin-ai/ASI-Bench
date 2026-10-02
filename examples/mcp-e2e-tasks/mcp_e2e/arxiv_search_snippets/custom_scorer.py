"""Scorers for the arXiv search → PDF-snippets MCP E2E fake task.

They only compare the agent's result.json with the pre-generated reference and
never contact arXiv. Whether the values came from the MCP tools, whether the
snippet call used a paper returned by the search, and whether arXiv was
reached some other way is checked separately from the run artefacts by
scripts/mcp/e2e/verify_run.py.

Submission problems (missing/invalid result.json) are ordinary zero scores.
A missing or unreadable reference is an evaluator failure and is reported with
``scorer_internal_error: true``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ai4sci_bench.core.scorer import Scorer, register_scorer
from ai4sci_bench.core.types import ScoreDetail


class _PredictionError(ValueError):
    pass


class _ReferenceError(ValueError):
    pass


def normalize_id(value) -> str:
    """'1103.0291' from '1103.0291v1', 'arXiv:1103.0291' or an abs/pdf URL."""
    text = re.sub(r"^https?://(export\.)?arxiv\.org/(abs|pdf)/", "", str(value).strip())
    text = re.sub(r"\.pdf$", "", text).removeprefix("arXiv:")
    return re.sub(r"v\d+$", "", text)


def _count(value, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise _PredictionError(f"{what} must be an integer, got {value!r}")
    try:
        number = float(value)
    except ValueError:
        raise _PredictionError(f"{what} must be an integer, got {value!r}") from None
    if number != int(number) or number < 0:
        raise _PredictionError(f"{what} must be a non-negative integer, got {value!r}")
    return int(number)


def load_prediction(path: Path) -> dict:
    if not path.is_file():
        raise _PredictionError(f"{path.name} not found")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _PredictionError(f"{path.name} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise _PredictionError(f"{path.name} must contain a JSON object")
    ids = data.get("arxiv_ids")
    if not isinstance(ids, list) or not ids or not all(isinstance(i, str) and i.strip() for i in ids):
        raise _PredictionError("arxiv_ids must be a non-empty list of strings")
    selected = data.get("selected_id")
    if not isinstance(selected, str) or not selected.strip():
        raise _PredictionError("selected_id must be a non-empty string")
    counts = data.get("snippet_counts")
    if not isinstance(counts, dict) or not counts:
        raise _PredictionError("snippet_counts must be a non-empty object")
    return {"arxiv_ids": [normalize_id(i) for i in ids], "selected_id": normalize_id(selected),
            "snippet_counts": {str(k).strip().lower(): _count(v, f"snippet_counts[{k!r}]")
                               for k, v in counts.items()}}


def load_reference(path: Path) -> dict:
    if not path.is_file():
        raise _ReferenceError(f"reference not found: {path.name}")
    try:
        ref = json.loads(path.read_text(encoding="utf-8"))
        return {"arxiv_ids": [normalize_id(i) for i in ref["arxiv_ids"]],
                "selected_id": normalize_id(ref["selected_id"]),
                "snippet_counts": {str(k).lower(): int(v) for k, v in ref["snippet_counts"].items()}}
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError) as exc:
        raise _ReferenceError(f"unreadable reference: {exc}") from exc


def _zero(name: str, weight: float, message: str) -> ScoreDetail:
    return ScoreDetail(scorer_name=name, score=0.0, max_score=weight, passed=False,
                       details={"error": message}, message=message)


def _evaluator_failure(name: str, weight: float, message: str) -> ScoreDetail:
    return ScoreDetail(scorer_name=name, score=0.0, max_score=weight, passed=False,
                       details={"scorer_internal_error": True, "failure_kind": "missing_evaluator_input",
                                "error": message}, message=message)


def _load_both(name: str, weight: float, pred_dir: Path, ref_dir: Path, config: dict):
    """(prediction, reference, None) or (None, None, failure ScoreDetail)."""
    try:
        ref = load_reference(ref_dir / config.get("ref_file", "reference.json"))
    except _ReferenceError as exc:
        return None, None, _evaluator_failure(name, weight, str(exc))
    try:
        pred = load_prediction(pred_dir / config.get("pred_file", "result.json"))
    except _PredictionError as exc:
        return None, None, _zero(name, weight, str(exc))
    return pred, ref, None


@register_scorer("arxiv_e2e_schema")
class ArxivSchema(Scorer):
    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        name = "arxiv_e2e_schema"
        weight = float(config.get("weight", 1.0))
        try:
            data = load_prediction(pred_dir / config.get("pred_file", "result.json"))
        except _PredictionError as exc:
            return _zero(name, weight, str(exc))
        return ScoreDetail(scorer_name=name, score=weight, max_score=weight, passed=True,
                           details={"num_ids": len(data["arxiv_ids"]), "terms": sorted(data["snippet_counts"])},
                           message="result.json has arxiv_ids, selected_id and snippet_counts")


@register_scorer("arxiv_e2e_ids")
class ArxivIds(Scorer):
    """All-or-nothing: the full result list, in order."""

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        name = "arxiv_e2e_ids"
        weight = float(config.get("weight", 1.0))
        pred, ref, failure = _load_both(name, weight, pred_dir, ref_dir, config)
        if failure:
            return failure
        ok = pred["arxiv_ids"] == ref["arxiv_ids"]
        return ScoreDetail(scorer_name=name, score=weight if ok else 0.0, max_score=weight, passed=ok,
                           details={"pred_ids": pred["arxiv_ids"], "ref_ids": ref["arxiv_ids"]},
                           message="search results match" if ok else
                           f"{pred['arxiv_ids']} != reference {ref['arxiv_ids']}")


@register_scorer("arxiv_e2e_selected")
class ArxivSelected(Scorer):
    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        name = "arxiv_e2e_selected"
        weight = float(config.get("weight", 1.0))
        pred, ref, failure = _load_both(name, weight, pred_dir, ref_dir, config)
        if failure:
            return failure
        ok = pred["selected_id"] == ref["selected_id"]
        return ScoreDetail(scorer_name=name, score=weight if ok else 0.0, max_score=weight, passed=ok,
                           details={"pred_selected": pred["selected_id"], "ref_selected": ref["selected_id"]},
                           message=f"selected {pred['selected_id']}, reference {ref['selected_id']}")


@register_scorer("arxiv_e2e_snippet_counts")
class ArxivSnippetCounts(Scorer):
    """Equal share per reference term whose count matches exactly; unknown extra terms are ignored."""

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        name = "arxiv_e2e_snippet_counts"
        weight = float(config.get("weight", 1.0))
        pred, ref, failure = _load_both(name, weight, pred_dir, ref_dir, config)
        if failure:
            return failure
        per_term = {term: pred["snippet_counts"].get(term) == count for term, count in ref["snippet_counts"].items()}
        fraction = sum(per_term.values()) / len(per_term)
        return ScoreDetail(scorer_name=name, score=weight * fraction, max_score=weight, passed=fraction == 1.0,
                           details={"pred_counts": pred["snippet_counts"], "ref_counts": ref["snippet_counts"],
                                    "per_term_correct": per_term},
                           message=f"{sum(per_term.values())}/{len(per_term)} term counts match")
