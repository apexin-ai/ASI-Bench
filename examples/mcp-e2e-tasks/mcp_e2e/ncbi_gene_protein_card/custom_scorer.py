"""Scorers for the NCBI gene → protein card MCP E2E fake task.

They only compare the agent's result.json with the pre-generated reference and
never contact NCBI. Whether the values came from the MCP tools, whether the
gene record was read for the id the search returned, and whether NCBI was
reached some other way is checked separately from the run artefacts by
scripts/mcp/e2e/verify_run.py.

Numbers are compared numerically (``452`` == ``"452"``) and identifiers as
written: the task asks for the band and the accessions **as the record reports
them**, and the verifier's ``answer_from_tool`` check compares them with the
tool's own spelling, so a scorer that accepted ``np_000268.1`` would call a
submission perfect that the verifier reports as not coming from the tool.

The hard gate is structural only, like the other MCP E2E fake tasks: it asks
for the seven keys with usable types, so a wrong or unversioned identifier
costs that field's own weight instead of zeroing the instance.

Submission problems (missing/invalid result.json) are ordinary zero scores.
A missing or unreadable reference is an evaluator failure and is reported with
``scorer_internal_error: true``.
"""
from __future__ import annotations

import json
from pathlib import Path

from ai4sci_bench.core.scorer import Scorer, register_scorer
from ai4sci_bench.core.types import ScoreDetail

GENE_FIELDS = ("gene_id", "symbol", "maplocation", "chr_accession", "taxid")
PROTEIN_FIELDS = ("protein_accver", "protein_len")
INTEGER_FIELDS = ("gene_id", "taxid", "protein_len")
TEXT_FIELDS = ("symbol", "maplocation", "chr_accession", "protein_accver")


class _PredictionError(ValueError):
    pass


class _ReferenceError(ValueError):
    pass


def _integer(value, what: str) -> int:
    """A whole number, however it is spelled. Whether it is a *plausible* gene id or
    length is a value question, not a schema one: a nonsensical number scores zero for
    its field and leaves the rest of the card its credit."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise _PredictionError(f"{what} must be an integer, got {value!r}")
    try:
        number = float(str(value).strip())
    except ValueError:
        raise _PredictionError(f"{what} must be an integer, got {value!r}") from None
    if number != int(number):
        raise _PredictionError(f"{what} must be a whole number, got {value!r}")
    return int(number)


def _text(value, what: str) -> str:
    """A non-empty identifier, stripped but not re-cased (see the module docstring)."""
    if not isinstance(value, str) or not value.strip():
        raise _PredictionError(f"{what} must be a non-empty string, got {value!r}")
    return value.strip()


def _normalize(data: dict, where: str) -> dict:
    out = {field: _integer(data.get(field), f"{where}[{field!r}]") for field in INTEGER_FIELDS}
    out.update({field: _text(data.get(field), f"{where}[{field!r}]") for field in TEXT_FIELDS})
    return out


def load_prediction(path: Path) -> dict:
    if not path.is_file():
        raise _PredictionError(f"{path.name} not found")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _PredictionError(f"{path.name} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise _PredictionError(f"{path.name} must contain a JSON object")
    return _normalize(data, path.name)


def load_reference(path: Path) -> dict:
    if not path.is_file():
        raise _ReferenceError(f"reference not found: {path.name}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("not a JSON object")
        return _normalize(data, path.name)
    except (OSError, UnicodeError, json.JSONDecodeError, _PredictionError, ValueError) as exc:
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


def _per_field(name: str, weight: float, fields: tuple[str, ...], pred: dict, ref: dict) -> ScoreDetail:
    correct = {field: pred[field] == ref[field] for field in fields}
    fraction = sum(correct.values()) / len(fields)
    return ScoreDetail(scorer_name=name, score=weight * fraction, max_score=weight,
                       passed=fraction == 1.0,
                       details={"per_field_correct": correct,
                                "prediction": {field: pred[field] for field in fields},
                                "reference": {field: ref[field] for field in fields}},
                       message=f"{sum(correct.values())}/{len(fields)} fields match the reference")


@register_scorer("ncbi_e2e_schema")
class NcbiSchema(Scorer):
    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        name = "ncbi_e2e_schema"
        weight = float(config.get("weight", 1.0))
        try:
            data = load_prediction(pred_dir / config.get("pred_file", "result.json"))
        except _PredictionError as exc:
            return _zero(name, weight, str(exc))
        return ScoreDetail(scorer_name=name, score=weight, max_score=weight, passed=True,
                           details={"fields": sorted(GENE_FIELDS + PROTEIN_FIELDS),
                                    "gene_id": data["gene_id"], "protein_accver": data["protein_accver"]},
                           message="result.json carries all seven card fields")


@register_scorer("ncbi_e2e_gene_card")
class NcbiGeneCard(Scorer):
    """Equal credit per gene field: id, symbol, band, chromosome accession, taxonomy id."""

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        name = "ncbi_e2e_gene_card"
        weight = float(config.get("weight", 1.0))
        pred, ref, failure = _load_both(name, weight, pred_dir, ref_dir, config)
        if failure:
            return failure
        return _per_field(name, weight, GENE_FIELDS, pred, ref)


@register_scorer("ncbi_e2e_protein_card")
class NcbiProteinCard(Scorer):
    """Equal credit per protein field: accession with version, residue count."""

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        name = "ncbi_e2e_protein_card"
        weight = float(config.get("weight", 1.0))
        pred, ref, failure = _load_both(name, weight, pred_dir, ref_dir, config)
        if failure:
            return failure
        return _per_field(name, weight, PROTEIN_FIELDS, pred, ref)
