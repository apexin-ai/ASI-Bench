"""Local ASI scoring implementation maintained by this adapter.

Initially copied from apexin-ai/ASI-Bench runner/orchestrator.py at
5935b5f33549e348a8505bb355d3b6f4fe4a273c. No upstream synchronization or
AST-equality contract is required; local behavior tests define the contract.
"""
from __future__ import annotations

import ast
import traceback
from pathlib import Path
from typing import Any

from ai4sci_bench.core.logger import get_logger
from ai4sci_bench.core.scorer import get_scorer
from ai4sci_bench.core.types import PromptLevel, ScoreDetail

logger = get_logger(__name__)


def _eval_param_expr(expr: str, parameters: dict[str, Any]) -> Any:
    """Safely evaluate a simple arithmetic expression against instance parameters."""
    node = ast.parse(expr, mode="eval")

    def eval_node(n: ast.AST) -> Any:
        if isinstance(n, ast.Expression):
            return eval_node(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.Name):
            if n.id not in parameters:
                raise KeyError(f"Unknown parameter in expression: {n.id}")
            return parameters[n.id]
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            value = eval_node(n.operand)
            return value if isinstance(n.op, ast.UAdd) else -value
        if isinstance(n, ast.BinOp) and isinstance(
            n.op,
            (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod),
        ):
            left = eval_node(n.left)
            right = eval_node(n.right)
            if isinstance(n.op, ast.Add):
                return left + right
            if isinstance(n.op, ast.Sub):
                return left - right
            if isinstance(n.op, ast.Mult):
                return left * right
            if isinstance(n.op, ast.Div):
                return left / right
            if isinstance(n.op, ast.FloorDiv):
                return left // right
            return left % right
        raise ValueError(f"Unsupported parameter expression: {expr}")

    return eval_node(node)


def _resolve_config_templates(value: Any, parameters: dict[str, Any]) -> Any:
    """Resolve task config values like '{n_frames+1}' against instance parameters."""
    if isinstance(value, dict):
        return {
            key: _resolve_config_templates(inner_value, parameters)
            for key, inner_value in value.items()
        }

    if isinstance(value, list):
        return [_resolve_config_templates(item, parameters) for item in value]

    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            expr_text = stripped[1:-1].strip()
            if "," in expr_text:
                return [
                    _eval_param_expr(part.strip(), parameters)
                    for part in expr_text.split(",")
                ]
            return _eval_param_expr(expr_text, parameters)

    return value


def _resolve_gate_severity(gate_cfg: dict[str, Any]) -> str:
    """Normalize gate severity with a backward-compatible default."""
    severity = str(gate_cfg.get("severity", "hard")).strip().lower()
    if severity not in {"hard", "soft"}:
        raise ValueError(f"Unsupported gate severity: {severity}")
    return severity


def _prompt_level_config_value(prompt_level: PromptLevel | str | None) -> str | None:
    """Normalize prompt level context before passing it into scorer configs."""
    if prompt_level is None:
        return None
    if isinstance(prompt_level, PromptLevel):
        return prompt_level.value
    return str(prompt_level)


def _evaluate_gates_and_scores(
    evaluation: dict[str, Any],
    pred_dir: Path,
    ref_dir: Path,
    parameters: dict[str, Any],
    *,
    prompt_level: PromptLevel | str | None = None,
) -> tuple[list[ScoreDetail], bool, int, list[ScoreDetail], float]:
    """Evaluate hard/soft gates, then scoring if hard gates pass."""
    gate_results: list[ScoreDetail] = []
    hard_gates_passed = True
    soft_gate_failures = 0
    prompt_level_value = _prompt_level_config_value(prompt_level)

    for gate_cfg in evaluation.get("gates", []):
        scorer_name = gate_cfg["scorer"]
        config = _resolve_config_templates(
            gate_cfg.get("config", {}),
            parameters,
        )
        if prompt_level_value is not None:
            config["prompt_level"] = prompt_level_value
        config["weight"] = 1.0  # gates are binary regardless of scorer mode

        scorer = get_scorer(scorer_name)
        try:
            result = scorer.score(pred_dir, ref_dir, config)
        except Exception as exc:
            logger.warning(
                "Gate scorer %s crashed on %s: %s",
                scorer_name, pred_dir, exc,
            )
            logger.debug("Traceback:\n%s", traceback.format_exc())
            result = ScoreDetail(
                scorer_name=scorer_name,
                score=0.0,
                max_score=1.0,
                passed=False,
                message=f"Scorer crashed: {type(exc).__name__}: {exc}",
                details={
                    "scorer_internal_error": True,
                    "exception_type": type(exc).__name__,
                    "traceback_tail": traceback.format_exc().splitlines()[-5:],
                },
            )
        severity = _resolve_gate_severity(gate_cfg)
        result.severity = severity
        gate_results.append(result)

        if result.passed:
            continue
        if severity == "hard":
            hard_gates_passed = False
        else:
            soft_gate_failures += 1

    score_results: list[ScoreDetail] = []
    final_score = 0.0
    hard_fail_score_mode = str(evaluation.get("hard_fail_score_mode", "zero"))

    if hard_gates_passed:
        for score_cfg in evaluation.get("scoring", []):
            scorer_name = score_cfg["scorer"]
            config = _resolve_config_templates(
                score_cfg.get("config", {}),
                parameters,
            )
            if prompt_level_value is not None:
                config["prompt_level"] = prompt_level_value
            config["weight"] = score_cfg.get("weight", 1.0)

            scorer = get_scorer(scorer_name)
            try:
                result = scorer.score(pred_dir, ref_dir, config)
            except Exception as exc:
                logger.warning(
                    "Scorer %s crashed on %s: %s",
                    scorer_name, pred_dir, exc,
                )
                logger.debug("Traceback:\n%s", traceback.format_exc())
                result = ScoreDetail(
                    scorer_name=scorer_name,
                    score=0.0,
                    max_score=config.get("weight", 1.0),
                    passed=False,
                    message=f"Scorer crashed: {type(exc).__name__}: {exc}",
                    details={
                        "scorer_internal_error": True,
                        "exception_type": type(exc).__name__,
                        "traceback_tail": traceback.format_exc().splitlines()[-5:],
                    },
                )
            score_results.append(result)
            final_score += result.score
    elif hard_fail_score_mode == "gate_score_sum":
        # Some tasks still want minimal credit for deliverable/format progress
        # even when a hard gate blocks the main weighted scoring stage.
        final_score = sum(result.score for result in gate_results)

    return gate_results, hard_gates_passed, soft_gate_failures, score_results, final_score
