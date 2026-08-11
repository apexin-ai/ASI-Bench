"""LLM-as-judge scorer — multi-model, rubric-based evaluation."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import litellm
from dotenv import load_dotenv

from ai4sci_bench.core.logger import get_logger

load_dotenv()
from ai4sci_bench.core.scorer import Scorer, register_scorer
from ai4sci_bench.core.types import ScoreDetail
from ai4sci_bench.scorers._judge_common import (
    BACKOFF_BASE,
    DEFAULT_MODEL,
    MAX_RETRIES,
    SUPPORTED_MODELS,
    aggregate_judge_scores,
    evaluator_unavailable_result,
    resolve_model,
    resolve_scorer_api_params,
)
from ai4sci_bench.scorers._parse_utils import parse_judge_json

logger = get_logger(__name__)

# Backward-compatible aliases
_MAX_RETRIES = MAX_RETRIES
_BACKOFF_BASE = BACKOFF_BASE


def _resolve_model(model: str) -> str:
    return resolve_model(model)


DEFAULT_RUBRIC = """\
Evaluate the agent's output on a scale of 0 to 10.
Criteria:
  - Correctness (40%): Does the output match the reference?
  - Completeness (30%): Are all required aspects addressed?
  - Quality (30%): Is the approach sound and well-implemented?

Return ONLY a JSON object: {"score": <0-10>, "reasoning": "<brief explanation>"}
"""

JUDGE_SYSTEM_PROMPT = """\
You are an expert scientific computing evaluator. You will be given an agent's \
output and a reference answer. Evaluate the agent's work according to the \
provided rubric. Be fair but rigorous. Return ONLY valid JSON."""


@register_scorer("llm_judge")
class LLMJudgeScorer(Scorer):
    """Use an LLM as judge for semantic evaluation."""

    def score(self, pred_dir: Path, ref_dir: Path, config: dict) -> ScoreDetail:
        scorer_name = "llm_judge"
        weight = config.get("weight", 1.0)
        model = resolve_model(config.get("model", DEFAULT_MODEL))
        rubric = config.get("rubric", DEFAULT_RUBRIC)
        num_judges = config.get("num_judges", 1)
        temperature = config.get("temperature", 0.0)
        max_tokens = config.get("max_tokens", 1024)
        max_score_value = config.get("max_score_value", 10)
        api_base, api_key = resolve_scorer_api_params(config)
        threshold = config.get("threshold", 0.5)
        max_chars = config.get("max_chars", 10000)

        if num_judges < 3:
            logger.warning(
                "num_judges=%d provides no parse-failure tolerance; consider >= 3",
                num_judges,
            )

        pred_file = config.get("pred_file")
        pred_content = self._read_content(pred_dir, pred_file, max_chars=max_chars)
        if pred_content is None:
            return ScoreDetail(
                scorer_name=scorer_name,
                score=0.0,
                max_score=weight,
                passed=False,
                details={"error": f"Prediction file not found: {pred_file}"},
                message="Prediction file not found",
            )

        ref_file = config.get("ref_file")
        ref_content = self._read_content(ref_dir, ref_file, max_chars=max_chars)
        extra_pred_contents = self._read_named_contents(
            pred_dir, config.get("extra_pred_files"), max_chars=max_chars
        )
        extra_ref_contents = self._read_named_contents(
            ref_dir, config.get("extra_ref_files"), max_chars=max_chars
        )
        judge_prompt = self._build_judge_prompt(
            rubric,
            pred_file,
            pred_content,
            ref_file,
            ref_content,
            extra_pred_contents,
            extra_ref_contents,
        )

        try:
            scores, raw_responses, _parse_failures = self._call_judges(
                model=model,
                prompt=judge_prompt,
                num_judges=num_judges,
                temperature=temperature,
                max_tokens=max_tokens,
                max_score_value=max_score_value,
                api_key=api_key,
                api_base=api_base,
            )
        except Exception as e:
            logger.error("LLM judge call failed: %s", e)
            return evaluator_unavailable_result(
                scorer_name=scorer_name,
                weight=weight,
                error=e,
                model=model,
            )

        result = aggregate_judge_scores(
            scores=scores,
            raw_responses=raw_responses,
            scorer_name=scorer_name,
            model=model,
            num_judges=num_judges,
            max_score_value=max_score_value,
            weight=weight,
            threshold=threshold,
        )

        if hasattr(self, "_judge_cost_accumulator") and self._judge_cost_accumulator.get("num_invocations", 0) > 0:
            result.details["judge_cost"] = dict(self._judge_cost_accumulator)

        logger.info(
            "LLM judge result: model=%s, judges=%d, scores=%s, median=%.2f, passed=%s",
            model, num_judges, scores, result.details["median_score"], result.passed,
        )

        return result

    def _read_content(
        self, directory: Path, filename: str | None, max_chars: int = 10000
    ) -> str | None:
        if not filename:
            return None
        filepath = directory / filename
        if not filepath.exists():
            return None
        try:
            content = filepath.read_text(encoding="utf-8")
            if len(content) > max_chars:
                logger.warning(
                    "Truncating %s from %d to %d chars for judge evaluation",
                    filepath.name, len(content), max_chars,
                )
                content = (
                    content[:max_chars]
                    + f"\n\n[... truncated, {len(content) - max_chars} chars omitted ...]"
                )
            return content
        except Exception:
            return None

    def _build_judge_prompt(
        self,
        rubric: str,
        pred_file: str | None,
        pred_content: str,
        ref_file: str | None,
        ref_content: str | None,
        extra_pred_contents: list[tuple[str, str]],
        extra_ref_contents: list[tuple[str, str]],
    ) -> str:
        pred_label = pred_file or "agent_output"
        parts = [
            f"## Rubric\n{rubric}\n",
            f"## Agent Output: {pred_label}\n```\n{pred_content}\n```\n",
        ]
        for filename, content in extra_pred_contents:
            parts.append(f"## Additional Agent Context: {filename}\n```\n{content}\n```\n")
        if ref_content:
            ref_label = ref_file or "reference_answer"
            parts.append(f"## Reference Answer: {ref_label}\n```\n{ref_content}\n```\n")
        for filename, content in extra_ref_contents:
            parts.append(f"## Additional Reference Context: {filename}\n```\n{content}\n```\n")
        parts.append(
            'Score the agent output. Return ONLY a JSON object: {"score": <number>, "reasoning": "<brief>"}'
        )
        return "\n".join(parts)

    def _read_named_contents(
        self,
        directory: Path,
        filenames: str | list[str] | None,
        *,
        max_chars: int,
    ) -> list[tuple[str, str]]:
        if not filenames:
            return []
        if isinstance(filenames, str):
            filenames = [filenames]
        contents: list[tuple[str, str]] = []
        for filename in filenames:
            content = self._read_content(directory, filename, max_chars=max_chars)
            if content is not None:
                contents.append((filename, content))
        return contents

    def _call_judges(
        self,
        model: str,
        prompt: str,
        num_judges: int,
        temperature: float,
        max_tokens: int,
        max_score_value: float = 10,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> tuple[list[float | None], list[str], int]:
        scores: list[float | None] = []
        raw_responses: list[str] = []
        parse_failures = 0
        self._judge_cost_accumulator: dict[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "num_invocations": 0,
        }

        for i in range(num_judges):
            logger.debug("Calling judge %d/%d with model %s", i + 1, num_judges, model)
            raw = self._call_with_retry(model, prompt, temperature, max_tokens, api_key=api_key, api_base=api_base)
            raw_responses.append(raw)
            score = parse_judge_json(raw, max_score=max_score_value)
            if score is None:
                parse_failures += 1
            scores.append(score)

        return scores, raw_responses, parse_failures

    def _call_with_retry(
        self,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> str:
        last_error: Exception | None = None
        extra_kwargs: dict[str, Any] = {}
        if api_key:
            extra_kwargs["api_key"] = api_key
        elif api_base:
            # Support OpenAI-compatible gateways that are authenticated with
            # OPENROUTER_API_KEY but are not addressed via LiteLLM's
            # openrouter/... provider shim.
            env_key = os.environ.get("OPENROUTER_API_KEY")
            if env_key:
                extra_kwargs["api_key"] = env_key
        elif model.startswith("openrouter/"):
            env_key = os.environ.get("OPENROUTER_API_KEY")
            if env_key:
                extra_kwargs["api_key"] = env_key
        if api_base:
            extra_kwargs["api_base"] = api_base
        for attempt in range(MAX_RETRIES):
            try:
                response = litellm.completion(
                    model=model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **extra_kwargs,
                )
                if hasattr(self, "_judge_cost_accumulator") and hasattr(response, "usage") and response.usage:
                    self._judge_cost_accumulator["input_tokens"] += getattr(response.usage, "prompt_tokens", 0) or 0
                    self._judge_cost_accumulator["output_tokens"] += getattr(response.usage, "completion_tokens", 0) or 0
                    self._judge_cost_accumulator["num_invocations"] += 1
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    wait = BACKOFF_BASE ** attempt
                    logger.warning(
                        "Judge API call attempt %d failed: %s, retrying in %.1fs",
                        attempt + 1, e, wait,
                    )
                    time.sleep(wait)
        raise last_error  # type: ignore[misc]
