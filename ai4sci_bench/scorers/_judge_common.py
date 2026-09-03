"""Judge scorer public infrastructure shared by llm_judge, multimodal, and agent_judge."""

from __future__ import annotations

import os
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai4sci_bench.core.judge_api import (
    JudgeAPIConfigurationError,
    VALID_JUDGE_API_PROTOCOLS,
    get_judge_api_override,
    validate_judge_api_base,
)
from ai4sci_bench.core.logger import get_logger
from ai4sci_bench.core.types import ScoreDetail

logger = get_logger(__name__)

MAX_RETRIES = 3
BACKOFF_BASE = 2.0

_scoring_defaults_cache: dict[str, Any] | None = None


@dataclass(frozen=True)
class ResolvedJudgeAPI:
    """Fully resolved Judge transport; ``api_key`` is deliberately non-repr."""

    model: str
    api_base: str | None = None
    api_key: str | None = field(default=None, repr=False)
    api_protocol: str | None = None
    api_key_env: str | None = None

    def public_metadata(self) -> dict[str, str]:
        metadata: dict[str, str] = {}
        if self.api_base is not None:
            metadata["api_base"] = self.api_base
        if self.api_protocol is not None:
            metadata["api_protocol"] = self.api_protocol
        if self.api_key_env is not None:
            metadata["api_key_env"] = self.api_key_env
        return metadata


def get_scoring_defaults() -> dict[str, Any]:
    """Load global scoring defaults from configs/scoring.yaml.

    Returns a dict with optional keys: api_base, api_key, api_protocol.
    The file is loaded once and cached for the process lifetime.
    """
    global _scoring_defaults_cache
    if _scoring_defaults_cache is not None:
        return _scoring_defaults_cache

    config_path = Path(__file__).resolve().parents[2] / "configs" / "scoring.yaml"
    if not config_path.exists():
        _scoring_defaults_cache = {}
        return _scoring_defaults_cache

    try:
        import yaml
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        _scoring_defaults_cache = data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning("Failed to load scoring defaults from %s: %s", config_path, exc)
        _scoring_defaults_cache = {}

    return _scoring_defaults_cache


def resolve_scorer_api_params(config: dict[str, Any]) -> tuple[str | None, str | None]:
    """Resolve api_base and api_key for a scorer, with global defaults as fallback.

    Priority: runtime override > task.yaml scorer config > configs/scoring.yaml > None.

    ``api_key_env`` is intentionally not read from task YAML.  Only the
    operator-supplied runtime override may name a secret environment variable.
    """
    api_base, api_key, _api_protocol, _api_key_env = _resolve_scorer_api_parts(config)
    return api_base, api_key

SUPPORTED_MODELS: dict[str, str] = {
    "gemini-3.1-pro-preview": "gemini/gemini-3.1-pro-preview",
    "claude-opus-4-6": "anthropic/claude-opus-4-6",
    "gpt-5.4": "openai/gpt-5.4",
    "openrouter/claude-opus-4-6": "openrouter/anthropic/claude-opus-4-6",
    "openrouter/gpt-5.4": "openrouter/openai/gpt-5.4",
    "openrouter/gemini-3.1-pro-preview": "openrouter/google/gemini-3.1-pro-preview",
}

DEFAULT_MODEL = "gemini/gemini-3.1-pro-preview"

_PROVIDER_API_KEY_ENVS: dict[str, tuple[str, ...]] = {
    "gemini/": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "anthropic/": ("ANTHROPIC_API_KEY",),
    "openai/": ("OPENAI_API_KEY",),
    "openrouter/": ("OPENROUTER_API_KEY",),
}


def _resolve_scorer_api_parts(
    config: dict[str, Any],
) -> tuple[str | None, str | None, str | None, str | None]:
    defaults = get_scoring_defaults()
    override = get_judge_api_override()

    # Keep legacy direct values for compatibility.  Deliberately do not accept
    # api_key_env from a public task contract: only the operator-controlled
    # runtime override may name a host secret environment variable.
    configured_base = config.get("api_base") or defaults.get("api_base")
    configured_key = config.get("api_key") or defaults.get("api_key")
    configured_protocol = config.get("api_protocol") or defaults.get("api_protocol")

    if configured_key is not None and not isinstance(configured_key, str):
        raise JudgeAPIConfigurationError("Judge api_key must be a string.")
    if configured_protocol is not None and not isinstance(configured_protocol, str):
        raise JudgeAPIConfigurationError("Judge api_protocol must be a string.")

    api_base = (
        override.api_base
        if override is not None and override.api_base is not None
        else configured_base
    )
    if override is not None and override.api_key_env is not None:
        api_key = override.resolve_api_key()
        api_key_env = override.api_key_env
    else:
        api_key = str(configured_key) if configured_key else None
        api_key_env = None
    api_protocol = (
        override.api_protocol
        if override is not None and override.api_protocol is not None
        else configured_protocol.strip().lower() if configured_protocol else None
    )
    if api_base is not None:
        if not isinstance(api_base, str):
            raise JudgeAPIConfigurationError(
                "Judge api_base must be a string."
            )
        api_base = validate_judge_api_base(api_base.strip())

    if api_protocol is not None:
        if api_protocol not in VALID_JUDGE_API_PROTOCOLS:
            valid = ", ".join(sorted(VALID_JUDGE_API_PROTOCOLS))
            raise JudgeAPIConfigurationError(
                f"Unsupported Judge API protocol {api_protocol!r}; choose one of: {valid}."
            )

    return api_base, api_key, api_protocol, api_key_env


def _canonical_openai_compatible_model(model: str) -> str:
    """Return the model ID a generic OpenAI-compatible gateway should receive."""
    resolved = SUPPORTED_MODELS.get(model, model)
    if resolved.startswith("openrouter/"):
        return resolved[len("openrouter/"):]
    if resolved.startswith("gemini/"):
        return "google/" + resolved[len("gemini/"):]
    if resolved.startswith("hosted_vllm/"):
        return resolved[len("hosted_vllm/"):]
    if resolved.startswith("openai/"):
        return resolved[len("openai/"):]
    # LiteLLM's native Gemini provider accepts both ``gemini/<id>`` and a
    # short model name, while gateways generally expect Google's provider
    # namespace in the model field (for example ``google/gemini-3.5-flash``).
    if resolved.startswith("gemini-"):
        return "google/" + resolved
    return resolved


def resolve_model(model: str, *, api_protocol: str | None = None) -> str:
    """Resolve a short model name to a litellm model identifier."""
    if api_protocol is not None:
        api_protocol = str(api_protocol).strip().lower()
    if api_protocol == "openai":
        # Keep the protocol visible in the model identifier.  LiteLLM's
        # ``openai`` route sends the standard /chat/completions payload for
        # the nested provider IDs used by gateways such as TokenRouter.
        return "openai/" + _canonical_openai_compatible_model(model)
    if api_protocol not in {None, "native"}:
        raise JudgeAPIConfigurationError(
            f"Unsupported Judge API protocol: {api_protocol!r}."
        )
    resolved = SUPPORTED_MODELS.get(model, model)
    if resolved.startswith("google/"):
        resolved = "gemini/" + resolved[len("google/"):]
    elif resolved.startswith("gemini-"):
        resolved = "gemini/" + resolved
    return resolved


def resolve_judge_api(config: dict[str, Any]) -> ResolvedJudgeAPI:
    """Resolve model, endpoint, protocol and credential for one Judge call."""
    runtime_override = get_judge_api_override()
    api_base, api_key, api_protocol, api_key_env = _resolve_scorer_api_parts(config)
    model = resolve_model(
        str(config.get("model", DEFAULT_MODEL)),
        api_protocol=api_protocol,
    )
    # New runtime endpoint overrides are deliberately fail-closed: accepting
    # an unauthenticated endpoint here tends to produce opaque provider errors
    # after a long scoring run.  Legacy task/config values remain permissive
    # for compatibility with LiteLLM's standard provider environment lookup.
    if runtime_override is not None and runtime_override.api_base and not api_key:
        raise JudgeAPIConfigurationError(
            "A custom Judge API endpoint requires a key. Use "
            "--judge-api-key-env (recommended) instead of storing a key in the task."
        )
    if not api_key:
        # Resolve conventional provider credentials ourselves instead of
        # leaving them implicit inside LiteLLM.  This lets every error and raw
        # response use the same redaction path even when the user only exports
        # GEMINI_API_KEY (or another provider-standard variable).
        for prefix, env_names in _PROVIDER_API_KEY_ENVS.items():
            if not model.startswith(prefix):
                continue
            for env_name in env_names:
                provider_key = os.environ.get(env_name)
                if provider_key:
                    api_key = provider_key
                    api_key_env = env_name
                    break
            break
    return ResolvedJudgeAPI(
        model=model,
        api_base=api_base,
        api_key=api_key,
        api_protocol=api_protocol,
        api_key_env=api_key_env,
    )


def judge_completion_api_kwargs(
    *,
    model: str,
    api_base: str | None,
    api_key: str | None,
) -> dict[str, str]:
    """Build identical credential/endpoint kwargs for text and VLM judges."""
    kwargs: dict[str, str] = {}
    if api_key:
        kwargs["api_key"] = api_key
    elif api_base is None and model.startswith("openrouter/"):
        # Preserve the historical provider fallback for direct/private helper
        # callers.  Never apply it to an explicit endpoint: that endpoint may
        # be TokenRouter (or another gateway) while an unrelated OpenRouter
        # credential happens to exist in the process environment.
        provider_key = os.environ.get("OPENROUTER_API_KEY")
        if provider_key:
            kwargs["api_key"] = provider_key
    if api_base:
        kwargs["api_base"] = api_base
    return kwargs


def sanitize_judge_error(
    error: Exception | str,
    *,
    secrets: tuple[str | None, ...] = (),
) -> str:
    """Remove known Judge credentials before an error is logged or persisted."""
    text = str(error)
    candidates = list(secrets)
    try:
        override = get_judge_api_override()
        if override is not None:
            candidates.append(override.resolve_api_key())
    except JudgeAPIConfigurationError:
        pass
    for secret in candidates:
        if secret:
            text = text.replace(secret, "<redacted>")
    return text


def aggregate_judge_scores(
    scores: list[float | None],
    raw_responses: list[str],
    *,
    scorer_name: str,
    model: str,
    num_judges: int,
    max_score_value: float,
    weight: float,
    threshold: float,
    extra_details: dict[str, Any] | None = None,
    secrets: tuple[str | None, ...] = (),
) -> ScoreDetail:
    """Aggregate multiple judge scores into a single ScoreDetail.

    Shared by llm_judge, vlm_judge, and agent_judge.
    """
    valid_scores = [s for s in scores if s is not None]
    parse_failures = len(scores) - len(valid_scores)

    if not valid_scores:
        logger.error("All %d judge responses failed to parse", len(scores))
        median_score = 0.0
    else:
        median_score = statistics.median(valid_scores)
        if parse_failures > 0:
            logger.warning(
                "%d/%d judge responses failed to parse, "
                "median based on %d valid scores",
                parse_failures, len(scores), len(valid_scores),
            )

    if max_score_value <= 0:
        logger.error("max_score_value=%s is invalid, treating as 0 score", max_score_value)
        normalized = 0.0
    else:
        normalized = median_score / max_score_value
    final_score = normalized * weight
    passed = normalized >= threshold

    fallback_secret = (
        os.environ.get("OPENROUTER_API_KEY")
        if model.startswith("openrouter/")
        else None
    )
    redaction_secrets = (*secrets, fallback_secret)
    safe_raw_responses = [
        sanitize_judge_error(response, secrets=redaction_secrets)
        for response in raw_responses
    ]
    details: dict[str, Any] = {
        "model": model,
        "num_judges": num_judges,
        "individual_scores": [s if s is not None else "PARSE_FAILED" for s in scores],
        "raw_responses": safe_raw_responses,
        "parse_failures": parse_failures,
        "valid_judge_count": len(valid_scores),
        "median_score": median_score,
        "normalized_score": round(normalized, 4),
        "threshold": threshold,
    }
    if extra_details:
        details.update(extra_details)

    return ScoreDetail(
        scorer_name=scorer_name,
        score=round(final_score, 4),
        max_score=weight,
        passed=passed,
        details=details,
        message=f"Judge median: {median_score}/{max_score_value}",
    )


def evaluator_unavailable_result(
    *,
    scorer_name: str,
    weight: float,
    error: Exception | str,
    model: str | None = None,
    secrets: tuple[str | None, ...] = (),
    extra_details: dict[str, Any] | None = None,
) -> ScoreDetail:
    """Return a zero score that is explicitly marked as evaluator infrastructure failure."""
    error_text = sanitize_judge_error(error, secrets=secrets)
    details: dict[str, Any] = {
        "error": error_text,
        "evaluator_status": "unavailable",
        "evaluator_unavailable": True,
    }
    if model:
        details["model"] = model
    if extra_details:
        details.update(extra_details)

    return ScoreDetail(
        scorer_name=scorer_name,
        score=0.0,
        max_score=weight,
        passed=False,
        details=details,
        message=f"Evaluator unavailable: {error_text}",
    )
