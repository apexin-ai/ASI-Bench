"""Shared OS network-policy selection and host-only model endpoint resolution."""

from __future__ import annotations

import os


def validate_network_policy(policy: str) -> str:
    if policy not in {"auto", "model-only", "inherit"}:
        raise ValueError("network_policy must be auto, model-only or inherit")
    return policy


def select_network_policy(policy: str, sandbox: str, tool_mode) -> str:
    validate_network_policy(policy)
    if policy == "model-only" and sandbox != "os":
        raise ValueError("model-only requires --sandbox os")
    if policy == "auto":
        return (
            "model-only"
            if sandbox == "os"
            and getattr(tool_mode, "value", tool_mode) == "restricted"
            else "inherit"
        )
    return policy


# Known key-only providers. Other endpoints require an explicit protocol/base.
PROVIDERS = {
    "anthropic": ("anthropic", "https://api.anthropic.com", "ANTHROPIC_API_KEY"),
    "openai": ("openai_responses", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    "deepseek": ("openai", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "openrouter": ("openai", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "groq": ("openai", "https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "mistral": ("openai", "https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    "xai": ("openai", "https://api.x.ai/v1", "XAI_API_KEY"),
    "zai": ("openai", "https://api.z.ai/api/paas/v4", "ZAI_API_KEY"),
    "kimi-coding": ("anthropic", "https://api.kimi.com/coding", "KIMI_API_KEY"),
}


def model_connection(adapter, agent_type: str) -> dict:
    """Resolve trusted routing without reading/mounting ambient CLI configs.

    Returned credentials stay on the host. Only protocol/model reach the client.
    """
    if agent_type == "agy":
        raise ValueError(
            "model-only: Antigravity has no verified custom model endpoint; use a supported harness or explicit network_policy=inherit"
        )
    custom_home = next(
        (
            getattr(adapter, name, None)
            for name in ("codex_home", "pi_config_dir", "kimi_home")
            if getattr(adapter, name, None)
        ),
        None,
    )
    if custom_home:
        raise ValueError(
            "model-only does not load custom harness homes; configure api_key/api_base explicitly"
        )
    key = getattr(adapter, "api_key", None)
    if agent_type == "openhands":
        anthropic = adapter.api_protocol == "anthropic" or adapter.model.startswith(
            "anthropic/"
        )
        key = adapter._explicit_api_key or os.environ.get(
            "ANTHROPIC_API_KEY" if anthropic else "OPENAI_API_KEY"
        )
    base = getattr(adapter, "api_base", None)
    protocol = getattr(adapter, "api_protocol", None)
    model = adapter.model
    provider = getattr(adapter, "provider", None) or (
        model.split("/", 1)[0] if "/" in model else None
    )
    if agent_type == "codex":
        protocol = "openai_responses"
        key = key or os.environ.get("OPENAI_API_KEY")
        if base and not adapter.provider:
            # Existing Chat-to-Responses translation remains host-side.
            base = adapter._ensure_proxy()
            key = "sk-proxy-placeholder"
        base = base or "https://api.openai.com/v1"
    elif agent_type == "mimo_code":
        env = adapter._build_api_env()
        protocol = "openai"
        key = env.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        base = env.get("OPENAI_BASE_URL") or "https://api.xiaomimimo.com/v1"
    elif agent_type == "kimi_code":
        protocol = adapter._resolved_type or "anthropic"
        protocol = {"kimi": "openai"}.get(protocol, protocol)
        base = adapter._resolved_base or "https://api.kimi.com/coding"
        key = key or os.environ.get("MOONSHOT_API_KEY")
    elif agent_type in {"pi", "opencode", "openhands"}:
        if not base:
            default = "openai" if agent_type == "openhands" else "anthropic"
            provider = provider or default
            if provider not in PROVIDERS:
                raise ValueError(
                    "model-only requires explicit api_base and api_protocol for this provider"
                )
            protocol, base, key_env = PROVIDERS[provider]
            if agent_type == "openhands" and provider == "openai":
                protocol = "openai"
            key = key or os.environ.get(key_env)
        if (
            agent_type in {"pi", "opencode"}
            and not adapter._uses_native_provider
            and "/" in model
        ):
            model = model.split("/", 1)[1]
        if agent_type == "openhands":
            # SDK model keeps the LiteLLM protocol prefix in its config.
            protocol = protocol or (
                "anthropic" if provider == "anthropic" else "openai"
            )
    else:
        raise ValueError(
            f"model-only has no model transport for agent type {agent_type!r}"
        )
    if protocol not in {"anthropic", "openai", "openai_responses"}:
        raise ValueError(
            "model-only supports Anthropic Messages, OpenAI Chat Completions and Responses; this protocol is unsupported"
        )
    if not key:
        raise ValueError(
            "model-only requires explicit API-key authentication; OAuth login is unsupported"
        )
    return {
        "protocol": protocol,
        "base_url": base,
        "api_key": key,
        "agent_type": agent_type,
        "model": model,
        "reasoning": getattr(adapter, "reasoning", True),
    }
