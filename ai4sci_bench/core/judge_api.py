"""Runtime configuration for LLM/VLM judge API endpoints.

The public task contract describes what model performs the evaluation.  This
module carries the operator-selected transport (for example TokenRouter) at
runtime so credentials never need to be written into ``task_eval.yaml``.
"""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, Mapping, cast
from urllib.parse import urlsplit

from dotenv import load_dotenv


# Public names for the three runtime settings.  ``JUDGE_API_KEY_ENV`` means
# the *name* of the process environment variable that contains the secret;
# it deliberately does not contain the secret itself.
JUDGE_API_BASE_ENV = "ASIBENCH_JUDGE_API_BASE"
JUDGE_API_KEY_ENV = "ASIBENCH_JUDGE_API_KEY_ENV"
JUDGE_API_PROTOCOL_ENV = "ASIBENCH_JUDGE_API_PROTOCOL"
VALID_JUDGE_API_PROTOCOLS = frozenset({"native", "openai"})

_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class JudgeAPIConfigurationError(ValueError):
    """Raised when a runtime Judge API override is incomplete or unsafe."""


@dataclass(frozen=True)
class JudgeAPIOverride:
    """Credential-safe runtime override for LLM and VLM judge calls.

    Only the environment-variable *name* is retained.  The secret is read at
    the point of use and is never represented by this object.
    """

    api_base: str | None = None
    api_key_env: str | None = None
    api_protocol: str | None = None

    def __post_init__(self) -> None:
        """Validate even manually constructed overrides.

        The CLI uses :func:`resolve_judge_api_override`, but this dataclass is
        also a public library entry point.  Validating here prevents callers
        from accidentally bypassing the endpoint/protocol/key coupling by
        constructing an object directly.
        """
        api_base = _clean(self.api_base, field_name="Judge API base")
        api_key_env = _clean(
            self.api_key_env,
            field_name="Judge API key environment variable",
        )
        api_protocol = _clean(self.api_protocol, field_name="Judge API protocol")
        if api_protocol is not None:
            api_protocol = api_protocol.lower()

        if api_key_env is not None and not _ENV_NAME_RE.fullmatch(api_key_env):
            raise JudgeAPIConfigurationError(
                "Invalid Judge API key environment variable name. Pass a name "
                "such as GEMINI_API_KEY, never the secret value."
            )
        if api_protocol is not None and api_protocol not in VALID_JUDGE_API_PROTOCOLS:
            valid = ", ".join(sorted(VALID_JUDGE_API_PROTOCOLS))
            raise JudgeAPIConfigurationError(
                f"Unsupported Judge API protocol {api_protocol!r}; choose one of: {valid}."
            )
        if api_base is not None:
            api_base = validate_judge_api_base(api_base)
            if api_protocol is None:
                raise JudgeAPIConfigurationError(
                    "--judge-api-protocol is required when --judge-api-base is set."
                )
            if api_key_env is None:
                raise JudgeAPIConfigurationError(
                    "--judge-api-key-env is required when --judge-api-base is set."
                )

        object.__setattr__(self, "api_base", api_base)
        object.__setattr__(self, "api_key_env", api_key_env)
        object.__setattr__(self, "api_protocol", api_protocol)

    def resolve_api_key(self, environ: Mapping[str, str] | None = None) -> str | None:
        if self.api_key_env is None:
            return None
        if environ is None:
            # Library users (as opposed to the CLI entry point) may import a
            # scorer directly.  Load the conventional .env lazily so the
            # same documented configuration works in both paths.  The default
            # ``override=False`` preserves explicitly exported shell values.
            load_dotenv()
            source = os.environ
        else:
            source = environ
        value = source.get(self.api_key_env)
        if not isinstance(value, str) or not value.strip():
            raise JudgeAPIConfigurationError(
                "The Judge API key environment variable named by "
                "--judge-api-key-env is unset or empty."
            )
        return value

    def public_metadata(self) -> dict[str, str]:
        """Return reproducibility metadata that never contains the API key."""
        metadata: dict[str, str] = {}
        if self.api_base is not None:
            metadata["api_base"] = self.api_base
        if self.api_protocol is not None:
            metadata["api_protocol"] = self.api_protocol
        if self.api_key_env is not None:
            metadata["api_key_env"] = self.api_key_env
        return metadata


_NO_ACTIVE_OVERRIDE = object()

_active_override: ContextVar[object] = ContextVar(
    "asibench_judge_api_override",
    default=_NO_ACTIVE_OVERRIDE,
)


def _clean(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise JudgeAPIConfigurationError(
            f"{field_name} must be a string, not {type(value).__name__}."
        )
    stripped = value.strip()
    return stripped or None


def validate_judge_api_base(api_base: str) -> str:
    if not isinstance(api_base, str):
        raise JudgeAPIConfigurationError(
            f"Judge API base must be a string, not {type(api_base).__name__}."
        )
    if any(character.isspace() or ord(character) < 0x20 for character in api_base):
        raise JudgeAPIConfigurationError(
            "Judge API base must not contain whitespace or control characters."
        )
    try:
        parsed = urlsplit(api_base)
        # Accessing these properties forces urllib to validate malformed ports
        # and bracketed IPv6 hosts, which ``urlsplit`` otherwise defers.
        _ = parsed.hostname
        _ = parsed.port
    except ValueError as exc:
        raise JudgeAPIConfigurationError(
            "Judge API base is not a valid HTTP(S) URL."
        ) from exc
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or not parsed.hostname:
        raise JudgeAPIConfigurationError(
            "Judge API base must be an absolute HTTP(S) URL."
        )
    if parsed.username is not None or parsed.password is not None:
        raise JudgeAPIConfigurationError(
            "Judge API base must not contain credentials; use --judge-api-key-env."
        )
    if parsed.query or parsed.fragment:
        raise JudgeAPIConfigurationError(
            "Judge API base must not contain a query string or fragment."
        )
    return api_base.rstrip("/")


# Kept private-name compatible for callers from the initial implementation.
_validate_api_base = validate_judge_api_base


def resolve_judge_api_override(
    api_base: str | None = None,
    api_key_env: str | None = None,
    api_protocol: str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> JudgeAPIOverride | None:
    """Validate and construct a runtime override.

    A key-only override is useful for native providers with a non-standard
    environment-variable name.  A custom endpoint must provide all three
    values so protocol selection and authentication are explicit.
    """
    api_base = _clean(api_base, field_name="Judge API base")
    api_key_env = _clean(api_key_env, field_name="Judge API key environment variable")
    api_protocol = _clean(api_protocol, field_name="Judge API protocol")
    if api_protocol is not None:
        api_protocol = api_protocol.lower()

    if api_base is None and api_key_env is None and api_protocol is None:
        return None

    # A protocol-only override is valid: it selects a provider's standard
    # endpoint while still allowing the operator to use a non-standard key
    # environment variable.  Only a *custom base* requires all three fields.
    override = JudgeAPIOverride(
        api_base=api_base,
        api_key_env=api_key_env,
        api_protocol=api_protocol,
    )
    # Fail before scoring or any network call when the named secret is absent.
    override.resolve_api_key(environ)
    return override


def judge_api_override_from_environment(
    environ: Mapping[str, str] | None = None,
) -> JudgeAPIOverride | None:
    if environ is None:
        load_dotenv()
        source = os.environ
    else:
        source = environ
    return resolve_judge_api_override(
        source.get(JUDGE_API_BASE_ENV),
        source.get(JUDGE_API_KEY_ENV),
        source.get(JUDGE_API_PROTOCOL_ENV),
        environ=source,
    )


def get_judge_api_override() -> JudgeAPIOverride | None:
    """Return the active scoped override, falling back to ASIBENCH_* env vars."""
    active = _active_override.get()
    if active is _NO_ACTIVE_OVERRIDE:
        return judge_api_override_from_environment()
    # ``None`` is meaningful here: a caller can explicitly suppress ambient
    # ASIBENCH_JUDGE_* settings for one scoring scope.
    return cast(JudgeAPIOverride | None, active)


@contextmanager
def use_judge_api_override(
    override: JudgeAPIOverride | None,
) -> Iterator[JudgeAPIOverride | None]:
    """Apply one override for the current context and restore it afterwards."""
    token = _active_override.set(override)
    try:
        yield override
    finally:
        _active_override.reset(token)
