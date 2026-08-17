"""Validate a manually pasted Portal PAT before storing it locally."""

from __future__ import annotations

import json
import urllib.error
import urllib.request


class TokenLoginError(RuntimeError):
    """The pasted value is malformed, revoked, or rejected by Portal."""


def validate_portal_token(
    api_base: str,
    token: str,
    *,
    timeout: float = 30.0,
) -> dict:
    """Return the authenticated user for a live PAT; never persist it here."""
    candidate = token.strip()
    if not candidate.startswith("asi_pat_") or len(candidate) < 20:
        raise TokenLoginError(
            "The pasted value is not a complete ASI-Bench token (expected asi_pat_...).",
        )
    request = urllib.request.Request(
        f"{api_base.rstrip('/')}/me",
        headers={"Authorization": f"Bearer {candidate}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", "replace")
            user = json.loads(body)
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            raise TokenLoginError(
                "Portal rejected this token. It may be incomplete, revoked, or from another environment.",
            ) from exc
        raise TokenLoginError(f"Portal token check failed (HTTP {exc.code}).") from exc
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise TokenLoginError(f"Could not validate the token with Portal: {exc}") from exc
    if not isinstance(user, dict) or not user.get("id"):
        raise TokenLoginError("Portal returned an invalid account response.")
    return user
