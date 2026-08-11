"""Credential resolution for portal-facing commands (design doc §2)."""

from __future__ import annotations

import os
import sys
from typing import Callable

from ai4sci_bench.auth.credentials import load_credential, save_credential
from ai4sci_bench.auth.device_login import DeviceLoginError, device_login
from ai4sci_bench.branding import submit_token


def _tty() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def login_allowed(interactive: bool | None = None) -> bool:
    """Device flow only in a real TTY, and never when explicitly disabled."""
    if os.environ.get("ASIBENCH_NO_BROWSER"):
        return False
    return _tty() if interactive is None else interactive


def resolve_token(
    api_base: str,
    explicit: str | None = None,
    *,
    interactive: bool | None = None,
    echo: Callable[[str], None] = print,
) -> tuple[str | None, str]:
    """Return ``(token, source)`` following the §2 precedence.

    ``source`` is one of ``explicit`` / ``env`` / ``stored`` / ``device_login`` /
    ``none`` — callers use it to decide 401 handling (only ``stored`` credentials
    are auto-cleared and re-logged-in).
    """
    tok = submit_token(explicit)
    if tok:
        return tok, ("explicit" if explicit else "env")

    saved = load_credential(api_base)
    if saved:
        return saved["token"], "stored"

    if not login_allowed(interactive):
        return None, "none"

    try:
        grant = device_login(api_base, echo=echo,
                             open_browser=not os.environ.get("ASIBENCH_NO_BROWSER"))
    except DeviceLoginError as exc:
        echo(f"Login failed: {exc}")
        return None, "none"
    user = grant.get("user") or {}
    path = save_credential(api_base, grant["access_token"], user)
    who = user.get("email") or user.get("display_name") or "you"
    echo(f"✅ Signed in as {who} — credential saved to {path}")
    return grant["access_token"], "device_login"
