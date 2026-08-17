"""CLI authentication: credential resolution + device-flow login.

Resolution order for any command that talks to the portal
(design doc cli_device_login_design.md §2):

1. ``--token`` (explicit argument)
2. ``ASIBENCH_SUBMIT_TOKEN`` environment variable (CI / headless)
3. ``~/.asibench/credentials`` (saved by a previous device login)
4. Nothing → interactive TTY starts a device-flow login; non-interactive
   callers fail loud instead (never hang waiting for a browser).
"""

from ai4sci_bench.auth.credentials import (
    clear_credential,
    load_credential,
    save_credential,
)
from ai4sci_bench.auth.device_login import DeviceLoginError, device_login
from ai4sci_bench.auth.resolve import resolve_token
from ai4sci_bench.auth.token_login import TokenLoginError, validate_portal_token

__all__ = [
    "clear_credential",
    "load_credential",
    "save_credential",
    "DeviceLoginError",
    "device_login",
    "resolve_token",
    "TokenLoginError",
    "validate_portal_token",
]
