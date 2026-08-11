"""Device-flow login client (RFC 8628, design doc §4.2).

Talks to the portal's ``/cli/device-code`` + ``/cli/device-token`` endpoints
with stdlib ``urllib`` only (same no-new-deps rule as submission/upload.py).
The user approves the short code at the portal's ``/activate`` page in any
browser — including on a different machine, which is why this beats a
localhost-callback flow for SSH boxes.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import webbrowser
from typing import Callable


class DeviceLoginError(RuntimeError):
    """Terminal device-flow failure (expired, denied, network)."""


def _post_json(url: str, payload: dict, timeout: float = 30.0) -> tuple[int, dict]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read().decode("utf-8") or "{}")
        except (json.JSONDecodeError, OSError):
            body = {}
        return exc.code, body
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise DeviceLoginError(f"Cannot reach the portal: {reason}") from exc


def device_login(
    api_base: str,
    *,
    echo: Callable[[str], None] = print,
    open_browser: bool = True,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Run the full device flow against ``api_base`` (…/api/v1).

    Returns ``{"access_token": ..., "user": {...}}`` on success; raises
    :class:`DeviceLoginError` on expiry/denial/network failure. Ctrl-C simply
    propagates (the pending code expires server-side on its own).
    """
    api = api_base.rstrip("/")
    status, grant = _post_json(f"{api}/cli/device-code",
                               {"client_name": "asibench-cli", "scope": "submit"})
    if status != 200 or "device_code" not in grant:
        raise DeviceLoginError(
            f"Could not start a login with the portal ({status}) — check the endpoint URL.")

    uri = grant["verification_uri"]
    uri_complete = grant.get("verification_uri_complete", uri)
    echo("\nFirst-time use needs a one-time sign-in. In a browser, open:\n")
    echo(f"    {uri}")
    echo(f"    and enter the code:   {grant['user_code']}\n")
    if open_browser:
        try:
            if webbrowser.open(uri_complete):
                echo("(A browser tab should have opened — approve the code there.)")
        except Exception:  # noqa: BLE001 — headless boxes: printing is enough
            pass
    echo("Waiting for authorization (Ctrl-C to abort) ...")

    interval = max(int(grant.get("interval", 5)), 1)
    deadline = time.monotonic() + int(grant.get("expires_in", 600))
    while time.monotonic() < deadline:
        sleep(interval)
        status, body = _post_json(f"{api}/cli/device-token",
                                  {"device_code": grant["device_code"]})
        if status == 200 and body.get("access_token"):
            return body
        error = body.get("detail") or body.get("error") or ""
        if error == "authorization_pending":
            continue
        if error == "slow_down":
            interval += 5
            continue
        if error == "access_denied":
            raise DeviceLoginError("The request was denied in the browser.")
        if error == "expired_token":
            raise DeviceLoginError(
                "The code expired before it was approved — run the command again.")
        raise DeviceLoginError(f"Login failed ({status}): {error or 'unexpected response'}")
    raise DeviceLoginError("Timed out waiting for browser approval — run the command again.")
