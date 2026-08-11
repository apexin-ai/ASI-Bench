"""Local CLI credential store — ``~/.asibench/credentials`` (design doc §4.1).

A JSON object keyed by portal API base (so multiple environments coexist):

    {"https://portal/api/v1": {"token": "asi_pat_…", "user": {...},
                               "created_at": "2026-07-10T12:00:00Z"}}

The file is chmod 0600, like ``~/.aws/credentials`` / ``gh`` hosts.yml. It holds
the PAT minted by a device login; the browser session is unrelated (web logout
does NOT invalidate this file — only ``asibench logout`` or a Settings-page
revoke does).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from ai4sci_bench.branding import CONFIG_DIR_NAME

_FILENAME = "credentials"


def _store_path() -> Path:
    return Path(os.environ.get("ASIBENCH_CREDENTIALS_DIR",
                               str(Path.home() / CONFIG_DIR_NAME))) / _FILENAME


def _normalize(endpoint_api_base: str) -> str:
    return endpoint_api_base.rstrip("/")


def _read_all() -> dict:
    path = _store_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def load_credential(endpoint_api_base: str) -> dict | None:
    """Return {"token", "user", "created_at"} for this endpoint, or None."""
    entry = _read_all().get(_normalize(endpoint_api_base))
    if isinstance(entry, dict) and entry.get("token"):
        return entry
    return None


def save_credential(endpoint_api_base: str, token: str, user: dict | None = None) -> Path:
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _read_all()
    data[_normalize(endpoint_api_base)] = {
        "token": token,
        "user": user or {},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.chmod(path, 0o600)
    return path


def clear_credential(endpoint_api_base: str | None = None) -> bool:
    """Drop one endpoint's credential (or the whole file). True if anything was removed."""
    path = _store_path()
    if endpoint_api_base is None:
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False
    data = _read_all()
    removed = data.pop(_normalize(endpoint_api_base), None) is not None
    if removed:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.chmod(path, 0o600)
    return removed


def list_credentials() -> dict:
    """All saved entries keyed by endpoint (tokens redacted to a prefix)."""
    out = {}
    for endpoint, entry in _read_all().items():
        if isinstance(entry, dict):
            tok = entry.get("token") or ""
            out[endpoint] = {
                "token_prefix": tok[:12] + "…" if tok else "",
                "user": entry.get("user") or {},
                "created_at": entry.get("created_at"),
            }
    return out
