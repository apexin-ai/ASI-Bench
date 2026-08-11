"""Upload a submission bundle to the official ASI-Bench website.

The endpoint defaults to ``https://asibench.apexin.ai`` and may be overridden
with ``--endpoint`` or ``ASIBENCH_SUBMIT_ENDPOINT`` for development deployments.

Transport uses the standard library only (``urllib``) to avoid adding a runtime
dependency: the ``.tar.gz`` is POSTed as ``application/gzip`` with an optional
``Authorization: Bearer <token>`` header.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class UploadResult:
    """Outcome of an upload attempt."""

    ok: bool
    status_code: int | None
    endpoint: str
    response_body: str = ""
    response_json: dict | None = None
    error: str | None = None
    headers: dict[str, str] = field(default_factory=dict)


def upload_bundle(
    archive_path: str | Path,
    endpoint: str,
    *,
    token: str | None = None,
    timeout: float = 300.0,
    extra_headers: dict[str, str] | None = None,
    filename: str | None = None,
) -> UploadResult:
    """POST a bundle ``.tar.gz`` to ``endpoint`` and return the outcome.

    A 2xx status is ``ok``; any other status or a network/timeout error yields
    ``ok=False`` with ``error`` populated (this function never raises for a
    failed request — the caller decides how to surface it).
    """
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        return UploadResult(
            ok=False, status_code=None, endpoint=endpoint,
            error=f"Bundle archive not found: {archive_path}",
        )

    data = archive_path.read_bytes()
    headers = {
        "Content-Type": "application/gzip",
        "X-ASIBench-Filename": filename or archive_path.name,
        "Content-Length": str(len(data)),
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if extra_headers:
        headers.update(extra_headers)

    request = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            parsed = _maybe_json(body)
            return UploadResult(
                ok=200 <= resp.status < 300,
                status_code=resp.status,
                endpoint=endpoint,
                response_body=body,
                response_json=parsed,
                headers=dict(resp.headers.items()),
            )
    except urllib.error.HTTPError as exc:  # server responded with 4xx/5xx
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        return UploadResult(
            ok=False, status_code=exc.code, endpoint=endpoint,
            response_body=body, response_json=_maybe_json(body),
            error=f"HTTP {exc.code}: {exc.reason}",
        )
    except (urllib.error.URLError, TimeoutError, OSError) as exc:  # network layer
        return UploadResult(
            ok=False, status_code=None, endpoint=endpoint,
            error=f"{type(exc).__name__}: {exc}",
        )


def _maybe_json(body: str) -> dict | None:
    try:
        parsed = json.loads(body)
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None
