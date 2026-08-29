"""Upload one locally-authored Task as a Portal Draft for browser confirmation."""

from __future__ import annotations

import json
import hashlib
import mimetypes
import re
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_ALLOWED_EXTENSIONS = frozenset({
    ".py", ".md", ".yaml", ".yml", ".txt", ".npy", ".dat", ".csv", ".json",
})
_IGNORED_PARTS = frozenset({
    ".git", ".asibench", ".ai4sci-bench", "__pycache__", "node_modules",
    "venv", ".venv", "results", "instances", "reference",
})
_TASK_ID_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)?$")
_MAX_FILE_BYTES = 10 * 1024 * 1024
_MAX_TASK_FILES = 1000
_MAX_TASK_BYTES = 100 * 1024 * 1024


@dataclass
class TaskSubmitResult:
    ok: bool
    proposal_id: str | None = None
    web_url: str | None = None
    uploaded: list[str] = field(default_factory=list)
    status: str = "draft"
    status_code: int | None = None
    files_ok: bool = True
    file_error: str | None = None
    reused: bool = False
    completeness_percent: int | None = None
    missing: list[str] = field(default_factory=list)
    error: str | None = None


def task_relative_name(task_dir: Path, path: Path) -> str:
    """Return a safe NFC-normalized POSIX path below *task_dir*."""
    try:
        relative = path.relative_to(task_dir)
    except ValueError as exc:
        raise ValueError(f"Task file is outside the selected directory: {path}") from exc
    name = unicodedata.normalize("NFC", relative.as_posix())
    if not name or len(name) > 255:
        raise ValueError(f"Task-relative path must contain 1-255 characters: {name!r}")
    if any(ord(char) < 32 or ord(char) == 127 for char in name):
        raise ValueError(f"Task-relative path contains control characters: {name!r}")
    if "\\" in name or any(part in {"", ".", ".."} for part in name.split("/")):
        raise ValueError(f"Task-relative path is not portable: {name!r}")
    return name


def _nested_files(task_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(task_dir.rglob("*")):
        relative = path.relative_to(task_dir)
        if any(part.startswith(".") or part in _IGNORED_PARTS for part in relative.parts):
            continue
        name = task_relative_name(task_dir, path)
        if path.is_symlink():
            raise ValueError(f"Task files may not be symlinks: {name}")
        if not path.is_file():
            continue
        if path.suffix.lower() not in _ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Task file type {path.suffix.lower() or '(none)'} is not supported: {name}",
            )
        files.append(path)
    return files


def collect_task_files(task_dir: str | Path) -> list[Path]:
    """Collect and preflight the exact local Task snapshot sent to Portal."""
    root = Path(task_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"Task directory not found: {root}")
    files = _nested_files(root)
    names: dict[str, Path] = {}
    total = 0
    for path in files:
        if path.is_symlink():
            raise ValueError(f"Task definitions may not be symlinks: {path.name}")
        name = task_relative_name(root, path)
        if name in names:
            raise ValueError(f"Task paths collide after normalization: {names[name]} and {path}")
        names[name] = path
        size = path.stat().st_size
        if size > _MAX_FILE_BYTES:
            raise ValueError(f"Task file exceeds {_MAX_FILE_BYTES} bytes: {name}")
        total += size
    if len(files) > _MAX_TASK_FILES:
        raise ValueError(f"A Task may contain at most {_MAX_TASK_FILES} uploaded files")
    if total > _MAX_TASK_BYTES:
        raise ValueError(f"Task files may total at most {_MAX_TASK_BYTES} bytes")
    return files


def _request(
    method: str,
    url: str,
    token: str,
    *,
    json_body=None,
    timeout: float = 60.0,
) -> tuple[int, dict]:
    headers = {"Authorization": f"Bearer {token}"}
    data = None
    if json_body is not None:
        data = json.dumps(json_body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8", "replace")
        parsed = json.loads(body) if body.strip() else {}
        return response.status, parsed if isinstance(parsed, dict) else {}


def _upload_files(
    api_base: str,
    proposal_id: str,
    token: str,
    files: list[Path],
    *,
    task_dir: Path,
    base_snapshot: str,
    force_file_sync: bool = False,
    timeout: float = 300.0,
) -> tuple[int, dict[str, str]]:
    boundary = "----asibench" + uuid.uuid4().hex
    chunks: list[bytes] = []
    uploaded_hashes: dict[str, str] = {}
    for path in files:
        name = task_relative_name(task_dir, path)
        quoted = name.replace("\\", "\\\\").replace('"', '\\"')
        content_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
        data = path.read_bytes()
        uploaded_hashes[name] = hashlib.sha256(data).hexdigest()
        chunks.extend([
            f"--{boundary}\r\n".encode(),
            (
                f'Content-Disposition: form-data; name="files"; filename="{quoted}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode(),
            data,
            b"\r\n",
        ])
    chunks.append(f"--{boundary}--\r\n".encode())
    body = b"".join(chunks)
    query = urllib.parse.urlencode({
        "base_snapshot": base_snapshot,
        "force": str(force_file_sync).lower(),
    })
    request = urllib.request.Request(
        f"{api_base}/proposals/{proposal_id}/files?{query}",
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
        method="PUT",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, uploaded_hashes


def _error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", "replace")[:500]
        parsed = json.loads(body)
        if isinstance(parsed, dict) and parsed.get("detail"):
            return str(parsed["detail"])
        return body
    except Exception:
        return str(exc.reason or exc)


def _metadata_path(task_dir: Path) -> Path | None:
    for name in ("task_meta.yaml", "task.yaml", "task.yml"):
        path = task_dir / name
        if path.is_file():
            return path
    return None


def validate_task_os_evidence(task_dir: str | Path) -> str | None:
    """Return an error unless Task author evidence records OS-sandbox trials."""
    task_dir = Path(task_dir).resolve()
    path = task_dir / "task_submission.yaml"
    if not path.is_file():
        return (
            "task_submission.yaml is required and must record local testing "
            "performed with `asibench difficulty-check --sandbox os`."
        )
    try:
        evidence = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        return f"Could not read task_submission.yaml: {exc}"
    if not isinstance(evidence, dict):
        return "task_submission.yaml must contain a YAML mapping"
    if evidence.get("local_testing_done") is not True:
        return (
            "task_submission.yaml must set local_testing_done: true after an "
            "OS-sandbox difficulty check."
        )
    results = evidence.get("local_test_results")
    if not isinstance(results, list) or not results:
        return (
            "task_submission.yaml must include at least one local_test_results "
            "entry produced with `--sandbox os`."
        )
    for index, result in enumerate(results):
        if not isinstance(result, dict) or result.get("sandbox") != "os":
            return (
                f"task_submission.yaml local_test_results[{index}] must set "
                "sandbox: os; Task contribution evidence must be produced with "
                "`asibench difficulty-check --sandbox os`."
            )
    return None


def submit_task(
    task_dir: str | Path,
    endpoint: str,
    token: str,
    *,
    force_file_sync: bool = False,
) -> TaskSubmitResult:
    """Create/update a Draft, exact-sync files, and return its browser review URL."""
    from ai4sci_bench.submission.endpoints import (
        portal_api_base,
        task_proposal_url,
    )

    root = Path(task_dir).resolve()
    metadata_path = _metadata_path(root)
    if metadata_path is None:
        return TaskSubmitResult(ok=False, error=f"task_meta.yaml or task.yaml not found in {root}")
    try:
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        return TaskSubmitResult(ok=False, error=f"Could not read {metadata_path.name}: {exc}")
    if not isinstance(metadata, dict):
        return TaskSubmitResult(ok=False, error=f"{metadata_path.name} must contain a YAML mapping")
    task_id = str(metadata.get("id") or metadata.get("task_id") or "").strip()
    if not _TASK_ID_RE.fullmatch(task_id):
        return TaskSubmitResult(ok=False, error=f"Invalid or missing stable Task id: {task_id!r}")
    evidence_error = validate_task_os_evidence(root)
    if evidence_error is not None:
        return TaskSubmitResult(ok=False, error=evidence_error)
    try:
        files = collect_task_files(root)
    except ValueError as exc:
        return TaskSubmitResult(ok=False, error=str(exc))
    api = portal_api_base(endpoint)
    if api is None:
        return TaskSubmitResult(ok=False, error="The endpoint is not a recognized ASI-Bench Portal URL")

    proposal_id: str | None = None
    web_url: str | None = None
    try:
        status, response = _request(
            "POST",
            f"{api}/cli/submit",
            token,
            json_body={
                "task_name": str(metadata.get("name") or task_id or root.name)[:80],
                "task_id": task_id,
                "domain": str(metadata.get("domain") or "other")[:50],
                "subdomain": metadata.get("subdomain"),
                "prefilled": {},
            },
        )
        proposal = response.get("proposal") or {}
        proposal_id = proposal.get("id")
        if not proposal_id:
            return TaskSubmitResult(
                ok=False, status_code=status,
                error=f"Portal did not return a Draft id: {response}",
            )
        web_url = task_proposal_url(endpoint, proposal_id)
        _, before = _request(
            "GET", f"{api}/proposals/{proposal_id}/file-snapshot", token,
        )
        snapshot = before.get("snapshot") if isinstance(before, dict) else None
        if not isinstance(snapshot, str) or len(snapshot) != 64:
            raise ValueError("Portal returned an invalid file snapshot token")
        _, uploaded_hashes = _upload_files(
            api, proposal_id, token, files,
            task_dir=root,
            base_snapshot=snapshot,
            force_file_sync=force_file_sync,
        )
        _, after = _request(
            "GET", f"{api}/proposals/{proposal_id}/file-snapshot", token,
        )
        stored_hashes = {
            str(item.get("file_name")): str(item.get("content_hash") or "")
            for item in (after.get("files") or [])
            if isinstance(item, dict) and item.get("file_name")
        }
        expected = {task_relative_name(root, path) for path in files}
        stored = set(stored_hashes)
        mismatched = sorted(
            name for name in expected & stored
            if stored_hashes[name] != uploaded_hashes.get(name)
        )
        if stored != expected or mismatched:
            missing = sorted(expected - stored)
            unexpected = sorted(stored - expected)
            details = []
            if missing:
                details.append("missing: " + ", ".join(missing))
            if unexpected:
                details.append("unexpected: " + ", ".join(unexpected))
            if mismatched:
                details.append("content mismatch: " + ", ".join(mismatched))
            return TaskSubmitResult(
                ok=True, proposal_id=proposal_id, status_code=status,
                files_ok=False, file_error="Portal snapshot differs (" + "; ".join(details) + ")",
                reused=bool(response.get("reused")),
                web_url=web_url,
            )
        _, refreshed = _request("GET", f"{api}/proposals/{proposal_id}", token)
        completeness = refreshed.get("completeness") or {}
        return TaskSubmitResult(
            ok=True,
            proposal_id=proposal_id,
            status_code=status,
            uploaded=sorted(expected),
            reused=bool(response.get("reused")),
            completeness_percent=completeness.get("percent"),
            missing=list(completeness.get("missing") or []),
            web_url=web_url,
        )
    except urllib.error.HTTPError as exc:
        return TaskSubmitResult(
            ok=False, proposal_id=proposal_id, web_url=web_url,
            status_code=exc.code,
            error=f"HTTP {exc.code}: {_error_detail(exc)}",
        )
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return TaskSubmitResult(
            ok=False, proposal_id=proposal_id, web_url=web_url,
            error=f"{type(exc).__name__}: {exc}",
        )
