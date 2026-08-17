"""CLI Task Draft upload and browser-review regression tests."""

import hashlib
from pathlib import Path
import urllib.error

from click.testing import CliRunner

import ai4sci_bench.submission.task_submit as task_submit
from ai4sci_bench.cli import cli
from ai4sci_bench.submission.task_submit import (
    TaskSubmitResult,
    collect_task_files,
    submit_task,
    task_relative_name,
)


def _task_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "task_meta.yaml").write_text(
        "id: physics.cli_demo\nname: CLI Demo\ndomain: physics\n",
        encoding="utf-8",
    )
    (root / "task_eval.yaml").write_text(
        "task_id: physics.cli_demo\nevaluation:\n  scoring:\n"
        "  - {scorer: numerical, weight: 100}\n",
        encoding="utf-8",
    )
    (root / "task_submission.yaml").write_text(
        "schema_version: 1\nscientific_goal: Test the CLI upload.\n",
        encoding="utf-8",
    )
    (root / "generate_gt.py").write_text("print('gt')\n", encoding="utf-8")
    for level in ("b1", "b2", "b3", "b4"):
        (root / f"prompt_{level}.md").write_text(
            f"# {level}\n", encoding="utf-8")
    return root


def test_collect_task_files_preserves_relative_paths_and_private_manifest(tmp_path):
    task_dir = _task_dir(tmp_path / "task")
    nested = task_dir / "data" / "nested" / "input.csv"
    nested.parent.mkdir(parents=True)
    nested.write_text("x\n", encoding="utf-8")
    root_input = task_dir / "input.csv"
    root_input.write_text("root\n", encoding="utf-8")

    names = sorted(
        task_relative_name(task_dir, path)
        for path in collect_task_files(task_dir)
    )

    assert "task_submission.yaml" in names
    assert "data/nested/input.csv" in names
    assert "input.csv" in names


def test_submit_task_creates_draft_exact_sync_and_returns_review_url(
    tmp_path, monkeypatch,
):
    task_dir = _task_dir(tmp_path / "task")
    calls = []
    uploaded = []
    uploaded_hashes = {}

    def fake_request(method, url, token, *, json_body=None, timeout=60.0):
        calls.append((method, url, token, json_body))
        if url.endswith("/cli/submit"):
            return 200, {"proposal": {"id": "proposal-1"}, "reused": False}
        if url.endswith("/file-snapshot"):
            return 200, {
                "snapshot": "a" * 64,
                "files": [
                    {"file_name": name, "content_hash": uploaded_hashes[name]}
                    for name in uploaded
                ],
            }
        if url.endswith("/proposals/proposal-1"):
            return 200, {
                "completeness": {
                    "percent": 100,
                    "missing": [],
                    "stopped_at": "Review & Submit",
                },
            }
        raise AssertionError(url)

    def fake_upload(api, proposal_id, token, files, **kwargs):
        uploaded.extend(task_relative_name(task_dir, path) for path in files)
        assert kwargs["base_snapshot"] == "a" * 64
        uploaded_hashes.update({
            task_relative_name(task_dir, path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        })
        return 200, uploaded_hashes

    monkeypatch.setattr(task_submit, "_request", fake_request)
    monkeypatch.setattr(task_submit, "_upload_files", fake_upload)

    result = submit_task(task_dir, "https://portal.example/submit", "asi_pat_x")

    assert result.ok is True
    assert result.status == "draft"
    assert result.completeness_percent == 100
    assert result.missing == []
    assert result.web_url == "https://portal.example/submit/proposals/proposal-1"
    body = next(call[3] for call in calls if call[1].endswith("/cli/submit"))
    assert body["task_id"] == "physics.cli_demo"
    assert body["prefilled"] == {}
    assert set(uploaded) == {
        task_relative_name(task_dir, path) for path in collect_task_files(task_dir)
    }


def test_submit_task_rejects_same_names_with_wrong_server_hashes(tmp_path, monkeypatch):
    task_dir = _task_dir(tmp_path / "task")
    names = [task_relative_name(task_dir, path) for path in collect_task_files(task_dir)]

    def fake_request(method, url, token, *, json_body=None, timeout=60.0):
        if url.endswith("/cli/submit"):
            return 200, {"proposal": {"id": "proposal-1"}}
        if url.endswith("/file-snapshot"):
            return 200, {
                "snapshot": "a" * 64,
                "files": [
                    {"file_name": name, "content_hash": "0" * 64}
                    for name in names
                ],
            }
        raise AssertionError(url)

    monkeypatch.setattr(task_submit, "_request", fake_request)
    monkeypatch.setattr(
        task_submit,
        "_upload_files",
        lambda *args, **kwargs: (200, {name: "f" * 64 for name in names}),
    )

    result = submit_task(task_dir, "https://portal.example/submit", "asi_pat_x")

    assert result.ok is True
    assert result.files_ok is False
    assert "content mismatch" in (result.file_error or "")
    assert result.web_url == "https://portal.example/submit/proposals/proposal-1"


def test_submit_task_keeps_recoverable_draft_url_when_file_sync_fails(
    tmp_path, monkeypatch,
):
    task_dir = _task_dir(tmp_path / "task")

    def fake_request(method, url, token, *, json_body=None, timeout=60.0):
        if url.endswith("/cli/submit"):
            return 200, {"proposal": {"id": "proposal-1"}}
        if url.endswith("/file-snapshot"):
            return 200, {"snapshot": "a" * 64, "files": []}
        raise AssertionError(url)

    monkeypatch.setattr(task_submit, "_request", fake_request)
    error = urllib.error.HTTPError(
        "url", 409, "Conflict", {}, None,
    )
    error.read = lambda: b'{"detail":"Portal files changed"}'
    monkeypatch.setattr(
        task_submit,
        "_upload_files",
        lambda *args, **kwargs: (_ for _ in ()).throw(error),
    )

    result = submit_task(task_dir, "https://portal.example/submit", "asi_pat_x")

    assert result.ok is False
    assert result.proposal_id == "proposal-1"
    assert result.web_url == "https://portal.example/submit/proposals/proposal-1"
    assert "Portal files changed" in (result.error or "")


def test_submit_task_reports_auth_failure_without_masking_status(tmp_path, monkeypatch):
    task_dir = _task_dir(tmp_path / "task")
    error = urllib.error.HTTPError("url", 401, "Unauthorized", {}, None)
    error.read = lambda: b'{"detail":"Invalid token"}'
    monkeypatch.setattr(task_submit, "_request", lambda *a, **k: (_ for _ in ()).throw(error))

    result = submit_task(task_dir, "https://portal.example", "bad")

    assert result.ok is False
    assert result.status_code == 401
    assert "Invalid token" in result.error


def test_cli_task_submit_uses_saved_token_uploads_draft_and_opens_review(
    tmp_path, monkeypatch,
):
    task_dir = _task_dir(tmp_path / "task")
    opened = []
    seen = {}

    monkeypatch.setattr(
        "ai4sci_bench.auth.resolve.resolve_token",
        lambda *args, **kwargs: ("asi_pat_saved", "stored"),
    )
    monkeypatch.setattr(
        "ai4sci_bench.submission.task_submit.submit_task",
        lambda path, endpoint, token, **kwargs: (
            seen.update(path=path, endpoint=endpoint, token=token) or
            TaskSubmitResult(
                ok=True,
                proposal_id="proposal-1",
                uploaded=["task_meta.yaml", "generate_gt.py"],
                web_url="https://portal.example/submit/proposals/proposal-1",
                completeness_percent=100,
            )
        ),
    )
    monkeypatch.setattr(
        "webbrowser.open",
        lambda url, new=0: opened.append((url, new)) or True,
    )

    result = CliRunner().invoke(
        cli,
        [
            "task", "submit",
            "--task-dir", str(task_dir),
            "--endpoint", "https://portal.example/submit",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["token"] == "asi_pat_saved"
    assert "DRAFT UPLOAD COMPLETE" in result.output
    assert "final confirmation is still required" in result.output
    assert opened == [
        ("https://portal.example/submit/proposals/proposal-1", 2),
    ]


def test_cli_task_submit_missing_token_prompts_manual_login_then_resumes(
    tmp_path, monkeypatch,
):
    task_dir = _task_dir(tmp_path / "task")
    prompted = []

    monkeypatch.setattr(
        "ai4sci_bench.auth.resolve.resolve_token",
        lambda *args, **kwargs: (None, "none"),
    )
    monkeypatch.setattr("ai4sci_bench.cli._stdio_interactive", lambda: True)
    monkeypatch.setattr(
        "ai4sci_bench.cli._prompt_and_save_portal_token",
        lambda api, **kwargs: prompted.append(api) or "asi_pat_pasted",
    )
    monkeypatch.setattr(
        "ai4sci_bench.submission.task_submit.submit_task",
        lambda path, endpoint, token, **kwargs: TaskSubmitResult(
            ok=True,
            proposal_id="proposal-1",
            uploaded=["task_meta.yaml"],
            web_url="https://portal.example/submit/proposals/proposal-1",
            completeness_percent=80,
            missing=["Local testing"],
        ),
    )

    result = CliRunner().invoke(
        cli,
        ["task", "submit", "--task-dir", str(task_dir),
         "--endpoint", "https://portal.example/submit"],
        env={"ASIBENCH_NO_BROWSER": "1"},
    )

    assert result.exit_code == 0, result.output
    assert prompted == ["https://portal.example/api/v1"]
    assert "Missing: Local testing" in result.output


def test_cli_task_submit_revoked_saved_token_clears_reauthenticates_and_resumes(
    tmp_path, monkeypatch,
):
    task_dir = _task_dir(tmp_path / "task")
    cleared = []
    submitted_tokens = []

    monkeypatch.setattr(
        "ai4sci_bench.auth.resolve.resolve_token",
        lambda *args, **kwargs: ("asi_pat_revoked", "stored"),
    )
    monkeypatch.setattr("ai4sci_bench.cli._stdio_interactive", lambda: True)
    monkeypatch.setattr(
        "ai4sci_bench.auth.clear_credential",
        lambda api: cleared.append(api) or True,
    )
    monkeypatch.setattr(
        "ai4sci_bench.cli._prompt_and_save_portal_token",
        lambda api, **kwargs: "asi_pat_replacement",
    )

    def fake_submit(path, endpoint, token, **kwargs):
        submitted_tokens.append(token)
        if token == "asi_pat_revoked":
            return TaskSubmitResult(
                ok=False, status_code=401, error="HTTP 401: Invalid token",
            )
        return TaskSubmitResult(
            ok=True,
            proposal_id="proposal-1",
            uploaded=["task_meta.yaml"],
            web_url="https://portal.example/submit/proposals/proposal-1",
            completeness_percent=80,
        )

    monkeypatch.setattr(
        "ai4sci_bench.submission.task_submit.submit_task", fake_submit,
    )

    result = CliRunner().invoke(
        cli,
        ["task", "submit", "--task-dir", str(task_dir),
         "--endpoint", "https://portal.example/submit"],
        env={"ASIBENCH_NO_BROWSER": "1"},
    )

    assert result.exit_code == 0, result.output
    assert submitted_tokens == ["asi_pat_revoked", "asi_pat_replacement"]
    assert cleared == ["https://portal.example/api/v1"]
    assert "saved token is no longer valid" in result.output
    assert "DRAFT UPLOAD COMPLETE" in result.output


def test_cli_task_submit_noninteractive_without_token_fails_with_settings_url(
    tmp_path, monkeypatch,
):
    task_dir = _task_dir(tmp_path / "task")
    monkeypatch.setattr(
        "ai4sci_bench.auth.resolve.resolve_token",
        lambda *args, **kwargs: (None, "none"),
    )
    monkeypatch.setattr("ai4sci_bench.cli._stdio_interactive", lambda: False)

    result = CliRunner().invoke(
        cli,
        ["task", "submit", "--task-dir", str(task_dir),
         "--endpoint", "https://portal.example/submit"],
    )

    assert result.exit_code != 0
    assert "ASIBENCH_SUBMIT_TOKEN" in result.output
    assert "https://portal.example/submit/settings" in result.output


def test_cli_task_submit_sync_failure_prints_recoverable_draft_url(
    tmp_path, monkeypatch,
):
    task_dir = _task_dir(tmp_path / "task")
    monkeypatch.setattr(
        "ai4sci_bench.auth.resolve.resolve_token",
        lambda *args, **kwargs: ("asi_pat_saved", "stored"),
    )
    monkeypatch.setattr(
        "ai4sci_bench.submission.task_submit.submit_task",
        lambda *args, **kwargs: TaskSubmitResult(
            ok=False,
            proposal_id="proposal-1",
            web_url="https://portal.example/submit/proposals/proposal-1",
            status_code=409,
            error="HTTP 409: Portal files changed",
        ),
    )

    result = CliRunner().invoke(
        cli,
        ["task", "submit", "--task-dir", str(task_dir),
         "--endpoint", "https://portal.example/submit"],
    )

    assert result.exit_code != 0
    assert "HTTP 409: Portal files changed" in result.output
    assert "A recoverable Draft may exist here:" in result.output
    assert "https://portal.example/submit/proposals/proposal-1" in result.output


def test_cli_task_submit_help_exposes_draft_upload_options_only():
    result = CliRunner().invoke(cli, ["task", "submit", "--help"])

    assert result.exit_code == 0
    assert "--task-dir" in result.output
    assert "--endpoint" in result.output
    assert "--force-file-sync" in result.output
    assert "--submit" not in result.output
    assert "--token" not in result.output
