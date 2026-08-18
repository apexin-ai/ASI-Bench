"""Tests for the submission bundle builder and uploader (network-free).

Fabricates minimal produce-only results trees (a result JSON + persisted
``.outputs/`` dir, with and without private ``framework_task_info.json``), then
asserts ``build_submission`` creates a verifiable bundle. Upload is tested
against a throwaway local HTTP server so no external domain is contacted.
"""

import gzip
import json
import shutil
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from click.testing import CliRunner


from ai4sci_bench.cli import _format_file_size, cli
from ai4sci_bench.submission import build_submission, upload_bundle
from ai4sci_bench.submission.upload import UploadResult


def _make_results_dir(
    root: Path,
    *,
    produce_all: bool = True,
    include_framework_task_info: bool = True,
    seed: str = "seed42",
    task_id: str = "physics.demo_task",
) -> Path:
    """Create a fake produce-only `run` output dir with one instance."""
    results = root / "out"
    instance_id = f"{task_id}__{seed}"
    base = f"{instance_id}__b2"

    task_dir = results / task_id
    task_dir.mkdir(parents=True)

    # Persisted outputs next to the result JSON.
    outputs = task_dir / f"{base}.outputs"
    (outputs / "results").mkdir(parents=True)
    (outputs / "solution.py").write_text("print('hi')\n", encoding="utf-8")
    data_files = ["results/answer.npy"]
    if produce_all:
        (outputs / "results" / "answer.npy").write_bytes(b"\x00\x01\x02\x03")
        files_manifest = [
            {"path": "solution.py", "sha256": "x", "bytes": 12},
            {"path": "results/answer.npy", "sha256": "y", "bytes": 4},
        ]
    else:
        # Agent failed to produce the declared data output.
        files_manifest = [
            {"path": "solution.py", "sha256": "x", "bytes": 12},
            {"path": "results/answer.npy", "missing": True},
        ]

    result = {
        "result_schema_version": 1,
        "instance_id": instance_id,
        "task_id": task_id,
        "attempt": 1,
        "prompt_level": "b2",
        "agent_name": "direct_llm",
        "parameters": {"seed": 42, "n": 10},
        "final_score": 0.0,
        "status": "completed",
        "provenance": {"agent": {"agent_name": "direct_llm"}, "sandbox": {"effective_mode": "none"}},
        "agent_output": {
            "code_files": ["solution.py"],
            "data_files": data_files,
            "persisted_outputs": {"dir": f"{base}.outputs", "files": files_manifest},
        },
    }
    (task_dir / f"{base}.json").write_text(json.dumps(result), encoding="utf-8")

    # framework_task_info.json snapshot.
    inst_dir = results / "instances" / instance_id
    inst_dir.mkdir(parents=True)
    if include_framework_task_info:
        (inst_dir / "framework_task_info.json").write_text(
            json.dumps({"instance_id": instance_id, "parameters": {"seed": 42}}),
            encoding="utf-8",
        )

    (results / "run_metadata.json").write_text(json.dumps({"run": "meta"}), encoding="utf-8")
    return results


@pytest.mark.parametrize(
    ("size_bytes", "expected"),
    [
        (512, "512 bytes"),
        (10_732, "10.5 KiB"),
        (2 * 1024 * 1024, "2.0 MiB"),
    ],
)
def test_format_file_size_does_not_round_small_archives_to_zero(size_bytes, expected):
    assert _format_file_size(size_bytes) == expected

def test_submit_no_upload_builds_local_bundle(tmp_path, monkeypatch):
    results = _make_results_dir(tmp_path)
    monkeypatch.delenv("ASIBENCH_SUBMIT_ENDPOINT", raising=False)
    monkeypatch.delenv("AI4SCI_SUBMIT_ENDPOINT", raising=False)

    result = CliRunner().invoke(
        cli, ["submit", "--results-dir", str(results), "--no-upload"]
    )

    assert result.exit_code == 0, result.output
    assert "Upload skipped (--no-upload)" in result.output


def test_submit_rejects_seed31415_before_upload_or_bundle(tmp_path, monkeypatch):
    results = _make_results_dir(tmp_path, seed="seed31415")
    monkeypatch.setattr(
        "ai4sci_bench.auth.resolve_token",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("authentication must not start")
        ),
    )

    result = CliRunner().invoke(cli, ["submit", "--results-dir", str(results)])

    assert result.exit_code != 0
    assert "only accepts seed42" in result.output
    assert "seed31415" in result.output
    assert "Bundle built" not in result.output


def test_submit_defaults_to_official_site_and_requires_identity(tmp_path, monkeypatch):
    results = _make_results_dir(tmp_path)
    monkeypatch.delenv("ASIBENCH_SUBMIT_ENDPOINT", raising=False)
    monkeypatch.delenv("AI4SCI_SUBMIT_ENDPOINT", raising=False)
    monkeypatch.setattr(
        "ai4sci_bench.auth.resolve_token",
        lambda *args, **kwargs: (None, "none"),
    )

    result = CliRunner().invoke(cli, ["submit", "--results-dir", str(results)])

    assert result.exit_code != 0
    assert "asibench login" in result.output


def test_submit_uploads_to_official_site_with_authenticated_identity(
    tmp_path, monkeypatch
):
    results = _make_results_dir(tmp_path)
    captured = {}

    def fake_upload(path, endpoint, *, token=None, **kwargs):
        captured.update(endpoint=endpoint, token=token)
        return UploadResult(
            ok=True,
            status_code=201,
            endpoint=endpoint,
            response_json={
                "submission_id": "sub-official",
                "status": "draft",
                "confirm_url": "https://asibench.apexin.ai/submit/submissions/sub-official",
            },
        )

    monkeypatch.delenv("ASIBENCH_SUBMIT_ENDPOINT", raising=False)
    monkeypatch.delenv("AI4SCI_SUBMIT_ENDPOINT", raising=False)
    monkeypatch.setattr("ai4sci_bench.submission.upload_bundle", fake_upload)

    result = CliRunner().invoke(
        cli,
        ["submit", "--results-dir", str(results), "--token", "asi_pat_user"],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "endpoint": "https://asibench.apexin.ai/api/v1/submissions/bundle",
        "token": "asi_pat_user",
    }
    assert "sub-official" in result.output
    assert "click Confirm to enter the scoring queue" in result.output



class TestBuildSubmission:
    def test_seed31415_results_are_rejected(self, tmp_path):
        results = _make_results_dir(tmp_path, seed="seed31415")

        with pytest.raises(ValueError, match="only accepts seed42.*seed31415"):
            build_submission(results)

    def test_unknown_seed_results_are_rejected(self, tmp_path):
        results = _make_results_dir(tmp_path, seed="adhoc")

        with pytest.raises(ValueError, match="official seed suffix"):
            build_submission(results)

    def test_mixed_seed_results_are_rejected(self, tmp_path):
        results = _make_results_dir(tmp_path, seed="seed42")
        other_root = tmp_path / "other"
        other = _make_results_dir(
            other_root,
            seed="seed31415",
            task_id="math.other_task",
        )
        source = other / "math.other_task"
        destination = results / "math.other_task"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)

        with pytest.raises(ValueError, match="only accepts seed42.*seed31415"):
            build_submission(results)

    def test_non_seed42_benchmark_repo_is_rejected(self, tmp_path):
        results = _make_results_dir(tmp_path)

        with pytest.raises(ValueError, match="benchmark_repo.*seed42"):
            build_submission(
                results,
                benchmark_repo="Apexintelligence-AI/ASI-Bench-seed31415",
            )

    @pytest.mark.parametrize(
        "benchmark_repo",
        ["seed42", "Apexintelligence-AI/ASI-Bench-seed42"],
    )
    def test_seed42_benchmark_repo_names_are_accepted(
        self, tmp_path, benchmark_repo,
    ):
        results = _make_results_dir(tmp_path)

        bundle = build_submission(
            results,
            benchmark_repo=benchmark_repo,
            archive=False,
        )

        manifest = json.loads((bundle.bundle_dir / "manifest.json").read_text())
        assert manifest["benchmark_repo"] == benchmark_repo

    def test_public_run_without_private_framework_info_is_bundleable(self, tmp_path):
        results = _make_results_dir(tmp_path, include_framework_task_info=False)

        bundle = build_submission(results, archive=False)

        assert bundle.instance_count == 1
        entry = bundle.instances[0]
        assert entry.framework_task_info_file is None
        assert not (
            bundle.bundle_dir / "instances" / entry.base_name / "framework_task_info.json"
        ).exists()

    def test_bundle_collects_everything(self, tmp_path):
        results = _make_results_dir(tmp_path)
        bundle = build_submission(results)

        assert bundle.instance_count == 1
        entry = bundle.instances[0]
        assert entry.task_id == "physics.demo_task"
        assert entry.missing_outputs == []
        assert bundle.run_metadata_included

        d = bundle.bundle_dir
        assert (d / "manifest.json").is_file()
        assert (d / "run_metadata.json").is_file()
        assert (d / "instances" / f"{entry.base_name}" / "result.json").is_file()
        assert (d / "instances" / f"{entry.base_name}" / "framework_task_info.json").is_file()
        # Actual output bytes are present, not just referenced by path.
        assert (d / "instances" / entry.base_name / "outputs" / "solution.py").is_file()
        assert (d / "instances" / entry.base_name / "outputs" / "results" / "answer.npy").read_bytes() == b"\x00\x01\x02\x03"

    def test_manifest_hashes_every_file(self, tmp_path):
        results = _make_results_dir(tmp_path)
        bundle = build_submission(results, archive=False)
        manifest = json.loads((bundle.bundle_dir / "manifest.json").read_text())

        assert manifest["submission_schema_version"] == 1
        assert manifest["instance_count"] == 1
        # Every listed file exists and hashes match.
        import hashlib
        for rec in manifest["files"]:
            fp = bundle.bundle_dir / rec["path"]
            assert fp.is_file()
            assert hashlib.sha256(fp.read_bytes()).hexdigest() == rec["sha256"]

    def test_archive_is_valid_targz(self, tmp_path):
        results = _make_results_dir(tmp_path)
        bundle = build_submission(results)
        assert bundle.archive_path is not None and bundle.archive_path.is_file()
        with tarfile.open(bundle.archive_path, "r:gz") as tar:
            names = tar.getnames()
        assert any(n.endswith("manifest.json") for n in names)

    def test_missing_outputs_flagged_not_fatal(self, tmp_path):
        results = _make_results_dir(tmp_path, produce_all=False)
        bundle = build_submission(results)
        assert bundle.instance_count == 1
        flagged = bundle.instances_with_missing_outputs
        assert len(flagged) == 1
        assert "results/answer.npy" in flagged[0].missing_outputs

    def test_empty_results_dir_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="No per-instance result JSON"):
            build_submission(empty)

    def test_custom_targz_output_path(self, tmp_path):
        results = _make_results_dir(tmp_path)
        dest = tmp_path / "mysub.tar.gz"
        bundle = build_submission(results, output_path=dest)
        assert bundle.archive_path == dest
        assert dest.is_file()


class _EchoHandler(BaseHTTPRequestHandler):
    received: dict = {}

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        type(self).received = {
            "auth": self.headers.get("Authorization"),
            "content_type": self.headers.get("Content-Type"),
            "size": len(body),
            "gzip_ok": body[:2] == b"\x1f\x8b",
        }
        payload = json.dumps({"submission_id": "sub-123", "status": "received"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


class TestUpload:
    def test_upload_posts_bundle(self, tmp_path):
        results = _make_results_dir(tmp_path)
        bundle = build_submission(results)

        server = HTTPServer(("127.0.0.1", 0), _EchoHandler)
        threading.Thread(target=server.handle_request, daemon=True).start()
        port = server.server_address[1]
        try:
            res = upload_bundle(
                bundle.archive_path,
                f"http://127.0.0.1:{port}/submit",
                token="secret-tok",
            )
        finally:
            server.server_close()

        assert res.ok and res.status_code == 200
        assert res.response_json == {"submission_id": "sub-123", "status": "received"}
        assert _EchoHandler.received["auth"] == "Bearer secret-tok"
        assert _EchoHandler.received["content_type"] == "application/gzip"
        assert _EchoHandler.received["gzip_ok"]

    def test_upload_missing_archive(self, tmp_path):
        res = upload_bundle(tmp_path / "nope.tar.gz", "http://127.0.0.1:1/x")
        assert not res.ok
        assert "not found" in res.error

    def test_upload_network_error_no_raise(self, tmp_path):
        results = _make_results_dir(tmp_path)
        bundle = build_submission(results)
        # Port 1 is not listenable -> connection refused, surfaced as error.
        res = upload_bundle(bundle.archive_path, "http://127.0.0.1:1/x", timeout=2)
        assert not res.ok and res.error
