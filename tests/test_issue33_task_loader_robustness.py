"""Regression tests for issue #33: IndentationError in TaskLoader fallback scan.

Both discover_tasks() and load_task_by_id() had:
  1. Duplicate load_task_metadata() calls (once unprotected, once in try/except)
  2. Missing `continue` after _is_template_metadata() check

These tests ensure:
  - Template-like tasks are skipped (not returned) in both methods
  - Bad YAML is gracefully skipped (not crash) in both methods
  - load_task_metadata() is called exactly once per task.yaml in the scan
  - Mixed scenarios (good + bad + template) work end-to-end
"""

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from ai4sci_bench.core.task import TaskLoader


def _write_task(tasks_dir: Path, rel_path: str, meta: dict) -> Path:
    task_dir = tasks_dir / rel_path
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.yaml").write_text(yaml.dump(meta))
    return task_dir


class TestIssue33DiscoverTasksRobustness:
    """discover_tasks() must skip bad YAML and template-like tasks cleanly."""

    def test_template_task_excluded_from_results(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        _write_task(tasks_dir, "physics/real", {
            "id": "physics.real", "name": "Real", "status": "final", "domain": "physics",
        })
        _write_task(tasks_dir, "physics/copied_template", {
            "id": "DOMAIN.TASK_NAME", "name": "Human-readable task name",
            "status": "final", "domain": "physics",
        })

        loader = TaskLoader(tasks_dir)
        tasks = loader.discover_tasks()
        ids = [t["id"] for t in tasks]
        assert ids == ["physics.real"]

    def test_bad_yaml_skipped_without_crash(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        _write_task(tasks_dir, "physics/good", {
            "id": "physics.good", "name": "Good", "status": "final", "domain": "physics",
        })
        bad_dir = tasks_dir / "physics" / "bad"
        bad_dir.mkdir(parents=True)
        (bad_dir / "task.yaml").write_text("id: bad\n  broken: {invalid\n")

        loader = TaskLoader(tasks_dir)
        tasks = loader.discover_tasks(include_dev=True)
        assert len(tasks) == 1
        assert tasks[0]["id"] == "physics.good"

    def test_load_task_metadata_called_once_per_file(self, tmp_path):
        """Regression: the old code called load_task_metadata twice per file."""
        tasks_dir = tmp_path / "tasks"
        _write_task(tasks_dir, "a/t1", {
            "id": "a.t1", "name": "T1", "status": "final", "domain": "a",
        })
        _write_task(tasks_dir, "a/t2", {
            "id": "a.t2", "name": "T2", "status": "final", "domain": "a",
        })

        loader = TaskLoader(tasks_dir)
        with patch.object(loader, "load_task_metadata", wraps=loader.load_task_metadata) as mock:
            loader.discover_tasks()
            assert mock.call_count == 2

    def test_mixed_good_bad_template(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        _write_task(tasks_dir, "a/good1", {
            "id": "a.good1", "name": "Good1", "status": "final", "domain": "a",
        })
        _write_task(tasks_dir, "a/good2", {
            "id": "a.good2", "name": "Good2", "status": "test", "domain": "a",
        })
        _write_task(tasks_dir, "a/template_copy", {
            "id": "DOMAIN.TASK_NAME", "name": "Whatever", "status": "final", "domain": "a",
        })
        bad_dir = tasks_dir / "a" / "corrupt"
        bad_dir.mkdir(parents=True)
        (bad_dir / "task.yaml").write_text("not: valid: yaml: {{")
        _write_task(tasks_dir, "_template", {
            "id": "template", "name": "template", "status": "final", "domain": "a",
        })

        loader = TaskLoader(tasks_dir)
        tasks = loader.discover_tasks(include_test=True)
        ids = sorted(t["id"] for t in tasks)
        assert ids == ["a.good1", "a.good2"]

    def test_template_with_domain_prefix_skipped(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        _write_task(tasks_dir, "phys/stub", {
            "id": "DOMAIN.something", "name": "Stub", "status": "final", "domain": "phys",
        })
        _write_task(tasks_dir, "phys/stub2", {
            "id": "phys.TASK_NAME", "name": "Stub2", "status": "final", "domain": "phys",
        })

        loader = TaskLoader(tasks_dir)
        tasks = loader.discover_tasks()
        assert tasks == []

    def test_sample_task_is_valid_and_opt_in(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        _write_task(tasks_dir, "physics/sample", {
            "id": "physics.sample", "name": "Sample", "status": "sample",
            "domain": "physics",
        })

        loader = TaskLoader(tasks_dir)
        assert loader.discover_tasks() == []
        assert [t["id"] for t in loader.discover_tasks(include_sample=True)] == [
            "physics.sample"
        ]
        assert loader.load_task_by_id("physics.sample")["status"] == "sample"


class TestIssue33LoadTaskByIdRobustness:
    """load_task_by_id() fallback scan must skip bad/template tasks cleanly."""

    def test_finds_task_despite_bad_yaml_in_other_dir(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        _write_task(tasks_dir, "chem/good", {
            "id": "chem.good", "name": "Good", "status": "final", "domain": "chem",
        })
        bad_dir = tasks_dir / "chem" / "broken"
        bad_dir.mkdir(parents=True)
        (bad_dir / "task.yaml").write_text("{{broken yaml")

        loader = TaskLoader(tasks_dir)
        meta = loader.load_task_by_id("chem.good")
        assert meta["id"] == "chem.good"

    def test_template_task_not_found_by_id(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        _write_task(tasks_dir, "astronomy/GL", {
            "id": "DOMAIN.TASK_NAME", "name": "Human-readable task name",
            "status": "in_development", "domain": "astronomy",
        })

        loader = TaskLoader(tasks_dir)
        with pytest.raises(ValueError, match="not found"):
            loader.load_task_by_id("DOMAIN.TASK_NAME")

    def test_load_metadata_called_once_per_file_in_fallback(self, tmp_path):
        """Regression: old code called load_task_metadata twice per file in scan."""
        tasks_dir = tmp_path / "tasks"
        # Use non-standard directory structure to force fallback scan
        _write_task(tasks_dir, "nonstandard_dir", {
            "id": "a.target", "name": "Target", "status": "final", "domain": "a",
        })
        _write_task(tasks_dir, "other_dir", {
            "id": "a.other", "name": "Other", "status": "final", "domain": "a",
        })

        loader = TaskLoader(tasks_dir)
        with patch.object(loader, "load_task_metadata", wraps=loader.load_task_metadata) as mock:
            meta = loader.load_task_by_id("a.target")
            assert meta["id"] == "a.target"
            # Direct path lookup fails (nonstandard dir), so fallback scans.
            # Each file should be loaded at most once.
            for call in mock.call_args_list:
                yaml_path = call[0][0]
                count = sum(1 for c in mock.call_args_list if c[0][0] == yaml_path)
                assert count == 1, f"load_task_metadata called {count} times for {yaml_path}"

    def test_fallback_skips_template_finds_real_task(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        _write_task(tasks_dir, "bio/template_copy", {
            "id": "DOMAIN.TASK_NAME", "name": "Human-readable task name",
            "status": "final", "domain": "bio",
        })
        # Non-standard path forces fallback
        _write_task(tasks_dir, "bio_custom_dir", {
            "id": "bio.real", "name": "Real", "status": "final", "domain": "bio",
        })

        loader = TaskLoader(tasks_dir)
        meta = loader.load_task_by_id("bio.real")
        assert meta["id"] == "bio.real"

    def test_fallback_with_only_bad_yaml_raises_not_found(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        bad_dir = tasks_dir / "a" / "broken"
        bad_dir.mkdir(parents=True)
        (bad_dir / "task.yaml").write_text("broken: {yaml")

        loader = TaskLoader(tasks_dir)
        with pytest.raises(ValueError, match="not found"):
            loader.load_task_by_id("a.broken")

    def test_direct_path_still_works(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        _write_task(tasks_dir, "physics/leapfrog", {
            "id": "physics.leapfrog", "name": "Leapfrog", "status": "final", "domain": "physics",
        })

        loader = TaskLoader(tasks_dir)
        with patch.object(loader, "load_task_metadata", wraps=loader.load_task_metadata) as mock:
            meta = loader.load_task_by_id("physics.leapfrog")
            assert meta["id"] == "physics.leapfrog"
            assert mock.call_count == 1

    def test_direct_path_bad_yaml_falls_through_to_scan(self, tmp_path):
        """If direct path resolves to bad YAML, fallback scan should still work."""
        tasks_dir = tmp_path / "tasks"
        # Bad YAML at the direct-path location
        bad_dir = tasks_dir / "a" / "target"
        bad_dir.mkdir(parents=True)
        (bad_dir / "task.yaml").write_text("broken: {yaml")
        # Good task in a different location that the scan will find
        _write_task(tasks_dir, "elsewhere", {
            "id": "a.target", "name": "Target", "status": "final", "domain": "a",
        })

        loader = TaskLoader(tasks_dir)
        meta = loader.load_task_by_id("a.target")
        assert meta["id"] == "a.target"

    def test_mixed_bad_template_good_in_fallback(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        bad_dir = tasks_dir / "x" / "bad"
        bad_dir.mkdir(parents=True)
        (bad_dir / "task.yaml").write_text("{{invalid")
        _write_task(tasks_dir, "x/template_copy", {
            "id": "DOMAIN.TASK_NAME", "name": "Human-readable task name",
            "status": "final", "domain": "x",
        })
        # Non-standard path forces fallback
        _write_task(tasks_dir, "x_custom", {
            "id": "x.target", "name": "Target", "status": "final", "domain": "x",
        })

        loader = TaskLoader(tasks_dir)
        meta = loader.load_task_by_id("x.target")
        assert meta["id"] == "x.target"


class TestIsTemplateMetadata:
    """Unit tests for _is_template_metadata edge cases."""

    @pytest.mark.parametrize("task_id,expected", [
        ("DOMAIN.TASK_NAME", True),
        ("DOMAIN.something", True),
        ("something.TASK_NAME", True),
        ("physics.leapfrog", False),
        ("", False),
    ])
    def test_template_id_detection(self, task_id, expected):
        meta = {"id": task_id, "name": "Some Name"}
        assert TaskLoader._is_template_metadata(meta) == expected

    def test_template_name_detection(self):
        meta = {"id": "physics.real", "name": "Human-readable task name"}
        assert TaskLoader._is_template_metadata(meta) is True

    def test_normal_task_not_template(self):
        meta = {"id": "physics.leapfrog", "name": "VIC Leapfrog"}
        assert TaskLoader._is_template_metadata(meta) is False

    def test_missing_id_and_name(self):
        assert TaskLoader._is_template_metadata({}) is False

    def test_whitespace_handling(self):
        meta = {"id": " DOMAIN.TASK_NAME ", "name": "x"}
        assert TaskLoader._is_template_metadata(meta) is True
