"""Tests for the HF pull module and produce-only (submit) reporting.

Network-free: only the pure helpers (repo aliasing, allow-pattern translation,
token resolution) and the unscored-submission report path are exercised.
"""

import pytest

from ai4sci_bench.hf.pull import (
    REPO_ALIASES,
    _allow_patterns_for_tasks,
    pull_instances,
    resolve_repo_id,
    resolve_token,
)


class TestResolveRepoId:
    def test_aliases_map_to_full_ids(self):
        assert resolve_repo_id("demo") == REPO_ALIASES["demo"]
        assert resolve_repo_id("seed42") == "Apexintelligence-AI/ASI-Bench-seed42"
        assert resolve_repo_id("seed31415") == "Apexintelligence-AI/ASI-Bench-seed31415"
        assert set(REPO_ALIASES) == {"demo", "seed42", "seed31415"}

    def test_retired_ambiguous_benchmark_alias_raises(self):
        with pytest.raises(ValueError, match="Unknown repo alias"):
            resolve_repo_id("benchmark")

    def test_full_id_passthrough(self):
        assert resolve_repo_id("some-org/Custom-Repo") == "some-org/Custom-Repo"

    def test_unknown_alias_raises(self):
        with pytest.raises(ValueError, match="Unknown repo alias"):
            resolve_repo_id("bogus")


class TestAllowPatterns:
    def test_none_when_no_tasks(self):
        assert _allow_patterns_for_tasks(None) is None
        assert _allow_patterns_for_tasks([]) is None

    def test_task_id_to_pattern(self):
        assert _allow_patterns_for_tasks(["physics.example_task"]) == [
            "tasks/physics.example_task__*/**",
            "tasks/physics.example_task/**",
        ]

    def test_tid_used_verbatim_dots_preserved(self):
        # The full '<domain>.<name>' id is used verbatim in the flat dir name.
        assert _allow_patterns_for_tasks(["math.example_task"]) == [
            "tasks/math.example_task__*/**",
            "tasks/math.example_task/**",
        ]

    def test_missing_domain_raises(self):
        with pytest.raises(ValueError, match="must be"):
            _allow_patterns_for_tasks(["no_domain_here"])


class TestFlattenFlatLayout:
    """pull_instances must flatten the flat HF layout tasks/<instance_id>/ →
    <output_dir>/<instance_id>/. Regression guard: the module previously assumed
    a nested tasks/<domain>/<name>/instances/<id>/ layout and pulled 0 instances.
    """

    def _fake_snapshot(self, tmp_path):
        # Mimic the real HF repo shape: flat tasks/<instance_id>/ dirs.
        root = tmp_path / "snapshot"
        for inst in ("astronomy.example_task__seed478163327",
                     "physics.example_task__grid255_seed42"):
            d = root / "tasks" / inst
            (d / "data").mkdir(parents=True)
            (d / "prompt_b1.md").write_text("# task", encoding="utf-8")
            (d / "data" / "input_0.npy").write_text("x", encoding="utf-8")
            (d / "framework_task_info.json").write_text("{}", encoding="utf-8")
        (root / ".gitattributes").write_text("", encoding="utf-8")
        return root

    def test_flattens_instances(self, tmp_path, monkeypatch):
        snap = self._fake_snapshot(tmp_path)
        import ai4sci_bench.hf.pull as pull_mod
        monkeypatch.setattr(
            "huggingface_hub.snapshot_download", lambda **kw: str(snap)
        )
        out = tmp_path / "out"
        res = pull_instances("demo", output_dir=out)

        assert sorted(res.instance_ids) == [
            "astronomy.example_task__seed478163327",
            "physics.example_task__grid255_seed42",
        ]
        # task_id derived from the "<domain>.<name>" prefix before "__"
        assert sorted(res.task_ids) == ["astronomy.example_task", "physics.example_task"]
        # files materialized flat under <output>/<instance_id>/
        inst = out / "astronomy.example_task__seed478163327"
        assert (inst / "prompt_b1.md").exists()
        assert (inst / "data" / "input_0.npy").exists()
        # top-level non-instance files (.gitattributes) are not copied
        assert not (out / ".gitattributes").exists()

    def test_skips_existing_without_overwrite(self, tmp_path, monkeypatch):
        snap = self._fake_snapshot(tmp_path)
        monkeypatch.setattr(
            "huggingface_hub.snapshot_download", lambda **kw: str(snap)
        )
        out = tmp_path / "out"
        (out / "physics.example_task__grid255_seed42").mkdir(parents=True)
        res = pull_instances("demo", output_dir=out)
        assert "physics.example_task__grid255_seed42" in res.skipped
        assert "astronomy.example_task__seed478163327" in res.instance_ids

    def test_filters_cached_snapshot_dirs_when_tasks_are_requested(
        self, tmp_path, monkeypatch
    ):
        """A reused HF snapshot may contain dirs outside allow_patterns."""
        snap = self._fake_snapshot(tmp_path)
        monkeypatch.setattr(
            "huggingface_hub.snapshot_download", lambda **kw: str(snap)
        )
        out = tmp_path / "out"

        res = pull_instances(
            "demo",
            output_dir=out,
            tasks=["astronomy.example_task"],
        )

        assert res.task_ids == ["astronomy.example_task"]
        assert res.instance_ids == ["astronomy.example_task__seed478163327"]
        assert (out / "astronomy.example_task__seed478163327").is_dir()
        assert not (out / "physics.example_task__grid255_seed42").exists()

    @pytest.mark.parametrize(
        ("repo", "reference_expected"),
        [("seed42", False), ("seed31415", True)],
    )
    def test_reference_policy_is_enforced_during_copy(
        self, tmp_path, monkeypatch, repo, reference_expected
    ):
        snap = self._fake_snapshot(tmp_path)
        for instance in (snap / "tasks").iterdir():
            reference = instance / "reference"
            reference.mkdir()
            (reference / "answer.json").write_text("{}", encoding="utf-8")
        captured = {}

        def fake_download(**kwargs):
            captured.update(kwargs)
            return str(snap)

        monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
        out = tmp_path / "out"
        pull_instances(repo, output_dir=out)

        copied = out / "astronomy.example_task__seed478163327" / "reference"
        assert copied.exists() is reference_expected
        if repo == "seed42":
            assert "**/reference/**" in captured["ignore_patterns"]
        else:
            assert captured["ignore_patterns"] is None


class TestRootFlatLayout:
    """Only the public tasks/<instance-id>/ dataset contract is accepted."""

    def test_ignores_root_level_instances(self, tmp_path, monkeypatch):
        snap = tmp_path / "snapshot"
        inst = snap / "math.example_task__seed31415"
        (inst / "data").mkdir(parents=True)
        (inst / "prompt_b1.md").write_text("# task", encoding="utf-8")
        (inst / "data" / "input.json").write_text("{}", encoding="utf-8")
        (snap / ".gitattributes").write_text("", encoding="utf-8")
        monkeypatch.setattr(
            "huggingface_hub.snapshot_download", lambda **kw: str(snap)
        )

        out = tmp_path / "out"
        res = pull_instances("demo", output_dir=out)

        assert res.task_ids == []
        assert res.instance_ids == []
        assert not (out / "math.example_task__seed31415").exists()
        assert not (out / ".gitattributes").exists()


class TestResolveToken:
    def test_explicit_wins(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "from_env")
        assert resolve_token("explicit") == "explicit"

    def test_falls_back_to_env(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setenv("HF_TOKEN", "env_tok")
        assert resolve_token(None) == "env_tok"

    def test_none_when_absent(self, monkeypatch):
        for var in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            monkeypatch.delenv(var, raising=False)
        assert resolve_token(None) is None


class TestUnscoredSubmissionReporting:
    def _make_result(self, *, unscored: bool):
        from ai4sci_bench.core.types import EvalResult, PromptLevel, RunStatus, ScoreDetail

        details = {"unscored_submission": True} if unscored else {"scorer_internal_error": True}
        return EvalResult(
            instance_id="astronomy.x__seed1",
            task_id="astronomy.x",
            prompt_level=PromptLevel.B1,
            agent_name="TestAgent",
            parameters={},
            gate_results=[],
            gates_passed=False,
            score_results=[ScoreDetail(
                scorer_name="unscored_submission" if unscored else "boom",
                score=0.0, max_score=100.0, passed=False, message="", details=details,
            )],
            final_score=0.0,
            execution_time_seconds=0.0,
            status=RunStatus.COMPLETED,
        )

    def test_detector(self):
        from ai4sci_bench.reporting.results import is_unscored_submission

        assert is_unscored_submission(self._make_result(unscored=True)) is True
        assert is_unscored_submission(self._make_result(unscored=False)) is False

    def test_aggregator_excludes_unscored_from_gate_failures(self):
        from ai4sci_bench.reporting.aggregator import aggregate_results

        report = aggregate_results([self._make_result(unscored=True)])
        assert report.unscored_submission_instances == 1
        assert report.gate_failed_instances == 0
        assert "Unscored submissions (submit mode): 1" in report.format_summary()
