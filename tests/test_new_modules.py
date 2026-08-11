"""Tests for leaderboard, visualizer, contamination, and new adapters."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai4sci_bench.core.types import (
    AgentOutput,
    EvalResult,
    PromptLevel,
    RunStatus,
    TaskInstance,
)
from ai4sci_bench.generators.contamination import (
    ContaminationGuard,
    detect_canary,
    generate_canary,
    inject_canary,
    perturb_prompt,
)
from ai4sci_bench.reporting.leaderboard import Leaderboard, LeaderboardEntry
from ai4sci_bench.reporting.results import DomainSummary, RunReport, TaskSummary
from ai4sci_bench.reporting.visualizer import ResultVisualizer


# ── Contamination Tests ───────────────────────────────────────────────


class TestContamination:
    def test_generate_canary(self):
        """Canary is generated as HTML comment."""
        canary = generate_canary("test_instance_1")
        assert canary.startswith("<!-- CANARY ")
        assert canary.endswith(" -->")

    def test_canary_uniqueness(self):
        """Different instances produce different canaries."""
        c1 = generate_canary("instance_1")
        c2 = generate_canary("instance_2")
        assert c1 != c2

    def test_inject_canary(self):
        """Canary is injected at the end of prompt."""
        prompt = "# Test Prompt\n\nDo something."
        canary = generate_canary("test")
        result = inject_canary(prompt, canary)
        assert prompt in result
        assert canary in result

    def test_detect_canary(self):
        """Canary IDs are detected in text."""
        canary = "<!-- CANARY abc123def4560a78 -->"
        text = f"Some text\n{canary}\nMore text"
        ids = detect_canary(text)
        assert "abc123def4560a78" in ids

    def test_detect_no_canary(self):
        """No canary returns empty list."""
        assert detect_canary("regular text without canary") == []

    def test_perturb_prompt_deterministic(self):
        """Same seed produces same perturbation."""
        prompt = "Write a simulation. Save the following output files."
        p1 = perturb_prompt(prompt, seed=42)
        p2 = perturb_prompt(prompt, seed=42)
        assert p1 == p2

    def test_perturb_prompt_different_seeds(self):
        """Different seeds may produce different perturbations."""
        prompt = "implement the algorithm. Save the following required output files in the current directory."
        results = set()
        for seed in range(20):
            results.add(perturb_prompt(prompt, seed=seed))
        # With enough seeds, at least some should differ
        assert len(results) >= 2

    def test_contamination_guard_protect(self):
        """ContaminationGuard applies all protections."""
        guard = ContaminationGuard(secret="test-secret")
        prompt = "# Task\nWrite code."
        protected = guard.protect_prompt(prompt, "inst_1", seed=42)
        assert "CANARY" in protected
        assert guard.canary_registry.get("inst_1") is not None

    def test_contamination_guard_detect(self):
        """ContaminationGuard detects its own canaries."""
        guard = ContaminationGuard(secret="test")
        prompt = "# Task"
        protected = guard.protect_prompt(prompt, "inst_1")
        contaminated = guard.check_contamination(protected)
        assert "inst_1" in contaminated

    def test_contamination_guard_no_false_positive(self):
        """Clean text does not trigger contamination detection."""
        guard = ContaminationGuard(secret="test")
        guard.protect_prompt("# Task", "inst_1")
        contaminated = guard.check_contamination("clean text without any canaries")
        assert contaminated == []

    def test_contamination_guard_disable(self):
        """Guards can be selectively disabled."""
        guard = ContaminationGuard(enable_canary=False, enable_perturbation=False)
        prompt = "# Task"
        protected = guard.protect_prompt(prompt, "inst_1")
        assert "CANARY" not in protected
        assert protected == prompt


# ── Leaderboard Tests ─────────────────────────────────────────────────


def _make_report(agent_name, score, domains=None, levels=None):
    """Helper to create a RunReport for testing."""
    task_summaries = []
    domain_summaries = []
    if domains:
        for d, s in domains.items():
            ts = TaskSummary(
                task_id=f"{d}.task1", n_instances=5, mean_score=s,
                min_score=s - 10, max_score=s + 10, std_score=5.0,
                gates_pass_rate=0.8,
            )
            task_summaries.append(ts)
            domain_summaries.append(DomainSummary(
                domain=d, n_tasks=1, mean_score=s, task_summaries=[ts],
            ))

    return RunReport(
        agent_name=agent_name,
        n_tasks=len(task_summaries) or 1,
        n_instances=5,
        overall_mean_score=score,
        by_task=task_summaries,
        by_domain=domain_summaries,
        by_prompt_level=levels or {},
    )


class TestLeaderboard:
    def test_add_entry(self):
        """Entries can be added from RunReport."""
        lb = Leaderboard()
        report = _make_report("AgentA", 85.0)
        lb.add_from_report(report)
        assert len(lb.entries) == 1
        assert lb.entries[0].agent_name == "AgentA"
        assert lb.entries[0].rank == 1

    def test_ranking_order(self):
        """Entries are ranked by score descending."""
        lb = Leaderboard()
        lb.add_from_report(_make_report("Low", 40.0))
        lb.add_from_report(_make_report("High", 90.0))
        lb.add_from_report(_make_report("Mid", 65.0))
        assert lb.entries[0].agent_name == "High"
        assert lb.entries[0].rank == 1
        assert lb.entries[1].agent_name == "Mid"
        assert lb.entries[2].agent_name == "Low"

    def test_format_table(self):
        """Table formatting produces readable output."""
        lb = Leaderboard()
        lb.add_from_report(_make_report("AgentA", 85.0, {"physics": 90.0}))
        lb.add_from_report(_make_report("AgentB", 65.0, {"physics": 60.0}))
        table = lb.format_table()
        assert "AgentA" in table
        assert "AgentB" in table
        assert "Leaderboard" in table

    def test_empty_leaderboard(self):
        """Empty leaderboard returns message."""
        lb = Leaderboard()
        assert "No entries" in lb.format_table()

    def test_save_and_load(self, tmp_dir):
        """Leaderboard can be saved and loaded."""
        lb = Leaderboard()
        lb.add_from_report(_make_report("AgentA", 85.0))
        lb.add_from_report(_make_report("AgentB", 65.0))

        path = tmp_dir / "leaderboard.json"
        lb.save(path)
        assert path.exists()

        loaded = Leaderboard.load(path)
        assert len(loaded.entries) == 2
        assert loaded.entries[0].agent_name == "AgentA"

    def test_to_dict(self):
        """Leaderboard serializes to list of dicts."""
        lb = Leaderboard()
        lb.add_from_report(_make_report("A", 90.0))
        data = lb.to_dict()
        assert len(data) == 1
        assert data[0]["agent_name"] == "A"
        assert data[0]["rank"] == 1

    def test_domain_and_level_breakdown(self):
        """Domain and prompt level breakdowns are included."""
        lb = Leaderboard()
        lb.add_from_report(_make_report(
            "Agent", 80.0,
            domains={"physics": 85.0, "chemistry": 75.0},
            levels={"b1": 90.0, "b2": 80.0, "b3": 70.0},
        ))
        entry = lb.entries[0]
        assert entry.scores_by_domain["physics"] == 85.0
        assert entry.scores_by_level["b1"] == 90.0


# ── Visualizer Tests ──────────────────────────────────────────────────


class TestVisualizer:
    def _make_report(self):
        return _make_report(
            "TestAgent", 75.0,
            domains={"physics": 80.0, "chemistry": 70.0},
            levels={"b1": 90.0, "b2": 75.0, "b3": 60.0},
        )

    def test_task_table(self):
        """Task table includes all tasks."""
        vis = ResultVisualizer(self._make_report())
        table = vis.task_table()
        assert "physics.task1" in table
        assert "chemistry.task1" in table

    def test_domain_table(self):
        """Domain table includes all domains."""
        vis = ResultVisualizer(self._make_report())
        table = vis.domain_table()
        assert "physics" in table
        assert "chemistry" in table

    def test_prompt_level_table(self):
        """Prompt level table includes all levels."""
        vis = ResultVisualizer(self._make_report())
        table = vis.prompt_level_table()
        assert "B1" in table
        assert "B2" in table
        assert "B3" in table

    def test_full_report(self):
        """Full report combines all sections."""
        vis = ResultVisualizer(self._make_report())
        report = vis.full_report()
        assert "Results Summary" in report
        assert "Per-Task" in report
        assert "Per-Domain" in report

    def test_save_json(self, tmp_dir):
        """JSON export works."""
        vis = ResultVisualizer(self._make_report())
        path = tmp_dir / "report.json"
        vis.save_json(path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["agent_name"] == "TestAgent"
        assert data["overall_mean_score"] == 75.0

    def test_empty_report_tables(self):
        """Tables handle empty data gracefully."""
        report = RunReport(
            agent_name="Empty", n_tasks=0, n_instances=0, overall_mean_score=0.0,
        )
        vis = ResultVisualizer(report)
        assert "No task" in vis.task_table()
        assert "No domain" in vis.domain_table()
        assert "No prompt" in vis.prompt_level_table()
        assert "No error" in vis.error_distribution_table()

    def test_error_distribution_table(self):
        """Error distribution table shows categories and percentages."""
        report = _make_report("Agent", 50.0)
        report.error_distribution = {"algorithm_error": 5, "implementation_bug": 3}
        vis = ResultVisualizer(report)
        table = vis.error_distribution_table()
        assert "algorithm_error" in table
        assert "implementation_bug" in table
        assert "Total" in table


# ── HTTP Agent Adapter Tests ──────────────────────────────────────────


class TestHTTPAgentAdapter:
    def test_init(self):
        """HTTPAgentAdapter initializes with base_url."""
        from ai4sci_bench.adapters.http_agent import HTTPAgentAdapter
        adapter = HTTPAgentAdapter(base_url="http://localhost:8080")
        assert adapter.base_url == "http://localhost:8080"
        assert adapter.timeout == 10800

    def test_init_with_api_key(self):
        """API key sets Authorization header."""
        from ai4sci_bench.adapters.http_agent import HTTPAgentAdapter
        adapter = HTTPAgentAdapter(
            base_url="http://localhost:8080", api_key="sk-test"
        )
        assert adapter.headers["Authorization"] == "Bearer sk-test"


# ── Docker Agent Adapter Tests ────────────────────────────────────────


class TestDockerAgentAdapter:
    def test_init(self):
        """DockerAgentAdapter initializes with image."""
        from ai4sci_bench.adapters.docker_agent import DockerAgentAdapter
        adapter = DockerAgentAdapter(image="my-agent:latest")
        assert adapter.image == "my-agent:latest"
        assert adapter.timeout == 10800

    def test_build_docker_cmd(self, tmp_dir):
        """Docker command includes image and workspace mount."""
        from ai4sci_bench.adapters.docker_agent import DockerAgentAdapter
        adapter = DockerAgentAdapter(
            image="my-agent:latest",
            cmd_template="python /app/agent.py --workspace /workspace",
        )
        instance = TaskInstance(
            task_id="test.task",
            instance_id="test__inst",
            task_dir=tmp_dir,
            workspace_dir=tmp_dir,
            reference_dir=tmp_dir,
            prompt_level=PromptLevel.B2,
            parameters={},
            metadata={"output": {"files": []}},
        )
        cmd = adapter._build_docker_cmd(tmp_dir, instance)
        assert "docker" in cmd
        assert "my-agent:latest" in cmd
        assert any("/workspace" in str(c) for c in cmd)
