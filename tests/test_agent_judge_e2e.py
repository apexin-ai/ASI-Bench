"""End-to-end integration test for agent_judge scorer.

Runs a real agent CLI (codex or claude) with a trivial rubric and
pre-staged workspace to verify the full pipeline works:
  _prepare_workspace → subprocess.run(agent CLI) → _extract_score → aggregate

Requires: `codex` or `claude` CLI in PATH.
Skip with: pytest -m "not e2e"
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from ai4sci_bench.core.scorer import get_scorer

import ai4sci_bench.scorers  # noqa: F401

pytestmark = pytest.mark.e2e


def _has_cli(name: str) -> bool:
    return shutil.which(name) is not None


@pytest.fixture
def workspace(tmp_path):
    pred_dir = tmp_path / "pred"
    ref_dir = tmp_path / "ref"
    pred_dir.mkdir()
    ref_dir.mkdir()

    (pred_dir / "solution.py").write_text(
        "import math\nresult = math.sqrt(144)\nprint(f'Result: {result}')\n"
    )
    (pred_dir / "output.txt").write_text("Result: 12.0\n")

    (ref_dir / "expected_output.txt").write_text("Result: 12.0\n")

    return pred_dir, ref_dir


SIMPLE_RUBRIC = """\
Compare the agent's output to the reference.

agent_output/output.txt should contain "Result: 12.0".
reference/expected_output.txt has the expected value.

Also check that agent_output/solution.py computes sqrt(144) correctly.

Scoring (0-10):
- Output matches reference exactly: 8-10 pts
- Output is close but not exact: 4-7 pts
- Output is wrong or missing: 0-3 pts
"""


@pytest.mark.skipif(not _has_cli("codex"), reason="codex CLI not in PATH")
class TestE2ECodex:
    def test_codex_simple_rubric(self, workspace):
        pred_dir, ref_dir = workspace
        scorer = get_scorer("agent_judge")

        result = scorer.score(pred_dir, ref_dir, {
            "agent": "codex",
            "rubric": SIMPLE_RUBRIC,
            "timeout": 120,
            "num_judges": 1,
            "max_score_value": 10,
            "weight": 10.0,
            "threshold": 0.5,
            "sandbox": "danger-full-access",
        })

        print(f"\n=== Codex E2E Result ===")
        print(f"  score: {result.score}/{result.max_score}")
        print(f"  passed: {result.passed}")
        print(f"  median_score: {result.details.get('median_score')}")
        print(f"  score_sources: {result.details.get('score_sources')}")
        print(f"  parse_failures: {result.details.get('parse_failures')}")
        print(f"  raw_log (first 500): {result.details.get('raw_responses', [''])[0][:500]}")

        assert result.scorer_name == "agent_judge"
        assert result.max_score == 10.0
        assert isinstance(result.score, float)
        assert result.details["num_judges"] == 1
        assert result.details["agent_type"] == "codex"
        assert result.details["valid_judge_count"] <= 1

        if result.details["valid_judge_count"] == 1:
            assert result.details["median_score"] >= 0
            assert result.details["median_score"] <= 10
            assert result.details["score_sources"][0] in ("score.json", "stdout_parse")


@pytest.mark.skipif(not _has_cli("claude"), reason="claude CLI not in PATH")
class TestE2EClaude:
    def test_claude_simple_rubric(self, workspace):
        pred_dir, ref_dir = workspace
        scorer = get_scorer("agent_judge")

        result = scorer.score(pred_dir, ref_dir, {
            "agent": "claude_code",
            "rubric": SIMPLE_RUBRIC,
            "timeout": 120,
            "num_judges": 1,
            "max_score_value": 10,
            "weight": 10.0,
            "threshold": 0.5,
        })

        print(f"\n=== Claude E2E Result ===")
        print(f"  score: {result.score}/{result.max_score}")
        print(f"  passed: {result.passed}")
        print(f"  median_score: {result.details.get('median_score')}")
        print(f"  score_sources: {result.details.get('score_sources')}")
        print(f"  parse_failures: {result.details.get('parse_failures')}")
        print(f"  raw_log (first 500): {result.details.get('raw_responses', [''])[0][:500]}")

        assert result.scorer_name == "agent_judge"
        assert result.max_score == 10.0
        assert isinstance(result.score, float)
        assert result.details["num_judges"] == 1
        assert result.details["agent_type"] == "claude_code"
        assert result.details["valid_judge_count"] <= 1

        if result.details["valid_judge_count"] == 1:
            assert result.details["median_score"] >= 0
            assert result.details["median_score"] <= 10
            assert result.details["score_sources"][0] in ("score.json", "stdout_parse")
