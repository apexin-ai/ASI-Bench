"""Regression checks for commands and task IDs advertised in the README."""

import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
AUTHOR_GUIDE = (ROOT / "docs" / "guide" / "authoring-a-task.md").read_text(encoding="utf-8")
GETTING_STARTED = (ROOT / "docs" / "guide" / "getting-started.md").read_text(
    encoding="utf-8"
)
TROUBLESHOOTING = (ROOT / "docs" / "guide" / "troubleshooting-results.md").read_text(
    encoding="utf-8"
)
EXAMPLE_TASKS = (
    "astronomy.nbody_close_encounters",
    "math.homotopy_poly_roots",
)


def test_readme_uses_canonical_cli():
    assert "asibench --help" in README
    assert "ai4sci-bench --help" not in README


def test_readme_links_the_arxiv_paper():
    assert "## Paper" in README
    assert "**ASI-Bench: At the Dawn of Artificial Superintelligence**" in README
    assert "https://arxiv.org/abs/2608.17271" in README


def test_readme_uses_produce_only_flow_for_official_tasks():
    assert "asibench run \\" in README
    assert "--score" not in README
    assert "--no-score" not in README
    assert "asibench submit --results-dir example_results/" in README


def test_docs_distinguish_public_and_private_seed_scoring():
    for text in (README, GETTING_STARTED):
        assert "asibench score" in text
        assert "--repo seed31415" in text
        assert "seed42" in text
        assert "asibench submit" in text
        assert "non-official" in text


def test_task_author_docs_keep_local_b1_b4_submission_gate():
    for text in (README, AUTHOR_GUIDE):
        assert "asibench validate --pre-submit" in text
        assert "asibench difficulty-check --task" in text
        assert "B1" in text and "B2" in text and "B3" in text and "B4" in text
        assert re.search(r"strictly\s+below\s+40", text)
        assert "B1" in text and "B2" in text
        assert re.search(r"no\s+score\s+ceiling", text)
        assert "--agent codex_cli" in text
        assert "gpt-5.6-sol" in text
        assert "direct_llm" in text
        assert "multi-turn" in text
    assert "not a hard gate" not in README
    assert "do not gate creation of a draft" not in AUTHOR_GUIDE
    assert "requirement, not an optional note" in AUTHOR_GUIDE


def test_readme_example_tasks_are_final():
    for task_id in EXAMPLE_TASKS:
        domain, name = task_id.split(".", maxsplit=1)
        metadata_path = ROOT / "tasks" / domain / name / "task_meta.yaml"
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))

        assert task_id in README
        assert metadata["id"] == task_id
        assert metadata["status"] == "final"


def test_readme_does_not_advertise_abandoned_examples():
    for task_id in (
        "math.toeplitz",
        "physics.poisson_capacitor_field",
        "computer_science.diffusion_mode_audit",
    ):
        assert task_id not in README


def test_agent_docs_distinguish_new_models_from_new_harnesses():
    for document in (README, GETTING_STARTED):
        assert "New model, existing harness" in document
        assert "New agent harness" in document
        assert "`--agent-cmd`" in document
        assert "`api_protocol`" in document

    assert re.search(r"first-class built-in\s+integration", README, re.IGNORECASE)
    assert re.search(r"compatible\s+adapters", GETTING_STARTED, re.IGNORECASE)


def test_docs_explain_low_direct_llm_scores_and_agentic_alternatives():
    assert "docs/guide/troubleshooting-results.md" in README
    assert "troubleshooting-results.md" in GETTING_STARTED
    assert "single-turn" in TROUBLESHOOTING
    assert "no agentic tools" in TROUBLESHOOTING
    for adapter in ("codex_cli", "claude_code_cli", "kimi_code_cli"):
        assert f"`{adapter}`" in TROUBLESHOOTING


def test_readme_fences_do_not_swallow_markdown_sections():
    """Headings must remain outside CommonMark fenced code blocks."""
    fence: tuple[str, int] | None = None
    swallowed_headings: list[tuple[int, str]] = []

    for line_number, line in enumerate(README.splitlines(), start=1):
        if fence is None:
            opening = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
            if opening:
                marker = opening.group(1)
                fence = (marker[0], len(marker))
            continue

        marker, minimum_length = fence
        if re.match(
            rf"^ {{0,3}}{re.escape(marker)}{{{minimum_length},}}[ \t]*$", line
        ):
            fence = None
        elif re.match(r"^#{2,6}\s", line):
            swallowed_headings.append((line_number, line))

    assert fence is None, "README contains an unclosed fenced code block"
    assert swallowed_headings == [], (
        f"README headings are inside fenced code: {swallowed_headings}"
    )
