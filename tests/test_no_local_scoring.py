"""Local scoring is public only for seed31415; seed42 remains website-owned."""

from importlib.util import find_spec
from pathlib import Path
import ast


ROOT = Path(__file__).resolve().parents[1]


def test_private_submission_scoring_module_is_not_distributed():
    assert not (ROOT / "ai4sci_bench" / "submission" / "score.py").exists()
    assert find_spec("ai4sci_bench.submission.score") is None


def test_public_local_scoring_module_is_distributed():
    assert (ROOT / "ai4sci_bench" / "local_scoring.py").is_file()
    assert find_spec("ai4sci_bench.local_scoring") is not None


def test_local_scoring_defers_optional_scorer_imports():
    """The lightweight wheel must reject seed42 without importing NumPy/scorers."""
    source = ROOT / "ai4sci_bench" / "local_scoring.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    top_level_imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert not any(
        isinstance(node, ast.ImportFrom)
        and (node.module or "").startswith("ai4sci_bench.scorers")
        for node in top_level_imports
    )
