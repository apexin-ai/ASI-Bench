"""Public benchmark execution must not expose maintainer-side scoring APIs."""

from importlib.util import find_spec
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_submission_scoring_module_is_not_distributed():
    assert not (ROOT / "ai4sci_bench" / "submission" / "score.py").exists()
    assert find_spec("ai4sci_bench.submission.score") is None
