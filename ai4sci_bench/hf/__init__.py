"""HuggingFace integration: pull pre-generated instances from HF datasets.

The CLI exposes this as ``asibench task pull``.

GitHub stores catalog metadata and public scoring logic; HuggingFace stores
instance-level prompts and input data. This package only reads instance data.
Ground-truth generators, generation settings, reference specifications,
reference answers, and private solver assets remain on the scoring service and
are used only after submission to the ASI-Bench website.
"""

from ai4sci_bench.hf.pull import (
    REPO_ALIASES,
    PullResult,
    pull_instances,
    resolve_repo_id,
)

__all__ = [
    "REPO_ALIASES",
    "PullResult",
    "pull_instances",
    "resolve_repo_id",
]
