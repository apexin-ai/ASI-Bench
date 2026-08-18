"""HuggingFace integration: pull pre-generated instances from HF datasets.

The CLI exposes this as ``asibench task pull``.

GitHub stores catalog metadata and public scoring logic; HuggingFace stores
instance-level prompts and input data. seed31415 additionally stores public
references for local scoring. seed42 references, all ground-truth generators,
generation settings, reference specifications, and private solver assets remain
on the scoring service.
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
