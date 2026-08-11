"""HuggingFace integration: pull pre-generated instances from HF datasets.

The CLI exposes this as ``asibench task pull``.

GitHub stores code/config; HuggingFace stores instance-level data. This package
only *reads* public/private instance data (prompts + input data, and for the
self-contained demo repo also reference answers). It never carries the private
scoring rules (``task_eval.yaml`` / ``evaluation``) — those stay in the private
scoring service and are used only after submission to the ASI-Bench website.
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
