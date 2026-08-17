"""Package a produce-only run into a submission bundle for official scoring.

The public repository exposes task contracts and scoring logic, while prompts
and inputs are pulled separately. Ground-truth generators, generation settings,
and reference answers stay in the website scoring service. External runners
therefore: ``pull`` → ``run`` (produce output) → authenticated ``submit`` →
confirm the draft on ``https://asibench.apexin.ai/`` for scoring.

``build_submission`` collects the public result JSON (identity, resolved params,
provenance) and the agent's actual output artifacts. It preserves a
``framework_task_info.json`` snapshot when one is available, but public
fixed-seed runs do not require that private scoring-side file. Every bundled
file is hashed so the receiving service can verify integrity.
"""

from __future__ import annotations

from ai4sci_bench.submission.bundle import (
    SUBMISSION_SCHEMA_VERSION,
    BundleResult,
    build_submission,
)
from ai4sci_bench.submission.upload import UploadResult, upload_bundle
from ai4sci_bench.submission.task_submit import (
    TaskSubmitResult,
    collect_task_files,
    submit_task,
)

__all__ = [
    "SUBMISSION_SCHEMA_VERSION",
    "BundleResult",
    "build_submission",
    "UploadResult",
    "upload_bundle",
    "TaskSubmitResult",
    "collect_task_files",
    "submit_task",
]
