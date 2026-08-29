"""Package a produce-only run into a submission bundle for official scoring.

The public repository exposes task contracts and scoring logic, while prompts
and inputs are pulled separately. Submission bundles never contain references:
the website joins seed42 outputs to private references. External runners use
``pull`` → ``run`` → authenticated ``submit`` → browser confirmation for
seed42. seed31415 may instead use the separate public local-scoring command.

``build_submission`` collects the public result JSON (identity, resolved params,
provenance) and the agent's actual output artifacts. It preserves a
``framework_task_info.json`` snapshot when one is available, but public
fixed-seed runs do not require that private scoring-side file. Every bundled
file is hashed so the receiving service can verify integrity. Official seed42
bundles require verified Docker OS provenance, including the image identity.
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
    validate_task_os_evidence,
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
    "validate_task_os_evidence",
]
