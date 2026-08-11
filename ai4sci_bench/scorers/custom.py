"""Custom scorer loader — loads task-specific scorers from task directories."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def load_custom_scorer(task_dir: Path) -> None:
    """Load custom_scorer.py from a task directory.

    The custom scorer module should use @register_scorer to register
    its scorer classes into the global registry.
    """
    scorer_path = task_dir / "custom_scorer.py"
    if not scorer_path.exists():
        return

    spec = importlib.util.spec_from_file_location(
        f"custom_scorer_{task_dir.name}", scorer_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
