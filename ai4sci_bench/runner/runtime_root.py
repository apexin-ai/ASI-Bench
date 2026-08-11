"""Resolve a writable runtime root for source and wheel installations."""

from __future__ import annotations

from pathlib import Path

from ai4sci_bench.branding import config_path


def _source_project_root(path: Path) -> Path | None:
    """Find an ASI-Bench source checkout containing *path*, if one exists."""
    candidate = path.resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for parent in (candidate, *candidate.parents):
        project_file = (parent / "pyproject.toml").is_file() or (
            parent / "setup.py"
        ).is_file()
        if project_file and (parent / "ai4sci_bench").is_dir():
            return parent
    return None


def resolve_runtime_root(*hints: Path) -> Path:
    """Return the checkout root, or a writable per-user wheel runtime directory.

    ``__file__`` ancestors are not assumed to be a repository: for wheel
    installations they point into ``site-packages``.  Caller-provided paths and
    the current directory are checked for a real source checkout first.
    """
    package_location = Path(__file__)
    for candidate in (*hints, Path.cwd(), package_location):
        root = _source_project_root(Path(candidate))
        if root is not None:
            return root

    root = config_path("runtime")
    root.mkdir(parents=True, exist_ok=True)
    return root
