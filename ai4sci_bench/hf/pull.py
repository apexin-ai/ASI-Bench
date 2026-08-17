"""Pull pre-generated task instances from a HuggingFace dataset repo.

Supported layouts on HF::

    tasks/<instance_id>/
        prompt_b*.md
        data/...
        task_info.json
        framework_task_info.json  # optional; private scoring datasets only
        instance_meta.json      # demo only (carries params_used)
        reference/...           # demo only (self-contained)

where ``<instance_id>`` is ``"<domain>.<name>__<params/seed>"`` — the same flat
directory name the run-side loaders expect.

``pull_instances`` downloads the requested tasks and copies every instance
directory into ``<output_dir>/<instance_id>/`` so the result can be fed straight
to ``asibench run --instances-dir`` (whose loader globs
``<instances_dir>/<task_id>__*``).

Security: this module only reads instance *data*. It never fetches or writes GT
generators, generation settings, or reference answers. The HF token is read from
the ``token`` argument or the ``HF_TOKEN`` / ``HUGGINGFACE_HUB_TOKEN``
environment variables — never hardcoded. Public repos need no token.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from ai4sci_bench.branding import (
    HF_ORG as DEFAULT_ORG,
    HF_REPO_ALIASES as REPO_ALIASES,
    HF_TOKEN_ENV_VARS as _TOKEN_ENV_VARS,
)


@dataclass
class PullResult:
    """Outcome of a pull: where instances landed and what was found."""

    repo_id: str
    output_dir: Path
    instance_ids: list[str] = field(default_factory=list)
    task_ids: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def resolve_repo_id(repo: str) -> str:
    """Map a known dataset alias to a full HF repo id.

    A value already containing ``/`` is treated as an explicit ``org/name`` id
    and returned unchanged.
    """
    if "/" in repo:
        return repo
    try:
        return REPO_ALIASES[repo]
    except KeyError:
        raise ValueError(
            f"Unknown repo alias {repo!r}. "
            f"Use one of {sorted(REPO_ALIASES)} or a full 'org/name' id."
        ) from None


def resolve_token(token: str | None) -> str | None:
    """Resolve an HF token from the argument or environment; never hardcoded."""
    if token:
        return token
    for env in _TOKEN_ENV_VARS:
        val = os.environ.get(env)
        if val:
            return val
    return None


def _allow_patterns_for_tasks(task_ids: list[str] | None) -> list[str] | None:
    """Translate ``domain.name`` task ids into HF snapshot allow-patterns.

    Both fixed-seed datasets store instances below ``tasks/``. The ``__``
    delimiter keeps ``foo.bar`` from also matching ``foo.bar_baz``.
    """
    if not task_ids:
        return None
    patterns: list[str] = []
    for tid in task_ids:
        if "." not in tid:
            raise ValueError(
                f"Task id {tid!r} must be '<domain>.<name>' (e.g. 'physics.example_task')."
            )
        patterns.append(f"tasks/{tid}__*/**")
        patterns.append(f"tasks/{tid}/**")
    return patterns


def pull_instances(
    repo: str,
    *,
    output_dir: str | os.PathLike[str],
    tasks: list[str] | None = None,
    token: str | None = None,
    revision: str = "main",
    overwrite: bool = False,
) -> PullResult:
    """Download instances from an HF dataset repo and flatten them locally.

    Parameters
    ----------
    repo:
        Known alias (``demo``, ``seed42``, or ``seed31415``) or
        full ``org/name`` repo id.
    output_dir:
        Directory to place ``<instance_id>/`` folders into (created if missing).
    tasks:
        Optional list of ``domain.name`` task ids to restrict the download.
        ``None`` pulls every task in the repo.
    token:
        HF token; falls back to ``HF_TOKEN`` env. Public repos need none.
    revision:
        Git revision (branch/tag/sha) to pull. Defaults to ``main``.
    overwrite:
        If ``True``, replace an existing ``<instance_id>/`` dir; otherwise skip it.
    """
    # Imported lazily so the rest of the CLI works without huggingface_hub installed.
    from huggingface_hub import snapshot_download

    repo_id = resolve_repo_id(repo)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    snapshot_path = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            token=resolve_token(token),
            allow_patterns=_allow_patterns_for_tasks(tasks),
        )
    )

    result = PullResult(repo_id=repo_id, output_dir=out)
    seen_tasks: set[str] = set()
    requested_task_ids = set(tasks) if tasks else None

    # The public dataset contract is tasks/<instance_id>/ for every seed.
    tasks_root = snapshot_path / "tasks"
    candidates = (
        {
            path.name: path
            for path in tasks_root.iterdir()
            if path.is_dir() and "__" in path.name and "." in path.name.split("__", 1)[0]
        }
        if tasks_root.is_dir()
        else {}
    )

    # Copy each instance into output_dir under the same name (the shape run expects).
    # Filter here as well as in snapshot_download(): a reused HF snapshot can
    # retain directories fetched by an earlier, broader allow_patterns call.
    for instance_id, inst_dir in sorted(candidates.items()):
        # <instance_id> == "<domain>.<name>__<suffix>" -> task id "<domain>.<name>"
        task_id = instance_id.split("__", 1)[0]
        if requested_task_ids is not None and task_id not in requested_task_ids:
            continue
        dest = out / instance_id
        if dest.exists():
            if not overwrite:
                result.skipped.append(instance_id)
                continue
            shutil.rmtree(dest)
        # snapshot files are symlinks into the HF cache, so follow them and
        # materialize real files in output_dir.
        shutil.copytree(inst_dir, dest, symlinks=False)
        result.instance_ids.append(instance_id)
        seen_tasks.add(task_id)

    result.task_ids = sorted(seen_tasks)
    return result
