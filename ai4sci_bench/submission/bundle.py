"""Build a self-contained submission bundle from a produce-only run.

Input layout (an ``asibench run`` output directory)::

    <results-dir>/
        run_metadata.json
        <task_id>/
            <base>.json                 # per-instance result JSON (unscored)
            <base>.outputs/...          # persisted agent output artifacts
        instances/
            <instance_id>/framework_task_info.json  # optional; private runs only

Output bundle layout::

    <bundle>/
        manifest.json
        run_metadata.json               # copied verbatim when present
        instances/
            <base>/
                result.json
                framework_task_info.json   # when found
                outputs/<declared outputs...>

The bundle is what the ASI-Bench website scores against its private instance
metadata and reference answers, using the published scoring contract. A public
run must carry the agent's declared outputs (bytes, not just paths), result
identity, resolved params, and provenance; it does not need
``framework_task_info.json``. No GT-generation material or reference answers
ever enter a public produce-only bundle.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tarfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ai4sci_bench import __version__ as FRAMEWORK_VERSION
from ai4sci_bench.branding import HF_REPO_ALIASES

SUBMISSION_SCHEMA_VERSION = 1
OFFICIAL_SUBMISSION_SEED = "seed42"

_SEED_SUFFIX_RE = re.compile(r"(?:^|_)seed(?P<number>[0-9]+)$")

# Result-directory files that are not per-instance result JSONs.
_NON_RESULT_JSON = {
    "run_metadata.json",
    "eval_config.json",
    "task_info.json",
    "framework_task_info.json",
    "instance_meta.json",
}


@dataclass
class InstanceEntry:
    """One evaluated instance as it appears in the bundle manifest."""

    base_name: str
    instance_id: str
    task_id: str
    prompt_level: str
    attempt: int
    agent_name: str
    status: str
    result_file: str
    framework_task_info_file: str | None
    output_files: list[dict] = field(default_factory=list)
    missing_outputs: list[str] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)


@dataclass
class BundleResult:
    """Outcome of :func:`build_submission`."""

    bundle_dir: Path
    archive_path: Path | None
    instance_count: int
    instances: list[InstanceEntry] = field(default_factory=list)
    run_metadata_included: bool = False

    @property
    def instances_with_missing_outputs(self) -> list[InstanceEntry]:
        return [e for e in self.instances if e.missing_outputs]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_result_files(results_dir: Path):
    """Yield per-instance result JSON files under ``results_dir``.

    A result JSON lives at ``<results-dir>/<task_id>/<base>.json`` and carries
    both ``task_id`` and ``instance_id``. Sidecars (``*.trajectory.json``,
    ``*.agent_model_output.*``) and run/task metadata are skipped.
    """
    for json_file in sorted(results_dir.glob("*/*.json")):
        if json_file.name in _NON_RESULT_JSON:
            continue
        if ".trajectory." in json_file.name or ".agent_model_output." in json_file.name:
            continue
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        if "task_id" not in data or "instance_id" not in data:
            continue
        yield json_file, data


def _find_framework_task_info(results_dir: Path, instance_id: str) -> Path | None:
    """Locate the framework_task_info.json snapshot for an instance, if any."""
    candidate = results_dir / "instances" / instance_id / "framework_task_info.json"
    if candidate.is_file():
        return candidate
    # Fall back to a shallow search (pulled instances may nest differently).
    for match in results_dir.glob(f"instances/{instance_id}/**/framework_task_info.json"):
        if match.is_file():
            return match
    return None


def _validate_official_submission(
    records: list[tuple[Path, dict]],
    benchmark_repo: str | None,
) -> None:
    """Reject anything outside the private-reference seed42 contract."""
    missing_seed: list[str] = []
    wrong_seed: list[str] = []
    invalid_sandbox: list[str] = []
    for _, data in records:
        instance_id = str(data.get("instance_id") or "")
        match = _SEED_SUFFIX_RE.search(instance_id)
        if match is None:
            missing_seed.append(instance_id or "<missing instance_id>")
        elif match.group("number") != "42":
            wrong_seed.append(f"{instance_id} (seed{match.group('number')})")

        provenance = data.get("provenance")
        sandbox = provenance.get("sandbox") if isinstance(provenance, dict) else None
        sandbox = sandbox if isinstance(sandbox, dict) else {}
        if not (
            sandbox.get("effective_mode") == "os"
            and sandbox.get("enforcement_status") == "fail_closed"
            and sandbox.get("verification_status") == "docker_container"
            and isinstance(sandbox.get("image_identity"), str)
            and sandbox["image_identity"].strip()
        ):
            invalid_sandbox.append(instance_id or "<missing instance_id>")

    if missing_seed:
        shown = ", ".join(missing_seed[:5])
        raise ValueError(
            "Official result submission requires every instance_id to end in an "
            f"official seed suffix; expected {OFFICIAL_SUBMISSION_SEED}. Invalid: {shown}"
        )
    if wrong_seed:
        shown = ", ".join(wrong_seed[:5])
        raise ValueError(
            f"Official result submission only accepts {OFFICIAL_SUBMISSION_SEED}; "
            f"score seed31415 locally with `asibench score`. Rejected: {shown}"
        )
    if invalid_sandbox:
        shown = ", ".join(invalid_sandbox[:5])
        raise ValueError(
            "Official seed42 result submission requires every result to be "
            "produced with `asibench run --sandbox os` and carry verified Docker "
            f"provenance, including an image identity. Invalid: {shown}"
        )

    if benchmark_repo is not None:
        allowed_repos = {
            OFFICIAL_SUBMISSION_SEED,
            HF_REPO_ALIASES[OFFICIAL_SUBMISSION_SEED],
        }
        if benchmark_repo not in allowed_repos:
            raise ValueError(
                "benchmark_repo must identify the official seed42 dataset "
                f"({HF_REPO_ALIASES[OFFICIAL_SUBMISSION_SEED]}), got "
                f"{benchmark_repo!r}"
            )


def build_submission(
    results_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    note: str | None = None,
    benchmark_repo: str | None = None,
    archive: bool = True,
) -> BundleResult:
    """Package a produce-only run into a submission bundle.

    Parameters
    ----------
    results_dir:
        An ``asibench run`` output directory.
    output_path:
        Destination. If it ends in ``.tar.gz``/``.tgz`` it is used as the
        archive path (bundle dir derived by stripping the suffix). Otherwise it
        is treated as the bundle directory. Defaults to
        ``<results-dir>/submission_<UTC timestamp>``.
    note:
        Free-form submitter note recorded in the manifest (e.g. team / model).
    benchmark_repo:
        Optional HF repo id the run was pulled from, recorded for provenance.
    archive:
        When ``True`` (default) also produce a ``.tar.gz`` next to the dir.

    Raises
    ------
    FileNotFoundError
        If ``results_dir`` does not exist.
    ValueError
        If no per-instance result JSON is found (nothing to submit).
    """
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    records = list(_iter_result_files(results_dir))
    if not records:
        raise ValueError(
            f"No per-instance result JSON found under {results_dir}. "
            "Run `asibench run ...` first."
        )
    _validate_official_submission(records, benchmark_repo)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if output_path is not None:
        output_path = Path(output_path)
        if output_path.name.endswith((".tar.gz", ".tgz")):
            archive_path = output_path
            suffix = ".tar.gz" if output_path.name.endswith(".tar.gz") else ".tgz"
            bundle_dir = output_path.with_name(output_path.name[: -len(suffix)])
        else:
            bundle_dir = output_path
            archive_path = bundle_dir.with_name(bundle_dir.name + ".tar.gz")
    else:
        bundle_dir = results_dir / f"submission_{stamp}"
        archive_path = bundle_dir.with_name(bundle_dir.name + ".tar.gz")

    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    (bundle_dir / "instances").mkdir(parents=True, exist_ok=True)

    entries: list[InstanceEntry] = []
    for json_file, data in records:
        base_name = json_file.stem
        task_dir = json_file.parent
        inst_bundle = bundle_dir / "instances" / base_name
        inst_bundle.mkdir(parents=True, exist_ok=True)

        # 1. Result JSON — identity, resolved params, provenance, status.
        shutil.copy2(json_file, inst_bundle / "result.json")

        # 2. Declared output artifacts (the bytes scoring runs against).
        agent_output = data.get("agent_output") or {}
        persisted = agent_output.get("persisted_outputs") or {}
        output_files: list[dict] = []
        missing: list[str] = []
        src_outputs = task_dir / f"{base_name}.outputs"
        for rec in persisted.get("files", []):
            rel = rec.get("path")
            if not rel:
                continue
            src = src_outputs / rel
            if rec.get("missing") or not src.is_file():
                missing.append(rel)
                output_files.append({"path": rel, "missing": True})
                continue
            dest = inst_bundle / "outputs" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            output_files.append({
                "path": f"outputs/{rel}",
                "sha256": _sha256_file(dest),
                "bytes": dest.stat().st_size,
            })

        # 3. framework_task_info.json — expected outputs + params snapshot.
        fti_rel: str | None = None
        fti_src = _find_framework_task_info(results_dir, data["instance_id"])
        if fti_src is not None:
            shutil.copy2(fti_src, inst_bundle / "framework_task_info.json")
            fti_rel = f"instances/{base_name}/framework_task_info.json"

        entries.append(InstanceEntry(
            base_name=base_name,
            instance_id=data["instance_id"],
            task_id=data["task_id"],
            prompt_level=data.get("prompt_level", ""),
            attempt=data.get("attempt", 1),
            agent_name=data.get("agent_name", ""),
            status=data.get("status", ""),
            result_file=f"instances/{base_name}/result.json",
            framework_task_info_file=fti_rel,
            output_files=output_files,
            missing_outputs=missing,
            provenance=data.get("provenance", {}),
        ))

    # Copy run-level metadata verbatim when present.
    run_meta_included = False
    run_meta = results_dir / "run_metadata.json"
    if run_meta.is_file():
        shutil.copy2(run_meta, bundle_dir / "run_metadata.json")
        run_meta_included = True

    # Flat integrity list over every file actually written into the bundle.
    files_index: list[dict] = []
    for path in sorted(bundle_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files_index.append({
                "path": str(path.relative_to(bundle_dir).as_posix()),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            })

    manifest = {
        "submission_schema_version": SUBMISSION_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "framework_version": FRAMEWORK_VERSION,
        "benchmark_repo": benchmark_repo,
        "note": note,
        "run_metadata_included": run_meta_included,
        "instance_count": len(entries),
        "instances_missing_outputs": [
            e.base_name for e in entries if e.missing_outputs
        ],
        "instances": [
            {
                "base_name": e.base_name,
                "instance_id": e.instance_id,
                "task_id": e.task_id,
                "prompt_level": e.prompt_level,
                "attempt": e.attempt,
                "agent_name": e.agent_name,
                "status": e.status,
                "result_file": e.result_file,
                "framework_task_info_file": e.framework_task_info_file,
                "output_files": e.output_files,
                "missing_outputs": e.missing_outputs,
                "provenance": e.provenance,
            }
            for e in entries
        ],
        "files": files_index,
    }
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    final_archive: Path | None = None
    if archive:
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(bundle_dir, arcname=bundle_dir.name)
        final_archive = archive_path

    return BundleResult(
        bundle_dir=bundle_dir,
        archive_path=final_archive,
        instance_count=len(entries),
        instances=entries,
        run_metadata_included=run_meta_included,
    )
