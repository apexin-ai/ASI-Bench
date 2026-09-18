"""ASI seed31415 contract, single Docker attempt and frozen output export.

Uses native BenchFlow Agent/ACP lifecycle without Evaluation or scoring.
Result paths match ASI; the adapter result schema is explicitly distinct.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import stat
import tempfile
import traceback
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import Any
from uuid import uuid4

from benchflow.rollout_planes import DefaultRolloutPlanes
from benchmarks.asi_bench.prepare import (
    InventoryError,
    build_task_info,
    check_inventory,
    is_inventory_noise,
)


@dataclass(frozen=True)
class OutputSpec:
    name: str
    type: str


def output_specs(metadata: dict[str, Any]) -> tuple[OutputSpec, ...]:
    """Use ASI's authority (task metadata), restricted to safe code/data files."""
    entries = metadata.get("output", {}).get("files", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError("output.files must be a nonempty list")
    result = []
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("output.files entries must be objects")
        name = entry.get("name")
        if (not isinstance(name, str) or not name or "\\" in name
                or ":" in name or any(ord(c) < 32 for c in name)
                or any(part in {"", ".", ".."} for part in name.split("/"))):
            raise ValueError(f"unsafe output path: {name!r}")
        if name.split("/")[0] in {"data", "prompt.md", "task_info.json"}:
            raise ValueError(f"output overlaps task input: {name}")
        if name.casefold() in seen:
            raise ValueError(f"duplicate output: {name}")
        seen.add(name.casefold())
        kind = entry.get("type", "data")
        if kind not in {"code", "data"}:
            raise ValueError(f"unsupported output type: {kind!r}")
        result.append(OutputSpec(name, kind))
    for name in seen:
        if any(name.startswith(other + "/") for other in seen):
            raise ValueError("output file paths overlap")
    return tuple(result)


def _regular_file(root: Path, name: str) -> Path:
    path = root
    parts = name.split("/")
    for i, part in enumerate(parts):
        path = path / part
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError(f"symlink is not an artifact/input: {name}")
        expected = stat.S_ISREG if i == len(parts) - 1 else stat.S_ISDIR
        if not expected(mode):
            raise ValueError(f"not a regular file path: {name}")
    return path


def load_contract(prepared_dir: Path, metadata: dict[str, Any], level: str = "b1") -> tuple[str, tuple[OutputSpec, ...], int]:
    """Read exact UTF-8 prompt text and verify task_info against trusted metadata.

    Returns prompt, output specs and the validated timeout (no second JSON read).
    Full source/manifest integrity checks are the S6 caller's responsibility.
    """
    root = Path(prepared_dir).resolve(strict=True)
    task_id = metadata.get("id")
    specs = output_specs(metadata)
    info_path = _regular_file(root, "environment/inputs/task_info.json")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if not isinstance(info, dict):
        raise ValueError("task_info must be an object")
    timeout = info.get("timeout_seconds")
    if type(timeout) is not int or timeout <= 0:
        raise ValueError("timeout_seconds must be a positive integer")
    expected = build_task_info(task_id, level, metadata, timeout)
    if info != expected:
        raise ValueError("task_info differs from anonymous ASI input/output contract")
    prompt_path = _regular_file(root, "environment/inputs/prompt.md")
    # Match upstream read_text UTF-8 semantics, not TaskDocument's strip().
    prompt = prompt_path.read_text(encoding="utf-8")
    if not prompt.strip():
        raise ValueError("empty prompt")
    return prompt, specs, timeout


def collect_outputs(workspace: Path, specs: tuple[OutputSpec, ...]) -> dict[str, Any]:
    """Describe declared files in an already frozen local workspace.

    Missing/invalid predictions do not change attempt status. I/O errors propagate
    as collection failures. No code/data validation or scientific scoring occurs.
    """
    root = Path(workspace).resolve(strict=True)
    # Revalidate caller-provided specs, not only load_contract's return value.
    specs = output_specs({"output": {"files": [vars(s) for s in specs]}})
    artifacts, missing, invalid = [], [], []
    for spec in specs:
        try:
            path = _regular_file(root, spec.name)
        except FileNotFoundError:
            missing.append(spec.name)
            continue
        except ValueError as exc:
            invalid.append({"name": spec.name, "reason": str(exc)})
            continue
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        artifacts.append({"name": spec.name, "type": spec.type,
                          "size": size, "sha256": digest.hexdigest()})
    return {"artifacts": artifacts, "missing": missing, "invalid": invalid,
            "prediction_status": "invalid" if invalid else "missing" if missing else "present"}


def runtime_override(profile: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
    """Map cpu-v1 task no-web policy, not a blanket CLI API network ban.

    Native Rollout retains model connectivity and enforces its no-web policy.
    Prepared files stay read-only; network=False describes task/tool permission.
    """
    required = {"profile_id": "cpu-v1", "variant": "os", "workdir": "/workspace",
                "base_image": "python:3.11-slim", "network": False}
    if any(profile.get(k) != v for k, v in required.items()) or profile.get("network") is not False:
        raise ValueError("unsupported runtime profile; expected cpu-v1 with network=false")
    for key in ("cpus", "memory_mb"):
        if type(profile.get(key)) is not int or profile[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if type(timeout_seconds) is not int or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be a positive integer")
    return {"sandbox": {"workdir": "/workspace", "cpus": profile["cpus"],
                        "memory_mb": profile["memory_mb"], "network_mode": "no-network",
                        "allow_internet": False},
            "agent": {"timeout_sec": timeout_seconds, "prompt_prefix": None}}


def validate_runtime_launch(sandbox: Any, *, backend: str, preserve_agent_network: bool) -> None:
    """Validate the task policy before native setup applies the CLI exception.

    Both network-preserving CLI and offline paths are valid. This adapter does
    not bypass BenchFlow's provider proxy/firewall or no-web tool restrictions.
    """
    if backend != "docker" or type(preserve_agent_network) is not bool:
        raise ValueError("ASI runtime requires Docker and an explicit network decision")
    if sandbox.allow_internet or sandbox.network_mode != "no-network" or sandbox.workdir != "/workspace":
        raise ValueError("effective sandbox does not enforce the requested runtime policy")


def attempt_result(outcome: str) -> dict[str, str | None]:
    """Map an execution outcome to persisted attempt status and failure reason."""
    if outcome not in {"completed", "agent_error", "timeout", "setup_error", "cancelled", "execution_error"}:
        raise ValueError(f"unknown attempt outcome: {outcome}")
    return {
        "attempt_status": "completed" if outcome == "completed" else "execution_failed",
        "failure_reason": None if outcome == "completed" else outcome,
    }


# Keep the supported backend boundary; delegate network policy to native Rollout.
class ASIDockerPlanes(DefaultRolloutPlanes):
    """ASI CLI API-network exception with BenchFlow native no-web enforcement."""
    def create_environment(
        self, environment, task, task_path, rollout_name, rollout_paths, *,
        preserve_agent_network, environment_manifest,
    ):
        if environment_manifest is not None:
            raise ValueError("S5 does not support environment manifests")
        validate_runtime_launch(task.config.sandbox, backend=environment,
                                preserve_agent_network=preserve_agent_network)
        launch_path = task_path
        if preserve_agent_network:
            if rollout_paths is None:
                raise ValueError("ASI execution view requires rollout paths")
            # Native custom Compose support, in a run-owned copy, not prepared.
            # Keep the original Task for prompt/config/verifier handling.
            # Native Docker derives the image name from this directory name.
            # tempfile suffixes may end with an underscore (invalid image ref).
            launch_path = Path(rollout_paths.rollout_dir) / ("asi-execution-" + uuid4().hex)
            launch_path.mkdir(mode=0o700)
            context = launch_path / "environment"
            shutil.copytree(
                Path(task_path) / "environment", context,
                ignore=lambda parent, names: [name for name in names
                                              if is_inventory_noise(Path(parent) / name)],
            )
            compose = context / "docker-compose.yaml"
            if compose.exists():
                raise ValueError("unexpected existing Compose configuration")
            compose.write_text(
                "services:\n  main:\n    cap_add:\n      - NET_ADMIN\n",
                encoding="utf-8",
            )
        return super().create_environment(
            environment, task, launch_path, rollout_name, rollout_paths,
            preserve_agent_network=preserve_agent_network, environment_manifest=None,
        )


@dataclass(frozen=True)
class PreparedContract:
    task_path: Path
    prompt: str
    outputs: tuple[OutputSpec, ...]
    inputs: tuple[str, ...]
    config_override: dict[str, Any]
    sources: dict[str, Any]


def inspect_prepared_task(prepared_dir: Path) -> PreparedContract:
    """Validate the current prepared package, without writing or executing it.

    This checks self-consistency, not authenticity of a locally modified
    manifest. Source re-download/signature verification is outside this helper.
    """
    import re

    import yaml

    from benchflow._utils.config_override import apply_config_override
    from benchflow.task import Task

    root = Path(prepared_dir).resolve(strict=True)
    manifest = json.loads(_regular_file(root, "task_manifest.json").read_text())
    task_id = manifest.get("task", {}).get("id", "")
    from benchmarks.asi_bench.prepare import _ids

    _ids(task_id, manifest.get("instance", {}).get("id", ""))
    if (manifest.get("schema_version") != 1 or manifest.get("status") != "prepared"
            or type(manifest.get("seed")) is not int or manifest["seed"] != 31415
            or manifest.get("task", {}).get("level") not in {"b1", "b2", "b3", "b4"}
            or manifest.get("instance", {}).get("id") != task_id + "__seed31415"):
        raise ValueError("expected prepared seed31415 task with B1-B4 level")
    if manifest.get("converter_version") != "prepare-v5":
        raise ValueError("expected prepare-v5; rerun prepare (old packages are not migrated)")
    scoring = manifest.get("scoring")
    expected_scoring_keys = {"profile_id", "launcher", "entrypoint",
                             "evaluator_spec_sha256", "expected_evaluator_bundle_digest"}
    if (not isinstance(scoring, dict) or set(scoring) != expected_scoring_keys
            or scoring.get("profile_id") != "asi-seed31415-v1"
            or scoring.get("launcher") != "verifier/test.sh"
            or scoring.get("entrypoint") != "verifier/score_entry.py"
            or any(not isinstance(scoring.get(k), str) or not re.fullmatch(r"[0-9a-f]{64}", scoring[k])
                   for k in ("evaluator_spec_sha256", "expected_evaluator_bundle_digest"))):
        raise ValueError("invalid prepared scoring contract")
    expected_paths = {"task_document": "task.md", "agent_inputs": "environment/inputs",
                      "instance_parameters": "verifier/instance_parameters.json"}
    if manifest.get("paths") != expected_paths:
        raise ValueError("unexpected prepared layout")
    expected_sources = {
        "hf": ("Apexintelligence-AI/ASI-Bench-seed31415", "tasks/" + task_id + "__seed31415"),
        "github": ("apexin-ai/ASI-Bench", "tasks/" + task_id.replace(".", "/")),
    }
    sources = manifest.get("sources", {})
    for name, (repo, path) in expected_sources.items():
        source = sources.get(name, {})
        if (source.get("repo") != repo or source.get("path") != path
                or not re.fullmatch(r"[0-9a-f]{40}", source.get("resolved_revision", ""))):
            raise ValueError(f"invalid {name} source identity")
    indexed = {}
    for entry in manifest.get("files", []):
        name = entry["target"]
        if (not isinstance(name, str) or "\\" in name
                or any(p in {"", ".", ".."} for p in name.split("/")) or name in indexed):
            raise ValueError("invalid/duplicate manifest target")
        indexed[name] = entry
    actual = set()
    for p in root.rglob("*"):
        if p.is_symlink():
            raise ValueError(f"prepared symlink: {p.relative_to(root)}")
        if not (p.is_file() or p.is_dir()):
            raise ValueError(f"unsafe prepared file: {p.relative_to(root)}")
        if p.is_file() and p != root / "task_manifest.json":
            actual.add(p.relative_to(root).as_posix())
    actual = check_inventory(root, actual, set(indexed), "prepared file inventory differs from manifest")
    for name, entry in indexed.items():
        p = _regular_file(root, name)
        if p.stat().st_size != entry["size"] or hashlib.sha256(p.read_bytes()).hexdigest() != entry["sha256"]:
            raise InventoryError("prepared integrity mismatch: " + json.dumps(name))
    verifier_files = {"verifier/instance_parameters.json", "verifier/test.sh", "verifier/score_entry.py", "verifier/scoring.py"}
    if {n for n in actual if n.startswith("verifier/")} != verifier_files:
        raise ValueError("unexpected verifier file inventory; rerun prepare")
    if any(indexed[n].get("visibility") != ["scoring"] for n in verifier_files):
        raise ValueError("verifier files must be scoring-only")
    from benchmarks.asi_bench.prepare import resolve_source_assets

    raw = resolve_source_assets(manifest)
    meta_path = "task_bundle/" + sources["github"]["path"] + "/task_meta.yaml"
    metadata = yaml.safe_load(_regular_file(raw, meta_path).read_text())
    prompt, outputs, timeout = load_contract(root, metadata, manifest["task"]["level"])
    from benchmarks.asi_bench.prepare import resolve_input_files

    declared_inputs = resolve_input_files(raw / 'instance' / manifest['instance']['id'], metadata)
    inputs = ('prompt.md', 'task_info.json', *(rel.as_posix() for _, rel in declared_inputs))
    prefix = "environment/inputs/"
    if {n[len(prefix):] for n in actual if n.startswith(prefix)} != set(inputs):
        raise ValueError("unexpected agent input inventory")
    if {n for n in actual if n.startswith("environment/")} != (
            {prefix + n for n in inputs} | {"environment/Dockerfile", "environment/runtime-profile.json", "environment/runtime-constraints.txt"}):
        raise ValueError("unexpected build context files")
    dockerfile = _regular_file(root, "environment/Dockerfile").read_text()
    from benchmarks.asi_bench.prepare import build_dockerfile
    from benchmarks.asi_bench.scorer import ADAPTER_ROOT

    if _regular_file(root, "environment/runtime-constraints.txt").read_bytes() != (ADAPTER_ROOT / "runtime-constraints.txt").read_bytes():
        raise ValueError("runtime constraints changed; rerun prepare")
    if dockerfile != build_dockerfile(metadata):
        raise ValueError("unsupported S5 Dockerfile")
    profile = json.loads(_regular_file(root, "environment/runtime-profile.json").read_text())
    if timeout != manifest["task"].get("timeout_seconds"):
        raise ValueError("timeout contract mismatch")
    override = runtime_override(profile, timeout)
    task = Task(root)
    if task.instruction != prompt.strip():
        raise ValueError("task.md instruction differs from prompt")
    effective = apply_config_override(task.config, override)
    validate_runtime_launch(effective.sandbox, backend="docker", preserve_agent_network=False)
    return PreparedContract(root, prompt, outputs, inputs, override, sources)


def rollout_contract_options(contract: PreparedContract) -> dict[str, Any]:
    """Options for S6's RolloutConfig; no Rollout is created or executed here.

    CLI agents retain native model connectivity; task web tools remain disabled.
    Native no-web proxy/firewall prerequisites still apply (also to dummy agents).
    """
    return {"task_path": contract.task_path, "environment": "docker",
            "prompts": [contract.prompt], "config_override": contract.config_override,
            "planes": ASIDockerPlanes(), "skip_verify": True}


@dataclass(frozen=True)
class ResultLayout:
    """ASI native result paths; calculation does not create or publish files."""

    task_dir: Path
    result_file: Path
    outputs_dir: Path
    run_metadata_file: Path
    benchflow_dir: Path


def result_layout(
    run_dir: Path, task_id: str, instance_id: str, level: str, attempt: int = 1,
) -> ResultLayout:
    """Match ASI _save_result naming for the seed31415 tasks and B1-B4 levels.

    An attempt suffix is a naming contract, not retry support. S6 must reserve a
    fresh run directory and publish atomically; these checks are not race-proof.
    """
    from benchmarks.asi_bench.prepare import _ids

    _ids(task_id, instance_id)
    if level not in {'b1', 'b2', 'b3', 'b4'}:
        raise ValueError('unsupported result prompt level')
    if type(attempt) is not int or attempt < 1:
        raise ValueError('attempt must be a positive integer')
    root = Path(run_dir).resolve()
    base = f'{instance_id}__{level}'
    if attempt > 1:
        base += f'__attempt{attempt}'
    task_dir = root / task_id
    layout = ResultLayout(task_dir, task_dir / f'{base}.json',
                          task_dir / f'{base}.outputs', root / 'run_metadata.json',
                          root / 'benchflow')
    for path in (task_dir, layout.result_file, layout.outputs_dir,
                 layout.run_metadata_file, layout.benchflow_dir):
        if path.is_symlink():
            raise ValueError(f'symlink in result destination: {path.name}')
        if path.exists():
            expect_directory = path in (task_dir, layout.outputs_dir, layout.benchflow_dir)
            if (expect_directory and not path.is_dir()) or (not expect_directory and not path.is_file()):
                raise ValueError(f'invalid result destination type: {path.name}')
    return layout


@dataclass(frozen=True)
class SolveConfig:
    """One Docker attempt; credentials are read only at execution time.

    Budgets include infrastructure separately from the task's native prompt
    timeout. No arbitrary Rollout/Compose/firewall overrides are accepted.
    """

    prepared_dir: Path
    run_dir: Path
    agent: str
    model: str
    effort: str | None = None
    model_env_file: Path | None = None
    lifecycle_timeout: float = 1800
    export_timeout: float = 120
    cleanup_timeout: float = 120
    max_artifact_bytes: int = 64 * 1024 * 1024


def reserve_run(run_dir: Path, prepared_dir: Path) -> Path:
    """Reserve an exclusive run directory outside immutable prepared/assets roots."""
    if Path(run_dir).is_symlink():
        raise ValueError('run directory must not be a symlink')
    run, prepared = Path(run_dir).resolve(), Path(prepared_dir).resolve(strict=True)
    if run == prepared or run.is_relative_to(prepared) or prepared.is_relative_to(run):
        raise ValueError('run directory overlaps prepared inputs')
    # The downloaded source is external to prepared; protect its default cache.
    raw = (Path.home() / '.cache/benchflow/asi_bench/downloads').resolve()
    if run == raw or run.is_relative_to(raw) or raw.is_relative_to(run):
        raise ValueError('run directory overlaps raw asset cache')
    for parent in run.parents:
        if (parent / 'sources.json').is_file() or (parent / 'task_manifest.json').is_file():
            raise ValueError('run directory is inside downloaded/prepared assets')
    run.mkdir(parents=True, exist_ok=False)
    run.chmod(0o700)
    return run


def decode_artifact_tar(payload: bytes, name: str, limit: int) -> bytes:
    """Decode docker cp's *single flat regular file*, never extract paths/links."""
    import io
    import tarfile

    if '/' in name or '\\' in name or name in {'', '.', '..'}:
        raise ValueError('S6 exports flat declared files only')
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode='r:') as archive:
            member = archive.next()
            if (member is None or member.name != name or not member.isreg()
                    or member.size > limit or member.size < 0 or member.issparse()):
                raise ValueError('invalid artifact archive member')
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError('artifact archive has no data')
            body = stream.read(limit + 1)
            if len(body) != member.size or len(body) > limit or archive.next() is not None:
                raise ValueError('artifact archive size/inventory mismatch')
            return body
    except (tarfile.TarError, EOFError) as exc:
        raise ValueError('invalid artifact archive') from exc


class UnsettledPhase(RuntimeError):
    """Cancellation did not settle a phase; publishing a snapshot is forbidden."""


async def _bounded(awaitable: Any, seconds: float) -> Any:
    """Join cancellation with a grace bound instead of unbounded wait_for."""
    import asyncio

    task = asyncio.ensure_future(awaitable)
    try:
        done, _ = await asyncio.wait({task}, timeout=max(0, seconds))
        if done:
            return task.result()
        raise TimeoutError('phase deadline exceeded')
    except BaseException:
        task.cancel()
        done, _ = await asyncio.wait({task}, timeout=5)
        if not done:
            # Consume a possible eventual exception, never use the late result.
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
            raise UnsettledPhase('phase did not settle after cancellation') from None
        if not task.cancelled():
            task.exception()
        raise


async def _docker(args: list[str], *, limit: int = 1024 * 1024) -> tuple[int, bytes, bytes]:
    """Bound Docker output; kill and join the CLI on cancellation/overflow.

    Used only for inspecting/stopping/exporting this Rollout's exact container.
    Lifecycle creation and teardown remain native BenchFlow operations.
    """
    import asyncio

    process = await asyncio.create_subprocess_exec(
        'docker', *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )

    async def read(stream: Any, cap: int) -> bytes:
        chunks, size = [], 0
        while chunk := await stream.read(65536):
            size += len(chunk)
            if size > cap:
                raise ValueError('Docker output exceeded byte limit')
            chunks.append(chunk)
        return b''.join(chunks)

    readers = [asyncio.create_task(read(process.stdout, limit)),
               asyncio.create_task(read(process.stderr, 65536))]
    try:
        out, err = await asyncio.gather(*readers)
        code = await process.wait()
        return code, out, err
    finally:
        for task in readers:
            if not task.done():
                task.cancel()
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()
        await process.wait()
        await asyncio.gather(*readers, return_exceptions=True)


async def _container_info(cid: str) -> dict[str, Any]:
    code, out, _ = await _docker(['inspect', cid])
    if code:
        raise RuntimeError('cannot inspect owned Docker container')
    return json.loads(out)[0]


async def _container_id(rollout: Any) -> str | None:
    import re

    env = getattr(rollout, '_env', None)
    if env is None:
        return None
    # Native Docker has no public container-id API. Isolate this single private
    # dependency here and test it against the installed core without modifying it.
    cid = await env._main_container_id()
    if cid is not None and not re.fullmatch(r'[0-9a-f]{64}', cid):
        raise ValueError('native Docker returned an invalid container identity')
    return cid


async def _runtime_evidence(rollout: Any, cid: str) -> dict[str, Any]:
    info = await _container_info(cid)
    host = info['HostConfig']
    if host['Privileged']:
        raise ValueError('privileged ASI container is forbidden')
    probe = await rollout.env.exec('id -u; grep CapEff /proc/self/status', user='agent')
    lines = probe.stdout.strip().splitlines()
    if probe.return_code or lines[0] == '0' or int(lines[1].split(':')[1], 16) != 0:
        raise ValueError('Agent must be non-root with zero effective capabilities')
    offline = host['NetworkMode'] == 'none'
    firewall = None
    if not offline:
        rules = await rollout.env.exec('iptables -S OUTPUT; ip6tables -S OUTPUT', user='root')
        if rules.return_code or rules.stdout.count(f'--uid-owner {lines[0]} -j REJECT') != 2:
            raise ValueError('native IPv4/IPv6 Agent firewall is missing')
        firewall = rules.stdout
    return {'task_network': False, 'preserve_agent_network': not offline,
            'network_mode': host['NetworkMode'], 'cap_add': host['CapAdd'],
            'privileged': False, 'agent_uid': int(lines[0]), 'agent_cap_eff': 0,
            'firewall': firewall, 'container_id': cid, 'image_id': info['Image']}


async def _freeze_export(cid: str, layout: ResultLayout, specs: tuple[OutputSpec, ...],
                         limit: int) -> dict[str, Any]:
    """Stop all container writers, validate each cp stream, then atomic rename.

    Workspace bind/volume mounts are rejected: stopping this container would
    not stop other writers to a shared mount. Transfer failure publishes nothing.
    Missing/invalid declarations are recorded separately from transfer failure.
    """
    info = await _container_info(cid)
    for mount in info.get('Mounts', []):
        path = Path(mount['Destination'])
        if path == Path('/workspace') or path.is_relative_to('/workspace') or Path('/workspace').is_relative_to(path):
            raise ValueError('cannot freeze shared workspace mount')
    code, _, _ = await _docker(['stop', '--time', '5', cid])
    if code:
        raise RuntimeError('cannot stop owned container for export')
    state = (await _container_info(cid))['State']
    if state.get('Running') is not False or state.get('Pid') != 0:
        raise RuntimeError('container writers are not stopped')
    layout.task_dir.mkdir(exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.' + layout.outputs_dir.name + '.staging-',
                                   dir=layout.task_dir))
    invalid = []
    expected_hashes = {}
    try:
        for spec in specs:
            code, payload, error = await _docker(
                ['cp', f'{cid}:/workspace/{spec.name}', '-'], limit=limit + 1024 * 1024,
            )
            if code:
                # Do not misreport daemon/transport failures as missing files.
                if b'Could not find the file ' in error and b' in container ' in error:
                    continue
                raise RuntimeError('artifact transfer failed')
            try:
                body = decode_artifact_tar(payload, spec.name, limit)
            except ValueError:
                invalid.append({'name': spec.name, 'reason': 'unsafe_artifact_archive'})
                continue
            path = staging / spec.name
            path.write_bytes(body)
            expected_hashes[spec.name] = hashlib.sha256(body).hexdigest()
        collected = collect_outputs(staging, specs)
        if {item['name']: item['sha256'] for item in collected['artifacts']} != expected_hashes:
            raise ValueError('artifact staging checksum mismatch')
        collected['freeze'] = {'method': 'docker-stop', 'container_id': cid,
                               'running': False, 'pid': 0, 'shared_workspace': False}
        collected['invalid'] = invalid
        invalid_names = {item['name'] for item in invalid}
        collected['missing'] = [name for name in collected['missing'] if name not in invalid_names]
        collected['prediction_status'] = ('invalid' if invalid else
                                          'missing' if collected['missing'] else 'present')
        if layout.outputs_dir.exists():
            raise FileExistsError('refusing to overwrite published artifacts')
        if collected['artifacts']:
            staging.rename(layout.outputs_dir)
        return collected
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def persisted_outputs(layout: ResultLayout, specs: tuple[OutputSpec, ...],
                      collection: dict[str, Any]) -> dict[str, Any] | None:
    """Serialize a validated collection; shared by S5 smoke and production S6."""
    if collection.get('prediction_status') == 'unavailable':
        return None
    found = {item['name']: item for item in collection.get('artifacts', [])}
    invalid = {item['name']: item['reason'] for item in collection.get('invalid', [])}
    files = []
    for spec in specs:
        if spec.name in found:
            item = found[spec.name]
            files.append({'path': spec.name, 'bytes': item['size'], 'sha256': item['sha256']})
        else:
            entry = {'path': spec.name, 'missing': True}
            if spec.name in invalid:
                entry['reason'] = invalid[spec.name]
            files.append(entry)
    return {'dir': layout.outputs_dir.name if found else None, 'files': files}


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    import os

    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        Path(name).replace(path)
    finally:
        Path(name).unlink(missing_ok=True)


def _model_route(config: SolveConfig) -> tuple[dict[str, str], dict[str, Any]]:
    import os
    from urllib.parse import urlsplit

    from dotenv import dotenv_values

    values = dotenv_values(config.model_env_file) if config.model_env_file else os.environ
    base, key = values.get('ASI_MODEL_BASE_URL'), values.get('ASI_MODEL_API_KEY')
    if bool(base) != bool(key):
        raise ValueError('ASI_MODEL_BASE_URL and ASI_MODEL_API_KEY must be set together')
    if config.model_env_file and not (base and key):
        raise ValueError('model env file is missing route configuration')
    env: dict[str, str] = {}
    if base:
        url = urlsplit(base)
        if url.scheme not in {'http', 'https'} or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError('model endpoint must be HTTP(S), without credentials/query/fragment')
        env = {'BENCHFLOW_PROVIDER_BASE_URL': base, 'BENCHFLOW_PROVIDER_API_KEY': key}
    return env, {'endpoint': base, 'api_key_env': 'ASI_MODEL_API_KEY' if key else None,
                 'routing': 'explicit' if env else 'native_environment'}


@dataclass(frozen=True)
class _FinishResult:
    native: Any
    cleanup: str
    collection: dict[str, Any]
    collection_status: str
    outcome: str


async def _finish_attempt(rollout: Any, cid: str | None, settled: bool, attempt: str,
                          layout: ResultLayout, specs: tuple[OutputSpec, ...],
                          config: SolveConfig, diagnostics: list[dict[str, str]]) -> _FinishResult:
    """Disconnect, freeze/export and finalize; preserve partial state on failure."""
    import asyncio

    native, cleanup, collection_status = None, 'pending', 'not_collected'
    collection = {'artifacts': [], 'missing': [], 'invalid': [], 'prediction_status': 'unavailable'}

    def diagnostic(stage: str, exc: BaseException) -> None:
        traceback.print_exception(exc)
        diagnostics.append({'phase': stage, 'error_type': type(exc).__name__})

    try:
        if rollout is None:
            cleanup = 'not_needed'
            return _FinishResult(native, cleanup, collection, collection_status, attempt)
        if cid is None:
            try:
                cid = await _bounded(_container_id(rollout), 15)
            except (Exception, asyncio.CancelledError) as exc:
                diagnostic('container_identity', exc)
        try:
            await _bounded(rollout.disconnect(), 20)
        except (Exception, asyncio.CancelledError) as exc:
            diagnostic('disconnect', exc)
            if attempt == 'completed':
                attempt = 'execution_error'
        if cid and settled:
            try:
                collection = await _bounded(_freeze_export(
                    cid, layout, specs, config.max_artifact_bytes,
                ), config.export_timeout)
                collection_status = 'collected'
            except (Exception, asyncio.CancelledError) as exc:
                collection_status = 'failed'
                diagnostic('export', exc)
        try:
            native = await _bounded(rollout.finalize(), config.cleanup_timeout)
            cleanup = 'completed'
        except (Exception, asyncio.CancelledError) as exc:
            cleanup = 'failed'
            diagnostic('finalize', exc)
            try:
                await _bounded(rollout.cleanup(), config.cleanup_timeout)
            except (Exception, asyncio.CancelledError) as secondary:
                diagnostic('cleanup', secondary)
        if cid:
            try:
                code, out, _ = await _bounded(_docker(
                    ['ps', '-a', '-q', '--no-trunc', '--filter', f'id={cid}'],
                ), 15)
                if code:
                    raise RuntimeError('cannot verify container cleanup')
                if out.strip():
                    cleanup = 'failed'
                    diagnostics.append({'phase': 'cleanup', 'error_type': 'ContainerRemaining'})
                    # Last-resort removal is scoped to this native container.
                    code, _, _ = await _bounded(_docker(['rm', '-f', cid]), 15)
                    if code:
                        raise RuntimeError('owned container removal failed')
            except (Exception, asyncio.CancelledError) as exc:
                cleanup = 'failed'
                diagnostic('cleanup_verification', exc)
        if not settled:
            cleanup = 'unconfirmed'
        if native is not None:
            if native.error and attempt == 'completed':
                attempt = 'execution_error'
            if native.export_error:
                diagnostic('native_export', RuntimeError())
            if native.verifier_error is not None or native.scoring is not None:
                diagnostic('unexpected_verifier', RuntimeError())
    except (Exception, asyncio.CancelledError) as exc:
        cleanup = 'failed'
        diagnostic('finish', exc)
    return _FinishResult(native, cleanup, collection, collection_status, attempt)


async def solve_task(config: SolveConfig) -> dict[str, Any]:
    """Execute and collect one unscored attempt through native Rollout phases.

    This writer shares ASI's directory and persisted_outputs field names, NOT
    its full EvalResult/provenance schema. Cancellation is re-raised after
    bounded collection/cleanup and result persistence. No verifier is invoked.
    """
    import asyncio
    import math
    import time
    from datetime import datetime

    import benchflow as bf

    for budget in (config.lifecycle_timeout, config.export_timeout, config.cleanup_timeout):
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError('phase budgets must be finite and positive')
    if type(config.max_artifact_bytes) is not int or config.max_artifact_bytes <= 0:
        raise ValueError('artifact limit must be a positive integer')
    if not config.agent.strip() or not config.model.strip():
        raise ValueError('explicit agent and model are required')
    manifest_path = _regular_file(Path(config.prepared_dir), 'task_manifest.json')
    prepared_bytes = manifest_path.read_bytes()
    contract = inspect_prepared_task(config.prepared_dir)
    if _regular_file(contract.task_path, 'task_manifest.json').read_bytes() != prepared_bytes:
        raise ValueError('prepared manifest changed during inspection')
    prepared_identity = {'converter_version': 'prepare-v5',
                         'manifest_sha256': hashlib.sha256(prepared_bytes).hexdigest()}
    agent_env, route = _model_route(config)
    root = reserve_run(config.run_dir, contract.task_path)
    prepared_manifest = json.loads(prepared_bytes)
    task_id = prepared_manifest['task']['id']
    level = prepared_manifest['task']['level']
    layout = result_layout(root, task_id, prepared_manifest['instance']['id'], level)
    layout.task_dir.mkdir()
    rollout, cid = None, None
    phase, attempt = 'create', 'execution_error'
    network = None
    diagnostics: list[dict[str, str]] = []
    cancelled, settled = False, True
    started = time.monotonic()
    metadata = {'adapter_schema_version': 1, 'mode': 'produce-only', 'official': False,
                'task_id': task_id, 'instance_id': task_id + '__seed31415',
                'prompt_level': level, 'attempt': 1, 'agent': config.agent,
                'model': config.model, 'requested_effort': config.effort,
                'model_route': route, 'sources': contract.sources, 'prepared': prepared_identity,
                'started_at': datetime.now(UTC).isoformat(),
                'evaluation_status': 'pending', 'status': 'running',
                'budgets': {'lifecycle_seconds': config.lifecycle_timeout,
                            'export_seconds': config.export_timeout,
                            'cleanup_seconds': config.cleanup_timeout,
                            'max_artifact_bytes': config.max_artifact_bytes}}
    _atomic_json(layout.run_metadata_file, metadata)

    def diagnostic(stage: str, exc: BaseException) -> None:
        # Print the original exception to stderr; persisted metadata keeps categories only.
        traceback.print_exception(exc)
        diagnostics.append({'phase': stage, 'error_type': type(exc).__name__})

    try:
        rollout = await _bounded(bf.Rollout.create(bf.RolloutConfig(
            **rollout_contract_options(contract), agent=config.agent, model=config.model,
            reasoning_effort=config.effort, agent_env=agent_env or None,
            jobs_dir=layout.benchflow_dir, job_name='asi-solve',
        )), config.lifecycle_timeout)
        for phase in ('setup', 'start', 'install_agent', 'connect', 'execute'):
            remaining = config.lifecycle_timeout - (time.monotonic() - started)
            await _bounded(getattr(rollout, phase)(), remaining)
            if phase == 'start':
                cid = await _bounded(_container_id(rollout), min(30, remaining))
                if cid is None:
                    raise RuntimeError('native Rollout did not expose its Docker container')
            if phase == 'connect':
                network = await _bounded(_runtime_evidence(rollout, cid), min(30, remaining))
        attempt = 'completed'
    except asyncio.CancelledError as exc:
        cancelled, attempt = True, 'cancelled'
        diagnostic(phase, exc)
    except Exception as exc:
        settled = not isinstance(exc, UnsettledPhase)
        attempt = 'timeout' if isinstance(exc, TimeoutError) or 'Timeout' in type(exc).__name__ else 'execution_error'
        diagnostic(phase, exc)
    finally:
        # Shield the bounded finishing job from repeated caller cancellation.
        # It cannot extend its own collection/cleanup budgets indefinitely.
        finishing = asyncio.create_task(_finish_attempt(
            rollout, cid, settled, attempt, layout, contract.outputs, config, diagnostics,
        ))
        while not finishing.done():
            try:
                await asyncio.shield(finishing)
            except asyncio.CancelledError:
                cancelled, attempt = True, 'cancelled'
        finished = finishing.result()
        native, cleanup = finished.native, finished.cleanup
        collection, collection_status = finished.collection, finished.collection_status
        attempt = 'cancelled' if cancelled else finished.outcome

    native_dir = getattr(rollout, '_rollout_dir', None) if rollout else None
    native_result = Path(native_dir) / 'result.json' if native_dir else None
    evidence = (str(native_result.relative_to(root))
                if native_result and native_result.is_file() else None)
    cost_fields = ('n_input_tokens', 'n_output_tokens', 'n_cache_read_tokens',
                   'n_cache_creation_tokens', 'total_tokens', 'cost_usd', 'usage_source')
    success = (attempt == 'completed' and collection_status == 'collected'
               and collection['prediction_status'] == 'present' and cleanup == 'completed'
               and not diagnostics and native is not None and evidence is not None)
    produced = {item['name'] for item in collection['artifacts']}
    result = {**metadata, 'status': 'completed' if success else 'failed',
              **attempt_result(attempt),
              'collection_status': collection_status,
              'prediction_status': collection['prediction_status'], 'cleanup_status': cleanup,
              'execution_time_seconds': time.monotonic() - started,
              'diagnostics': diagnostics, 'collection': collection,
              'native_result': evidence,
              'native_error_category': getattr(native, 'error_category', None),
              'native_error_present': bool(getattr(native, 'error', None)),
              'network': network,
              'agent_output': {
                  'code_files': [s.name for s in contract.outputs if s.type == 'code' and s.name in produced],
                  'data_files': [s.name for s in contract.outputs if s.type == 'data' and s.name in produced],
                  'persisted_outputs': persisted_outputs(layout, contract.outputs, collection)},
              'cost': {key: getattr(native, key, None) for key in cost_fields}}
    _atomic_json(layout.result_file, result)
    _atomic_json(layout.run_metadata_file, {
        **metadata, 'status': result['status'], 'attempt_status': result['attempt_status'],
        'failure_reason': result['failure_reason'],
        'evaluation_status': 'pending', 'result_file': str(layout.result_file.relative_to(root)),
        'finished_at': datetime.now(UTC).isoformat(),
    })
    if cancelled:
        raise asyncio.CancelledError
    return result


def main() -> int:
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description='Run one unscored ASI seed31415/b1 Docker attempt')
    parser.add_argument('--prepared-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True, help='Must not already exist')
    parser.add_argument('--agent', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--effort')
    parser.add_argument('--model-env-file', type=Path)
    parser.add_argument('--lifecycle-timeout', type=float, default=1800)
    args = parser.parse_args()
    try:
        result = asyncio.run(solve_task(SolveConfig(**vars(args))))
    except (KeyboardInterrupt, asyncio.CancelledError):
        return 130
    except Exception as exc:
        traceback.print_exception(exc)
        error = {'status': 'failed', 'error_type': 'ValueError' if isinstance(exc, InventoryError) else type(exc).__name__}
        if isinstance(exc, InventoryError):
            error['error'] = str(exc)
        print(json.dumps(error))
        return 1
    print(json.dumps({key: result[key] for key in (
        'status', 'attempt_status', 'collection_status', 'prediction_status',
        'cleanup_status', 'evaluation_status',
    )}, indent=2))
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
