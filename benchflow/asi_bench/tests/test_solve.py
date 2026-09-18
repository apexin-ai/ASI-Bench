"""S5 contracts against executable source excerpts pinned to ASI a797bf69.

Regression cases guard BenchFlow 9b90d16c's implicit stripped prompt/public
network defaults; S5 does not launch a model, import generators, or score.
"""
import hashlib
import json
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow._utils.config_override import apply_config_override
from benchflow.rollout._setup import _resolve_prompts
from benchflow.task.config import TaskConfig
from benchmarks.asi_bench.prepare import build_task_info, build_task_md
from benchmarks.asi_bench.solve import (
    OutputSpec,
    collect_outputs,
    load_contract,
    output_specs,
    runtime_override,
    validate_runtime_launch,
)

FIXTURES = Path(__file__).parent / "fixtures/asi_contract"
META = {"id": "math.mpsc_safety_filter", "output": {"files": [
    {"name": "analysis.py", "type": "code"},
    {"name": "certificate.json", "type": "data"},
]}}
PROFILE = {"profile_id": "cpu-v1", "variant": "os", "workdir": "/workspace",
           "base_image": "python:3.11-slim", "network": False, "cpus": 2, "memory_mb": 4096}


def upstream(fixture, *, as_class=False):
    """Execute only reviewed exact excerpts, without ASI imports/dependencies."""
    entry = next(x for x in json.loads((FIXTURES / "provenance.json").read_text())
                 if x["fixture"] == fixture + ".txt")
    raw = (FIXTURES / entry["fixture"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == entry["fixture_sha256"]
    text = textwrap.dedent(raw.decode())
    if as_class:
        text = "class Upstream:\n" + textwrap.indent(text, "    ")
    namespace = {}
    exec("from __future__ import annotations\n" + text, namespace)
    return namespace["Upstream"]() if as_class else namespace["collect_output_files"]


def prepared(tmp_path):
    (tmp_path / "task.md").write_text(
        build_task_md("  hello\n\n", META["id"], "b1", 3600, 600)
    )
    inputs = tmp_path / "environment/inputs"
    inputs.mkdir(parents=True)
    (inputs / "prompt.md").write_bytes(b"  hello\n\n")
    (inputs / "task_info.json").write_text(json.dumps(build_task_info(META["id"], "b1", META, 3600)))
    return tmp_path


def test_prompt_and_task_info_against_pinned_upstream(tmp_path):
    root = prepared(tmp_path)
    prompt, specs, timeout = load_contract(root, META)
    assert timeout == 3600
    inputs = root / "environment/inputs"
    pi = upstream("pi_cli", as_class=True)
    assert prompt == pi._get_stdin_input(SimpleNamespace(workspace_dir=inputs))
    assert prompt == "  hello\n\n"
    # Explicit native prompts avoid default TaskDocument.strip().
    assert _resolve_prompts(root, [prompt]) == [prompt]
    generator = upstream("instance_generator", as_class=True)
    actual = json.loads((inputs / "task_info.json").read_text())
    expected = generator._build_agent_task_info(META, SimpleNamespace(value="b1"), effective_timeout_seconds=3600)
    assert actual == expected
    assert [vars(s) for s in specs] == expected["expected_outputs"]


@pytest.mark.parametrize("names", [[], ["analysis.py"], ["analysis.py", "certificate.json"]])
def test_collection_parity_with_missing_and_extra_files(tmp_path, names):
    for name in [*names, "extra.txt"]:
        (tmp_path / name).write_bytes((name + "\n").encode())
    specs = output_specs(META)
    actual = collect_outputs(tmp_path, specs)
    native_names = upstream("subprocess_base")(tmp_path, SimpleNamespace(metadata=META))
    assert [a["name"] for a in actual["artifacts"]] == native_names
    for artifact in actual["artifacts"]:
        raw = (tmp_path / artifact["name"]).read_bytes()
        assert artifact["sha256"] == hashlib.sha256(raw).hexdigest()
        assert artifact["size"] == len(raw)
    assert actual["missing"] == [s.name for s in specs if s.name not in names]
    assert actual["prediction_status"] == ("present" if len(names) == 2 else "missing")


def test_nested_and_empty_regular_files_match_upstream(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub/file").touch()
    meta = {"output": {"files": [{"name": "sub/file", "type": "data"}]}}
    actual = collect_outputs(tmp_path, output_specs(meta))
    assert [a["name"] for a in actual["artifacts"]] == upstream("subprocess_base")(tmp_path, SimpleNamespace(metadata=meta))
    assert actual["artifacts"][0]["size"] == 0


@pytest.mark.parametrize("kind", ["directory", "symlink", "broken_symlink", "parent_symlink", "fifo"])
def test_security_differences_are_explicit(tmp_path, kind):
    import os
    name = "analysis.py"
    if kind == "directory":
        (tmp_path / name).mkdir()
    elif kind == "fifo":
        os.mkfifo(tmp_path / name)
    elif kind == "parent_symlink":
        (tmp_path / "real").mkdir()
        (tmp_path / "real/file").write_text("x")
        (tmp_path / "linked").symlink_to(tmp_path / "real", target_is_directory=True)
        name = "linked/file"
    else:
        (tmp_path / "real").write_text("x")
        (tmp_path / name).symlink_to(tmp_path / ("real" if kind == "symlink" else "absent"))
    meta = {"output": {"files": [{"name": name, "type": "code"}]}}
    result = collect_outputs(tmp_path, output_specs(meta))
    assert result["artifacts"] == []
    assert result["prediction_status"] == "invalid"
    assert upstream("subprocess_base")(tmp_path, SimpleNamespace(metadata=meta)) == ([] if kind == "broken_symlink" else [name])


@pytest.mark.parametrize("name", ["/x", "../x", "a/../x", "a\\x", "./x", "a//x", "a/", "", "C:x", "x\n", "prompt.md", "data/x"])
def test_rejects_unsafe_output_names(name):
    with pytest.raises(ValueError):
        output_specs({"output": {"files": [{"name": name, "type": "data"}]}})


@pytest.mark.parametrize("files", [
    [{"name": "a", "type": "directory"}],
    [{"name": "a"}, {"name": "A"}],
    [{"name": "a"}, {"name": "a/b"}],
])
def test_rejects_unsupported_and_overlapping_declarations(files):
    with pytest.raises(ValueError):
        output_specs({"output": {"files": files}})


def test_task_info_cannot_override_collector_authority(tmp_path):
    root = prepared(tmp_path)
    path = root / "environment/inputs/task_info.json"
    info = json.loads(path.read_text())
    info["expected_outputs"][0]["name"] = "extra.txt"
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="contract"):
        load_contract(root, META)


def test_direct_specs_cannot_bypass_path_checks(tmp_path):
    with pytest.raises(ValueError):
        collect_outputs(tmp_path, (OutputSpec("../secret", "data"),))


def test_io_error_is_collection_failure_not_missing(tmp_path, monkeypatch):
    (tmp_path / "analysis.py").touch()
    def denied(*args, **kwargs):
        raise PermissionError("cannot read artifact")
    monkeypatch.setattr(Path, "open", denied)
    with pytest.raises(PermissionError):
        collect_outputs(tmp_path, output_specs(META))


def test_runtime_maps_native_config_without_mutating_defaults():
    original = TaskConfig()
    effective = apply_config_override(original, runtime_override(PROFILE, 3600))
    assert original.sandbox.allow_internet is True
    assert effective.sandbox.allow_internet is False
    assert effective.sandbox.network_mode == "no-network"
    assert effective.sandbox.cpus == 2
    assert effective.sandbox.memory_mb == 4096
    assert effective.agent.timeout_sec == 3600
    assert effective.agent.prompt_prefix is None
    validate_runtime_launch(effective.sandbox, backend="docker", preserve_agent_network=False)
    # ASI CLI agents retain model connectivity; task no-web remains requested.
    validate_runtime_launch(effective.sandbox, backend="docker", preserve_agent_network=True)
    with pytest.raises(ValueError):
        validate_runtime_launch(original.sandbox, backend="docker", preserve_agent_network=False)


@pytest.mark.parametrize("patch", [{"network": True}, {"network": 0}, {"workdir": "/tmp"}, {"cpus": 0}, {"memory_mb": True}])
def test_invalid_runtime_fails_closed(patch):
    with pytest.raises(ValueError):
        runtime_override(PROFILE | patch, 3600)


@pytest.mark.parametrize("outcome", [
    "completed", "agent_error", "timeout", "setup_error", "cancelled", "execution_error",
])
def test_attempt_result_only_maps_execution_status(outcome):
    """Guards narrow outcome mapping from solve review after 9b90d16c."""
    from benchmarks.asi_bench.solve import attempt_result
    assert attempt_result(outcome) == {
        "attempt_status": "completed" if outcome == "completed" else "execution_failed",
        "failure_reason": None if outcome == "completed" else outcome,
    }


def test_attempt_result_rejects_unknown_outcome():
    """Guards outcome validation retained from the contract after 9b90d16c."""
    from benchmarks.asi_bench.solve import attempt_result
    with pytest.raises(ValueError, match="unknown attempt outcome"):
        attempt_result("unknown")


@pytest.mark.parametrize("preserve", [True, False])
def test_asi_plane_preserves_native_network_decision(tmp_path, monkeypatch, preserve):
    """Guards the ASI network parity correction after commit 9b90d16c."""
    from benchflow.rollout_planes import DefaultRolloutPlanes
    from benchmarks.asi_bench.solve import ASIDockerPlanes
    task = SimpleNamespace(config=apply_config_override(TaskConfig(), runtime_override(PROFILE, 3600)))
    received = {}
    def create(self, environment, task, task_path, rollout_name, rollout_paths, **kwargs):
        received.update(kwargs)
        return SimpleNamespace(task_env_config=task.config.sandbox)
    monkeypatch.setattr(DefaultRolloutPlanes, "create_environment", create)
    from benchflow.task import RolloutPaths
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    (tmp_path / "environment").mkdir()
    plane = ASIDockerPlanes()
    env = plane.create_environment("docker", task, tmp_path, "test", paths,
                                   preserve_agent_network=preserve, environment_manifest=None)
    assert received["preserve_agent_network"] is preserve
    assert env.task_env_config.allow_internet is False
    with pytest.raises(ValueError):
        plane.create_environment("daytona", task, tmp_path, "test", None,
                                 preserve_agent_network=True, environment_manifest=None)
    with pytest.raises(ValueError):
        plane.create_environment("docker", task, tmp_path, "test", None,
                                 preserve_agent_network=True, environment_manifest=object())


def pi_os_output(workspace, metadata, *, success=True, log="done", error=None):
    """Actual pinned pi.solve; only external run_agent is stubbed (no model)."""
    import time
    pi = upstream("pi_os_solve", as_class=True)
    namespace = pi.solve.__func__.__globals__
    namespace.update(time=time, AgentOutput=SimpleNamespace,
                     RunStatus=SimpleNamespace(COMPLETED="completed", FAILED="failed", TIMEOUT="timeout"),
                     collect_output_files=upstream("subprocess_base"))
    calls = []
    def run_agent(**kwargs):
        calls.append(kwargs)
        if error:
            raise error
        return success, log, None, None, "fixture-image"
    pi.sandbox = "os"
    pi._os_sandbox = SimpleNamespace(run_agent=run_agent)
    pi.allow_external_tools = False
    pi._get_effective_timeout = lambda instance: 3600
    pi._build_os_agent_cmd = lambda workspace: ["not-executed"]
    pi._build_os_api_env = lambda: {}
    pi._build_os_extra_mounts = lambda: []
    pi._extract_terminal_error_from_jsonl = lambda raw: None
    instance = SimpleNamespace(metadata=metadata, workspace_dir=workspace, instance_id="fixture__seed31415")
    return pi.solve(instance), calls


@pytest.mark.parametrize("success,log,status", [
    (True, "done", "completed"),
    (False, "failed", "failed"),
    (False, "agent timed out", "timeout"),
])
def test_pi_os_failure_collection_and_final_stdin_parity(tmp_path, success, log, status):
    (tmp_path / "prompt.md").write_text("\n  exact prompt\n")
    (tmp_path / "analysis.py").write_bytes(b"partial output\n")
    output, calls = pi_os_output(tmp_path, META, success=success, log=log)
    assert output.status == status
    assert calls[0]["stdin_input"] == "\n  exact prompt\n"
    actual = collect_outputs(tmp_path, output_specs(META))
    assert [a["name"] for a in actual["artifacts"]] == output.code_files + output.data_files
    assert actual["missing"] == ["certificate.json"]


def test_pi_os_unhandled_exception_is_not_claimed_as_parity(tmp_path):
    # Native pi propagates transport errors; S6 partial collection on these
    # errors is an adapter extension, not native pi behavior.
    (tmp_path / "prompt.md").write_text("prompt")
    with pytest.raises(RuntimeError, match="transport"):
        pi_os_output(tmp_path, META, error=RuntimeError("transport"))


def upstream_workspace(instance, destination, metadata, framework_dir):
    generator = upstream("instance_generator", as_class=True)
    workspace = upstream("workspace", as_class=True)
    namespace = workspace._prepare_workspace.__func__.__globals__
    namespace.update(json=json, AGENT_TASK_INFO_FILENAME="task_info.json",
                     FRAMEWORK_TASK_INFO_FILENAME="framework_task_info.json")
    workspace._build_agent_task_info = generator._build_agent_task_info
    workspace._AGENT_MEMORY_FILENAMES = ("CLAUDE.md", "AGENTS.md", ".claude")
    destination.mkdir()
    workspace._prepare_workspace(instance, destination, META["id"] + "__seed31415",
                                 SimpleNamespace(value="b1"), metadata, {},
                                 effective_timeout_seconds=3600, framework_task_info_dir=framework_dir)


@pytest.fixture
def full_prepared(tmp_path):
    """Build synthetic complete package using the production converter."""
    import yaml

    from benchmarks.asi_bench.prepare import PrepareConfig, prepare_task
    raw = tmp_path / "raw"
    instance = raw / "instance" / (META["id"] + "__seed31415")
    bundle = raw / "task_bundle/tasks/math/mpsc_safety_filter"
    (instance / "data").mkdir(parents=True)
    (instance / "reference").mkdir()
    bundle.mkdir(parents=True)
    for name in ("system.json", "public_cases.json"):
        (instance / "data" / name).write_text('{"fixture": true}\n')
    (instance / "reference/synthetic.json").write_text('{}')
    (instance / "instance_meta.json").write_text('{"params_used": {}}')
    (instance / "prompt_b1.md").write_text('  fixture prompt\n')
    meta = META | {"input": {"files": [{"name": "system.json"}, {"name": "public_cases.json"}]}}
    (bundle / "task_meta.yaml").write_text(yaml.safe_dump(meta))
    (bundle / "task_eval.yaml").write_text('task_id: math.mpsc_safety_filter\n')
    sources = {"seed": 31415, "task_id": META["id"], "instance_id": META["id"] + "__seed31415",
               "hf": {"repo": "Apexintelligence-AI/ASI-Bench-seed31415", "path": "tasks/" + META["id"] + "__seed31415", "resolved_revision": "a" * 40},
               "github": {"repo": "apexin-ai/ASI-Bench", "path": "tasks/math/mpsc_safety_filter", "resolved_revision": "a797bf69683400bed0f38c4add507bb3d404b833"},
               "files": [{"path": p.relative_to(raw).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in raw.rglob('*') if p.is_file()]}
    (raw / "sources.json").write_text(json.dumps(sources))
    config = PrepareConfig(raw, META["id"], META["id"] + "__seed31415", "b1", tmp_path / "prepared")
    return prepare_task(config).output_dir, instance, meta


def test_prepared_tree_parity_against_asi_workspace(full_prepared, tmp_path):
    from benchflow.rollout import RolloutConfig
    from benchmarks.asi_bench.solve import (
        ASIDockerPlanes,
        inspect_prepared_task,
        rollout_contract_options,
    )
    root, instance, meta = full_prepared
    before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    contract = inspect_prepared_task(root)
    workspace = tmp_path / "asi-workspace"
    upstream_workspace(instance, workspace, meta, tmp_path / "framework")
    actual = root / "environment/inputs"
    assert {p.relative_to(workspace).as_posix() for p in workspace.rglob('*') if p.is_file()} == set(contract.inputs)
    for name in contract.inputs:
        if name == "task_info.json":  # JSON semantics; converter appends a newline.
            assert json.loads((actual / name).read_text()) == json.loads((workspace / name).read_text())
        else:
            assert (actual / name).read_bytes() == (workspace / name).read_bytes()
    cfg = RolloutConfig(**rollout_contract_options(contract), agent="dummy", model="dummy")
    assert isinstance(cfg.planes, ASIDockerPlanes)
    assert cfg.skip_verify and cfg.prompts == [contract.prompt]
    assert before == {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize("mutation", ["seed42", "data", "extra_input", "extra_context", "symlink", "revision"])
def test_prepared_contract_rejects_drift(full_prepared, mutation):
    from benchmarks.asi_bench.solve import inspect_prepared_task
    root, _, _ = full_prepared
    manifest = root / "task_manifest.json"
    if mutation in {"seed42", "revision"}:
        data = json.loads(manifest.read_text())
        if mutation == "seed42":
            data["seed"] = 42
        else:
            data["sources"]["github"]["resolved_revision"] = "0" * 40
        manifest.write_text(json.dumps(data))
    elif mutation == "data":
        (root / "environment/inputs/data/system.json").write_text("modified")
    elif mutation == "extra_input":
        (root / "environment/inputs/secret.txt").write_text("extra")
    elif mutation == "extra_context":
        (root / "environment/docker-compose.yaml").write_text("services: {}")
    else:
        (root / "environment/inputs/link").symlink_to(root / "verifier")
    with pytest.raises(ValueError):
        inspect_prepared_task(root)


@pytest.mark.parametrize("preserve", [True, False])
def test_native_docker_network_matches_agent_policy(full_prepared, tmp_path, preserve):
    """Guards ASI CLI network parity after 9b90d16c without a daemon."""
    from benchflow.task import RolloutPaths, Task
    from benchmarks.asi_bench.solve import (
        inspect_prepared_task,
        rollout_contract_options,
    )
    root, _, _ = full_prepared
    options = rollout_contract_options(inspect_prepared_task(root))
    task = Task(root)
    task.config = apply_config_override(task.config, options["config_override"])
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    env = options["planes"].create_environment(
        "docker", task, root, "s5-no-daemon", paths,
        preserve_agent_network=preserve, environment_manifest=None,
    )
    assert env.task_env_config.allow_internet is preserve
    # Retain the task-level no-web request; native setup copies its config.
    assert task.config.sandbox.allow_internet is False
    from benchflow.rollout._setup import _task_disallows_internet
    assert _task_disallows_internet(task) is True
    assert env.task_env_config.network_mode == "no-network"
    assert env.task_env_config.cpus == 2
    assert env.task_env_config.memory_mb == 4096
    import yaml
    overlays = [p for p in env._docker_compose_paths if p.name == "docker-compose-no-network.yaml"]
    assert len(overlays) == (0 if preserve else 1)
    # Guards the no-web Docker NET_ADMIN failure found after 9b90d16c.
    compose = [p for p in env._docker_compose_paths if p.name == "docker-compose.yaml"]
    assert len(compose) == (1 if preserve else 0)
    assert not (root / "environment/docker-compose.yaml").exists()
    if preserve:
        assert yaml.safe_load(compose[0].read_text()) == {
            "services": {"main": {"cap_add": ["NET_ADMIN"]}}}
        assert env.environment_dir.is_relative_to(paths.rollout_dir)
        for source in (root / "environment/inputs").rglob('*'):
            if source.is_file():
                assert (env.environment_dir / "inputs" / source.relative_to(root / "environment/inputs")).read_bytes() == source.read_bytes()
        assert (env.environment_dir / "Dockerfile").read_bytes() == (root / "environment/Dockerfile").read_bytes()
        assert sorted(p.relative_to(env.environment_dir / "inputs") for p in (env.environment_dir / "inputs").rglob('*')) == sorted(p.relative_to(root / "environment/inputs") for p in (root / "environment/inputs").rglob('*'))
    if not preserve:
        assert yaml.safe_load(overlays[0].read_text())["services"]["main"]["network_mode"] == "none"


def test_real_prepared_parity_opt_in(tmp_path):
    """Read-only local acceptance; set ASI_S5_PREPARED and ASI_S5_RAW."""
    import os

    import yaml

    from benchmarks.asi_bench.solve import inspect_prepared_task
    if not os.environ.get("ASI_S5_PREPARED") or not os.environ.get("ASI_S5_RAW"):
        pytest.skip("set ASI_S5_PREPARED and ASI_S5_RAW for real-asset acceptance")
    root = Path(os.environ["ASI_S5_PREPARED"])
    raw = Path(os.environ["ASI_S5_RAW"])
    def hashes(directory):
        return {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in directory.rglob('*') if p.is_file()}
    before, raw_before = hashes(root), hashes(raw)
    contract = inspect_prepared_task(root)
    source_manifest = json.loads((raw / "sources.json").read_text())
    assert source_manifest["hf"] == contract.sources["hf"]
    assert source_manifest["github"] == contract.sources["github"]
    for record in source_manifest["files"]:
        assert raw_before[record["path"]] == record["sha256"]
    metadata = yaml.safe_load((raw / "task_bundle/tasks/math/mpsc_safety_filter/task_meta.yaml").read_text())
    workspace = tmp_path / "asi-workspace"
    upstream_workspace(raw / "instance/math.mpsc_safety_filter__seed31415", workspace,
                       metadata, tmp_path / "framework")
    assert set(hashes(workspace)) == set(contract.inputs)
    for name in contract.inputs:
        left, right = root / "environment/inputs" / name, workspace / name
        if name == "task_info.json":
            assert json.loads(left.read_text()) == json.loads(right.read_text())
        else:
            assert left.read_bytes() == right.read_bytes()
    assert before == hashes(root) and raw_before == hashes(raw)


@pytest.mark.parametrize('attempt,suffix', [(1, ''), (2, '__attempt2')])
def test_native_result_layout(tmp_path, attempt, suffix):
    """Guards native ASI result naming added after BenchFlow 9b90d16c."""
    from benchmarks.asi_bench.solve import result_layout
    layout = result_layout(tmp_path, 'math.mpsc_safety_filter',
                           'math.mpsc_safety_filter__seed31415', 'b1', attempt)
    base = 'math.mpsc_safety_filter__seed31415__b1' + suffix
    assert layout.result_file == tmp_path / 'math.mpsc_safety_filter' / (base + '.json')
    assert layout.outputs_dir == layout.result_file.with_suffix('.outputs')
    assert layout.run_metadata_file == tmp_path / 'run_metadata.json'
    assert layout.benchflow_dir == tmp_path / 'benchflow'
    assert not layout.task_dir.exists()  # Calculation has no publishing side effects.


@pytest.mark.parametrize('task,instance,level,attempt', [
    ('../escape', '../escape__seed31415', 'b1', 1),
    ('math.mpsc_safety_filter', 'other__seed31415', 'b1', 1),
    ('math.mpsc_safety_filter', 'math.mpsc_safety_filter__seed42', 'b1', 1),
    ('math.mpsc_safety_filter', 'math.mpsc_safety_filter__seed31415', '../b1', 1),
    ('math.mpsc_safety_filter', 'math.mpsc_safety_filter__seed31415', 'b1', 0),
    ('math.mpsc_safety_filter', 'math.mpsc_safety_filter__seed31415', 'b1', True),
])
def test_native_result_layout_rejects_invalid_identity(tmp_path, task, instance, level, attempt):
    """Guards native layout path/identity safety after 9b90d16c."""
    from benchmarks.asi_bench.solve import result_layout
    with pytest.raises(ValueError):
        result_layout(tmp_path, task, instance, level, attempt)


def test_native_persisted_outputs(tmp_path):
    """Guards ASI persisted_outputs field/byte/hash parity after 9b90d16c."""
    from benchmarks.asi_bench.solve import persisted_outputs, result_layout
    layout = result_layout(tmp_path, 'math.mpsc_safety_filter',
                           'math.mpsc_safety_filter__seed31415', 'b1')
    specs = output_specs(META)
    with pytest.raises(FileNotFoundError):
        persisted_outputs(layout, specs, collect_outputs(layout.outputs_dir, specs))
    layout.outputs_dir.mkdir(parents=True)
    (layout.outputs_dir / 'analysis.py').write_bytes(b'# dummy\n')
    (layout.outputs_dir / 'extra.txt').write_text('not collected')
    expected = {'dir': layout.outputs_dir.name, 'files': [
        {'path': 'analysis.py', 'bytes': 8,
         'sha256': hashlib.sha256(b'# dummy\n').hexdigest()},
        {'path': 'certificate.json', 'missing': True}]}
    assert persisted_outputs(layout, specs, collect_outputs(layout.outputs_dir, specs)) == expected
    assert collect_outputs(layout.outputs_dir, specs)['missing'] == ['certificate.json']
    (layout.outputs_dir / 'certificate.json').symlink_to(layout.outputs_dir / 'extra.txt')
    expected['files'][1]['reason'] = 'symlink is not an artifact/input: certificate.json'
    assert persisted_outputs(layout, specs, collect_outputs(layout.outputs_dir, specs)) == expected
    assert collect_outputs(layout.outputs_dir, specs)['prediction_status'] == 'invalid'


def test_native_layout_rejects_symlink(tmp_path):
    """Guards native result destination containment after 9b90d16c."""
    from benchmarks.asi_bench.solve import result_layout
    outside = tmp_path / 'outside'
    outside.mkdir()
    (tmp_path / 'math.mpsc_safety_filter').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        result_layout(tmp_path, 'math.mpsc_safety_filter',
                      'math.mpsc_safety_filter__seed31415', 'b1')


def test_prepared_rejects_changed_external_bundle(full_prepared):
    """Guards prepare-v3: solve must revalidate referenced task metadata."""
    from benchmarks.asi_bench.solve import inspect_prepared_task
    root, instance, _ = full_prepared
    inspect_prepared_task(root)
    raw = instance.parent.parent
    metadata = raw / 'task_bundle/tasks/math/mpsc_safety_filter/task_meta.yaml'
    metadata.write_text(metadata.read_text() + '\n# changed after prepare\n')
    with pytest.raises(ValueError, match='integrity'):
        inspect_prepared_task(root)


@pytest.mark.parametrize('mutation', ['old_version', 'missing_entry', 'tamper', 'escape', 'no_scoring', 'visibility', 'extra_verifier'])
def test_p21_unique_prepared_contract(full_prepared, mutation):
    """Guards P2.1: only v4 is accepted; verifier stays outside Agent context."""
    from benchmarks.asi_bench.solve import inspect_prepared_task
    root, _, _ = full_prepared
    path = root / 'task_manifest.json'
    manifest = json.loads(path.read_text())
    assert manifest['scoring']['profile_id'] == 'asi-seed31415-v1'
    if mutation == 'old_version':
        manifest['converter_version'] = 'prepare-v3'
    elif mutation == 'missing_entry':
        (root / 'verifier/test.sh').unlink()
    elif mutation == 'tamper':
        (root / 'verifier/score_entry.py').write_text('tampered')
    elif mutation == 'escape':
        manifest['scoring']['launcher'] = '../test.sh'
    elif mutation == 'no_scoring':
        manifest['scoring'] = None
    elif mutation == 'visibility':
        next(e for e in manifest['files'] if e['target'] == 'verifier/test.sh')['visibility'] = ['agent']
    else:
        (root / 'verifier/extra').write_text('extra')
    path.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, FileNotFoundError)):
        inspect_prepared_task(root)


def test_prepared_ignores_only_finder_metadata(full_prepared):
    """Guards Finder inventory handling following BenchFlow 9b90d16c."""
    from benchmarks.asi_bench.solve import inspect_prepared_task
    root, _, _ = full_prepared
    for relative in ('.DS_Store', 'environment/inputs/.DS_Store', 'verifier/.DS_Store'):
        (root / relative).write_bytes(b'Finder')
    inspect_prepared_task(root)
    (root / 'debug.py').write_text('extra')
    with pytest.raises(ValueError, match=r'debug\.py'):
        inspect_prepared_task(root)


def test_prepared_finder_symlink_rejected(full_prepared):
    """Guards symlink rejection following BenchFlow 9b90d16c."""
    from benchmarks.asi_bench.solve import inspect_prepared_task
    root, _, _ = full_prepared
    (root / '.DS_Store').symlink_to('task.md')
    with pytest.raises(ValueError, match='symlink'):
        inspect_prepared_task(root)
