"""S6 regression coverage against the adapter on BenchFlow 9b90d16c.

Guard manual Rollout lifecycle, fail-closed export and native ASI path binding;
never modify or emulate scientific scoring.
"""
import asyncio
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import benchmarks.asi_bench.solve as solve
from benchmarks.asi_bench.solve import decode_artifact_tar, reserve_run


def archive(name='analysis.py', body=b'print(1)\n', kind=tarfile.REGTYPE, extra=False):
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode='w') as tar:
        member = tarfile.TarInfo(name)
        member.type = kind
        member.linkname = '/etc/passwd' if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE) else ''
        member.size = len(body) if kind == tarfile.REGTYPE else 0
        tar.addfile(member, io.BytesIO(body) if member.size else None)
        if extra:
            tar.addfile(tarfile.TarInfo('extra'))
    return out.getvalue()


def test_archive_exact_regular_bytes():
    assert decode_artifact_tar(archive(), 'analysis.py', 100) == b'print(1)\n'


@pytest.mark.parametrize('name,kind,extra', [
    ('../analysis.py', tarfile.REGTYPE, False),
    ('/analysis.py', tarfile.REGTYPE, False),
    ('other', tarfile.REGTYPE, False),
    ('analysis.py', tarfile.SYMTYPE, False),
    ('analysis.py', tarfile.LNKTYPE, False),
    ('analysis.py', tarfile.DIRTYPE, False),
    ('analysis.py', tarfile.FIFOTYPE, False),
    ('analysis.py', tarfile.REGTYPE, True),
])
def test_archive_rejects_unsafe_members(name, kind, extra):
    with pytest.raises(ValueError):
        decode_artifact_tar(archive(name, kind=kind, extra=extra), 'analysis.py', 100)


def test_archive_size_limit():
    with pytest.raises(ValueError):
        decode_artifact_tar(archive(body=b'x' * 101), 'analysis.py', 100)


def test_run_is_fresh_and_not_inside_inputs(tmp_path):
    prepared = tmp_path / 'prepared'
    prepared.mkdir()
    with pytest.raises(ValueError):
        reserve_run(prepared / 'output', prepared)
    with pytest.raises(ValueError):
        reserve_run(tmp_path, prepared)
    run = tmp_path / 'run'
    reserve_run(run, prepared)
    with pytest.raises(FileExistsError):
        reserve_run(run, prepared)
    assert run.is_dir()


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    prepared = tmp_path / 'prepared'
    prepared.mkdir()
    (prepared / 'task_manifest.json').write_text(json.dumps({'converter_version': 'prepare-v5', 'task': {'id': 'math.mpsc_safety_filter', 'level': 'b1'}, 'instance': {'id': 'math.mpsc_safety_filter__seed31415'}}))
    outputs = (solve.OutputSpec('analysis.py', 'code'), solve.OutputSpec('certificate.json', 'data'))
    contract = SimpleNamespace(task_path=prepared, outputs=outputs, sources={})
    monkeypatch.setattr(solve, 'inspect_prepared_task', lambda p: contract)
    monkeypatch.setattr(solve, 'rollout_contract_options', lambda c: {
        'task_path': prepared, 'skip_verify': True, 'prompts': ['exact\n'],
    })
    monkeypatch.setattr(solve, '_model_route', lambda c: ({}, {'endpoint': None}))
    config = solve.SolveConfig(prepared, tmp_path / 'run', 'dummy', 'model')
    events = []
    r = SimpleNamespace(_env=object())

    def operation(name):
        async def call():
            events.append(name)
        return AsyncMock(side_effect=call)
    for phase in ('setup', 'start', 'install_agent', 'connect', 'execute', 'disconnect', 'cleanup'):
        setattr(r, phase, operation(phase))
    native = SimpleNamespace(error=None, error_category=None, export_error=None,
                             verifier_error=None, scoring=None)

    async def finalize():
        events.append('finalize')
        r._rollout_dir.mkdir(parents=True, exist_ok=True)
        (r._rollout_dir / 'result.json').write_text('{}')
        return native
    r.finalize = AsyncMock(side_effect=finalize)

    async def create(cfg):
        assert cfg.skip_verify is True
        assert cfg.prompts == ['exact\n']
        r._rollout_dir = Path(cfg.jobs_dir) / 'job/rollout'
        return r
    import benchflow
    monkeypatch.setattr(benchflow.Rollout, 'create', AsyncMock(side_effect=create))
    monkeypatch.setattr(solve, '_container_id', AsyncMock(return_value='a' * 64))
    monkeypatch.setattr(solve, '_runtime_evidence', AsyncMock(return_value={'agent_uid': 1000, 'image_id': 'sha256:' + 'b' * 64}))
    monkeypatch.setattr(solve, '_docker', AsyncMock(return_value=(0, b'', b'')))

    async def export(cid, layout, specs, limit):
        events.append('export')
        layout.outputs_dir.mkdir()
        for spec in specs:
            (layout.outputs_dir / spec.name).write_bytes(b'test')
        return solve.collect_outputs(layout.outputs_dir, specs)
    exporter = AsyncMock(side_effect=export)
    monkeypatch.setattr(solve, '_freeze_export', exporter)
    return SimpleNamespace(config=config, r=r, native=native, events=events,
                           exporter=exporter, outputs=outputs)


async def test_solve_native_layout_order_and_unscored_result(runtime):
    result = await solve.solve_task(runtime.config)
    assert result['status'] == 'completed'
    assert result['evaluation_status'] == 'pending'
    assert 'final_score' not in result
    assert runtime.events == ['setup', 'start', 'install_agent', 'connect', 'execute',
                              'disconnect', 'export', 'finalize']
    path = runtime.config.run_dir / 'math.mpsc_safety_filter/math.mpsc_safety_filter__seed31415__b1.json'
    persisted = json.loads(path.read_text())['agent_output']['persisted_outputs']
    assert persisted['dir'] == 'math.mpsc_safety_filter__seed31415__b1.outputs'
    assert {item['path'] for item in persisted['files']} == {'analysis.py', 'certificate.json'}
    assert (runtime.config.run_dir / result['native_result']).is_file()
    assert len(list(path.parent.glob('*.json'))) == 1


@pytest.mark.parametrize('phase', ['setup', 'start', 'install_agent', 'connect', 'execute'])
async def test_phase_failure_keeps_partial_artifacts_and_primary_error(runtime, phase, capsys):
    getattr(runtime.r, phase).side_effect = RuntimeError('secret-value-never-persist')
    result = await solve.solve_task(runtime.config)
    assert result['attempt_status'] == 'execution_failed'
    assert result['status'] == 'failed'
    assert result['collection_status'] == 'collected'
    assert result['diagnostics'][0] == {'phase': phase, 'error_type': 'RuntimeError'}
    assert 'finalize' in runtime.events
    stderr = capsys.readouterr().err
    assert 'Traceback (most recent call last)' in stderr
    assert 'RuntimeError: secret-value-never-persist' in stderr
    for path in runtime.config.run_dir.rglob('*.json'):
        assert 'secret-value-never-persist' not in path.read_text()


async def test_native_error_not_ignored(runtime):
    runtime.native.error = 'agent failed'
    result = await solve.solve_task(runtime.config)
    assert result['status'] == 'failed'
    assert result['attempt_status'] == 'execution_failed'
    assert result['prediction_status'] == 'present'


async def test_transfer_failure_does_not_claim_missing_predictions(runtime):
    runtime.exporter.side_effect = OSError('transport')
    result = await solve.solve_task(runtime.config)
    assert result['status'] == 'failed'
    assert result['collection_status'] == 'failed'
    assert result['prediction_status'] == 'unavailable'
    assert result['agent_output']['persisted_outputs'] is None
    assert result['attempt_status'] == 'completed'
    assert 'finalize' in runtime.events


async def test_cleanup_failure_keeps_primary_agent_error(runtime):
    runtime.r.execute.side_effect = RuntimeError('agent')
    runtime.r.finalize.side_effect = OSError('cleanup')
    result = await solve.solve_task(runtime.config)
    assert result['diagnostics'][:2] == [
        {'phase': 'execute', 'error_type': 'RuntimeError'},
        {'phase': 'finalize', 'error_type': 'OSError'},
    ]
    assert result['cleanup_status'] == 'failed'
    runtime.r.cleanup.assert_awaited_once()


async def test_native_suppressed_cleanup_failure_is_detected(runtime, monkeypatch):
    cli = AsyncMock(side_effect=[(0, b'a' * 64, b''), (0, b'', b'')])
    monkeypatch.setattr(solve, '_docker', cli)
    result = await solve.solve_task(runtime.config)
    assert result['cleanup_status'] == 'failed'
    assert result['status'] == 'failed'
    assert cli.await_args_list[1].args[0] == ['rm', '-f', 'a' * 64]


async def test_timeout_is_not_success(runtime):
    """Guards S5/S6 status drift identified in review after 9b90d16c."""
    runtime.r.execute.side_effect = TimeoutError()
    result = await solve.solve_task(runtime.config)
    assert result['attempt_status'] == 'execution_failed'
    assert result['failure_reason'] == 'timeout'
    assert result['status'] == 'failed'
    assert result['collection_status'] == 'collected'


async def test_cancellation_persists_partial_result_then_reraises(runtime):
    runtime.r.execute.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await solve.solve_task(runtime.config)
    meta = json.loads((runtime.config.run_dir / 'run_metadata.json').read_text())
    assert meta['attempt_status'] == 'execution_failed'
    assert meta['failure_reason'] == 'cancelled'
    result = json.loads((runtime.config.run_dir / meta['result_file']).read_text())
    assert result['collection_status'] == 'collected'
    assert result['cleanup_status'] == 'completed'


async def test_unsettled_phase_never_publishes(runtime):
    runtime.r.execute.side_effect = solve.UnsettledPhase()
    result = await solve.solve_task(runtime.config)
    runtime.exporter.assert_not_awaited()
    assert result['collection_status'] == 'not_collected'
    assert result['cleanup_status'] == 'unconfirmed'
    assert result['status'] == 'failed'


async def test_bounded_cancels_and_joins():
    stopped = asyncio.Event()
    async def hang():
        try:
            await asyncio.sleep(100)
        finally:
            stopped.set()
    with pytest.raises(TimeoutError):
        await solve._bounded(hang(), 0.01)
    assert stopped.is_set()


async def test_export_stops_before_copy_and_excludes_missing(tmp_path, monkeypatch):
    layout = solve.result_layout(tmp_path, 'math.mpsc_safety_filter',
                                'math.mpsc_safety_filter__seed31415', 'b1')
    calls = []
    monkeypatch.setattr(solve, '_container_info', AsyncMock(side_effect=[
        {'Mounts': []}, {'State': {'Running': False, 'Pid': 0}},
    ]))
    async def cli(args, **kwargs):
        calls.append(args)
        if args[0] == 'stop':
            return 0, b'', b''
        if 'analysis.py' in args[1]:
            return 0, archive(), b''
        return 1, b'', b'Error response from daemon: Could not find the file /workspace/certificate.json in container abc'
    monkeypatch.setattr(solve, '_docker', cli)
    specs = (solve.OutputSpec('analysis.py', 'code'), solve.OutputSpec('certificate.json', 'data'))
    result = await solve._freeze_export('abc', layout, specs, 100)
    assert calls[0] == ['stop', '--time', '5', 'abc']
    assert result['missing'] == ['certificate.json']
    assert (layout.outputs_dir / 'analysis.py').read_bytes() == b'print(1)\n'
    persisted = solve.persisted_outputs(layout, specs, result)
    assert persisted['files'][1] == {'path': 'certificate.json', 'missing': True}
    assert not list(tmp_path.rglob('*.staging-*'))


@pytest.mark.parametrize('fault', ['transport', 'overflow', 'running', 'mount'])
async def test_export_failure_never_publishes_partial_directory(tmp_path, monkeypatch, fault):
    layout = solve.result_layout(tmp_path, 'math.mpsc_safety_filter',
                                'math.mpsc_safety_filter__seed31415', 'b1')
    monkeypatch.setattr(solve, '_container_info', AsyncMock(side_effect=[
        {'Mounts': [{'Destination': '/workspace'}] if fault == 'mount' else []},
        {'State': {'Running': fault == 'running', 'Pid': 1 if fault == 'running' else 0}},
    ]))
    async def cli(args, **kwargs):
        if args[0] == 'stop':
            return 0, b'', b''
        if 'analysis.py' in args[1]:
            return 0, archive(), b''
        if fault == 'overflow':
            raise ValueError('output limit')
        return 1, b'', b'connection reset'
    monkeypatch.setattr(solve, '_docker', cli)
    with pytest.raises((RuntimeError, ValueError)):
        await solve._freeze_export('abc', layout, (
            solve.OutputSpec('analysis.py', 'code'), solve.OutputSpec('certificate.json', 'data')), 100)
    assert not layout.outputs_dir.exists()
    assert not list(tmp_path.rglob('*.staging-*'))


def test_model_route_does_not_rewrite_url_or_persist_secret(tmp_path):
    env = tmp_path / 'model.env'
    env.write_text('ASI_MODEL_BASE_URL=https://example.test/v\nASI_MODEL_API_KEY=top-secret\n')
    cfg = solve.SolveConfig(tmp_path, tmp_path / 'run', 'agent', 'model', model_env_file=env)
    values, provenance = solve._model_route(cfg)
    assert values['BENCHFLOW_PROVIDER_BASE_URL'] == 'https://example.test/v'
    assert provenance['endpoint'] == 'https://example.test/v'
    assert 'top-secret' not in json.dumps(provenance)


async def test_firewall_checked_after_connect_before_execute(runtime, monkeypatch):
    async def evidence(*args):
        assert runtime.events[-1] == 'connect'
        assert 'execute' not in runtime.events
        return {'agent_uid': 1000, 'image_id': 'sha256:' + 'b' * 64}
    monkeypatch.setattr(solve, '_runtime_evidence', evidence)
    assert (await solve.solve_task(runtime.config))['status'] == 'completed'


async def test_repeated_cancellation_does_not_interrupt_export_and_cleanup(runtime):
    started, allow_finish = asyncio.Event(), asyncio.Event()
    original = runtime.exporter.side_effect
    async def export(*args):
        started.set()
        await allow_finish.wait()
        return await original(*args)
    runtime.r.execute.side_effect = asyncio.CancelledError()
    runtime.exporter.side_effect = export
    task = asyncio.create_task(solve.solve_task(runtime.config))
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    allow_finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    metadata = json.loads((runtime.config.run_dir / 'run_metadata.json').read_text())
    result = json.loads((runtime.config.run_dir / metadata['result_file']).read_text())
    assert result['attempt_status'] == 'execution_failed'
    assert result['failure_reason'] == 'cancelled'
    assert result['cleanup_status'] == 'completed'
    assert result['prediction_status'] == 'present'


async def test_self_cancelled_finalize_does_not_lose_result(runtime):
    runtime.r.finalize.side_effect = asyncio.CancelledError()
    result = await solve.solve_task(runtime.config)
    assert result['cleanup_status'] == 'failed'
    assert result['status'] == 'failed'
    runtime.r.cleanup.assert_awaited_once()


async def test_missing_all_has_no_directory_or_fake_produced_files(runtime):
    runtime.exporter.side_effect = None
    runtime.exporter.return_value = {
        'artifacts': [], 'missing': ['analysis.py', 'certificate.json'],
        'invalid': [], 'prediction_status': 'missing',
    }
    result = await solve.solve_task(runtime.config)
    assert result['agent_output']['code_files'] == []
    assert result['agent_output']['data_files'] == []
    assert result['agent_output']['persisted_outputs']['dir'] is None
    assert all(item['missing'] for item in result['agent_output']['persisted_outputs']['files'])
    assert result['attempt_status'] == 'completed'
    assert result['status'] == 'failed'


def test_reserve_protects_custom_raw_assets(tmp_path):
    raw = tmp_path / 'custom-raw'
    raw.mkdir()
    (raw / 'sources.json').write_text('{}')
    prepared = tmp_path / 'prepared'
    prepared.mkdir()
    with pytest.raises(ValueError):
        solve.reserve_run(raw / 'new-run', prepared)


async def test_no_container_is_not_reported_as_export_success(runtime, monkeypatch):
    monkeypatch.setattr(solve, '_container_id', AsyncMock(return_value=None))
    result = await solve.solve_task(runtime.config)
    runtime.exporter.assert_not_awaited()
    assert result['attempt_status'] == 'execution_failed'
    assert result['collection_status'] == 'not_collected'
    assert result['agent_output']['persisted_outputs'] is None


async def test_freeze_refuses_symlink_but_collects_valid_sibling(tmp_path, monkeypatch):
    layout = solve.result_layout(tmp_path, 'math.mpsc_safety_filter',
                                'math.mpsc_safety_filter__seed31415', 'b1')
    monkeypatch.setattr(solve, '_container_info', AsyncMock(side_effect=[
        {'Mounts': []}, {'State': {'Running': False, 'Pid': 0}},
    ]))
    async def cli(args, **kwargs):
        if args[0] == 'stop':
            return 0, b'', b''
        name = args[1].rsplit('/', 1)[-1]
        return 0, archive(name, kind=tarfile.SYMTYPE if name == 'certificate.json' else tarfile.REGTYPE), b''
    monkeypatch.setattr(solve, '_docker', cli)
    result = await solve._freeze_export('abc', layout, (
        solve.OutputSpec('analysis.py', 'code'), solve.OutputSpec('certificate.json', 'data')), 100)
    assert result['prediction_status'] == 'invalid'
    assert result['missing'] == []
    assert result['invalid'] == [{'name': 'certificate.json', 'reason': 'unsafe_artifact_archive'}]
    assert not (layout.outputs_dir / 'certificate.json').exists()
    assert (layout.outputs_dir / 'analysis.py').read_bytes() == b'print(1)\n'


async def test_create_failure_still_writes_diagnostic_result(runtime, monkeypatch):
    import benchflow
    monkeypatch.setattr(benchflow.Rollout, 'create', AsyncMock(side_effect=ValueError('key')))
    result = await solve.solve_task(runtime.config)
    assert result['cleanup_status'] == 'not_needed'
    assert result['native_result'] is None
    assert result['collection_status'] == 'not_collected'
    assert result['status'] == 'failed'


@pytest.mark.parametrize('field', ['verifier_error', 'scoring', 'export_error'])
async def test_native_secondary_failures_not_ignored(runtime, field):
    setattr(runtime.native, field, 'unexpected')
    result = await solve.solve_task(runtime.config)
    assert result['status'] == 'failed'
    assert result['diagnostics']


def test_execution_view_name_is_valid_docker_image_component(tmp_path, monkeypatch):
    """Guards S6 Docker smoke regression on 9b90d16c: tempfile suffix may end '_'."""
    import re

    from benchflow._utils.config_override import apply_config_override
    from benchflow.task import RolloutPaths
    from benchflow.task.config import TaskConfig
    from benchmarks.asi_bench.tests.test_solve import PROFILE

    # Reproduce the suffix seen in a failed real Docker build, deterministically.
    def bad_temp(*, prefix, dir):
        path = Path(dir) / (prefix + 'aagmygl_')
        path.mkdir()
        return str(path)
    monkeypatch.setattr(solve.tempfile, 'mkdtemp', bad_temp)
    monkeypatch.setattr(solve.DefaultRolloutPlanes, 'create_environment',
                        lambda self, environment, task, task_path, *args, **kw: task_path)
    (tmp_path / 'environment').mkdir()
    paths = RolloutPaths(rollout_dir=tmp_path / 'rollout')
    paths.mkdir()
    task = SimpleNamespace(config=apply_config_override(TaskConfig(), solve.runtime_override(PROFILE, 300)))
    view = solve.ASIDockerPlanes().create_environment(
        'docker', task, tmp_path, 'test', paths,
        preserve_agent_network=True, environment_manifest=None,
    )
    assert re.fullmatch(r'[a-z0-9]+(?:[._-][a-z0-9]+)*', view.name)


@pytest.mark.parametrize('status,expected', [('completed', 0), ('failed', 1)])
def test_cli_exit_status_tracks_aggregate_outcome(tmp_path, monkeypatch, capsys, status, expected):
    import sys
    monkeypatch.setattr(sys, 'argv', ['solve', '--prepared-dir', str(tmp_path),
                                     '--run-dir', str(tmp_path / 'run'),
                                     '--agent', 'dummy', '--model', 'model'])
    monkeypatch.setattr(solve, 'solve_task', AsyncMock(return_value={
        'status': status, 'attempt_status': 'completed', 'collection_status': 'collected',
        'prediction_status': 'missing' if status == 'failed' else 'present',
        'cleanup_status': 'completed', 'evaluation_status': 'pending',
    }))
    assert solve.main() == expected
    assert json.loads(capsys.readouterr().out)['status'] == status


def test_background_fixture_child_source_compiles():
    """Guards S6 dummy child fixture on 9b90d16c: nested exec newline quoting."""
    import ast
    tree = ast.parse((Path(__file__).parent / 'fixtures/s6_dummy.py').read_text())
    assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'child_code' for t in n.targets))
    compile(ast.literal_eval(assignment.value), '<s6-background-child>', 'exec')


def test_reserve_rejects_dangling_run_symlink(tmp_path):
    prepared = tmp_path / 'prepared'
    prepared.mkdir()
    target, link = tmp_path / 'target', tmp_path / 'link'
    link.symlink_to(target)
    with pytest.raises(ValueError):
        solve.reserve_run(link, prepared)
    assert not target.exists()


async def test_export_detects_corruption_with_single_collection_read(tmp_path, monkeypatch):
    """Guards checksum consolidation from solve review after 9b90d16c."""
    layout = solve.result_layout(tmp_path, 'math.mpsc_safety_filter',
                                'math.mpsc_safety_filter__seed31415', 'b1')
    monkeypatch.setattr(solve, '_container_info', AsyncMock(side_effect=[
        {'Mounts': []}, {'State': {'Running': False, 'Pid': 0}},
    ]))
    monkeypatch.setattr(solve, '_docker', AsyncMock(side_effect=[
        (0, b'', b''), (0, archive(), b''),
    ]))
    original_write = Path.write_bytes

    def corrupt(path, body):
        return original_write(path, b'corrupted')

    monkeypatch.setattr(Path, 'write_bytes', corrupt)
    with pytest.raises(ValueError, match='checksum mismatch'):
        await solve._freeze_export('abc', layout, (solve.OutputSpec('analysis.py', 'code'),), 100)
    assert not layout.outputs_dir.exists()
    assert not list(tmp_path.rglob('*.staging-*'))


async def test_cancellation_during_successful_finishing_is_not_overwritten(runtime):
    """Guards extracted finish result overriding cancellation after 9b90d16c."""
    started, release = asyncio.Event(), asyncio.Event()
    original = runtime.exporter.side_effect

    async def export(*args):
        started.set()
        await release.wait()
        return await original(*args)

    runtime.exporter.side_effect = export
    task = asyncio.create_task(solve.solve_task(runtime.config))
    await started.wait()
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    metadata = json.loads((runtime.config.run_dir / 'run_metadata.json').read_text())
    result = json.loads((runtime.config.run_dir / metadata['result_file']).read_text())
    assert result['attempt_status'] == 'execution_failed'
    assert result['failure_reason'] == 'cancelled'
    assert result['collection_status'] == 'collected'
    assert result['cleanup_status'] == 'completed'
    assert result['status'] == 'failed'


async def test_p22_prepared_identity_is_persisted(runtime):
    """Guards P2.2: a completed result binds the prepared bytes used at start."""
    import hashlib
    result = await solve.solve_task(runtime.config)
    expected = {'converter_version': 'prepare-v5', 'manifest_sha256': hashlib.sha256(
        (runtime.config.prepared_dir / 'task_manifest.json').read_bytes()).hexdigest()}
    assert result['prepared'] == expected
    meta = json.loads((runtime.config.run_dir / 'run_metadata.json').read_text())
    assert meta['prepared'] == expected


async def test_p22_prepared_mutation_rejected_before_run(runtime, monkeypatch):
    """Guards P2.2 prepared binding against mutation during initial inspection."""
    original = solve.inspect_prepared_task

    def mutate(path):
        contract = original(path)
        (path / 'task_manifest.json').write_text('{"changed":true}')
        return contract

    monkeypatch.setattr(solve, 'inspect_prepared_task', mutate)
    with pytest.raises(ValueError, match='manifest changed'):
        await solve.solve_task(runtime.config)
    assert not runtime.config.run_dir.exists()
    assert not runtime.events


@pytest.mark.parametrize('error,visible', [
    (solve.InventoryError('inventory mismatch: missing ["data/input.json"]'), True),
    (ValueError('secret-model-key'), False),
])
def test_cli_prints_raw_traceback_without_persisting_it(tmp_path, monkeypatch, capsys, error, visible):
    """Guards raw stderr diagnostics following BenchFlow 9b90d16c."""
    import sys
    monkeypatch.setattr(sys, 'argv', ['solve', '--prepared-dir', str(tmp_path),
                                     '--run-dir', str(tmp_path / 'run'),
                                     '--agent', 'dummy', '--model', 'model'])
    monkeypatch.setattr(solve, 'solve_task', AsyncMock(side_effect=error))
    assert solve.main() == 1
    captured = capsys.readouterr()
    assert f'{type(error).__name__}: {error}' in captured.err
    assert 'Traceback (most recent call last)' in captured.err
    output = captured.out
    report = json.loads(output)
    assert report['error_type'] == 'ValueError'
    assert ('error' in report) == visible
    assert 'secret-model-key' not in output


def test_execution_view_excludes_finder_metadata(tmp_path, monkeypatch):
    """Guards Docker context filtering following BenchFlow 9b90d16c."""
    from benchflow._utils.config_override import apply_config_override
    from benchflow.task import RolloutPaths
    from benchflow.task.config import TaskConfig
    from benchmarks.asi_bench.tests.test_solve import PROFILE

    monkeypatch.setattr(solve.DefaultRolloutPlanes, 'create_environment',
                        lambda self, environment, task, task_path, *args, **kw: task_path)
    inputs = tmp_path / 'environment/inputs'
    inputs.mkdir(parents=True)
    (inputs / '.DS_Store').write_text('Finder')
    (inputs / 'prompt.md').write_text('task')
    paths = RolloutPaths(rollout_dir=tmp_path / 'rollout')
    paths.mkdir()
    task = SimpleNamespace(config=apply_config_override(TaskConfig(), solve.runtime_override(PROFILE, 300)))
    view = solve.ASIDockerPlanes().create_environment(
        'docker', task, tmp_path, 'test', paths,
        preserve_agent_network=True, environment_manifest=None,
    )
    assert not list(view.rglob('.DS_Store'))
    assert (view / 'environment/inputs/prompt.md').read_text() == 'task'
    assert (inputs / '.DS_Store').is_file()

