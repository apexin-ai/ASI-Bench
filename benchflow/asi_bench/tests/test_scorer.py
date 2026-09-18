"""P1.2 materialization contract for ASI revision 5935b5f33549 (no task execution)."""
import copy
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.asi_bench import scorer


@pytest.fixture(scope='module')
def spec():
    return scorer.load_evaluator_spec()


@pytest.fixture
def source(tmp_path, spec):
    upstream = os.environ.get('ASI_BENCH_SOURCE')
    assert upstream, 'Set ASI_BENCH_SOURCE to the pinned ASI Git checkout'
    root = tmp_path / 'source'
    root.mkdir()
    subprocess.run(['git', 'init', '-q', str(root)], check=True)
    # Keep the actual pinned Git history available without copying user files.
    git_dir = subprocess.check_output(['git', '-C', upstream, 'rev-parse', '--absolute-git-dir'], text=True).strip()
    (root / '.git/objects/info/alternates').write_text(str(Path(git_dir) / 'objects') + '\n')
    subprocess.run(['git', '-C', str(root), 'read-tree', spec['upstream']['revision']], check=True)
    for entry in spec['files']:
        if entry['kind'] != 'copy':
            continue
        path = root / entry['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(subprocess.check_output(
            ['git', '-C', upstream, 'show', f"{spec['upstream']['revision']}:{entry['source']}"]))
    return root


def test_materialize_direct_copy_and_manifest(source, tmp_path, spec):
    """Guards simplification of 9b90d16c: explicit destination, no shared cache."""
    destination = tmp_path / 'evaluator'
    bundle = scorer.materialize_evaluator(source, destination, spec)
    report = json.loads((bundle / 'evaluator_manifest.json').read_text())
    assert bundle == destination
    assert report == scorer._manifest(spec)
    assert len(report['files']) == 11
    for entry in spec['files']:
        data = (bundle / entry['path']).read_bytes()
        assert scorer._sha(data) == entry['sha256']
        if entry['kind'] == 'copy':
            assert data == (source / entry['source']).read_bytes()
        elif entry['kind'] == 'runtime_adapter':
            assert data == (scorer.ADAPTER_ROOT / 'templates/verifier/task_env.py').read_bytes()
        else:
            assert data == scorer.INITIALIZERS[entry['path']].encode()
    assert {p.name for p in tmp_path.iterdir()} == {'source', 'evaluator'}
    with pytest.raises(FileExistsError):
        scorer.materialize_evaluator(source, destination, spec)


@pytest.mark.parametrize('damage', ['changed', 'missing', 'symlink', 'fifo', 'parent_symlink'])
def test_reject_source_damage(source, tmp_path, spec, damage):
    path = source / 'ai4sci_bench/core/types.py'
    if damage == 'changed':
        path.write_text('changed')
    elif damage == 'missing':
        path.unlink()
    elif damage == 'symlink':
        path.unlink()
        path.symlink_to('/dev/null')
    elif damage == 'fifo':
        path.unlink()
        os.mkfifo(path)
    elif damage == 'parent_symlink':
        parent = path.parent
        parent.rename(source / 'external-core')
        parent.symlink_to(source / 'external-core', target_is_directory=True)
    cache = tmp_path / 'cache'
    with pytest.raises((ValueError, FileNotFoundError)):
        scorer.materialize_evaluator(source, cache, spec)
    assert not cache.exists() or not list(cache.iterdir())


@pytest.mark.parametrize('path', ['../escape.py', '/abs.py', 'ai4sci_bench/../x.py',
                                  'ai4sci_bench//x.py', 'ai4sci_bench\\x.py'])
def test_unsafe_spec_path(tmp_path, spec, path):
    bad = copy.deepcopy(spec)
    bad['files'][0]['path'] = path
    location = tmp_path / 'bad.json'
    location.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        scorer.load_evaluator_spec(location)


def test_failed_copy_removes_partial_destination(source, tmp_path, spec, monkeypatch):
    """Guards direct-copy cleanup replacing the cache publisher from 9b90d16c."""
    write = Path.write_bytes
    destination = tmp_path / 'evaluator'
    def fail(path, data):
        if path.name == 'types.py':
            raise OSError('simulated copy failure')
        return write(path, data)
    monkeypatch.setattr(Path, 'write_bytes', fail)
    with pytest.raises(OSError, match='simulated'):
        scorer.materialize_evaluator(source, destination, spec)
    assert not destination.exists()


@pytest.mark.parametrize('entry', ['bridge', 'launcher', 'scoring'])
def test_local_hash_is_derived(tmp_path, spec, entry):
    """Guards simplification of 9b90d16c: local code needs no manual hash update."""
    changed = copy.deepcopy(spec)
    changed[entry].pop('sha256')
    location = tmp_path / 'spec.json'
    location.write_text(json.dumps(changed))
    actual = scorer.load_evaluator_spec(location)
    assert actual[entry]['sha256'] == scorer._sha(
        (scorer.ADAPTER_ROOT / actual[entry]['path']).read_bytes())


def test_p21_static_scoring_identity(spec, monkeypatch):
    """Guards P2.1: reuse the bundle identity without materializing or running it."""
    def forbidden(*args, **kwargs):
        raise AssertionError('static identity must not execute or materialize')
    monkeypatch.setattr(scorer, 'materialize_evaluator', forbidden)
    monkeypatch.setattr(scorer.subprocess, 'run', forbidden)
    identity = scorer.scoring_profile_identity(spec)
    expected = scorer._manifest(spec)
    assert identity['expected_evaluator_bundle_digest'] == expected['bundle_digest']
    assert identity['evaluator_spec_sha256'] == expected['spec_sha256']
    assert identity['launcher'] == 'verifier/test.sh'
    assert identity == scorer.scoring_profile_identity(spec)


@pytest.fixture
def scoring_chain(tmp_path, request):
    """Synthetic execution evidence, never a real Agent acceptance claim (P2.2)."""
    import hashlib

    import yaml

    from benchmarks.asi_bench.prepare import PrepareConfig, prepare_task
    from benchmarks.asi_bench.solve import OutputSpec, collect_outputs
    task, level, code, data = getattr(request, 'param', ('math.mpsc_safety_filter', 'b1', 'analysis.py', 'certificate.json'))
    iid = task + '__seed31415'
    raw = tmp_path / 'raw'
    instance = raw / 'instance' / iid
    bundle = raw / ('task_bundle/tasks/' + task.replace('.', '/'))
    bundle.mkdir(parents=True)
    metadata = {'id': task, 'runtime': {'python': '>=3.11', 'packages': []},
                'input': {'files': [{'name': 'system.json'}, {'name': 'public_cases.json'}]},
                'output': {'files': [{'name': code, 'type': 'code'}, {'name': data, 'type': 'data'}]}}
    if task != 'math.mpsc_safety_filter':
        metadata['input'] = {'files': []}
    (bundle / 'task_meta.yaml').write_text(yaml.safe_dump(metadata))
    (bundle / 'task_eval.yaml').write_text('task_id: ' + task)
    for name in (('custom_scorer.py', 'mpsc_eval_runtime.py') if task == 'math.mpsc_safety_filter' else ()):
        (bundle / name).write_text('raise AssertionError("never import task code on host")')
    instance_files = ('data/system.json', 'data/public_cases.json', 'reference/certificate.json',
                 'reference/hidden_cases.json', 'reference/data/system.json', 'reference/data/public_cases.json')
    for name in (instance_files if task == 'math.mpsc_safety_filter' else ('reference/answer.txt',)):
        p = instance / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('{}')
    (instance / 'instance_meta.json').write_text(json.dumps({'task_id': task, 'params_used': {}}))
    (instance / f'prompt_{level}.md').write_text('test prompt')
    sources = {'hf': {'repo': 'Apexintelligence-AI/ASI-Bench-seed31415', 'path': 'tasks/' + iid, 'resolved_revision': 'a'*40},
               'github': {'repo': 'apexin-ai/ASI-Bench', 'path': ('tasks/' + task.replace('.', '/')),
                          'resolved_revision': ('a797bf69683400bed0f38c4add507bb3d404b833' if task == 'math.mpsc_safety_filter' else 'b'*40)}}
    raw_manifest = {'seed': 31415, 'task_id': task, 'instance_id': iid, **sources,
                    'files': [{'path': p.relative_to(raw).as_posix(), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                              for p in raw.rglob('*') if p.is_file()]}
    (raw / 'sources.json').write_text(json.dumps(raw_manifest))
    prepared = prepare_task(PrepareConfig(raw, task, iid, level, tmp_path / 'prepared')).output_dir
    run = tmp_path / 'run'
    task_dir = run / task
    prediction = task_dir / (iid + f'__{level}.outputs')
    prediction.mkdir(parents=True)
    (prediction / code).parent.mkdir(parents=True, exist_ok=True)
    (prediction / data).parent.mkdir(parents=True, exist_ok=True)
    (prediction / code).write_text('raise AssertionError("never import prediction on host")')
    (prediction / data).write_text('{}')
    collection = collect_outputs(prediction, (OutputSpec(code, 'code'), OutputSpec(data, 'data')))
    collection['freeze'] = {'method': 'docker-stop', 'container_id': 'c'*64, 'running': False, 'pid': 0, 'shared_workspace': False}
    native_path = Path('benchflow/job') / (prepared.name + '__test') / 'result.json'
    native = {'task_name': prepared.name, 'rollout_name': native_path.parent.name, 'agent': 'dummy', 'model': 'dummy',
              **dict.fromkeys(('error', 'error_category', 'verifier_error', 'export_error', 'scoring', 'rewards')),
              'finished_at': '2026-09-17T00:00:01+00:00'}
    (run / native_path).parent.mkdir(parents=True)
    (run / native_path).write_text(json.dumps(native))
    result_path = task_dir / (iid + f'__{level}.json')
    meta = {'adapter_schema_version': 1, 'official': False, 'mode': 'produce-only', 'task_id': task,
            'instance_id': iid, 'prompt_level': level, 'attempt': 1, 'agent': 'dummy', 'model': 'dummy',
            'requested_effort': None, 'model_route': {'endpoint': 'https://example.test/v1', 'api_key_env': 'ASI_MODEL_API_KEY', 'routing': 'explicit'},
            'sources': sources, 'prepared': {'converter_version': 'prepare-v5',
                'manifest_sha256': hashlib.sha256((prepared/'task_manifest.json').read_bytes()).hexdigest()},
            'started_at': '2026-09-17T00:00:00+00:00', 'budgets': {},
            'status': 'completed', 'attempt_status': 'completed', 'failure_reason': None, 'evaluation_status': 'pending'}
    result = {**meta, 'collection_status': 'collected', 'prediction_status': 'present', 'cleanup_status': 'completed',
              'native_result': native_path.as_posix(), 'native_error_present': False, 'native_error_category': None,
              'diagnostics': [], 'collection': collection, 'network': {'container_id': 'c'*64, 'image_id': 'sha256:'+'d'*64},
              'agent_output': {'code_files': [code], 'data_files': [data],
                   'persisted_outputs': {'dir': prediction.name, 'files': [
                       {'path': a['name'], 'bytes': a['size'], 'sha256': a['sha256']} for a in collection['artifacts']]}}}
    result_path.write_text(json.dumps(result))
    (run/'run_metadata.json').write_text(json.dumps({**meta, 'result_file': result_path.relative_to(run).as_posix(), 'finished_at': native['finished_at']}))
    return prepared, result_path, raw


def test_p22_inspection_snapshot_and_cleanup(scoring_chain, monkeypatch):
    """Guards P2.2: freeze only bound inputs, without imports or Docker."""
    prepared, result, raw = scoring_chain
    def forbidden(*args, **kwargs):
        raise AssertionError('no Docker or source execution during inspection')
    monkeypatch.setattr(scorer.subprocess, 'run', forbidden)
    before = {p: scorer._input_tree(p) for p in (prepared, raw, result.parent.parent)}
    checked = scorer.inspect_scoring_inputs(prepared, result)
    assert checked['host_provenance']['model_route']['endpoint'] == 'https://example.test/v1'
    with scorer.scoring_input_snapshot(checked) as snapshot:
        root = snapshot['input_dir']
        assert scorer._input_tree(root) == snapshot['request_fields']['files']
        assert 'runtime' not in snapshot['request_fields']
        assert not (root/'evaluation_manifest.json').exists()
        assert not any('prompt' in p for p in scorer._input_tree(root))
        assert all(not (p.stat().st_mode & 0o222) for p in root.rglob('*'))
    assert not root.parent.exists()
    assert before == {p: scorer._input_tree(p) for p in before}


@pytest.mark.parametrize('mutation', ['prepared', 'metadata', 'seed42', 'attempt', 'cleanup', 'native_error',
    'native_missing', 'native_verifier', 'native_identity', 'prediction_hash', 'prediction_missing', 'extra',
    'duplicate', 'collection', 'escape', 'symlink', 'fifo', 'reference', 'freeze', 'missing_prepared',
    'route_secret', 'scoring', 'old_version'])
def test_p22_reject_invalid_evidence(scoring_chain, mutation):
    """Guards P2.2 fail-closed result, source and persisted-output binding."""
    prepared, result_path, raw = scoring_chain
    run = result_path.parent.parent
    result = json.loads(result_path.read_text())
    meta_path = run/'run_metadata.json'
    meta = json.loads(meta_path.read_text())
    outputs = result_path.parent / result['agent_output']['persisted_outputs']['dir']
    native_path = run / result['native_result']
    if mutation == 'prepared':
        result['prepared']['manifest_sha256'] = '0'*64
    elif mutation == 'metadata':
        meta['result_file'] = 'other.json'
    elif mutation == 'seed42':
        result['instance_id'] = 'math.mpsc_safety_filter__seed42'
    elif mutation == 'attempt':
        result['attempt_status'] = 'execution_failed'
    elif mutation == 'cleanup':
        result['cleanup_status'] = 'failed'
    elif mutation == 'native_missing':
        native_path.unlink()
    elif mutation.startswith('native_'):
        native = json.loads(native_path.read_text())
        native[{'native_error': 'error', 'native_verifier': 'rewards', 'native_identity': 'model'}[mutation]] = 'invalid'
        native_path.write_text(json.dumps(native))
    elif mutation == 'prediction_hash':
        (outputs/'analysis.py').write_text('changed')
    elif mutation == 'prediction_missing':
        (outputs/'analysis.py').unlink()
    elif mutation == 'extra':
        (outputs/'extra.txt').write_text('extra')
    elif mutation == 'duplicate':
        result['agent_output']['persisted_outputs']['files'][1] = result['agent_output']['persisted_outputs']['files'][0]
    elif mutation == 'collection':
        result['collection']['artifacts'][0]['sha256'] = '0'*64
    elif mutation == 'escape':
        result['agent_output']['persisted_outputs']['dir'] = '../outside'
    elif mutation in ('symlink', 'fifo'):
        p = outputs/'analysis.py'
        p.unlink()
        if mutation == 'symlink':
            p.symlink_to(prepared/'task.md')
        else:
            os.mkfifo(p)
    elif mutation == 'reference':
        (raw/'instance/math.mpsc_safety_filter__seed31415/reference/hidden_cases.json').unlink()
    elif mutation == 'freeze':
        result['collection']['freeze']['running'] = True
    elif mutation == 'missing_prepared':
        result.pop('prepared')
        meta.pop('prepared')
    elif mutation == 'route_secret':
        result['model_route']['api_key'] = 'never-persist'
        meta['model_route'] = result['model_route']
    elif mutation in ('scoring', 'old_version'):
        p = prepared/'task_manifest.json'
        m = json.loads(p.read_text())
        if mutation == 'scoring':
            m['scoring']['evaluator_spec_sha256'] = '0'*64
        else:
            m['converter_version'] = 'prepare-v3'
        p.write_text(json.dumps(m))
    result_path.write_text(json.dumps(result))
    meta_path.write_text(json.dumps(meta))
    with pytest.raises((ValueError, OSError)):
        scorer.inspect_scoring_inputs(prepared, result_path)


def test_p22_snapshot_race_cleans_partial(scoring_chain, monkeypatch, tmp_path):
    """Guards P2.2: mutation during copy must never publish a snapshot."""
    prepared, result, _ = scoring_chain
    checked = scorer.inspect_scoring_inputs(prepared, result)
    original = scorer._input_bytes
    original_mkdtemp = scorer.tempfile.mkdtemp
    created = []
    def mkdtemp(**kwargs):
        path = original_mkdtemp(dir=tmp_path, **kwargs)
        created.append(Path(path))
        return path
    monkeypatch.setattr(scorer.tempfile, 'mkdtemp', mkdtemp)
    def race(root, name, *args):
        data = original(root, name, *args)
        if created and name.endswith('analysis.py'):
            return data + b'changed'
        return data
    monkeypatch.setattr(scorer, '_input_bytes', race)
    with pytest.raises(ValueError, match='changed'), scorer.scoring_input_snapshot(checked):
        pytest.fail('snapshot must not be published')
    assert created and not created[0].exists()


def test_p22_existing_evaluations_not_immutable_input(scoring_chain):
    """Guards P2.2: adding separate evaluations cannot mutate original evidence."""
    prepared, result, _ = scoring_chain
    checked = scorer.inspect_scoring_inputs(prepared, result)
    extra = result.parent.parent/'evaluations/new'
    extra.mkdir(parents=True)
    (extra/'diagnostic.txt').write_text('new evaluation')
    with scorer.scoring_input_snapshot(checked):
        pass


def test_p22_input_limits(scoring_chain, monkeypatch):
    """Guards P2.2 bounded inventory independently of caller-provided budgets."""
    prepared, result, _ = scoring_chain
    monkeypatch.setattr(scorer, 'MAX_INPUT_FILES', 1)
    with pytest.raises(ValueError, match='count'):
        scorer.inspect_scoring_inputs(prepared, result)


@pytest.mark.parametrize('target', ['metadata', 'native', 'raw', 'prepared', 'prediction'])
def test_p22_change_after_inspection_rejected(scoring_chain, target):
    """Guards P2.2 against mutations between validation and snapshot creation."""
    prepared, result_path, raw = scoring_chain
    result = json.loads(result_path.read_text())
    checked = scorer.inspect_scoring_inputs(prepared, result_path)
    run = result_path.parent.parent
    paths = {
        'metadata': run / 'run_metadata.json',
        'native': run / result['native_result'],
        'raw': raw / 'sources.json',
        'prepared': prepared / 'task_manifest.json',
        'prediction': result_path.parent / result['agent_output']['persisted_outputs']['dir'] / 'analysis.py',
    }
    path = paths[target]
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='changed'), scorer.scoring_input_snapshot(checked):
        pytest.fail('changed input must not be published')


@pytest.mark.parametrize('limit', ['MAX_INPUT_FILE_BYTES', 'MAX_INPUT_TOTAL_BYTES'])
def test_p22_byte_limits(scoring_chain, monkeypatch, limit):
    """Guards P2.2 byte bounds before parsing source or result JSON."""
    prepared, result, _ = scoring_chain
    monkeypatch.setattr(scorer, limit, 1)
    with pytest.raises(ValueError, match=r'oversized|total bytes'):
        scorer.inspect_scoring_inputs(prepared, result)


def test_p22_snapshot_consumer_exception_cleanup(scoring_chain):
    """Guards P2.2 cleanup of read-only temporary copies after consumer failure."""
    prepared, result, _ = scoring_chain
    checked = scorer.inspect_scoring_inputs(prepared, result)
    with pytest.raises(RuntimeError, match='consumer failed'), scorer.scoring_input_snapshot(checked) as snapshot:
        path = snapshot['input_dir']
        raise RuntimeError('consumer failed')
    assert not path.parent.exists()


def test_p23a_no_test_module_dependency():
    """Guards P2.3a production build independence from smoke code and evidence."""
    import ast

    tree = ast.parse(Path(scorer.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert 'tests' not in (node.module or '').split('.')
        elif isinstance(node, ast.Import):
            assert all('tests' not in alias.name.split('.') for alias in node.names)


def test_scorer_review_staged_content_is_not_a_build_input(source, tmp_path, spec):
    """Guards the 2026-09-17 scorer review item 4 (pending commit)."""
    path = source / 'ai4sci_bench/core/types.py'
    original = path.read_bytes()
    path.write_text('unrelated staged change')
    subprocess.run(['git', '-C', str(source), 'add', str(path)], check=True)
    path.write_bytes(original)
    bundle = scorer.materialize_evaluator(source, tmp_path / 'cache', spec)
    assert (bundle / 'ai4sci_bench/core/types.py').read_bytes() == original
    assert subprocess.check_output(['git', '-C', str(source), 'show', ':ai4sci_bench/core/types.py']) != original


def test_scorer_review_input_checks_are_not_duplicated(scoring_chain, monkeypatch):
    """Guards review item 5 (pending commit): one source/metadata validation."""
    from benchmarks.asi_bench import prepare

    prepared, result, _ = scoring_chain
    calls, reads = [], []
    resolve, read = prepare.resolve_source_assets, scorer._input_json

    def resolve_once(manifest):
        calls.append(manifest)
        return resolve(manifest)

    def read_once(root, name):
        reads.append(name)
        return read(root, name)

    monkeypatch.setattr(prepare, 'resolve_source_assets', resolve_once)
    monkeypatch.setattr(scorer, '_input_json', read_once)
    scorer.inspect_scoring_inputs(prepared, result)
    assert len(calls) == 1
    assert reads.count('instance/math.mpsc_safety_filter__seed31415/instance_meta.json') == 1


@pytest.mark.parametrize('scoring_chain', [
    ('demo.generic', level, 'nested/solution.py', 'answer.json')
    for level in ('b1', 'b2', 'b3', 'b4')
], indirect=True)
def test_generic_scoring_entry(scoring_chain):
    """Guards generic-entry change: non-mpsc B1-B4 use declared nested output contracts."""
    prepared, result, _ = scoring_chain
    checked = scorer.inspect_scoring_inputs(prepared, result)
    assert checked['request_fields']['task_id'] == 'demo.generic'
    assert checked['request_fields']['prompt_level'] == json.loads(result.read_text())['prompt_level']
    with scorer.scoring_input_snapshot(checked) as snapshot:
        assert (snapshot['input_dir'] / 'prediction/nested/solution.py').is_file()
        assert (snapshot['input_dir'] / 'prediction/answer.json').is_file()


def test_direct_copy_from_exported_source(source, tmp_path, spec, monkeypatch):
    """Guards simplification of baseline 9b90d16c: no Git/cache publication."""
    import shutil

    shutil.rmtree(source / '.git')
    def forbidden(*args, **kwargs):
        raise AssertionError('copy must not invoke Git')
    monkeypatch.setattr(scorer.subprocess, 'run', forbidden)
    destination = tmp_path / 'evaluator'
    assert scorer.materialize_evaluator(source, destination, spec) == destination
    assert (destination / 'ai4sci_bench/core/types.py').read_bytes() == (
        source / 'ai4sci_bench/core/types.py').read_bytes()
    with pytest.raises(FileExistsError):
        scorer.materialize_evaluator(source, destination, spec)


def test_unrelated_logs_not_scanned(scoring_chain, monkeypatch):
    """Guards baseline 9b90d16c: logs outside scoring inputs are not evidence."""
    prepared, result, _ = scoring_chain
    run = result.parent.parent
    (run / 'benchflow' / 'unrelated.log').symlink_to('/dev/null')
    (result.parent / 'unrelated.log').symlink_to('/dev/null')
    checked = scorer.inspect_scoring_inputs(prepared, result)
    def forbidden(*args, **kwargs):
        raise AssertionError('snapshot must not rescan source trees')
    monkeypatch.setattr(scorer, '_input_tree', forbidden)
    with scorer.scoring_input_snapshot(checked) as snapshot:
        assert (snapshot['input_dir'] / 'prediction/analysis.py').is_file()


@pytest.mark.parametrize('image', [None, 'python:3.11-slim', 'sha256:bad'])
def test_shared_image_requires_solve_digest(scoring_chain, image):
    """Guards shared-image migration from baseline commit 9b90d16c: no tag fallback."""
    prepared, result, _ = scoring_chain
    value = json.loads(result.read_text())
    value['network']['image_id'] = image
    result.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='immutable solve image'):
        scorer.inspect_scoring_inputs(prepared, result)


def test_build_failure_preserves_original_exception(scoring_chain, monkeypatch, tmp_path):
    """Guards scoring rebuild after 9b90d16c: build failure cannot start scoring."""
    import asyncio

    prepared, result, _ = scoring_chain
    calls = []

    def absent(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, 'run', absent)
    output = tmp_path / 'score'
    with pytest.raises(subprocess.CalledProcessError):
        asyncio.run(scorer.score_result(prepared, result, output, source_root=tmp_path))
    assert len(calls) == 1
    assert calls[0][:3] == ['docker', 'build', '--iidfile']
    assert not output.exists()


@pytest.mark.parametrize('outcome', ['zero', 'timeout', 'failed', 'start-failed'])
def test_shared_image_lifecycle(scoring_chain, source, spec, monkeypatch, tmp_path, outcome):
    """Guards shared-image replacement of commit 9b90d16c: isolation, cleanup and real zero."""
    import asyncio

    from benchflow.sandbox.docker import DockerSandbox

    prepared, result, _ = scoring_chain
    checked = scorer.inspect_scoring_inputs(prepared, result)
    monkeypatch.setattr(scorer, 'load_evaluator_spec', lambda: spec)
    monkeypatch.setattr(scorer, 'inspect_scoring_image', lambda image: {'Id': image})
    events = []
    mounts = {}

    async def start(self, force_build):
        assert not force_build
        assert self.task_env_config.docker_image == 'sha256:' + 'd' * 64
        service = json.loads((self.environment_dir / 'docker-compose.yaml').read_text())['services']['main']
        assert service['image'] == self.task_env_config.docker_image
        assert service['pull_policy'] == 'never'
        assert service['read_only'] and service['network_mode'] == 'none'
        assert service['deploy']['resources']['limits']['pids'] == 128
        assert 'pids_limit' not in service  # Compose rejects conflicting representations.
        assert service['user'] == '10001:10001' and service['cap_drop'] == ['ALL']
        mounts.update({entry['target']: entry for entry in service['volumes']})
        assert set(mounts) == {'/input', '/output', '/opt/asi-evaluator', '/opt/bridge', '/prediction'}
        assert all(entry['read_only'] == (target != '/output') for target, entry in mounts.items())
        manifest = json.loads((Path(mounts['/input']['source']) / 'evaluation_manifest.json').read_text())
        assert manifest['runtime']['image_id'] == 'sha256:' + 'd' * 64
        assert 'runtime_manifest_sha256' not in manifest['runtime']
        workspace = Path(mounts['/prediction']['source'])
        assert (workspace / 'analysis.py').read_bytes() == (Path(mounts['/input']['source']) / 'prediction/analysis.py').read_bytes()
        assert not (self.environment_dir / 'Dockerfile').exists()
        events.append('start')
        if outcome == 'start-failed':
            raise RuntimeError('partial start')

    async def execute(self, command, **kwargs):
        assert kwargs['user'] == '10001:10001'
        events.append('exec')
        if outcome == 'timeout':
            raise TimeoutError('scorer timeout')
        output = Path(mounts['/output']['source'])
        failed = outcome == 'failed'
        (output / 'asibench_score.json').write_text(json.dumps({
            'evaluation_status': 'failed' if failed else 'completed',
            'score': None if failed else 0, 'reward': None if failed else 0.0}))
        # Even a failure that writes a reward must never publish it.
        (output / 'reward.txt').write_text('0.0\n')
        return SimpleNamespace(return_code=1 if failed else 0, stdout='', stderr='')

    async def stop(self, delete):
        events.append('stop')

    monkeypatch.setattr(DockerSandbox, 'start', start)
    monkeypatch.setattr(DockerSandbox, 'exec', execute)
    monkeypatch.setattr(DockerSandbox, 'stop', stop)
    output = tmp_path / 'score'
    with scorer.scoring_input_snapshot(checked) as snapshot:
        call = scorer._score_snapshot(snapshot, checked['image_id'], source, output, timeout=1)
        if outcome == 'zero':
            assert asyncio.run(call)['score'] == 0
            assert float((output / 'reward.txt').read_text()) == 0
        else:
            with pytest.raises((TimeoutError, RuntimeError)):
                asyncio.run(call)
            assert not (output / 'reward.txt').exists()
    assert events == (['start', 'stop'] if outcome == 'start-failed' else ['start', 'exec', 'stop'])
    assert not Path(mounts['/input']['source']).exists()
    assert not Path(mounts['/opt/asi-evaluator']['source']).exists()


@pytest.mark.skipif(not os.environ.get('ASI_SHARED_IMAGE'), reason='opt-in: existing shared Docker image')
@pytest.mark.parametrize('scoring_chain', [('demo.generic', 'b2', 'solution.py', 'answer.json')], indirect=True)
def test_docker_shared_image_clean_container(scoring_chain, tmp_path):
    """Guards shared-image replacement of commit 9b90d16c with real Docker, no model calls."""
    import asyncio
    import hashlib

    from benchmarks.asi_bench.prepare import PrepareConfig, prepare_task

    prepared, result_file, raw = scoring_chain
    image = os.environ['ASI_SHARED_IMAGE']
    task = 'demo.generic'
    bundle = raw / 'task_bundle/tasks/demo/generic'
    (bundle / 'task_eval.yaml').write_text('evaluation:\n  scoring:\n    - scorer: isolation_probe\n      weight: 1\n')
    (bundle / 'custom_scorer.py').write_text('''
import os
from pathlib import Path
from ai4sci_bench.core.scorer import Scorer, register_scorer
from ai4sci_bench.core.types import ScoreDetail
@register_scorer('isolation_probe')
class Probe(Scorer):
    def score(self, pred_dir, ref_dir, config):
        import numpy
        assert os.getuid() == os.getgid() == 10001
        assert not Path('/workspace/agent-layer-marker').exists()
        for target in [pred_dir / 'new', ref_dir / 'new', Path('/opt/bridge/new'),
                       Path('/opt/asi-evaluator/new'), Path('/workspace/new')]:
            try:
                target.write_text('must fail')
            except OSError:
                pass
            else:
                raise AssertionError('writable protected path: ' + str(target))
        return ScoreDetail(scorer_name=self.name, score=0, max_score=1, passed=True, details={})
''')
    manifest_path = raw / 'sources.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['files'] = [{'path': p.relative_to(raw).as_posix(),
                          'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                         for p in raw.rglob('*') if p.is_file() and p != manifest_path]
    manifest_path.write_text(json.dumps(manifest))
    prepared = prepare_task(PrepareConfig(raw, task, task + '__seed31415', 'b2',
                                         tmp_path / 'new-prepared')).output_dir
    run = result_file.parent.parent
    result = json.loads(result_file.read_text())
    result['prepared']['manifest_sha256'] = hashlib.sha256((prepared / 'task_manifest.json').read_bytes()).hexdigest()
    result['network']['image_id'] = image
    result_file.write_text(json.dumps(result))
    metadata_path = run / 'run_metadata.json'
    metadata = json.loads(metadata_path.read_text())
    metadata['prepared'] = result['prepared']
    metadata_path.write_text(json.dumps(metadata))
    # Simulate Agent-side dependency corruption and workspace changes. Nothing
    # from this writable layer may enter the fresh production scoring container.
    cid = subprocess.check_output(['docker', 'create', '--network=none', image, 'sleep', 'infinity'], text=True).strip()
    try:
        subprocess.run(['docker', 'start', cid], check=True, capture_output=True)
        script = "import pathlib,numpy; pathlib.Path(numpy.__file__).write_text('raise RuntimeError(\"agent corruption\")'); pathlib.Path('/workspace/agent-layer-marker').touch(); assert not pathlib.Path('/opt/asi-evaluator').exists(); assert not pathlib.Path('/opt/bridge').exists(); assert not pathlib.Path('/input').exists()"
        subprocess.run(['docker', 'exec', cid, 'python', '-c', script], check=True)
        assert subprocess.run(['docker', 'exec', cid, 'python', '-c', 'import numpy'], capture_output=True).returncode != 0
        inspected = json.loads(subprocess.check_output(['docker', 'inspect', cid]))[0]
        assert inspected['Image'] == image
        subprocess.run(['docker', 'stop', cid], check=True, capture_output=True)
        output = tmp_path / 'evaluation'
        report = asyncio.run(scorer.score_result(prepared, result_file, output, timeout=60))
        assert report['evaluation_status'] == 'completed' and report['score'] == 0
        assert report['provenance']['runtime']['image_id'].startswith('sha256:')
        assert float((output / 'reward.txt').read_text()) == 0
    finally:
        subprocess.run(['docker', 'rm', '-f', cid], check=True, capture_output=True)


def test_input_tree_filters_only_regular_finder_metadata(tmp_path):
    """Guards scoring snapshot inputs following BenchFlow 9b90d16c."""
    (tmp_path / 'analysis.py').write_text('answer')
    (tmp_path / '.DS_Store').write_text('Finder')
    assert set(scorer._input_tree(tmp_path)) == {'analysis.py'}
    (tmp_path / '.DS_Store').unlink()
    (tmp_path / '.DS_Store').symlink_to('analysis.py')
    with pytest.raises((ValueError, OSError)):
        scorer._input_tree(tmp_path)


def test_finder_noise_not_copied_to_scoring_snapshot(scoring_chain):
    """Guards snapshot filtering following BenchFlow 9b90d16c."""
    prepared, result, raw = scoring_chain
    prediction = result.with_suffix('.outputs')
    for root in (prepared, raw, prediction, prepared / 'verifier'):
        (root / '.DS_Store').write_text('Finder')
    checked = scorer.inspect_scoring_inputs(prepared, result)
    with scorer.scoring_input_snapshot(checked) as snapshot:
        assert not list(snapshot['input_dir'].rglob('.DS_Store'))
        assert scorer._input_tree(snapshot['input_dir']) == snapshot['request_fields']['files']


def test_input_tree_keeps_declared_finder_file(tmp_path):
    """Guards declared asset retention following BenchFlow 9b90d16c."""
    (tmp_path / '.DS_Store').write_text('declared')
    assert '.DS_Store' in scorer._input_tree(tmp_path, {'.DS_Store'})


@pytest.mark.parametrize('value', ['absent', None, {}, {'status': 'complete'}])
def test_native_optional_scoring_contract(scoring_chain, value):
    """Guards real produce-only serialization on BenchFlow 9b90d16c."""
    prepared, result_path, _ = scoring_chain
    result = json.loads(result_path.read_text())
    native_path = result_path.parent.parent / result['native_result']
    native = json.loads(native_path.read_text())
    if value == 'absent':
        native.pop('scoring')
    else:
        native['scoring'] = value
    native_path.write_text(json.dumps(native))
    if value == 'absent' or value is None:
        scorer.inspect_scoring_inputs(prepared, result_path)
    else:
        with pytest.raises(ValueError, match='scoring'):
            scorer.inspect_scoring_inputs(prepared, result_path)


def test_rebuild_uses_verified_context(scoring_chain, monkeypatch):
    """Guards rebuild from prepared, rather than solve image reuse, after 9b90d16c."""
    prepared, result, _ = scoring_chain
    checked = scorer.inspect_scoring_inputs(prepared, result)
    image = 'sha256:' + 'e' * 64
    calls = []
    def build(command, **kwargs):
        calls.append(command)
        context = Path(command[-1])
        assert (context / 'Dockerfile').read_bytes() == (prepared / 'environment/Dockerfile').read_bytes()
        assert {p.relative_to(context).as_posix() for p in context.rglob('*') if p.is_file()} == {
            name.removeprefix('environment/') for name in checked['environment_files']}
        Path(command[3]).write_text(image)
    monkeypatch.setattr(subprocess, 'run', build)
    monkeypatch.setattr(scorer, 'inspect_scoring_image', lambda value: calls.append(value))
    assert scorer.build_scoring_image(checked) == image
    assert calls[-1] == image
    assert checked['host_provenance']['solve_image_id'] != image


def test_rebuild_rejects_changed_context(scoring_chain, monkeypatch):
    """Guards the verified build copy after baseline 9b90d16c."""
    prepared, result, _ = scoring_chain
    checked = scorer.inspect_scoring_inputs(prepared, result)
    (prepared / 'environment/Dockerfile').write_text('FROM changed')
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: pytest.fail('must not build'))
    with pytest.raises(ValueError, match='environment changed'):
        scorer.build_scoring_image(checked)


async def test_score_uses_rebuilt_image(scoring_chain, monkeypatch, tmp_path):
    """Guards independence from deleted solve images after baseline 9b90d16c."""
    from unittest.mock import AsyncMock
    prepared, result, _ = scoring_chain
    image = 'sha256:' + 'e' * 64
    monkeypatch.setattr(scorer, 'build_scoring_image', lambda *a, **kw: image)
    score = AsyncMock(return_value={'evaluation_status': 'completed'})
    monkeypatch.setattr(scorer, '_score_snapshot', score)
    await scorer.score_result(prepared, result, tmp_path / 'scored', source_root=tmp_path)
    assert score.call_args.args[1] == image


def test_scoring_workspace_combines_outputs_and_public_data(tmp_path):
    """Guards Levin missing training data after baseline 9b90d16c."""
    inputs = tmp_path / 'input'
    for name, text in [('prediction/analysis.py', 'agent answer'),
                       ('instance/data/training_levels.json', 'public training'),
                       ('instance/reference/hidden.json', 'hidden answer')]:
        path = inputs / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    workspace = tmp_path / 'workspace'
    scorer.materialize_scoring_workspace(inputs, workspace)
    assert (workspace / 'analysis.py').read_text() == 'agent answer'
    assert (workspace / 'data/training_levels.json').read_text() == 'public training'
    assert {p.relative_to(workspace).as_posix() for p in workspace.rglob('*') if p.is_file()} == {
        'analysis.py', 'data/training_levels.json'}
    assert not (inputs / 'prediction/data').exists()


def test_scoring_workspace_without_data(tmp_path):
    """Guards tasks without public data after baseline 9b90d16c."""
    inputs = tmp_path / 'input'
    (inputs / 'prediction').mkdir(parents=True)
    (inputs / 'prediction/answer.json').write_text('{}')
    scorer.materialize_scoring_workspace(inputs, tmp_path / 'workspace')
    assert (tmp_path / 'workspace/answer.json').read_text() == '{}'


def test_scoring_workspace_rejects_output_data_collision(tmp_path):
    """Guards trusted public data against output replacement after 9b90d16c."""
    inputs = tmp_path / 'input'
    (inputs / 'prediction/data').mkdir(parents=True)
    (inputs / 'instance/data').mkdir(parents=True)
    with pytest.raises(FileExistsError):
        scorer.materialize_scoring_workspace(inputs, tmp_path / 'workspace')


@pytest.mark.parametrize('unknown', [False, True])
def test_pre_workspace_prepared_is_exactly_bound(scoring_chain, unknown):
    """Guards existing solve reuse for the workspace fix after baseline 9b90d16c."""
    import hashlib
    prepared, result, _ = scoring_chain
    bridge = prepared / 'verifier/score_entry.py'
    old_bytes = bridge.read_bytes().replace(b"evaluation, Path('/prediction'),", b"evaluation, input_root / 'prediction',")
    bridge.write_bytes(old_bytes)
    spec = scorer.load_evaluator_spec()
    spec['bridge'] = {**spec['bridge'], 'adaptation_version': 6, 'sha256': hashlib.sha256(old_bytes).hexdigest()}
    legacy = scorer._manifest(spec)
    manifest_path = prepared / 'task_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['scoring'].update(evaluator_spec_sha256=legacy['spec_sha256'],
                               expected_evaluator_bundle_digest=legacy['bundle_digest'])
    if unknown:
        manifest['scoring']['evaluator_spec_sha256'] = 'f' * 64
    for entry in manifest['files']:
        if entry['target'] == 'verifier/score_entry.py':
            entry.update(size=len(old_bytes), sha256=hashlib.sha256(old_bytes).hexdigest())
    manifest_path.write_text(json.dumps(manifest))
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    for path in (result, result.parent.parent / 'run_metadata.json'):
        value = json.loads(path.read_text())
        value['prepared']['manifest_sha256'] = digest
        path.write_text(json.dumps(value))
    if unknown:
        with pytest.raises(ValueError, match='unsupported prepared scoring identity'):
            scorer.inspect_scoring_inputs(prepared, result)
    else:
        checked = scorer.inspect_scoring_inputs(prepared, result)
        assert checked['scoring'] == scorer.scoring_profile_identity()
        assert checked['scoring'] != manifest['scoring']


@pytest.fixture
def downloads(source, tmp_path, monkeypatch):
    import io
    calls = []
    def fetch(url, timeout):
        assert timeout == 30
        prefix = f"https://raw.githubusercontent.com/apexin-ai/ASI-Bench/{scorer.load_evaluator_spec()['upstream']['revision']}/"
        assert url.startswith(prefix)
        relative = url.removeprefix(prefix)
        calls.append(relative)
        return io.BytesIO((source / relative).read_bytes())
    monkeypatch.setattr(scorer, 'urlopen', fetch)
    monkeypatch.setattr(scorer, 'EVALUATOR_CACHE', tmp_path / 'cache')
    return calls


def test_download_and_offline_cache(downloads, tmp_path, monkeypatch, spec):
    """Guards GitHub delivery replacing vendoring after baseline 9b90d16c."""
    bundle = scorer.materialize_evaluator(None, tmp_path / 'bundle')
    assert set(downloads) == {e['source'] for e in spec['files'] if e['kind'] == 'copy'}
    assert len(downloads) == 7
    for entry in spec['files']:
        assert scorer._sha((bundle / entry['path']).read_bytes()) == entry['sha256']
    def no_network(*args, **kwargs):
        pytest.fail('valid cache must work offline')
    monkeypatch.setattr(scorer, 'urlopen', no_network)
    scorer.materialize_evaluator(None, tmp_path / 'second')
    assert len(list(scorer.EVALUATOR_CACHE.rglob('*.py'))) == 7
    assert not (bundle / 'ai4sci_bench/generators').exists()


def test_corrupt_cache_redownload(downloads, spec):
    """Guards cache integrity after baseline 9b90d16c."""
    cache = scorer.download_evaluator_sources(spec)
    (cache / 'ai4sci_bench/core/types.py').write_text('corrupt')
    scorer.download_evaluator_sources(spec)
    assert downloads[7:] == ['ai4sci_bench/core/types.py']


@pytest.mark.parametrize('failure', ['digest', 'interrupted', 'oversize'])
def test_download_failure_leaves_no_bundle(downloads, tmp_path, monkeypatch, failure):
    """Guards failed downloads after baseline 9b90d16c."""
    import io
    def fail(*args, **kwargs):
        if failure == 'interrupted':
            raise OSError('network interrupted')
        return io.BytesIO(b'x' * (8 * 1024 * 1024 + 1) if failure == 'oversize' else b'wrong')
    monkeypatch.setattr(scorer, 'urlopen', fail)
    with pytest.raises((ValueError, OSError)):
        scorer.materialize_evaluator(None, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()
    assert not list(scorer.EVALUATOR_CACHE.rglob('*'))


def test_default_score_downloads_before_build(scoring_chain, downloads, tmp_path, monkeypatch):
    """Guards host-side prefetch ordering after baseline 9b90d16c."""
    import asyncio
    from unittest.mock import AsyncMock
    prepared, result, _ = scoring_chain
    def build(*args, **kwargs):
        assert len(downloads) == 7
        return 'image'
    monkeypatch.setattr(scorer, 'build_scoring_image', build)
    score = AsyncMock(return_value={'status': 'completed'})
    monkeypatch.setattr(scorer, '_score_snapshot', score)
    asyncio.run(scorer.score_result(prepared, result, tmp_path / 'score'))
    assert score.call_args.args[2] == scorer.EVALUATOR_CACHE / scorer.load_evaluator_spec()['upstream']['revision']
