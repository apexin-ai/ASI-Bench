"""Contracts for the prepare-v1 implementation (following d154f6aa acquisition)."""
import hashlib
import json
from dataclasses import replace

import pytest
import yaml

from benchflow.task import TaskDocument
from benchmarks.asi_bench.prepare import PrepareConfig, prepare_task, validate_runtime


def snapshot(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


def refresh(raw):
    manifest = json.loads((raw / 'sources.json').read_text())
    manifest['files'] = [dict(path=path, size=(raw/path).stat().st_size, sha256=digest)
                         for path, digest in snapshot(raw).items() if path != 'sources.json']
    (raw/'sources.json').write_text(json.dumps(manifest))


@pytest.fixture
def config(tmp_path):
    raw = tmp_path/'raw'
    instance = raw/'instance/math.demo__seed31415'
    bundle = raw/'task_bundle/tasks/math/demo'
    for path, content in {
        'data/input.json': '{"input": 1}',
        'reference/answer.json': '{"hidden": 42}',
        'reference/nested/extra.txt': 'keep all reference',
        'instance_meta.json': '{"task_id":"math.demo","params_used":{"seed":123}}',
        **{f'prompt_b{i}.md': f'# Task B{i}\n\nSolve it.\n' for i in range(1, 5)},
    }.items():
        target = instance/path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    bundle.mkdir(parents=True)
    (bundle/'task_meta.yaml').write_text(yaml.safe_dump({
        'id': 'math.demo', 'version': '1', 'runtime': {'python': '>=3.11'},
        'input': {'files': [{'name': 'input.json'}]},
        'output': {'files': [{'name': 'result.json', 'type': 'data', 'shape': [3]}]},
    }))
    (bundle/'task_eval.yaml').write_text('task_id: math.demo\n')
    (bundle/'custom_scorer.py').write_text('raise RuntimeError("must not import")\n')
    (raw/'sources.json').write_text(json.dumps({
        'schema_version': 1, 'status': 'downloaded', 'seed': 31415,
        'task_id': 'math.demo', 'instance_id': 'math.demo__seed31415',
        'instance_dir': 'instance/math.demo__seed31415',
        'task_bundle_dir': 'task_bundle/tasks/math/demo',
        'hf': {'repo': 'Apexintelligence-AI/ASI-Bench-seed31415',
               'resolved_revision': 'a'*40, 'path': 'tasks/math.demo__seed31415'},
        'github': {'repo': 'apexin-ai/ASI-Bench',
                   'resolved_revision': 'b'*40, 'path': 'tasks/math/demo'},
    }))
    refresh(raw)
    return PrepareConfig(raw, 'math.demo', 'math.demo__seed31415', 'b1', tmp_path/'out')


def test_layout_parity_and_parser(config):
    before = snapshot(config.raw_dir)
    result = prepare_task(config)
    root = result.output_dir
    prompt = (config.raw_dir/'instance'/config.instance_id/'prompt_b1.md').read_bytes()
    assert (root/'environment/inputs/prompt.md').read_bytes() == prompt
    document = TaskDocument.from_path(root/'task.md')
    assert document.instruction == prompt.decode().strip()
    assert document.config.agent.timeout_sec == 3600
    info = json.loads((root/'environment/inputs/task_info.json').read_text())
    assert info['expected_outputs'] == [{'name': 'result.json', 'type': 'data'}]
    assert info['task_id'].startswith('task_')
    assert json.loads((root/'verifier/instance_parameters.json').read_text()) == {'seed': 123}
    assert not (root/'verifier/reference').exists()
    assert not (root/'verifier/task_bundle').exists()
    assert (root/'verifier/test.sh').is_file()
    assert set(snapshot(root/'environment/inputs')) == {'prompt.md', 'task_info.json', 'data/input.json'}
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest['sources']['hf']['resolved_revision'] == 'a'*40
    for entry in manifest['files']:
        assert snapshot(root)[entry['target']] == entry['sha256']
    assert snapshot(config.raw_dir) == before
    before_prepared = snapshot(root)
    assert prepare_task(config).output_dir == root
    assert snapshot(root) == before_prepared


@pytest.mark.parametrize('level', ['b1', 'b2', 'b3', 'b4'])
def test_levels(config, level):
    result = prepare_task(replace(config, level=level))
    assert f'Task B{level[1]}' in TaskDocument.from_path(result.output_dir/'task.md').instruction


@pytest.mark.parametrize('mutation', ['seed', 'hash', 'missing_reference'])
def test_reject_invalid_sources(config, mutation):
    raw = config.raw_dir
    if mutation == 'seed':
        m = json.loads((raw/'sources.json').read_text()); m['seed'] = 42
        (raw/'sources.json').write_text(json.dumps(m))
    elif mutation == 'hash':
        (raw/'instance'/config.instance_id/'data/input.json').write_text('tampered')
    else:
        for p in (raw/'instance'/config.instance_id/'reference').rglob('*'):
            if p.is_file(): p.unlink()
        refresh(raw)
    with pytest.raises(ValueError): prepare_task(config)
    assert not list(config.output_dir.rglob('task_manifest.json'))


@pytest.mark.parametrize('name', ['../escape', '/abs', 'data/../escape', 'data//input.json'])
def test_bad_input_path(config, name):
    p = config.raw_dir/'task_bundle/tasks/math/demo/task_meta.yaml'
    m = yaml.safe_load(p.read_text()); m['input']['files'][0]['name'] = name
    p.write_text(yaml.safe_dump(m)); refresh(config.raw_dir)
    with pytest.raises(ValueError): prepare_task(config)


def test_cache_identity(config):
    first = prepare_task(config)
    other = prepare_task(replace(config, timeout_seconds=100))
    assert first.output_dir != other.output_dir
    source = config.raw_dir/'sources.json'
    m = json.loads(source.read_text()); m['hf']['resolved_revision'] = 'c'*40
    source.write_text(json.dumps(m))
    assert prepare_task(config).output_dir != first.output_dir


def test_reserved_headings(config):
    prompt = config.raw_dir/'instance'/config.instance_id/'prompt_b1.md'
    prompt.write_text('Intro\n\n## user-persona\n\nThis is task text.\n')
    refresh(config.raw_dir)
    result = prepare_task(config)
    assert TaskDocument.from_path(result.output_dir/'task.md').instruction == prompt.read_text().strip()


@pytest.mark.parametrize('runtime', [{'python': '>=3.10'}, {'gpu': True}])
def test_incompatible_runtime(runtime):
    with pytest.raises(ValueError): validate_runtime({'runtime': runtime}, 'cpu-v1')


def test_runtime_packages_installed_at_build_time(config):
    """Guards S8 missing-certificate fix: prepare must honor task dependencies."""
    from benchmarks.asi_bench.prepare import build_dockerfile
    packages = ['numpy>=1.26', 'scipy>=1.11', 'cvxpy>=1.4', 'clarabel>=0.7', 'scs>=3.2']
    path = config.raw_dir/'task_bundle/tasks/math/demo/task_meta.yaml'
    meta = yaml.safe_load(path.read_text())
    meta['runtime']['packages'] = packages
    path.write_text(yaml.safe_dump(meta))
    refresh(config.raw_dir)
    result = prepare_task(config)
    dockerfile = (result.output_dir/'environment/Dockerfile').read_text()
    assert dockerfile == build_dockerfile(meta)
    assert 'RUN ["python", "-m", "pip", "install", "--no-cache-dir"' in dockerfile
    for package in packages:
        assert f'"{package}"' in dockerfile
    assert dockerfile.index('RUN [') < dockerfile.index('COPY inputs/')


@pytest.mark.parametrize('packages', ['numpy', [None], ['--index-url=evil'], ['numpy; touch /tmp/x'], ['numpy\\nRUN echo bad'], ['x @ https://example.com/x.whl']])
def test_runtime_packages_reject_unsafe_specs(packages):
    """Guards S8 build-time dependency fix against command/URL injection."""
    from benchmarks.asi_bench.prepare import build_dockerfile
    with pytest.raises(ValueError, match=r'runtime\.packages'):
        build_dockerfile({'runtime': {'packages': packages}})


def test_external_assets_binding(config):
    """Guards prepare-v3 removal of duplicate raw assets: references stay bound."""
    from benchmarks.asi_bench.prepare import resolve_source_assets
    result = prepare_task(config)
    manifest = json.loads(result.manifest_path.read_text())
    assert resolve_source_assets(manifest) == config.raw_dir.resolve()
    assert manifest['source_assets']['manifest_sha256'] == hashlib.sha256((config.raw_dir/'sources.json').read_bytes()).hexdigest()
    (config.raw_dir/'instance'/config.instance_id/'reference/answer.json').write_text('{}')
    with pytest.raises(ValueError, match='integrity'):
        resolve_source_assets(manifest)


@pytest.mark.parametrize('mutation', ['missing', 'manifest', 'extra', 'symlink', 'traversal'])
def test_external_assets_fail_closed(config, mutation):
    """Guards prepare-v3 external references against stale or unsafe sources."""
    from benchmarks.asi_bench.prepare import resolve_source_assets
    result = prepare_task(config)
    manifest = json.loads(result.manifest_path.read_text())
    source = config.raw_dir/'sources.json'
    if mutation == 'missing':
        source.unlink()
    elif mutation == 'manifest':
        source.write_text('{}')
    elif mutation == 'extra':
        (config.raw_dir/'untracked.py').write_text('raise RuntimeError')
    elif mutation == 'symlink':
        ref = config.raw_dir/'instance'/config.instance_id/'reference'
        ref.rename(ref.with_name('reference-original'))
        ref.symlink_to(ref.with_name('reference-original'), target_is_directory=True)
    else:
        data = json.loads(source.read_text())
        data['files'][0]['path'] = '../outside'
        source.write_text(json.dumps(data))
        manifest['source_assets']['manifest_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        resolve_source_assets(manifest)


@pytest.mark.parametrize('old_version', ['prepare-v3', 'prepare-v4'])
def test_v5_template_identity_and_version(config, old_version):
    """Guards shared-image migration from commit 9b90d16c: reject old prepared labels."""
    from benchmarks.asi_bench.scorer import ADAPTER_ROOT
    with pytest.raises(ValueError, match='prepare-v5'):
        prepare_task(replace(config, converter_version=old_version))
    result = prepare_task(config)
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest['converter_version'] == 'prepare-v5'
    assert manifest['scoring']['profile_id'] == 'asi-seed31415-v1'  # generic entry, including demo tasks
    for name in ('test.sh', 'score_entry.py', 'scoring.py'):
        assert (result.output_dir / 'verifier' / name).read_bytes() == (ADAPTER_ROOT / 'templates/verifier' / name).read_bytes()
        entry = next(e for e in manifest['files'] if e['target'] == 'verifier/' + name)
        assert entry['visibility'] == ['scoring']


def test_v4_reuse_checks_integrity(config):
    """Guards P2.1: a cache hit never hides altered prepared files."""
    result = prepare_task(config)
    (result.output_dir / 'verifier/test.sh').write_text('tampered')
    with pytest.raises(ValueError, match='integrity'):
        prepare_task(config)
    assert (result.output_dir / 'verifier/test.sh').read_text() == 'tampered'


def test_v4_scoring_identity_changes_prepared_id(config, monkeypatch):
    """Guards P2.1: reviewed evaluator changes cannot reuse stale prepared IDs."""
    from benchmarks.asi_bench import scorer
    first = prepare_task(config)
    original = scorer.scoring_profile_identity
    monkeypatch.setattr(scorer, 'scoring_profile_identity', lambda spec: {
        **original(spec), 'expected_evaluator_bundle_digest': 'f' * 64})
    second = prepare_task(config)
    assert first.output_dir != second.output_dir
    assert snapshot(first.output_dir / 'environment') == snapshot(second.output_dir / 'environment')


def test_v4_invalid_template_rejected_before_publish(config, monkeypatch):
    """Guards P2.1: no package is published from an unreviewed bridge."""
    from benchmarks.asi_bench import scorer
    def reject():
        raise ValueError('bridge digest mismatch')
    monkeypatch.setattr(scorer, 'load_evaluator_spec', reject)
    with pytest.raises(ValueError, match='bridge digest'):
        prepare_task(config)
    assert not config.output_dir.exists()


def test_shared_runtime_has_one_public_dependency_path(config):
    """Guards shared-image migration from commit 9b90d16c against private build assets."""
    from benchmarks.asi_bench.prepare import build_dockerfile
    from benchmarks.asi_bench.scorer import ADAPTER_ROOT

    result = prepare_task(config)
    environment = result.output_dir / 'environment'
    assert {p.name for p in environment.iterdir()} == {
        'Dockerfile', 'runtime-profile.json', 'runtime-constraints.txt', 'inputs'}
    assert (environment / 'runtime-constraints.txt').read_bytes() == (ADAPTER_ROOT / 'runtime-constraints.txt').read_bytes()
    dockerfile = build_dockerfile({'runtime': {'packages': ['numpy', 'numpy>=1.26', 'scipy>=1.11']}})
    command = json.loads(next(line[4:] for line in dockerfile.splitlines() if line.startswith('RUN ')))
    assert command.count('numpy') == 1
    assert {'numpy', 'PyYAML', 'packaging', 'numpy>=1.26', 'scipy>=1.11'} <= set(command)
    assert command.count('install') == 1 and '-c' in command
    assert 'asibench' not in dockerfile and 'verifier' not in dockerfile and 'reference' not in dockerfile


def test_finder_metadata_and_reuse(config):
    """Guards the Finder inventory fix following d154f6aa acquisition."""
    for relative in ('.DS_Store', 'instance/.DS_Store'):
        (config.raw_dir / relative).write_bytes(b'Finder metadata')
    result = prepare_task(config)
    for relative in ('.DS_Store', 'environment/inputs/.DS_Store', 'verifier/.DS_Store'):
        (result.output_dir / relative).write_bytes(b'Finder metadata')
    assert prepare_task(config).output_dir == result.output_dir


def test_inventory_error_lists_paths(config):
    """Guards diagnostic detail following d154f6aa acquisition."""
    from benchmarks.asi_bench.prepare import validate_source_files
    manifest = json.loads((config.raw_dir / 'sources.json').read_text())
    missing = 'instance/math.demo__seed31415/data/input.json'
    (config.raw_dir / missing).unlink()
    (config.raw_dir / 'debug.py').write_text('extra')
    with pytest.raises(ValueError) as error:
        validate_source_files(config.raw_dir, manifest)
    assert missing in str(error.value)
    assert 'debug.py' in str(error.value)


def test_declared_finder_file_is_still_verified(config):
    """Guards declared asset integrity following d154f6aa acquisition."""
    from benchmarks.asi_bench.prepare import validate_source_files
    path = config.raw_dir / '.DS_Store'
    path.write_text('declared')
    refresh(config.raw_dir)
    manifest = json.loads((config.raw_dir / 'sources.json').read_text())
    path.write_text('modified')
    with pytest.raises(ValueError, match='integrity mismatch'):
        validate_source_files(config.raw_dir, manifest)


def test_finder_symlink_not_ignored(config):
    """Guards symlink rejection following d154f6aa acquisition."""
    (config.raw_dir / '.DS_Store').symlink_to('sources.json')
    with pytest.raises(ValueError, match='unsafe'):
        prepare_task(config)
