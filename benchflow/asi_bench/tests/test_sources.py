"""Offline tests for the ASI raw asset acquisition contract."""
from unittest.mock import patch

import pytest

from benchmarks.asi_bench.sources import acquire, validate_task_id


@pytest.mark.parametrize('task_id', ['../escape', 'a.b/c', 'a..b', 'a', '/a.b'])
def test_reject_invalid_task(task_id):
    with pytest.raises(ValueError):
        validate_task_id(task_id)


def fake_hf(task_id, destination, revision, cache_dir):
    destination.mkdir(parents=True)
    (destination / 'prompt_b1.md').write_text('Solve this task.')
    (destination / 'reference').mkdir()
    (destination / 'reference' / 'answer.bin').write_bytes(b'raw-reference')
    return 'a' * 40


def fake_github(task_id, destination, revision):
    destination.mkdir(parents=True)
    (destination / 'task_meta.yaml').write_text(f'id: {task_id}\n')
    (destination / 'task_eval.yaml').write_text(f'task_id: {task_id}\n')
    (destination / 'helpers').mkdir()
    (destination / 'helpers' / 'extra.py').write_text('# retained, not executed\n')
    return 'b' * 40


def test_acquire_raw_layout_and_provenance(tmp_path):
    out = tmp_path / 'assets'
    with patch('benchmarks.asi_bench.sources.download_instance', fake_hf), patch(
        'benchmarks.asi_bench.sources.download_task_bundle', fake_github
    ):
        manifest = acquire('astronomy.example', out)
    assert (out / 'instance/astronomy.example__seed31415/reference/answer.bin').read_bytes() == b'raw-reference'
    assert (out / 'task_bundle/tasks/astronomy/example/helpers/extra.py').is_file()
    assert manifest['status'] == 'downloaded'
    assert manifest['hf']['resolved_revision'] == 'a' * 40
    assert manifest['github']['resolved_revision'] == 'b' * 40
    assert len(manifest['files']) == 5
    assert all(len(f['sha256']) == 64 for f in manifest['files'])
    assert not (out / 'task.md').exists()
    assert not (out / 'task_manifest.json').exists()
    assert (out / 'sources.json').is_file()


def test_failure_does_not_publish_partial_output(tmp_path):
    out = tmp_path / 'assets'
    with patch('benchmarks.asi_bench.sources.download_instance', fake_hf), patch(
        'benchmarks.asi_bench.sources.download_task_bundle', side_effect=RuntimeError('offline')
    ), pytest.raises(RuntimeError):
        acquire('astronomy.example', out)
    assert list(tmp_path.iterdir()) == []


def test_reject_existing_output_before_network(tmp_path):
    with patch('benchmarks.asi_bench.sources.download_instance') as download:
        with pytest.raises(FileExistsError):
            acquire('astronomy.example', tmp_path)
        download.assert_not_called()


def test_identity_mismatch(tmp_path):
    with patch('benchmarks.asi_bench.sources.download_instance', fake_hf), patch(
        'benchmarks.asi_bench.sources.download_task_bundle',
        lambda task, dest, rev: fake_github('astronomy.wrong', dest, rev),
    ), pytest.raises(ValueError, match='identity'):
        acquire('astronomy.example', tmp_path / 'assets')
    assert list(tmp_path.iterdir()) == []


def test_github_downloads_whole_subtree_without_allowlist(tmp_path, monkeypatch):
    import base64
    import hashlib
    import httpx
    from benchmarks.asi_bench.sources import download_task_bundle

    data = b'# helper bytes\n'
    blob_sha = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
    routes = {
        'commits/main': {'sha': 'c' * 40, 'commit': {'tree': {'sha': 'root'}}},
        'git/trees/root': {'tree': [{'path': 'tasks', 'type': 'tree', 'sha': 'tasks'}]},
        'git/trees/tasks': {'tree': [{'path': 'astronomy', 'type': 'tree', 'sha': 'domain'}]},
        'git/trees/domain': {'tree': [{'path': 'example', 'type': 'tree', 'sha': 'task'}]},
        'git/trees/task': {'tree': [{'path': 'helpers', 'type': 'tree', 'sha': 'helpers'}]},
        'git/trees/helpers': {'tree': [{'path': 'extra.py', 'type': 'blob', 'mode': '100644',
                                      'sha': blob_sha, 'size': len(data)}]},
        f'git/blobs/{blob_sha}': {'encoding': 'base64', 'content': base64.b64encode(data).decode()},
    }
    calls = []
    monkeypatch.setattr("benchmarks.asi_bench.sources.time.sleep", lambda _: None)

    def handle(request):
        key = request.url.path.split('/ASI-Bench/')[1]
        calls.append(key)
        if len(calls) == 1:
            return httpx.Response(504)  # transient read failure must be retried
        assert key in routes, key  # no allowlist or whole-repository download
        return httpx.Response(200, json=routes[key])

    client = httpx.Client
    monkeypatch.setattr(httpx, 'Client', lambda **kwargs: client(
        **kwargs, transport=httpx.MockTransport(handle)))
    assert download_task_bundle('astronomy.example', tmp_path, 'main') == 'c' * 40
    assert (tmp_path / 'helpers/extra.py').read_bytes() == data
    assert len(calls) == 8
    routes['git/trees/helpers']['tree'][0]['mode'] = '120000'
    with pytest.raises(ValueError, match='unsupported Git entry'):
        download_task_bundle('astronomy.example', tmp_path / 'symlink', 'main')
    routes['git/trees/task']['truncated'] = True
    with pytest.raises(ValueError, match='incomplete tree'):
        download_task_bundle('astronomy.example', tmp_path / 'truncated', 'main')


def test_hf_exact_instance_at_resolved_revision(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from benchmarks.asi_bench import sources

    prefix = 'tasks/astronomy.example__seed31415'
    sha = 'd' * 40
    payloads = {f'prompt_b{level}.md': b'prompt' for level in range(1, 5)}
    payloads['reference/unlisted.bin'] = b'answer'
    calls = []

    class FakeApi:
        def __init__(self, **kwargs):
            pass

        def dataset_info(self, repo, revision):
            assert repo == sources.HF_REPO
            assert revision == 'main'
            return SimpleNamespace(sha=sha)

        def list_repo_tree(self, repo, **kwargs):
            assert kwargs == dict(path_in_repo=prefix, recursive=True,
                                  repo_type='dataset', revision=sha)
            return [sources.RepoFile(path=f'{prefix}/{p}', size=len(data), oid='e' * 40)
                    for p, data in payloads.items()]

    def download(repo, path, **kwargs):
        assert kwargs['revision'] == sha
        calls.append(path)
        cache = tmp_path / 'cached-file'
        cache.write_bytes(payloads[path.removeprefix(prefix + '/')])
        return str(cache)

    monkeypatch.setattr(sources, 'HfApi', FakeApi)
    monkeypatch.setattr(sources, 'hf_hub_download', download)
    destination = tmp_path / 'instance'
    assert sources.download_instance('astronomy.example', destination, 'main', None) == sha
    assert (destination / 'reference/unlisted.bin').read_bytes() == b'answer'
    assert len(calls) == 5
    payloads.pop('reference/unlisted.bin')
    with pytest.raises(ValueError, match='populated reference'):
        sources.download_instance('astronomy.example', tmp_path / 'missing-ref', 'main', None)


@pytest.mark.parametrize('path', ['../x', '/x', 'a//b', 'a/./b', 'a\\b'])
def test_unsafe_source_paths(path):
    from benchmarks.asi_bench.sources import safe_path
    with pytest.raises(ValueError):
        safe_path(path)


def test_limits_and_case_collisions():
    from benchmarks.asi_bench.sources import Budget, MAX_FILE_BYTES
    budget = Budget()
    budget.add('input.csv', 1)
    with pytest.raises(ValueError, match='colliding'):
        budget.add('INPUT.csv', 1)
    with pytest.raises(ValueError, match='size'):
        budget.add('huge.bin', MAX_FILE_BYTES + 1)


@pytest.mark.parametrize('cached_kind', ['missing', 'directory', 'wrong_size'])
def test_hf_download_validation_reports_specific_cause(tmp_path, monkeypatch, cached_kind):
    """Guards ambiguous HF file errors introduced in commit c2f7220d."""
    from types import SimpleNamespace
    from benchmarks.asi_bench import sources

    entry = sources.RepoFile(
        path='tasks/astronomy.example__seed31415/prompt_b1.md',
        size=1024, oid='e' * 40,
    )
    api = SimpleNamespace(
        dataset_info=lambda *args, **kwargs: SimpleNamespace(sha='d' * 40),
        list_repo_tree=lambda *args, **kwargs: [entry],
    )
    cached = tmp_path / 'cached-file'
    if cached_kind == 'directory':
        cached.mkdir()
    elif cached_kind == 'wrong_size':
        cached.write_bytes(b'x' * 800)
    monkeypatch.setattr(sources, 'HfApi', lambda **kwargs: api)
    monkeypatch.setattr(sources, 'hf_hub_download', lambda *args, **kwargs: str(cached))

    destination = tmp_path / 'instance'
    with pytest.raises(ValueError) as error:
        sources.download_instance('astronomy.example', destination, 'main', None)
    if cached_kind == 'wrong_size':
        assert str(error.value) == (
            'HF file size mismatch: prompt_b1.md; expected=1024, actual=800'
        )
    else:
        assert str(error.value) == 'HF downloaded file missing or invalid: prompt_b1.md'
    assert not destination.exists()


@pytest.mark.parametrize('missing_level', [1, 2, 3, 4])
def test_hf_requires_four_matching_prompt_files(tmp_path, monkeypatch, missing_level):
    """Guards the B1-only completeness check introduced in commit c2f7220d."""
    from types import SimpleNamespace
    from benchmarks.asi_bench import sources

    # Four prompt files in total, but B5 must not substitute for a missing B1–B4.
    names = [f'prompt_b{level}.md' for level in range(1, 5) if level != missing_level]
    names += ['prompt_b5.md', 'reference/answer.bin']
    entries = [sources.RepoFile(
        path=f'tasks/astronomy.example__seed31415/{name}', size=1, oid='e' * 40,
    ) for name in names]
    api = SimpleNamespace(
        dataset_info=lambda *args, **kwargs: SimpleNamespace(sha='d' * 40),
        list_repo_tree=lambda *args, **kwargs: entries,
    )
    cached = tmp_path / 'cached-file'
    cached.write_bytes(b'x')
    monkeypatch.setattr(sources, 'HfApi', lambda **kwargs: api)
    monkeypatch.setattr(sources, 'hf_hub_download', lambda *args, **kwargs: str(cached))
    with pytest.raises(ValueError, match=r'prompt count mismatch: expected=4, actual=3'):
        sources.download_instance('astronomy.example', tmp_path / 'instance', 'main', None)


def test_acquire_uses_minimal_default_paths(tmp_path, monkeypatch):
    from benchmarks.asi_bench import sources
    monkeypatch.setattr(sources, 'DEFAULT_DOWNLOADS_DIR', tmp_path / 'downloads')
    monkeypatch.setattr(sources, 'DEFAULT_HF_CACHE_DIR', tmp_path / 'hf')
    seen = {}

    def hf(task, destination, revision, cache_dir):
        seen['hf_cache'] = cache_dir
        return fake_hf(task, destination, revision, cache_dir)

    def github(task, destination, revision):
        seen['output'] = destination
        return fake_github(task, destination, revision)

    with patch('benchmarks.asi_bench.sources.download_instance', hf), patch(
        'benchmarks.asi_bench.sources.download_task_bundle', github
    ):
        manifest = acquire('astronomy.example')
    assert manifest['status'] == 'downloaded'
    assert seen['hf_cache'] == tmp_path / 'hf'
    assert seen['output'].name == 'example'
    assert (tmp_path / 'downloads' / 'astronomy.example').is_dir()
