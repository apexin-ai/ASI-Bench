"""Offline checks for the opt-in, revision-bound upstream patch bundle."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

BUNDLE = Path(__file__).resolve().parents[1] / 'scripts/mcp/cad-repairs'
spec = importlib.util.spec_from_file_location('cad_repair_apply', BUNDLE / 'apply.py')
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


def test_patch_manifest():
    manifest = json.loads((BUNDLE / 'manifest.json').read_text())
    assert {e['id'] for e in manifest['servers']} == {'sketchup', 'fusion360', 'cadquery'}
    for entry in manifest['servers']:
        patch = (BUNDLE / entry['patch']).read_bytes()
        assert hashlib.sha256(patch).hexdigest() == entry['patch_sha256']
        assert len(entry['revision']) == 40
        assert b'/Users/' not in patch
        constraints = (BUNDLE / entry['constraints']).read_text()
        assert 'mcp==1.30.0' in constraints
        assert '/Users/' not in constraints and ' @ ' not in constraints


def make_fixture(tmp_path, count=1):
    bundle = tmp_path / 'bundle'
    root = tmp_path / 'sources'
    bundle.mkdir()
    entries = []
    for i in range(count):
        source = root / f'server{i}'
        source.mkdir(parents=True)
        def git(*args):
            return repair.git(source, *args)
        git('init')
        git('config', 'user.name', 'Test')
        git('config', 'user.email', 'test@example.invalid')
        (source / 'test.txt').write_text('before\n')
        git('add', 'test.txt')
        git('commit', '-m', 'fixture')
        revision = git('rev-parse', 'HEAD')
        (source / 'test.txt').write_text('after\n')
        patch = (git('diff') + '\n').encode()
        git('checkout', '--', 'test.txt')
        filename = f'server{i}.patch'
        (bundle / filename).write_bytes(patch)
        entries.append(dict(id=f'server{i}', revision=revision, patch=filename,
                            patch_sha256=hashlib.sha256(patch).hexdigest()))
    (bundle / 'manifest.json').write_text(json.dumps({'servers': entries}))
    return root, bundle


def test_apply_and_refuse_repeat(tmp_path):
    root, bundle = make_fixture(tmp_path)
    repair.apply(root, bundle)
    assert (root / 'server0/test.txt').read_text() == 'after\n'
    with pytest.raises(subprocess.CalledProcessError):
        repair.apply(root, bundle)


def test_checksum_fail_closed(tmp_path):
    root, bundle = make_fixture(tmp_path)
    (bundle / 'server0.patch').write_text('tampered')
    with pytest.raises(ValueError, match='checksum'):
        repair.apply(root, bundle)
    assert (root / 'server0/test.txt').read_text() == 'before\n'


def test_preflight_all_before_modifying(tmp_path):
    root, bundle = make_fixture(tmp_path, count=2)
    manifest = json.loads((bundle / 'manifest.json').read_text())
    manifest['servers'][1]['revision'] = '0' * 40
    (bundle / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='revision'):
        repair.apply(root, bundle)
    assert (root / 'server0/test.txt').read_text() == 'before\n'


def test_dirty_checkout_fail_closed(tmp_path):
    root, bundle = make_fixture(tmp_path)
    (root / 'server0/test.txt').write_text('user edits\n')
    with pytest.raises(subprocess.CalledProcessError):
        repair.apply(root, bundle)
    assert (root / 'server0/test.txt').read_text() == 'user edits\n'
