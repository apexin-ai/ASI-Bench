"""Image-environment compatibility, without an installed ASI distribution."""
import importlib.util
import sys
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest

from benchmarks.asi_bench.scorer import ADAPTER_ROOT


@pytest.fixture
def runtime(monkeypatch):
    path = ADAPTER_ROOT / 'templates/verifier/task_env.py'
    spec = importlib.util.spec_from_file_location('image_task_env', path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_existing_image_environment_never_installs(runtime, tmp_path, monkeypatch):
    """Guards image-only replacement of native caches from baseline commit 9b90d16c."""
    import subprocess

    def forbidden(*args, **kwargs):
        raise AssertionError('must not create a subprocess or install anything')

    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    cache = tmp_path / 'unused-cache'
    manager = runtime.TaskEnvironmentManager(tmp_path, cache_root=cache)
    env = manager.ensure_env({'_runtime_python': '>=3.11', '_runtime_packages': ['packaging>=20']})
    assert env.env_dir == Path(sys.prefix)
    assert env.python_executable == Path(sys.executable)
    assert not cache.exists()
    original = {'PATH': '/bin', 'MY_VAR': 'preserved'}
    child = env.build_subprocess_env(original)
    assert child['PATH'].split(':')[0] == str(Path(sys.executable).parent)
    assert child['MY_VAR'] == 'preserved' and original == {'PATH': '/bin', 'MY_VAR': 'preserved'}
    assert env.build_subprocess_env({}).keys() == {'PATH'}


@pytest.mark.parametrize('declaration,error', [
    ({'_runtime_python': '>=99'}, ValueError),
    ({'_runtime_packages': ['packaging>=9999']}, ValueError),
    ({'_runtime_packages': ['asi-missing-dependency-9b90d16c']}, PackageNotFoundError),
    ({'_runtime_packages': ['example @ https://invalid.example/pkg.whl']}, ValueError),
])
def test_unsatisfied_runtime_is_not_repaired(runtime, tmp_path, declaration, error):
    """Guards fail-fast image dependency checks replacing commit 9b90d16c's cache path."""
    with pytest.raises(error):
        runtime.TaskEnvironmentManager(tmp_path).ensure_env(declaration)
    assert not list(tmp_path.iterdir())


def test_runtime_marker_uses_current_environment(runtime, tmp_path):
    """Guards image compatibility introduced after baseline commit 9b90d16c."""
    env = runtime.TaskEnvironmentManager(tmp_path).ensure_env({
        '_runtime_packages': ['asi-missing-dependency; python_version < "1"']})
    assert env.env_dir == Path(sys.prefix)
