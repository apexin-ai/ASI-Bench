"""Regression tests for task environments from source and wheel installs."""

import shutil
from pathlib import Path
from unittest.mock import patch

from ai4sci_bench.runner.task_env import TaskEnvironmentManager
from ai4sci_bench.runner.runtime_root import resolve_runtime_root


def _build_with_mocked_commands(manager: TaskEnvironmentManager, env_dir: Path):
    commands: list[list[str]] = []

    with (
        patch.object(manager, "_run", side_effect=lambda command, cwd: commands.append(command)),
        patch.object(manager, "_resolve_python_version", return_value="3.13.0"),
        patch.object(manager, "_write_metadata"),
    ):
        manager._build_env(
            env_dir=env_dir,
            cache_key="test-key",
            runtime_packages=[],
            python_requirement=None,
        )

    return commands


def test_source_checkout_is_installed_editable(tmp_path):
    project_root = tmp_path / "checkout"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text("[project]\nname='asibench'\n")
    manager = TaskEnvironmentManager(project_root, cache_root=tmp_path / "cache")

    commands = _build_with_mocked_commands(manager, tmp_path / "env-source")

    assert commands[1][-2:] == ["-e", str(project_root)]


def test_wheel_install_uses_pinned_distribution_not_site_packages(tmp_path):
    installed_root = tmp_path / "venv" / "lib" / "python3.13" / "site-packages"
    installed_root.mkdir(parents=True)
    manager = TaskEnvironmentManager(installed_root, cache_root=tmp_path / "cache")

    with patch("importlib.metadata.version", return_value="0.1.0"):
        commands = _build_with_mocked_commands(manager, tmp_path / "env-wheel")

    install_command = commands[1]
    assert install_command[-1] == "asibench==0.1.0"
    assert "-e" not in install_command
    assert str(installed_root) not in install_command


def test_runtime_root_prefers_source_checkout_hint(tmp_path):
    project_root = tmp_path / "checkout"
    tasks_dir = project_root / "tasks"
    (project_root / "ai4sci_bench").mkdir(parents=True)
    tasks_dir.mkdir()
    (project_root / "pyproject.toml").write_text("[project]\nname='asibench'\n")

    assert resolve_runtime_root(tasks_dir) == project_root


def test_runtime_root_falls_back_to_user_config_for_wheel(tmp_path):
    fallback = tmp_path / "config" / "runtime"
    with (
        patch(
            "ai4sci_bench.runner.runtime_root._source_project_root",
            return_value=None,
        ),
        patch(
            "ai4sci_bench.runner.runtime_root.config_path",
            return_value=fallback,
        ),
    ):
        assert resolve_runtime_root() == fallback

    assert fallback.is_dir()


def test_lock_survives_departing_holder_deleting_the_new_dir(tmp_path):
    """A stale reaper / departing holder's rmtree must not kill the new holder.

    ``shutil.rmtree`` is not atomic, so its final ``rmdir`` can remove the
    directory another process just created. The metadata write then failed with
    ``FileNotFoundError`` straight out of the context manager, aborting the
    instance with a zero-second run.
    """
    manager = TaskEnvironmentManager(tmp_path / "repo", cache_root=tmp_path / "cache")
    lock_path = manager._lock_path("abc123")
    manager.LOCK_POLL_INTERVAL_SECONDS = 0

    real_write = manager._write_lock_metadata
    calls = {"n": 0}

    def flaky_write(path, token):
        calls["n"] += 1
        if calls["n"] == 1:
            shutil.rmtree(path, ignore_errors=True)  # simulate the racing rmdir
        real_write(path, token)

    with patch.object(manager, "_write_lock_metadata", side_effect=flaky_write):
        with manager._acquire_lock(lock_path):
            assert lock_path.is_dir()
            assert manager._read_lock_token(lock_path) is not None

    assert calls["n"] == 2  # first attempt was retried, not raised
    assert not lock_path.exists()


def test_release_does_not_delete_a_lock_owned_by_someone_else(tmp_path):
    """If a reaper handed the lock on, releasing ours must leave theirs alone."""
    manager = TaskEnvironmentManager(tmp_path / "repo", cache_root=tmp_path / "cache")
    lock_path = manager._lock_path("abc123")

    with manager._acquire_lock(lock_path):
        manager._write_lock_metadata(lock_path, "someone-elses-token")

    assert lock_path.is_dir()
    assert manager._read_lock_token(lock_path) == "someone-elses-token"
