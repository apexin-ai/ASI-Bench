"""Regression tests for task environments from source and wheel installs."""

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
