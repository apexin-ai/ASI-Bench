"""Regression tests for the web-only ``asibench task submit`` command."""

import importlib.util

from click.testing import CliRunner

import ai4sci_bench.submission as submission
from ai4sci_bench.cli import cli


def test_cli_task_submit_opens_web_form_without_upload_options(monkeypatch):
    opened = []
    monkeypatch.setattr(
        "webbrowser.open",
        lambda url, new=0: opened.append((url, new)) or True,
    )

    result = CliRunner().invoke(
        cli,
        ["task", "submit", "--endpoint", "https://portal.example/api/v1"],
    )

    assert result.exit_code == 0, result.output
    assert opened == [("https://portal.example/submit/proposals/new", 2)]
    assert "Task submission is available only in the web portal" in result.output
    assert "requires careful review" in result.output
    assert "https://portal.example/submit/proposals/new" in result.output
    assert "Uploading" not in result.output
    assert "synchronized" not in result.output


def test_cli_task_submit_no_browser_prints_web_form_without_opening(monkeypatch):
    def unexpected_open(*args, **kwargs):
        raise AssertionError("browser must not open when ASIBENCH_NO_BROWSER is set")

    monkeypatch.setattr("webbrowser.open", unexpected_open)
    result = CliRunner().invoke(
        cli,
        ["task", "submit"],
        env={
            "ASIBENCH_SUBMIT_ENDPOINT": "https://portal.example/submit/settings",
            "ASIBENCH_NO_BROWSER": "1",
        },
    )

    assert result.exit_code == 0, result.output
    assert "https://portal.example/submit/proposals/new" in result.output
    assert "Automatic browser opening is disabled" in result.output


def test_cli_task_submit_help_exposes_only_web_endpoint():
    result = CliRunner().invoke(cli, ["task", "submit", "--help"])

    assert result.exit_code == 0
    assert "--endpoint" in result.output
    assert "--task-dir" not in result.output
    assert "--token" not in result.output
    assert "--force-file-sync" not in result.output


def test_cli_task_submit_defaults_to_official_portal(monkeypatch):
    opened = []
    monkeypatch.setattr(
        "webbrowser.open",
        lambda url, new=0: opened.append((url, new)) or True,
    )

    result = CliRunner().invoke(
        cli,
        ["task", "submit"],
        env={"ASIBENCH_SUBMIT_ENDPOINT": "", "AI4SCI_SUBMIT_ENDPOINT": ""},
    )

    expected = "https://asibench.apexin.ai/submit/proposals/new"
    assert result.exit_code == 0, result.output
    assert opened == [(expected, 2)]
    assert expected in result.output


def test_task_upload_module_and_public_exports_are_removed():
    assert importlib.util.find_spec("ai4sci_bench.submission.task_submit") is None
    assert not hasattr(submission, "submit_task")
    assert not hasattr(submission, "collect_task_files")
