"""Tests for sandbox_support — banner output, validation, and descriptions."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from ai4sci_bench.runner.sandbox_support import (
    KNOWN_SANDBOX_MODES,
    SANDBOX_DESCRIPTIONS,
    SANDBOX_HELP,
    print_sandbox_banner,
    validate_sandbox_mode,
)


class TestSandboxDescriptions:
    """Verify SANDBOX_DESCRIPTIONS covers all known modes."""

    def test_all_known_modes_have_descriptions(self):
        for mode in KNOWN_SANDBOX_MODES:
            assert mode in SANDBOX_DESCRIPTIONS, f"Missing description for mode '{mode}'"

    def test_descriptions_have_required_keys(self):
        required_keys = {"label", "description", "pros", "cons"}
        for mode, info in SANDBOX_DESCRIPTIONS.items():
            assert set(info.keys()) == required_keys, (
                f"Mode '{mode}' has keys {set(info.keys())}, expected {required_keys}"
            )

    def test_sandbox_help_mentions_all_modes(self):
        for mode in KNOWN_SANDBOX_MODES:
            assert mode in SANDBOX_HELP


class TestPrintSandboxBanner:
    """Test print_sandbox_banner output for each mode."""

    def test_banner_none_mode(self, capsys):
        print_sandbox_banner("none")
        output = capsys.readouterr().out
        assert "Sandbox mode: --sandbox none" in output
        assert "No isolation" in output
        assert "Tip:" in output  # Only 'none' mode has a tip

    def test_banner_task_mode(self, capsys):
        print_sandbox_banner("task")
        output = capsys.readouterr().out
        assert "Sandbox mode: --sandbox task" in output
        assert "Python venv" in output or "task" in output.lower()
        assert "Tip:" not in output  # Only 'none' has a tip

    def test_banner_os_mode(self, capsys):
        print_sandbox_banner("os")
        output = capsys.readouterr().out
        assert "Sandbox mode: --sandbox os" in output
        assert "Docker" in output
        assert "Tip:" not in output

    def test_banner_unknown_mode_no_output(self, capsys):
        print_sandbox_banner("bogus")
        output = capsys.readouterr().out
        assert output == ""

    def test_banner_includes_pros_and_cons(self, capsys):
        for mode in KNOWN_SANDBOX_MODES:
            print_sandbox_banner(mode)
            output = capsys.readouterr().out
            assert "Pros:" in output
            assert "Cons:" in output


class TestValidateSandboxMode:
    """Test validate_sandbox_mode for various scenarios."""

    def test_valid_mode_passes(self):
        validate_sandbox_mode("none", supported_modes=("none", "task"), component="Test")

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown sandbox mode 'bogus'"):
            validate_sandbox_mode("bogus", supported_modes=("none",), component="Test")

    def test_unsupported_mode_raises(self):
        with pytest.raises(ValueError, match="Test does not support sandbox 'os'"):
            validate_sandbox_mode("os", supported_modes=("none", "task"), component="Test")

    def test_error_lists_known_modes(self):
        with pytest.raises(ValueError, match="none, task, os, linux_ns"):
            validate_sandbox_mode("bogus", supported_modes=("none",), component="Test")

    def test_error_lists_supported_modes(self):
        with pytest.raises(ValueError, match="none, task"):
            validate_sandbox_mode("os", supported_modes=("none", "task"), component="Test")
