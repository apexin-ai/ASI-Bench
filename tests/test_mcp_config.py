import json
import tomllib
from pathlib import Path

import pytest
from click.testing import CliRunner

from ai4sci_bench.adapters.claude_code_cli import ClaudeCodeCLIAdapter
from ai4sci_bench.adapters.codex_cli import CodexCLIAdapter
from ai4sci_bench.cli import cli
from ai4sci_bench.mcp_config import load_mcp_config, load_science_mcp_catalog


@pytest.fixture
def mcp_config(tmp_path: Path) -> Path:
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({
        "mcpServers": {
            "netlogo": {
                "command": "netlogo-mcp",
                "args": ["--headless"],
                "env": {"NETLOGO_HOME": "/opt/netlogo"},
            },
            "pubchem": {
                "url": "http://127.0.0.1:8765/mcp",
                "headers": {"X-Test": "value"},
            },
        }
    }), encoding="utf-8")
    return path


def test_load_mcp_config_rejects_empty_and_unknown_fields(tmp_path: Path):
    empty = tmp_path / "empty.json"
    empty.write_text('{"mcpServers": {}}', encoding="utf-8")
    with pytest.raises(ValueError, match="at least one"):
        load_mcp_config(empty)

    bad = tmp_path / "bad.json"
    bad.write_text(
        '{"mcpServers":{"x":{"command":"x","secret":"nope"}}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported field"):
        load_mcp_config(bad)


def test_claude_explicit_mcp_config_is_strict_and_allowlisted(
    mcp_config: Path, sample_task_instance,
):
    adapter = ClaudeCodeCLIAdapter(tool_mode="search", mcp_config=str(mcp_config))
    cmd = adapter._build_command(sample_task_instance, task_env=None)
    assert cmd[cmd.index("--mcp-config") + 1] == str(mcp_config.resolve())
    assert "--strict-mcp-config" in cmd
    tools = cmd[cmd.index("--tools") + 1]
    assert "mcp__netlogo__*" in tools
    assert "mcp__pubchem__*" in tools


def test_codex_explicit_mcp_config_is_written_to_isolated_home(
    mcp_config: Path, sample_task_instance, tmp_path: Path,
):
    adapter = CodexCLIAdapter(tool_mode="search", mcp_config=str(mcp_config))
    adapter.setup({"sandbox": "none", "repo_root": str(tmp_path / "repo")})
    env = adapter._build_run_env(sample_task_instance, task_env=None)
    config = (Path(env["CODEX_HOME"]) / "config.toml").read_text(encoding="utf-8")
    parsed = tomllib.loads(config)
    assert set(parsed["mcp_servers"]) == {"netlogo", "pubchem"}
    assert '[mcp_servers.netlogo]' in config
    assert 'command = "netlogo-mcp"' in config
    assert 'args = ["--headless"]' in config
    assert '[mcp_servers.netlogo.env]' in config
    assert 'NETLOGO_HOME = "/opt/netlogo"' in config
    assert '[mcp_servers.pubchem]' in config
    assert 'url = "http://127.0.0.1:8765/mcp"' in config
    assert 'http_headers = { X-Test = "value" }' in config
    cmd = adapter._build_command(sample_task_instance, task_env=None)
    assert "--ignore-user-config" not in cmd
    assert "--ignore-rules" in cmd


@pytest.mark.parametrize("adapter_cls", [ClaudeCodeCLIAdapter, CodexCLIAdapter])
def test_explicit_mcp_rejected_in_os_sandbox(adapter_cls, mcp_config: Path):
    adapter = adapter_cls(tool_mode="search", mcp_config=str(mcp_config))
    with pytest.raises(ValueError, match="not supported with --sandbox os"):
        adapter.setup({"sandbox": "os"})


def test_run_help_exposes_mcp_config():
    result = CliRunner().invoke(cli, ["run", "--help"])
    assert result.exit_code == 0
    assert "--mcp-config" in result.output


def test_run_rejects_mcp_for_unsupported_agent(mcp_config: Path):
    result = CliRunner().invoke(cli, [
        "run", "--agent", "direct_llm", "--mcp-config", str(mcp_config),
    ])
    assert result.exit_code != 0
    assert "requires --agent claude_code_cli or codex_cli" in result.output


def test_run_rejects_mcp_os_before_docker_probe(mcp_config: Path):
    result = CliRunner().invoke(cli, [
        "run", "--agent", "codex_cli", "--mcp-config", str(mcp_config),
        "--sandbox", "os",
    ])
    assert result.exit_code != 0
    assert "not supported with --sandbox os" in result.output


def test_mcp_catalog_and_init_commands(tmp_path: Path):
    runner = CliRunner()
    catalog_result = runner.invoke(cli, ["mcp", "catalog"])
    assert catalog_result.exit_code == 0
    assert "matlab" in catalog_result.output
    assert "energyplus" in catalog_result.output

    output = tmp_path / "science-mcp.json"
    init_result = runner.invoke(cli, [
        "mcp", "init", "--servers", "netlogo,pubchem", "--output", str(output),
    ])
    assert init_result.exit_code == 0
    assert set(load_mcp_config(output)) == {"netlogo", "pubchem"}


def test_mcp_check_reports_unconfigured_template(tmp_path: Path):
    output = tmp_path / "science-mcp.json"
    result = CliRunner().invoke(cli, [
        "mcp", "init", "--servers", "comsol", "--output", str(output),
    ])
    assert result.exit_code == 0
    checked = CliRunner().invoke(cli, ["mcp", "check", "--config", str(output)])
    assert checked.exit_code != 0
    assert "need local installation/configuration" in checked.output


def test_science_catalog_covers_requested_servers():
    catalog = load_science_mcp_catalog()
    expected = {
        "comsol", "openfoam", "matlab", "simulink", "mworks", "pynite",
        "energyplus", "text2sim", "netlogo", "afsim", "blender", "freecad",
        "autocad", "fusion360", "sketchup", "pubchem", "gns3",
    }
    assert set(catalog) == expected
    for name, entry in catalog.items():
        assert entry["source"].startswith("https://")
        assert entry["prerequisites"]
        assert "server" in entry
        assert load_mcp_config_document({"mcpServers": {name: entry["server"]}})


def load_mcp_config_document(document: dict) -> dict:
    """Exercise the public file parser for one in-memory catalog fixture."""
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "mcp.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return load_mcp_config(path)
