"""Tests for KimiCodeCLIAdapter endpoint resolution and config generation."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from ai4sci_bench.adapters.kimi_code_cli import KimiCodeCLIAdapter


# ── _resolve_endpoint: official endpoints, no api_protocol ─────

@pytest.mark.parametrize("api_base,expected_base,expected_type", [
    # Regular pay-as-you-go
    ("https://api.moonshot.cn/v1",     "https://api.moonshot.cn/v1",     "kimi"),
    ("https://api.moonshot.ai/v1",     "https://api.moonshot.ai/v1",     "kimi"),
    # Coding Plan
    ("https://api.kimi.com/coding/v1", "https://api.kimi.com/coding/v1", "kimi"),
    ("https://api.kimi.com/coding",    "https://api.kimi.com/coding",    "anthropic"),
    ("https://api.kimi.com/coding/",   "https://api.kimi.com/coding",    "anthropic"),
    # Trailing slash on OpenAI-shape endpoints
    ("https://api.moonshot.cn/v1/",    "https://api.moonshot.cn/v1",     "kimi"),
    ("https://api.kimi.com/coding/v1/", "https://api.kimi.com/coding/v1", "kimi"),
])
def test_official_endpoint_auto_resolves(api_base, expected_base, expected_type):
    base, cli_type = KimiCodeCLIAdapter._resolve_endpoint(api_base, None)
    assert base == expected_base
    assert cli_type == expected_type


# ── _resolve_endpoint: third-party requires api_protocol ───────

def test_third_party_without_protocol_raises():
    with pytest.raises(ValueError, match="not a recognized official"):
        KimiCodeCLIAdapter._resolve_endpoint("https://api.tokenrouter.com/v1", None)


def test_third_party_with_openai():
    base, cli_type = KimiCodeCLIAdapter._resolve_endpoint(
        "https://api.tokenrouter.com/v1", "openai"
    )
    assert base == "https://api.tokenrouter.com/v1"
    assert cli_type == "openai"


def test_third_party_with_anthropic_strips_v1():
    # /v1 is stripped because Kimi CLI's Anthropic client appends /v1/messages.
    base, cli_type = KimiCodeCLIAdapter._resolve_endpoint(
        "https://api.tokenrouter.com/v1", "anthropic"
    )
    assert base == "https://api.tokenrouter.com"
    assert cli_type == "anthropic"


def test_third_party_anthropic_without_v1_untouched():
    base, cli_type = KimiCodeCLIAdapter._resolve_endpoint(
        "https://api.tokenrouter.com", "anthropic"
    )
    assert base == "https://api.tokenrouter.com"
    assert cli_type == "anthropic"


# ── _resolve_endpoint: explicit override on official endpoints ─

def test_api_protocol_overrides_official_table():
    # User forces openai on Coding Plan OpenAI-shape (bypasses native "kimi")
    base, cli_type = KimiCodeCLIAdapter._resolve_endpoint(
        "https://api.kimi.com/coding/v1", "openai"
    )
    assert base == "https://api.kimi.com/coding/v1"
    assert cli_type == "openai"


def test_api_protocol_anthropic_on_coding_v1_strips_v1():
    # Coding Plan OpenAI URL with anthropic override → /v1 stripped
    base, cli_type = KimiCodeCLIAdapter._resolve_endpoint(
        "https://api.kimi.com/coding/v1", "anthropic"
    )
    assert base == "https://api.kimi.com/coding"
    assert cli_type == "anthropic"


# ── __init__ validation ────────────────────────────────────────

def test_local_login_mode():
    adapter = KimiCodeCLIAdapter()
    assert not adapter._uses_native_provider
    assert adapter._resolved_base is None
    assert adapter._resolved_type is None
    assert adapter._build_api_env() == {}


def test_moonshot_key_only_mode():
    adapter = KimiCodeCLIAdapter(api_key="sk-test")
    assert not adapter._uses_native_provider
    env = adapter._build_api_env()
    assert env == {"MOONSHOT_API_KEY": "sk-test"}


def test_official_coding_openai_shape_mode():
    adapter = KimiCodeCLIAdapter(
        api_key="sk-test",
        api_base="https://api.kimi.com/coding/v1",
    )
    assert adapter._uses_native_provider
    assert adapter._resolved_base == "https://api.kimi.com/coding/v1"
    assert adapter._resolved_type == "kimi"


def test_official_coding_anthropic_shape_mode():
    adapter = KimiCodeCLIAdapter(
        api_key="sk-test",
        api_base="https://api.kimi.com/coding/",
    )
    assert adapter._uses_native_provider
    assert adapter._resolved_base == "https://api.kimi.com/coding"
    assert adapter._resolved_type == "anthropic"


def test_third_party_requires_api_protocol():
    with pytest.raises(ValueError, match="not a recognized official"):
        KimiCodeCLIAdapter(
            api_key="sk-test",
            api_base="https://api.tokenrouter.com/v1",
        )


def test_third_party_with_api_protocol():
    adapter = KimiCodeCLIAdapter(
        api_key="sk-test",
        api_base="https://api.tokenrouter.com/v1",
        api_protocol="openai",
    )
    assert adapter._resolved_base == "https://api.tokenrouter.com/v1"
    assert adapter._resolved_type == "openai"


def test_api_base_without_api_key_raises():
    with pytest.raises(ValueError, match="api_base was given but api_key"):
        KimiCodeCLIAdapter(api_base="https://api.tokenrouter.com/v1")


def test_api_protocol_without_api_base_raises():
    with pytest.raises(ValueError, match="api_protocol was given but api_base"):
        KimiCodeCLIAdapter(api_key="sk-test", api_protocol="openai")


def test_invalid_api_protocol_raises():
    with pytest.raises(ValueError, match="Invalid api_protocol"):
        KimiCodeCLIAdapter(
            api_key="sk-test",
            api_base="https://api.tokenrouter.com/v1",
            api_protocol="totally-made-up",
        )


# ── Legacy `provider` alias ────────────────────────────────────

def test_provider_alias_emits_deprecation_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        adapter = KimiCodeCLIAdapter(
            api_key="sk-test",
            api_base="https://api.tokenrouter.com/v1",
            provider="openai",
        )
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert adapter.api_protocol == "openai"
    # Backwards-compat property still works
    assert adapter.provider == "openai"


def test_provider_and_api_protocol_conflict_raises():
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="disagree"):
        KimiCodeCLIAdapter(
            api_key="sk-test",
            api_base="https://api.tokenrouter.com/v1",
            api_protocol="openai",
            provider="anthropic",
        )


def test_provider_and_api_protocol_agree_ok():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        adapter = KimiCodeCLIAdapter(
            api_key="sk-test",
            api_base="https://api.tokenrouter.com/v1",
            api_protocol="openai",
            provider="openai",
        )
    assert adapter.api_protocol == "openai"


# ── config.toml generation ────────────────────────────────────

def test_generated_config_uses_resolved_fields():
    adapter = KimiCodeCLIAdapter(
        model="kimi-k2.7-code",
        api_key="sk-my-key",
        api_base="https://api.kimi.com/coding/v1",
    )
    config = adapter._generate_kimi_config()
    assert 'type = "kimi"' in config
    assert 'base_url = "https://api.kimi.com/coding/v1"' in config
    assert 'api_key = "sk-my-key"' in config
    assert 'model = "kimi-k2.7-code"' in config


def test_generated_config_anthropic_shape_strips_v1_on_override():
    adapter = KimiCodeCLIAdapter(
        model="kimi-k2.7-code",
        api_key="sk-my-key",
        api_base="https://api.kimi.com/coding/v1",  # OpenAI-shape URL...
        api_protocol="anthropic",                    # ...but user wants Anthropic
    )
    config = adapter._generate_kimi_config()
    assert 'type = "anthropic"' in config
    # /v1 must be stripped so Kimi CLI doesn't double-append
    assert 'base_url = "https://api.kimi.com/coding"' in config
    assert 'base_url = "https://api.kimi.com/coding/v1"' not in config


def test_generated_config_third_party():
    adapter = KimiCodeCLIAdapter(
        model="deepseek-chat",
        api_key="sk-ds",
        api_base="https://api.deepseek.com/v1",
        api_protocol="openai",
    )
    config = adapter._generate_kimi_config()
    assert 'type = "openai"' in config
    assert 'base_url = "https://api.deepseek.com/v1"' in config


# ── kimi_home behavior ────────────────────────────────────────

def test_user_kimi_home_used_as_is(tmp_path):
    # Pre-populate a user config dir
    user_home = tmp_path / "my-kimi"
    user_home.mkdir()
    (user_home / "config.toml").write_text("# my hand-tuned config\n")

    adapter = KimiCodeCLIAdapter(kimi_home=str(user_home))
    env = adapter._build_api_env()
    assert env["KIMI_CODE_HOME"] == str(user_home)
    # We must not overwrite the user's existing config.toml
    assert (user_home / "config.toml").read_text() == "# my hand-tuned config\n"


def test_native_mode_writes_temp_config(tmp_path):
    adapter = KimiCodeCLIAdapter(
        api_key="sk-test",
        api_base="https://api.kimi.com/coding/v1",
    )
    try:
        env = adapter._build_api_env()
        kimi_home = env["KIMI_CODE_HOME"]
        # config.toml exists with the expected content
        with open(f"{kimi_home}/config.toml") as f:
            content = f.read()
        assert 'type = "kimi"' in content
        assert 'base_url = "https://api.kimi.com/coding/v1"' in content
    finally:
        adapter.teardown()


def test_teardown_removes_temp_kimi_home():
    import os
    adapter = KimiCodeCLIAdapter(
        api_key="sk-test",
        api_base="https://api.kimi.com/coding/v1",
    )
    adapter._build_api_env()
    tmp = adapter._temp_kimi_home
    assert tmp is not None and os.path.isdir(tmp)
    adapter.teardown()
    assert not os.path.exists(tmp)
    assert adapter._temp_kimi_home is None


class _RunKeyStub:
    """Minimal task-instance stand-in exposing only run_key."""

    def __init__(self, run_key: str):
        self.run_key = run_key


def test_native_mode_homes_differ_per_instance(tmp_path):
    """Sequentially executed instances must not share a KIMI_CODE_HOME.

    Kimi CLI writes sessions/ and logs/ under its home; a shared dir would
    leak state from one instance into the next.
    """
    import os
    adapter = KimiCodeCLIAdapter(
        api_key="sk-test",
        api_base="https://api.kimi.com/coding/v1",
    )
    try:
        inst_a = _RunKeyStub("task__inst1__seed0__b2")
        inst_b = _RunKeyStub("task__inst2__seed1__b2")

        env_a = adapter._build_api_env(inst_a)
        env_b = adapter._build_api_env(inst_b)

        home_a = Path(env_a["KIMI_CODE_HOME"])
        home_b = Path(env_b["KIMI_CODE_HOME"])
        assert home_a != home_b
        assert home_a.is_relative_to(adapter._temp_kimi_home)
        assert home_b.is_relative_to(adapter._temp_kimi_home)
        assert home_a.is_dir() and home_b.is_dir()
        # Both homes carry the generated provider config.
        for home in (home_a, home_b):
            content = (home / "config.toml").read_text()
            assert 'base_url = "https://api.kimi.com/coding/v1"' in content
        # Same instance resolves to the same home (retries are deterministic).
        assert adapter._build_api_env(inst_a)["KIMI_CODE_HOME"] == str(home_a)
    finally:
        adapter.teardown()


def test_os_sandbox_mounts_are_per_instance():
    """OS-sandbox rw mounts must map a per-instance host dir into the container."""
    adapter = KimiCodeCLIAdapter(
        api_key="sk-test",
        api_base="https://api.kimi.com/coding/v1",
    )
    try:
        mounts_a = adapter._build_os_extra_mounts(_RunKeyStub("task__inst1__seed0__b2"))
        mounts_b = adapter._build_os_extra_mounts(_RunKeyStub("task__inst2__seed1__b2"))

        assert len(mounts_a) == 2 and len(mounts_b) == 2
        assert mounts_a[1].split(":")[0] != mounts_b[1].split(":")[0]
        assert mounts_a[1].split(":")[1] == adapter._CONTAINER_KIMI_HOME
        assert mounts_a[1].endswith(":rw")
    finally:
        adapter.teardown()


def test_user_kimi_home_is_shared_across_instances(tmp_path):
    """An explicit kimi_home is shared by choice and never isolated."""
    user_home = tmp_path / "my-kimi"
    user_home.mkdir()
    (user_home / "config.toml").write_text("# my hand-tuned config\n")

    adapter = KimiCodeCLIAdapter(kimi_home=str(user_home))
    env_a = adapter._build_api_env(_RunKeyStub("task__inst1__seed0__b2"))
    env_b = adapter._build_api_env(_RunKeyStub("task__inst2__seed1__b2"))

    assert env_a["KIMI_CODE_HOME"] == str(user_home)
    assert env_b["KIMI_CODE_HOME"] == str(user_home)
    assert (user_home / "config.toml").read_text() == "# my hand-tuned config\n"


def test_teardown_removes_all_instance_homes():
    import os
    adapter = KimiCodeCLIAdapter(
        api_key="sk-test",
        api_base="https://api.kimi.com/coding/v1",
    )
    adapter._build_api_env(_RunKeyStub("task__inst1__seed0__b2"))
    adapter._build_api_env(_RunKeyStub("task__inst2__seed1__b2"))
    root = adapter._temp_kimi_home
    assert root is not None and os.path.isdir(root)

    adapter.teardown()

    assert not os.path.exists(root)
    assert adapter._temp_kimi_home is None


# ── build_command uses "bench-model" alias only when native ────

class _StubTaskInstance:
    def __init__(self, workspace_dir):
        self.workspace_dir = workspace_dir


def test_build_command_uses_bench_model_in_native_mode(tmp_path):
    (tmp_path / "prompt.md").write_text("hello")
    adapter = KimiCodeCLIAdapter(
        model="kimi-k2.7-code",
        api_key="sk-test",
        api_base="https://api.kimi.com/coding/v1",
    )
    cmd = adapter._build_command(_StubTaskInstance(tmp_path), None)
    # -m argument should reference the bench-model alias, not the raw model name
    assert "bench-model" in cmd
    assert "kimi-k2.7-code" not in cmd


def test_build_command_uses_raw_model_in_moonshot_only_mode(tmp_path):
    (tmp_path / "prompt.md").write_text("hello")
    adapter = KimiCodeCLIAdapter(model="kimi-k2.7", api_key="sk-test")
    cmd = adapter._build_command(_StubTaskInstance(tmp_path), None)
    assert "kimi-k2.7" in cmd
