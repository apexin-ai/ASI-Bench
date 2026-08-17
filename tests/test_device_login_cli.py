"""CLI-side device-flow tests (design doc cli_device_login_design.md §7).

The portal is stubbed at the HTTP layer (a real localhost http.server), so these
exercise the actual urllib transport in auth/device_login.py.
"""
from __future__ import annotations

import json
import os
import stat
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from click.testing import CliRunner

from ai4sci_bench.auth import credentials as creds
from ai4sci_bench.auth.device_login import DeviceLoginError, device_login
from ai4sci_bench.auth.resolve import resolve_token
from ai4sci_bench.cli import cli


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("ASIBENCH_CREDENTIALS_DIR", str(tmp_path / ".asibench"))
    monkeypatch.delenv("ASIBENCH_SUBMIT_TOKEN", raising=False)
    monkeypatch.delenv("AI4SCI_SUBMIT_TOKEN", raising=False)
    monkeypatch.setenv("ASIBENCH_NO_BROWSER", "1")  # never open a browser in tests


class _StubPortal:
    """Minimal /cli/device-code + /cli/device-token portal. `states` is the
    scripted sequence of poll answers, e.g. ["authorization_pending", "ok"]."""

    def __init__(self, states):
        self.states = list(states)
        self.polls = 0
        handler_self = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                json.loads(self.rfile.read(n) or b"{}")
                if self.path.endswith("/cli/device-code"):
                    body, code = {
                        "device_code": "dev-secret", "user_code": "WDJB-MJHT",
                        "verification_uri": "http://stub/activate",
                        "verification_uri_complete": "http://stub/activate?code=WDJB-MJHT",
                        "expires_in": 600, "interval": 0,
                    }, 200
                elif self.path.endswith("/cli/device-token"):
                    handler_self.polls += 1
                    state = handler_self.states.pop(0) if handler_self.states else "expired_token"
                    if state == "ok":
                        body, code = {"access_token": "asi_pat_test123",
                                      "token_type": "bearer",
                                      "user": {"email": "alice@uni.edu"}}, 200
                    else:
                        body, code = {"detail": state}, 400
                else:
                    body, code = {}, 404
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def api(self):
        return f"http://127.0.0.1:{self.server.server_port}/api/v1"

    def stop(self):
        self.server.shutdown()


@pytest.fixture
def portal_ok():
    p = _StubPortal(["authorization_pending", "ok"])
    yield p
    p.stop()


class TestCredentialStore:
    def test_roundtrip_and_permissions(self):
        path = creds.save_credential("http://h/api/v1", "asi_pat_x", {"email": "a@b"})
        assert creds.load_credential("http://h/api/v1/")["token"] == "asi_pat_x"
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600

    def test_clear(self):
        creds.save_credential("http://h/api/v1", "asi_pat_x")
        assert creds.clear_credential("http://h/api/v1") is True
        assert creds.load_credential("http://h/api/v1") is None
        assert creds.clear_credential("http://h/api/v1") is False


class TestDeviceLoginClient:
    def test_pending_then_success(self, portal_ok):
        out = []
        grant = device_login(portal_ok.api, echo=out.append,
                             open_browser=False, sleep=lambda s: None)
        assert grant["access_token"] == "asi_pat_test123"
        assert portal_ok.polls == 2
        joined = "\n".join(out)
        assert "WDJB-MJHT" in joined and "http://stub/activate" in joined

    def test_slow_down_backs_off(self):
        p = _StubPortal(["slow_down", "ok"])
        try:
            sleeps = []
            device_login(p.api, echo=lambda s: None, open_browser=False,
                         sleep=sleeps.append)
            assert sleeps[-1] > sleeps[0]  # interval grew after slow_down
        finally:
            p.stop()

    @pytest.mark.parametrize("state,msg", [
        ("access_denied", "denied"),
        ("expired_token", "expired"),
    ])
    def test_terminal_errors(self, state, msg):
        p = _StubPortal([state])
        try:
            with pytest.raises(DeviceLoginError, match=msg):
                device_login(p.api, echo=lambda s: None, open_browser=False,
                             sleep=lambda s: None)
        finally:
            p.stop()


class TestResolveOrder:
    def test_explicit_wins(self):
        creds.save_credential("http://h/api/v1", "asi_pat_stored")
        tok, src = resolve_token("http://h/api/v1", "asi_pat_cli", interactive=False)
        assert (tok, src) == ("asi_pat_cli", "explicit")

    def test_env_beats_stored(self, monkeypatch):
        creds.save_credential("http://h/api/v1", "asi_pat_stored")
        monkeypatch.setenv("ASIBENCH_SUBMIT_TOKEN", "asi_pat_env")
        tok, src = resolve_token("http://h/api/v1", None, interactive=False)
        assert (tok, src) == ("asi_pat_env", "env")

    def test_stored_used(self):
        creds.save_credential("http://h/api/v1", "asi_pat_stored")
        tok, src = resolve_token("http://h/api/v1", None, interactive=False)
        assert (tok, src) == ("asi_pat_stored", "stored")

    def test_non_interactive_never_logs_in(self):
        tok, src = resolve_token("http://127.0.0.1:1/api/v1", None, interactive=False)
        assert (tok, src) == (None, "none")

    def test_interactive_runs_device_flow(self, portal_ok, monkeypatch):
        # ASIBENCH_NO_BROWSER disables *auto* login entirely (design §2), so lift
        # it here and stub the actual browser call instead.
        monkeypatch.delenv("ASIBENCH_NO_BROWSER", raising=False)
        import importlib
        dl = importlib.import_module("ai4sci_bench.auth.device_login")
        monkeypatch.setattr(dl.webbrowser, "open", lambda *_: False)
        tok, src = resolve_token(portal_ok.api, None, interactive=True,
                                 echo=lambda s: None)
        assert tok == "asi_pat_test123" and src == "device_login"
        assert creds.load_credential(portal_ok.api)["token"] == "asi_pat_test123"

    def test_no_browser_env_disables_auto_login(self, portal_ok):
        # With ASIBENCH_NO_BROWSER=1 (fixture default) even interactive resolution
        # must not start a device flow (design §2: it turns auto-login off).
        tok, src = resolve_token(portal_ok.api, None, interactive=True,
                                 echo=lambda s: None)
        assert (tok, src) == (None, "none")
        assert portal_ok.polls == 0


class TestAuthCommands:
    def test_login_with_pasted_token_and_status_and_logout(self, monkeypatch):
        monkeypatch.setattr(
            "ai4sci_bench.auth.validate_portal_token",
            lambda api, token: {
                "id": "user-1", "primary_email": "alice@example.org",
            },
        )
        monkeypatch.setattr("webbrowser.open", lambda *args, **kwargs: True)
        r = CliRunner().invoke(
            cli,
            ["login", "--endpoint", "http://h/api/v1"],
            input="asi_pat_test\n",
        )
        assert r.exit_code == 0 and "saved" in r.output
        r = CliRunner().invoke(cli, ["auth", "status"])
        assert "http://h/api/v1" in r.output and "asi_pat_test" in r.output
        assert "alice@example.org" in r.output
        r = CliRunner().invoke(cli, ["logout", "--endpoint", "http://h/api/v1"])
        assert r.exit_code == 0 and "Signed out" in r.output
        r = CliRunner().invoke(cli, ["auth", "status"])
        assert "Not logged in" in r.output

    def test_login_device_flow_e2e(self, portal_ok):
        r = CliRunner().invoke(
            cli, ["login", "--device", "--endpoint", portal_ok.api])
        # CliRunner stdin/stdout are not TTYs → login must refuse, not hang.
        assert r.exit_code != 0 and "interactive" in r.output

    def test_logout_all(self):
        creds.save_credential("http://a/api/v1", "asi_pat_1")
        creds.save_credential("http://b/api/v1", "asi_pat_2")
        r = CliRunner().invoke(cli, ["logout", "--all"])
        assert r.exit_code == 0 and creds.list_credentials() == {}

    def test_login_defaults_to_official_portal(self, monkeypatch):
        monkeypatch.setattr(
            "ai4sci_bench.auth.validate_portal_token",
            lambda api, token: {"id": "user-1"},
        )
        monkeypatch.setattr("webbrowser.open", lambda *args, **kwargs: True)
        r = CliRunner().invoke(
            cli, ["login"], input="asi_pat_test\n")
        assert r.exit_code == 0 and "saved" in r.output
        saved = creds.load_credential("https://asibench.apexin.ai/api/v1")
        assert saved and saved["token"] == "asi_pat_test"

    def test_login_help_does_not_accept_token_in_process_arguments(self):
        r = CliRunner().invoke(cli, ["login", "--help"])
        assert r.exit_code == 0
        assert "--token" not in r.output
        assert "--device" in r.output
