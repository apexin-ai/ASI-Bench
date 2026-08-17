"""Manual PAT validation never saves malformed or rejected clipboard text."""

import io
import urllib.error

import pytest

from ai4sci_bench.auth.token_login import TokenLoginError, validate_portal_token

FAKE_PAT = "asi_" + "pat_" + "abcdefghijklmnopqrstuvwxyz"


class _Response:
    status = 200

    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self._body


def test_validate_portal_token_returns_account_and_sends_bearer(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        return _Response(b'{"id":"user-1","primary_email":"alice@example.org"}')

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    user = validate_portal_token(
        "https://portal.example/api/v1",
        FAKE_PAT,
    )

    assert user["id"] == "user-1"
    assert captured["request"].full_url.endswith("/api/v1/me")
    assert captured["request"].headers["Authorization"].startswith("Bearer asi_pat_")


def test_validate_portal_token_rejects_overwritten_clipboard_before_network(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network called")),
    )
    with pytest.raises(TokenLoginError, match="not a complete"):
        validate_portal_token("https://portal.example/api/v1", "new clipboard text")


def test_validate_portal_token_explains_revoked_token(monkeypatch):
    error = urllib.error.HTTPError(
        "url", 401, "Unauthorized", {}, io.BytesIO(b'{"detail":"Invalid token"}'),
    )
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(error),
    )
    with pytest.raises(TokenLoginError, match="incomplete, revoked"):
        validate_portal_token(
            "https://portal.example/api/v1",
            FAKE_PAT,
        )
