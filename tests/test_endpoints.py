"""Endpoint normalization — web-page/origin URLs must resolve to the API.

Regression for the real-world 405: a runner set ASIBENCH_SUBMIT_ENDPOINT to the
portal web root (``…/submit``) and the bundle was POSTed at a Next.js page.
"""

from __future__ import annotations

import pytest

from ai4sci_bench.branding import submit_endpoint
from ai4sci_bench.submission.endpoints import (
    bundle_upload_url,
    portal_api_base,
    task_proposal_url,
    task_submission_url,
    token_settings_url,
)

STAGING = "https://staging.portal.example.org"
OFFICIAL = "https://asibench.apexin.ai"


def test_official_submission_endpoint_is_default(monkeypatch):
    monkeypatch.delenv("ASIBENCH_SUBMIT_ENDPOINT", raising=False)
    monkeypatch.delenv("AI4SCI_SUBMIT_ENDPOINT", raising=False)

    assert submit_endpoint() == OFFICIAL
    assert bundle_upload_url(submit_endpoint())[0] == (
        f"{OFFICIAL}/api/v1/submissions/bundle"
    )


class TestPortalApiBase:
    @pytest.mark.parametrize(
        "url",
        [
            f"{STAGING}/submit",
            f"{STAGING}/submit/",
            f"{STAGING}/submit/submit-results",
            f"{STAGING}/submit/settings",
            f"{STAGING}/submit/dashboard",
            f"{STAGING}/submit-results",
            f"{STAGING}",
            f"{STAGING}/",
            f"{STAGING}/api/v1",
            f"{STAGING}/api/v1/",
            f"{STAGING}/api",
            f"{STAGING}/api/v1/submissions/bundle",
            f"{STAGING}/api/v1/proposals",
            f"{STAGING}/api/v1/submissions",
        ],
    )
    def test_portal_shaped_urls_resolve_to_api_v1(self, url):
        assert portal_api_base(url) == f"{STAGING}/api/v1"

    def test_query_and_fragment_are_dropped(self):
        assert portal_api_base(f"{STAGING}/submit?tab=1#top") == f"{STAGING}/api/v1"

    @pytest.mark.parametrize(
        "url",
        [
            "https://scoring.example.com/upload",  # custom receiver
            f"{STAGING}/docs",  # unknown page
            "not-a-url",
            "",
        ],
    )
    def test_unrecognized_urls_return_none(self, url):
        assert portal_api_base(url) is None


class TestBundleUploadUrl:
    def test_web_root_is_rewritten_with_note(self):
        url, note = bundle_upload_url(f"{STAGING}/submit")
        assert url == f"{STAGING}/api/v1/submissions/bundle"
        assert note and f"{STAGING}/submit" in note

    def test_bare_origin_is_rewritten(self):
        url, note = bundle_upload_url(STAGING)
        assert url == f"{STAGING}/api/v1/submissions/bundle"
        assert note

    def test_api_base_gets_bundle_path(self):
        url, note = bundle_upload_url(f"{STAGING}/api/v1")
        assert url == f"{STAGING}/api/v1/submissions/bundle"
        assert note

    def test_exact_bundle_url_passes_verbatim(self):
        url, note = bundle_upload_url(f"{STAGING}/api/v1/submissions/bundle")
        assert url == f"{STAGING}/api/v1/submissions/bundle"
        assert note is None

    def test_custom_receiver_passes_verbatim(self):
        url, note = bundle_upload_url("https://scoring.example.com/upload")
        assert url == "https://scoring.example.com/upload"
        assert note is None


class TestTaskSubmissionUrl:
    @pytest.mark.parametrize(
        "endpoint",
        [
            STAGING,
            f"{STAGING}/submit",
            f"{STAGING}/submit/settings",
            f"{STAGING}/api/v1",
            f"{STAGING}/api/v1/proposals",
        ],
    )
    def test_portal_endpoint_opens_new_task_page(self, endpoint):
        assert task_submission_url(endpoint) == f"{STAGING}/submit/proposals/new"

    def test_unrecognized_endpoint_has_no_safe_web_target(self):
        assert task_submission_url("https://receiver.example/upload") is None

    def test_cli_draft_and_token_settings_urls_keep_portal_base_path(self):
        assert task_proposal_url(STAGING, "proposal-1") == (
            f"{STAGING}/submit/proposals/proposal-1"
        )
        assert token_settings_url(STAGING) == (
            f"{STAGING}/submit/settings#cli-tokens"
        )
