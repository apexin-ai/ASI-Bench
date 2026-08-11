"""Portal endpoint normalization — accept whatever URL a user pastes.

Users naturally copy the portal URL from the browser address bar: the web root
(``…/submit`` — the portal's basePath on shared deployments where the bench
site owns ``/``), a page inside it (``…/submit/submit-results``), or the bare
domain. But the API is a separate service mounted at ``/api/v1`` on the same
origin, so POSTing a bundle to a page URL yields ``405 Method Not Allowed``
from the web server.

These helpers recognize portal-shaped URLs and rewrite them to the API;
anything unrecognized (a genuinely custom receiver) passes through untouched.
"""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

# Pages a user plausibly copies from the address bar. ``/submit`` itself (the
# web basePath) is handled separately below because pages nest under it.
_PAGE_SUFFIXES = (
    "/proposals/new",
    "/submit-results",
    "/settings",
    "/activate",
    "/dashboard",
)

# Known API paths whose base we can strip back to.
_API_SUFFIXES = ("/submissions/bundle", "/proposals", "/submissions")


def portal_api_base(endpoint: str) -> str | None:
    """Return the ``…/api/v1`` base when *endpoint* is recognizably the portal
    (an API URL, a web-UI URL, or a bare origin); ``None`` when it isn't ours.
    """
    e = (endpoint or "").strip().rstrip("/")
    parts = urlsplit(e)
    if not parts.scheme or not parts.netloc:
        return None
    path = parts.path.rstrip("/")
    for suffix in _API_SUFFIXES:
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    else:
        for page in _PAGE_SUFFIXES:
            if path.endswith(page):
                path = path[: -len(page)]
                break
    if path.endswith("/submit"):  # the portal web basePath
        path = path[: -len("/submit")]
    if path.endswith("/api/v1"):
        api_path = path
    elif path.endswith("/api"):
        api_path = path + "/v1"
    elif path == "":
        api_path = "/api/v1"
    else:
        return None
    return urlunsplit((parts.scheme, parts.netloc, api_path, "", ""))


def bundle_upload_url(endpoint: str) -> tuple[str, str | None]:
    """Resolve the actual bundle-upload URL for ``asibench submit``.

    Returns ``(url, note)``. *note* is a human-readable message when the
    endpoint was rewritten (it pointed at a web page, a bare origin, or an API
    base); ``None`` when the endpoint is used verbatim — either it is already
    the bundle URL, or it is an unrecognized custom receiver we must trust.
    """
    e = (endpoint or "").strip().rstrip("/")
    if e.endswith("/submissions/bundle"):
        return e, None
    api = portal_api_base(e)
    if api is None:
        return e, None
    fixed = api + "/submissions/bundle"
    note = f"'{endpoint}' is not the upload API — uploading to {fixed} instead"
    return fixed, note


def task_submission_url(endpoint: str) -> str | None:
    """Return the Portal page used to author and submit a new task.

    Task submission is intentionally web-only: unlike run-bundle upload, it
    requires guided metadata, local-test evidence, file review, and explicit
    author confirmations. Return ``None`` when *endpoint* cannot safely be
    identified as the ASI-Bench Portal.
    """
    api = portal_api_base(endpoint)
    if api is None:
        return None
    parts = urlsplit(api)
    return urlunsplit(
        (parts.scheme, parts.netloc, "/submit/proposals/new", "", "")
    )
