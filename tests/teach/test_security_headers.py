"""Security headers on every Teach response; no inline script or styles in templates (design 16)."""

from __future__ import annotations

import re

import pytest

from teach.blueprint import CSP

from .conftest import REPO, post, submit

TEMPLATES = sorted((REPO / "teach" / "templates" / "teach").glob("*.html"))


def _check(resp, noindex=False, nostore=False):
    h = resp.headers
    assert h["Content-Security-Policy"] == CSP
    assert "default-src 'self'" in CSP and "img-src 'self' data:" in CSP and "script-src 'self'" in CSP
    assert "unsafe-inline" not in CSP
    assert h["X-Content-Type-Options"] == "nosniff" and h["Referrer-Policy"] == "no-referrer"
    assert (h.get("X-Robots-Tag") == "noindex") == noindex
    assert (h.get("Cache-Control") == "no-store") == nostore


def test_pages(teach):
    body, sub_id, _ = submit(teach)
    n = body["number"]
    cases = [("/teach", False, False), ("/teach/spec", False, False), ("/teach/openapi.json", False, False),
             ("/teach/schema/submission.json", False, False), ("/teach/example/image.png", False, False),
             ("/teach/static/teach/teach.js", False, False), ("/teach/queue", True, False),
             (f"/teach/q/{n}", True, False), (f"/teach/manage/{sub_id}", True, True),
             ("/teach/api/queue", False, True), (f"/teach/api/submissions/{n}", False, True),
             ("/teach/q/9999", True, False)]
    for url, noindex, nostore in cases:
        r = teach.client.get(url)
        assert r.status_code in (200, 404), url
        _check(r, noindex, nostore)


def test_api_posts_and_errors(teach):
    _check(post(teach.client, "/teach/api/validate"), nostore=True)
    _check(teach.client.post("/teach/api/submissions", data={}, content_type="multipart/form-data"), nostore=True)


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda p: p.name)
def test_templates_have_no_inline_script_or_style(path):
    html = path.read_text(encoding="utf-8")
    for m in re.finditer(r"<script\b[^>]*>(.*?)</script>", html, re.S | re.I):
        assert "src=" in m.group(0).split(">")[0] and not m.group(1).strip(), path.name
    assert not re.search(r"\son[a-z]+\s*=", html, re.I), path.name
    assert not re.search(r"\sstyle\s*=", html, re.I) and "<style" not in html.lower(), path.name
    assert not re.search(r"=\s*[\"']?\s*javascript:", html, re.I), path.name


def test_static_js_has_no_html_injection():
    for js in (REPO / "teach" / "static" / "teach").glob("*.js"):
        text = js.read_text(encoding="utf-8")
        assert "innerHTML" not in text and "eval(" not in text and "document.write" not in text, js.name
