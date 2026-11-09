"""Main page (templates/index.html) with Teach off is byte-identical to before Stage 5c, and the
Teach additions appear only when Teach is registered."""

from __future__ import annotations

import hashlib
import re

from .conftest import REPO

# sha-256 of GET / (Teach off) rendered from templates/index.html at 55071d5, before the Teach
# checkbox was added. Jinja renders with "\n" line endings on every platform. If index.html is
# changed on purpose later, re-derive this from the new commit.
PRE_CHANGE_SHA256 = "fefd2901406ac496f3e90f03689f54147136ae50881a8c894d307ff9db78f620"
TEACH_BLOCK = re.compile(r"\{% if teach_main_ui %\}.*?\{% endif %\}", re.S)
MAIN_JS = REPO / "teach" / "static" / "teach" / "main_teach.js"


def _page(mod) -> bytes:
    r = mod.app.test_client().get("/")
    assert r.status_code == 200
    return r.data


def test_main_page_unchanged_with_teach_off(main_off):
    body = _page(main_off)
    assert hashlib.sha256(body).hexdigest() == PRE_CHANGE_SHA256
    assert b"teach" not in body.lower()


def test_teach_blocks_are_the_only_template_change(main_off):
    source = (REPO / "templates" / "index.html").read_text(encoding="utf-8")
    blocks = TEACH_BLOCK.findall(source)
    assert len(blocks) == 2
    stripped = TEACH_BLOCK.sub("", source)
    assert "teach" not in stripped.lower()
    with main_off.app.app_context():
        rendered = main_off.app.jinja_env.from_string(stripped).render()
    assert rendered.encode("utf-8") == _page(main_off)


def test_main_page_with_teach_on(main_on):
    body = _page(main_on).decode("utf-8")
    assert '<div class="parameter-group" id="teachPanel" hidden>' in body
    toggle = re.search(r'<input type="checkbox" id="teachToggle"[^>]*>', body).group(0)
    assert "checked" not in toggle  # unchecked by default
    assert '<p id="teachSendWrap" hidden>' in body and 'target="_blank"' in body
    assert body.index('src="/teach/static/teach/fragment.js"') < body.index('src="/teach/static/teach/main_teach.js"')
    assert main_on.app.test_client().get("/teach/static/teach/main_teach.js").status_code == 200


def test_main_page_script_only_reads():
    js = MAIN_JS.read_text(encoding="utf-8")
    assert "method" not in js  # every fetch is a plain GET
    urls = re.findall(r'fetch\(\s*"([^"]+)"', js)
    assert urls and all(u.startswith("/teach/api/") for u in urls)
    assert "window.open" not in js and "alert(" not in js  # no automatic pop-ups
    # It shows the checkbox only after the status check returns 200.
    assert 'fetch("/teach/api/status"' in js and "r.status === 200" in js
