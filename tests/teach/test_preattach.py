"""Pre-attaching the cropped bitmap on the Teach page (Stage 5d A).

The main app's "Send this to Teach" link carries "#s=...&sid=...". The Teach page parses it,
removes it from the address bar, fetches GET /teach/api/crop/<sid>/image.png (read-only),
checks its sha-256 against the fragment and offers it as the attached bitmap. These tests run
the real fragment.js, preattach.js and teach.js under node against a minimal fake DOM
(tests/teach/js/teach_page_harness.js), with the crop route served by the real app. They skip
if node is not installed."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import shutil
import subprocess

import pytest
from PIL import Image

from teach import crop, schema

from .conftest import REPO, png_bytes
from .test_crop import LESSON_IMAGE, REGION, _crop_info, _crop_png, _process, _upload

NODE = shutil.which("node")
HARNESS = REPO / "tests" / "teach" / "js" / "teach_page_harness.js"
FRAGMENT_JS = REPO / "teach" / "static" / "teach" / "fragment.js"
PREATTACH_JS = REPO / "teach" / "static" / "teach" / "preattach.js"
TEACH_JS = REPO / "teach" / "static" / "teach" / "teach.js"
UNAVAILABLE = ("The app's copy of this image is no longer available. "
               "Attach the downloaded cropped image instead.")

pytestmark = pytest.mark.skipif(not NODE, reason="node is not installed")


def _settings_spec() -> list[dict]:
    out = []
    for name, spec in schema.SETTINGS.items():
        kind = spec["type"]
        out.append({"name": name, "kind": kind, "default": spec["default"], "min": spec.get("minimum"),
                    "max": spec.get("maximum"), "options": spec.get("enum")})
    return out


def _node(script: str, payload: dict):
    proc = subprocess.run([NODE, "-e", script, str(FRAGMENT_JS), str(PREATTACH_JS)], input=json.dumps(payload),
                          capture_output=True, text=True, encoding="utf-8", timeout=60, check=True)
    return json.loads(proc.stdout)


def build_hash(sha: str | None, crop_region: dict | None = None, sid: str | None = None,
               settings: dict | None = None) -> str:
    script = ("const f=require(process.argv[1]);let d='';process.stdin.on('data',c=>d+=c).on('end',()=>{"
              "const a=JSON.parse(d);process.stdout.write(JSON.stringify(f.buildSettingsFragment(a.s,a.sha,a.crop,a.sid)));});")
    return _node(script, {"s": settings or schema.default_settings(), "sha": sha, "crop": crop_region, "sid": sid})


def crop_url(sid: str, crop_region: dict | None) -> str:
    script = ("const p=require(process.argv[2]);let d='';process.stdin.on('data',c=>d+=c).on('end',()=>{"
              "const a=JSON.parse(d);process.stdout.write(JSON.stringify(p.cropImageUrl(a.sid,a.crop)));});")
    return _node(script, {"sid": sid, "crop": crop_region})


def run_page(hash_: str, crop_responses: dict | None = None, steps=(), subtle: bool = True) -> list[dict]:
    payload = {"root": str(REPO), "hash": hash_, "settings": _settings_spec(), "crop": crop_responses or {},
               "subtle": subtle, "steps": list(steps)}
    proc = subprocess.run([NODE, str(HARNESS)], input=json.dumps(payload), capture_output=True, text=True,
                          encoding="utf-8", timeout=60)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def served(client, url: str) -> dict:
    r = client.get(url)
    return {"status": r.status_code, "b64": base64.b64encode(r.data).decode("ascii"),
            "disposition": r.headers.get("Content-Disposition")}


def image_entry(form: list[dict] | None):
    return next((e for e in form or [] if e["key"] == "image"), None)


@pytest.fixture
def cropped_run(main_on):
    """A real cropped run in the main app: (client, sid, crop info, downloaded crop bytes)."""
    c = main_on.app.test_client()
    sid = _upload(c, LESSON_IMAGE.read_bytes())
    _process(main_on, c, sid, REGION)
    info = _crop_info(c, sid, REGION)
    assert info["equivalent"] is True
    yield c, sid, info, _crop_png(c, sid, REGION)
    dict.pop(main_on.sessions, sid, None)


# ------------------------------------------------------------------ success


def test_preattach_success_equals_downloaded_crop(cropped_run):
    c, sid, info, download = cropped_run
    url = crop_url(sid, REGION)
    resp = served(c, url)
    assert resp["status"] == 200 and base64.b64decode(resp["b64"]) == download

    load, validate = run_page(build_hash(info["image_sha256"], REGION, sid), {url: resp}, [{"do": "validate"}])
    assert load["replaceCalls"] == ["/teach"] and load["hash"] == ""
    assert load["fetchUrls"] == ["GET " + url]
    assert load["events"][0] == "replaceState /teach"  # removed before the fetch
    assert load["settingsFromApp"] is True
    assert load["panelHidden"] is False and load["pickHidden"] is True and load["status"] is None
    w, h = info["width"], info["height"]
    assert load["name"] == f"cropped-{w}x{h}.png" and load["size"] == f"{w} × {h} px"
    assert load["preview"] == "data:image/png;base64," + base64.b64encode(download).decode("ascii")

    entry = image_entry(validate["form"])
    assert entry and base64.b64decode(entry["b64"]) == download and entry["name"] == f"cropped-{w}x{h}.png"
    assert hashlib.sha256(base64.b64decode(entry["b64"])).hexdigest() == info["image_sha256"]
    # The session id is only ever in the path of the crop route, never in a submitted request.
    assert validate["fetchUrls"][-1] == "POST /teach/api/validate"
    assert all(sid not in u for u in validate["fetchUrls"][1:])
    assert all(sid not in e.get("text", "") for e in validate["form"])


# ------------------------------------------------------------------ fallbacks


def test_expired_session_falls_back_to_manual_attach(main_on, cropped_run):
    c, sid, info, _download = cropped_run
    url = crop_url(sid, REGION)
    dict.pop(main_on.sessions, sid)  # expired, evicted or the app restarted
    resp = served(c, url)
    assert resp["status"] == 404
    for responses in ({url: resp}, {url: "reject"}):
        load, validate = run_page(build_hash(info["image_sha256"], REGION, sid), responses, [{"do": "validate"}])
        assert load["status"] == UNAVAILABLE and load["statusClass"] == "error"
        assert load["panelHidden"] is True and load["pickHidden"] is False
        assert load["replaceCalls"] == ["/teach"]
        assert image_entry(validate["form"]) is None  # nothing attached; the user attaches the download


def test_mismatch_falls_back_with_an_error(cropped_run):
    c, sid, info, download = cropped_run
    url = crop_url(sid, REGION)
    wrong = hashlib.sha256(download + b"x").hexdigest()
    (load,) = run_page(build_hash(wrong, REGION, sid), {url: served(c, url)})
    assert load["statusClass"] == "error" and "not the one you sent" in load["status"]
    assert load["status"].endswith("Attach the downloaded cropped image instead.")
    assert load["panelHidden"] is True and load["pickHidden"] is False


def test_no_subtle_crypto_falls_back(cropped_run):
    c, sid, info, _download = cropped_run
    url = crop_url(sid, REGION)
    (load,) = run_page(build_hash(info["image_sha256"], REGION, sid), {url: served(c, url)}, subtle=False)
    assert load["statusClass"] == "error" and load["panelHidden"] is True and load["fetchUrls"] == []


def test_replace_then_manual_attach(cropped_run):
    c, sid, info, download = cropped_run
    url = crop_url(sid, REGION)
    other = png_bytes(40, 40, 200)
    steps = [{"do": "replace"}, {"do": "validate"},
             {"do": "pick", "name": "mine.png", "b64": base64.b64encode(other).decode("ascii")}, {"do": "validate"}]
    load, replaced, v1, picked, v2 = run_page(build_hash(info["image_sha256"], REGION, sid), {url: served(c, url)}, steps)
    assert load["panelHidden"] is False
    assert replaced["panelHidden"] is True and replaced["pickHidden"] is False
    assert replaced["inputClicks"] == 1 and replaced["focused"] == "image"
    assert image_entry(v1["form"]) is None  # the pre-attached file is gone
    assert picked["mixupHidden"] is False  # the existing mix-up warning still works for manual files
    entry = image_entry(v2["form"])
    assert entry["name"] == "mine.png" and base64.b64decode(entry["b64"]) == other


def test_manual_attach_with_matching_file_has_no_warning(cropped_run):
    """Without sid (a fragment from before pre-attach) the page still pre-fills settings and checks manual files."""
    _c, _sid, info, download = cropped_run
    steps = [{"do": "pick", "name": "cropped.png", "b64": base64.b64encode(download).decode("ascii")}]
    load, picked = run_page(build_hash(info["image_sha256"], REGION), {}, steps)
    assert load["fetchUrls"] == [] and load["status"] is None and load["panelHidden"] is True
    assert load["settingsFromApp"] is True and load["replaceCalls"] == ["/teach"]
    assert picked["mixupHidden"] is True


# ------------------------------------------------------------------ the fragment and the address bar


@pytest.mark.parametrize("hash_,stripped", [
    ("", False),
    ("#main", False),                      # the skip link; not a hand-over fragment
    ("#s=not-valid", True),                # invalid hand-over fragments are removed too
    ("#s=abc&sid=00000000-0000-0000-0000-000000000000", True),
])
def test_fragment_removed_from_address_bar(hash_, stripped):
    (load,) = run_page(hash_)
    assert load["replaceCalls"] == (["/teach"] if stripped else [])
    assert load["fetchUrls"] == [] and load["settingsFromApp"] is False and load["status"] is None


def test_with_and_without_sid(cropped_run):
    c, sid, info, _download = cropped_run
    url = crop_url(sid, REGION)
    (with_sid,) = run_page(build_hash(info["image_sha256"], REGION, sid), {url: served(c, url)})
    (without,) = run_page(build_hash(info["image_sha256"], REGION))
    assert with_sid["fetchUrls"] == ["GET " + url] and with_sid["panelHidden"] is False
    assert without["fetchUrls"] == [] and without["panelHidden"] is True and without["settingsFromApp"] is True


def test_uncropped_run_fetches_without_a_query(main_on):
    c = main_on.app.test_client()
    sid = _upload(c, LESSON_IMAGE.read_bytes())
    try:
        _process(main_on, c, sid, None)
        info = _crop_info(c, sid, None)
        url = crop_url(sid, None)
        assert url == f"/teach/api/crop/{sid}/image.png"
        (load,) = run_page(build_hash(info["image_sha256"], None, sid), {url: served(c, url)})
        assert load["panelHidden"] is False and load["size"] == "{} × {} px".format(*Image.open(LESSON_IMAGE).size)
    finally:
        dict.pop(main_on.sessions, sid, None)


# ------------------------------------------------------------------ static checks and logging


def test_page_has_preattach_elements(teach):
    html = teach.client.get("/teach").get_data(as_text=True)
    for snippet in ('<div id="preattach" hidden>', '<button type="button" id="preattach-replace">Replace</button>',
                    '<p id="preattach-status" role="status" hidden></p>', 'id="preattach-preview"',
                    'teach/preattach.js'):
        assert snippet in html, snippet
    assert html.index("teach/fragment.js") < html.index("teach/preattach.js") < html.index("teach/teach.js")


def test_sid_is_never_put_in_a_submitted_request():
    js = TEACH_JS.read_text(encoding="utf-8")
    body = js[js.index("function buildMeta"):js.index("function problemList")]
    assert "sid" not in body and "handover" not in body
    assert js.index("stripFragment(") < js.index("fetchCropped(")
    pre = PREATTACH_JS.read_text(encoding="utf-8")
    assert '"/teach/api/crop/" + encodeURIComponent(sid) + "/image.png"' in pre
    for path in (TEACH_JS, PREATTACH_JS, FRAGMENT_JS):
        assert "console." not in path.read_text(encoding="utf-8")


def test_request_logs_redact_the_session_id(main_on, caplog):
    sid = "0123abcd-0000-4000-8000-00000000beef"
    record = logging.LogRecord("werkzeug", logging.INFO, __file__, 1, '%s - - "%s" %s -',
                               ("127.0.0.1", f"GET /teach/api/crop/{sid}/image.png?top=0.1 HTTP/1.1", "200"), None)
    assert logging.getLogger("werkzeug").filter(record)
    assert sid not in record.getMessage() and "/teach/api/crop/<sid>/image.png?top=0.1" in record.getMessage()
    access = logging.LogRecord("gunicorn.access", logging.INFO, __file__, 1,
                               f'1.2.3.4 "GET /teach/api/crop/{sid} HTTP/1.1" 404', None, None)
    assert logging.getLogger("gunicorn.access").filter(access) and sid not in access.getMessage()
    # Teach's own code never logs it either.
    with caplog.at_level(logging.DEBUG):
        main_on.app.test_client().get(f"/teach/api/crop/{sid}/image.png")
    assert all(sid not in r.getMessage() for r in caplog.records)


def test_redaction_filter_installed_once(main_on):
    crop.install_log_redaction()
    crop.install_log_redaction()
    filters = [f for f in logging.getLogger("werkzeug").filters if isinstance(f, crop.RedactCropSessionIds)]
    assert len(filters) == 1