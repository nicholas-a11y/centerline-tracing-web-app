"""The "#s=..." hand-over fragment (Stage 5c): parsed and bounds-checked in the browser by
teach/static/teach/fragment.js, never sent to the server, ignored silently when invalid.

The JavaScript is exercised with node; the tests skip if node is not installed."""

from __future__ import annotations

import base64
import json
import shutil
import subprocess

import pytest

from teach import schema

from .conftest import REPO

FRAGMENT_JS = REPO / "teach" / "static" / "teach" / "fragment.js"
TEACH_JS = REPO / "teach" / "static" / "teach" / "teach.js"
NODE = shutil.which("node")
SHA = "ab" * 32


def specs() -> dict:
    """The same spec teach.js builds from the Advanced inputs (rendered from schema.SETTINGS)."""
    out = {}
    for name, spec in schema.SETTINGS.items():
        if spec["type"] == "string":
            out[name] = {"kind": "string", "options": spec["enum"]}
        else:
            out[name] = {"kind": spec["type"], "min": spec.get("minimum", 0), "max": spec.get("maximum", 0)}
    return out


def encode(doc) -> str:
    raw = doc if isinstance(doc, bytes) else json.dumps(doc).encode("utf-8")
    return "#s=" + base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def good(**changes) -> dict:
    s = schema.default_settings()
    s.update(changes)
    return {"v": 1, "settings": s, "image_sha256": SHA}


def parse_all(hashes: list[str]) -> list:
    if not NODE:
        pytest.skip("node is not installed")
    script = ("const f=require(process.argv[1]);let d='';process.stdin.on('data',c=>d+=c).on('end',()=>{"
              "const a=JSON.parse(d);process.stdout.write(JSON.stringify(a.h.map(h=>f.parseSettingsFragment(h,a.s))));});")
    proc = subprocess.run([NODE, "-e", script, str(FRAGMENT_JS)], input=json.dumps({"h": hashes, "s": specs()}),
                          capture_output=True, text=True, timeout=60, check=True)
    return json.loads(proc.stdout)


def test_valid_fragment_round_trips():
    custom = good(dark_threshold=0.35, min_path_length=7, enable_pruning=True, normalization_mode="off",
                  enable_post_fit_export=True, source_smoothing=40, enable_curve_fitting=True,
                  cubic_fit_tolerance=1.25, force_orthogonal_as_lines=True)
    no_sha = good()
    del no_sha["image_sha256"]
    a, b, c = parse_all([encode(good()), encode(custom), encode(no_sha)])
    assert a == {"settings": schema.default_settings(), "imageSha256": SHA, "crop": None, "sid": None}
    assert b["settings"] == custom["settings"]
    assert c == {"settings": schema.default_settings(), "imageSha256": None, "crop": None, "sid": None}


SID = "0123abcd-0000-4000-8000-00000000beef"
CROP = {"top": 0.1, "bottom": 0.83, "left": 0.2, "right": 0.9}


def test_fragment_with_sid_and_crop():
    with_crop = {**good(), "crop": CROP}
    a, b, c = parse_all([encode(good()) + "&sid=" + SID, encode(with_crop) + "&sid=" + SID, encode(with_crop)])
    assert a == {"settings": schema.default_settings(), "imageSha256": SHA, "crop": None, "sid": SID}
    assert b == {"settings": schema.default_settings(), "imageSha256": SHA, "crop": CROP, "sid": SID}
    assert c["crop"] == CROP and c["sid"] is None


BAD_SID_OR_CROP = {
    "sid uppercase": encode(good()) + "&sid=" + SID.upper(),
    "sid short": encode(good()) + "&sid=" + SID[:-1],
    "sid empty": encode(good()) + "&sid=",
    "sid not uuid": encode(good()) + "&sid=../../admin",
    "sid escaped": encode(good()) + "&sid=" + SID.replace("-", "%2D"),
    "sid twice": encode(good()) + "&sid=" + SID + "&sid=" + SID,
    "sid first": "#sid=" + SID + "&" + encode(good())[1:],
    "sid without sha": encode({k: v for k, v in good().items() if k != "image_sha256"}) + "&sid=" + SID,
    "crop missing key": encode({**good(), "crop": {k: v for k, v in CROP.items() if k != "left"}}),
    "crop extra key": encode({**good(), "crop": {**CROP, "x": 0}}),
    "crop out of range": encode({**good(), "crop": {**CROP, "right": 1.2}}),
    "crop negative": encode({**good(), "crop": {**CROP, "top": -0.1}}),
    "crop string": encode({**good(), "crop": {**CROP, "top": "0.1"}}),
    "crop array": encode({**good(), "crop": [0.1, 0.8, 0.2, 0.9]}),
    "crop null": encode({**good(), "crop": None}),
}


@pytest.mark.parametrize("label", list(BAD_SID_OR_CROP))
def test_invalid_sid_or_crop_is_ignored(label):
    assert parse_all([BAD_SID_OR_CROP[label]]) == [None]


MALFORMED = {
    "empty": "",
    "no prefix": "#x=abc",
    "not base64url": "#s=abc+/=",
    "not json": "#s=" + base64.urlsafe_b64encode(b"{nope").decode().rstrip("="),
    "bad utf8": encode(b"\xff\xfe{}"),
    "json array": encode([1, 2]),
    "json string": encode("x"),
    "trailing junk": encode(good()) + "&x=1",
    "oversized": "#s=" + "A" * 2100,
}


@pytest.mark.parametrize("label", list(MALFORMED))
def test_malformed_or_oversized_is_ignored(label):
    assert parse_all([MALFORMED[label]]) == [None]


def _without(key):
    d = good()
    del d["settings"][key]
    return d


BAD_DOCS = {
    "wrong version": {**good(), "v": 2},
    "no version": {k: v for k, v in good().items() if k != "v"},
    "extra top key": {**good(), "extra": 1},
    "extra setting": {**good(), "settings": {**good()["settings"], "merge_gap": 5}},
    "missing setting": _without("source_smoothing"),
    "settings not object": {**good(), "settings": [1]},
    "bad sha": {**good(), "image_sha256": "XYZ"},
    "short sha": {**good(), "image_sha256": "ab" * 31},
    "sha not string": {**good(), "image_sha256": 5},
    "dark_threshold low": good(dark_threshold=0.01),
    "dark_threshold high": good(dark_threshold=0.81),
    "min_path_length fraction": good(min_path_length=2.5),
    "min_path_length high": good(min_path_length=21),
    "source_smoothing high": good(source_smoothing=101),
    "source_smoothing negative": good(source_smoothing=-1),
    "cubic low": good(cubic_fit_tolerance=0.3),
    "cubic high": good(cubic_fit_tolerance=4.5),
    "number as string": good(dark_threshold="0.2"),
    "bool as number": good(enable_pruning=1),
    "bool as string": good(enable_post_fit_export="true"),
    "unknown enum": good(normalization_mode="sometimes"),
    "null value": good(cubic_fit_tolerance=None),
}


@pytest.mark.parametrize("label", list(BAD_DOCS))
def test_invalid_content_is_ignored(label):
    assert parse_all([encode(BAD_DOCS[label])]) == [None]


def test_page_loads_fragment_parser_without_inline_script(teach):
    html = teach.client.get("/teach").get_data(as_text=True)
    scripts = [s for s in html.split("<script")[1:]]
    assert len(scripts) == 3 and all(s.lstrip().startswith("src=") for s in scripts)
    assert html.index("teach/fragment.js") < html.index("teach/teach.js")
    assert teach.client.get("/teach/static/teach/fragment.js").status_code == 200
    assert teach.client.get("/teach/static/teach/preattach.js").status_code == 200


def test_fragment_is_never_sent():
    js = TEACH_JS.read_text(encoding="utf-8")
    # The hash is read once, for parsing, and is not put in any request (it is then removed
    # from the address bar; see test_preattach.py).
    assert js.count("location.hash") == 1
    assert "parseSettingsFragment(window.location.hash" in js
    assert "imageSha256" in js and "fd.append(\"sha" not in js
    meta_fn = js[js.index("function buildMeta"):js.index("function buildBody")]
    assert "expectedSha" not in meta_fn and "handover" not in meta_fn


def test_mixup_warning_is_friendly_and_never_blocks(teach):
    html = teach.client.get("/teach").get_data(as_text=True)
    assert '<p id="mixup-warning" class="warn" role="status" hidden>This doesn\'t look like the cropped image from the app.</p>' in html
    js = TEACH_JS.read_text(encoding="utf-8")
    assert 'crypto.subtle.digest("SHA-256"' in js
    submit_handler = js[js.index('getElementById("btn-submit")'):js.index('form.addEventListener("submit"')]
    assert "mixup" not in submit_handler and "expectedSha" not in submit_handler


def test_page_text_for_preparing_a_case(teach):
    html = teach.client.get("/teach").get_data(as_text=True)
    for text in ("use the crop tools to crop to the problem area", "leaving a little context",
                 "trace over that original (cropped) image", "Keep the image in the exported SVG"):
        assert text in html, text


def test_build_fragment_round_trips_and_matches_python_encoding():
    """The main app's encoder (buildSettingsFragment) produces what the Teach page parses."""
    if not NODE:
        pytest.skip("node is not installed")
    settings = dict(schema.default_settings(), dark_threshold=0.35, normalization_mode="off", source_smoothing=40)
    script = ("const f=require(process.argv[1]);let d='';process.stdin.on('data',c=>d+=c).on('end',()=>{"
              "const a=JSON.parse(d);const h=f.buildSettingsFragment(a.s,a.sha);"
              "process.stdout.write(JSON.stringify({h:h,p:f.parseSettingsFragment(h,a.specs)}));});")
    proc = subprocess.run([NODE, "-e", script, str(FRAGMENT_JS)],
                          input=json.dumps({"s": settings, "sha": SHA, "specs": specs()}),
                          capture_output=True, text=True, timeout=60, check=True)
    out = json.loads(proc.stdout)
    assert out["p"] == {"settings": settings, "imageSha256": SHA, "crop": None, "sid": None}
    raw = out["h"][3:]
    doc = json.loads(base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)))
    assert doc == {"v": 1, "settings": settings, "image_sha256": SHA}
    assert len(out["h"]) < 2048


def test_build_fragment_with_crop_and_sid_round_trips():
    if not NODE:
        pytest.skip("node is not installed")
    sid = "0123abcd-0000-4000-8000-00000000beef"
    crop = {"top": 0.123456789, "bottom": 0.9, "left": 0.0019, "right": 1}
    script = ("const f=require(process.argv[1]);let d='';process.stdin.on('data',c=>d+=c).on('end',()=>{"
              "const a=JSON.parse(d);const h=f.buildSettingsFragment(a.s,a.sha,a.crop,a.sid);"
              "process.stdout.write(JSON.stringify({h:h,p:f.parseSettingsFragment(h,a.specs)}));});")
    proc = subprocess.run([NODE, "-e", script, str(FRAGMENT_JS)],
                          input=json.dumps({"s": schema.default_settings(), "sha": SHA, "crop": crop, "sid": sid,
                                            "specs": specs()}),
                          capture_output=True, text=True, timeout=60, check=True)
    out = json.loads(proc.stdout)
    assert out["h"].endswith("&sid=" + sid) and len(out["h"]) < 2048
    assert out["p"] == {"settings": schema.default_settings(), "imageSha256": SHA, "crop": crop, "sid": sid}
