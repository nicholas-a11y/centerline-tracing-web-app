"""The Teach form (Stage 5c, 5d): one optional text field and a required title, no focus box on the page, a read-only
"Settings used" line with an Advanced section. The API still accepts focus_box."""

from __future__ import annotations

import re

import pytest

from teach import schema
from teach.validate import Problems, validate_submission

from .conftest import IMAGE, META, REFERENCE, post, submit

LABEL = "What's wrong? (optional)"
HINT = "What you expected, and any notes"


def _validate(**changes):
    import json

    meta = dict(META)
    meta.update(changes)
    return validate_submission(IMAGE, REFERENCE, json.dumps(meta))


def test_single_text_field_in_schema():
    assert list(schema.TEXT_FIELDS) == ["title", "description", "not_duplicate_reason"]
    assert schema.TEXT_FIELDS["description"][:2] == (False, 4000)
    assert schema.TEXT_FIELDS["title"][:2] == (True, 120)
    required = schema.submission_meta_schema()["required"]
    assert required == ["title", "consent"]


def test_description_optional_and_limited():
    import json

    assert _validate(description="   ").fields["description"] == ""
    meta = {k: v for k, v in META.items() if k != "description"}
    assert validate_submission(IMAGE, REFERENCE, json.dumps(meta)).fields["description"] == ""
    with pytest.raises(Problems) as info:
        _validate(title="  ")
    assert [(p.field, p.code) for p in info.value.problems] == [("title", "field_missing")]
    with pytest.raises(Problems) as info:
        _validate(description="x" * 4001)
    assert [(p.field, p.code) for p in info.value.problems] == [("description", "field_too_long")]
    assert _validate(description="x" * 4000).fields["description"] == "x" * 4000


@pytest.mark.parametrize("old", ["what_is_wrong", "what_is_expected", "notes"])
def test_old_text_fields_are_rejected(old):
    with pytest.raises(Problems) as info:
        _validate(**{old: "text"})
    assert [(p.field, p.code) for p in info.value.problems] == [(old, "meta_invalid")]


def test_api_still_accepts_focus_box(teach):
    body, sub_id, _ = submit(teach, focus_box={"x0": 10, "y0": 10, "x1": 100, "y1": 100})
    meta = teach.svc.load_meta(sub_id)
    assert meta["scoring"]["region"] == {"x0": 10.0, "y0": 10.0, "x1": 100.0, "y1": 100.0}
    assert meta["description"] == META["description"]
    assert "what_is_wrong" not in meta and "notes" not in meta


def test_page_has_single_field_and_no_focus_box(teach):
    html = teach.client.get("/teach").get_data(as_text=True)
    assert LABEL in html and HINT in html and 'id="description"' in html and 'maxlength="4000"' in html
    assert not re.search(r'<textarea id="description"[^>]*\brequired\b', html)
    assert re.search(r'<input id="title"[^>]*\brequired\b', html)
    for gone in ("what_is_wrong", "what_is_expected", 'id="notes"', "fx0", "Focus box"):
        assert gone not in html
    form = html[html.index('<form id="case-form"'):html.index("</form>")]
    names = set(re.findall(r'<(?:input|textarea|select)[^>]*\bid="([^"]+)"', form))
    expected = {"image", "reference", "title", "description", "public_thumbnail", "share_bitmap",
                "not_duplicate_reason", "consent"} | {f"s-{n}" for n in schema.SETTINGS}
    assert names == expected


def test_page_settings_line_and_advanced_defaults(teach):
    html = teach.client.get("/teach").get_data(as_text=True)
    line = re.search(r'<span id="settings-used">([^<]*)</span>', html).group(1)
    assert line == schema.settings_summary(schema.default_settings())
    assert "Dark threshold 0.2" in line and "Pruning off" in line
    advanced = html[html.index('<details id="advanced">'):html.index("</details>", html.index('<details id="advanced">'))]
    for name, spec in schema.SETTINGS.items():
        assert f'data-setting="{name}"' in advanced
        if spec["type"] in ("number", "integer"):
            assert re.search(rf'id="s-{name}"[^>]*min="{spec["minimum"]}" max="{spec["maximum"]}"[^>]*value="{spec["default"]}"',
                             advanced), name


def test_consent_and_visibility_unchanged(teach):
    from markupsafe import escape

    html = teach.client.get("/teach").get_data(as_text=True)
    assert str(escape(schema.CONSENT_TEXT)) in html
    assert "Show the overlay thumbnail publicly after review" in html
    assert "Allow the full bitmap to be public after review" in html


def test_submit_with_single_field_over_http(teach):
    r = post(teach.client, "/teach/api/submissions", description="One field only")
    assert r.status_code == 201
