"""Stage 5d B: users see "hand traced SVG", never "reference SVG". The machine field stays `reference`."""

from __future__ import annotations

import json
import re

import pytest

from teach import schema
from teach.validate import Problems, validate_reference, validate_submission

from .conftest import IMAGE, META, REFERENCE, REPO, post, submit

TEACH = REPO / "teach"
OLD = re.compile(r"reference SVG|hand-traced|your reference|the reference|variant image and reference|"
                 r"image \+ reference", re.IGNORECASE)
HELP = "Attach the SVG you drew by hand over the bitmap."


def _no_old_wording(text: str, where: str) -> None:
    hit = OLD.search(text)
    assert hit is None, f"{where}: {text[max(0, hit.start() - 40):hit.end() + 40]!r}"


def test_page_label_and_help(teach):
    html = teach.client.get("/teach").get_data(as_text=True)
    assert '<label for="reference">Hand traced SVG</label>' in html
    assert re.search(r'<span id="reference-help" class="help">' + re.escape(HELP), html)
    assert re.search(r'<input id="reference" name="reference"[^>]*aria-describedby="reference-help"', html)
    _no_old_wording(html, "/teach")


def test_spec_names_the_field(teach):
    text = teach.client.get("/teach/spec").get_data(as_text=True)
    assert "| `reference` | yes | The hand traced SVG (the field keeps the name `reference`)" in text
    assert "## The hand traced SVG" in text and "**hand traced SVG**" in text
    _no_old_wording(text, "/teach/spec")


def test_openapi_keeps_field_and_renames_text(teach):
    doc = teach.client.get("/teach/openapi.json").get_json()
    form = doc["paths"]["/teach/api/submissions"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
    assert "reference" in form["properties"] and "reference" in form["required"]
    assert form["properties"]["reference"]["description"].startswith("The hand traced SVG")
    _no_old_wording(json.dumps(doc), "openapi.json")
    _no_old_wording(json.dumps(teach.client.get("/teach/schema/submission.json").get_json()), "schema")


def test_rules_and_templates_and_scripts():
    for rule in schema.RULES:
        _no_old_wording(rule, "RULES")
    assert any(r.startswith("The hand traced SVG must be registered") for r in schema.RULES)
    for path in list((TEACH / "templates" / "teach").glob("*.html")) + list((TEACH / "static" / "teach").glob("*.js")):
        _no_old_wording(path.read_text(encoding="utf-8"), path.name)
    _no_old_wording((TEACH / "spec.md").read_text(encoding="utf-8"), "spec.md")


def test_manage_page_wording(teach):
    _, sub_id, _ = submit(teach)
    html = teach.client.get(f"/teach/manage/{sub_id}").get_data(as_text=True)
    assert "answer-form" not in html
    assert "your hand traced SVG (green)" in html
    _no_old_wording(html, "manage page")


def test_missing_svg_messages():
    with pytest.raises(Problems) as info:
        validate_reference(b"", (256, 256))
    (p,) = info.value.problems
    assert (p.field, p.code, p.message) == ("reference", "field_missing", "No hand traced SVG was uploaded.")
    assert p.hint.startswith("Send the hand traced SVG as 'reference'")
    with pytest.raises(Problems) as info:
        validate_submission(IMAGE, b"", json.dumps(META))
    assert [(p.field, p.code, p.message) for p in info.value.problems] == [
        ("reference", "field_missing", "No hand traced SVG was uploaded.")]


def test_registration_hints_have_no_old_wording():
    from teach.validate import _SVG_HINTS

    for code, hint in _SVG_HINTS.items():
        _no_old_wording(hint, code)


def test_manage_has_no_answer_method(teach):
    _, sub_id, _ = submit(teach)
    assert not hasattr(teach.svc, "manage_answer")
    assert teach.client.patch(f"/teach/api/manage/{sub_id}").status_code == 405


def test_api_still_accepts_reference_field(teach):
    r = post(teach.client, "/teach/api/validate")
    assert r.status_code == 200 and r.get_json()["ok"] is True
    r = post(teach.client, "/teach/api/validate", reference=None)
    body = r.get_json()
    assert body["errors"][0]["field"] == "reference"
    assert body["errors"][0]["message"] == "No hand traced SVG was uploaded."
    body, _, _ = submit(teach)
    assert body["number"] >= 1
