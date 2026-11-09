"""The trace template was removed in Stage 5c: no route, no module, no page link, no result hint."""

from __future__ import annotations

import importlib.util
import io

from .conftest import IMAGE, REPO, post


def test_trace_template_route_is_gone(teach):
    r = teach.client.post("/teach/api/trace-template", data={"image": (io.BytesIO(IMAGE), "a.png")},
                          content_type="multipart/form-data")
    assert r.status_code == 404
    assert teach.client.get("/teach/api/trace-template").status_code == 404
    assert not [rule for rule in teach.app.url_map.iter_rules() if "template" in rule.rule]


def test_trace_template_module_is_gone():
    assert importlib.util.find_spec("lessonkit.trace_template") is None
    assert not (REPO / "tests" / "pipeline" / "trace_template.py").exists()


def test_no_template_mentions_in_page_spec_or_results(teach):
    page = teach.client.get("/teach").get_data(as_text=True).lower()
    spec = teach.client.get("/teach/spec").get_data(as_text=True).lower()
    api = teach.client.get("/teach/openapi.json").get_data(as_text=True).lower()
    for text in (page, spec, api):
        assert "trace template" not in text and "trace-template" not in text
    body = post(teach.client, "/teach/api/validate").get_json()
    assert "trace_template_url" not in body["result"]
