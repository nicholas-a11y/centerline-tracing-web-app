"""The post_fit variant (Stage 5c): fitted geometry from the same route as the app's Download
SVG button, read back with the safe reader, reported with its own overlay."""

from __future__ import annotations

import json

import numpy as np
import pytest

from lessonkit.lesson import load_lesson
from lessonkit.runner import run_live, run_post_fit
from teach.service import public_result
from teach.storage import submission_key
from teach.validate import effective_params

from .conftest import LESSON, THUMB, FakeQueue, post, submit


def _densify(path: np.ndarray, step: float = 0.25) -> np.ndarray:
    pts = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        pts.extend(a + (b - a) * t for t in np.linspace(0, 1, n + 1)[1:])
    return np.asarray(pts)


@pytest.mark.parametrize("curves,prune", [(False, False), (True, False), (True, True)])
def test_run_post_fit_follows_the_extracted_paths(curves, prune):
    lesson = load_lesson(LESSON)
    params = effective_params({"enable_post_fit_export": True, "enable_curve_fitting": curves,
                               "enable_pruning": prune})
    fitted = run_post_fit(lesson.image_path, params)
    raw = run_live(lesson.image_path, params)
    assert fitted.error is None and fitted.route == "download_svg"
    assert fitted.image_shape == raw.image_shape
    assert len(fitted.paths) == len(raw.paths) == fitted.counts["svg_paths"]
    dense = np.vstack([_densify(p) for p in raw.paths])
    for f, r in zip(fitted.paths, raw.paths):
        # Same orientation and endpoints (writer +0.5 offset undone by the reader). Smoothing
        # without curve fitting moves junction ends by up to ~2.3 px (app behaviour, recorded here).
        assert np.allclose(f[0], r[0], atol=3.0) and np.allclose(f[-1], r[-1], atol=3.0)
        d = np.sqrt(((f[:, None, :] - dense[None, :, :]) ** 2).sum(-1)).min(1)
        assert d.max() < 3.0  # source smoothing (70) moves staircase points by at most a few px


class PostFitQueue(FakeQueue):
    def run(self, ip, image_png, reference_json, params, scoring):
        out = super().run(ip, image_png, reference_json, params, scoring)
        if params.get("enable_post_fit_export"):
            out["variants"]["post_fit"] = {"summary": {"path_count": 5, "passed": False}, "lines": []}
            out["post_fit_overlay_png"] = THUMB
        else:
            out["post_fit_overlay_png"] = None
        return out


def test_public_result_includes_post_fit_only_when_present():
    worker = {"outcome": "fail", "variants": {"prune_off": {"lines": []}, "post_fit": {"summary": {"x": 1}}}}
    with_pf = public_result(worker, "size_match", "data:o", "data:p")
    assert with_pf["post_fit_overlay_png"] == "data:p" and with_pf["other_variants"]["post_fit"] == {"x": 1}
    assert with_pf["variant"] == "prune_off"
    assert "post_fit_overlay_png" not in public_result(worker, "size_match", "data:o")


def test_validate_and_submit_with_post_fit(make_teach):
    t = make_teach(queue=PostFitQueue())
    settings = {"enable_post_fit_export": True}
    r = post(t.client, "/teach/api/validate", settings=settings)
    res = r.get_json()["result"]
    assert res["post_fit_overlay_png"].startswith("data:image/png;base64,")
    assert "post_fit" in res["other_variants"]
    body, sub_id, token = submit(t, settings=settings)
    assert body["result"]["post_fit_overlay_png"].startswith("data:image/png;base64,")
    assert t.storage.get(submission_key(sub_id, "overlay_post_fit.png")) == THUMB
    meta = t.svc.load_meta(sub_id)
    json.dumps(meta)  # no raw bytes leak into stored metadata
    assert "post_fit" in meta["results"][0]["metrics"]
    assert meta["params"]["enable_post_fit_export"] is True
    view = t.client.get(f"/teach/api/manage/{sub_id}", headers={"X-Teach-Manage-Token": token}).get_json()["result"]
    assert view["post_fit_overlay_png"].startswith("data:image/png;base64,")


def test_post_fit_off_changes_nothing(make_teach):
    t = make_teach(queue=PostFitQueue())
    res = post(t.client, "/teach/api/validate").get_json()["result"]
    assert "post_fit_overlay_png" not in res and "post_fit" not in res["other_variants"]
    body, sub_id, _ = submit(t)
    assert "post_fit_overlay_png" not in body["result"]
    assert t.storage.get(submission_key(sub_id, "overlay_post_fit.png")) is None
