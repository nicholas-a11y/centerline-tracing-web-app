"""Scoring worker (design 8): same numbers as the lesson suite, isolated subprocess, clean env."""

from __future__ import annotations

import io
import json

from PIL import Image

from lessonkit.lesson import load_lesson
from teach import scoring_worker
from teach.scoring_worker import ScoringQueue, score_job, worker_env
from teach.validate import effective_params

from .conftest import LESSON

COMPARED = ("error", "timed_out", "counts", "graph", "summary", "lines")


def _job(tmp_path, lesson):
    ref = [p.tolist() for p in lesson.load_reference()]
    (tmp_path / "reference.json").write_text(json.dumps({"paths": ref}), encoding="utf-8")
    return {"image": str(lesson.image_path), "reference": str(tmp_path / "reference.json"),
            "params": lesson.params, "scoring": lesson.scoring}


def test_score_job_matches_lesson_suite(tmp_path):
    from tests.pipeline.evaluate import evaluate, evaluation_record

    lesson = load_lesson(LESSON)
    result, overlay, thumb, post_overlay = score_job(_job(tmp_path, lesson))
    expected = evaluation_record(evaluate(lesson))
    assert set(result["variants"]) == set(expected["variants"])
    for name, rec in expected["variants"].items():
        got = result["variants"][name]
        for key in COMPARED:
            assert got.get(key) == rec.get(key), (name, key)
    assert result["outcome"] == ("pass" if expected["outcome"] in ("pass", "UNEXPECTED PASS") else "fail")
    w, h = Image.open(lesson.image_path).size
    assert overlay.size[0] % w == 0 and overlay.size[0] // w == overlay.size[1] // h
    assert max(thumb.size) <= scoring_worker.THUMB_MAX_PX
    assert post_overlay is None


def _compared(result):
    return {name: {k: rec.get(k) for k in COMPARED} for name, rec in result["variants"].items()}


def test_all_ten_default_settings_score_exactly_as_before(tmp_path):
    """Teach now passes all ten settings explicitly; with post-fit off the result is unchanged."""
    lesson = load_lesson(LESSON)
    before, *_ = score_job(_job(tmp_path, lesson))
    job = _job(tmp_path, lesson)
    job["params"] = effective_params({})
    assert len(job["params"]) == 10
    after, _, _, post_overlay = score_job(job)
    assert set(after["variants"]) == {"prune_off", "prune_on"}
    assert _compared(after) == _compared(before)
    assert after["outcome"] == before["outcome"] and after["summary"] == before["summary"]
    assert post_overlay is None


def test_post_fit_variant_when_enabled(tmp_path):
    lesson = load_lesson(LESSON)
    off, *_ = score_job(_job(tmp_path, lesson))
    job = _job(tmp_path, lesson)
    job["params"] = effective_params({"enable_post_fit_export": True, "enable_curve_fitting": True})
    on, overlay, _, post_overlay = score_job(job)
    assert set(on["variants"]) == {"prune_off", "prune_on", "post_fit"}
    # The existing variants and the outcome are untouched by the extra variant.
    assert {k: v for k, v in _compared(on).items() if k != "post_fit"} == _compared(off)
    assert on["outcome"] == off["outcome"] and on["variant"] == "prune_off"
    pf = on["variants"]["post_fit"]
    assert pf["error"] is None and set(pf) == set(on["variants"]["prune_off"])
    assert pf["counts"]["svg_paths"] == pf["summary"]["path_count"] == 5
    assert len(pf["lines"]) == len(lesson.load_reference())
    assert post_overlay is not None and post_overlay.size == overlay.size


def test_real_worker_subprocess():
    lesson = load_lesson(LESSON)
    ref = {"paths": [p.tolist() for p in lesson.load_reference()]}
    out = ScoringQueue().run("1.1.1.1", lesson.image_path.read_bytes(), ref, lesson.params, lesson.scoring)
    assert out["outcome"] in ("pass", "fail"), out
    assert set(out["variants"]) == {"prune_off", "prune_on"}
    assert Image.open(io.BytesIO(out["overlay_png"])).format == "PNG"
    assert out["thumb_png"] and out["engine_commit"]
    assert out["post_fit_overlay_png"] is None
    assert out["app_centerline_svg"].startswith(b'<?xml version="1.0"')
    assert b'<svg xmlns="http://www.w3.org/2000/svg"' in out["app_centerline_svg"]


def test_real_worker_subprocess_post_fit():
    lesson = load_lesson(LESSON)
    ref = {"paths": [p.tolist() for p in lesson.load_reference()]}
    params = effective_params({"enable_post_fit_export": True})
    out = ScoringQueue().run("1.1.1.1", lesson.image_path.read_bytes(), ref, params, lesson.scoring)
    assert set(out["variants"]) == {"prune_off", "prune_on", "post_fit"}, out
    assert Image.open(io.BytesIO(out["post_fit_overlay_png"])).format == "PNG"


def test_worker_error_is_reported(tmp_path):
    out = ScoringQueue().run("1.1.1.1", b"not a png", {"paths": [[[0, 0], [1, 1]]]}, {}, {})
    assert out["outcome"] == "error" and out["overlay_png"] is None


def test_worker_env_strips_secrets(monkeypatch):
    for name in ("TEACH_ADMIN_TOKEN", "R2_SECRET_ACCESS_KEY", "R2_ACCESS_KEY_ID", "TEACH_ALERT_WEBHOOK_URL"):
        monkeypatch.setenv(name, "secret-value")
    monkeypatch.setenv("TEACH_ENABLED", "1")
    env = worker_env()
    assert env["TEACH_ENABLED"] == "0"
    assert "secret-value" not in json.dumps(env)
