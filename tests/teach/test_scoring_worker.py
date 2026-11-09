"""Scoring worker (design 8): same numbers as the lesson suite, isolated subprocess, clean env."""

from __future__ import annotations

import io
import json

from PIL import Image

from lessonkit.lesson import load_lesson
from teach import scoring_worker
from teach.scoring_worker import ScoringQueue, score_job, worker_env

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
    result, overlay, thumb = score_job(_job(tmp_path, lesson))
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


def test_real_worker_subprocess():
    lesson = load_lesson(LESSON)
    ref = {"paths": [p.tolist() for p in lesson.load_reference()]}
    out = ScoringQueue().run("1.1.1.1", lesson.image_path.read_bytes(), ref, lesson.params, lesson.scoring)
    assert out["outcome"] in ("pass", "fail"), out
    assert set(out["variants"]) == {"prune_off", "prune_on"}
    assert Image.open(io.BytesIO(out["overlay_png"])).format == "PNG"
    assert out["thumb_png"] and out["engine_commit"]


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
