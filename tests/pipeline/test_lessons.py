"""Every lesson in tests/lessons/ runs through the live pipeline (/upload -> /process_immediate).

status "active"   -> must pass
status "baseline" -> non-strict xfail (reports current behaviour; XPASS is listed at the end)
status "draft"    -> skipped (reference not trusted yet)
"""

from __future__ import annotations

import pytest

from tests.pipeline.evaluate import PRIMARY_VARIANT, evaluate
from tests.pipeline.lesson import discover_lessons, validate_meta

LESSONS = discover_lessons()


def _param(lesson):
    marks = []
    if lesson.status == "baseline":
        marks.append(pytest.mark.xfail(reason="baseline: fails with the current engine", strict=False))
    elif lesson.status == "draft":
        marks.append(pytest.mark.skip(reason="draft: reference not trusted yet (see lesson notes)"))
    return pytest.param(lesson, id=lesson.id, marks=marks)


def test_lessons_exist():
    assert LESSONS, "no lessons found in tests/lessons/"


@pytest.mark.parametrize("lesson", [pytest.param(ls, id=ls.id) for ls in LESSONS])
def test_lesson_metadata_valid(lesson):
    assert validate_meta(lesson.meta, lesson.folder) == []
    assert lesson.load_reference(), "reference.svg has no geometry"


@pytest.mark.parametrize("lesson", [_param(ls) for ls in LESSONS])
def test_lesson(lesson):
    ev = evaluate(lesson)
    prim = ev.variants[PRIMARY_VARIANT]
    assert prim.result.error is None, prim.result.error
    reasons = [f"line {ln.index}: {ln.reason}" for ln in prim.score.lines if not ln.passed]
    assert prim.score.passed, f"{lesson.id}: " + "; ".join(reasons or prim.score.notes)
