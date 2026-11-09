"""The ten-setting whitelist (Stage 5c): same ranges and defaults as the main UI, always
passed to the engine explicitly, and additive in the lesson format."""

from __future__ import annotations

import json
import re

import pytest

from lessonkit.lesson import ALLOWED_PARAMS, DEFAULT_PARAMS, OPTIONAL_PARAMS, discover_lessons, validate_meta
from lessonkit.runner import UI_DEFAULT_PARAMS
from teach import schema
from teach.validate import Problems, effective_params, validate_settings, validate_submission

from .conftest import IMAGE, META, REFERENCE, REPO, post

TEN = ("dark_threshold", "min_path_length", "enable_pruning", "normalization_mode", "normalization_sensitivity",
       "enable_post_fit_export", "source_smoothing", "enable_curve_fitting", "cubic_fit_tolerance",
       "force_orthogonal_as_lines")
UI_IDS = {"dark_threshold": "darkThreshold", "min_path_length": "minPathLength", "source_smoothing": "sourceSmoothing",
          "cubic_fit_tolerance": "cubicFitTolerance"}
UI_CHECKBOXES = {"enable_pruning": "enablePruning", "enable_post_fit_export": "enablePostFitExport",
                 "force_orthogonal_as_lines": "forceOrthogonalAsLines"}
UI_SELECTS = {"normalization_mode": "normalizationMode", "normalization_sensitivity": "normalizationSensitivity"}
MAIN_HTML = (REPO / "templates" / "index.html").read_text(encoding="utf-8")


def test_whitelist_is_exactly_the_ten_settings():
    assert tuple(schema.SETTINGS) == TEN
    assert tuple(schema.SETTING_UI) == TEN
    assert set(schema.submission_meta_schema()["properties"]["settings"]["properties"]) == set(TEN)


@pytest.mark.parametrize("name", list(UI_IDS))
def test_number_ranges_and_defaults_match_main_ui(name):
    tag = re.search(rf'<input type="range" id="{UI_IDS[name]}"[^>]*>', MAIN_HTML).group(0)
    attr = dict(re.findall(r'(\w+)="([^"]*)"', tag))
    spec = schema.SETTINGS[name]
    assert float(attr["min"]) == spec["minimum"] and float(attr["max"]) == spec["maximum"]
    assert float(attr["value"]) == spec["default"]
    assert float(attr["step"]) == schema.SETTING_UI[name][1]


@pytest.mark.parametrize("name", list(UI_CHECKBOXES))
def test_checkbox_defaults_match_main_ui(name):
    tag = re.search(rf'<input type="checkbox" id="{UI_CHECKBOXES[name]}"[^>]*>', MAIN_HTML).group(0)
    assert ("checked" in tag) == schema.SETTINGS[name]["default"]


@pytest.mark.parametrize("name", list(UI_SELECTS))
def test_select_options_match_main_ui(name):
    start = MAIN_HTML.index(f'<select id="{UI_SELECTS[name]}"')
    block = MAIN_HTML[start:MAIN_HTML.index("</select>", start)]
    assert re.findall(r'<option value="([^"]+)"', block) == schema.SETTINGS[name]["enum"]
    selected = re.search(r'<option value="([^"]+)" selected', block)
    assert (selected.group(1) if selected else schema.SETTINGS[name]["enum"][0]) == schema.SETTINGS[name]["default"]


def test_curve_fitting_default_is_the_effective_ui_value():
    # The UI box is ticked by default, but getParameters() sends postFit && ticked, i.e. false.
    assert re.search(r'id="enableCurveFitting" checked', MAIN_HTML)
    assert "enable_curve_fitting: isPostFitExportEnabled() && document.getElementById('enableCurveFitting').checked" in MAIN_HTML
    assert schema.SETTINGS["enable_curve_fitting"]["default"] is False


def test_defaults_equal_the_lesson_runner_ui_defaults():
    assert schema.default_settings() == {k: UI_DEFAULT_PARAMS[k] for k in TEN}


def test_effective_params_always_has_all_ten():
    assert effective_params({}) == schema.default_settings()
    assert effective_params(None) == schema.default_settings()
    p = effective_params({"dark_threshold": 0.3})
    assert tuple(p) == TEN and p["dark_threshold"] == 0.3 and p["source_smoothing"] == 70


def test_curve_fitting_is_sent_as_its_effective_value():
    assert effective_params({"enable_curve_fitting": True})["enable_curve_fitting"] is False
    both = {"enable_curve_fitting": True, "enable_post_fit_export": True}
    assert effective_params(both)["enable_curve_fitting"] is True


def test_omitting_settings_would_give_different_server_defaults():
    """Why Teach must pass all ten: the app's own fallbacks differ from the UI defaults."""
    from lessonkit.runner import _app_module

    resolve = _app_module()._resolve_svg_render_settings
    absent = resolve({"enable_post_fit_export": True})
    explicit = resolve(effective_params({"enable_post_fit_export": True}))
    assert absent["source_smoothing"] == 0.0 and explicit["source_smoothing"] == 70.0
    assert absent["curve_fit_tolerance"] == 1.0 and explicit["curve_fit_tolerance"] == 0.35
    ticked = resolve(effective_params({"enable_post_fit_export": True, "enable_curve_fitting": True}))
    assert absent["enable_curve_fitting"] is False and ticked["enable_curve_fitting"] is True


@pytest.mark.parametrize("settings", [None, {}, {"dark_threshold": 0.4}, {"enable_post_fit_export": True}])
def test_scoring_job_always_receives_all_ten(teach, settings):
    meta = dict(META)
    meta.pop("settings", None)
    if settings is not None:
        meta["settings"] = settings
    r = post(teach.client, "/teach/api/validate", meta=meta)
    assert r.status_code == 200, r.get_json()
    params = teach.queue.calls[-1]["params"]
    assert tuple(params) == TEN
    assert params == effective_params(settings or {})


@pytest.mark.parametrize("key,value", [
    ("source_smoothing", -1), ("source_smoothing", 100.5), ("source_smoothing", "70"),
    ("cubic_fit_tolerance", 0.3), ("cubic_fit_tolerance", 4.01), ("cubic_fit_tolerance", None),
    ("enable_post_fit_export", "maybe"), ("enable_curve_fitting", 2), ("force_orthogonal_as_lines", "x"),
])
def test_new_settings_out_of_range(key, value):
    _, problems = validate_settings({key: value})
    assert [(p.field, p.code) for p in problems] == [(f"settings.{key}", "param_out_of_range")]


@pytest.mark.parametrize("key", ["enable_optimization", "merge_gap", "rdp_tolerance", "include_image",
                                 "endpoint_tangent_strictness"])
def test_optimisation_settings_stay_out(key):
    with pytest.raises(Problems) as info:
        validate_submission(IMAGE, REFERENCE, json.dumps({**META, "settings": {key: 1}}))
    assert [(p.field, p.code) for p in info.value.problems] == [(f"settings.{key}", "param_out_of_range")]


def test_lesson_params_whitelist_is_additive():
    assert set(ALLOWED_PARAMS) == set(TEN)
    assert set(DEFAULT_PARAMS) | set(OPTIONAL_PARAMS) == set(TEN)
    assert OPTIONAL_PARAMS == {k: UI_DEFAULT_PARAMS[k] for k in OPTIONAL_PARAMS}
    lessons = discover_lessons()
    assert lessons
    for lesson in lessons:
        assert lesson.params == {**DEFAULT_PARAMS, **(lesson.meta.get("params") or {})}
        assert set(lesson.params) == set(DEFAULT_PARAMS), lesson.id
    meta = dict(lessons[0].meta)
    meta["params"] = {**DEFAULT_PARAMS, "enable_post_fit_export": True, "source_smoothing": 50}
    assert validate_meta(meta) == []
    meta["params"] = {**DEFAULT_PARAMS, "merge_gap": 5}
    assert any("unknown params" in e for e in validate_meta(meta))
