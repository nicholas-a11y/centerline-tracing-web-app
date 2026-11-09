"""Smoke tests for the light-engine port of the /test-ui rig and the import-rebinding helper."""

import pytest

import centerline_core
import centerline_web_app
from tests.light_port import rig_backend_light as rig
from tests.light_port.light_app import bound_engine_modules, light_engine_imports


def test_rig_port_uses_light_engine():
    assert rig.extract_skeleton_paths.__module__ == "centerline_engine_light"
    assert rig.fit_curve_segments.__module__ == "centerline_engine_light"
    assert not str(rig.RUNS_ROOT).startswith(str(rig.REPO_ROOT))


def test_rebinding_is_active_in_light_port_and_restored_afterwards():
    # The autouse fixture is active here.
    assert set(bound_engine_modules().values()) == {"centerline_engine_light"}
    with light_engine_imports():
        pass
    # Nested use restores to the fixture's (light) bindings, not the originals.
    assert centerline_web_app.merge_nearby_paths.__module__ == "centerline_engine_light"
    assert centerline_core.merge_nearby_paths.__module__ == "centerline_engine_light"


@pytest.mark.parametrize("mode", ["curve_fitter", "main_app"])
@pytest.mark.parametrize("fixture_id", rig.ALL_FIXTURE_IDS)
def test_rig_fixture_analysis_runs_on_light_engine(fixture_id, mode):
    params = rig._normalize_fitting_parameters(dict(rig.DEFAULT_FITTING_PARAMETERS, analysis_mode=mode))
    result = rig._fixture_analysis(fixture_id, params)
    assert result["fixture_id"] == fixture_id
    assert result["layers"]["golden_analytic_path"]
    assert any(len(p) >= 2 for p in result["layers"]["skeleton_paths"])
