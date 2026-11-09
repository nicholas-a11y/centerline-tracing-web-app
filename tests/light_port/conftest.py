"""Every test under tests/light_port runs with the app's old-engine names rebound to the light engine."""

import pytest

from tests.light_port.light_app import bound_engine_modules, light_engine_imports

# Copies of tests that already fail on the original engine (docs/migration/stage0-baseline.md).
# Non-strict xfail so a port that starts passing shows up as XPASS instead of hiding.
MIRRORED_BASELINE_FAILURES = {
    "test_path_optimization.py": {
        "test_prune_extracted_paths_preserves_closed_loop_tangent_anchors_for_true_circle",
        "test_prune_extracted_paths_removes_tiny_open_corner_arising_segment",
        "test_prune_extracted_paths_removes_rotated_open_corner_cluster",
    },
    "test_web_app_optimization_state.py": {
        "test_download_svg_disabled_optimization_uses_direct_single_layer",
        "test_download_svg_final_results_disabled_optimization_uses_direct_single_layer",
        "test_download_svg_query_overrides_include_image",
        "test_generate_svg_defaults_curve_fitting_off",
    },
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        names = MIRRORED_BASELINE_FAILURES.get(item.path.name)
        if names and item.originalname in names and "light_port" in item.path.parts:
            item.add_marker(pytest.mark.xfail(reason="mirrors Stage 0 baseline failure of the original test", strict=False))


@pytest.fixture(autouse=True)
def _light_engine_app_imports():
    with light_engine_imports():
        assert set(bound_engine_modules().values()) == {"centerline_engine_light"}
        yield
