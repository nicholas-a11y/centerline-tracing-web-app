import numpy as np

import centerline_core
from centerline_core import (
    AUTO_TUNE_RANDOM_TILE_DIM,
    _build_random_tile_mosaic,
    auto_tune_extraction_parameters,
    resolve_auto_tune_sampling_plan,
    resolve_parameter_scale,
)
from centerline_web_app import _effective_merge_gap, _effective_min_path_length


def test_resolution_scale_is_neutral_for_typical_images():
    assert resolve_parameter_scale((1200, 900)) == 1.0


def test_effective_parameters_expand_for_large_images():
    extraction_profile = {'max_min_path_length': None}
    params = {'min_path_length': 3, 'merge_gap': 25}

    assert _effective_min_path_length(params, extraction_profile, image_shape=(3200, 2400)) == 6
    assert _effective_merge_gap(params, image_shape=(3200, 2400)) == 50


def test_profile_cap_applies_before_resolution_scaling():
    extraction_profile = {'max_min_path_length': 2}
    params = {'min_path_length': 8, 'merge_gap': 25}

    assert _effective_min_path_length(params, extraction_profile, image_shape=(3200, 2400)) == 4


def test_auto_tune_returns_logical_min_length_with_scaled_effective_length(monkeypatch):
    gray = np.ones((64, 64), dtype=np.float32)
    merged_paths = [
        [(0, 0)] * 8,
        [(0, 0)] * 8,
        [(0, 0)] * 8,
        [(0, 0)] * 20,
    ]

    monkeypatch.setattr(centerline_core, 'extract_skeleton_paths', lambda *args, **kwargs: list(merged_paths))
    monkeypatch.setattr(centerline_core, 'merge_nearby_paths', lambda paths, **kwargs: list(paths))

    result = auto_tune_extraction_parameters(
        gray,
        threshold_range=(0.10, 0.20),
        num_thresholds=2,
        base_min_lengths=[3, 5],
        preview_max_dim=64,
        parameter_scale=2.0,
        confidence_target=2.0,
    )

    assert result['best_min_length'] == 3
    assert result['effective_min_path_length'] == 6


def test_random_tile_mosaic_uses_larger_tiles_for_large_images():
    gray = np.ones((4096, 4096), dtype=np.float32)

    _, metadata = _build_random_tile_mosaic(gray, tile_count=4)

    assert metadata['sampling_mode'] == 'random_tiles_mosaic'
    assert metadata['adaptive_tile_dim'] > AUTO_TUNE_RANDOM_TILE_DIM
    assert metadata['tile_shape'][0] > AUTO_TUNE_RANDOM_TILE_DIM


def test_sampling_plan_increases_tile_budget_for_large_images():
    medium_plan = resolve_auto_tune_sampling_plan((1200, 1200))
    large_plan = resolve_auto_tune_sampling_plan((4096, 4096))

    assert large_plan['tile_dim'] >= medium_plan['tile_dim']
    assert large_plan['tile_count'] >= medium_plan['tile_count']
    assert large_plan['target_coverage_ratio'] <= medium_plan['target_coverage_ratio']
    assert large_plan['coverage_tile_target'] >= 1


def test_random_tile_mosaic_places_hotspot_tiles_over_active_regions():
    gray = np.ones((512, 512), dtype=np.float32)
    gray[320:450, 340:420] = 0.0

    _, metadata = _build_random_tile_mosaic(gray, tile_count=6)

    assert metadata['selected_hotspot_tiles'] >= 1
    assert metadata['selected_coverage_tiles'] >= 1
    hotspot_origins = [
        origin
        for origin, mode in zip(metadata['tile_origins'], metadata['tile_selection_modes'])
        if mode == 'hotspot'
    ]
    assert hotspot_origins
    assert any(origin[0] >= 220 and origin[1] >= 220 for origin in hotspot_origins)


def test_random_tile_mosaic_scatter_coverage_tiles_away_from_perimeter_only():
    gray = np.ones((1024, 1024), dtype=np.float32)

    _, metadata = _build_random_tile_mosaic(gray)

    tile_w = int(metadata['tile_shape'][1])
    tile_h = int(metadata['tile_shape'][0])
    max_x = gray.shape[1] - tile_w
    max_y = gray.shape[0] - tile_h
    coverage_origins = [
        origin
        for origin, mode in zip(metadata['tile_origins'], metadata['tile_selection_modes'])
        if mode == 'coverage'
    ]

    assert coverage_origins
    assert any(0 < origin[0] < max_x and 0 < origin[1] < max_y for origin in coverage_origins)