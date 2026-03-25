import numpy as np

import centerline_core
from centerline_core import (
    AUTO_TUNE_RANDOM_TILE_DIM,
    _build_random_tile_mosaic,
    auto_detect_dark_threshold,
    auto_tune_extraction_parameters,
    resolve_auto_tune_sampling_plan,
    resolve_parameter_scale,
)
from centerline_web_app import (
    _effective_min_object_size,
    _effective_merge_gap,
    _effective_merge_gap_from_stroke,
    _effective_min_path_length,
    _estimate_median_stroke_width_from_masks,
)


def test_resolution_scale_is_neutral_for_typical_images():
    assert resolve_parameter_scale((1200, 900)) == 1.0


def test_effective_parameters_expand_for_large_images():
    extraction_profile = {'max_min_path_length': None}
    params = {'min_path_length': 3, 'merge_gap': 25}

    assert _effective_min_path_length(params, extraction_profile, image_shape=(3200, 2400)) == 6
    assert _effective_merge_gap(params, image_shape=(3200, 2400)) == 50


def test_stroke_width_can_reduce_merge_gap_for_large_thin_lines():
    params = {'merge_gap': 25}

    assert _effective_merge_gap_from_stroke(
        params,
        image_shape=(3200, 2400),
        median_stroke_width_px=2.0,
    ) == 17


def test_stroke_factor_mode_uses_merge_gap_as_stroke_multiplier():
    params = {'merge_gap': 2, 'merge_gap_mode': 'stroke_factor'}

    assert _effective_merge_gap_from_stroke(
        params,
        image_shape=(3200, 2400),
        median_stroke_width_px=2.5,
    ) == 5


def test_mask_based_stroke_width_estimate_uses_skeleton_distance():
    binary = np.zeros((9, 9), dtype=bool)
    binary[3:6, 1:8] = True
    skeleton = np.zeros_like(binary)
    skeleton[4, 1:8] = True

    estimate = _estimate_median_stroke_width_from_masks(binary, skeleton)

    assert estimate is not None
    assert 2.0 <= estimate <= 4.0


def test_profile_cap_applies_before_resolution_scaling():
    extraction_profile = {'max_min_path_length': 2}
    params = {'min_path_length': 8, 'merge_gap': 25}

    assert _effective_min_path_length(params, extraction_profile, image_shape=(3200, 2400)) == 4


def test_effective_min_object_size_is_capped_by_logical_min_path_length():
    extraction_profile = {'max_min_path_length': None}
    params = {'min_path_length': 2}

    assert _effective_min_object_size(5, params, extraction_profile, image_shape=(1200, 900)) == 1


def test_effective_min_object_size_relaxes_for_large_images():
    extraction_profile = {'max_min_path_length': None}
    params = {'min_path_length': 3}

    assert _effective_min_object_size(2, params, extraction_profile, image_shape=(3200, 2400)) == 1


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


def test_auto_detect_dark_threshold_uses_preview_sampling_for_large_images(monkeypatch):
    gray = np.ones((2400, 2400), dtype=np.float32)
    called_shapes = []

    def fake_extract(sample_image, threshold, min_object_size=3):
        called_shapes.append(tuple(sample_image.shape))
        if abs(float(threshold) - 0.33) <= 0.06:
            return [[(0, 0)] * 24 for _ in range(8)]
        if abs(float(threshold) - 0.33) <= 0.14:
            return [[(0, 0)] * 14 for _ in range(4)]
        return []

    monkeypatch.setattr(centerline_core, 'extract_skeleton_paths', fake_extract)
    monkeypatch.setattr(centerline_core, 'merge_nearby_paths', lambda paths, **kwargs: list(paths))

    result = auto_detect_dark_threshold(
        gray,
        num_thresholds=8,
        preview_max_dim=512,
        parameter_scale=resolve_parameter_scale(gray.shape),
        confidence_target=0.90,
    )

    assert result['best_score'] > 0
    assert result['sample_metadata']['sampling_mode'] == 'random_tiles_mosaic'
    assert result['thresholds_evaluated'] < 8
    assert called_shapes
    assert all(max(shape) <= 512 for shape in called_shapes)


def test_auto_detect_dark_threshold_respects_cancellation_before_work(monkeypatch):
    gray = np.ones((256, 256), dtype=np.float32)
    extract_calls = {'count': 0}

    def fake_extract(*args, **kwargs):
        extract_calls['count'] += 1
        return []

    monkeypatch.setattr(centerline_core, 'extract_skeleton_paths', fake_extract)

    result = auto_detect_dark_threshold(
        gray,
        num_thresholds=6,
        should_continue=lambda: False,
    )

    assert result['cancelled'] is True
    assert result['thresholds_evaluated'] == 0
    assert extract_calls['count'] == 0