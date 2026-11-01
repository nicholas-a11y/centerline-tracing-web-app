from __future__ import annotations

import time

import numpy as np

import centerline_engine_light as engine


def _supersample_path(points, scale_factor):
    scale = float(scale_factor)
    return [
        ((((float(y) + 0.5) * scale) - 0.5), (((float(x) + 0.5) * scale) - 0.5))
        for y, x in points
    ]


def test_downscale_extracted_paths_round_trips_known_supersampled_path():
    native_path = [(0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (1.0, 2.0)]
    supersampled_paths = [_supersample_path(native_path, scale_factor=4)]

    assert engine._downscale_extracted_paths(supersampled_paths, scale_factor=4) == [native_path]


def test_downscale_extracted_paths_is_noop_at_scale_one():
    native_paths = [[(1.0, 1.0), (1.0, 2.0)]]

    assert engine._downscale_extracted_paths(native_paths, scale_factor=1) == native_paths


def test_resolve_small_image_supersample_factor_extends_into_mid_band_but_skips_large_images():
    assert engine._resolve_small_image_supersample_factor((16, 16)) == 3
    assert engine._resolve_small_image_supersample_factor((260, 260)) == 3
    assert engine._resolve_small_image_supersample_factor((450, 450)) == 2
    assert engine._resolve_small_image_supersample_factor((800, 800)) == 1


def test_extract_skeleton_paths_downscales_legacy_results_when_supersampling_is_active(monkeypatch):
    expected_native_path = [(0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (0.0, 3.0)]
    image_shape = (16, 16)
    scale_factor = engine._resolve_small_image_supersample_factor(image_shape)
    observed = {}

    def _fake_create_fast_paths(skeleton, min_path_length=1):
        observed["shape"] = tuple(skeleton.shape)
        return [_supersample_path(expected_native_path, scale_factor=scale_factor)]

    monkeypatch.setattr(engine, "USE_VECTORIZED_FAST_PATHS", False)
    monkeypatch.setattr("centerline_engine.create_fast_paths", _fake_create_fast_paths)

    result = engine.extract_skeleton_paths(
        np.zeros(image_shape, dtype=np.float32),
        dark_threshold=0.5,
        min_object_size=0,
        min_path_length=2,
        enable_small_image_supersample=True,
    )

    assert observed["shape"] == (image_shape[0] * scale_factor, image_shape[1] * scale_factor)
    assert result == [expected_native_path]


def test_extract_skeleton_paths_scales_min_path_length_with_supersample_factor(monkeypatch):
    image_shape = (260, 260)
    scale_factor = engine._resolve_small_image_supersample_factor(image_shape)
    observed = {}
    expected_native_path = [(0.0, 0.0), (0.0, 1.0)]

    def _fake_create_fast_paths(skeleton, min_path_length=1):
        observed["shape"] = tuple(skeleton.shape)
        observed["min_path_length"] = min_path_length
        return [_supersample_path(expected_native_path, scale_factor=scale_factor)]

    monkeypatch.setattr(engine, "USE_VECTORIZED_FAST_PATHS", False)
    monkeypatch.setattr("centerline_engine.create_fast_paths", _fake_create_fast_paths)

    result = engine.extract_skeleton_paths(
        np.zeros(image_shape, dtype=np.float32),
        dark_threshold=0.5,
        min_object_size=0,
        min_path_length=2,
        enable_small_image_supersample=True,
    )

    assert observed["shape"] == (image_shape[0] * scale_factor, image_shape[1] * scale_factor)
    assert observed["min_path_length"] == 2 * scale_factor
    assert result == [expected_native_path]


def test_extract_skeleton_paths_scales_min_object_size_with_supersample_factor(monkeypatch):
    image_shape = (260, 260)
    scale_factor = engine._resolve_small_image_supersample_factor(image_shape)
    observed = {}
    expected_native_path = [(0.0, 0.0), (0.0, 1.0)]

    def _fake_remove_small_objects(binary, min_size):
        observed["binary_shape"] = tuple(binary.shape)
        observed["min_size"] = min_size
        return binary

    monkeypatch.setattr(engine, "USE_VECTORIZED_FAST_PATHS", False)
    monkeypatch.setattr(engine.morphology, "remove_small_objects", _fake_remove_small_objects)
    monkeypatch.setattr(engine.morphology, "skeletonize", lambda binary: np.zeros_like(binary, dtype=bool))
    monkeypatch.setattr(
        "centerline_engine.create_fast_paths",
        lambda skeleton, min_path_length=1: [_supersample_path(expected_native_path, scale_factor=scale_factor)],
    )

    result = engine.extract_skeleton_paths(
        np.zeros(image_shape, dtype=np.float32),
        dark_threshold=0.5,
        min_object_size=2,
        min_path_length=1,
        enable_small_image_supersample=True,
    )

    assert observed["binary_shape"] == (image_shape[0] * scale_factor, image_shape[1] * scale_factor)
    assert observed["min_size"] == 2 * scale_factor * scale_factor
    assert result == [expected_native_path]


def test_extract_skeleton_paths_keeps_native_thresholds_for_large_images(monkeypatch):
    image_shape = (1500, 1500)
    observed = {}
    expected_native_path = [(0.0, 0.0), (0.0, 1.0)]

    def _fake_remove_small_objects(binary, min_size):
        observed["binary_shape"] = tuple(binary.shape)
        observed["min_size"] = min_size
        return binary

    def _fake_create_fast_paths(skeleton, min_path_length=1):
        observed["skeleton_shape"] = tuple(skeleton.shape)
        observed["min_path_length"] = min_path_length
        return [expected_native_path]

    monkeypatch.setattr(engine, "USE_VECTORIZED_FAST_PATHS", False)
    monkeypatch.setattr(engine.morphology, "remove_small_objects", _fake_remove_small_objects)
    monkeypatch.setattr(engine.morphology, "skeletonize", lambda binary: np.zeros_like(binary, dtype=bool))
    monkeypatch.setattr("centerline_engine.create_fast_paths", _fake_create_fast_paths)

    result = engine.extract_skeleton_paths(
        np.zeros(image_shape, dtype=np.float32),
        dark_threshold=0.5,
        min_object_size=2,
        min_path_length=2,
        enable_small_image_supersample=True,
    )

    assert engine._resolve_small_image_supersample_factor(image_shape) == 1
    assert observed["binary_shape"] == image_shape
    assert observed["min_size"] == 2
    assert observed["skeleton_shape"] == image_shape
    assert observed["min_path_length"] == 2
    assert result == [expected_native_path]


def test_supersampled_speckle_filter_is_not_more_permissive_than_native_run():
    image = np.ones((260, 260), dtype=np.float32)
    image[130, 130] = 0.0

    native_paths = engine.extract_skeleton_paths(
        image,
        dark_threshold=0.5,
        min_object_size=2,
        min_path_length=1,
        enable_small_image_supersample=False,
    )
    supersampled_paths = engine.extract_skeleton_paths(
        image,
        dark_threshold=0.5,
        min_object_size=2,
        min_path_length=1,
        enable_small_image_supersample=True,
    )

    assert native_paths == []
    assert supersampled_paths == []


def test_supersample_timing_prints_mid_band_and_large_case_metrics(capsys):
    for size in (260, 1500):
        image = np.ones((size, size), dtype=np.float32)
        image[size // 2, size // 4:size - (size // 4)] = 0.0
        factor = engine._resolve_small_image_supersample_factor(image.shape)

        started_at = time.perf_counter()
        paths = engine.extract_skeleton_paths(
            image,
            dark_threshold=0.5,
            min_object_size=0,
            min_path_length=2,
            enable_small_image_supersample=True,
        )
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        print(
            f"supersample_timing size={size} factor={factor} path_count={len(paths)} elapsed_ms={elapsed_ms:.3f}"
        )

    captured = capsys.readouterr()
    assert "size=260" in captured.out
    assert "factor=3" in captured.out
    assert "size=1500" in captured.out
    assert "factor=1" in captured.out

def test_small_object_filter_keeps_one_pixel_diagonal_lines_but_drops_corner_specks():
    binary = np.zeros((40, 40), dtype=bool)
    for i in range(5, 30):
        binary[i, i] = True
    binary[5:10, 32:38] = True
    binary[10, 38] = True

    filtered = engine.remove_small_objects_preserving_thin_lines(binary, 2)

    diagonal = np.array([filtered[i, i] for i in range(5, 30)])
    assert diagonal[1:-1].all()
    assert filtered[5:10, 32:38].all()
    assert not filtered[10, 38]
