from __future__ import annotations

import time

import numpy as np
import pytest

import centerline_engine_light as engine
from tests.fixtures import load_fixture


def _path_point_coverage(paths):
    return {tuple(point) for path in paths for point in path}


def _open_path_endpoint_set(paths):
    endpoints = set()
    for path in paths:
        if not path:
            continue
        if tuple(path[0]) == tuple(path[-1]):
            continue
        endpoints.add(tuple(path[0]))
        endpoints.add(tuple(path[-1]))
    return endpoints


def _closed_path_count(paths):
    return sum(1 for path in paths if path and tuple(path[0]) == tuple(path[-1]))


def _total_path_length(paths):
    total = 0.0
    for path in paths:
        for start_point, end_point in zip(path, path[1:]):
            dy = float(end_point[0]) - float(start_point[0])
            dx = float(end_point[1]) - float(start_point[1])
            total += (dy * dy + dx * dx) ** 0.5
    return total


def _legacy_junction_points(mask):
    ys, xs = np.nonzero(mask)
    skeleton_points = set(zip(ys, xs))
    neighbors = engine._build_fast_path_neighbors(skeleton_points)
    _node_points, junction_cluster_ids, _junction_cluster_sizes, _junction_cluster_anchors = engine._build_fast_path_junction_state(neighbors)
    return set(junction_cluster_ids)


def _vectorized_junction_points(mask):
    degree_map = engine._compute_fast_path_degree_map(mask)
    junction_mask, _junction_cluster_ids, _junction_cluster_sizes, _junction_cluster_anchors = engine._build_vectorized_junction_state(mask, degree_map)
    return {tuple(point) for point in np.argwhere(junction_mask)}


def _assert_fast_path_equivalence(mask, min_path_length=2, allowed_path_count_delta=0):
    mask = np.asarray(mask, dtype=bool)
    legacy_paths = engine.create_fast_paths(mask, min_path_length=min_path_length)
    vectorized_paths = engine.create_fast_paths_vectorized(mask, min_path_length=min_path_length)

    assert _legacy_junction_points(mask) == _vectorized_junction_points(mask)
    assert _path_point_coverage(legacy_paths) == _path_point_coverage(vectorized_paths)
    assert _open_path_endpoint_set(legacy_paths) == _open_path_endpoint_set(vectorized_paths)
    assert _closed_path_count(legacy_paths) == _closed_path_count(vectorized_paths)
    assert abs(len(legacy_paths) - len(vectorized_paths)) <= allowed_path_count_delta
    assert _total_path_length(vectorized_paths) == pytest.approx(
        _total_path_length(legacy_paths),
        rel=0.01,
        abs=1.0,
    )

    return legacy_paths, vectorized_paths


def _make_horizontal_line(size):
    mask = np.zeros((size, size), dtype=bool)
    margin = max(10, size // 12)
    mask[size // 2, margin:size - margin] = True
    return mask


def _make_branching_field(size, branch_spacing=None, branch_half_length=None):
    mask = np.zeros((size, size), dtype=bool)
    margin = max(10, size // 12)
    row = size // 2
    mask[row, margin:size - margin] = True

    if branch_spacing is None:
        branch_spacing = max(24, size // 12)
    if branch_half_length is None:
        branch_half_length = max(20, size // 5)
    branch_top = max(1, row - branch_half_length)
    branch_bottom = min(size - 2, row + branch_half_length)
    for column in range(margin + branch_spacing // 2, size - margin, branch_spacing):
        mask[branch_top:branch_bottom + 1, column] = True

    return mask


def _time_extractor(extractor, mask, repeats=2):
    best_elapsed = None
    last_paths = None
    for _ in range(repeats):
        started_at = time.perf_counter()
        last_paths = extractor(mask, min_path_length=2)
        elapsed = time.perf_counter() - started_at
        if best_elapsed is None or elapsed < best_elapsed:
            best_elapsed = elapsed
    return float(best_elapsed or 0.0), last_paths


@pytest.mark.parametrize("size", [300, 800, 1500])
def test_create_fast_paths_vectorized_matches_legacy_horizontal_line(size):
    _assert_fast_path_equivalence(_make_horizontal_line(size), min_path_length=2)


@pytest.mark.parametrize("size", [300, 800])
def test_create_fast_paths_vectorized_matches_legacy_branching_field(size):
    _assert_fast_path_equivalence(_make_branching_field(size), min_path_length=2)


@pytest.mark.parametrize("fixture_id", ["corner-l-shape", "curve-true-circle", "square-perfect"])
def test_create_fast_paths_vectorized_matches_legacy_fixture_skeletons(fixture_id):
    fixture = load_fixture(fixture_id, size=64, stroke_width=3)
    mask = fixture.gray < 0.5
    skeleton = engine.morphology.skeletonize(mask)

    _assert_fast_path_equivalence(skeleton, min_path_length=2)


def test_extract_skeleton_paths_uses_vectorized_walker_above_threshold(monkeypatch):
    sentinel_paths = [[(1, 1), (1, 2)]]

    monkeypatch.setattr(engine, "USE_VECTORIZED_FAST_PATHS", True)
    monkeypatch.setattr(engine, "VECTORIZED_FAST_PATHS_MIN_PIXELS", 0)
    monkeypatch.setattr(engine, "create_fast_paths", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy walker should not run")))

    def _fake_vectorized(skeleton, min_path_length=1):
        assert skeleton.dtype == bool
        return sentinel_paths

    monkeypatch.setattr(engine, "create_fast_paths_vectorized", _fake_vectorized)

    paths = engine.extract_skeleton_paths(
        np.zeros((16, 16), dtype=np.float32),
        dark_threshold=0.5,
        min_object_size=0,
        min_path_length=2,
        enable_small_image_supersample=False,
    )

    assert paths == sentinel_paths


def test_extract_skeleton_paths_uses_legacy_walker_below_threshold(monkeypatch):
    sentinel_paths = [[(2, 2), (2, 3)]]

    monkeypatch.setattr(engine, "USE_VECTORIZED_FAST_PATHS", True)
    monkeypatch.setattr(engine, "VECTORIZED_FAST_PATHS_MIN_PIXELS", 10**9)
    monkeypatch.setattr(engine, "create_fast_paths_vectorized", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("vectorized walker should not run")))

    def _fake_legacy(skeleton, min_path_length=1):
        assert skeleton.dtype == bool
        return sentinel_paths

    monkeypatch.setattr(engine, "create_fast_paths", _fake_legacy)

    paths = engine.extract_skeleton_paths(
        np.zeros((16, 16), dtype=np.float32),
        dark_threshold=0.5,
        min_object_size=0,
        min_path_length=2,
        enable_small_image_supersample=False,
    )

    assert paths == sentinel_paths


@pytest.mark.parametrize("fixture_id", ["corner-l-shape", "curve-true-circle"])
def test_extract_skeleton_paths_flag_toggle_preserves_fixture_summary(monkeypatch, fixture_id):
    fixture = load_fixture(fixture_id, size=64, stroke_width=3)
    image = fixture.gray

    monkeypatch.setattr(engine, "USE_VECTORIZED_FAST_PATHS", False)
    monkeypatch.setattr(engine, "VECTORIZED_FAST_PATHS_MIN_PIXELS", 0)
    legacy_paths = engine.extract_skeleton_paths(
        image,
        dark_threshold=0.5,
        min_object_size=3,
        min_path_length=2,
        enable_small_image_supersample=False,
    )

    monkeypatch.setattr(engine, "USE_VECTORIZED_FAST_PATHS", True)
    vectorized_paths = engine.extract_skeleton_paths(
        image,
        dark_threshold=0.5,
        min_object_size=3,
        min_path_length=2,
        enable_small_image_supersample=False,
    )

    assert _path_point_coverage(legacy_paths) == _path_point_coverage(vectorized_paths)
    assert _open_path_endpoint_set(legacy_paths) == _open_path_endpoint_set(vectorized_paths)
    assert _closed_path_count(legacy_paths) == _closed_path_count(vectorized_paths)
    assert abs(len(legacy_paths) - len(vectorized_paths)) <= 1
    assert _total_path_length(vectorized_paths) == pytest.approx(
        _total_path_length(legacy_paths),
        rel=0.01,
        abs=1.0,
    )


def test_create_fast_paths_vectorized_benchmark_large_branching_field():
    mask = _make_branching_field(1500, branch_spacing=20, branch_half_length=650)
    legacy_paths, vectorized_paths = _assert_fast_path_equivalence(mask, min_path_length=2)

    legacy_time, legacy_timed_paths = _time_extractor(engine.create_fast_paths, mask, repeats=2)
    vectorized_time, vectorized_timed_paths = _time_extractor(engine.create_fast_paths_vectorized, mask, repeats=2)

    assert _path_point_coverage(legacy_paths) == _path_point_coverage(legacy_timed_paths)
    assert _path_point_coverage(vectorized_paths) == _path_point_coverage(vectorized_timed_paths)
    print(
        "fast-path benchmark "
        f"legacy_ms={legacy_time * 1000.0:.2f} "
        f"vectorized_ms={vectorized_time * 1000.0:.2f} "
        f"speedup={legacy_time / max(vectorized_time, 1e-9):.2f}x"
    )
    assert vectorized_time < legacy_time