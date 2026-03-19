from __future__ import annotations

import centerline_engine as engine


class _ConstantCircleSystem:
    def evaluate_path(self, path):
        return 100.0


def test_precondition_path_for_optimization_simplifies_staircase_noise():
    path = [
        (0, 0),
        (0, 1),
        (1, 1),
        (1, 2),
        (2, 2),
        (2, 3),
        (3, 3),
        (3, 4),
    ]

    conditioned = engine.precondition_path_for_optimization(path, base_tolerance=0.8)

    assert conditioned[0] == path[0]
    assert conditioned[-1] == path[-1]
    assert len(conditioned) == 2
    assert not engine._path_has_self_intersection(conditioned)


def test_precondition_path_for_optimization_projects_through_terminal_spurs():
    path = [
        (1150.50, 138.50),
        (1152.73, 141.50),
        (1598.08, 154.50),
        (1603.35, 156.50),
        (2048.28, 170.50),
        (2050.50, 172.50),
    ]

    conditioned = engine.precondition_path_for_optimization(path, base_tolerance=0.8)

    assert len(conditioned) < len(path)
    assert conditioned[0][0] < conditioned[-1][0]
    assert not engine._path_has_self_intersection(conditioned)

    optimized, _score = engine.optimize_path_with_custom_params(
        path,
        _ConstantCircleSystem(),
        {"min_path_length": 2, "simplification_strength": 60.0, "line_fit_strength": 35.0, "rdp_tolerance": 2.0},
        initial_score=100.0,
    )

    assert len(optimized) == 2
    assert optimized[0] == path[0]
    assert optimized[-1] == path[-1]


def test_precondition_path_for_optimization_collapses_monotone_staircase_line():
    path = [
        (1149.5, 136.5),
        (1152.5, 141.5),
        (1183.5, 141.5),
        (1184.5, 142.5),
        (1215.5, 142.5),
        (1216.5, 143.5),
        (1247.5, 143.5),
        (1248.5, 144.5),
        (1279.5, 144.5),
        (1280.5, 145.5),
        (1311.5, 145.5),
        (1312.5, 146.5),
        (1343.5, 146.5),
        (1344.5, 147.5),
    ]

    conditioned = engine.precondition_path_for_optimization(path, base_tolerance=0.8)

    assert len(conditioned) < len(path)
    assert conditioned[0] == path[0]
    assert conditioned[-1] == path[-1]
    assert not engine._path_has_self_intersection(conditioned)

    optimized, _score = engine.optimize_path_with_custom_params(
        path,
        _ConstantCircleSystem(),
        {"min_path_length": 3, "simplification_strength": 60.0, "line_fit_strength": 35.0, "rdp_tolerance": 2.0},
        initial_score=100.0,
    )

    assert len(optimized) <= 3
    assert optimized[0] == path[0]
    assert optimized[-1] == path[-1]


def test_optimizer_allows_two_point_result_for_long_open_staircase():
    path = []
    y = 0
    for block in range(12):
        start_x = block * 6
        path.extend((start_x + offset, y) for offset in range(5))
        y += 1
        path.append((start_x + 5, y))

    path = [path[0]] + [point for idx, point in enumerate(path[1:], start=1) if point != path[idx - 1]]

    optimized, _score = engine.optimize_path_with_custom_params(
        path,
        _ConstantCircleSystem(),
        {"min_path_length": 3, "simplification_strength": 60.0, "line_fit_strength": 35.0, "rdp_tolerance": 2.0},
        initial_score=100.0,
    )

    assert len(optimized) == 2
    assert optimized[0] == path[0]
    assert optimized[-1][0] > optimized[0][0]


def test_optimizer_rejects_self_intersecting_candidate(monkeypatch):
    path = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    looping_candidate = [(0, 0), (3, 3), (0, 3), (3, 0), (4, 4)]

    monkeypatch.setattr(engine, "smooth_path_spline", lambda current, *_args, **_kwargs: looping_candidate)
    monkeypatch.setattr(engine, "fit_curve_to_path", lambda current, *_args, **_kwargs: current)
    monkeypatch.setattr(engine, "rdp_simplify", lambda current, _epsilon: current)

    optimized, _score = engine.optimize_path_with_custom_params(
        path,
        _ConstantCircleSystem(),
        {"min_path_length": 3, "simplification_strength": 0.0, "line_fit_strength": 0.0},
        initial_score=100.0,
    )

    assert not engine._path_has_self_intersection(optimized)


def test_optimizer_skips_arc_candidates_for_straight_paths(monkeypatch):
    path = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]
    smooth_call_count = {'count': 0}
    fit_curve_call_count = {'count': 0}

    def _counting_smooth(current, *_args, **_kwargs):
        smooth_call_count['count'] += 1
        return current

    def _counting_fit(current, *_args, **_kwargs):
        fit_curve_call_count['count'] += 1
        return current

    monkeypatch.setattr(engine, "smooth_path_spline", _counting_smooth)
    monkeypatch.setattr(engine, "fit_curve_to_path", _counting_fit)

    optimized, _score = engine.optimize_path_with_custom_params(
        path,
        _ConstantCircleSystem(),
        {"min_path_length": 3, "simplification_strength": 60.0, "line_fit_strength": 35.0},
        initial_score=100.0,
    )

    assert optimized[0] == path[0]
    assert optimized[-1] == path[-1]
    assert smooth_call_count['count'] == 0
    assert fit_curve_call_count['count'] == 0


def test_merge_nearby_paths_rejects_self_intersecting_bridge():
    path_a = [(0, 0), (1, 1), (2, 2), (3, 3)]
    path_b = [(3, 0), (2, 1), (1, 2), (0, 3)]

    merged = engine.merge_nearby_paths([path_a, path_b], max_gap=3.5, verbose=False)

    assert len(merged) == 2


def test_merge_nearby_paths_reports_metrics_and_progress():
    path_a = [(0, 0), (1, 0), (2, 0)]
    path_b = [(3, 0), (4, 0), (5, 0)]
    metrics = {}
    progress_updates = []

    merged = engine.merge_nearby_paths(
        [path_a, path_b],
        max_gap=1.5,
        verbose=False,
        metrics=metrics,
        progress_callback=lambda payload: progress_updates.append(dict(payload)),
    )

    assert len(merged) == 1
    assert metrics['merged_pairs'] == 1
    assert metrics['candidate_paths_scanned'] >= 1
    assert metrics['cheap_candidates_ranked'] >= 1
    assert metrics['endpoint_distance_checks'] >= 1
    assert metrics['safety_checks'] >= 1
    assert metrics['seed_paths_processed'] >= 1
    assert metrics['elapsed_sec'] >= 0.0
    assert progress_updates


def test_merge_nearby_paths_skips_any_long_candidate_in_cheap_pass():
    path_a = [(0, idx) for idx in range(8)]
    path_b = [(0, idx) for idx in range(8, 10)]
    metrics = {}

    merged = engine.merge_nearby_paths(
        [path_a, path_b],
        max_gap=2.0,
        verbose=False,
        metrics=metrics,
        allow_long_path_merges=False,
        long_path_threshold=5,
    )

    assert len(merged) == 2
    assert metrics['merged_pairs'] == 0
    assert metrics['long_fragment_candidates_skipped'] >= 1
    assert metrics['merge_scope'] == 'cheap_only'


def test_merge_nearby_paths_skips_long_long_candidates_in_cheap_pass():
    path_a = [(0, idx) for idx in range(8)]
    path_b = [(0, idx) for idx in range(8, 16)]
    metrics = {}

    merged = engine.merge_nearby_paths(
        [path_a, path_b],
        max_gap=2.0,
        verbose=False,
        metrics=metrics,
        allow_long_path_merges=False,
        long_path_threshold=5,
    )

    assert len(merged) == 2
    assert metrics['merged_pairs'] == 0
    assert metrics['long_fragment_candidates_skipped'] >= 1
    assert metrics['merge_scope'] == 'cheap_only'


def test_merge_nearby_paths_long_only_pass_can_merge_long_fragments():
    path_a = [(0, idx) for idx in range(8)]
    path_b = [(0, idx) for idx in range(8, 10)]
    metrics = {}

    merged = engine.merge_nearby_paths(
        [path_a, path_b],
        max_gap=2.0,
        verbose=False,
        metrics=metrics,
        allow_long_path_merges=True,
        only_long_path_merges=True,
        long_path_threshold=5,
    )

    assert len(merged) == 1
    assert metrics['merged_pairs'] == 1
    assert metrics['merge_scope'] == 'long_only'