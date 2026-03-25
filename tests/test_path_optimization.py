from __future__ import annotations

import centerline_engine as engine
from tests.fixtures import load_fixture


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


def test_prune_extracted_paths_removes_terminal_spur_artifacts():
    path = [
        (1150.50, 138.50),
        (1152.73, 141.50),
        (1598.08, 154.50),
        (1603.35, 156.50),
        (2048.28, 170.50),
        (2050.50, 172.50),
    ]

    pruned = engine.prune_extracted_paths([path], min_output_points=2)

    assert len(pruned) == 1
    assert len(pruned[0]) < len(path)
    assert pruned[0][0] != path[0]
    assert not engine._path_has_self_intersection(pruned[0])


def test_prune_extracted_paths_collapses_staircase_noise_before_fitting():
    path = [
        (0, 0),
        (0, 1),
        (1, 1),
        (1, 2),
        (2, 2),
        (2, 3),
        (3, 3),
        (3, 4),
        (4, 4),
        (4, 5),
    ]

    pruned = engine.prune_extracted_paths([path], min_output_points=2)

    assert len(pruned) == 1
    assert len(pruned[0]) <= 3
    assert pruned[0][0] == path[0]
    assert pruned[0][-1] == path[-1]


def test_prune_extracted_paths_drops_projected_terminal_spur_endpoints():
    path = [
        (0, 0),
        (1, 2),
        (2, 4),
        (2, 20),
        (2, 40),
        (2, 60),
        (2, 80),
        (2, 100),
        (3, 102),
        (4, 104),
    ]

    pruned, debug_payload = engine.prune_extracted_paths(
        [path],
        min_output_points=2,
        return_diagnostics=True,
    )

    assert len(pruned) == 1
    assert len(pruned[0]) == 2
    assert pruned[0] == [(2.0, 4.0), (2.0, 100.0)]
    assert debug_payload['paths'][0]['path_state'] == 'pruned'


def test_prune_extracted_paths_drops_longer_shallow_terminal_runs():
    path = [
        (0, 0),
        (0.8, 1.2),
        (1.6, 2.4),
        (2.4, 3.6),
        (3.2, 4.8),
        (3.2, 25),
        (3.2, 45),
        (3.2, 65),
        (3.2, 85),
        (3.2, 105),
        (4.0, 106.2),
        (4.8, 107.4),
        (5.6, 108.6),
        (6.4, 109.8),
    ]

    pruned, debug_payload = engine.prune_extracted_paths(
        [path],
        min_output_points=2,
        return_diagnostics=True,
    )

    assert len(pruned) == 1
    assert pruned[0] == [(2.4, 3.6), (3.2, 105.0)]
    assert debug_payload['paths'][0]['path_state'] == 'pruned'


def test_critical_path_indices_ignore_line_like_micro_inflections():
    path = [
        (0.0, 0.0),
        (0.3, 3.0),
        (0.0, 6.0),
        (0.3, 9.0),
        (0.0, 12.0),
        (0.3, 15.0),
        (0.0, 18.0),
    ]

    critical = engine._critical_path_indices(path, endpoint_buffer=0)

    assert critical == {0, len(path) - 1}


def test_critical_path_indices_keep_stronger_sawtooth_inflections():
    path = [
        (0.0, 0.0),
        (0.8, 3.0),
        (0.0, 6.0),
        (0.8, 9.0),
        (0.0, 12.0),
        (0.8, 15.0),
        (0.0, 18.0),
    ]

    critical = engine._critical_path_indices(path, endpoint_buffer=0)

    assert critical == set(range(len(path)))


def test_prune_extracted_paths_keeps_multiple_points_for_curves():
    path = [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (3, 4),
        (4, 5),
        (5, 5),
        (6, 6),
    ]

    pruned = engine.prune_extracted_paths([path], min_output_points=2)

    assert len(pruned) == 1
    assert len(pruned[0]) > 2
    assert pruned[0][0] == path[0]
    assert pruned[0][-1] == path[-1]


def test_prune_extracted_paths_can_return_debug_diagnostics():
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

    pruned, debug_payload = engine.prune_extracted_paths(
        [path],
        min_output_points=2,
        return_diagnostics=True,
        max_debug_paths=3,
    )

    assert len(pruned) == 1
    assert isinstance(debug_payload, dict)
    assert debug_payload['summary']['path_count'] == 1
    assert debug_payload['summary']['input_points'] == len(path)
    assert debug_payload['summary']['output_points'] == len(pruned[0])
    assert len(debug_payload['paths']) == 1
    assert debug_payload['paths'][0]['input_points'][0] == path[0]
    assert debug_payload['paths'][0]['output_points'][0] == pruned[0][0]
    assert debug_payload['paths'][0]['bbox'] is not None
    assert len(debug_payload['paths'][0]['output_point_provenance']) == len(pruned[0])
    assert len(debug_payload['paths'][0]['kept_vertex_guards']) == len(pruned[0])
    assert debug_payload['paths'][0]['path_state'] == 'pruned'
    assert debug_payload['paths'][0]['changed'] is True
    assert debug_payload['paths'][0]['pruning_config']['min_output_points'] == 2
    assert 'criteria_explanations' in debug_payload['paths'][0]['kept_vertex_guards'][0]
    assert debug_payload['paths'][0]['dropped_counts_by_reason']
    assert debug_payload['summary']['dropped_counts_by_reason']
    assert debug_payload['summary']['path_state_counts']['pruned'] == 1
    assert debug_payload['summary']['line_span_hint_count'] >= 1
    assert debug_payload['paths'][0]['line_span_hints']


def test_prune_extracted_paths_reports_unchanged_path_reasons():
    path = [(0, 0), (10, 0)]

    pruned, debug_payload = engine.prune_extracted_paths(
        [path],
        min_output_points=2,
        return_diagnostics=True,
    )

    assert pruned == [path]
    assert debug_payload['paths'][0]['path_state'] == 'unchanged'
    assert debug_payload['paths'][0]['changed'] is False
    assert 'pruning_thresholds_not_exceeded' in debug_payload['paths'][0]['unchanged_reasons']
    assert 'no_terminal_spur_detected' in debug_payload['paths'][0]['unchanged_reasons']
    assert debug_payload['summary']['path_state_counts']['unchanged'] == 1


def test_build_line_span_hints_marks_two_point_path_as_full_line():
    path = [(3, 4), (9, 12)]

    hints = engine.build_line_span_hints(path)

    assert hints == [{
        'start_idx': 0,
        'end_idx': 1,
        'kind': 'line',
        'confidence': 1.0,
        'source': 'two_point_path',
        'start_point': [3, 4],
        'end_point': [9, 12],
    }]


def test_build_line_span_hints_marks_collinear_anchor_run():
    path = [(0, 0), (1, 1), (2, 2), (3, 3)]

    hints = engine.build_line_span_hints(path)

    assert len(hints) == 1
    assert hints[0]['start_idx'] == 0
    assert hints[0]['end_idx'] == 3
    assert hints[0]['kind'] == 'line'
    assert hints[0]['confidence'] > 0.9


def test_prune_extracted_paths_preserves_closed_loop_tangent_anchors_for_true_circle():
    fixture = load_fixture('curve-true-circle', size=64, stroke_width=3)
    paths = engine.extract_skeleton_paths(fixture.gray, 0.5, min_object_size=3)
    closed_paths = [path for path in paths if len(path) >= 2]

    pruned, debug_payload = engine.prune_extracted_paths(
        closed_paths,
        min_output_points=2,
        return_diagnostics=True,
    )

    assert len(pruned) == 1
    reduced = pruned[0]
    reduced_points = list(reduced)

    def _has_point_near(target, max_dist=1.5):
        target_row, target_col = target
        return any(
            (((point[0] - target_row) ** 2) + ((point[1] - target_col) ** 2)) ** 0.5 <= max_dist
            for point in reduced_points
        )

    assert len(reduced) <= 45
    assert _has_point_near((49, 16))
    assert _has_point_near((27, 8), max_dist=2.0)
    assert _has_point_near((16, 14))
    assert _has_point_near((8, 36), max_dist=1.5)
    assert _has_point_near((13, 46), max_dist=1.5)
    assert _has_point_near((34, 55), max_dist=1.5)
    assert _has_point_near((46, 50), max_dist=1.5)
    assert _has_point_near((55, 34), max_dist=1.5)
    assert not engine._path_has_self_intersection(reduced)


def test_prune_extracted_paths_removes_tiny_square_corner_arising_segments():
    fixture = load_fixture('square-perfect', size=64, stroke_width=3)
    paths = engine.extract_skeleton_paths(fixture.gray, 0.5, min_object_size=3)
    closed_paths = [path for path in paths if len(path) >= 2]

    pruned = engine.prune_extracted_paths(closed_paths, min_output_points=2)

    assert len(pruned) == 1
    reduced = pruned[0]
    assert len(reduced) == 5

    expected_corners = [(12, 51), (12, 12), (51, 12), (51, 51)]
    reduced_points = set(reduced)
    for corner in expected_corners:
        assert corner in reduced_points

    segment_lengths = [
        ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5
        for start, end in zip(reduced, reduced[1:])
    ]
    assert min(segment_lengths) >= 10.0


def test_prune_extracted_paths_removes_tiny_open_corner_arising_segment():
    fixture = load_fixture('corner-l-shape', size=64, stroke_width=3)
    paths = engine.extract_skeleton_paths(fixture.gray, 0.5, min_object_size=3)
    open_paths = [path for path in paths if len(path) >= 2]

    pruned, debug_payload = engine.prune_extracted_paths(open_paths, min_output_points=2, return_diagnostics=True)

    assert len(pruned) == 1
    reduced = pruned[0]
    assert reduced == [(54.0, 52.0), (54, 9), (9.0, 9.0)]
    assert debug_payload['paths'][0]['output_point_provenance'] == ['preserved', 'inferred', 'preserved']
    assert debug_payload['paths'][0]['inferred_output_count'] == 1
    guard_details = debug_payload['paths'][0]['kept_vertex_guards']
    assert len(guard_details) == len(reduced)
    assert 'inferred_corner' in guard_details[1]['guards']


def test_prune_extracted_paths_removes_rotated_open_corner_cluster():
    fixture = load_fixture('corner-l-shape-r45', size=64, stroke_width=3)
    paths = engine.extract_skeleton_paths(fixture.gray, 0.5, min_object_size=3)
    open_paths = [path for path in paths if len(path) >= 2]

    pruned, debug_payload = engine.prune_extracted_paths(open_paths, min_output_points=2, return_diagnostics=True)

    assert len(pruned) == 1
    reduced = pruned[0]
    assert len(reduced) == 3
    assert reduced[0] == (7.0, 31.0)
    assert reduced[-1] == (56.0, 31.0)
    corner = reduced[1]
    assert abs(corner[0] - 32.0) <= 0.5
    assert 6.2 <= corner[1] <= 6.8
    assert debug_payload['paths'][0]['output_point_provenance'] == ['preserved', 'inferred', 'preserved']
    assert debug_payload['summary']['inferred_output_points'] >= 1
    assert 'inferred_corner' in debug_payload['paths'][0]['kept_vertex_guards'][1]['guards']


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