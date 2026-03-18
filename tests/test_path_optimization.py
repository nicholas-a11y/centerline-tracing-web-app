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
    assert len(conditioned) < len(path)
    assert not engine._path_has_self_intersection(conditioned)


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
    assert smooth_call_count['count'] == 1
    assert fit_curve_call_count['count'] == 0


def test_merge_nearby_paths_rejects_self_intersecting_bridge():
    path_a = [(0, 0), (1, 1), (2, 2), (3, 3)]
    path_b = [(3, 0), (2, 1), (1, 2), (0, 3)]

    merged = engine.merge_nearby_paths([path_a, path_b], max_gap=3.5, verbose=False)

    assert len(merged) == 2