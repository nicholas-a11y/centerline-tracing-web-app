from __future__ import annotations

import numpy as np

from tests.pipeline.scoring import densify, graph_metrics, score_paths

SCORING = {"region": None, "tolerance_px": 1.0, "min_single_path_coverage": 0.95,
           "require_single_path": True, "trim_ends_px": 0.0}


def _line(r0, c0, r1, c1):
    return np.array([[r0, c0], [r1, c1]], dtype=float)


def test_perfect_match_passes():
    ref = [_line(10, 0, 10, 50)]
    s = score_paths([densify(ref[0], 1.0)], ref, SCORING)
    assert s.passed
    assert s.lines[0].best_single_coverage == 1.0
    assert s.lines[0].ref_to_best["p95"] <= 0.125 + 1e-9  # densify step 0.25 px bounds sampling error


def test_lateral_offset_fails_on_error():
    ref = [_line(10, 0, 10, 50)]
    s = score_paths([_line(12, 0, 12, 50)], ref, SCORING)
    assert not s.passed
    assert abs(s.lines[0].ref_to_best["p95"] - 2.0) < 1e-6


def test_split_path_fails_single_but_full_any_coverage():
    ref = [_line(10, 0, 10, 50)]
    s = score_paths([_line(10, 0, 10, 24), _line(10, 26, 10, 50)], ref, SCORING)
    ln = s.lines[0]
    assert not s.passed
    assert ln.any_coverage == 1.0
    assert ln.best_single_coverage < 0.6
    assert ln.fragments == 2
    relaxed = score_paths([_line(10, 0, 10, 24), _line(10, 26, 10, 50)], ref,
                          {**SCORING, "require_single_path": False, "tolerance_px": 1.5})
    assert relaxed.lines[0].any_coverage == 1.0


def test_region_restricts_scoring():
    ref = [_line(10, 0, 10, 100)]
    out = [np.array([[10, 0], [10, 60], [15, 100]], dtype=float)]  # wrong only on the right
    whole = score_paths(out, ref, SCORING)
    left = score_paths(out, ref, {**SCORING, "region": {"x0": 0, "y0": 0, "x1": 50, "y1": 20}})
    assert not whole.passed
    assert left.passed


def test_trim_ends():
    ref = [_line(10, 0, 10, 50)]
    out = [_line(10, 4, 10, 46)]  # skeleton shortened at both ends
    assert not score_paths(out, ref, SCORING).passed
    assert score_paths(out, ref, {**SCORING, "trim_ends_px": 5.0}).passed


def test_graph_metrics_two_junction_bridge():
    # Crossing extracted as four arms plus a bridge between two junction points
    j1, j2 = np.array([50.0, 45.0]), np.array([50.0, 55.0])
    arms = [np.array([[50, 0], j1]), np.array([[20, 30], j1]), np.array([[80, 70], j2]),
            np.array([[50, 99], j2]), np.array([j1, j2])]
    g = graph_metrics(arms, {"x0": 30, "y0": 30, "x1": 70, "y1": 70}, image_shape=(100, 100))
    assert g["path_count"] == 5
    assert g["junctions_in_region"] == 2
    assert g["short_paths_in_region"] == 1
    assert len(g["endpoints"]) == 2  # two arm ends on the image border are not counted


def test_no_output_paths():
    s = score_paths([], [_line(0, 0, 0, 10)], SCORING)
    assert not s.passed
    assert s.lines[0].reason == "no output paths"
