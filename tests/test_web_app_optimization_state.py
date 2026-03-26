import os
import time
import numpy as np
from io import BytesIO

import centerline_core

from centerline_web_app import (
    BENCHMARK_FILENAME_PREFIXES,
    CenterlineSession,
    SVG_VIEW_PATH_RENDER_LIMIT,
    _benchmark_metrics_response,
    _should_capture_benchmark_metrics,
    _should_force_pruning_debug,
    app,
    background_optimization,
    process_centerlines,
    sessions,
)


class _StubCircleSystem:
    def __init__(self, image, threshold, *args, **kwargs):
        self.image = image
        self.threshold = threshold

    def evaluate_path(self, path):
        return 1.0


def test_benchmark_filename_prefix_detection():
    assert _should_capture_benchmark_metrics(f"{BENCHMARK_FILENAME_PREFIXES[0]}sample.png") is True
    assert _should_capture_benchmark_metrics("ordinary_sample.png") is False


def test_debug_filename_marker_detection():
    assert _should_force_pruning_debug("_debug_sample.png") is True
    assert _should_force_pruning_debug("fixture_debug_line.png") is True
    assert _should_force_pruning_debug("ordinary_sample.png") is False


def test_process_centerlines_force_enables_pruning_debug_for_debug_filename():
    session = CenterlineSession("debug-pruning-session")
    session.original_filename = "fixture_debug_line.png"
    session.image = np.ones((16, 16), dtype=np.float32)
    session.image[4:12, 8] = 0.0
    session.display_image = session.image.copy()
    session.parameters.update({
        'dark_threshold': 0.5,
        'min_path_length': 2,
        'enable_pruning': True,
        'show_pruning_debug_grid': False,
        'enable_optimization': False,
    })

    results = process_centerlines(session)

    assert isinstance(results, dict)
    assert not results.get('error')
    assert isinstance(results.get('pruning_debug'), dict)
    assert results['pruning_debug']['summary']['path_count'] >= 1
    assert 'path_state_counts' in results['pruning_debug']['summary']

    first_path = results['pruning_debug']['paths'][0]
    assert 'original_path_index' in first_path
    assert 'path_state' in first_path
    assert 'changed' in first_path
    assert 'state_reasons' in first_path
    assert 'unchanged_reasons' in first_path
    assert 'pruning_config' in first_path

    if first_path['kept_vertex_guards']:
        assert 'criteria_explanations' in first_path['kept_vertex_guards'][0]


def test_execute_test_run_passes_fitting_parameters(monkeypatch):
    captured = {}

    def _fake_create_run(update_goldens=False, fixture_ids=None, fitting_parameters=None):
        captured["update_goldens"] = update_goldens
        captured["fixture_ids"] = fixture_ids
        captured["fitting_parameters"] = fitting_parameters
        return {
            "run_id": "run-123",
            "status": "queued",
            "created_at": "2026-03-18T00:00:00+00:00",
            "update_goldens": bool(update_goldens),
            "fixture_ids": fixture_ids or [],
            "fixture_count": len(fixture_ids or []),
            "fitting_parameters": fitting_parameters or {},
        }

    monkeypatch.setattr("centerline_web_app.create_run", _fake_create_run)

    client = app.test_client()
    response = client.post(
        "/api/test-runs/execute",
        json={
            "update_goldens": False,
            "fixture_ids": ["corner-l-shape"],
            "fitting_parameters": {
                "analysis_mode": "main_app",
                "main_app_enable_optimization": True,
                "enable_curve_fitting": True,
                "cubic_fit_tolerance": 1.25,
                "endpoint_tangent_strictness": 72,
                "force_orthogonal_as_lines": False,
            },
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert payload["run_id"] == "run-123"
    assert captured["fixture_ids"] == ["corner-l-shape"]
    assert captured["fitting_parameters"] == {
        "analysis_mode": "main_app",
        "main_app_enable_optimization": True,
        "enable_curve_fitting": True,
        "cubic_fit_tolerance": 1.25,
        "endpoint_tangent_strictness": 72,
        "force_orthogonal_as_lines": False,
    }


def test_test_ui_backend_normalizes_main_app_analysis_parameters():
    from test_ui_backend import _normalize_fitting_parameters

    normalized = _normalize_fitting_parameters({
        "analysis_mode": "main_app",
        "main_app_enable_optimization": True,
        "enable_curve_fitting": True,
        "cubic_fit_tolerance": 1.4,
        "endpoint_tangent_strictness": 66,
        "force_orthogonal_as_lines": True,
    })

    assert normalized == {
        "analysis_mode": "main_app",
        "main_app_enable_optimization": True,
        "enable_curve_fitting": True,
        "cubic_fit_tolerance": 1.4,
        "endpoint_tangent_strictness": 66.0,
        "force_orthogonal_as_lines": True,
    }


def test_execute_test_run_rejects_non_object_fitting_parameters():
    client = app.test_client()
    response = client.post(
        "/api/test-runs/execute",
        json={
            "update_goldens": False,
            "fixture_ids": ["corner-l-shape"],
            "fitting_parameters": ["bad-shape"],
        },
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "fitting_parameters must be an object"


def test_auto_tune_tile_preview_reuses_cached_tile_entries(monkeypatch):
    session = CenterlineSession("tile-preview-cache-session")
    session.image = np.ones((20, 20), dtype=np.float32)
    sessions[session.session_id] = session

    calls = []

    def _fake_extract_single_tile_preview_entry(
        gray_image,
        *,
        tile_index,
        origin_x,
        origin_y,
        tile_width,
        tile_height,
        dark_threshold,
        min_path_length,
        merge_gap=None,
        min_object_size=3,
        full_resolution=False,
    ):
        calls.append((tile_index, origin_x, origin_y, dark_threshold, min_path_length, merge_gap, min_object_size))
        tile_id = f"tile-{tile_index}-{origin_x}-{origin_y}-{tile_width}-{tile_height}"
        return {
            "tile_id": tile_id,
            "tile_index": tile_index,
            "valid_count": 1,
            "orphan_count": 0,
            "total_length": 4,
            "paths": [
                {
                    "path_id": f"{tile_id}-path-0",
                    "length": 4,
                    "points": [[origin_y, origin_x], [origin_y + 1, origin_x + 1]],
                }
            ],
        }

    monkeypatch.setattr("centerline_web_app._extract_single_tile_preview_entry", _fake_extract_single_tile_preview_entry)

    request_payload = {
        "session_id": session.session_id,
        "dark_threshold": 0.2,
        "min_path_length": 3,
        "sample_metadata": {
            "sampled": True,
            "sample_shape": [20, 20],
            "sample_origin": [0, 0],
            "source_shape": [20, 20],
            "sampling_mode": "random_tiles_mosaic",
            "overlay_supported": True,
            "tile_count": 2,
            "tile_shape": [10, 10],
            "tile_origins": [[0, 0], [10, 0]],
        },
    }

    try:
        client = app.test_client()
        first_response = client.post("/auto_tune_tile_preview", json=request_payload)
        second_response = client.post("/auto_tune_tile_preview", json=request_payload)

        assert first_response.status_code == 200
        assert second_response.status_code == 200
        assert len(calls) == 2

        first_payload = first_response.get_json()
        second_payload = second_response.get_json()

        assert first_payload["computed_tile_count"] == 2
        assert first_payload["cached_tile_count"] == 0
        assert second_payload["computed_tile_count"] == 0
        assert second_payload["cached_tile_count"] == 2
        assert second_payload["tile_path_count"] == 2
        assert second_payload["stats"]["cached_tile_count"] == 2
        assert second_payload["stats"]["computed_tile_count"] == 0
    finally:
        sessions.pop(session.session_id, None)


def test_benchmark_metrics_route_returns_filename_triggered_metrics():
    session = CenterlineSession("benchmark-session")
    session.original_filename = f"{BENCHMARK_FILENAME_PREFIXES[0]}sample.png"
    session.image_file_size = 321
    session.benchmark_enabled = True
    session.image = np.ones((8, 8), dtype=np.float32)
    session.benchmark_metrics = {
        'session_id': session.session_id,
        'filename': session.original_filename,
        'enabled': True,
        'created_at_epoch': float(session.created_at),
        'created_at_ms': int(round(session.created_at * 1000.0)),
        'upload_size_bytes': int(session.image_file_size),
        'preprocessing': {},
        'stages': {'upload': {'elapsed_ms': 12.5}},
        'latest_svg': {},
        'latest_download': {},
        'optimization': {},
    }

    sessions[session.session_id] = session
    try:
        client = app.test_client()
        response = client.get(f"/benchmark_metrics/{session.session_id}")

        assert response.status_code == 200
        payload = response.get_json()
        assert payload['enabled'] is True
        assert payload['filename'] == session.original_filename
        assert payload['upload_size_bytes'] == 321
        assert payload['image_shape'] == [8, 8]
        assert payload['summary']['current_phase'] == 'uploaded'
    finally:
        sessions.pop(session.session_id, None)


def test_benchmark_metrics_route_rejects_non_benchmark_session():
    session = CenterlineSession("non-benchmark-session")
    session.original_filename = "sample.png"
    sessions[session.session_id] = session
    try:
        client = app.test_client()
        response = client.get(f"/benchmark_metrics/{session.session_id}")
        assert response.status_code == 404
    finally:
        sessions.pop(session.session_id, None)


def test_upload_response_exposes_benchmark_metrics_url_for_triggered_filename():
    client = app.test_client()
    response = client.post(
        "/upload",
        data={
            'normalization_mode': 'off',
            'normalization_sensitivity': 'medium',
            'file': (BytesIO(
                b'\x89PNG\r\n\x1a\n'
                b'\x00\x00\x00\rIHDR'
                b'\x00\x00\x00\x01\x00\x00\x00\x01\x08\x00\x00\x00\x00'
                b'\x3a\x7e\x9b\x55'
                b'\x00\x00\x00\x0bIDATx\x9cc`\x00\x02\x00\x00\x05\x00\x01'
                b'\x0d\x0a\x2d\xb4'
                b'\x00\x00\x00\x00IEND\xaeB`\x82'
            ), f"{BENCHMARK_FILENAME_PREFIXES[0]}tiny.png"),
        },
        content_type='multipart/form-data',
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['benchmark_enabled'] is True
    assert payload['benchmark_metrics_url'].endswith(payload['session_id'])
    assert payload['normalization_applied'] is False
    assert 'raw_image_data' not in payload
    assert 'normalized_image_data' not in payload
    assert payload['image_data']

    session = sessions.pop(payload['session_id'], None)
    if session and session.image_path and os.path.exists(session.image_path):
        os.remove(session.image_path)


def test_upload_remote_fetches_http_image(monkeypatch):
    tiny_png = (
        b'\x89PNG\r\n\x1a\n'
        b'\x00\x00\x00\rIHDR'
        b'\x00\x00\x00\x01\x00\x00\x00\x01\x08\x00\x00\x00\x00'
        b'\x3a\x7e\x9b\x55'
        b'\x00\x00\x00\x0bIDATx\x9cc`\x00\x02\x00\x00\x05\x00\x01'
        b'\x0d\x0a\x2d\xb4'
        b'\x00\x00\x00\x00IEND\xaeB`\x82'
    )

    class _FakeRemoteResponse:
        def __init__(self, payload, url, content_type='image/png'):
            self._payload = payload
            self._url = url
            self.headers = {
                'Content-Type': content_type,
                'Content-Length': str(len(payload)),
            }

        def read(self, size=-1):
            if size is None or size < 0:
                return self._payload
            return self._payload[:size]

        def geturl(self):
            return self._url

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        'centerline_web_app.urlopen',
        lambda request, timeout=0: _FakeRemoteResponse(tiny_png, 'https://cdn.example.com/drop/test-image.png'),
    )

    client = app.test_client()
    response = client.post(
        '/upload_remote',
        json={
            'url': 'https://example.com/test-image.png',
            'normalization_mode': 'off',
            'normalization_sensitivity': 'medium',
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['success'] is True
    assert payload['image_data']
    assert payload['normalization_applied'] is False

    session = sessions.pop(payload['session_id'], None)
    if session and session.image_path and os.path.exists(session.image_path):
        os.remove(session.image_path)


def test_extract_tile_preview_paths_returns_all_kept_paths_without_rank(monkeypatch):
    source_image = np.zeros((12, 12), dtype=np.float32)
    sample_metadata = {
        'sampled': True,
        'sample_shape': [12, 12],
        'sample_origin': [0, 0],
        'source_shape': [12, 12],
        'sampling_mode': 'random_tiles_mosaic',
        'tile_count': 1,
        'tile_shape': [12, 12],
        'tile_origins': [[0, 0]],
    }

    extracted_paths = [
        [(0, 0), (0, 1), (0, 2), (0, 3)],
        [(3, 0), (3, 1), (3, 2), (3, 3)],
        [(6, 0), (6, 1), (6, 2), (6, 3)],
        [(9, 0), (9, 1)],
    ]

    monkeypatch.setattr('centerline_core.extract_skeleton_paths', lambda *args, **kwargs: list(extracted_paths))
    monkeypatch.setattr('centerline_core.merge_nearby_paths', lambda paths, **kwargs: list(paths))
    monkeypatch.setattr('centerline_core.prune_extracted_paths', lambda paths, **kwargs: list(paths))

    tile_data, stats = centerline_core._extract_tile_preview_paths(
        source_image,
        sample_metadata,
        dark_threshold=0.2,
        min_path_length=3,
        merge_gap=6,
        min_object_size=1,
        full_resolution=False,
    )

    assert len(tile_data) == 1
    tile_entry = tile_data[0]
    assert tile_entry['valid_count'] == 3
    assert tile_entry['orphan_count'] == 1
    assert len(tile_entry['paths']) == 3
    assert tile_entry['tile_id'].startswith('tile-1-0-0-12-12')
    assert all('rank' not in path_entry for path_entry in tile_entry['paths'])
    assert all(path_entry['path_id'].startswith(tile_entry['tile_id']) for path_entry in tile_entry['paths'])
    assert tile_entry['paths'][0]['points'] == [[0, 0], [0, 1], [0, 2], [0, 3]]
    assert stats['total_valid_paths'] == 3
    assert stats['total_orphan_paths'] == 1


def test_upload_remote_falls_back_from_html_page_to_og_image(monkeypatch):
    tiny_png = (
        b'\x89PNG\r\n\x1a\n'
        b'\x00\x00\x00\rIHDR'
        b'\x00\x00\x00\x01\x00\x00\x00\x01\x08\x00\x00\x00\x00'
        b'\x3a\x7e\x9b\x55'
        b'\x00\x00\x00\x0bIDATx\x9cc`\x00\x02\x00\x00\x05\x00\x01'
        b'\x0d\x0a\x2d\xb4'
        b'\x00\x00\x00\x00IEND\xaeB`\x82'
    )
    html_payload = b'''<!doctype html><html><head><meta property="og:image" content="https://cdn.example.com/assets/hero.png"></head></html>'''

    class _FakeRemoteResponse:
        def __init__(self, payload, url, content_type):
            self._payload = payload
            self._url = url
            self.headers = {
                'Content-Type': content_type,
                'Content-Length': str(len(payload)),
            }

        def read(self, size=-1):
            if size is None or size < 0:
                return self._payload
            return self._payload[:size]

        def geturl(self):
            return self._url

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def _fake_urlopen(request, timeout=0):
        url = request.full_url
        if url == 'https://example.com/gallery/item':
            return _FakeRemoteResponse(html_payload, url, 'text/html; charset=utf-8')
        if url == 'https://cdn.example.com/assets/hero.png':
            return _FakeRemoteResponse(tiny_png, url, 'image/png')
        raise AssertionError(f'Unexpected remote URL {url!r}')

    monkeypatch.setattr('centerline_web_app.urlopen', _fake_urlopen)

    client = app.test_client()
    response = client.post(
        '/upload_remote',
        json={
            'url': 'https://example.com/gallery/item',
            'normalization_mode': 'off',
            'normalization_sensitivity': 'medium',
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['success'] is True
    assert payload['image_data']

    session = sessions.pop(payload['session_id'], None)
    if session and session.image_path and os.path.exists(session.image_path):
        os.remove(session.image_path)


def test_upload_remote_rejects_non_http_scheme():
    client = app.test_client()
    response = client.post(
        '/upload_remote',
        json={
            'url': 'file:///tmp/example.png',
            'normalization_mode': 'off',
            'normalization_sensitivity': 'medium',
        },
    )

    assert response.status_code == 400
    assert response.get_json()['error'] == 'Only http and https URLs are supported'


def test_progress_reports_failed_optimization(monkeypatch):
    session = CenterlineSession("test-session")
    session.image = np.ones((8, 8), dtype=np.float32)
    session.initial_paths = [
        [[0, 0], [0, 1], [0, 2]],
    ]
    session.optimization_generation = 1

    monkeypatch.setattr("centerline_web_app.CircleEvaluationSystem", _StubCircleSystem)

    def _raise_during_optimization(path, circle_system, params, initial_score=None):
        raise RuntimeError("boom")

    monkeypatch.setattr("centerline_web_app.optimize_path_with_custom_params", _raise_during_optimization)

    sessions[session.session_id] = session
    try:
        background_optimization(session, 1)

        client = app.test_client()
        response = client.get(f"/progress/{session.session_id}")
        assert response.status_code == 200

        payload = response.get_json()
        assert payload["optimization_complete"] is False
        assert any("Optimization error: boom" in message for message in payload["messages"])
    finally:
        sessions.pop(session.session_id, None)


def test_progress_reports_live_preview_metadata():
    session = CenterlineSession("live-preview-progress")
    session.initial_paths = [
        [[0, 0], [0, 1], [0, 2]],
        [[1, 0], [1, 1], [1, 2]],
    ]
    session.live_preview_paths = [
        [[0, 0], [0, 2]],
    ]
    session.live_preview_kind = "merge"
    session.live_preview_frame_id = 4

    sessions[session.session_id] = session
    try:
        client = app.test_client()
        response = client.get(f"/progress/{session.session_id}")

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["live_preview_kind"] == "merge"
        assert payload["live_preview_frame_id"] == 4
        assert payload["live_preview_count"] == 1
        assert payload["live_preview_paths"] == [[[0.0, 0.0], [0.0, 2.0]]]
    finally:
        sessions.pop(session.session_id, None)


def test_progress_can_skip_live_preview_path_payload():
    session = CenterlineSession("live-preview-progress-lightweight")
    session.initial_paths = [
        [[0, 0], [0, 1], [0, 2]],
    ]
    session.live_preview_paths = [
        [[0, 0], [0, 2]],
    ]
    session.live_preview_kind = "merge"
    session.live_preview_frame_id = 4

    sessions[session.session_id] = session
    try:
        client = app.test_client()
        response = client.get(f"/progress/{session.session_id}?include_preview_paths=0")

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["live_preview_kind"] == "merge"
        assert payload["live_preview_frame_id"] == 4
        assert payload["live_preview_count"] == 1
        assert payload["live_preview_paths"] == []
    finally:
        sessions.pop(session.session_id, None)


def test_centerline_session_defaults_to_disabled_optimization():
    session = CenterlineSession("default-disabled-optimization")

    assert session.parameters["enable_optimization"] is False
    assert session.parameters["show_pre_optimization"] is True
    assert session.parameters["enable_pruning"] is False
    assert session.parameters["min_path_length"] == 2
    assert session.parameters["source_smoothing"] == 70.0
    assert session.parameters["cubic_fit_tolerance"] == 0.35


def test_process_immediate_reports_disabled_optimization_state(monkeypatch):
    session = CenterlineSession("immediate-disabled-state")
    session.image = np.zeros((16, 16), dtype=np.float32)

    def _fake_create_fast_paths(skeleton, min_path_length=1):
        return [
            [[0, 0], [0, 1], [0, 2], [0, 3]],
        ]

    monkeypatch.setattr("centerline_engine.create_fast_paths", _fake_create_fast_paths)

    sessions[session.session_id] = session
    try:
        client = app.test_client()
        response = client.post(
            "/process_immediate",
            json={
                "session_id": session.session_id,
                "parameters": {
                    "enable_optimization": False,
                    "enable_pruning": True,
                },
            },
        )

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["success"] is True
        assert payload["optimization_enabled"] is False
        assert payload["optimization_started"] is False
        assert {
            "extraction",
            "initial_preview_publish",
            "length_filter",
            "pruning",
            "final_path_serialization",
            "hint_generation",
            "final_preview_publish",
            "progress_queue_clear",
            "post_extraction",
            "response_assembly",
            "total_server",
        }.issubset(payload["timing_breakdown_ms"].keys())
        assert payload["pruning_timing_breakdown_ms"]["path_count"] == 1
        assert {
            "terminal_trim",
            "straighten",
            "anchor_selection",
            "corner_collapse",
            "minimum_output_guard",
            "geometry_rejection",
            "finalize",
        }.issubset(payload["pruning_timing_breakdown_ms"]["stage_ms"].keys())
        assert payload["timing_breakdown_ms"]["total_server"] >= payload["timing_breakdown_ms"]["extraction"]

        progress_response = client.get(f"/progress/{session.session_id}")
        assert progress_response.status_code == 200
        progress_payload = progress_response.get_json()
        assert progress_payload["live_preview_kind"] == "initial"
        assert progress_payload["live_preview_count"] == 1
        assert progress_payload["live_preview_paths"] == [[[0.0, 0.0], [0.0, 3.0]]]
    finally:
        sessions.pop(session.session_id, None)


def test_process_immediate_skips_path_simplification_when_disabled(monkeypatch):
    session = CenterlineSession("immediate-path-simplification-disabled")
    session.image = np.zeros((16, 16), dtype=np.float32)

    def _fake_create_fast_paths(skeleton, min_path_length=1):
        return [
            [[0, 0], [0, 1], [0, 2], [0, 3]],
        ]

    monkeypatch.setattr("centerline_engine.create_fast_paths", _fake_create_fast_paths)

    sessions[session.session_id] = session
    try:
        client = app.test_client()
        response = client.post(
            "/process_immediate",
            json={
                "session_id": session.session_id,
                "parameters": {
                    "enable_optimization": False,
                    "enable_pruning": False,
                },
            },
        )

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["success"] is True
        assert payload["path_simplification_enabled"] is False
        assert payload["paths"] == [[[0, 0], [0, 1], [0, 2], [0, 3]]]
        assert payload["pruning_debug"] is None

        progress_response = client.get(f"/progress/{session.session_id}")
        assert progress_response.status_code == 200
        progress_payload = progress_response.get_json()
        assert progress_payload["live_preview_kind"] == "initial"
        assert progress_payload["live_preview_count"] == 1
        assert progress_payload["live_preview_paths"] == [[[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.0, 3.0]]]
    finally:
        sessions.pop(session.session_id, None)


def test_process_centerlines_skips_pruning_when_path_simplification_disabled(monkeypatch):
    session = CenterlineSession("process-centerlines-pruning-disabled")
    session.original_filename = "fixture_debug_line.png"
    session.image = np.ones((16, 16), dtype=np.float32)
    session.display_image = session.image.copy()
    session.parameters.update({
        'dark_threshold': 0.5,
        'min_path_length': 2,
        'enable_optimization': False,
        'enable_pruning': False,
        'show_pruning_debug_grid': True,
    })

    monkeypatch.setattr(
        "centerline_engine.create_fast_paths",
        lambda *args, **kwargs: [[(0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (0.0, 3.0)]],
    )

    results = process_centerlines(session)

    assert isinstance(results, dict)
    assert not results.get('error')
    assert results['optimized_paths'] == [[(0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (0.0, 3.0)]]
    assert results['pre_optimization_paths'] == [[(0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (0.0, 3.0)]]
    assert results['pruning_debug'] is None


def test_auto_detect_threshold_success_keeps_optimization_disabled(monkeypatch):
    session = CenterlineSession("auto-detect-disabled-state")
    session.image = np.ones((32, 32), dtype=np.float32)

    monkeypatch.setattr(
        "centerline_web_app.auto_detect_dark_threshold",
        lambda *args, **kwargs: {
            "best_score": 0.84,
            "best_threshold": 0.31,
            "recommendation": "use detected threshold",
            "elapsed_sec": 0.12,
            "thresholds_evaluated": 8,
            "preview_shape": [32, 32],
            "preview_scale": 1.0,
            "sample_metadata": {"sampled": False, "source_shape": [32, 32]},
        },
    )

    sessions[session.session_id] = session
    try:
        client = app.test_client()
        response = client.post(
            "/auto_detect_threshold",
            json={"session_id": session.session_id},
        )

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["success"] is True
        assert payload["updated_parameters"]["enable_optimization"] is False
        assert payload["updated_parameters"]["show_pre_optimization"] is True
        assert session.parameters["enable_optimization"] is False
        assert session.parameters["show_pre_optimization"] is True
    finally:
        sessions.pop(session.session_id, None)


def test_auto_tune_extraction_success_keeps_optimization_disabled(monkeypatch):
    session = CenterlineSession("auto-tune-disabled-state")
    session.image = np.ones((32, 32), dtype=np.float32)

    monkeypatch.setattr(
        "centerline_web_app.auto_tune_extraction_parameters",
        lambda *args, **kwargs: {
            "best_threshold": 0.24,
            "best_min_length": 5,
            "confidence_score": 0.93,
            "quality_score": 0.88,
            "recommendation": "good fit",
            "preview_shape": [32, 32],
            "preview_scale": 1.0,
            "longest_path": 21,
            "valid_paths": 8,
            "sample_metadata": {"sampled": False, "source_shape": [32, 32]},
            "timed_out": False,
            "elapsed_sec": 0.11,
        },
    )
    monkeypatch.setattr("centerline_web_app._estimate_median_stroke_width", lambda *args, **kwargs: 2.5)

    sessions[session.session_id] = session
    try:
        client = app.test_client()
        response = client.post("/auto_tune_extraction", json={"session_id": session.session_id})

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["success"] is True
        assert payload["updated_parameters"]["enable_optimization"] is False
        assert payload["updated_parameters"]["show_pre_optimization"] is True
        assert session.parameters["enable_optimization"] is False
        assert session.parameters["show_pre_optimization"] is True
    finally:
        sessions.pop(session.session_id, None)


def test_generate_svg_immediate_disabled_optimization_uses_direct_single_layer(monkeypatch):
    session = CenterlineSession("svg-immediate-disabled-direct")
    session.image = np.ones((8, 8), dtype=np.float32)
    session.display_image = session.image
    session.initial_paths = [
        [[0, 0], [0, 1], [0, 2]],
    ]
    session.initial_path_hints = [[{'start_idx': 0, 'end_idx': 2, 'kind': 'line', 'confidence': 1.0}]]
    session.parameters.update({
        'enable_optimization': False,
        'show_pre_optimization': True,
    })

    captured = {}

    def _stub_create_svg_output(*args, **kwargs):
        captured['optimized_paths'] = kwargs.get('optimized_paths', args[2] if len(args) > 2 else None)
        captured['pre_optimization_paths'] = kwargs.get('pre_optimization_paths', args[4] if len(args) > 4 else None)
        captured['fit_optimized_paths'] = kwargs.get('fit_optimized_paths')
        return '<svg xmlns="http://www.w3.org/2000/svg"></svg>'

    monkeypatch.setattr('centerline_web_app.create_svg_output', _stub_create_svg_output)

    sessions[session.session_id] = session
    try:
        client = app.test_client()
        response = client.post(
            '/generate_svg',
            json={
                'session_id': session.session_id,
                'mode': 'immediate',
                'parameters': session.parameters,
            },
        )

        assert response.status_code == 200
        assert captured['optimized_paths'] == session.initial_paths
        assert captured['pre_optimization_paths'] == []
        assert captured['fit_optimized_paths'] is False
    finally:
        sessions.pop(session.session_id, None)


def test_download_svg_disabled_optimization_uses_direct_single_layer(monkeypatch):
    session = CenterlineSession("download-disabled-direct")
    session.image = np.ones((8, 8), dtype=np.float32)
    session.display_image = session.image
    session.original_filename = 'sample_input.png'
    session.initial_paths = [
        [[1, 1], [1, 2], [1, 3]],
    ]
    session.initial_path_hints = [[{'start_idx': 0, 'end_idx': 2, 'kind': 'line', 'confidence': 1.0}]]
    session.parameters.update({
        'enable_optimization': False,
        'show_pre_optimization': True,
    })

    captured = {}

    def _stub_create_svg_output(*args, **kwargs):
        captured['optimized_paths'] = kwargs.get('optimized_paths', args[2] if len(args) > 2 else None)
        captured['pre_optimization_paths'] = kwargs.get('pre_optimization_paths', args[4] if len(args) > 4 else None)
        captured['fit_optimized_paths'] = kwargs.get('fit_optimized_paths')
        return '<svg xmlns="http://www.w3.org/2000/svg"></svg>'

    monkeypatch.setattr('centerline_web_app.create_svg_output', _stub_create_svg_output)

    sessions[session.session_id] = session
    try:
        client = app.test_client()
        response = client.get(f'/download_svg/{session.session_id}')

        assert response.status_code == 200
        assert captured['optimized_paths'] == session.initial_paths
        assert captured['pre_optimization_paths'] == []
        assert captured['fit_optimized_paths'] is False
    finally:
        sessions.pop(session.session_id, None)


def test_download_svg_final_results_disabled_optimization_uses_direct_single_layer(monkeypatch):
    session = CenterlineSession("download-final-disabled-direct")
    session.image = np.ones((8, 8), dtype=np.float32)
    session.display_image = session.image
    session.original_filename = 'sample_input.png'
    session.initial_paths = []
    session.parameters.update({
        'enable_optimization': False,
        'show_pre_optimization': True,
    })
    session.results = {
        'pre_optimization_paths': [
            [[0, 0], [0, 1], [0, 2]],
        ],
        'optimized_paths': [
            [[1, 1], [1, 2], [1, 3]],
        ],
        'optimized_scores': [1.0],
        'optimized_path_hints': [[{'start_idx': 0, 'end_idx': 2, 'kind': 'line', 'confidence': 1.0}]],
        'circle_system': None,
    }

    captured = {}

    def _stub_create_svg_output(*args, **kwargs):
        captured['optimized_paths'] = kwargs.get('optimized_paths', args[2] if len(args) > 2 else None)
        captured['pre_optimization_paths'] = kwargs.get('pre_optimization_paths', args[4] if len(args) > 4 else None)
        captured['fit_optimized_paths'] = kwargs.get('fit_optimized_paths')
        return '<svg xmlns="http://www.w3.org/2000/svg"></svg>'

    monkeypatch.setattr('centerline_web_app.create_svg_output', _stub_create_svg_output)

    sessions[session.session_id] = session
    try:
        client = app.test_client()
        response = client.get(f'/download_svg/{session.session_id}')

        assert response.status_code == 200
        assert captured['optimized_paths'] == session.results['optimized_paths']
        assert captured['pre_optimization_paths'] == []
        assert captured['fit_optimized_paths'] is False
    finally:
        sessions.pop(session.session_id, None)


def test_generate_svg_disabled_optimization_can_enable_post_fit_export(monkeypatch):
    session = CenterlineSession("svg-disabled-post-fit-export")
    session.image = np.ones((8, 8), dtype=np.float32)
    session.display_image = session.image
    session.initial_paths = [
        [[0, 0], [0, 1], [0, 2]],
    ]
    session.initial_path_hints = [[{'start_idx': 0, 'end_idx': 2, 'kind': 'line', 'confidence': 1.0}]]
    session.parameters.update({
        'enable_optimization': False,
        'enable_post_fit_export': True,
        'source_smoothing': 38.0,
    })

    captured = {}

    def _stub_create_svg_output(*args, **kwargs):
        captured['fit_optimized_paths'] = kwargs.get('fit_optimized_paths')
        captured['source_smoothing'] = kwargs.get('source_smoothing')
        return '<svg xmlns="http://www.w3.org/2000/svg"></svg>'

    monkeypatch.setattr('centerline_web_app.create_svg_output', _stub_create_svg_output)

    sessions[session.session_id] = session
    try:
        client = app.test_client()
        response = client.post(
            '/generate_svg',
            json={
                'session_id': session.session_id,
                'mode': 'immediate',
                'parameters': session.parameters,
            },
        )

        assert response.status_code == 200
        assert captured['fit_optimized_paths'] is True
        assert captured['source_smoothing'] == 38.0
    finally:
        sessions.pop(session.session_id, None)


def test_progress_reports_merge_progress_metadata():
    session = CenterlineSession("merge-progress-state")
    session.initial_paths = [
        [[0, 0], [0, 1], [0, 2]],
        [[1, 0], [1, 1], [1, 2]],
        [[2, 0], [2, 1], [2, 2]],
    ]
    session.merge_progress = {
        'active': True,
        'phase': 'cheap_merge',
        'processed': 2,
        'total': 3,
        'percent': 66.7,
    }

    sessions[session.session_id] = session
    try:
        client = app.test_client()
        response = client.get(f"/progress/{session.session_id}")

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["merge_progress"] == {
            'active': True,
            'phase': 'cheap_merge',
            'processed': 2,
            'total': 3,
            'percent': 66.7,
        }
    finally:
        sessions.pop(session.session_id, None)


def test_auto_tune_progress_allows_running_state_past_90_seconds():
    session = CenterlineSession("auto-tune-running-90s")
    session.auto_tune_active = True
    session.auto_tune_progress.update({
        'running': True,
        'started_at': 0.0,
        'finished': False,
        'elapsed_sec': 91.2,
        'timed_out': False,
        'cancelled': False,
        'message': 'Still evaluating candidates',
    })

    sessions[session.session_id] = session
    try:
        client = app.test_client()
        response = client.get(f"/auto_tune_progress/{session.session_id}")

        assert response.status_code == 200
        payload = response.get_json()
        assert payload['running'] is True
        assert payload['finished'] is False
        assert payload['active'] is True
        assert payload['elapsed_sec'] == 91.2
        assert payload['timed_out'] is False
    finally:
        sessions.pop(session.session_id, None)


def test_auto_tune_progress_reports_terminal_timeout_from_server_state():
    session = CenterlineSession("auto-tune-timeout-finished")
    session.auto_tune_active = False
    session.auto_tune_progress.update({
        'running': False,
        'started_at': 0.0,
        'finished': True,
        'success': False,
        'elapsed_sec': 90.4,
        'timed_out': True,
        'cancelled': False,
        'message': 'Auto-tune completed within server time budget handling.',
    })

    sessions[session.session_id] = session
    try:
        client = app.test_client()
        response = client.get(f"/auto_tune_progress/{session.session_id}")

        assert response.status_code == 200
        payload = response.get_json()
        assert payload['running'] is False
        assert payload['finished'] is True
        assert payload['active'] is False
        assert payload['timed_out'] is True
        assert payload['elapsed_sec'] == 90.4
    finally:
        sessions.pop(session.session_id, None)


def test_auto_tune_sets_merge_gap_to_two_times_stroke_factor(monkeypatch):
    session = CenterlineSession("auto-tune-merge-gap")
    session.image = np.ones((32, 32), dtype=np.float32)

    monkeypatch.setattr(
        'centerline_web_app.auto_tune_extraction_parameters',
        lambda *args, **kwargs: {
            'best_threshold': 0.24,
            'best_min_length': 5,
            'confidence_score': 0.93,
            'quality_score': 0.88,
            'recommendation': 'good fit',
            'preview_shape': [32, 32],
            'preview_scale': 1.0,
            'longest_path': 21,
            'valid_paths': 8,
            'sample_metadata': {'sampled': False, 'source_shape': [32, 32]},
            'timed_out': False,
            'elapsed_sec': 0.11,
        },
    )
    monkeypatch.setattr('centerline_web_app._estimate_median_stroke_width', lambda *args, **kwargs: 2.5)

    sessions[session.session_id] = session
    try:
        client = app.test_client()
        response = client.post('/auto_tune_extraction', json={'session_id': session.session_id})

        assert response.status_code == 200
        payload = response.get_json()
        assert payload['success'] is True
        assert payload['updated_parameters']['merge_gap'] == 2
        assert payload['updated_parameters']['merge_gap_mode'] == 'stroke_factor'
        assert session.parameters['merge_gap'] == 2
        assert session.parameters['merge_gap_mode'] == 'stroke_factor'
    finally:
        sessions.pop(session.session_id, None)


class _ImmediateThread:
    def __init__(self, target=None, args=(), daemon=None):
        self._target = target
        self._args = args
        self._alive = False

    def start(self):
        self._alive = True
        try:
            if self._target is not None:
                self._target(*self._args)
        finally:
            self._alive = False

    def is_alive(self):
        return self._alive


def test_auto_detect_threshold_start_runs_background_job_and_updates_session(monkeypatch):
    session = CenterlineSession("auto-detect-start-session")
    session.image = np.ones((32, 32), dtype=np.float32)

    monkeypatch.setattr('centerline_web_app.threading.Thread', _ImmediateThread)
    monkeypatch.setattr(
        'centerline_web_app.auto_detect_dark_threshold',
        lambda *args, **kwargs: {
            'best_threshold': 0.31,
            'best_score': 0.72,
            'recommendation': 'high confidence',
            'elapsed_sec': 0.08,
            'thresholds_evaluated': 4,
            'preview_shape': [32, 32],
            'preview_scale': 1.0,
            'sample_metadata': {'sampled': False, 'source_shape': [32, 32]},
            'cancelled': False,
        },
    )

    sessions[session.session_id] = session
    try:
        client = app.test_client()
        response = client.post('/auto_detect_threshold_start', json={'session_id': session.session_id})

        assert response.status_code == 200
        assert response.get_json()['success'] is True
        assert session.parameters['dark_threshold'] == 0.31

        progress_response = client.get(f'/auto_detect_threshold_progress/{session.session_id}')
        assert progress_response.status_code == 200
        payload = progress_response.get_json()
        assert payload['finished'] is True
        assert payload['success'] is True
        assert payload['cancelled'] is False
        assert payload['detected_threshold'] == 0.31
        assert payload['updated_parameters']['dark_threshold'] == 0.31
    finally:
        sessions.pop(session.session_id, None)


def test_stop_auto_detect_marks_progress_cancelled_without_overwriting_threshold():
    session = CenterlineSession("auto-detect-stop-session")
    session.parameters['dark_threshold'] = 0.27
    session.auto_detect_active = True
    session.auto_detect_generation = 3
    session.auto_detect_progress.update({
        'running': True,
        'started_at': float(time.perf_counter() - 0.25),
        'finished': False,
        'cancelled': False,
        'message': 'Auto-detect started. Evaluating threshold candidates...',
    })

    sessions[session.session_id] = session
    try:
        client = app.test_client()
        response = client.post('/stop_auto_detect', json={'session_id': session.session_id})

        assert response.status_code == 200
        assert response.get_json()['success'] is True
        assert session.auto_detect_active is False
        assert session.auto_detect_generation == 4
        assert session.parameters['dark_threshold'] == 0.27

        progress_response = client.get(f'/auto_detect_threshold_progress/{session.session_id}')
        assert progress_response.status_code == 200
        payload = progress_response.get_json()
        assert payload['finished'] is True
        assert payload['cancelled'] is True
        assert payload['success'] is False
        assert payload['detected_threshold'] == 0.27
        assert payload['active'] is False
    finally:
        sessions.pop(session.session_id, None)


def test_background_optimization_records_optimizer_phase_metrics(monkeypatch):
    session = CenterlineSession("benchmark-opt-session")
    session.benchmark_enabled = True
    session.original_filename = f"{BENCHMARK_FILENAME_PREFIXES[0]}opt.png"
    session.image = np.ones((8, 8), dtype=np.float32)
    session.initial_paths = [
        [[0, 0], [0, 1], [0, 2], [0, 3]],
        [[1, 0], [1, 1], [1, 2], [1, 3]],
    ]
    session.optimization_generation = 1

    monkeypatch.setattr("centerline_web_app.CircleEvaluationSystem", _StubCircleSystem)
    monkeypatch.setattr("centerline_web_app.merge_nearby_paths", lambda paths, **kwargs: list(paths))

    def _diagnostic_optimizer(path, circle_system, params, initial_score=None, return_diagnostics=False):
        assert return_diagnostics is True
        return path, 1.0, {
            'phase_ms': {
                'score_evaluation': 1.5,
                'adaptive_rdp': 2.0,
            },
            'counts': {
                'score_evaluations': 3,
                'candidates_considered': 2,
                'candidates_accepted': 1,
            },
        }

    monkeypatch.setattr("centerline_web_app.optimize_path_with_custom_params", _diagnostic_optimizer)

    background_optimization(session, 1)

    optimization_metrics = session.benchmark_metrics['optimization']
    assert optimization_metrics['status'] == 'complete'
    assert optimization_metrics['input_path_count'] == 2
    assert optimization_metrics['optimizer_phase_ms']['score_evaluation'] == 3.0
    assert optimization_metrics['optimizer_phase_ms']['adaptive_rdp'] == 4.0
    assert optimization_metrics['optimizer_counts']['score_evaluations'] == 6
    assert optimization_metrics['optimizer_counts']['candidates_considered'] == 4
    assert optimization_metrics['optimizer_counts']['candidates_accepted'] == 2
    assert session.results is not None
    assert len(session.results['optimized_paths']) == 2
    assert session.optimization_complete is True


def test_generate_svg_final_works_immediately_after_background_optimization(monkeypatch):
    session = CenterlineSession("svg-final-after-background-optimization")
    session.image = np.ones((8, 8), dtype=np.float32)
    session.display_image = session.image
    session.initial_paths = [
        [[0, 0], [0, 1], [0, 2], [0, 3]],
        [[1, 0], [1, 1], [1, 2], [1, 3]],
    ]
    session.optimization_generation = 1
    session.parameters['show_pre_optimization'] = True

    monkeypatch.setattr("centerline_web_app.CircleEvaluationSystem", _StubCircleSystem)
    monkeypatch.setattr("centerline_web_app.merge_nearby_paths", lambda paths, **kwargs: list(paths))
    monkeypatch.setattr(
        "centerline_web_app.optimize_path_with_custom_params",
        lambda path, circle_system, params, initial_score=None, return_diagnostics=False: (path, 1.0, {'phase_ms': {}, 'counts': {}})
        if return_diagnostics else (path, 1.0),
    )

    sessions[session.session_id] = session
    try:
        background_optimization(session, 1)

        client = app.test_client()
        response = client.post(
            "/generate_svg",
            json={
                "session_id": session.session_id,
                "mode": "final",
                "parameters": session.parameters,
            },
        )

        assert response.status_code == 200
        payload = response.get_json()
        assert "svg" in payload
        assert 'stroke="#0066CC"' in payload["svg"]
    finally:
        sessions.pop(session.session_id, None)


def test_benchmark_response_helper_returns_none_when_disabled():
    session = CenterlineSession("helper-disabled")
    assert _benchmark_metrics_response(session) is None


def test_benchmark_response_helper_includes_compact_summary():
    session = CenterlineSession("helper-summary")
    session.benchmark_enabled = True
    session.original_filename = f"{BENCHMARK_FILENAME_PREFIXES[0]}summary.png"
    session.created_at = 100.0
    session.benchmark_metrics = {
        'session_id': session.session_id,
        'filename': session.original_filename,
        'enabled': True,
        'created_at_epoch': 100.0,
        'created_at_ms': 100000,
        'upload_size_bytes': 123,
        'preprocessing': {},
        'stages': {
            'load_and_preprocess': {'elapsed_ms': 25.0, 'recorded_at_ms': 100100},
            'preview_image_encoding': {'elapsed_ms': 10.0, 'recorded_at_ms': 100120},
        },
        'latest_svg': {},
        'latest_download': {},
        'optimization': {'status': 'complete', 'elapsed_ms': 50.0, 'recorded_at_ms': 100300},
    }
    session.optimization_complete = True

    payload = _benchmark_metrics_response(session)

    assert payload is not None
    assert payload['summary']['current_phase'] == 'optimization_complete'
    assert payload['summary']['wall_clock_ms'] == 300.0
    assert payload['summary']['recorded_stage_ms'] == 35.0
    assert payload['summary']['optimization_ms'] == 50.0


def test_generate_svg_live_preview_includes_blue_when_under_limit():
    session = CenterlineSession("svg-preview-session")
    session.image = np.ones((8, 8), dtype=np.float32)
    session.display_image = session.image
    session.initial_paths = [
        [[0, 0], [0, 1], [0, 2]],
    ]
    session.partial_optimized_paths = [
        [[1, 1], [1, 2], [1, 3]],
    ]
    session.parameters['enable_optimization'] = True
    session.parameters['show_pre_optimization'] = True

    sessions[session.session_id] = session
    try:
        client = app.test_client()

        progressive_response = client.post(
            "/generate_svg",
            json={
                "session_id": session.session_id,
                "mode": "progressive",
                "parameters": session.parameters,
            },
        )
        assert progressive_response.status_code == 200
        progressive_payload = progressive_response.get_json()
        assert "svg" in progressive_payload
        assert 'stroke="magenta"' in progressive_payload["svg"]
        assert 'stroke="#0066CC"' in progressive_payload["svg"]

        session.results = {
            'pre_optimization_paths': session.initial_paths,
            'optimized_paths': session.partial_optimized_paths,
            'optimized_scores': [1.0],
            'circle_system': None,
        }
        session.optimization_complete = True

        final_response = client.post(
            "/generate_svg",
            json={
                "session_id": session.session_id,
                "mode": "final",
                "parameters": session.parameters,
            },
        )
        assert final_response.status_code == 200
        final_payload = final_response.get_json()
        assert "svg" in final_payload
        assert 'stroke="magenta"' in final_payload["svg"]
        assert 'stroke="#0066CC"' in final_payload["svg"]
    finally:
        sessions.pop(session.session_id, None)


def test_generate_svg_progressive_uses_live_merge_preview_before_optimization():
    session = CenterlineSession("svg-merge-preview-session")
    session.image = np.ones((8, 8), dtype=np.float32)
    session.display_image = session.image
    session.initial_paths = [
        [[0, 0], [0, 1], [0, 2]],
    ]
    session.live_preview_paths = [
        [[1, 1], [1, 2], [1, 3]],
    ]
    session.live_preview_kind = "merge"
    session.live_preview_frame_id = 1
    session.parameters['enable_optimization'] = True
    session.parameters['show_pre_optimization'] = True

    sessions[session.session_id] = session
    try:
        client = app.test_client()

        progressive_response = client.post(
            "/generate_svg",
            json={
                "session_id": session.session_id,
                "mode": "progressive",
                "parameters": session.parameters,
            },
        )
        assert progressive_response.status_code == 200
        progressive_payload = progressive_response.get_json()
        assert "svg" in progressive_payload
        assert 'stroke="magenta"' in progressive_payload["svg"]
        assert 'stroke="#0066CC"' in progressive_payload["svg"]
    finally:
        sessions.pop(session.session_id, None)


def test_generate_svg_progressive_stitches_optimized_prefix_with_live_preview_tail(monkeypatch):
    session = CenterlineSession("svg-progressive-stitch-session")
    session.image = np.ones((8, 8), dtype=np.float32)
    session.display_image = session.image
    session.initial_paths = [
        [[0, 0], [0, 1], [0, 2]],
        [[2, 0], [2, 1], [2, 2]],
    ]
    session.live_preview_paths = [
        [[10, 10], [10, 11], [10, 12]],
        [[20, 20], [20, 21], [20, 22]],
    ]
    session.partial_optimized_paths = [
        [[11, 10], [11, 11], [11, 12]],
    ]
    session.live_preview_kind = "optimization"
    session.live_preview_frame_id = 2
    session.parameters['enable_optimization'] = True
    session.parameters['show_pre_optimization'] = True

    captured = {}

    def _stub_create_svg_output(*args, **kwargs):
        captured['optimized_paths'] = kwargs.get('optimized_paths', args[2] if len(args) > 2 else None)
        return '<svg xmlns="http://www.w3.org/2000/svg"></svg>'

    monkeypatch.setattr('centerline_web_app.create_svg_output', _stub_create_svg_output)

    sessions[session.session_id] = session
    try:
        client = app.test_client()
        response = client.post(
            '/generate_svg',
            json={
                'session_id': session.session_id,
                'mode': 'progressive',
                'parameters': session.parameters,
            },
        )

        assert response.status_code == 200
        assert captured['optimized_paths'] == [
            [[11, 10], [11, 11], [11, 12]],
            [[20, 20], [20, 21], [20, 22]],
        ]
    finally:
        sessions.pop(session.session_id, None)


def test_download_svg_includes_completed_blue_and_selected_magenta_and_image():
    session = CenterlineSession("download-svg-session")
    session.image = np.ones((8, 8), dtype=np.float32)
    session.display_image = session.image
    session.original_filename = "sample_input.png"
    session.initial_paths = [
        [[0, 0], [0, 1], [0, 2]],
    ]
    session.partial_optimized_paths = [
        [[1, 1], [1, 2], [1, 3]],
    ]
    session.results = {
        'pre_optimization_paths': session.initial_paths,
        'optimized_paths': session.partial_optimized_paths,
        'optimized_scores': [1.0],
        'circle_system': None,
    }
    session.optimization_complete = True
    session.parameters['enable_optimization'] = True
    session.parameters['show_pre_optimization'] = True
    session.parameters['include_image'] = True

    sessions[session.session_id] = session
    try:
        client = app.test_client()
        response = client.get(f"/download_svg/{session.session_id}")

        assert response.status_code == 200
        svg_text = response.data.decode("utf-8")
        assert 'stroke="#0066CC"' in svg_text
        assert 'stroke="magenta"' in svg_text
        assert '<image' in svg_text
    finally:
        sessions.pop(session.session_id, None)


def test_download_svg_query_overrides_include_image(monkeypatch):
    session = CenterlineSession("download-svg-include-image-override")
    session.image = np.ones((8, 8), dtype=np.float32)
    session.display_image = session.image
    session.original_filename = "sample_input.png"
    session.initial_paths = [
        [[0, 0], [0, 1], [0, 2]],
    ]
    session.parameters['enable_optimization'] = False
    session.parameters['show_pre_optimization'] = True
    session.parameters['include_image'] = False

    captured = {}

    def _stub_create_svg_output(*args, **kwargs):
        captured['optimized_paths'] = kwargs.get('optimized_paths', args[2] if len(args) > 2 else None)
        return '<svg xmlns="http://www.w3.org/2000/svg"></svg>'

    monkeypatch.setattr('centerline_web_app.create_svg_output', _stub_create_svg_output)

    sessions[session.session_id] = session
    try:
        client = app.test_client()
        response = client.get(f"/download_svg/{session.session_id}?include_image=1")

        assert response.status_code == 200
        assert captured['optimized_paths'] == session.initial_paths
    finally:
        sessions.pop(session.session_id, None)


def test_download_svg_prefers_final_results_over_empty_progressive_state_when_complete(monkeypatch):
    session = CenterlineSession("download-prefers-final-results")
    session.image = np.ones((8, 8), dtype=np.float32)
    session.display_image = session.image
    session.original_filename = "sample_input.png"
    session.initial_paths = [
        [[0, 0], [0, 1], [0, 2]],
    ]
    session.partial_optimized_paths = []
    session.partial_optimized_path_hints = []
    session.optimization_complete = True
    session.parameters['enable_optimization'] = True
    session.parameters['show_pre_optimization'] = True
    session.results = {
        'pre_optimization_paths': session.initial_paths,
        'optimized_paths': [
            [[1, 1], [1, 2], [1, 3]],
        ],
        'optimized_scores': [1.0],
        'optimized_path_hints': [[{'start_idx': 0, 'end_idx': 2, 'kind': 'line', 'confidence': 1.0}]],
        'circle_system': None,
    }

    captured = {}

    def _stub_create_svg_output(*args, **kwargs):
        captured['optimized_paths'] = kwargs.get('optimized_paths', args[2] if len(args) > 2 else None)
        captured['pre_optimization_paths'] = kwargs.get('pre_optimization_paths', args[4] if len(args) > 4 else None)
        return '<svg xmlns="http://www.w3.org/2000/svg"></svg>'

    monkeypatch.setattr('centerline_web_app.create_svg_output', _stub_create_svg_output)

    sessions[session.session_id] = session
    try:
        client = app.test_client()
        response = client.get(f'/download_svg/{session.session_id}')

        assert response.status_code == 200
        assert captured['optimized_paths'] == session.results['optimized_paths']
        assert captured['pre_optimization_paths'] == session.results['pre_optimization_paths']
    finally:
        sessions.pop(session.session_id, None)


def test_generate_svg_samples_preview_when_detected_paths_exceed_limit():
    session = CenterlineSession("svg-preview-limit-session")
    session.image = np.ones((8, 8), dtype=np.float32)
    session.display_image = session.image
    session.initial_paths = [
        [[0, 0], [0, 1], [0, 2]]
        for _ in range(SVG_VIEW_PATH_RENDER_LIMIT + 1)
    ]
    session.parameters['enable_optimization'] = True
    session.parameters['show_pre_optimization'] = True
    session.results = {
        'pre_optimization_paths': session.initial_paths,
        'optimized_paths': [
            [[1, 1], [1, 2], [1, 3]],
        ],
        'optimized_scores': [1.0],
        'circle_system': None,
    }
    session.optimization_complete = True

    sessions[session.session_id] = session
    try:
        client = app.test_client()
        response = client.post(
            "/generate_svg",
            json={
                "session_id": session.session_id,
                "mode": "final",
                "parameters": session.parameters,
            },
        )

        assert response.status_code == 200
        payload = response.get_json()
        assert "svg" in payload
        assert payload["preview_paths_suppressed"] is True
        assert payload["detected_paths_count"] == SVG_VIEW_PATH_RENDER_LIMIT + 1
        assert payload["rendered_path_count"] == SVG_VIEW_PATH_RENDER_LIMIT + 1
        assert 'stroke="#0066CC"' in payload["svg"]
        assert 'stroke="magenta"' in payload["svg"]
    finally:
        sessions.pop(session.session_id, None)


def test_generate_svg_reuses_cached_output_for_identical_request(monkeypatch):
    session = CenterlineSession("svg-cache-session")
    session.image = np.ones((8, 8), dtype=np.float32)
    session.display_image = session.image
    session.initial_paths = [
        [[0, 0], [0, 1], [0, 2]],
    ]
    session.partial_optimized_paths = [
        [[1, 1], [1, 2], [1, 3]],
    ]
    session.parameters['enable_optimization'] = True
    session.parameters['show_pre_optimization'] = True

    render_calls = {'count': 0}

    def _stub_create_svg_output(*args, **kwargs):
        render_calls['count'] += 1
        return '<svg xmlns="http://www.w3.org/2000/svg"></svg>'

    monkeypatch.setattr('centerline_web_app.create_svg_output', _stub_create_svg_output)

    sessions[session.session_id] = session
    try:
        client = app.test_client()
        request_payload = {
            'session_id': session.session_id,
            'mode': 'progressive',
            'parameters': session.parameters,
        }

        first_response = client.post('/generate_svg', json=request_payload)
        second_response = client.post('/generate_svg', json=request_payload)

        assert first_response.status_code == 200
        assert second_response.status_code == 200
        assert first_response.get_json()['svg'] == second_response.get_json()['svg']
        assert render_calls['count'] == 1
    finally:
        sessions.pop(session.session_id, None)


def test_generate_svg_cache_invalidates_when_geometry_changes_with_same_counts(monkeypatch):
    session = CenterlineSession("svg-cache-geometry-session")
    session.image = np.ones((8, 8), dtype=np.float32)
    session.display_image = session.image
    session.initial_paths = [
        [[0, 0], [0, 1], [0, 2]],
    ]
    session.partial_optimized_paths = [
        [[1, 1], [1, 2], [1, 3]],
    ]
    session.parameters['enable_optimization'] = True
    session.parameters['show_pre_optimization'] = True

    render_calls = {'count': 0}

    def _stub_create_svg_output(*args, **kwargs):
        render_calls['count'] += 1
        return f'<svg xmlns="http://www.w3.org/2000/svg"><!-- {render_calls["count"]} --></svg>'

    monkeypatch.setattr('centerline_web_app.create_svg_output', _stub_create_svg_output)

    sessions[session.session_id] = session
    try:
        client = app.test_client()
        request_payload = {
            'session_id': session.session_id,
            'mode': 'progressive',
            'parameters': session.parameters,
        }

        first_response = client.post('/generate_svg', json=request_payload)
        session.partial_optimized_paths = [
            [[2, 1], [2, 2], [2, 3]],
        ]
        second_response = client.post('/generate_svg', json=request_payload)

        assert first_response.status_code == 200
        assert second_response.status_code == 200
        assert render_calls['count'] == 2
    finally:
        sessions.pop(session.session_id, None)


def test_generate_svg_defaults_curve_fitting_off(monkeypatch):
    session = CenterlineSession("svg-default-curve-fit-off")
    session.image = np.ones((8, 8), dtype=np.float32)
    session.display_image = session.image
    session.parameters.update({
        'enable_optimization': True,
        'show_pre_optimization': False,
    })
    session.partial_optimized_paths = [
        [[1, 1], [1, 2], [1, 3]],
    ]
    session.results = {
        'pre_optimization_paths': [],
        'optimized_paths': session.partial_optimized_paths,
        'optimized_scores': [1.0],
        'circle_system': None,
    }
    session.optimization_complete = True

    captured = {}

    def _stub_create_svg_output(*args, **kwargs):
        captured['enable_curve_fitting'] = kwargs.get('enable_curve_fitting')
        captured['fit_optimized_paths'] = kwargs.get('fit_optimized_paths')
        return '<svg xmlns="http://www.w3.org/2000/svg"></svg>'

    monkeypatch.setattr('centerline_web_app.create_svg_output', _stub_create_svg_output)

    sessions[session.session_id] = session
    try:
        client = app.test_client()
        response = client.post(
            '/generate_svg',
            json={
                'session_id': session.session_id,
                'mode': 'final',
                'parameters': session.parameters,
            },
        )

        assert response.status_code == 200
        assert captured['enable_curve_fitting'] is False
        assert captured['fit_optimized_paths'] is True
    finally:
        sessions.pop(session.session_id, None)


def test_generate_svg_preview_cache_invalidates_when_curve_fitting_toggles(monkeypatch):
    session = CenterlineSession("svg-curve-fit-cache-toggle")
    session.image = np.ones((8, 8), dtype=np.float32)
    session.display_image = session.image
    session.partial_optimized_paths = [
        [[1, 1], [1, 2], [1, 3]],
    ]
    session.results = {
        'pre_optimization_paths': [],
        'optimized_paths': session.partial_optimized_paths,
        'optimized_scores': [1.0],
        'circle_system': None,
    }
    session.optimization_complete = True

    render_calls = {'count': 0}

    def _stub_create_svg_output(*args, **kwargs):
        render_calls['count'] += 1
        return '<svg xmlns="http://www.w3.org/2000/svg"></svg>'

    monkeypatch.setattr('centerline_web_app.create_svg_output', _stub_create_svg_output)

    sessions[session.session_id] = session
    try:
        client = app.test_client()

        curve_off_payload = {
            'session_id': session.session_id,
            'mode': 'final',
            'parameters': {**session.parameters, 'enable_curve_fitting': False},
        }
        curve_on_payload = {
            'session_id': session.session_id,
            'mode': 'final',
            'parameters': {**session.parameters, 'enable_curve_fitting': True},
        }

        first_response = client.post('/generate_svg', json=curve_off_payload)
        second_response = client.post('/generate_svg', json=curve_on_payload)

        assert first_response.status_code == 200
        assert second_response.status_code == 200
        assert render_calls['count'] == 2
    finally:
        sessions.pop(session.session_id, None)


def test_generate_svg_view_and_download_share_render_settings(monkeypatch):
    session = CenterlineSession("svg-view-download-settings-match")
    session.image = np.ones((8, 8), dtype=np.float32)
    session.display_image = session.image
    session.original_filename = "sample_input.png"
    session.partial_optimized_paths = [
        [[1, 1], [1, 2], [1, 3]],
    ]
    session.initial_paths = [
        [[0, 0], [0, 1], [0, 2]],
    ]
    session.parameters.update({
        'show_pre_optimization': True,
        'include_image': True,
        'enable_post_fit_export': True,
        'source_smoothing': 42.0,
        'enable_curve_fitting': True,
        'force_orthogonal_as_lines': True,
        'cubic_fit_tolerance': 0.65,
        'endpoint_tangent_strictness': 93.0,
    })

    captured_calls = []

    def _stub_create_svg_output(*args, **kwargs):
        captured_calls.append({
            'source_smoothing': kwargs.get('source_smoothing'),
            'curve_fit_tolerance': kwargs.get('curve_fit_tolerance'),
            'endpoint_tangent_strictness': kwargs.get('endpoint_tangent_strictness'),
            'force_orthogonal_as_lines': kwargs.get('force_orthogonal_as_lines'),
            'enable_curve_fitting': kwargs.get('enable_curve_fitting'),
            'fit_optimized_paths': kwargs.get('fit_optimized_paths'),
            'combine_optimized_paths': kwargs.get('combine_optimized_paths'),
            'combine_pre_optimization_paths': kwargs.get('combine_pre_optimization_paths'),
            'coordinate_precision': kwargs.get('coordinate_precision'),
        })
        return '<svg xmlns="http://www.w3.org/2000/svg"></svg>'

    monkeypatch.setattr('centerline_web_app.create_svg_output', _stub_create_svg_output)

    sessions[session.session_id] = session
    try:
        client = app.test_client()

        view_response = client.post(
            '/generate_svg',
            json={
                'session_id': session.session_id,
                'mode': 'progressive',
                'parameters': session.parameters,
            },
        )
        download_response = client.get(f'/download_svg/{session.session_id}')

        assert view_response.status_code == 200
        assert 'svg' in view_response.get_json()
        assert download_response.status_code == 200
        assert len(captured_calls) == 2
        assert captured_calls[0]['source_smoothing'] == 42.0
        assert captured_calls[1]['source_smoothing'] == 42.0
        assert captured_calls[0] == captured_calls[1]
    finally:
        sessions.pop(session.session_id, None)