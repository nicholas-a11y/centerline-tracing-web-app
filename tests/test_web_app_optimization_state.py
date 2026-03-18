import os
import numpy as np
from io import BytesIO

from centerline_web_app import (
    BENCHMARK_FILENAME_PREFIXES,
    CenterlineSession,
    SVG_VIEW_PATH_RENDER_LIMIT,
    _benchmark_metrics_response,
    _should_capture_benchmark_metrics,
    app,
    background_optimization,
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
        "enable_curve_fitting": True,
        "cubic_fit_tolerance": 1.25,
        "endpoint_tangent_strictness": 72,
        "force_orthogonal_as_lines": False,
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


def test_generate_svg_suppresses_blue_preview_when_detected_paths_exceed_limit():
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
        assert 'stroke="#0066CC"' not in payload["svg"]
        assert 'stroke="magenta"' not in payload["svg"]
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
        import centerline_engine as engine

        render_calls['count'] += 1
        with open(engine.OUTPUT_PATH, 'w') as file_handle:
            file_handle.write('<svg xmlns="http://www.w3.org/2000/svg"></svg>')

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


def test_generate_svg_defaults_curve_fitting_off(monkeypatch):
    session = CenterlineSession("svg-default-curve-fit-off")
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

    captured = {}

    def _stub_create_svg_output(*args, **kwargs):
        import centerline_engine as engine

        captured['enable_curve_fitting'] = kwargs.get('enable_curve_fitting')
        with open(engine.OUTPUT_PATH, 'w') as file_handle:
            file_handle.write('<svg xmlns="http://www.w3.org/2000/svg"></svg>')

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
    finally:
        sessions.pop(session.session_id, None)


def test_generate_svg_cache_key_changes_when_curve_fitting_toggles(monkeypatch):
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
        import centerline_engine as engine

        render_calls['count'] += 1
        with open(engine.OUTPUT_PATH, 'w') as file_handle:
            file_handle.write('<svg xmlns="http://www.w3.org/2000/svg"></svg>')

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