#!/usr/bin/env python3
"""Backend utilities for test management and layered fixture visualization."""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
from typing import Any

import numpy as np

from centerline_engine import extract_skeleton_paths, fit_curve_segments
from tests.conftest import (
    ALL_FIXTURE_IDS,
    CURVE_FIXTURES,
    LINE_FIXTURES,
    analytical_deviation,
    sample_fitted_segments,
)
from tests.fixtures import load_fixture


REPO_ROOT = Path(__file__).resolve().parent
RUNS_ROOT = REPO_ROOT / "tests" / "fixtures" / "runs"
FIXTURE_SIZE = 64
FIXTURE_SW = 3
DARK_THRESHOLD = 0.5
IDEAL_SAMPLE_N = 400
DEFAULT_FITTING_PARAMETERS = {
    "analysis_mode": "main_app",
    "main_app_enable_optimization": False,
    "enable_curve_fitting": False,
    "cubic_fit_tolerance": 0.35,
    "endpoint_tangent_strictness": 85.0,
    "force_orthogonal_as_lines": False,
}


@dataclass
class TestRunState:
    run_id: str
    created_at: str
    artifact_dir: Path
    status: str = "queued"
    started_at: str | None = None
    finished_at: str | None = None
    progress_messages: Queue = field(default_factory=Queue)
    thread: threading.Thread | None = None
    summary: dict[str, Any] | None = None
    fixture_ids: list[str] = field(default_factory=list)
    fixture_count: int = 0
    completed_fixtures: int = 0
    current_fixture: str | None = None
    pytest_exit_code: int | None = None
    fixture_results: dict[str, str] = field(default_factory=dict)
    fitting_parameters: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_FITTING_PARAMETERS))


_RUNS: dict[str, TestRunState] = {}
_RUNS_LOCK = threading.Lock()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _make_run_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def _normalize_fixture_selection(fixture_ids: list[str] | None) -> list[str]:
    if fixture_ids is None:
        return list(ALL_FIXTURE_IDS)

    seen: set[str] = set()
    selected: list[str] = []
    for fid in fixture_ids:
        if not isinstance(fid, str):
            continue
        value = fid.strip()
        if not value or value in seen:
            continue
        if value in ALL_FIXTURE_IDS:
            selected.append(value)
            seen.add(value)
    return selected


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _coerce_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, number))


def _normalize_fitting_parameters(fitting_parameters: dict[str, Any] | None) -> dict[str, Any]:
    if fitting_parameters is None:
        return dict(DEFAULT_FITTING_PARAMETERS)
    if not isinstance(fitting_parameters, dict):
        raise ValueError("fitting_parameters must be an object")

    defaults = DEFAULT_FITTING_PARAMETERS
    raw_mode = str(fitting_parameters.get("analysis_mode", defaults["analysis_mode"]) or defaults["analysis_mode"]).strip().lower()
    analysis_mode = raw_mode if raw_mode in {"curve_fitter", "main_app"} else defaults["analysis_mode"]
    return {
        "analysis_mode": analysis_mode,
        "main_app_enable_optimization": _coerce_bool(
            fitting_parameters.get("main_app_enable_optimization"),
            defaults["main_app_enable_optimization"],
        ),
        "enable_curve_fitting": _coerce_bool(
            fitting_parameters.get("enable_curve_fitting"),
            defaults["enable_curve_fitting"],
        ),
        "cubic_fit_tolerance": _coerce_float(
            fitting_parameters.get("cubic_fit_tolerance"),
            defaults["cubic_fit_tolerance"],
            0.35,
            4.0,
        ),
        "endpoint_tangent_strictness": _coerce_float(
            fitting_parameters.get("endpoint_tangent_strictness"),
            defaults["endpoint_tangent_strictness"],
            0.0,
            100.0,
        ),
        "force_orthogonal_as_lines": _coerce_bool(
            fitting_parameters.get("force_orthogonal_as_lines"),
            defaults["force_orthogonal_as_lines"],
        ),
    }


def _fixture_pass_fail_from_results(
    fixture_ids: list[str], node_results: dict[str, str]
) -> dict[str, str]:
    """Derive per-fixture pass/fail from pytest node results."""
    results: dict[str, str] = {}
    for fid in fixture_ids:
        statuses = [st for node_id, st in node_results.items() if f"[{fid}]" in node_id]
        if not statuses:
            results[fid] = "unknown"
        elif "failed" in statuses:
            results[fid] = "failed"
        else:
            results[fid] = "passed"
    return results


def _run_pytest(
    update_goldens: bool,
    fixture_ids: list[str],
    fitting_parameters: dict[str, Any],
) -> dict[str, Any]:
    analysis_mode = str(fitting_parameters.get("analysis_mode", "curve_fitter"))
    if analysis_mode == "main_app":
        cmd = [
            ".venv/bin/python",
            "-m",
            "pytest",
            "tests/test_web_app_optimization_state.py",
            "tests/test_path_optimization.py",
            "tests/test_engine_path_thresholds.py",
            "-q",
            "--tb=short",
        ]
    else:
        cmd = [".venv/bin/python", "-m", "pytest", "tests/test_fit_curves.py", "-v", "--tb=short"]
    if update_goldens and analysis_mode != "main_app":
        cmd.append("--update-goldens")

    env = os.environ.copy()
    if fixture_ids:
        env["TEST_UI_FIXTURE_IDS"] = ",".join(fixture_ids)
    env["TEST_UI_ENABLE_CURVE_FITTING"] = "1" if fitting_parameters["enable_curve_fitting"] else "0"
    env["TEST_UI_CUBIC_FIT_TOLERANCE"] = str(fitting_parameters["cubic_fit_tolerance"])
    env["TEST_UI_ENDPOINT_TANGENT_STRICTNESS"] = str(fitting_parameters["endpoint_tangent_strictness"])
    env["TEST_UI_FORCE_ORTHOGONAL_AS_LINES"] = "1" if fitting_parameters["force_orthogonal_as_lines"] else "0"

    if analysis_mode == "main_app":
        env["TEST_UI_MAIN_APP_ENABLE_OPTIMIZATION"] = "1" if fitting_parameters["main_app_enable_optimization"] else "0"

    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    output = f"{proc.stdout}\n{proc.stderr}".strip()
    node_results: dict[str, str] = {}
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    status_tokens = (" PASSED", " FAILED", " SKIPPED")
    node_status_re = re.compile(r"^(tests/\S+)\s+(PASSED|FAILED|SKIPPED)\b")
    for line in output.splitlines():
        clean = ansi_escape.sub("", line).strip()
        if not clean.startswith("tests/") or not any(token in clean for token in status_tokens):
            continue

        m = node_status_re.match(clean)
        if m:
            node_id = m.group(1)
            status = m.group(2).lower()
            node_results[node_id] = status

    summary_line = ""
    for line in reversed(output.splitlines()):
        if " passed" in line or " failed" in line or " skipped" in line:
            summary_line = line.strip()
            break

    return {
        "exit_code": proc.returncode,
        "command": " ".join(cmd),
        "summary_line": summary_line,
        "node_results": node_results,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _to_points(arr: np.ndarray) -> list[list[float]]:
    if arr.size == 0:
        return []
    return [[float(r), float(c)] for r, c in arr]


def _orient_open_curve_to_start(points: np.ndarray, ref_start: np.ndarray | None) -> np.ndarray:
    """Return points reversed when that better matches the reference start.

    This is used for UI layer display only, so the first rendered analytic point
    aligns with the extracted/fitted path direction users inspect in the canvas.
    """
    if ref_start is None or points.shape[0] < 2:
        return points

    d_first = float(np.linalg.norm(points[0] - ref_start))
    d_last = float(np.linalg.norm(points[-1] - ref_start))
    if d_last < d_first:
        return points[::-1].copy()
    return points


def _to_python_number(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _normalize_point(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    return [float(_to_python_number(value[0])), float(_to_python_number(value[1]))]


def _normalize_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for seg in segments:
        seg_type = seg.get("type")
        end = _normalize_point(seg.get("end_point"))
        if not isinstance(seg_type, str) or end is None:
            continue

        out: dict[str, Any] = {
            "type": seg_type,
            "end_point": end,
        }
        if seg_type == "cubic":
            c1 = _normalize_point(seg.get("control1"))
            c2 = _normalize_point(seg.get("control2"))
            if c1 is not None and c2 is not None:
                out["control1"] = c1
                out["control2"] = c2
        sp = _normalize_point(seg.get("start_point"))
        if sp is not None:
            out["start_point"] = sp
        normalized.append(out)
    return normalized


def _all_segment_vertices(raw_path: list[list[int]], segments: list[dict[str, Any]]) -> list[list[float]]:
    verts: list[list[float]] = []
    if not raw_path:
        return verts
    # Prefer the snapped start stored by fit_curve_segments over raw_path[0],
    # which may be a 1-pixel off-axis skeleton artifact.
    sp = segments[0].get("start_point") if segments else None
    if isinstance(sp, (list, tuple)) and len(sp) == 2:
        verts.append([float(sp[0]), float(sp[1])])
    else:
        verts.append([float(raw_path[0][0]), float(raw_path[0][1])])
    for seg in segments:
        end = seg.get("end_point")
        if isinstance(end, (list, tuple)) and len(end) == 2:
            verts.append([float(end[0]), float(end[1])])
    return verts


def _collect_cubic_handles(raw_path: list[list[int]], segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    handles: list[dict[str, Any]] = []
    if not raw_path:
        return handles

    sp = segments[0].get("start_point") if segments else None
    cursor = [float(sp[0]), float(sp[1])] if sp is not None else [float(raw_path[0][0]), float(raw_path[0][1])]
    for seg_index, seg in enumerate(segments):
        end = seg.get("end_point")
        if not isinstance(end, (list, tuple)) or len(end) != 2:
            continue
        end_pt = [float(end[0]), float(end[1])]
        if seg.get("type") == "cubic":
            c1 = seg.get("control1")
            c2 = seg.get("control2")
            if (
                isinstance(c1, (list, tuple))
                and len(c1) == 2
                and isinstance(c2, (list, tuple))
                and len(c2) == 2
            ):
                handles.append(
                    {
                        "segment_index": seg_index,
                        "start_point": cursor,
                        "control1": [float(c1[0]), float(c1[1])],
                        "control2": [float(c2[0]), float(c2[1])],
                        "end_point": end_pt,
                    }
                )
        cursor = end_pt
    return handles


def _fixture_analysis(fixture_id: str, fitting_parameters: dict[str, Any]) -> dict[str, Any]:
    if str(fitting_parameters.get("analysis_mode", "curve_fitter")) == "main_app":
        return _fixture_analysis_main_app(fixture_id, fitting_parameters)

    fx = load_fixture(fixture_id, size=FIXTURE_SIZE, stroke_width=FIXTURE_SW)
    raw_paths = extract_skeleton_paths(fx.gray, DARK_THRESHOLD, min_object_size=3)
    non_empty = [p for p in raw_paths if len(p) >= 2]
    fitted_paths = fit_curve_segments(
        non_empty,
        tolerance_px=fitting_parameters["cubic_fit_tolerance"],
        endpoint_tangent_strictness=fitting_parameters["endpoint_tangent_strictness"],
        force_orthogonal_as_lines=fitting_parameters["force_orthogonal_as_lines"],
        enable_curve_fitting=fitting_parameters["enable_curve_fitting"],
    )
    fitted_paths = [_normalize_segments(path_segments) for path_segments in fitted_paths]

    sampled_paths: list[list[list[float]]] = []
    vertices: list[list[list[float]]] = []
    cubic_handles: list[dict[str, Any]] = []

    for path_index, (raw_path, segments) in enumerate(zip(non_empty, fitted_paths)):
        sp = segments[0].get("start_point") if segments else None
        path_start = sp if sp is not None else list(raw_path[0])
        sampled = sample_fitted_segments(path_start, segments, n_per_segment=50)
        sampled_paths.append(_to_points(sampled))
        vertices.append(_all_segment_vertices(raw_path, segments))
        for handle in _collect_cubic_handles(raw_path, segments):
            handle["path_index"] = path_index
            cubic_handles.append(handle)

    if sampled_paths:
        flat_fitted = np.vstack(
            [np.asarray(path_points, dtype=np.float32) for path_points in sampled_paths if path_points]
        )
    else:
        flat_fitted = np.zeros((0, 2), dtype=np.float32)

    ideal_pts = fx.defn.ideal_sample(FIXTURE_SIZE, IDEAL_SAMPLE_N).astype(np.float32)
    ref_start = None
    if sampled_paths and sampled_paths[0]:
        ref_start = np.asarray(sampled_paths[0][0], dtype=np.float32)
    elif non_empty and non_empty[0]:
        ref_start = np.asarray(non_empty[0][0], dtype=np.float32)
    display_ideal_pts = _orient_open_curve_to_start(ideal_pts, ref_start)
    dev = analytical_deviation(flat_fitted, ideal_pts)

    pixel_points = np.argwhere(fx.pixels < 128)
    skeleton_layers = [
        [[float(p[0]), float(p[1])] for p in path]
        for path in non_empty
    ]

    segment_count = int(sum(len(segments) for segments in fitted_paths))
    all_types = {seg.get("type") for path in fitted_paths for seg in path}

    contract_flags: dict[str, Any] = {
        "has_paths": bool(non_empty),
        "has_fitted_geometry": bool(flat_fitted.size),
        "mean_within_fixture_tolerance": bool(dev["mean_deviation_px"] <= fx.tolerance_px),
        "line_only_segments": True,
    }
    if fixture_id in CURVE_FIXTURES:
        contract_flags["curve_has_cubic"] = "cubic" in all_types
    if fixture_id in LINE_FIXTURES:
        contract_flags["line_only_segments"] = all(t == "line" for t in all_types)

    return {
        "fixture_id": fixture_id,
        "description": fx.description,
        "category": fx.category,
        "tolerance_px": float(fx.tolerance_px),
        "dark_threshold": DARK_THRESHOLD,
        "fitting_parameters": dict(fitting_parameters),
        "raw_path_count": len(raw_paths),
        "non_empty_path_count": len(non_empty),
        "segment_count": segment_count,
        "segment_types": sorted(t for t in all_types if isinstance(t, str)),
        "deviation": {
            "mean_deviation_px": float(dev["mean_deviation_px"]),
            "max_deviation_px": float(dev["max_deviation_px"]),
            "coverage": float(dev["coverage"]),
        },
        "contract_flags": contract_flags,
        "layers": {
            "golden_analytic_path": _to_points(display_ideal_pts),
            "golden_pixel_points": _to_points(pixel_points.astype(np.float32)),
            "skeleton_paths": skeleton_layers,
            "optimized_sampled_paths": sampled_paths,
            "optimized_segments": fitted_paths,
            "optimized_vertices": vertices,
            "cubic_handles": cubic_handles,
        },
    }


def _polyline_segments(path: list[list[float]] | list[tuple[float, float]]) -> list[dict[str, Any]]:
    if len(path) < 2:
        return []

    normalized_path = [[float(point[0]), float(point[1])] for point in path]
    segments = []
    for index, point in enumerate(normalized_path[1:]):
        segment = {
            "type": "line",
            "end_point": [float(point[0]), float(point[1])],
        }
        if index == 0:
            segment["start_point"] = [float(normalized_path[0][0]), float(normalized_path[0][1])]
        segments.append(segment)
    return segments


def _fixture_analysis_main_app(fixture_id: str, fitting_parameters: dict[str, Any]) -> dict[str, Any]:
    fx = load_fixture(fixture_id, size=FIXTURE_SIZE, stroke_width=FIXTURE_SW)

    from centerline_web_app import CenterlineSession, process_centerlines

    session = CenterlineSession(f"fixture-{fixture_id}")
    session.image = fx.gray.astype(np.float32)
    session.display_image = session.image
    session.parameters.update({
        "dark_threshold": DARK_THRESHOLD,
        "min_path_length": 3,
        "enable_optimization": bool(fitting_parameters.get("main_app_enable_optimization", False)),
        "enable_curve_fitting": bool(fitting_parameters.get("enable_curve_fitting", False)),
        "cubic_fit_tolerance": float(fitting_parameters.get("cubic_fit_tolerance", 1.0)),
        "endpoint_tangent_strictness": float(fitting_parameters.get("endpoint_tangent_strictness", 85.0)),
        "force_orthogonal_as_lines": bool(fitting_parameters.get("force_orthogonal_as_lines", False)),
    })

    results = process_centerlines(session)
    if not isinstance(results, dict) or results.get("error"):
        raise ValueError(results.get("error") if isinstance(results, dict) else "main app analysis failed")

    optimized_paths = [list(path) for path in list(results.get("optimized_paths") or []) if len(path) >= 2]
    pre_paths = [list(path) for path in list(results.get("pre_optimization_paths") or []) if len(path) >= 2]
    optimization_enabled = bool(fitting_parameters.get("main_app_enable_optimization", False))
    pruning_debug = results.get("pruning_debug") if isinstance(results.get("pruning_debug"), dict) else None

    if optimization_enabled:
        rendered_segments = fit_curve_segments(
            optimized_paths,
            tolerance_px=fitting_parameters["cubic_fit_tolerance"],
            endpoint_tangent_strictness=fitting_parameters["endpoint_tangent_strictness"],
            force_orthogonal_as_lines=fitting_parameters["force_orthogonal_as_lines"],
            enable_curve_fitting=fitting_parameters["enable_curve_fitting"],
            path_hints=list(results.get("optimized_path_hints") or []),
        )
        rendered_segments = [_normalize_segments(path_segments) for path_segments in rendered_segments]
    else:
        rendered_segments = [_polyline_segments(path) for path in optimized_paths]

    sampled_paths: list[list[list[float]]] = []
    vertices: list[list[list[float]]] = []
    vertex_provenance: list[list[str]] = []
    cubic_handles: list[dict[str, Any]] = []
    for path_index, (raw_path, segments) in enumerate(zip(optimized_paths, rendered_segments)):
        if optimization_enabled:
            start_point = segments[0].get("start_point") if segments else None
            path_start = start_point if start_point is not None else list(raw_path[0])
            sampled = sample_fitted_segments(path_start, segments, n_per_segment=50)
            sampled_paths.append(_to_points(sampled))
            vertices.append(_all_segment_vertices(raw_path, segments))
            vertex_provenance.append(["preserved"] * len(vertices[-1]))
            for handle in _collect_cubic_handles(raw_path, segments):
                handle["path_index"] = path_index
                cubic_handles.append(handle)
        else:
            sampled_paths.append([[float(point[0]), float(point[1])] for point in raw_path])
            vertices.append([[float(point[0]), float(point[1])] for point in raw_path])
            provenance = ["preserved"] * len(raw_path)
            if pruning_debug and isinstance(pruning_debug.get("paths"), list) and path_index < len(pruning_debug["paths"]):
                path_debug = pruning_debug["paths"][path_index]
                raw_provenance = list(path_debug.get("output_point_provenance") or []) if isinstance(path_debug, dict) else []
                if len(raw_provenance) == len(raw_path):
                    provenance = [
                        "inferred" if str(item).lower() == "inferred" else "preserved"
                        for item in raw_provenance
                    ]
            vertex_provenance.append(provenance)

    if sampled_paths:
        flat_fitted = np.vstack(
            [np.asarray(path_points, dtype=np.float32) for path_points in sampled_paths if path_points]
        )
    else:
        flat_fitted = np.zeros((0, 2), dtype=np.float32)

    ideal_pts = fx.defn.ideal_sample(FIXTURE_SIZE, IDEAL_SAMPLE_N).astype(np.float32)
    ref_start = None
    if sampled_paths and sampled_paths[0]:
        ref_start = np.asarray(sampled_paths[0][0], dtype=np.float32)
    elif optimized_paths and optimized_paths[0]:
        ref_start = np.asarray(optimized_paths[0][0], dtype=np.float32)
    display_ideal_pts = _orient_open_curve_to_start(ideal_pts, ref_start)
    dev = analytical_deviation(flat_fitted, ideal_pts)

    pixel_points = np.argwhere(fx.pixels < 128)
    skeleton_layers = [
        [[float(p[0]), float(p[1])] for p in path]
        for path in (pre_paths or optimized_paths)
    ]

    segment_count = int(sum(len(segments) for segments in rendered_segments))
    all_types = {seg.get("type") for path in rendered_segments for seg in path}

    contract_flags: dict[str, Any] = {
        "has_paths": bool(optimized_paths),
        "has_fitted_geometry": bool(flat_fitted.size),
        "mean_within_fixture_tolerance": bool(dev["mean_deviation_px"] <= fx.tolerance_px),
        "line_only_segments": all(t == "line" for t in all_types) if all_types else True,
        "uses_main_app_pipeline": True,
        "optimization_enabled": bool(optimization_enabled),
    }

    return {
        "fixture_id": fixture_id,
        "description": fx.description,
        "category": fx.category,
        "tolerance_px": float(fx.tolerance_px),
        "dark_threshold": DARK_THRESHOLD,
        "fitting_parameters": dict(fitting_parameters),
        "raw_path_count": int(results.get("initial_paths_count", 0)),
        "non_empty_path_count": int(len(optimized_paths)),
        "segment_count": segment_count,
        "segment_types": sorted(t for t in all_types if isinstance(t, str)),
        "deviation": {
            "mean_deviation_px": float(dev["mean_deviation_px"]),
            "max_deviation_px": float(dev["max_deviation_px"]),
            "coverage": float(dev["coverage"]),
        },
        "contract_flags": contract_flags,
        "layers": {
            "golden_analytic_path": _to_points(display_ideal_pts),
            "golden_pixel_points": _to_points(pixel_points.astype(np.float32)),
            "skeleton_paths": skeleton_layers,
            "optimized_sampled_paths": sampled_paths,
            "optimized_segments": rendered_segments,
            "optimized_vertices": vertices,
            "optimized_vertex_provenance": vertex_provenance,
            "cubic_handles": cubic_handles,
        },
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _worker(
    run_state: TestRunState,
    update_goldens: bool,
    fixture_ids: list[str],
    fitting_parameters: dict[str, Any],
) -> None:
    run_state.status = "running"
    run_state.started_at = _utc_now_iso()
    run_state.fixture_ids = list(fixture_ids)
    run_state.fixture_count = len(fixture_ids)
    run_state.completed_fixtures = 0
    run_state.current_fixture = None
    run_state.fitting_parameters = dict(fitting_parameters)
    run_state.progress_messages.put(
        f"Running pytest suite for {run_state.fixture_count} fixture(s)..."
    )

    pytest_result = {
        "exit_code": 2,
        "command": "",
        "summary_line": "",
        "node_results": {},
        "stdout": "",
        "stderr": "",
    }
    fixture_details: list[dict[str, Any]] = []
    fixture_errors: dict[str, str] = {}
    run_status = "failed"
    error_message = ""
    fixture_dir = run_state.artifact_dir / "fixtures"

    try:
        pytest_result = _run_pytest(
            update_goldens=update_goldens,
            fixture_ids=fixture_ids,
            fitting_parameters=fitting_parameters,
        )
        run_state.pytest_exit_code = int(pytest_result["exit_code"])
        run_state.fixture_results = _fixture_pass_fail_from_results(
            fixture_ids, pytest_result["node_results"]
        )
        run_state.progress_messages.put(
            f"Pytest finished with exit code {pytest_result['exit_code']}"
        )

        total = len(fixture_ids)
        for index, fixture_id in enumerate(fixture_ids, start=1):
            run_state.current_fixture = fixture_id
            run_state.progress_messages.put(f"Analyzing fixture {index}/{total}: {fixture_id}")
            try:
                detail = _fixture_analysis(fixture_id, fitting_parameters)
                fixture_details.append(detail)
                _write_json(fixture_dir / f"{fixture_id}.json", detail)
                run_state.completed_fixtures = index
            except Exception as exc:
                fixture_errors[fixture_id] = str(exc)
                run_state.fixture_results[fixture_id] = "error"
                run_state.completed_fixtures = index
                run_state.progress_messages.put(f"Fixture failed: {fixture_id} ({exc})")

        run_state.current_fixture = None

        run_status = "completed" if pytest_result["exit_code"] == 0 and not fixture_errors else "failed"
    except Exception as exc:
        error_message = str(exc)
        run_state.current_fixture = None
        run_state.progress_messages.put(f"Run failed during artifact generation: {error_message}")

    summary = {
        "run_id": run_state.run_id,
        "created_at": run_state.created_at,
        "started_at": run_state.started_at,
        "finished_at": _utc_now_iso(),
        "status": run_status,
        "pytest": {
            "exit_code": pytest_result["exit_code"],
            "command": pytest_result["command"],
            "summary_line": pytest_result["summary_line"],
            "node_results": pytest_result["node_results"],
        },
        "fixture_ids": fixture_ids,
        "fixture_count": len(fixture_ids),
        "fitting_parameters": dict(fitting_parameters),
        "fixture_results": run_state.fixture_results,
        "fixture_errors": fixture_errors,
        "artifact_error": error_message,
    }

    _write_json(run_state.artifact_dir / "summary.json", summary)
    _write_json(
        run_state.artifact_dir / "pytest_output.json",
        {"stdout": pytest_result["stdout"], "stderr": pytest_result["stderr"]},
    )

    # Ensure all successful fixture details exist on disk even if written earlier.
    for detail in fixture_details:
        _write_json(fixture_dir / f"{detail['fixture_id']}.json", detail)

    run_state.finished_at = summary["finished_at"]
    run_state.status = summary["status"]
    run_state.summary = summary
    run_state.progress_messages.put("Run artifacts written.")


def create_run(
    update_goldens: bool = False,
    fixture_ids: list[str] | None = None,
    fitting_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    run_id = _make_run_id()
    artifact_dir = RUNS_ROOT / run_id
    selected_fixtures = _normalize_fixture_selection(fixture_ids)
    normalized_fitting_parameters = _normalize_fitting_parameters(fitting_parameters)
    if not selected_fixtures:
        raise ValueError("No valid fixture IDs selected for run")

    run_state = TestRunState(
        run_id=run_id,
        created_at=_utc_now_iso(),
        artifact_dir=artifact_dir,
        fixture_ids=selected_fixtures,
        fixture_count=len(selected_fixtures),
        fitting_parameters=dict(normalized_fitting_parameters),
    )

    worker = threading.Thread(
        target=_worker,
        args=(run_state, update_goldens, selected_fixtures, normalized_fitting_parameters),
        daemon=True,
    )
    run_state.thread = worker

    with _RUNS_LOCK:
        _RUNS[run_id] = run_state

    worker.start()
    return {
        "run_id": run_id,
        "status": run_state.status,
        "created_at": run_state.created_at,
        "update_goldens": bool(update_goldens),
        "fixture_ids": selected_fixtures,
        "fixture_count": len(selected_fixtures),
        "fitting_parameters": dict(normalized_fitting_parameters),
    }


def _in_memory_run_status(run_state: TestRunState) -> dict[str, Any]:
    return {
        "run_id": run_state.run_id,
        "status": run_state.status,
        "created_at": run_state.created_at,
        "started_at": run_state.started_at,
        "finished_at": run_state.finished_at,
        "fixture_count": run_state.fixture_count,
        "fixture_ids": run_state.fixture_ids,
        "completed_fixtures": run_state.completed_fixtures,
        "current_fixture": run_state.current_fixture,
        "fixture_results": run_state.fixture_results,
        "fitting_parameters": run_state.fitting_parameters,
        "artifact_dir": str(run_state.artifact_dir.relative_to(REPO_ROOT)),
    }


def list_runs() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    for summary_path in sorted(RUNS_ROOT.glob("*/summary.json"), reverse=True):
        try:
            summary = json.loads(summary_path.read_text())
            runs.append(
                {
                    "run_id": summary.get("run_id", summary_path.parent.name),
                    "status": summary.get("status", "unknown"),
                    "created_at": summary.get("created_at"),
                    "started_at": summary.get("started_at"),
                    "finished_at": summary.get("finished_at"),
                    "fixture_count": summary.get("fixture_count", 0),
                    "fixture_ids": summary.get("fixture_ids", []),
                    "fixture_results": summary.get("fixture_results", {}),
                    "fitting_parameters": summary.get("fitting_parameters", dict(DEFAULT_FITTING_PARAMETERS)),
                    "artifact_dir": str(summary_path.parent.relative_to(REPO_ROOT)),
                    "pytest_summary": summary.get("pytest", {}).get("summary_line", ""),
                }
            )
        except Exception:
            continue

    with _RUNS_LOCK:
        known = {r["run_id"] for r in runs}
        for run_id, state in _RUNS.items():
            if run_id in known:
                continue
            runs.append(_in_memory_run_status(state))

    runs.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    return runs


def get_run_summary(run_id: str) -> dict[str, Any] | None:
    summary_path = RUNS_ROOT / run_id / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())

    with _RUNS_LOCK:
        state = _RUNS.get(run_id)
        if state and state.summary:
            return state.summary
        if state:
            return _in_memory_run_status(state)

    return None


def get_fixture_detail(run_id: str, fixture_id: str) -> dict[str, Any] | None:
    detail_path = RUNS_ROOT / run_id / "fixtures" / f"{fixture_id}.json"
    if not detail_path.exists():
        return None
    return json.loads(detail_path.read_text())


def get_run_progress(run_id: str) -> dict[str, Any] | None:
    with _RUNS_LOCK:
        state = _RUNS.get(run_id)

    if not state:
        summary = get_run_summary(run_id)
        if summary is None:
            return None
        return {
            "run_id": run_id,
            "status": summary.get("status", "unknown"),
            "messages": [],
            "started_at": summary.get("started_at"),
            "finished_at": summary.get("finished_at"),
            "fixture_count": summary.get("fixture_count", 0),
            "fixture_ids": summary.get("fixture_ids", []),
            "completed_fixtures": summary.get("fixture_count", 0),
            "current_fixture": None,
            "fixture_results": summary.get("fixture_results", {}),
            "fitting_parameters": summary.get("fitting_parameters", dict(DEFAULT_FITTING_PARAMETERS)),
        }

    # If a summary was already persisted and thread is no longer alive, prefer
    # persisted status over stale in-memory status.
    if state.thread is not None and not state.thread.is_alive():
        summary = get_run_summary(run_id)
        if summary and summary.get("finished_at"):
            state.status = str(summary.get("status", state.status))
            state.finished_at = str(summary.get("finished_at", state.finished_at))
            state.summary = summary

    messages: list[str] = []
    while not state.progress_messages.empty():
        messages.append(str(state.progress_messages.get()))

    return {
        "run_id": run_id,
        "status": state.status,
        "messages": messages,
        "started_at": state.started_at,
        "finished_at": state.finished_at,
        "fixture_count": state.fixture_count,
        "fixture_ids": state.fixture_ids,
        "completed_fixtures": state.completed_fixtures,
        "current_fixture": state.current_fixture,
        "pytest_exit_code": state.pytest_exit_code,
        "fixture_results": state.fixture_results,
        "fitting_parameters": state.fitting_parameters,
    }
