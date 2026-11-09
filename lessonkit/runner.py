"""Drive the live app pipeline in-process through Flask's test client.

Primary path (what the browser UI does):
    POST /upload (multipart) -> POST /process_immediate (JSON, UI default params)
Float paths are read from ``sessions[sid].initial_paths``; the JSON response
truncates coordinates with int(), which is reported separately.

Secondary path: POST /process with enable_optimization=true (merge + overlap
removal + prune + circle optimisation). It can be slow, so it runs in a
subprocess worker that is killed after a timeout.

No app code is modified; this only calls public routes and reads the
in-memory session registry.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Mirrors getParameters() defaults in templates/index.html.
UI_DEFAULT_PARAMS = {
    "dark_threshold": 0.20,
    "merge_gap": 25,
    "merge_angle_priority": 30,
    "enable_long_path_merging": False,
    "rdp_tolerance": 5.0,
    "simplification_strength": 100,
    "line_fit_strength": 0,
    "enable_post_fit_export": False,
    "source_smoothing": 70,
    "short_path_protection": 0,
    "score_preservation": 99,
    "enable_curve_fitting": False,
    "cubic_fit_tolerance": 0.35,
    "endpoint_tangent_strictness": 85.0,
    "force_orthogonal_as_lines": False,
    "show_pruning_debug_grid": False,
    "min_path_length": 2,
    "enable_pruning": False,
    "enable_optimization": False,
    "show_pre_optimization": True,
    "include_image": False,
    "normalization_mode": "auto",
    "normalization_sensitivity": "medium",
}

_UPLOAD_ONLY = ("normalization_mode", "normalization_sensitivity")


@dataclass
class PipelineResult:
    route: str
    paths: list[np.ndarray] = field(default_factory=list)          # float (row, col)
    response_paths: list[np.ndarray] = field(default_factory=list)  # as returned in JSON
    counts: dict = field(default_factory=dict)
    image_shape: tuple[int, int] | None = None
    elapsed_ms: float = 0.0
    error: str | None = None
    timed_out: bool = False


@contextlib.contextmanager
def _quiet():
    if os.environ.get("PIPELINE_VERBOSE"):
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()):
        yield


_APP_MODULE = None


def _app_module():
    global _APP_MODULE
    if _APP_MODULE is None:
        with _quiet():
            import centerline_web_app as mod  # noqa: WPS433 (deliberate lazy import)
        _APP_MODULE = mod
    return _APP_MODULE


def _merged_params(overrides: dict | None) -> dict:
    params = dict(UI_DEFAULT_PARAMS)
    params.update(overrides or {})
    return params


def _read_bytes(image) -> tuple[bytes, str]:
    if isinstance(image, (bytes, bytearray)):
        return bytes(image), "lesson.png"
    p = Path(image)
    return p.read_bytes(), p.name


def _upload(client, data: bytes, name: str, params: dict):
    resp = client.post(
        "/upload",
        data={
            "file": (io.BytesIO(data), name),
            "normalization_mode": params["normalization_mode"],
            "normalization_sensitivity": params["normalization_sensitivity"],
        },
        content_type="multipart/form-data",
    )
    body = resp.get_json(silent=True) or {}
    if "session_id" not in body:
        raise RuntimeError(f"/upload failed: {body.get('error') or resp.status_code}")
    return body["session_id"], tuple(body.get("image_shape") or ())


def _release(mod, sid: str) -> None:
    session = dict.pop(mod.sessions, sid, None)
    release = getattr(mod, "_release_session", None)
    if session is not None and callable(release):
        try:
            release(session)
        except Exception:
            pass


def _as_float_paths(paths) -> list[np.ndarray]:
    out = []
    for p in paths or []:
        arr = np.asarray(p, dtype=float)
        if arr.ndim == 2 and arr.shape[1] >= 2 and len(arr):
            out.append(arr[:, :2].copy())
    return out


def run_live(image, params: dict | None = None) -> PipelineResult:
    """Upload + /process_immediate exactly as the UI does (enable_optimization stays false)."""
    mod = _app_module()
    p = _merged_params(params)
    p["enable_optimization"] = False
    data, name = _read_bytes(image)
    result = PipelineResult(route="process_immediate")
    client = mod.app.test_client()
    sid = None
    t0 = time.perf_counter()
    try:
        with _quiet():
            sid, shape = _upload(client, data, name, p)
            result.image_shape = shape[:2] if shape else None
            body_params = {k: v for k, v in p.items() if k not in _UPLOAD_ONLY}
            resp = client.post("/process_immediate", json={"session_id": sid, "parameters": body_params})
        body = resp.get_json(silent=True) or {}
        if not body.get("success"):
            result.error = body.get("error") or f"HTTP {resp.status_code}"
            return result
        session = dict.get(mod.sessions, sid)
        result.paths = _as_float_paths(getattr(session, "initial_paths", None))
        result.response_paths = _as_float_paths(body.get("paths"))
        result.counts = {k: body.get(k) for k in ("initial_paths_count", "merged_paths_count", "valid_paths_count")}
        result.counts["median_stroke_width_px"] = body.get("median_stroke_width_px")
        result.counts["extraction_profile"] = body.get("extraction_profile")
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        result.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if sid:
            _release(mod, sid)
    return result


def run_live_both(image, params: dict | None = None) -> dict[str, PipelineResult]:
    """Live path with pruning off and on."""
    base = dict(params or {})
    return {
        "prune_off": run_live(image, {**base, "enable_pruning": False}),
        "prune_on": run_live(image, {**base, "enable_pruning": True}),
    }


def _process_in_this_process(image_path: str, params: dict) -> dict:
    mod = _app_module()
    p = _merged_params(params)
    p["enable_optimization"] = True
    data, name = _read_bytes(image_path)
    client = mod.app.test_client()
    sid = None
    try:
        with _quiet():
            sid, shape = _upload(client, data, name, p)
            body_params = {k: v for k, v in p.items() if k not in _UPLOAD_ONLY}
            resp = client.post("/process", json={"session_id": sid, "parameters": body_params})
        body = resp.get_json(silent=True) or {}
        if not body.get("success"):
            return {"error": body.get("error") or f"HTTP {resp.status_code}"}
        results = dict.get(mod.sessions, sid).results or {}
        return {
            "image_shape": list(shape[:2]) if shape else None,
            "optimized_paths": [np.asarray(x, float)[:, :2].tolist() for x in results.get("optimized_paths", [])],
            "pre_optimization_paths": [np.asarray(x, float)[:, :2].tolist()
                                       for x in results.get("pre_optimization_paths", [])],
            "counts": {k: body.get(k) for k in ("initial_paths_count", "merged_paths_count",
                                                 "valid_paths_count", "paths_count")},
        }
    finally:
        if sid:
            _release(mod, sid)


def run_process(image, params: dict | None = None, timeout_s: float = 60.0) -> dict[str, PipelineResult]:
    """Secondary /process route (enable_optimization=true) in a killable subprocess.

    Returns {"process_pre_opt": ..., "process_optimized": ...}.
    """
    image_path = str(Path(image).resolve())
    cmd = [sys.executable, "-m", "lessonkit.runner", "--process-worker", image_path,
           json.dumps(params or {})]
    t0 = time.perf_counter()
    pre = PipelineResult(route="process_pre_opt")
    opt = PipelineResult(route="process_optimized")
    try:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout_s)
        payload = None
        for line in reversed(proc.stdout.splitlines()):
            if line.startswith("@@RESULT@@"):
                payload = json.loads(line[len("@@RESULT@@"):])
                break
        if payload is None:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
            raise RuntimeError("worker produced no result: " + " | ".join(tail))
        if payload.get("error"):
            pre.error = opt.error = payload["error"]
        else:
            shape = tuple(payload.get("image_shape") or ()) or None
            for res, key in ((pre, "pre_optimization_paths"), (opt, "optimized_paths")):
                res.paths = _as_float_paths(payload[key])
                res.response_paths = res.paths
                res.counts = dict(payload.get("counts") or {})
                res.image_shape = shape
    except subprocess.TimeoutExpired:
        pre.timed_out = opt.timed_out = True
        pre.error = opt.error = f"timeout after {timeout_s:g}s"
    except Exception as exc:
        pre.error = opt.error = f"{type(exc).__name__}: {exc}"
    elapsed = (time.perf_counter() - t0) * 1000.0
    pre.elapsed_ms = opt.elapsed_ms = elapsed
    return {"process_pre_opt": pre, "process_optimized": opt}


def _worker_main(argv: list[str]) -> int:
    image_path, params_json = argv[0], argv[1]
    try:
        out = _process_in_this_process(image_path, json.loads(params_json))
    except Exception as exc:
        out = {"error": f"{type(exc).__name__}: {exc}"}
    sys.stdout.write("@@RESULT@@" + json.dumps(out) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--process-worker":
        raise SystemExit(_worker_main(sys.argv[2:]))
    print("usage: python -m lessonkit.runner --process-worker IMAGE PARAMS_JSON", file=sys.stderr)
    raise SystemExit(2)
