"""Teach scoring: a subprocess worker plus the in-process queue that runs it (design 8).

Worker: ``python -m teach.scoring_worker <job.json>`` (cwd = repo root, TEACH_ENABLED=0).
It drives the live pipeline through ``lessonkit.runner.run_live_both`` and scores with
``lessonkit.scoring.score_paths``, the same steps as ``tests/pipeline/evaluate.py``,
then writes ``result.json`` and ``overlay.png``/``thumb.png`` next to the job file.
With ``enable_post_fit_export`` on it also scores a ``post_fit`` variant from
``lessonkit.runner.run_post_fit`` (the Download SVG output) and writes ``overlay_post_fit.png``.

Parent: ``ScoringQueue`` runs one job at a time with at most 3 waiting, at most one
active-or-waiting job per client IP, and kills a worker after 30 s.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TIMEOUT_S = 30.0
MAX_WAITING = 3
RLIMIT_AS_BYTES = int(1.5 * 1024 ** 3)
NICE = 10
PRIMARY_VARIANT = "prune_off"
POST_FIT_VARIANT = "post_fit"
THUMB_MAX_PX = 256
# Never hand secrets to the worker; it needs none of them.
_SECRET_ENV = ("TEACH_ADMIN_TOKEN", "TEACH_ALERT_WEBHOOK_URL", "R2_ENDPOINT", "R2_BUCKET", "R2_ACCESS_KEY_ID",
               "R2_SECRET_ACCESS_KEY")


class Busy(Exception):
    def __init__(self, reason: str = "queue_full"):
        super().__init__(reason)
        self.reason = reason


# ======================================================================= worker side

def _limit_resources() -> None:
    if not sys.platform.startswith("linux"):
        return
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (RLIMIT_AS_BYTES, RLIMIT_AS_BYTES))
    except Exception:
        pass
    try:
        os.nice(NICE)
    except Exception:
        pass


def engine_commit() -> str:
    sha = (os.environ.get("RAILWAY_GIT_COMMIT_SHA") or "").strip()
    if sha:
        return sha[:7]
    try:
        from lessonkit.lesson import engine_commit as git_commit

        return git_commit() or "unknown"
    except Exception:
        return "unknown"


def _record(name: str, res, score, graph: dict) -> dict:
    """Same fields as tests/pipeline/evaluate.py:evaluation_record for one variant."""
    rec = {
        "error": res.error,
        "timed_out": res.timed_out,
        "elapsed_ms": round(res.elapsed_ms, 1),
        "counts": res.counts,
        "graph": {k: graph.get(k) for k in ("path_count", "paths_in_region", "endpoints_in_region",
                                            "junctions_in_region", "short_paths_in_region", "short_path_lengths")},
    }
    if score is not None:
        rec["summary"] = score.summary()
        rec["lines"] = [{
            "index": ln.index, "best_path": ln.best_path,
            "best_single_coverage": round(ln.best_single_coverage, 4),
            "any_coverage": round(ln.any_coverage, 4), "fragments": ln.fragments,
            "ref_to_best": ln.ref_to_best, "best_to_ref": ln.best_to_ref, "ref_to_any": ln.ref_to_any,
            "passed": ln.passed, "reason": ln.reason,
        } for ln in score.lines]
    return rec


def score_job(job: dict) -> tuple[dict, object, object, object]:
    """Run the pipeline and score.

    Returns (result dict, overlay image, thumbnail image, post-fit overlay image or None).
    Variants prune_off and prune_on are always scored. ``post_fit`` is added only when
    ``params["enable_post_fit_export"]`` is true: the SVG the Download button would
    produce, read back with the safe reader and scored with the same rules.
    """
    import numpy as np
    from PIL import Image

    from lessonkit.overlay import render_overlay
    from lessonkit.runner import run_live_both, run_post_fit
    from lessonkit.scoring import graph_metrics, score_paths

    image_path = Path(job["image"])
    ref_doc = json.loads(Path(job["reference"]).read_text(encoding="utf-8"))
    reference = [np.asarray(p, dtype=float) for p in ref_doc["paths"]]
    scoring = job["scoring"]
    t0 = time.perf_counter()
    runs = run_live_both(image_path, job["params"])
    if (job.get("params") or {}).get("enable_post_fit_export"):
        runs[POST_FIT_VARIANT] = run_post_fit(image_path, job["params"])
    variants, primary, post_fit = {}, None, None
    for name, res in runs.items():
        if res.error:
            variants[name] = _record(name, res, None, {})
            continue
        sc = score_paths(res.paths, reference, scoring, res.image_shape)
        graph = sc.graph
        variants[name] = _record(name, res, sc, graph)
        if name == PRIMARY_VARIANT:
            primary = (res, sc, graph)
        elif name == POST_FIT_VARIANT:
            post_fit = (res, sc, graph)
    out = {"variants": variants, "engine_commit": engine_commit(),
           "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1), "variant": PRIMARY_VARIANT}
    gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
    region = scoring.get("region")
    if primary is None:
        out["outcome"] = "error"
        out["error"] = variants.get(PRIMARY_VARIANT, {}).get("error") or "pipeline error"
        overlay = render_overlay(gray, [], reference, None, region)
    else:
        res, sc, graph = primary
        out["outcome"] = "pass" if sc.passed else "fail"
        out["summary"] = sc.summary()
        overlay = render_overlay(gray, res.paths, reference, graph, region)
    if region:
        thumb_src = render_overlay(gray, primary[0].paths if primary else [], reference,
                                   primary[2] if primary else None, region, zoom_to_region=True)
    else:
        thumb_src = overlay.copy()
    thumb_src.thumbnail((THUMB_MAX_PX, THUMB_MAX_PX))
    post_overlay = None
    if post_fit is not None:
        res, sc, graph = post_fit
        post_overlay = render_overlay(gray, res.paths, reference, graph, region)
    return out, overlay, thumb_src, post_overlay


def worker_main(argv: list[str]) -> int:
    _limit_resources()
    if len(argv) != 1:
        print("usage: python -m teach.scoring_worker <job.json>", file=sys.stderr)
        return 2
    job_path = Path(argv[0])
    out_dir = job_path.parent
    try:
        job = json.loads(job_path.read_text(encoding="utf-8"))
        result, overlay, thumb, post_overlay = score_job(job)
        overlay.save(out_dir / "overlay.png")
        thumb.save(out_dir / "thumb.png")
        if post_overlay is not None:
            post_overlay.save(out_dir / "overlay_post_fit.png")
    except Exception as exc:
        result = {"outcome": "error", "error": f"{type(exc).__name__}", "variants": {}}
    tmp = out_dir / "result.json.tmp"
    tmp.write_text(json.dumps(result), encoding="utf-8")
    os.replace(tmp, out_dir / "result.json")
    return 0


# ======================================================================= parent side

def worker_env() -> dict:
    env = {k: v for k, v in os.environ.items() if k not in _SECRET_ENV}
    env["TEACH_ENABLED"] = "0"
    env.pop("PIPELINE_VERBOSE", None)
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


class ScoringQueue:
    """One job at a time, at most ``max_waiting`` waiting, at most one per client IP."""

    def __init__(self, timeout_s: float = TIMEOUT_S, max_waiting: int = MAX_WAITING, command=None):
        self.timeout_s = timeout_s
        self.max_waiting = max_waiting
        self.command = command or [sys.executable, "-m", "teach.scoring_worker"]
        self._cond = threading.Condition()
        self._running = False
        self._waiting = 0
        self._ips: set[str] = set()

    @property
    def depth(self) -> int:
        with self._cond:
            return self._waiting + (1 if self._running else 0)

    def _enter(self, ip: str) -> None:
        with self._cond:
            if ip in self._ips:
                raise Busy("ip_busy")
            if self._running and self._waiting >= self.max_waiting:
                raise Busy("queue_full")
            self._ips.add(ip)
            self._waiting += 1
            try:
                while self._running:
                    self._cond.wait()
            finally:
                self._waiting -= 1
            self._running = True

    def _leave(self, ip: str) -> None:
        with self._cond:
            self._running = False
            self._ips.discard(ip)
            self._cond.notify()

    def run(self, ip: str, image_png: bytes, reference_json: dict, params: dict, scoring: dict) -> dict:
        """Score one case. Returns the worker result plus ``overlay_png``/``thumb_png`` bytes (or None)."""
        self._enter(ip)
        try:
            return self._run_worker(image_png, reference_json, params, scoring)
        finally:
            self._leave(ip)

    def _run_worker(self, image_png, reference_json, params, scoring) -> dict:
        job_dir = Path(tempfile.mkdtemp(prefix="teach-job-"))
        try:
            (job_dir / "image.png").write_bytes(image_png)
            (job_dir / "reference.json").write_text(json.dumps(reference_json), encoding="utf-8")
            job = {"image": str(job_dir / "image.png"), "reference": str(job_dir / "reference.json"),
                   "params": params, "scoring": scoring}
            job_path = job_dir / "job.json"
            job_path.write_text(json.dumps(job), encoding="utf-8")
            t0 = time.perf_counter()
            try:
                subprocess.run(self.command + [str(job_path)], cwd=str(REPO_ROOT), env=worker_env(),
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               timeout=self.timeout_s, check=False)
            except subprocess.TimeoutExpired:
                return {"outcome": "timeout", "timed_out": True, "variants": {}, "engine_commit": engine_commit(),
                        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                        "overlay_png": None, "thumb_png": None, "post_fit_overlay_png": None}
            result_path = job_dir / "result.json"
            if not result_path.exists():
                result = {"outcome": "error", "error": "worker produced no result", "variants": {}}
            else:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            result.setdefault("timed_out", False)
            result.setdefault("engine_commit", engine_commit())
            for key, name in (("overlay_png", "overlay.png"), ("thumb_png", "thumb.png"),
                              ("post_fit_overlay_png", "overlay_post_fit.png")):
                p = job_dir / name
                result[key] = p.read_bytes() if p.exists() else None
            return result
        finally:
            shutil.rmtree(job_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(worker_main(sys.argv[1:]))
