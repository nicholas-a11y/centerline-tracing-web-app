"""Stage 3: compare the old engine (centerline_engine) with centerline_engine_light.

Report only. Nothing here fixes behaviour, edits fixtures or regenerates goldens.

Sections
  A  pytest outcomes: each legacy test file vs. its tests/light_port copy
  B  /test-ui rig: test_ui_backend._fixture_analysis vs. rig_backend_light, all fixtures, both modes
  C  the four functions whose code differs (extract_skeleton_paths, create_fast_paths,
     fit_curve_segments, create_svg_output) on fixtures and lesson images
  D  process_centerlines (/process: merge + overlap + prune [+ optimise]) with the app's
     old-engine bindings vs. the same call with light bindings, on lesson images

Usage:
    python -m tests.light_port.compare_engines [--out DIR] [--skip-pytest] [--with-optimisation]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PYTEST_PAIRS = [
    ("tests/test_path_optimization.py", "tests/light_port/test_path_optimization.py"),
    ("tests/test_small_image_supersampling.py", "tests/light_port/test_small_image_supersampling.py"),
    ("tests/test_web_app_optimization_state.py", "tests/light_port/test_web_app_optimization_state.py"),
    ("tests/test_resolution_scaling.py", "tests/light_port/test_resolution_scaling.py"),
]


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def _arr(p) -> np.ndarray:
    return np.asarray(p, dtype=float).reshape(-1, 2)


def path_set_diff(old, new) -> dict:
    a = [_arr(p) for p in (old or []) if len(p)]
    b = [_arr(p) for p in (new or []) if len(p)]
    identical = len(a) == len(b) and all(x.shape == y.shape and np.array_equal(x, y) for x, y in zip(a, b))
    out = {"identical": identical, "count_old": len(a), "count_light": len(b), "max_point_dist": 0.0}
    if identical:
        return out
    if not a or not b:
        out["max_point_dist"] = float("inf")
        return out
    pa, pb = np.vstack(a), np.vstack(b)
    d1 = cKDTree(pb).query(pa)[0].max()
    d2 = cKDTree(pa).query(pb)[0].max()
    out["max_point_dist"] = round(float(max(d1, d2)), 4)
    return out


def _round(obj, nd=6):
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, (list, tuple)):
        return [_round(x, nd) for x in obj]
    if isinstance(obj, dict):
        return {k: _round(v, nd) for k, v in obj.items()}
    if isinstance(obj, np.generic):
        return _round(obj.item(), nd)
    if isinstance(obj, np.ndarray):
        return _round(obj.tolist(), nd)
    return obj


# --------------------------------------------------------------------------- A: pytest
def _junit(test_file: str, tmp: Path) -> dict[str, dict]:
    xml = tmp / (Path(test_file).stem + ("_light" if "light_port" in test_file else "_old") + ".xml")
    # --runxfail: compare the raw outcome/message, not the baseline xfail marker added by conftest.
    extra = ["--runxfail"] if "light_port" in test_file else []
    subprocess.run(
        [sys.executable, "-m", "pytest", test_file, "-q", "-p", "no:cacheprovider",
         "-W", "ignore::FutureWarning", f"--junitxml={xml}", *extra],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    results = {}
    for case in ET.parse(xml).getroot().iter("testcase"):
        outcome, message = "passed", ""
        for tag in ("failure", "error", "skipped"):
            el = case.find(tag)
            if el is not None:
                outcome = {"failure": "failed", "error": "error", "skipped": "skipped"}[tag]
                message = (el.get("message") or "").splitlines()[0][:200] if el.get("message") else ""
                break
        results[case.get("name")] = {"outcome": outcome, "message": message}
    return results


def section_pytest(tmp: Path) -> list[dict]:
    rows = []
    for old_file, new_file in PYTEST_PAIRS:
        old, new = _junit(old_file, tmp), _junit(new_file, tmp)
        diffs = []
        for name in sorted(set(old) | set(new)):
            o, n = old.get(name), new.get(name)
            if o != n:
                diffs.append({"test": name, "old": o, "light": n})
        rows.append({
            "original": old_file,
            "port": new_file,
            "old_counts": _count(old),
            "light_counts": _count(new),
            "differences": diffs,
        })
    return rows


def _count(results: dict) -> dict:
    out: dict[str, int] = {}
    for r in results.values():
        out[r["outcome"]] = out.get(r["outcome"], 0) + 1
    return out


# --------------------------------------------------------------------------- B: rig
RIG_MODES = {
    "curve_fitter/fit_off": {"analysis_mode": "curve_fitter", "enable_curve_fitting": False},
    "curve_fitter/fit_on": {"analysis_mode": "curve_fitter", "enable_curve_fitting": True},
    "main_app/opt_off": {"analysis_mode": "main_app", "main_app_enable_optimization": False},
    "main_app/opt_on": {"analysis_mode": "main_app", "main_app_enable_optimization": True},
}


def section_rig(with_optimisation: bool) -> list[dict]:
    with quiet():
        import test_ui_backend as old_rig
        from tests.light_port import rig_backend_light as new_rig
    rows = []
    for mode, overrides in RIG_MODES.items():
        if mode == "main_app/opt_on" and not with_optimisation:
            continue
        params = old_rig._normalize_fitting_parameters(dict(old_rig.DEFAULT_FITTING_PARAMETERS, **overrides))
        for fid in old_rig.ALL_FIXTURE_IDS:
            row = {"mode": mode, "fixture": fid}
            try:
                with quiet():
                    o = old_rig._fixture_analysis(fid, dict(params))
                    n = new_rig._fixture_analysis(fid, dict(params))
            except Exception as exc:  # report, never hide
                row["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)
                continue
            row.update(_rig_compare(o, n))
            rows.append(row)
    return rows


def _rig_compare(o: dict, n: dict) -> dict:
    out = {}
    for key in ("raw_path_count", "non_empty_path_count", "segment_count", "segment_types",
                "path_count", "contract_flags", "passed"):
        if key in o or key in n:
            if _round(o.get(key)) != _round(n.get(key)):
                out[key] = {"old": _round(o.get(key)), "light": _round(n.get(key))}
    od, nd = o.get("deviation") or {}, n.get("deviation") or {}
    for k in sorted(set(od) | set(nd)):
        ov, nv = od.get(k), nd.get(k)
        if isinstance(ov, (int, float)) and isinstance(nv, (int, float)):
            if abs(float(ov) - float(nv)) > 1e-6:
                out[f"deviation.{k}"] = {"old": round(float(ov), 4), "light": round(float(nv), 4)}
        elif ov != nv:
            out[f"deviation.{k}"] = {"old": _round(ov), "light": _round(nv)}
    ol, nl = o.get("layers") or {}, n.get("layers") or {}
    for layer in ("skeleton_paths", "optimized_sampled_paths", "pre_optimization_paths"):
        if layer in ol or layer in nl:
            d = path_set_diff(ol.get(layer), nl.get(layer))
            if not d["identical"]:
                out[f"layers.{layer}"] = d
    return {"identical": not out, "differences": out}


# --------------------------------------------------------------------------- C: differing functions
def _images() -> list[tuple[str, np.ndarray]]:
    from PIL import Image

    from tests.conftest import ALL_FIXTURE_IDS
    from tests.fixtures import load_fixture

    imgs = []
    for fid in ALL_FIXTURE_IDS:
        fx = load_fixture(fid, size=64, stroke_width=3, auto_generate=False)
        imgs.append((f"fixture:{fid}", np.asarray(fx.gray, dtype=np.float32)))
    for png in sorted((REPO_ROOT / "tests" / "lessons").glob("*/image.png")):
        with Image.open(png) as im:
            imgs.append((f"lesson:{png.parent.name}", np.asarray(im.convert("L"), dtype=np.float32) / 255.0))
    return imgs


EXTRACT_SETTINGS = {
    # (dark_threshold, min_object_size, min_path_length, supersample)
    "rig(0.5,obj3)": (0.5, 3, None, True),
    "ui(0.2,obj10)": (0.2, 10, None, True),
    "ui(0.2,obj10)/no-supersample": (0.2, 10, None, False),
}


def section_functions() -> dict:
    with quiet():
        import centerline_engine as old
        import centerline_engine_light as light
    from skimage import morphology

    images = _images()
    out = {"extract_skeleton_paths": [], "create_fast_paths": [], "fit_curve_segments": [], "create_svg_output": []}

    for name, gray in images:
        for setting, (thr, obj, mpl, ss) in EXTRACT_SETTINGS.items():
            kw = {"min_object_size": obj, "enable_small_image_supersample": ss}
            if mpl is not None:
                kw["min_path_length"] = mpl
            with quiet():
                po = old.extract_skeleton_paths(gray, thr, **kw)
                pn = light.extract_skeleton_paths(gray, thr, **kw)
            d = path_set_diff(po, pn)
            out["extract_skeleton_paths"].append({"image": name, "setting": setting, **d})

        skel = morphology.skeletonize(gray < 0.5)
        with quiet():
            fo = old.create_fast_paths(skel, min_path_length=3)
            fn = light.create_fast_paths(skel, min_path_length=3)
        out["create_fast_paths"].append({"image": name, **path_set_diff(fo, fn)})

        with quiet():
            paths = [p for p in light.extract_skeleton_paths(gray, 0.5, min_object_size=3) if len(p) >= 2]
        for fit in (False, True):
            kw = dict(tolerance_px=0.35, endpoint_tangent_strictness=85.0,
                      force_orthogonal_as_lines=False, enable_curve_fitting=fit)
            with quiet():
                so = _round(old.fit_curve_segments(paths, **kw))
                sn = _round(light.fit_curve_segments(paths, **kw))
            out["fit_curve_segments"].append({
                "image": name, "curve_fitting": fit, "identical": so == sn,
                "paths_differing": sum(1 for a, b in zip(so, sn) if a != b),
                "segments_old": sum(len(p) for p in so), "segments_light": sum(len(p) for p in sn),
            })

        for fit in (False, True):
            svgs = []
            for mod in (old, light):
                saved = (mod.SHOW_BITMAP, mod.SHOW_PRE_OPTIMIZATION_PATHS)
                mod.SHOW_BITMAP, mod.SHOW_PRE_OPTIMIZATION_PATHS = False, False
                try:
                    with quiet():
                        svgs.append(mod.create_svg_output(
                            gray, None, paths, [1.0] * len(paths), None,
                            enable_curve_fitting=fit, output_path=None,
                        ))
                finally:
                    mod.SHOW_BITMAP, mod.SHOW_PRE_OPTIMIZATION_PATHS = saved
            out["create_svg_output"].append({"image": name, "curve_fitting": fit, "identical": svgs[0] == svgs[1]})
    return out


# --------------------------------------------------------------------------- D: /process
def section_process(with_optimisation: bool) -> list[dict]:
    from PIL import Image

    with quiet():
        import centerline_web_app as app_mod
    from tests.light_port.light_app import light_engine_imports

    def run(gray, enable_opt):
        session = app_mod.CenterlineSession(f"stage3-{time.time_ns()}")
        session.image = gray
        session.display_image = gray
        session.parameters.update({"dark_threshold": 0.2, "min_path_length": 2, "enable_optimization": enable_opt})
        t0 = time.perf_counter()
        with quiet():
            res = app_mod.process_centerlines(session)
        return res, time.perf_counter() - t0

    rows = []
    for png in sorted((REPO_ROOT / "tests" / "lessons").glob("*/image.png")):
        with Image.open(png) as im:
            gray = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
        for enable_opt in ((False, True) if with_optimisation else (False,)):
            ro, to = run(gray, enable_opt)
            ro2, _ = run(gray, enable_opt)
            with light_engine_imports():
                rn, tn = run(gray, enable_opt)
            row = {"lesson": png.parent.name, "optimisation": enable_opt,
                   "seconds_old": round(to, 2), "seconds_light": round(tn, 2)}
            for key in ("pre_optimization_paths", "optimized_paths"):
                row[key] = path_set_diff(ro.get(key), rn.get(key))
                row[key + "_old_repeat_identical"] = path_set_diff(ro.get(key), ro2.get(key))["identical"]
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- E: UI helpers in centerline_core
_VOLATILE_KEYS = {"elapsed", "elapsed_sec", "elapsed_seconds", "duration", "duration_sec", "seconds", "time",
                  "timestamp", "time_budget_sec", "time_used_sec", "timed_out", "progress"}


def _strip_volatile(obj):
    if isinstance(obj, dict):
        return {k: _strip_volatile(v) for k, v in obj.items()
                if k not in _VOLATILE_KEYS and "elapsed" not in k and "duration" not in k}
    if isinstance(obj, (list, tuple)):
        return [_strip_volatile(v) for v in obj]
    return obj


def section_core_helpers() -> list[dict]:
    """auto-detect / auto-tune / tile preview are UI buttons that call the old-engine merge via centerline_core."""
    from PIL import Image

    with quiet():
        import centerline_core as core
    from tests.light_port.light_app import light_engine_imports

    calls = {
        "auto_detect_min_path_length": lambda g: core.auto_detect_min_path_length(g, 0.2),
        "auto_detect_dark_threshold": lambda g: core.auto_detect_dark_threshold(g, num_thresholds=6),
        "auto_tune_extraction_parameters": lambda g: core.auto_tune_extraction_parameters(
            g, num_thresholds=6, time_budget_sec=600.0),
        "_extract_single_tile_preview_entry": lambda g: core._extract_single_tile_preview_entry(
            g, tile_index=0, origin_x=0, origin_y=0, tile_width=g.shape[1], tile_height=g.shape[0],
            dark_threshold=0.2, min_path_length=2),
    }
    rows = []
    for png in sorted((REPO_ROOT / "tests" / "lessons").glob("*/image.png")):
        with Image.open(png) as im:
            gray = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
        for name, fn in calls.items():
            row = {"lesson": png.parent.name, "function": name}
            try:
                with quiet():
                    ro = _round(_strip_volatile(fn(gray)))
                    ro2 = _round(_strip_volatile(fn(gray)))
                    with light_engine_imports():
                        rn = _round(_strip_volatile(fn(gray)))
                row.update(identical=json.dumps(ro, sort_keys=True, default=str) == json.dumps(rn, sort_keys=True, default=str),
                           old_repeat_identical=json.dumps(ro, sort_keys=True, default=str) == json.dumps(ro2, sort_keys=True, default=str))
            except Exception as exc:  # noqa: BLE001 - recorded as a finding
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(Path(tempfile.gettempdir()) / "centerline-stage3"))
    ap.add_argument("--skip-pytest", action="store_true")
    ap.add_argument("--with-optimisation", action="store_true", help="also compare circle optimisation (slow)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    report: dict = {"python": sys.version.split()[0]}
    t0 = time.perf_counter()
    if not args.skip_pytest:
        report["pytest"] = section_pytest(out)
        print(f"[A] pytest pairs done ({time.perf_counter() - t0:.0f}s)")
    report["rig"] = section_rig(args.with_optimisation)
    print(f"[B] rig done ({time.perf_counter() - t0:.0f}s)")
    report["functions"] = section_functions()
    print(f"[C] functions done ({time.perf_counter() - t0:.0f}s)")
    report["process"] = section_process(args.with_optimisation)
    print(f"[D] /process done ({time.perf_counter() - t0:.0f}s)")
    report["core_helpers"] = section_core_helpers()
    print(f"[E] core UI helpers done ({time.perf_counter() - t0:.0f}s)")

    (out / "stage3-compare.json").write_text(json.dumps(_round(report), indent=2, default=str), encoding="utf-8")
    _print_summary(report)
    print(f"\nfull report: {out / 'stage3-compare.json'}")
    return 0


def _print_summary(report: dict) -> None:
    print(f"\npython {report['python']}")
    for row in report.get("pytest", []):
        print(f"[A] {row['original']}: old {row['old_counts']} light {row['light_counts']} "
              f"differences={len(row['differences'])}")
        for d in row["differences"]:
            print(f"      {d['test']}: old={d['old']} light={d['light']}")
    by_mode: dict[str, list] = {}
    for r in report["rig"]:
        by_mode.setdefault(r["mode"], []).append(r)
    for mode, rows in by_mode.items():
        diff = [r for r in rows if r.get("error") or not r.get("identical")]
        print(f"[B] rig {mode}: {len(rows) - len(diff)}/{len(rows)} identical")
        for r in diff:
            print(f"      {r['fixture']}: {r.get('error') or list(r['differences'])}")
    for fn, rows in report["functions"].items():
        diff = [r for r in rows if not r["identical"]]
        print(f"[C] {fn}: {len(rows) - len(diff)}/{len(rows)} identical")
    rows = report["process"]
    for key in ("pre_optimization_paths", "optimized_paths"):
        diff = [r for r in rows if not r[key]["identical"]]
        nondet = [r for r in rows if not r[key + "_old_repeat_identical"]]
        print(f"[D] /process {key}: {len(rows) - len(diff)}/{len(rows)} identical; "
              f"old-engine repeat non-deterministic in {len(nondet)}")
    by_fn: dict[str, list] = {}
    for r in report.get("core_helpers", []):
        by_fn.setdefault(r["function"], []).append(r)
    for fn, rows in by_fn.items():
        errs = [r for r in rows if "error" in r]
        diff = [r for r in rows if "error" not in r and not r["identical"]]
        nondet = [r for r in rows if "error" not in r and not r["old_repeat_identical"]]
        print(f"[E] {fn}: {len(rows) - len(diff) - len(errs)}/{len(rows)} identical; "
              f"old repeat non-deterministic in {len(nondet)}; errors {len(errs)}")
        for r in errs[:3]:
            print(f"      {r['lesson']}: {r['error']}")


if __name__ == "__main__":
    raise SystemExit(main())
