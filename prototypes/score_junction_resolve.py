"""Score the junction-resolution prototype against every lesson (and optional Teach submissions).

    .venv\\Scripts\\python -m prototypes.score_junction_resolve [--filter ID] [--submission DIR ...]
                                                              [--overlays OUT_DIR] [--json OUT.json]

For each case the live pipeline runs once with the case's settings (the same /upload +
/process_immediate route the UI uses). The returned paths are scored as they are
("baseline") and after prototypes.junction_resolve ("resolved"), with the lesson's own
scoring rules. Nothing in the engine or the app is changed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lessonkit.lesson import discover_lessons  # noqa: E402
from lessonkit.overlay import render_overlay  # noqa: E402
from lessonkit.runner import run_live  # noqa: E402
from lessonkit.scoring import score_paths  # noqa: E402
from prototypes.junction_resolve import resolve_junctions  # noqa: E402

DEFAULT_OUT = Path(tempfile.gettempdir()) / "centerline-junction-prototype"


class _Case:
    def __init__(self, cid, status, image_path, params, scoring, reference):
        self.id, self.status, self.image_path = cid, status, Path(image_path)
        self.params, self.scoring, self.reference = params, scoring, reference


def _lesson_cases(flt: str):
    for ls in discover_lessons():
        if flt in ls.id:
            yield _Case(ls.id, ls.status, ls.image_path, ls.params, ls.scoring, ls.load_reference())


def _submission_case(folder: Path) -> _Case:
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    ref = json.loads((folder / "reference.json").read_text(encoding="utf-8"))
    scoring = {"tolerance_px": 1.0, "min_single_path_coverage": 0.95, "require_single_path": True}
    scoring.update({k: v for k, v in (meta.get("scoring") or {}).items() if k in ("region", "tolerance_px")})
    return _Case(f"teach-{meta.get('number', folder.name)}", "submission", folder / "image.png",
                 meta.get("params") or {}, scoring, [np.asarray(p, float) for p in ref["paths"]])


def _summary(sc) -> dict:
    s = sc.summary()
    return {"passed": s["passed"], "cov": s["min_single_coverage"], "p95": s["max_p95_px"],
            "paths": s["path_count"], "junctions": s["junctions_in_region"], "short": s["short_paths_in_region"]}


def _cell(s: dict) -> str:
    p95 = "-" if s["p95"] is None else f"{s['p95']:.2f}"
    return f"{'PASS' if s['passed'] else 'fail'} cov={s['cov']:.2f} p95={p95:>6} paths={s['paths']} J={s['junctions']}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--filter", default="")
    ap.add_argument("--submission", action="append", default=[], help="Teach submission folder")
    ap.add_argument("--overlays", nargs="?", const=str(DEFAULT_OUT), default=None)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args(argv)

    cases = list(_lesson_cases(args.filter)) + [_submission_case(Path(d)) for d in args.submission]
    out_dir = Path(args.overlays) if args.overlays else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    rows, changed_pass = [], {"fixed": [], "broken": []}
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case.id}", file=sys.stderr)
        res = run_live(case.image_path, case.params)
        if res.error:
            rows.append({"id": case.id, "status": case.status, "error": res.error})
            continue
        with Image.open(case.image_path) as im:
            gray = np.asarray(im.convert("L"), dtype=float) / 255.0
        resolved, reports = resolve_junctions(res.paths, gray)
        base = score_paths(res.paths, case.reference, case.scoring, res.image_shape)
        new = score_paths(resolved, case.reference, case.scoring, res.image_shape)
        row = {"id": case.id, "status": case.status, "baseline": _summary(base), "resolved": _summary(new),
               "clusters": [r.__dict__ for r in reports]}
        rows.append(row)
        if new.passed and not base.passed:
            changed_pass["fixed"].append(case.id)
        if base.passed and not new.passed:
            changed_pass["broken"].append(case.id)
        if out_dir:
            region = case.scoring.get("region")
            for tag, paths, sc in (("baseline", res.paths, base), ("resolved", resolved, new)):
                render_overlay(gray, paths, case.reference, sc.graph, region,
                               title=f"{case.id} {tag}").save(out_dir / f"{case.id}.{tag}.png")

    width = max(len(r["id"]) for r in rows) if rows else 10
    print(f"{'case':<{width}}  {'status':<10}  {'baseline':<44}  resolved")
    for r in rows:
        if "error" in r:
            print(f"{r['id']:<{width}}  {r['status']:<10}  ERROR {r['error']}")
            continue
        print(f"{r['id']:<{width}}  {r['status']:<10}  {_cell(r['baseline']):<44}  {_cell(r['resolved'])}")
    print(f"\nnewly passing: {len(changed_pass['fixed'])} {changed_pass['fixed']}")
    print(f"newly failing: {len(changed_pass['broken'])} {changed_pass['broken']}")
    if out_dir:
        print(f"overlays: {out_dir}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"rows": rows, **changed_pass}, indent=2, default=float),
                                       encoding="utf-8")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    raise SystemExit(main())
