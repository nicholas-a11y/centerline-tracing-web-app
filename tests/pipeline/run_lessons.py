"""Run every lesson through the live pipeline and print a results table.

Usage (from the repo root):
    .venv\\Scripts\\python.exe -m tests.pipeline.run_lessons
    ... --filter cross          only lessons whose id contains "cross"
    ... --user                  also run test_images/user_supplied/ images
    ... --process               also run /process (enable_optimization=true) per lesson
    ... --process-timeout 60    seconds before a /process run is killed and reported as TIMEOUT
    ... --html [DIR]            write a static HTML report with overlays (default: %TEMP%/centerline-lessons-report)
    ... --json FILE             write machine-readable results
    ... --record                write last_run (engine commit, date, summary) into each lesson.json
    ... --mark-failing-as-baseline   set status "baseline" on active lessons that fail today

Normal runs never modify files inside the repository.
Exit status is 1 if an "active" lesson fails or errors; baseline lessons never fail the run.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import html
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.pipeline.evaluate import PRIMARY_VARIANT, evaluate, evaluation_record, variant_cell  # noqa: E402
from tests.pipeline.lesson import discover_lessons, engine_commit, write_lesson_json  # noqa: E402
from tests.pipeline.overlay import save_overlay  # noqa: E402
from tests.pipeline.user_images import discover_user_images  # noqa: E402

DEFAULT_HTML_DIR = Path(tempfile.gettempdir()) / "centerline-lessons-report"


def _print_table(evals, include_process: bool) -> None:
    cols = ["lesson", "status", "outcome", "prune_off (UI default)", "prune_on"]
    if include_process:
        cols += ["/process pre-opt", "/process optimised"]
    rows = []
    for ev in evals:
        row = [ev.id, ev.status, ev.outcome, variant_cell(ev.variants.get("prune_off")),
               variant_cell(ev.variants.get("prune_on"))]
        if include_process:
            row += [variant_cell(ev.variants.get("process_pre_opt")),
                    variant_cell(ev.variants.get("process_optimized"))]
        rows.append(row)
    widths = [max(len(str(r[i])) for r in rows + [cols]) for i in range(len(cols))]
    line = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))
    counts = {}
    for ev in evals:
        counts[ev.outcome] = counts.get(ev.outcome, 0) + 1
    print()
    print("summary:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    unexpected = [ev.id for ev in evals if ev.outcome == "UNEXPECTED PASS"]
    if unexpected:
        print("UNEXPECTED PASSES (baseline lessons that now pass; consider promoting to active):")
        for lid in unexpected:
            print("  -", lid)


def _write_html(evals, items, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = engine_commit() or "unknown"
    parts = [
        "<!doctype html><meta charset='utf-8'><title>Centerline lessons</title>",
        "<style>body{font-family:sans-serif;margin:16px}table{border-collapse:collapse}"
        "td,th{border:1px solid #ccc;padding:4px;vertical-align:top;font-size:12px}"
        "img{max-width:360px;image-rendering:pixelated}.UNEXPECTED{background:#ffd}.FAIL,.ERROR{background:#fdd}"
        "</style>",
        f"<h1>Centerline lessons</h1><p>engine commit {html.escape(commit)}; generated "
        f"{_dt.datetime.now().isoformat(timespec='seconds')}. Green = reference, coloured = engine paths, "
        "red circles = junctions, orange squares = free endpoints, cyan box = scoring region.</p>",
        "<table><tr><th>lesson</th><th>status / outcome</th>",
    ]
    variant_names = list(evals[0].variants) if evals else []
    parts += [f"<th>{html.escape(v)}</th>" for v in variant_names] + ["</tr>"]
    for ev, item in zip(evals, items):
        gray = item.load_gray()
        region = item.scoring.get("region")
        cls = ev.outcome.split()[0]
        parts.append(f"<tr class='{html.escape(cls)}'><td><b>{html.escape(ev.id)}</b><br>"
                     f"{html.escape(getattr(item, 'meta', {}).get('title', ''))}</td>"
                     f"<td>{html.escape(ev.status)}<br><b>{html.escape(ev.outcome)}</b></td>")
        for name in variant_names:
            v = ev.variants.get(name)
            if v is None or v.result.error:
                parts.append(f"<td>{html.escape(v.result.error if v else '-')}</td>")
                continue
            png = save_overlay(out_dir / "img" / f"{ev.id}__{name}.png", gray, v.result.paths, ev.reference,
                               v.graph, region)
            zoom = ""
            if region:
                zpng = save_overlay(out_dir / "img" / f"{ev.id}__{name}__zoom.png", gray, v.result.paths,
                                    ev.reference, v.graph, region, zoom_to_region=True)
                zoom = f"<br><img src='img/{zpng.name}'>"
            parts.append(f"<td><img src='img/{png.name}'>{zoom}<br>{html.escape(variant_cell(v))}</td>")
        parts.append("</tr>")
    parts.append("</table>")
    index = out_dir / "index.html"
    index.write_text("\n".join(parts), encoding="utf-8")
    (out_dir / "results.json").write_text(json.dumps([evaluation_record(e) for e in evals], indent=2),
                                          encoding="utf-8")
    return index


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run centerline lessons through the live pipeline.")
    ap.add_argument("--filter", default="")
    ap.add_argument("--user", action="store_true", help="include test_images/user_supplied images")
    ap.add_argument("--process", action="store_true", help="also run /process with enable_optimization=true")
    ap.add_argument("--process-timeout", type=float, default=60.0)
    ap.add_argument("--html", nargs="?", const=str(DEFAULT_HTML_DIR), default=None)
    ap.add_argument("--json", dest="json_out", default=None)
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--mark-failing-as-baseline", action="store_true")
    args = ap.parse_args(argv)

    items = [ls for ls in discover_lessons() if args.filter in ls.id]
    if args.user:
        items += [u for u in discover_user_images() if args.filter in u.id]
    if not items:
        print("no lessons found")
        return 0

    evals = []
    for i, item in enumerate(items, 1):
        print(f"[{i}/{len(items)}] {item.id}", file=sys.stderr)
        evals.append(evaluate(item, include_process=args.process, process_timeout_s=args.process_timeout))
    print()
    _print_table(evals, args.process)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps([evaluation_record(e) for e in evals], indent=2),
                                       encoding="utf-8")
    if args.html:
        index = _write_html(evals, items, Path(args.html))
        print(f"\nHTML report: {index}")

    lessons = [it for it in items if hasattr(it, "folder")]
    by_id = {ev.id: ev for ev in evals}
    if args.mark_failing_as_baseline:
        for ls in lessons:
            ev = by_id[ls.id]
            if ls.status == "active" and ev.outcome in ("FAIL", "ERROR"):
                ls.meta["status"] = "baseline"
                write_lesson_json(ls.folder, ls.meta)
                print(f"marked baseline: {ls.id}")
    if args.record:
        commit = engine_commit()
        stamp = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
        for ls in lessons:
            ev = by_id[ls.id]
            prim = ev.variants[PRIMARY_VARIANT]
            ls.meta["last_run"] = {
                "engine_commit": commit, "date": stamp, "outcome": ev.outcome,
                "summary": prim.score.summary() if prim.score else None,
            }
            write_lesson_json(ls.folder, ls.meta)
        print(f"recorded last_run for {len(lessons)} lessons (engine {commit})")

    failing_active = [ev.id for ev in evals if ev.status == "active" and ev.outcome in ("FAIL", "ERROR")]
    return 1 if failing_active and not args.mark_failing_as_baseline else 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    raise SystemExit(main())
