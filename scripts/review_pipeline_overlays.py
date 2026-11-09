"""Dev-only review tool: run the live pipeline on images and write overlays + metrics.

For each image the default UI pipeline (upload -> /process_immediate, pruning
off; plus pruning on) is run in-process. Output per image and variant:
  <out>/<id>__<variant>.png       centerlines over the original (reference in green
                                  if available, junctions red, free endpoints orange)
  <out>/<id>__<variant>__zoom.png crop of the scoring region, if one is defined
  <out>/<id>.json                 metrics: path count, endpoints, junctions, short
                                  bridge segments, path lengths, and scores if a
                                  reference exists
No network access, no API keys; nothing is written inside the repository by default.

Usage (repo root):
    .venv\\Scripts\\python.exe scripts\\review_pipeline_overlays.py                 # all lessons + user_supplied
    .venv\\Scripts\\python.exe scripts\\review_pipeline_overlays.py path\\to\\img.png  # specific images/folders
    ... --out DIR   (default: %TEMP%/centerline-review)
    ... --lessons-only | --user-only
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.pipeline.evaluate import evaluate, evaluation_record  # noqa: E402
from tests.pipeline.lesson import DEFAULT_PARAMS, DEFAULT_SCORING, discover_lessons  # noqa: E402
from tests.pipeline.overlay import save_overlay  # noqa: E402
from tests.pipeline.user_images import IMAGE_SUFFIXES, UserImage, discover_user_images  # noqa: E402

DEFAULT_OUT = Path(tempfile.gettempdir()) / "centerline-review"


def _adhoc_items(paths: list[str]) -> list[UserImage]:
    items = []
    for raw in paths:
        p = Path(raw)
        files = sorted(x for x in p.iterdir() if x.suffix.lower() in IMAGE_SUFFIXES) if p.is_dir() else [p]
        for f in files:
            svg = f.with_suffix(".svg")
            js = f.with_suffix(".json")
            meta = json.loads(js.read_text(encoding="utf-8")) if js.exists() else {}
            items.append(UserImage(id=f.stem, image_path=f, reference_path=svg if svg.exists() else None,
                                   meta=meta))
    return items


def review(items, out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i, item in enumerate(items, 1):
        print(f"[{i}/{len(items)}] {item.id}", file=sys.stderr)
        try:
            ev = evaluate(item)
        except Exception as exc:
            records.append({"id": item.id, "error": f"{type(exc).__name__}: {exc}"})
            continue
        gray = item.load_gray()
        region = item.scoring.get("region")
        rec = evaluation_record(ev)
        rec["image"] = str(item.image_path)
        rec["params"] = item.params
        rec["overlays"] = {}
        for name, v in ev.variants.items():
            if v.result.error:
                continue
            g = v.graph
            rec["variants"][name]["endpoints"] = g.get("endpoints")
            rec["variants"][name]["junctions"] = g.get("junctions")
            rec["variants"][name]["path_lengths"] = g.get("path_lengths")
            title = f"{item.id} [{name}] paths={g.get('path_count')} junctions={len(g.get('junctions') or [])}"
            files = [save_overlay(out_dir / f"{item.id}__{name}.png", gray, v.result.paths, ev.reference, g,
                                  region, title=title).name]
            if region:
                files.append(save_overlay(out_dir / f"{item.id}__{name}__zoom.png", gray, v.result.paths,
                                          ev.reference, g, region, zoom_to_region=True).name)
            rec["overlays"][name] = files
        (out_dir / f"{item.id}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        records.append(rec)
    (out_dir / "index.json").write_text(json.dumps(
        [{"id": r["id"], "outcome": r.get("outcome"), "error": r.get("error")} for r in records], indent=2),
        encoding="utf-8")
    return records


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Write centerline overlays and metrics for review.")
    ap.add_argument("inputs", nargs="*", help="image files or folders (default: lessons + user_supplied)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--lessons-only", action="store_true")
    group.add_argument("--user-only", action="store_true")
    args = ap.parse_args(argv)

    if args.inputs:
        items = _adhoc_items(args.inputs)
    else:
        items = [] if args.user_only else list(discover_lessons())
        items += [] if args.lessons_only else discover_user_images()
    if not items:
        print("nothing to review")
        return 0
    records = review(items, args.out)
    print(f"wrote {len(records)} metric files and overlays to {args.out}")
    print(f"defaults used: params={DEFAULT_PARAMS} scoring={DEFAULT_SCORING}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
