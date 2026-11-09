# User-supplied crops

Put your own cropped bitmaps here so they run through the live pipeline with the
lessons.

```
my-crossing.png     # required: the crop (png/jpg/bmp/tif)
my-crossing.svg     # optional: ideal centerline(s), same pixel size as the PNG
my-crossing.json    # optional: overrides, e.g. {"title": "...", "params": {...},
                    #           "scoring": {"region": {...}, "tolerance_px": 1.0}, "status": "baseline"}
```

* Without an SVG, the image is a `draft`. It still gets overlays and metrics, but
  it is not scored.
* With an SVG, it defaults to `baseline`: it is scored, and a failure is
  reported as an expected failure.
* The SVG rules are in `docs/migration/lesson-format.md`. In short: one stroke
  per path, the size must match the bitmap (or include a border rect), and no
  transforms or mm/in units for now.
* Crop with some context around the problem. Use
  `tests.pipeline.padding.padding_check` to confirm that the crop behaves like
  the full image.

Run:

```powershell
.venv\Scripts\python.exe -m tests.pipeline.run_lessons --user --filter my-crossing --html
.venv\Scripts\python.exe scripts\review_pipeline_overlays.py --user-only
```

Promoting a crop to a permanent lesson means copying it into
`tests/lessons/<id>/` with a `lesson.json`, which is a manual step for now.
