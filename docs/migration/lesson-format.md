# Lesson format (Stage 2)

A *lesson* is an engine-agnostic test case: a bitmap, the ideal centerline drawn as
SVG, and a `lesson.json` describing how to run and score it. Lessons live in
`tests/lessons/<id>/` and are run through the **live** UI pipeline
(`/upload` → `/process_immediate`) by `tests/pipeline/runner.py`.

## Folder layout

```
tests/lessons/<id>/
  image.png        # input bitmap (dark strokes on light background)
  reference.svg    # ideal centerline(s), one <path>/<polyline>/<line> per stroke
  lesson.json      # metadata, pinned params, scoring, status
```

## `lesson.json` fields

| field | meaning |
|---|---|
| `schema_version` | currently `1` |
| `id` | must equal the folder name |
| `title`, `description`, `notes` | free text |
| `category` | `crossing`, `placement` or `user` |
| `expected_behaviour` | plain-language statement of what correct output looks like |
| `source` | `{kind: generated/fixture/user/user-supplied, generator, fixture_id}`; hand-traced references add `tool` (e.g. `LightBurn`) |
| `licence`, `consent` | licence text and `{public, by}` |
| `image` | `{file, width, height}`; width/height are checked against the PNG |
| `reference` | `{file}` |
| `params` | pinned `/process_immediate` parameters. Defaults equal the UI defaults: `dark_threshold 0.2`, `min_path_length 2`, `enable_pruning false`, `normalization_mode auto`, `normalization_sensitivity medium`. Both pruning off and on are always run; `prune_off` is the scored variant. |
| `scoring.region` | `{x0,y0,x1,y1}` in SVG/pixel-edge coordinates, or `null` for the whole image |
| `scoring.tolerance_px` | max lateral error (p95) inside the region |
| `scoring.min_single_path_coverage` | fraction of each reference line that one single output path must cover within tolerance |
| `scoring.require_single_path` | if true, coverage counts only the best single path |
| `scoring.trim_ends_px` | reference length ignored at each open end (end caps) |
| `status` | `active` (must pass), `baseline` (known failing: non-strict xfail), `draft` (not run in pytest) |
| `last_run` | optional; written by `run_lessons --record`: engine commit, outcome, date |

Image size and threshold are pinned per lesson because path and junction counts
vary with both (for example, images under 600 px are supersampled).

## Coordinates

* Engine float paths are `(row, col)` in **original-image pixels**, with integer
  values at pixel centres. Supersampled extraction is mapped back with
  `(v + 0.5)/s − 0.5`.
* SVG uses pixel-edge coordinates: `x = col + 0.5`, `y = row + 0.5`, the same
  convention as the app's SVG export. The reader inverts this.
* The SVG must be registered onto the bitmap in one of three ways (tried in this
  order): an embedded `<image>` of the bitmap (its placed rectangle is the
  bitmap frame), a border rectangle (`id`, `class` or `data-role` = `border`)
  spanning the image, or `width`/`height` (or `viewBox`) equal to the image's
  pixel size. See [SVG reader](#svg-reader-testspipelinesvg_referencepy).

## Scoring

For each reference line:

1. Densify the reference and every output path (0.25 px step) and trim
   `trim_ends_px` from the reference ends.
2. Keep only the samples inside `scoring.region`.
3. **Single-path coverage**: the fraction of reference samples within
   `tolerance_px` of the best single output path.
4. **Lateral error**: report p50/p95/max of reference→best-path distance, and of
   reference→any-path distance. Uncovered samples are included, so a crossing
   that splits gives large values.

A lesson passes when every line has coverage ≥ `min_single_path_coverage` and
p95 ≤ `tolerance_px`. Graph metrics are also reported: path count, endpoints and
junctions (clustered at 1.5 px, image-border endpoints ignored), and short paths
(< 12 px).

## SVG reader (`tests/pipeline/svg_reference.py`)

stdlib + numpy only. It never calls `extract_manual_centerlines_from_svg`.

* Refuses any `<!DOCTYPE` / `<!ENTITY` before parsing.
* Caps: 2 MB file (the whole file, so an embedded base64 image counts), 200 shape
  elements, 200 000 sampled points, nesting depth 64.
* Supports path commands M/L/H/V/C/S/Q/T/A/Z, relative and absolute, with
  compact numbers and arc flags. Curves are sampled at a 0.5 **bitmap** px step
  (after transforms). Also supports `line`, `polyline`, `polygon`, `rect`,
  `circle` and `ellipse`.
* **Transforms** (Stage 2b): `matrix`, `translate`, `scale`, `rotate` (with
  optional centre), `skewX`, `skewY`, as lists and on nested groups, composed in
  document order. Malformed or degenerate transforms are rejected
  (`svg_transform_invalid`).
* **Units** (Stage 2b): unitless/`px`, `mm`, `cm`, `in`, `pt`, `pc` at 96 px/in.
  Percentages are only accepted on the root when a `viewBox` is present.
* **Registration**, in order:
  1. `image-frame`: exactly one rendered `<image>`. Its `x/y/width/height`
     rectangle, pushed through all transforms, is mapped onto the bitmap's full
     pixel size with the rendered top-left as pixel edge (0, 0). Flips are fine;
     rotation or skew is rejected (`svg_image_rotated`), aspect ratio must match
     the bitmap within 1 % (`svg_image_aspect`), and more than one image is
     rejected (`svg_multiple_images`). Root `width`/`height` and `viewBox` are
     not used. **The image data is never decoded**; it only marks the frame.
  2. `border-rect`: a `rect` marked `border` (through its transforms).
  3. `size-match`: root `width`/`height` in px (after unit conversion) or the
     `viewBox` size equal to the bitmap.
  4. Otherwise `svg_no_registration` with a hint to keep the image in the export
     or start from the [trace template](#trace-template).
* Ignores styles, `<text>` and non-rendered subtrees (`defs`, `clipPath`,
  `mask`, `pattern`, `marker`, `symbol`, `metadata`, ...).
* **Rejects** `<!DOCTYPE`/`<!ENTITY`, `<use>` and nested `<svg>`, each with a
  clear error and a stable `code` (`SvgReferenceError.code`, matching
  `docs/teach/design.md` §14.4).

### LightBurn export rules

Tested with `tests/pipeline/samples/lightburn-cross-t2-w8-a35.svg` (lesson
`user-lightburn-cross-t2-w8-a35`):

* **Keep the image in the export.** Do not hide or delete the traced bitmap
  before exporting; its placement is how the reader aligns your lines. LightBurn
  writes it with `matrix(1,0,0,-1,…)` and stores the PNG upside down so it renders
  upright; only the rendered rectangle matters.
* **Ungrouped paths**: draw one open path per stroke along the stroke centre.
  (Groups and transforms are read correctly, but ungrouped is what has been
  tested.)
* Document units **mm or px are both fine**; the image frame removes any
  dependence on units, dpi or the export's `viewBox`.
* Do not rotate, stretch or crop the bitmap in LightBurn after placing it.

Inspect any export with
`.venv\Scripts\python.exe -m tests.pipeline.inspect_svg <file.svg> --bitmap <image.png>`.

## Commands

```powershell
# all lessons, table to stdout (exit 1 only if an *active* lesson fails)
.venv\Scripts\python.exe -m tests.pipeline.run_lessons
# include user crops, /process comparison with a time limit, static HTML report outside the repo
.venv\Scripts\python.exe -m tests.pipeline.run_lessons --user --process --process-timeout 60 --html
# pytest (baseline lessons are non-strict xfail; unexpected passes are listed at the end)
.venv\Scripts\python.exe -m pytest tests/pipeline -q
# overlays + metrics JSON for review (default %TEMP%\centerline-review)
.venv\Scripts\python.exe scripts\review_pipeline_overlays.py --lessons-only
# regenerate generated lessons (keeps status/notes/last_run)
.venv\Scripts\python.exe -m tests.pipeline.generate_lessons
```

`--mark-failing-as-baseline` changes the `status` field of failing active
lessons to `baseline`. It was used once to create the initial baseline.

## Padding check

`tests.pipeline.padding.padding_check(image, inner_box, pad_px)` runs the full
image and a crop padded by `pad_px` around `inner_box`, then compares the
centerlines inside the inner box. A large difference means the crop has too
little context (or a different supersample factor) to be a faithful lesson.
