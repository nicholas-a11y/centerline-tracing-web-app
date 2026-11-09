# Centerline Extraction Web App

A Flask web app that turns bitmap line art (scans, photos, drawings) into single-stroke
centerline SVG paths. You upload an image, adjust a few sliders, preview the extracted
centerlines over the image, and download an SVG.

## Running locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python centerline_web_app.py          # http://localhost:5002 (override with PORT)
```

Production uses gunicorn: `gunicorn --workers 1 --threads 8 --timeout 300 centerline_web_app:app`
(see `Procfile`, `railway.json`, `app.yaml`). `runtime.txt` targets Python 3.11.

## How processing works (what the UI actually runs)

| Module | Role |
|---|---|
| `centerline_web_app.py` | Flask app, sessions, routes, SVG rendering |
| `centerline_core.py` | Preprocessing, auto-detect threshold, auto-tune |
| `centerline_engine_light.py` | **Current engine.** Skeleton extraction, pruning, SVG output |
| `centerline_engine.py` | **Legacy engine**, kept during migration (see below) |

The UI (`templates/index.html`) calls `POST /process_immediate` and always sends
`enable_optimization: false`. The live result is therefore:

1. threshold (`dark_threshold`, default 0.20) → `extract_skeleton_paths` (light engine);
2. drop paths shorter than `min_path_length` (default 2);
3. `prune_extracted_paths` (light engine), **only** if the "pruning" checkbox is ticked (off by default).

Path merging (`merge_nearby_paths`), overlap removal (`remove_overlapping_paths`) and circle-based
optimisation are imported from the legacy `centerline_engine.py`. They only run via `POST /process`
(not called by the UI), the background-optimisation thread (never started by the UI), auto-tune /
auto-detect in `centerline_core.py`, and the test rig. Those legacy functions are identical when
compared as parsed code (AST) to their copies in `centerline_engine_light.py`.

## Tests

```bash
pip install pytest
python -m pytest tests/
```

The last recorded baseline is in [docs/migration/stage0-baseline.md](docs/migration/stage0-baseline.md):
7 known failures on a clean clone. These are recorded, not fixed.

Note: `.gitignore` contains `test*.py`, so new test files are ignored unless they are force-added or
covered by an exception.

### Fixtures and goldens

- `tests/fixtures/definitions.py` defines analytic `curve`, `line` and `corner` shapes, each with an
  ideal sampled centerline.
- `python -m tests.fixtures.generate [--size 64 --stroke-width 3]` rasterises them to
  `tests/fixtures/data/<id>_<size>px_sw<sw>.png` plus a `.json` sidecar (tolerance, expected segment types).
- These fixtures test **curve fitting** (`fit_curve_segments`) against the ideal sample. They do not test
  centerline placement, crossings or junctions.
- `tests/fixtures/golden/*.json` are **orphaned**: the test that read and wrote them (`test_fit_curves.py`)
  was never committed. `--update-goldens` still exists in `tests/conftest.py` but nothing uses it.

### Test manager (`/test-ui`)

`/test-ui` (template `templates/test_management.html`, backend `test_ui_backend.py`) runs fixtures, shows
layered overlays (ideal path, stroke pixels, skeleton, output) and lets you change fitting parameters.
It is only enabled if `test_ui_backend` imports successfully (needs pytest); otherwise it serves the main page.

| Route | Purpose |
|---|---|
| `GET /test-ui` | Test manager page |
| `GET /api/test-runs` | List runs (persisted under `tests/fixtures/runs/`) |
| `POST /api/test-runs/execute` | Start a run: `{fixture_ids, fitting_parameters, update_goldens}` |
| `GET /api/test-runs/<run_id>[?fixture_id=..]` | Run summary / one fixture's layers |
| `GET /api/test-runs/<run_id>/progress` | Poll progress messages |

Known problems with the rig (unchanged; recorded only):

- It imports `extract_skeleton_paths` and `fit_curve_segments` from the **legacy** engine (these two
  differ from the light engine). Its default `main_app` mode instead calls `process_centerlines`, which is
  the `/process` path, not the UI's `/process_immediate` path.
- Its pytest step runs `.venv/bin/python`, which does not exist on Windows.
- It refers to `tests/test_fit_curves.py` and `tests/test_engine_path_thresholds.py`, which are not in the repository.

### Synthetic stress image

`generate_test_bitmap.py` draws a 2200×1700 scene: horizontal and diagonal lines with gaps of 2–44 px
(for tuning merge gap), crossings (+, X, skewed), T and Y junctions, open shapes (polyline, C, U), closed
shapes (rectangle, ellipse) and short stubs. It writes clean, variable-width and blurred/noisy variants
as PNG + BMP plus `centerline_gap_bridge_manifest.json` into `test_images/`.

These images have no reference paths and are not used by any test. The output directory is hard-coded to a
macOS path in `main()`, so edit it before running elsewhere.

## Known limitation: thin line crossing a thick line

Where a thin line (e.g. 2 px) crosses a thicker line (e.g. 8 px) at an angle, the skeleton usually has two
junction points joined by a short bridge segment rather than one clean X. The output then breaks into
several pieces at the crossing: four arms plus a bridge, sometimes with tiny stubs, and the thin line jogs
sideways. Near 90° it is usually four arms meeting at one point. The exact count depends on image size and
threshold. There is no committed fix; this is documented so test cases can track it.

## Migration plan (in progress)

Goal: a test suite that exercises the real engine and the live `/process_immediate` path, and then
retire `centerline_engine.py`. Stages 0–3 change no shipped code, parameters, API responses or UI.

0. Baseline: record current `pytest tests/` results (done).
1. This README.
2. Engine-agnostic "lesson" format (image + reference SVG + metadata) and a runner that drives the real
   pipeline via Flask's test client; generated crossing lessons; existing fixtures converted to lessons;
   currently failing lessons marked xfail.
3. Port legacy tests and rig logic to the light engine (copies only) and document every old-vs-new difference.
4. Design doc for a "Teach" submission area (users submit a bitmap + ideal SVG into a review queue).
   It is feature-flagged off by default and never changes any user's results.
5–6. Teach implementation and admin review/export into lessons (after approval).
7. Switch the five legacy imports to `centerline_engine_light` (after approval).
8. Remove `centerline_engine.py` and dead test/rig code once nothing imports it (after approval).

## Teach (feature-flagged, off by default)

Teach (`teach/`, design in [docs/teach/design.md](docs/teach/design.md)) is a public page where people
submit a bitmap and a hand traced SVG of the ideal centrelines. Each case is scored in a subprocess
against the live `/process_immediate` pipeline and stored for review. It is registered by one line at
the end of `centerline_web_app.py` and adds nothing to the app unless `TEACH_ENABLED=1`. If the
configuration is unsafe or incomplete (for example the admin token is missing or shorter than 32
characters), Teach stays off, `/teach` returns 404 and one log line says why. Admin review and lesson
export come in a later stage.

Try it locally (PowerShell):

```powershell
$env:TEACH_ENABLED = "1"
$env:TEACH_ADMIN_TOKEN = (py -c "import secrets; print(secrets.token_urlsafe(32))")
python centerline_web_app.py
# open http://localhost:5002/teach
```

With no `R2_*` variables set, data is stored in a local folder: `TEACH_LOCAL_DIR` if set, otherwise
`~/.centerline-teach` (outside the repo). The page shows a "LOCAL MODE" banner. Delete that folder to
start again. The Teach tests run with `pytest tests/teach`.

When Teach is enabled, the main page also shows an unchecked **Teach** checkbox (with Teach off the
main page is unchanged). Ticking it offers **Download the cropped image**, which gives exactly the
pixels the engine processed, and after a run a **Send this to Teach** link. The link opens `/teach` with
the run's settings filled in and the cropped image already attached (fetched from the app's session copy;
if that has expired, attach the downloaded file instead), so you only add the hand traced SVG. To prepare a case: crop to the problem area with the app's crop tools,
leaving a little context, then trace over the cropped image in your vector program. Keep the image in
the exported SVG (in LightBurn you can disable the image layer's output and it still stays in the
export). Details are in [docs/teach/design.md](docs/teach/design.md) §5.1.

## Deployment

### Heroku
1. Install Heroku CLI
2. `heroku create your-app-name`
3. `git init && git add . && git commit -m "Initial commit"`
4. `heroku git:remote -a your-app-name`
5. `git push heroku main`

### Railway
1. Connect GitHub repository
2. Deploy automatically

### Render
1. Connect GitHub repository
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `python centerline_web_app.py`

See `DEPLOYMENT.md` for more detail.