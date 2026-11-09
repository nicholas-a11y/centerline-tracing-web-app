# Stage 3: old vs light engine differences, and dead-code report

Status: **findings only**. Nothing in this document has been fixed, re-goldened or
removed. Every item below is something for the owner to review.

Environment: Windows, Python 3.12.3, numpy 1.26.4, scipy 1.17.1, scikit-image 0.26.0,
pillow 12.3.0, Flask 3.1.2, pytest 9.1.1. Python 3.11 was **not** available locally
(see [Python 3.11](#python-311-runtimetxt)).

How to reproduce (writes JSON outside the repo, takes about 4 minutes):

```
python -m tests.light_port.compare_engines --with-optimisation
python -m pytest tests/light_port
```

## What was built

`tests/light_port/` contains copies of the legacy tests and the `/test-ui` rig,
retargeted to `centerline_engine_light`. The originals are untouched.

| file | copied from | change |
|---|---|---|
| `light_app.py` | new | `light_engine_imports()` rebinds the 5 old-engine names in `centerline_web_app` and `merge_nearby_paths` in `centerline_core` to the light engine. It simulates the Stage 7 import switch in memory only. |
| `conftest.py` | new | Autouse fixture that applies the rebinding. Non-strict xfail for the 7 tests that also fail in the original (Stage 0 baseline). |
| `test_path_optimization.py` | `tests/test_path_optimization.py` | `import centerline_engine as engine` → light |
| `test_small_image_supersampling.py` | `tests/test_small_image_supersampling.py` | monkeypatch target `centerline_engine.create_fast_paths` → `centerline_engine_light.create_fast_paths` (×4) |
| `test_web_app_optimization_state.py` | `tests/test_web_app_optimization_state.py` | same monkeypatch target (×3), and `_normalize_fitting_parameters` imported from the ported rig |
| `test_resolution_scaling.py` | `tests/test_resolution_scaling.py` | none (the conftest rebinding is enough) |
| `rig_backend_light.py` | `test_ui_backend.py` | light `extract_skeleton_paths`/`fit_curve_segments`; `sys.executable` instead of `.venv/bin/python`; runs written to `%TEMP%`; never auto-generates fixtures; pytest targets point at the ports; missing `test_engine_path_thresholds.py` dropped |
| `test_rig_backend_light.py` | new | smoke tests for the ported rig (all 17 fixtures, both analysis modes) |
| `compare_engines.py` | new | the old-vs-light harness behind every table below |

Each copied file starts with a `PORTED COPY` header listing its substitutions.
These tests were already light-engine or core-only, so they were not ported:
`test_fast_paths_vectorized.py`, `test_line_only_fit.py` and `test_preprocessing_profiles.py`.
`test_recenter_to_crest.py` is out of scope.

## Summary of results

| comparison | corpus | result | affects live UI? |
|---|---|---|---|
| A. pytest, original vs port | 4 legacy files, 107 tests | **identical outcome and message for every test** (the 7 baseline failures fail identically) | no |
| B. `/test-ui` rig fixture analysis | 17 fixtures × 4 modes (curve_fitter fit off/on, main_app opt off/on) | **68/68 identical** | no (dev tool) |
| C1. `extract_skeleton_paths` | 45 images × 3 settings | 107/135 identical; **all 28 differences come from supersampling** | **no**: the UI already uses light |
| C2. `create_fast_paths` | 45 skeletons | 45/45 identical | no |
| C3. `fit_curve_segments` | 45 images × curve fitting off/on | 90/90 identical | no |
| C4. `create_svg_output` | 45 images × curve fitting off/on | 90/90 identical | no |
| D. `/process` (`process_centerlines`) with old vs light bindings | 28 lessons × optimisation off/on | 56/56 identical (pre-optimisation and optimised paths); old engine deterministic on repeat | no |
| E. `centerline_core` UI helpers with old vs light `merge_nearby_paths` | 28 lessons × 4 functions | 112/112 identical; deterministic on repeat | **these are UI buttons**, but no change |

Conclusion for Stage 7: on this corpus, switching the 5 imports in
`centerline_web_app.py` and the 1 import in `centerline_core.py` to
`centerline_engine_light` changes **no** observed output on any route (rows D and E).

This is **not** a claim that the two engines are equivalent. The old and light
`extract_skeleton_paths` differ in 28 of 135 cases (row C1, finding F1). That
function is not one of the imports Stage 7 switches, and the app already uses the
light version. The difference therefore matters only for code that still calls the
old extractor: the `/test-ui` rig and `tests/test_path_optimization.py`. It will
change their results when they move to light (Stage 8). On the 64 px fixtures those
callers use today, both engines supersample ×3 (the only band where the factors
match), so their outputs match (rows A and B).

## Findings

### F1. Supersampling thresholds differ (extract only)

| constant | old | light |
|---|---|---|
| `SMALL_IMAGE_SUPERSAMPLE_TARGET_MAX_DIM` | 160 | 600 |
| `SMALL_IMAGE_SUPERSAMPLE_MAX_FACTOR` | 4 | 3 |

So 64 px images are supersampled ×3 by both engines, but a 256 px image is
supersampled only by light. Of the 28 lesson images, 10 differ at
rig settings (threshold 0.5, min_object_size 3) and 14 at UI settings (0.2, 10). The
path counts are equal except in one case, and points move by 1.0–3.2 px.
With `enable_small_image_supersample=False`, **all 45 images are identical** in both
settings.

* `cross-t8-w8-a60-equal` at rig settings: old gives 5 paths, light gives 9.
* Affects live UI: **no**. `/process_immediate` and `/process` already call light
  extraction (`centerline_web_app.py` lines 1363, 1372, 3061).
* Affects non-UI callers: **yes, for every longest side under 600 px except
  54–79 px**. The supersample factor is `min(cap, ceil(target / max_dim))` when
  `max_dim < target` (engine lines 243 and 251):

  | longest side | old factor | light factor |
  |---|---|---|
  | ≤ 53 px | 4 | 3 |
  | 54–79 px | 3 | 3 |
  | 80–159 px | 2 | 3 |
  | 160–299 px | none | 3 |
  | 300–599 px | none | 2 |
  | ≥ 600 px | none | none |

  Two callers use the old extractor: the legacy rig (`test_ui_backend.py` line 21)
  and `tests/test_path_optimization.py` (lines 3, 333, 367, 390, 407). Both use
  64 px fixtures today, which fall in the one matching band, so they see no change.
  Any other image size in either would extract differently once retargeted to light.

### F2. Light scales `min_path_length` and `min_object_size` with the supersample factor

At UI settings (0.2, min_object_size 10), 4 of the 64 px fixtures give fewer paths
on light: `curve-gentle-convex` 5→1, `line-diagonal-45` 5→1, `corner-l-shape-r45` 9→2,
`square-opposing-curves` 4→2. The extra old-engine paths are short end spurs.
Scaling `min_path_length` (3 → 9 at ×3) removes them. Scaling `min_object_size` alone makes
no difference on this corpus. The same 4 appear as the `fixture-*-64px-sw3` lessons.
Affects live UI: **no** (UI is light already). On these 4 the light behaviour is arguably the better one.

### F3. Light uses the vectorized fast-path walker on large skeletons

`USE_VECTORIZED_FAST_PATHS = True` and `VECTORIZED_FAST_PATHS_MIN_PIXELS = 40000`
exist only in light. None of the 45 images reaches 40 000 skeleton pixels, so this
corpus does not exercise it. `tests/test_fast_paths_vectorized.py` covers it separately.
Affects live UI: yes in principle (large images), but it is already live; nothing changes in Stage 7.

### F4. `create_fast_paths` refactor

Light splits the function into `_build_fast_path_neighbors`,
`_build_fast_path_junction_state`, `_classify_fast_path` and
`_recover_discarded_same_cluster_paths`. Output is identical on all 45 skeletons.

### F5. `fit_curve_segments`: closed-loop handling

Light splits a closed loop at its far point before RDP
(`centerline_engine_light.py` around line 6331, under `if has_non_line_segment`).
It also uses `is_closed_loop` at ≥3 vertices instead of ≥4. Neither branch changed any of
the 90 fits, so **this corpus does not cover the change**. A lesson with a closed
curved loop and curve fitting on would be needed to observe it.
Affects live UI: only when curve fitting is enabled, and the UI already uses light.

### F6. `create_svg_output`: `return_metadata` added; `circle_system` unused in both

The output is identical. The `circle_system` parameter is accepted but never used by
either engine (a dead parameter).

### F7. Test-only hook steers light extraction through the OLD module

`_resolve_legacy_fast_path_extractor()` (`centerline_engine_light.py` line 3566, called
at line 379) returns `sys.modules["centerline_engine"].create_fast_paths` if a test has
monkeypatched it. The legacy tests `test_small_image_supersampling.py` and
`test_web_app_optimization_state.py` patch the old module to control the light engine.
They will break when `centerline_engine.py` is removed. The ports patch
`centerline_engine_light.create_fast_paths` instead, which works because the hook falls
back to the light module global at call time.
Affects live UI: no. The hook can be deleted in Stage 8, together with the original tests.

### F8. Auto-tune DOES depend on the old-engine merge, and is a UI feature

`centerline_core.py` line 16 imports `merge_nearby_paths` from `centerline_engine`. It is
called at the following lines:

| core line | function | reached from | UI? |
|---|---|---|---|
| 537 | `auto_detect_min_path_length` | `/auto_detect_min_path_length` (app 1568) | **yes**: `index.html` 5978 |
| 786 | `auto_detect_dark_threshold` | `/auto_detect_threshold` (app 1625) and the job at app 1722 | **yes**: `index.html` 5831, 5884 |
| 1108 | `auto_tune_extraction_parameters` › `_evaluate_threshold` | `/auto_tune_extraction` (app 1936) and `_run_auto_tune_job` (app 2348) | **yes**: `index.html` 5087 |
| 1733 | `_extract_single_tile_preview_entry` | `/auto_tune_tile_preview` (app 2200) | **yes**: `index.html` 3769 |

This corrects the earlier summary "merge runs only via /process, rig, background
optimisation and auto-tune". Auto-tune and auto-detect are UI buttons, so the
old-engine merge does shape the **parameter suggestions** the UI offers. However:

* `merge_nearby_paths` is identical in both engines when compared as parsed code
  (AST), and none of its helpers touch the 4 functions that differ; and
* comparison E ran all four functions on the 28 lessons with old and with light
  bindings: **112/112 identical**.

So switching this import is expected to be behaviour-neutral.

### F9. Where the other old-engine imports are reached

| name | call sites in `centerline_web_app.py` | reached from | UI? |
|---|---|---|---|
| `merge_nearby_paths`, `remove_overlapping_paths` | 1398, 1409 (`process_centerlines`) | `POST /process` (app 3942) | no (no UI caller; the rig and tests use it) |
| `merge_nearby_paths` | 3447, 3476 (`background_optimization`) | thread started in `/process_immediate` (app 3301–3303) only if `enable_optimization` is true | no: the UI always sends `false` |
| `CircleEvaluationSystem`, `optimize_path_with_custom_params` | 1482, 1511 (`process_centerlines`); 3546, 3601, 3611 (`background_optimization`) | as above | no |
| `optimize_path_with_circles` | imported, **never called** in the app | none | no |

### F10. `/process_immediate` truncates coordinates with `int()` (record only)

`centerline_web_app.py` lines 3075 and 3227:
`[int(point[0]) + crop_offset_row, int(point[1]) + crop_offset_col]`.
Supersampled paths are float sub-pixel coordinates in the original image frame. `int()`
floors them (they are non-negative), so the JSON returned to the UI is biased by about
−0.5 px per axis (mean about −0.25 to −0.5 depending on the image). Pinned by
`tests/pipeline/test_coordinate_frame.py`. The lesson scorer uses the unrounded paths.
Affects live UI: **yes**. This is current live behaviour, and it is unchanged and not fixed here.

### F11. 0.5 px offset on even-width strokes at ≥600 px (record only)

At or above 600 px, light does not supersample. The skeleton then lies on whole pixels,
so a stroke with an even width has its centreline 0.5 px off the true centre. The
`native-700px` lesson therefore uses a 0.5 px tolerance. Pinned in
`tests/pipeline/test_coordinate_frame.py`.
Affects live UI: **yes**. This is current live behaviour, and it is unchanged and not fixed here.

### F12. Mirrored baseline failures

The 7 original tests that fail in the Stage 0 baseline fail identically in the ports,
with the same assertion messages:

* `test_path_optimization.py`:
  * `…closed_loop_tangent_anchors_for_true_circle` (`87 <= 45`)
  * `…tiny_open_corner_arising_segment`
  * `…rotated_open_corner_cluster` (`17 == 3`)
* `test_web_app_optimization_state.py`:
  * `test_download_svg_disabled_optimization_uses_direct_single_layer`
  * `…final_results_disabled_optimization…`
  * `…query_overrides_include_image`
  * `test_generate_svg_defaults_curve_fitting_off`

So these failures are not caused by the engine split. They fail the same way on light.
In the port they are non-strict xfail; `compare_engines` runs them with `--runxfail`
to compare raw outcomes.

## Dead-code report

Evidence comes from `git ls-files` plus word-boundary searches across all tracked `.py`,
`.html` and `.js` files. "Refs" counts references excluding the definition itself.

### D1. Orphaned goldens and golden tooling

| item | evidence |
|---|---|
| `tests/fixtures/golden/*.json` (18 tracked files) | No tracked code reads `fixtures/golden` except `GOLDEN_DIR` (`tests/conftest.py` line 37), which has 1 reference: its definition. |
| `tests/test_fit_curves.py` | Never in git history (`git log --all -- tests/test_fit_curves.py` is empty). Only referenced by the rig at `test_ui_backend.py` line 193. |
| `--update-goldens` option and `update_goldens` fixture (`tests/conftest.py` lines 43, 51) | Only consumer would be `test_fit_curves.py`. The rig passes the flag (line 195) to a run that collects nothing. |
| `flatten_all_fitted_paths` (`tests/conftest.py` line 135) | 0 refs. |
| "Update goldens" checkbox (`templates/test_management.html` line 724) → `/api/test-runs` (app 2565) | Plumbing works, but the target test file is missing. |

### D2. `/test-ui` rig features that cannot run

| item | evidence |
|---|---|
| Rig pytest step, `main_app` mode | Command lists `tests/test_engine_path_thresholds.py`, which does not exist. Reproduced: pytest exits **4, "no tests ran"**, even with a valid interpreter. |
| Rig pytest step, `curve_fitter` mode | Command runs only `tests/test_fit_curves.py` (missing). Reproduced: exit 4. |
| `.venv/bin/python` (`test_ui_backend.py` lines 183, 193) | Unix-only path; fails on Windows before pytest starts. |
| `TEST_UI_*` env vars (`test_ui_backend.py` lines 199–206) | Set but read by nothing (no tracked file reads them). |
| Node-result parser (lines 216–228) | In `main_app` mode pytest runs with `-q`, so no `tests/... PASSED` lines are printed, and failure lines start with `FAILED`, not `tests/`. Parsing therefore returns nothing even when tests run. |
| Fixture analysis (`_fixture_analysis`, `_fixture_analysis_main_app`) | **Works** (all 17 fixtures, both modes, in the port and the original). This is the only useful part of the rig today. |

### D3. Legacy configuration and code in `centerline_engine_light.py`

All of these are also present in `centerline_engine.py`.

| item | line | refs outside the engines | notes |
|---|---|---|---|
| `INPUT_PATH = "/Users/nicholas/Downloads/tractor.png"` | 45 | 0 | used only by `main()` and the `__main__` block |
| `OUTPUT_PATH = "/Users/nicholas/Downloads/output_optimized.svg"` | 46 | 0 | **default** of `create_svg_output(output_path=…)` (line 3304). The app's only call (app line 1244) passes `output_path=None`, so it is never written. Any caller that omits the argument would try to write to a macOS path. |
| `MANUAL_CENTERLINE_PATH`, `ML_MODEL_PATH` | 81, 82 | 0 | `/Users/nicholas/Downloads/...`; used only by `load_training_paths` and the `__main__` block |
| `ENABLE_ML_TRAINING`, `SHOW_ML_PATHS`, `SHOW_TRAINING_PATHS` | 80, 63, 64 | 0 | constant `False` flags |
| `SKLEARN_AVAILABLE = False` | 27 | 0 | hard-coded; ML is unreachable |
| `CenterlineMLTrainer` | 4322 (160 lines) | 0 (0 inside) | never instantiated. `add_manual_training_data` uses `self.ml_trainer`, which the class never defines, so it would raise if called. |
| `create_annotation_svg` ("annotation template") | 4528 | 0 | called only from `__main__ --create-annotation` (4887) |
| `save_manual_centerline` | 4483 | 0 | called from `extract_manual_centerlines_from_svg` and `__main__` (4916) |
| `load_manual_centerlines` | 4508 | 0 (0 inside) | never called |
| `extract_manual_centerlines_from_svg`, `parse_svg_file_for_paths`, `parse_svg_path_data` | 4700, 4565, 4603 | 0 | chain reachable only from `__main__ --extract-from-svg` (4901); writes to `MANUAL_CENTERLINE_PATH`. The lesson suite uses its own `tests/pipeline/svg_reference.py` instead. |
| `parse_svg_path_to_centerline` | 4554 | 0 (0 inside) | never called |
| `load_training_paths` | 4820 | 0 (0 inside) | never called |
| `main()` and the `if __name__ == "__main__"` block | 4721, 4874 | 0 | script entry for the original desktop workflow; reads `INPUT_PATH` |
| `skeleton_to_paths` | 497 | 0 (0 inside) | never called |
| `create_super_long_paths` | 4145 (176 lines) | 0 (0 inside) | never called |
| `_path_length` defined twice | 626 and 675 | n/a | identical bodies; the second shadows the first (the old engine has the same at 555/604) |
| `OPTIMIZE_TOP_N_PATHS` | 77 | 0 | used only by `main()` |
| import-time banner `=== OPTIMIZED CIRCLE-BASED CENTERLINE EXTRACTION ===` | n/a | n/a | printed on import; noise in logs and tests |

Still used (not dead): `SHOW_BITMAP` and `SHOW_PRE_OPTIMIZATION_PATHS`, which
`centerline_web_app.py` lines 1238–1270 toggles as module globals around
`create_svg_output`. That is shared mutable state across concurrent requests, a
side finding and not dead code. `DARK_THRESHOLD` is also used, by `test_ui_backend.py`.
`USE_VECTORIZED_FAST_PATHS` and `VECTORIZED_FAST_PATHS_MIN_PIXELS` are used by tests.

### D4. Merge / overlap / optimise code reachable only from non-UI paths

* `remove_overlapping_paths`, `CircleEvaluationSystem`, `optimize_path_with_custom_params`:
  reached only from `POST /process` (no UI caller), from `background_optimization`
  (UI always sends `enable_optimization=false`), and from the `/test-ui` rig and tests.
* `optimize_path_with_circles`: imported by the app but **never called**. It is
  used only by `main()` in each engine.
* `merge_nearby_paths`: **not** non-UI-only. Auto-tune, auto-detect and tile preview use it (F8).
* Two code paths are live but dormant: the background-optimisation thread and its
  progressive-preview publishing. They are reachable only if a client sends
  `enable_optimization: true` to `/process_immediate`.

## Legacy fixtures and goldens: proposal

Based on the Stage 2 ground-truth audit (`fixture-ground-truth-audit.md`, 64 px):

* **Still test real behaviour (keep as lessons):**
  * at sw3, 10 of 17 fixtures have a valid `ideal_sample`:
    * `curve-quarter-arc`, `curve-half-arc`, `curve-true-circle`, `curve-s-cubic`;
    * `curve-gentle-convex`, `curve-tight-180`, `curve-perfect-circle`;
    * `line-horizontal`, `line-diagonal-45`, `corner-l-shape-r45`.
  * These already exist as `tests/lessons/fixture-*-64px-sw3` lessons.
* **Baggage as centerline ground truth:**
  * The 7 invalid at sw3, whose `inset_px` / offset curves sit 0.6–1.9 px (p95) off the stroke centre:
    * `curve-line-curve-line`, `curve-to-straight-to-curve`, `square-opposing-curves`;
    * `line-horizontal-r3`, `square-perfect`, `square-perfect-r3`, `corner-l-shape`.
  * They can still serve as **shape** tests for curve fitting. They do not test centreline placement.
* **Goldens (`tests/fixtures/golden/`):** dead. Their only consumer was never committed.
  Propose deleting them in Stage 8 unless `test_fit_curves.py` turns up.
* **Rig-only fixtures at sw2:** most curves are invalid at sw2 (≈0.5 px even-width bias, F11).
  Do not use sw2 for placement scoring.

## Python 3.11 (runtime.txt)

`runtime.txt` pins `python-3.11.13`. `py -0p` lists only 3.14, 3.12 and 3.10 on this
machine, so **3.11 results were not checked**. What was checked: every new file under
`tests/pipeline/`, `tests/light_port/` and `scripts/` parses with
`ast.parse(..., feature_version=(3, 11))`. That covers syntax only, not results.

Proposal (not added): a GitHub Actions workflow with a 3.11/3.12 matrix that runs the
same commands and uploads the comparison JSON so the two can be compared:

```yaml
# .github/workflows/lessons.yml  (PROPOSED, not committed)
name: lessons
on: [push, pull_request, workflow_dispatch]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix: { python: ["3.11", "3.12"] }
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "${{ matrix.python }}" }
      - run: pip install -r requirements.txt pytest
      - run: python -m pytest tests/pipeline tests/light_port tests/lessons -q
      - run: python -m tests.pipeline.run_lessons --json lessons-${{ matrix.python }}.json
      - run: python -m tests.light_port.compare_engines --skip-pytest --out out-${{ matrix.python }}
      - uses: actions/upload-artifact@v4
        with: { name: results-${{ matrix.python }}, path: "lessons-*.json\nout-*" }
```

Caveats:

* The legacy suite has 13 known failures, so it is left out of the gating step.
* `--mark-failing-as-baseline` is never run in CI.
* Linux and Windows may differ slightly in floating point. Compare 3.11 against 3.12 on the same OS.
* An alternative is to install Python 3.11 locally (`py install 3.11` with the new Python
  install manager, or the python.org installer), create a venv outside the repo and run
  the same commands.

## CAD-exported SVG

Pending: no CAD SVG has been supplied yet. When one is available:

```
python -m tests.pipeline.inspect_svg path\to\export.svg --bitmap tests\lessons\<id>\image.png
```

This prints the alignment mode, per-path point counts, extents and warnings, or the exact
rejection reason. Expected rejections today:

* any `transform=` attribute;
* `mm`/`in` units unless the width and height match the bitmap through `viewBox`;
* a `DOCTYPE` or `ENTITY` declaration.

Affine transforms and physical units must be supported before the public Teach feature.

> **Update (Stage 2b):** transforms, physical units and `<image>`-frame registration
> are now supported, and the owner's LightBurn export
> (`tests/pipeline/samples/lightburn-cross-t2-w8-a35.svg`) is accepted with no
> rejections. See `docs/migration/lesson-format.md` (SVG reader, LightBurn export rules).
