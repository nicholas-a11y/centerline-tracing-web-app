# Teach: design (Stage 4)

Status: **design approved (revision 2)**, with the owner's decisions applied. No code
has been written for Teach. Items marked **[unverified]** are assumptions that Stage 5
must check first. Decisions and everything still uncertain are collected in
[§19](#19-uncertainties-and-decisions-needed).

## 0. Summary

Teach is a public area where anyone can submit a bitmap and an SVG of the ideal
centerline. The server runs the **live** pipeline, scores the result against the
SVG, shows an overlay with pass/fail ("see it fail"), and puts the case into a
review queue. Accepted cases are exported into `tests/lessons/` and become permanent
tests.

Teach **never changes any user's results**. It does not touch `/process_immediate`
or any other existing route, session or parameter. It lives in a separate package
registered by one line, and it does nothing unless `TEACH_ENABLED` is true. The
default is off.

## 1. Goals and non-goals

Goals:
* Collect real failure cases with ground truth, scored the same way as the lesson
  suite (`docs/migration/lesson-format.md`).
* A visible queue, so people can see that their case was received and what happened
  to it.
* Make it easy for both people and AI agents to submit a correct case first time.
* Provide a one-step path from an accepted case to a committed lesson.

Non-goals:
* No accounts, no captchas (Turnstile is a documented fallback only, see §12.3), and
  no comments or discussion threads.
* No changes to engine behaviour, defaults, existing routes, the main UI or
  `templates/index.html`. Adding a link to Teach from the main UI would be a UI
  change and needs separate approval.
* No automatic commits from the server. The server never holds GitHub credentials.

## 2. Facts about the current app that shape this design

| fact | evidence | consequence |
|---|---|---|
| A single module-level `app = Flask(__name__)`, no blueprints, no app factory | `centerline_web_app.py` line 84 | Teach is a blueprint registered once, after all existing routes |
| A global 16 MB `MAX_CONTENT_LENGTH` | line 85 | Teach enforces its own, smaller caps per route; the global value is not changed |
| Live sessions use a global registry capped at `MAX_SESSIONS = 12` and evict the oldest first | lines 143–145, 382–417 | Teach must **not** create sessions in the live process, or it could evict real users' sessions. Scoring runs in a subprocess (§8). |
| Deployed on Railway: `gunicorn --workers 1 --threads 8 --timeout 300` | `railway.json`, `Procfile`, `IS_RAILWAY_DEPLOYMENT` at line 92 | One process, so in-memory rate limits and locks work. Scaling out breaks them (§19). |
| `app.yaml` (App Engine, `max_instances: 10`) is also present | `app.yaml` | It looks unused. If it were used, the single-process assumption would fail. **[unverified]** |
| No `ProxyFix`, no `SECRET_KEY`, no rate limiting | grep of `centerline_web_app.py` | Teach parses forwarded headers itself, inside the blueprint only. Admin does not use cookies (§15). |
| Railway's filesystem is ephemeral | Railway platform behaviour | Local-folder storage is for development only. In production, Teach refuses to start without R2 (§4). |
| No `LICENSE` file in the repo | `git ls-files` | Generated lessons say "same licence as this repository", but there is none. Submissions need their own explicit licence (§13). |
| `engine_commit()` calls `git` | `tests/pipeline/lesson.py` | Probably unavailable on Railway. Use `RAILWAY_GIT_COMMIT_SHA` instead (§6.3). **[unverified]** |
| SVG reader rejects transforms and mm/in units, and does not use `<image>` placement | `tests/pipeline/svg_reference.py` (Stage 2) | Stage 2b adds them. They are required **before Stage 5 starts** (§9.2) |

## 3. Architecture and isolation

### 3.1 Package layout (all new)

```
teach/
  __init__.py          # register_teach(app): reads env; returns immediately if disabled
  config.py            # env parsing and startup validation (§4)
  blueprint.py         # public routes (§14)
  admin.py             # admin routes (§15)
  storage.py           # Storage interface: LocalStorage, R2Storage (boto3, imported lazily)
  limits.py            # rate limits, storage quotas, admin lockout
  validate.py          # image/SVG/text validation -> structured errors (§9, §14.4)
  scoring_worker.py    # subprocess entry point: live pipeline + scoring (§8)
  similarity.py        # pHash, geometry and text similarity (§10)
  export_lesson.py     # submission -> lesson folder / zip; also a CLI (§16)
  notify.py            # ntfy alerts (§15.5)
  retention.py         # retention sweep (§13.3)
  sanitize.py          # public-title sanitiser (§11)
  templates/teach/*.html, static/teach/*   # blueprint-local; templates/ is not touched
  spec.md              # served at /teach/spec (§14.1)
lessonkit/             # shared, non-test code moved out of tests/pipeline (decided: option A)
  svg_reference.py, scoring.py, overlay.py, lesson.py, runner.py, trace_template.py
```

**Decided: option A.** Teach needs the SVG reader, scoring, overlay, live-runner and
trace-template code that lives in `tests/pipeline/` today. Those modules move to a new
top-level `lessonkit/` package, and one-line re-export shims stay in `tests/pipeline/`
so every existing command and test keeps working unchanged. Shipped code never imports
from the test tree.

The move is the **first commit of Stage 5, on its own**: the full suites
(`tests/pipeline`, `tests/light_port` and `tests/`) are run before and after it and
must give identical results. No Teach code is in that commit. These are all Stage 2
files written by us; no engine, app or legacy test file is involved.

### 3.2 Registration: the only change to existing app code

One line at the end of `centerline_web_app.py`, after every existing route and
before `if __name__ == '__main__':`:

```python
from teach import register_teach; register_teach(app)
```

`register_teach`:
* Returns immediately, registering nothing, unless `TEACH_ENABLED` is exactly `1`
  or `true` (case-insensitive).
* Imports nothing heavy at module level. boto3, numpy and PIL are imported lazily
  inside handlers and the worker.
* If any configuration check fails, it logs one line without secrets
  (`Teach disabled: <reason>`) and returns. **It never raises into app startup**, so
  a bad Teach configuration cannot take the main app down.
* Registers the blueprint under the URL prefix `/teach`, plus `/llms.txt` only if
  `TEACH_LLMS_TXT` is enabled (§14.1). No existing route uses either path; I checked
  all 27 `@app.route` rules.
* Does not add `before_request`/`after_request` hooks or error handlers at the app
  level. Teach's hooks are blueprint-scoped (`bp.before_request`), so they run only
  for Teach routes.
* Does not change `app.config`, `Image.MAX_IMAGE_PIXELS`, logging levels or any
  other global. Image-size limits are checked from the image header before decoding,
  without changing Pillow globals.

### 3.3 How isolation is proven

The Stage 5 tests (§18) include:
* With the flag off, the set of `app.url_map` rules equals the current 27 rules plus
  `static`, and `register_teach` registers nothing.
* With the flag on, every added rule starts with `/teach` (or is `/llms.txt`), and
  every existing rule is unchanged (same endpoint, methods and view function).
* Teach scoring never adds a key to `centerline_web_app.sessions`.
* The full existing suite and the lesson suite give the same results with the flag on
  and off.

### 3.4 Existing files that would be touched (Stage 5)

| file | change |
|---|---|
| `centerline_web_app.py` | **one line** (§3.2). No other edit. |
| `requirements.txt` | add `boto3` (needed only when R2 is configured; imported lazily) |
| `.env.example` | add the approved optional names: `TEACH_ALERT_WEBHOOK_URL`, `TEACH_LOCAL_DIR`, `TEACH_TRUSTED_PROXY_HOPS`, `TEACH_PUBLIC_BASE_URL`, `TEACH_LLMS_TXT` (§4). Names only, empty values. |
| `.gitignore` | one `!tests/teach/**/test*.py` exception, like the existing ones |
| `README.md` | short Teach section |
| `tests/pipeline/{svg_reference,scoring,overlay,lesson,runner,trace_template}.py` | re-export shims (option A, §3.1), in the first Stage 5 commit |

Not touched: `centerline_engine*.py`, `centerline_core.py`, `test_ui_backend.py`,
`templates/index.html` and other templates, `static/`, existing tests, fixtures and
goldens.

## 4. Configuration

Values are read **only** from environment variables, never from files in the repo.
Nothing reads `.env` automatically: no `python-dotenv` dependency is added, and local
developers export variables themselves. Secrets are never logged, echoed in errors or
included in responses.

| variable | required | meaning |
|---|---|---|
| `TEACH_ENABLED` | – | `1`/`true` enables Teach. Anything else, or unset, means off. |
| `TEACH_ADMIN_TOKEN` | when enabled | Admin bearer token. **Minimum length 32 characters**; shorter or missing means Teach is disabled (see below). |
| `R2_ENDPOINT` | prod | `https://<accountid>.r2.cloudflarestorage.com` |
| `R2_BUCKET` | prod | bucket name |
| `R2_ACCESS_KEY_ID` | prod | R2 API token key ID |
| `R2_SECRET_ACCESS_KEY` | prod | R2 API token secret |
| `TEACH_LOCAL_DIR` | – (new, optional) | Local-mode folder. Default: `~/.centerline-teach` (outside the repo). |
| `TEACH_ALERT_WEBHOOK_URL` | – (new, optional) | **Secret.** ntfy topic URL for admin alerts (§15.5). Handled exactly like `R2_SECRET_ACCESS_KEY`: set only in the deployment's variables, never committed, logged, echoed in errors or shown in the admin UI (the UI shows only "alerts: configured / not configured"). |
| `TEACH_TRUSTED_PROXY_HOPS` | – (new, optional) | Number of trusted proxies in front of the app. Default `1` on Railway, `0` locally. |
| `TEACH_PUBLIC_BASE_URL` | – (new, optional) | Absolute base URL for links in alerts and in the spec, e.g. `https://example.up.railway.app` |
| `TEACH_LLMS_TXT` | – (new, optional) | `1` to also serve `/llms.txt` |

The six names already in `.env.example` are used exactly as written. The five new
optional names are approved.

**Startup rules (fail closed):**
1. If `TEACH_ENABLED` is not true, register nothing.
2. If `TEACH_ADMIN_TOKEN` is missing or shorter than 32 characters, disable **all** of
   Teach (public and admin; decided). Nothing is registered, so `/teach` and every
   path under it return Flask's normal **404**, and `/llms.txt` is not served. One log
   line at WARNING level says why, naming the variable but never its value or length:
   * `Teach disabled: TEACH_ENABLED is set but TEACH_ADMIN_TOKEN is missing`
   * `Teach disabled: TEACH_ADMIN_TOKEN is shorter than 32 characters`

   The public side is not run without a working admin, because nothing could ever be
   reviewed or deleted. A token can be generated with
   `python -c "import secrets; print(secrets.token_urlsafe(32))"` (43 characters).
   Every other "disabled" reason below uses the same `Teach disabled: <reason>` form
   and also leaves `/teach` as a 404.
3. **Storage selection:**
   * If all four `R2_*` variables are set, use R2.
   * If **none** are set, use local-folder mode, but only when not on a deployment
     (`IS_RAILWAY_DEPLOYMENT` is false). On Railway with no R2 settings, disable
     Teach, because local files would be lost on every deploy.
   * If only **some** are set, disable Teach and log which names are missing (names
     only, never values). Teach never silently falls back to local mode.
4. On startup with R2, make one cheap `head_bucket` call. If it fails, disable Teach
   and log the error class only.

### 4.1 Local-development mode

When the R2 variables are absent and the app is not deployed, storage uses
`TEACH_LOCAL_DIR`, with the same layout as R2 (§7.1). This is how the owner runs Teach
locally against a folder:

```powershell
$env:TEACH_ENABLED = "1"
$env:TEACH_ADMIN_TOKEN = (python -c "import secrets; print(secrets.token_urlsafe(32))")
.venv\Scripts\python.exe centerline_web_app.py     # http://localhost:5002/teach
```

In local mode, HTTPS-only admin is relaxed for `localhost`/`127.0.0.1` only (§15.3),
and the page shows a "LOCAL MODE" banner.

## 5. User flow

1. **Teach page** (`/teach`): explains what makes a good case, shows the rules
   (§14.6), and links to the queue, the spec and the worked example. It has a search
   box over the public queue ("search the queue first").
2. **Form fields:**

   | field | required | notes |
   |---|---|---|
   | bitmap | yes | PNG or JPEG (§9.1) |
   | reference SVG | yes | registered to the bitmap by an embedded `<image>`, a size match or a border rect (§9.2) |
   | title | yes | ≤ 120 characters (a sanitised ≤ 80-character version is public straight away, §11) |
   | what is wrong | yes | ≤ 2000 characters |
   | what is expected | yes | ≤ 2000 characters |
   | notes | no | ≤ 4000 characters |
   | focus box | no | x0, y0, x1, y1 around the problem; used as the initial scoring region |
   | settings used | no | `dark_threshold`, `enable_pruning`, `normalization_mode`, `normalization_sensitivity`, `min_path_length`. Same whitelist and ranges as the UI; UI defaults otherwise. |
   | public thumbnail | no (off) | allow the overlay thumbnail to be public once reviewed |
   | share full bitmap | no (off) | allow the full bitmap to be public once reviewed |
   | consent and licence | yes | checkbox with the text in §13 |

   **Scoring tolerance is never a field.** The submitter cannot set it.
3. **Download trace template** (optional, stateless): as soon as a bitmap has been
   chosen (or after a validate call), the page offers **"Download trace template"**.
   The browser posts only the bitmap to `POST /teach/api/trace-template`. The server
   validates it with the same bitmap rules (§9.1) and returns a zip generated in
   memory:
   * `<name>-faded.png`: the bitmap faded towards white (default fade 0.35), at its
     exact pixel size, with 96 dpi metadata;
   * `<name>-template.svg`: `width`/`height` in px equal to the bitmap,
     `viewBox="0 0 W H"`, the faded PNG embedded at `0,0 W×H`, and no other shapes;
   * `<name>-README.txt`: draw one open path per stroke along the centre; keep the
     image in the export (do not hide or delete it); export SVG.

   **Nothing is stored**, no ID is issued and nothing is logged except the request
   count. It counts against the validate rate limit (§12.2), does not use the scoring
   queue (it is a resize-free fade and re-encode only), and the request body cap is
   the bitmap cap. It is the same code as the Stage 2c CLI
   (`python -m tests.pipeline.trace_template`, moved to `lessonkit` in Stage 5).
4. **Check** (optional, no storage): the browser calls `/teach/api/validate`. This
   returns precise, fixable errors, or the pass/fail result with an overlay preview.
   It also calls `/teach/api/similar`, which may say "looks similar to #N" (§10).
   The validate result repeats the "Download trace template" link, so a user whose
   SVG fails registration can start again from a template that is known to work.
5. **Submit:** the server validates again (it never trusts the client), stores the
   files, runs the scoring worker, and stores the result and overlay. The status moves
   to `pending_review`. The response shows the overlay with pass/fail, the submission
   number, the queue URL and a **manage link** (§13.3).
6. **Queue:** the case appears in the public queue under the visibility rules in §11.

If scoring times out or errors, the submission is still stored with outcome `timeout`
or `error`. This is useful information for the owner.

## 6. Data model

### 6.1 Statuses

```
submitted ──(scored)──▶ pending_review ──▶ accepted
                             │   ▲      └─▶ rejected
                             ▼   │
                          needs_info
(any) ──(submitter delete / admin delete)──▶ deleted (tombstone: number + "deleted" only)
```

| status | meaning | who sets it |
|---|---|---|
| `submitted` | stored; scoring in progress | server |
| `pending_review` | scored; waiting for the owner | server, or submitter answering `needs_info` |
| `needs_info` | the owner asked a question (admin note is public on the item) | admin |
| `accepted` | will become, or has become, a lesson | admin |
| `rejected` | not useful, out of scope or a duplicate (reason shown) | admin |
| `deleted` | content removed; number kept so links don't break | submitter or admin |

A submitter answers `needs_info` through the manage link (add a note or a replacement
SVG). This moves the case back to `pending_review` and keeps the history.

### 6.2 `meta.json` (one per submission; comments are illustrative only)

```jsonc
{
  "schema_version": 1,
  "id": "t_6f3k9q2m8w",          // random, unguessable; used in storage keys
  "number": 42,                   // human "#42", sequential (§7.3)
  "created_at": "2026-11-09T01:02:03Z",
  "status": "pending_review",
  "status_history": [{"status": "submitted", "at": "...", "by": "server"}],
  "reviewed": false,              // true once the admin has set any status (§11)
  "title": "...", "what_is_wrong": "...", "what_is_expected": "...", "notes": "...",
  "title_public": "...",          // sanitised title (§11), computed at submit time
  "title_hidden": false,          // admin one-click hide (§11)
  "admin_note_public": "", "admin_note_private": "",
  "visibility": {"thumbnail_opt_in": false, "bitmap_opt_in": false,
                 "thumbnail_public": false, "bitmap_public": false},
  "consent": {"licence": "CC0-1.0", "text_version": 1, "accepted_at": "..."},
  "image": {"width": 256, "height": 256, "sha256": "...", "phash": "c3a1..."},
  "reference": {"paths": 2, "points": 812, "sha256_geometry": "...",
                "registration": "image_frame"},   // image_frame | size_match | border_rect (§9.2)
  "params": {"dark_threshold": 0.2},
  "scoring": {"region": null, "tolerance_px": 1.0, "set_by": "default"},
  "results": [{"engine_commit": "de89229", "at": "...", "outcome": "fail",
               "variant": "prune_off", "summary": "...", "metrics": {},
               "elapsed_ms": 812, "timed_out": false}],
  "me_too": [{"at": "...", "note": "...", "variant_key": null}],
  "manage_token_sha256": "...",
  "lesson": {"lesson_id": null, "exported_at": null, "merged_commit": null,
             "private_copies_deleted_at": null},   // admin action (§13.3)
  "retention": {"status_since": "2026-11-09T01:02:03Z"},  // drives the sweep (§13.3)
  "quota_bytes": 183422
}
```

No IP address, user agent or contact detail is stored. The manage token is stored only
as a SHA-256 hash.

### 6.3 Engine commit and "failing since / now passing"

Every entry in `results[]` records the engine commit:
* `RAILWAY_GIT_COMMIT_SHA` (first 7 characters) if set **[unverified: confirm Railway
  sets it for GitHub deploys]**;
* otherwise `git rev-parse --short HEAD`;
* otherwise `"unknown"`.

The detail page derives:
* **failing since:** the first commit in the latest unbroken run of failures;
* **now passing (since X):** when the latest result passes but an earlier one failed.

Re-scoring is triggered by the admin ("re-score all open items at the current
commit"). It runs sequentially in the background using the same worker and limits.
There is no automatic re-score on deploy, to avoid a CPU spike after every release.

## 7. Storage

### 7.1 Layout (identical for R2 and local folder)

```
teach/
  counter.json                         # next submission number (§7.3)
  sweep/last.json                      # last retention sweep (§13.3)
  index/public.json                    # regenerated public index (§7.2)
  index/admin.json                     # admin index (includes private fields)
  submissions/<id>/meta.json
  submissions/<id>/image.png           # re-encoded, metadata stripped (§9.1); private by default
  submissions/<id>/reference.json      # parsed geometry (what scoring uses)
  submissions/<id>/reference.raw.svg   # original upload; admin-only, never served inline
  submissions/<id>/overlay.png         # server-rendered
  submissions/<id>/thumb.png           # ≤ 256 px overlay crop; public only per §11
  submissions/<id>/variants/<k>/...    # optional "me too" variants (same rules)
  quota/<yyyy-mm-dd>.json              # global daily counters (§12)
```

R2 is accessed with boto3 (`endpoint_url=R2_ENDPOINT`, `region_name="auto"`, signature
v4). Objects are **never** public, and there is no public bucket and no r2.dev URL.
Every public byte is served by Flask after a visibility check.

### 7.2 Index regeneration

On every status, visibility or submission change, the server rewrites
`index/public.json` and `index/admin.json` under an in-process lock. The update is
incremental: it loads the current index, replaces one entry and writes it back. A full
rebuild, which lists all `meta.json` files, is an admin action. The public index
contains only fields allowed by §11.

### 7.3 Submission numbers

`#N` comes from `counter.json`, read and incremented under the in-process lock (one
gunicorn worker), with an R2 conditional write (`If-Match` on the ETag) as a second
guard **[unverified: confirm R2 conditional PUT support through boto3]**. If the
conditional write fails, the server retries once and then returns 503. The random `id`
is the storage key. The number is only for display.

### 7.4 When to move metadata to D1

Stay on one JSON file per submission plus the index until **any** of these is true:
* more than **1,000** submissions;
* `index/admin.json` is larger than **1 MB**, or an index update takes more than
  **500 ms** (p95);
* the app runs more than one worker or instance, so writers can no longer be
  serialised in-process.

At that point, move metadata, counters, quotas and rate limits to Cloudflare D1, and
keep bitmaps and overlays in R2. From Flask, D1 is reachable only through the
Cloudflare HTTP API, which needs an extra API token **[decide later]**. The pipeline
stays in Flask either way. Python Workers are not assumed to be able to run
scikit-image.

## 8. Pipeline and scoring worker

* **Same code path as the lesson suite:** the worker calls the shared live runner
  (`run_live`), which drives `/upload` → `/process_immediate` with the UI's default
  parameters (or the whitelisted ones the submitter gave). Results match
  `tests/pipeline/run_lessons.py` exactly.
* **Separate subprocess**
  (`python -m teach.scoring_worker <job.json>`, the same pattern as the Stage 2
  `/process` worker):
  * Live sessions are untouched (§2). The worker has its own copy of `sessions`.
  * The worker's environment sets `TEACH_ENABLED=0`, so the imported app does not
    register Teach inside the worker.
  * There is a hard wall-clock timeout of **30 s**, after which the worker is killed.
  * On Linux, memory is capped with `RLIMIT_AS` = **1.5 GB**, and `nice +10` keeps
    live users' requests ahead of scoring **[unverified: Railway plan memory; tune]**.
* **Concurrency:** one scoring job at a time, with a waiting queue of at most **3**.
  Beyond that, return `503 busy` with `Retry-After: 30`. Validate and submit share this
  queue.
* **Variants:** `prune_off` (the primary variant, as in the UI) and `prune_on` are both
  scored. `/process` is not run for Teach.
* **Scoring:**
  * Uses `lessonkit.scoring.score_paths` with the submission's `scoring` block.
  * The tolerance defaults to **1.0 px** and only the admin can change it.
  * The region is the submitter's focus box if one was given, otherwise the whole
    image. The admin can replace it.
* **Overlay:** rendered server-side to PNG by `lessonkit.overlay`, with the
  reference, engine paths, junctions and endpoints. The user's SVG is never rendered
  by a browser.

## 9. Validation and SVG safety

### 9.1 Bitmap

* **Upload limits:**
  * at most **4 MB**; magic bytes must say PNG or JPEG;
  * Pillow opens it lazily, and the size is read **from the header before decoding**;
  * dimensions from **32×32 to 2048×2048**, and at most **4 megapixels**;
  * single frame only; animated or multi-page files are rejected.
* Converted to 8-bit greyscale, re-encoded as PNG, and stripped of all metadata
  (EXIF, ICC, text chunks). Only the re-encoded file is stored.
* Warnings (not errors): no dark pixels at the chosen threshold; more than 60 % dark
  (probably inverted); larger than 1024 px ("crop to the problem area").

### 9.2 Reference SVG

* Parsed only by the safe reader (`lessonkit.svg_reference`):
  * DOCTYPE/ENTITY are refused before parsing; `<use>` is rejected;
  * at most **512 KB** for Teach (the reader's general cap is 2 MB). The cap counts
    the **whole file, including any embedded base64 image**;
  * at most 200 shape elements and 200k sampled points.
* Only geometry is extracted. Styles, scripts, fonts, text and other non-geometry
  elements are ignored. The parsed geometry is stored as JSON (`reference.json`) and
  drives everything.
* **Transforms and units (required before Stage 5 starts, not just before public
  launch):** affine `transform`s (`matrix`, `translate`, `scale`, `rotate`, `skewX`,
  `skewY`) on paths and nested `<g>`, composed in document order; physical units
  (`mm`, `cm`, `in`, `pt`, `pc`, `px`, at 96 px/in) combined with `viewBox`. These
  land in Stage 2b (`tests/pipeline/svg_reference.py`) and move to `lessonkit` with
  the rest of the shared code.
* **Registration (how SVG coordinates are mapped onto bitmap pixels), in order:**
  1. **`<image>` placement frame (preferred; this is what LightBurn exports).** If the
     SVG contains exactly one `<image>`, its `x`/`y`/`width`/`height` rectangle is
     pushed through all its transforms and the viewBox. The resulting rectangle, as
     it appears on screen, is mapped onto the bitmap's full pixel size
     (`0..W × 0..H`) in the bitmap's own upright orientation: the rendered top-left
     corner is pixel edge (0, 0). The SVG's `width`/`height` units and its viewBox
     size are **not** relied on. A flipped placement (LightBurn writes
     `matrix(1,0,0,-1,…)` and stores the PNG upside down so it renders upright) is
     handled, because only the rendered rectangle matters.
     * The embedded pixel data is **never decoded** for scoring or registration.
     * **Stored-size check (header only):** the image's stored width/height is read
       from the PNG `IHDR` or JPEG `SOF` header (at most the first 256 KB of the
       base64 payload is decoded to find it; pixels are never decoded), with each
       side capped at 16384 px (`svg_image_dimensions`). An aspect ratio more than
       1 % off the bitmap's is an error (`svg_bitmap_mismatch`: traced over a
       different picture). The same aspect ratio with a different pixel size is a
       **warning** shown to the submitter, because the frame alone fits any bitmap
       of the same shape. Linked, non-base64 or non-PNG/JPEG images give a "could
       not check" warning.
     * Errors: `svg_image_rotated` (the rectangle is not axis-aligned after
       transforms), `svg_image_aspect` (its aspect ratio differs from the bitmap's by
       more than 1 %), `svg_multiple_images`, `svg_bitmap_mismatch`,
       `svg_image_dimensions`.
  2. **Border rectangle:** a `rect` with `id`, `class` or `data-role` = `border`
     spanning the image.
  3. **Size match:** `width`/`height`/`viewBox` equal to the bitmap's pixel size.
  4. Otherwise `svg_no_registration`, with a hint to keep the image in the export or
     to start from the trace template (§5 step 3).
* **Pixel convention:** SVG coordinates are pixel-edge based; the engine and scoring
  use pixel centres, so the reader subtracts 0.5 px after registration.
* **LightBurn exports** (the owner's tool) must keep the image in the export, with
  the traced paths ungrouped; document units mm or px are both fine. The rules are in
  `docs/migration/lesson-format.md` and are repeated in the spec (§14.1).
* The raw upload is stored privately for the admin's audit and is never served
  inline. If the admin downloads it, it is sent as `Content-Type: text/plain` and
  `Content-Disposition: attachment`.

### 9.3 Text

UTF-8 only; control characters are stripped and the lengths in §5 are enforced. Text
is stored as-is and **always rendered escaped** (Jinja autoescape, no `|safe`, no
Markdown rendering). Links are not made clickable.

## 10. Duplicate prevention

`/teach/api/similar` runs before submission, and again inside submit. It compares the
new case with every non-deleted submission:

| signal | method (no new dependencies) | threshold |
|---|---|---|
| exact | SHA-256 of the re-encoded PNG **and** of the canonical geometry | equal → **blocked**, offered "me too" |
| image | 64-bit DCT perceptual hash (numpy + scipy.fft) on a 32×32 downscale | Hamming ≤ **10** |
| geometry | when sizes are equal: symmetric mean chamfer distance between reference paths | ≤ **2 px** |
| text | token Jaccard on title + what is wrong (lower-cased, stop words removed) | ≥ **0.6** |

A match is shown if:
* the image signal matches; or
* the geometry matches and text Jaccard is at least 0.3; or
* text Jaccard is at least 0.8.

For each match the response says "looks similar to #N" and links to it:
* If #N is already reviewed, its public title and status are shown.
* If #N is still pending, it shows only what the public queue shows (§11): the
  number, the **sanitised title** (or "title hidden" if the admin hid it) and
  "pending review, details hidden". Descriptions, thumbnails and bitmaps never leak
  through the similarity check.
* The text signal compares against the full stored text on the server, but the
  response never returns any text beyond the public sanitised title.

The user can then:
* **"Me too":** add a note and optionally a variant bitmap and SVG (same limits,
  counted against their quota) to #N. #N's public "me too" count increases.
* **"Not a duplicate":** submit anyway, with a required one-line reason, which is
  shown to the admin.

## 11. Public queue and visibility

The public queue (`/teach/queue`, JSON at `/teach/api/queue`) lists every
non-deleted submission. What it shows depends on review state:

| item state | number, status, date, me-too count | title | descriptions, public admin note | thumbnail | full bitmap |
|---|---|---|---|---|---|
| not yet reviewed (`reviewed: false`) | yes | **sanitised title**, unless the admin hid it | **no** ("awaiting review") | no | no |
| reviewed (the admin has set any status) | yes | sanitised title, unless hidden | yes (escaped text) | only if the submitter opted in **and** the admin left it public | only if the submitter opted in **and** the status is `accepted` (or the admin made it public) |

**Title sanitiser** (`teach/sanitize.py`, decided). The public title is derived from
the submitted title once, at submit time, and stored as `title_public`:
1. Unicode NFKC normalisation.
2. Remove control and format characters (Unicode categories `Cc`, `Cf`, `Cs`, `Co`,
   `Cn`), which includes bidirectional overrides and zero-width characters.
3. Remove URLs and URL-like tokens: anything with a scheme (`\w+://…`), `www.…`,
   e-mail addresses, and bare `host.tld` / `host.tld/path` tokens whose last label
   is alphabetic and 2–24 letters long (so `v1.2` and `0.5px` survive). Each removed
   token is replaced by `[link removed]`.
4. Collapse whitespace and trim.
5. Truncate to **80 characters** (on a grapheme boundary), adding `…` if shortened.
6. If nothing is left, use `Untitled`.

The admin sees the original title as well. **Hide title** is one click in the admin
queue list and detail view: it sets `title_hidden: true`, the public index shows
"title hidden" instead, and it is undone with the same button. Hiding does not change
the item's status or `reviewed` flag.

The **public index builder** (§7.2) writes, for every non-deleted item: number,
status, dates, me-too count, `title_public` (or `null` with `"title_hidden": true`),
and only after review the description fields, public admin note and thumbnail URL.
The same builder output is used by the queue page, `/teach/api/queue`,
`/teach/api/submissions/<number>`, the duplicate check (§10) and ntfy alerts (§15.5),
so they cannot disagree.

People can see straight away that their case is in the queue and what it is about,
while descriptions, thumbnails and bitmaps stay private until the owner has looked at
them. The spam risk of public titles is limited by the sanitiser (no links survive),
the per-IP limits (§12.2) and one-click hiding.

Detail pages for unreviewed items, and the queue page, send `X-Robots-Tag: noindex`,
so unreviewed titles are not indexed by search engines.

The submitter always sees their own full case through the manage link.

## 12. Abuse limits (no captchas)

### 12.1 Request and size limits

| limit | value |
|---|---|
| bitmap | ≤ 4 MB, 32–2048 px per side, ≤ 4 MP |
| SVG | ≤ 512 KB, ≤ 200 elements, ≤ 200k points |
| text | title 120, what is wrong / what is expected 2000 each, notes 4000 characters |
| request body (Teach routes) | ≤ 5 MB, checked from `Content-Length` and while streaming |
| scoring | 30 s wall clock, 1 concurrent, queue ≤ 3, `RLIMIT_AS` 1.5 GB |

### 12.2 Per-IP and global limits

All limits are kept in memory, keyed by client IP (§15.4 explains how the IP is
derived), and reset on restart. The daily counters are also persisted in
`quota/<date>.json`, so a restart does not reset the daily caps.

| action | per IP | global |
|---|---|---|
| `/teach/api/validate` (and `/teach/api/trace-template`, which shares this counter) | **60 / hour** | **600 / hour** |
| `/teach/api/similar` | 60 / hour | – |
| submissions + me-too | 5 / hour, 20 / day | **200 / day** |
| stored bytes | **20 MB / day** | **500 MB / day** |
| manage-link actions | 30 / hour | – |
| admin auth failures | see §15.4 | see §15.4 |

When a global cap is reached, Teach submissions close for the rest of the UTC day
with a clear message, and the admin is alerted. Validation keeps working.

The validate limits are deliberately generous so agents can iterate on a submission.
CPU is still bounded by the scoring queue (§8: one job at a time, queue ≤ 3, then
`503 busy` with `Retry-After`), not by the rate limit: 600 validations an hour is one
every 6 s, which a single worker at roughly 2–5 s per job can sustain.

### 12.3 Fallback (documented only)

If abuse gets past these limits, add Cloudflare Turnstile to submit and me-too only.
This needs a site key and a secret (two new environment variables) and a server-side
Siteverify call. It is not built now.

## 13. Privacy, consent, licence and deletion

### 13.1 Consent text (checkbox, required)

> I made this image and drawing, or have the right to share them. I have cropped
> them to the problem area and they contain no personal or confidential information.
> I license them under **CC0 1.0** (public domain dedication) so they can be used as
> test data for this project. If the case is accepted, it may be published in the
> project's public repository. After that it can be removed from future versions but
> not from the repository's history. Until then I can delete it at any time with my
> manage link.

**Decided: CC0 1.0.** The repository has no LICENSE file, and attribution would need
a place to live. The licence ID `CC0-1.0` is stored in `meta.json` and copied into the
exported `lesson.json`.

### 13.2 Advice shown on the form

"Crop to the problem area, with some context around it (see the padding check). Remove
title blocks, names, addresses and logos. Never upload confidential drawings."

### 13.3 Manage link and deletion

* Every submission gets a manage link: `/teach/manage/<id>#k=<token>`, where the
  token comes from `secrets.token_urlsafe(32)`.
  * The token is in the URL **fragment**, so it is never sent in requests, proxy logs
    or Referer headers.
  * The page's JavaScript sends it in an `X-Teach-Manage-Token` header.
  * The server stores only its SHA-256 hash and compares it with
    `secrets.compare_digest`.
* The link is shown once after submitting, with a "copy" button and a warning that it
  cannot be recovered.
* **Delete** removes the image, SVG, overlay, thumbnail, variants and text, and leaves
  a tombstone (number and "deleted").
* **Retention rules:**
  * `rejected` items are deleted (tombstoned) **30 days** after they became rejected.
  * `needs_info` items with no submitter response are deleted **60 days** after the
    question was asked.
  * `accepted` items keep their files. After the lesson is merged, deleting the
    private copies is an **admin action** ("Delete private copies" on the item,
    §15.1), never automatic. Afterwards only the metadata and the lesson link are
    kept, and `lesson.private_copies_deleted_at` is set.
* **Retention mechanism (no scheduler):** a sweep (`teach/retention.py`) runs
  * once at startup, in a background thread so startup is never delayed, and
  * on every admin summary request (`GET /teach/api/admin/summary`, which the admin
    page calls on load), after the response data has been gathered.

  Each run:
  1. Takes a **non-blocking in-process lock**. If another sweep holds it, it returns
     at once.
  2. Reads `sweep/last.json` from storage. If the last sweep finished less than
     **6 hours** ago, it releases the lock and returns. This keeps admin page loads
     and restarts cheap.
  3. Finds candidates from `index/admin.json` (status plus `retention.status_since`),
     without listing every object in the bucket.
  4. Deletes at most **200 items** per run; the rest are picked up by the next run.
  5. Writes `sweep/last.json` with the finish time and counts, then releases the lock.

  The sweep logs only counts ("retention: 3 rejected and 1 needs_info item
  tombstoned"). The admin summary shows the last-swept time and the number of items
  due. Because there is no scheduler, nothing is deleted while the app is idle; the
  next startup or admin visit catches up. If the app ever runs more than one process,
  the lock moves to a storage lease with the D1 migration (§7.4).

### 13.4 Logging

Never log images, SVGs, text fields, tokens or full IPs. Rate-limit logs record only a
truncated hash of the IP.

## 14. Agent-friendly submission

### 14.1 Spec at a stable URL

* `GET /teach/spec` serves `text/markdown; charset=utf-8`. The same content is rendered
  on the Teach page, from one source file, `teach/spec.md`. It contains the rules,
  field list, limits, SVG requirements (including the LightBurn export rules, §9.2),
  the worked example and the error codes.
* The spec has a **"Trace template"** section for agents: when you have a bitmap but
  no SVG yet, `POST` the bitmap alone to `/teach/api/trace-template`; the zip contains
  a faded PNG and a template SVG already registered to the bitmap
  (`viewBox="0 0 W H"`, image at `0,0 W×H`). Add one open `<path>` per stroke, in the
  template's pixel coordinates (pixel edges, so the centre of pixel column 10 is
  x = 10.5), keep the `<image>`, and submit that file as `reference`. This is the
  most reliable way for an agent to produce a correctly registered SVG.
* Optionally, `GET /llms.txt` (only if `TEACH_LLMS_TXT=1`) returns a short pointer:
  the app name, one line on what Teach is, and links to `/teach/spec` and
  `/teach/openapi.json`.

### 14.2 Machine-readable schema

* `GET /teach/openapi.json`: OpenAPI 3.1 for every public Teach endpoint.
* `GET /teach/schema/submission.json`: JSON Schema for the form fields, using the same
  limits as the validator. Both are generated from one Python definition, so they
  cannot drift.

### 14.3 Endpoints

| method and path | purpose |
|---|---|
| `GET /teach` | Teach page and form |
| `GET /teach/spec`, `/teach/openapi.json`, `/teach/schema/submission.json` | docs |
| `GET /teach/example/{image.png,reference.svg,request.json}` | worked example files (§14.5) |
| `POST /teach/api/validate` | **dry run**: validate and score; stores nothing |
| `POST /teach/api/trace-template` | bitmap only → zip of faded PNG, template SVG and README (§5 step 3); stateless, stores nothing |
| `POST /teach/api/similar` | duplicate check only |
| `POST /teach/api/submissions` | submit |
| `POST /teach/api/submissions/<number>/me-too` | add to an existing item |
| `GET /teach/api/queue`, `GET /teach/api/submissions/<number>` | public data (§11) |
| `GET /teach/queue`, `GET /teach/q/<number>`, `GET /teach/q/<number>/thumb.png` | public pages |
| `GET /teach/manage/<id>` + `PATCH`/`DELETE /teach/api/manage/<id>` | submitter's own item (manage-token header) |
| `/teach/admin...` | admin (§15) |

Requests are `multipart/form-data`, with the files `image` and `reference` and a
`meta` field containing JSON. Responses are always JSON with a stable shape.

### 14.4 Validation response (the same shape for validate and submit)

```json
{
  "ok": false,
  "errors": [
    {"field": "reference", "code": "svg_no_registration",
     "message": "Cannot register the SVG to the 256x256 bitmap: it has no <image>, no border rect, and its size is 67.7x67.7 mm.",
     "hint": "Keep the traced image in the export, or start from /teach/api/trace-template."}
  ],
  "warnings": [{"code": "image_large", "message": "...", "hint": "Crop to the problem area."}],
  "result": null
}
```

On success, `result` contains the outcome, per-line coverage and p95, path, junction
and endpoint counts, the engine commit, the registration method used (§9.2), a
`trace_template_url` hint, and an `overlay_png` (a data URL for validate, a URL for
submit).

Error codes include:
* bitmap: `image_too_large`, `image_dimensions`, `image_format`, `image_multiframe`;
* SVG: `svg_too_large`, `svg_doctype`, `svg_parse`, `svg_use_unsupported`,
  `svg_transform_invalid`, `svg_units_unsupported`, `svg_no_registration`,
  `svg_multiple_images`, `svg_image_rotated`, `svg_image_aspect`, `svg_bitmap_mismatch`,
  `svg_image_dimensions`, `svg_no_geometry`,
  `svg_too_many_elements`;
* fields: `field_missing`, `field_too_long`, `param_out_of_range`, `consent_required`;
* limits and duplicates: `duplicate_exact`, `rate_limited` (with `retry_after_s`),
  `quota_exceeded`, `busy`, `scoring_timeout`.

Submit returns `201` with:

```json
{"ok": true, "number": 42, "status": "pending_review",
 "queue_url": "https://…/teach/q/42", "manage_url": "https://…/teach/manage/t_6f3k9q2m8w#k=…",
 "result": {...}}
```

### 14.5 Worked example

There is a fixed example based on the `cross-t2-w8-a35` lesson: a 256 px PNG, a
two-path `reference.svg` and `request.json`. The spec shows the exact `curl` call:

```bash
curl -sS -X POST https://<host>/teach/api/validate \
  -F image=@image.png -F reference=@reference.svg \
  -F 'meta={"title":"Thin line kinks where it crosses a thick line","what_is_wrong":"...","what_is_expected":"...","consent":true}'
```

It also shows the expected response, including the failing result.

### 14.6 Rules stated plainly (spec and page)

1. One problem per submission.
2. Search the queue first. If your case is already there, use "me too".
3. The SVG must be registered to the bitmap. Best: keep the traced image in the export
   (LightBurn does this as long as the image is not hidden or deleted), or start from
   the trace template. Otherwise match the bitmap's pixel size or include a border
   rect. Draw one open path per stroke, along the ideal centre.
4. Crop to the problem area with some context. No confidential or personal content.
5. You cannot set the scoring tolerance; the maintainer does.
6. Run `/teach/api/validate` first and fix every error before submitting.

## 15. Admin

### 15.1 Pages

`/teach/admin` is a static shell page, so loading it reveals nothing. All data comes
from `/teach/api/admin/*` with `Authorization: Bearer <token>`. The shell keeps the
token in `sessionStorage`, which is cleared when the tab closes. Because there are no
cookies, there is no CSRF exposure, and the token is never put in a URL.

The admin can:
* view the queue, filtered by status, with the **count awaiting review**;
* open the detail view: overlay, the submitter's text, the raw SVG (download only),
  the results history and me-too entries;
* set the status and public and private notes;
* set the scoring **tolerance** and **region**, and re-score;
* toggle thumbnail and bitmap visibility (only within the submitter's opt-ins);
* **hide or unhide the public title** with one click, from the queue list or the
  detail view (§11);
* after a lesson is merged, **delete private copies** with one click (§13.3);
* export as a lesson (§16), delete, rebuild the index, and see quota usage and the
  last retention sweep.

### 15.2 Token check

* The token is compared with
  `secrets.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))`.
  Bytes are used because `compare_digest` raises `TypeError` on non-ASCII strings.
* A missing or malformed header is treated as a failure, with the same timing path.
* The 32-character minimum length is enforced at startup (§4).

### 15.3 HTTPS only

Every admin route (page and API) is refused with `403` unless **one** of these holds:
* the request is HTTPS: `request.scheme == "https"`, or `X-Forwarded-Proto: https`
  from a trusted proxy hop (§15.4). Railway terminates TLS and sets this header
  **[unverified: confirm on the deployment]**;
* local mode, with the host `localhost`/`127.0.0.1` and the remote address loopback.

Teach responses also send `Strict-Transport-Security: max-age=31536000` over HTTPS,
plus `Cache-Control: no-store` on admin responses.

### 15.4 Client IP and admin attempt limits

* **Client IP:** with `TEACH_TRUSTED_PROXY_HOPS = n`, the client IP is the n-th entry
  from the **right** of `X-Forwarded-For`. With n = 0, it is `request.remote_addr`.
  The left-most entries are client-controlled and never trusted. This is done inside
  the blueprint only; no global `ProxyFix`. The admin page has a "what IP do you see
  for me" diagnostic, to confirm the hop count after deploy **[unverified: Railway
  hop count]**.
* **Per IP:** 5 failed attempts in 15 minutes locks that IP out of admin for **1 hour**
  (`429`). Successful requests do not reset the window.
* **Global:** 30 failed attempts in 1 hour locks admin for everyone for 1 hour and
  sends an alert. A locked request is refused **before** the token is compared.
* Failures are logged with a truncated IP hash and a reason code only.

### 15.5 Notifications

* **Minimum, always on:** the "awaiting review" count on the admin page, and in the
  admin API (`GET /teach/api/admin/summary`).
* **Decided: ntfy** (`TEACH_ALERT_WEBHOOK_URL` = the full topic URL, for example
  `https://ntfy.sh/<unguessable-topic>` or a self-hosted ntfy server). It is sent with
  stdlib `urllib`, so there is no new dependency. Email and a daily summary are not
  built; the summary endpoint can be added later.
* **Why the URL is a secret:** ntfy has no sign-up, so on ntfy.sh the topic name
  works as the password: anyone who knows it can read and publish. Generate the topic
  with `python -c "import secrets; print('teach-' + secrets.token_urlsafe(18))"`. If
  a protected topic or self-hosted server with an access token is used, ntfy's
  documented `auth` query parameter keeps the token inside the same URL, so no extra
  variable is needed **[unverified: check in Stage 5]**.
* **Debounce:** at most **one message per 30 minutes**. Events in between are
  collected in memory and the next message summarises them ("+N more" beyond 10
  items). Alerts about a global cap being hit or an admin lockout bypass the
  debounce, but are limited to one per hour per kind.
* **Message content (title, status and link only):**
  * Request: `POST <TEACH_ALERT_WEBHOOK_URL>`, body `text/plain; charset=utf-8`.
  * Headers (fixed ASCII, no user text): `Title: Teach: N updates`,
    `Tags: inbox_tray`, `Priority: default`, and `Click:` the admin page URL built
    from `TEACH_PUBLIC_BASE_URL` (no token in it).
  * Body: one line per item, `#42 pending_review: <public title> <base>/teach/admin#q=42`,
    using the same sanitised title as the public queue (§11), or "title hidden".
  * Total body at most 3,000 bytes, well under ntfy.sh's 4,096-byte message limit, so
    it is never turned into an attachment.
  * It never contains the drawing, thumbnail, overlay, descriptions, notes, IPs or
    manage links.
* **Sending:** a background thread with a 5 s timeout and one retry. Failures are
  logged as `Teach alert failed: <error class>`, never with the URL. If the variable
  is unset, alerts are off and the admin page says "alerts: not configured".

## 16. Export to `tests/lessons/` and commit workflow

### 16.1 What export produces

The lesson ID is `teach-<number>-<slug>`, where the slug comes from the title:
lower-case ASCII, at most 40 characters. It is a standard lesson folder
(`docs/migration/lesson-format.md`):

* `image.png`: the stored re-encoded bitmap.
* `reference.svg`: **re-serialised from the parsed geometry**
  (`write_reference_svg`), not the user's raw file. It is therefore normalised, safe
  and readable by the lesson suite.
* `lesson.json`:

  | field | value |
  |---|---|
  | `category` | `"user"` |
  | `title` | the submission title |
  | `description` | what is wrong |
  | `expected_behaviour` | what is expected |
  | `notes` | the submitter's notes + me-too summary + public admin note |
  | `source` | `{"kind": "teach", "submission": 42, "queue_url": "..."}` |
  | `licence` | e.g. `"CC0-1.0 (submitted via Teach)"` |
  | `consent` | `{"public": true, "by": "submitter via Teach form, consent text v1, <date>"}`. No personal data. |
  | `params` | the submission's pinned params |
  | `scoring` | the **admin's** region and tolerance |
  | `status` | `baseline` if the latest result fails, `active` if it passes. The admin can override this in the export dialog. |
  | `last_run` | the latest result's commit, date and outcome |

* Before writing, export runs `validate_meta` on the result and refuses to export if it
  fails.

### 16.2 Two ways to get the lesson out

1. **Admin button "Download lesson (.zip)":** a zip containing
   `tests/lessons/<lesson-id>/...`. Unzip it at the repo root.
2. **CLI (local, recommended):**
   `python -m teach.export_lesson <number> --dest tests/lessons`. It reads from R2,
   using the `R2_*` variables from your shell environment, or from `TEACH_LOCAL_DIR`.

The server never writes to the repo and never commits.

### 16.3 Exact commit workflow after an export

```powershell
git switch main; git pull --ff-only
git switch -c lessons/teach-42
.venv\Scripts\python.exe -m teach.export_lesson 42 --dest tests/lessons      # or unzip the download
.venv\Scripts\python.exe -m tests.pipeline.run_lessons --filter teach-42 --html   # check the overlay
.venv\Scripts\python.exe -m pytest tests/pipeline -q                              # suite stays green (baseline = xfail)
git add -- tests/lessons/teach-42-thin-line-kinks
git diff --cached --stat                                                          # only the 3 lesson files
git commit -m "Add lesson teach-42: Thin line kinks where it crosses a thick line" -m "Source: <queue URL>. Licence: CC0-1.0."
git push -u origin lessons/teach-42
# open a PR; CI (Appendix A) runs the suites; merge
```

Do **not** run `--mark-failing-as-baseline` or `--record` as part of this. The status
is set at export.

After the merge deploys, Teach detects on startup that the lesson folder exists, with
`source.submission == 42`. It records `lesson.merged_commit` and shows "now a
permanent test: teach-42" on the public item. It does **not** delete anything. The
admin item then offers **"Delete private copies"** (§13.3), which removes the stored
bitmap, raw SVG and variants and keeps the metadata and lesson link.

## 17. In-app lesson browser (deferred from Stage 2)

* `GET /teach/lessons` (only when Teach is enabled) gives a read-only list of
  `tests/lessons/*` from the deployed code: ID, title, category, status, and the
  `last_run` outcome and commit.
* `GET /teach/lessons/<id>` shows the description, expected behaviour, scoring
  settings and an overlay rendered **on demand** by the same scoring worker. The
  overlay is cached in memory per (commit, lesson). It is admin-triggered for the
  first render, to avoid CPU abuse; after that it is served from the cache.
* Lesson bitmaps are public: generated lessons, plus accepted Teach lessons whose
  licence allows it.
* Rendering uses the Teach templates only. `/test-ui` is unchanged.

## 18. Testing plan (Stage 5)

New folder `tests/teach/`, with a `.gitignore` exception like the others:
* **App setup:** each test builds a **fresh** `Flask` app and calls
  `register_teach(app, env=...)`, plus `LocalStorage(tmp_path)`. No R2 and no network.
  A fake storage class checks the R2 key layout. An optional live-R2 smoke test is
  skipped unless the `R2_*` variables are set.
* **Isolation:** the url_map checks in §3.3; Teach scoring adds no `sessions` keys; the
  main app's url_map is unchanged when the flag is off.
* **Config:** missing or short token, partial R2 settings, and Railway without R2
  each disable Teach without raising; in each case `/teach` returns 404 and the log
  line names the reason without the token's value or length.
* **Validation:** one test per error code, including DOCTYPE/ENTITY, `<use>`, an
  oversized SVG (counting embedded base64), each registration method (image frame,
  border rect, size match, none), a rotated or wrong-aspect `<image>`, a
  decompression-bomb header, multi-frame images, and PNG metadata being stripped. The
  LightBurn sample from Stage 2b is a fixed test case.
* **Trace template:** the endpoint returns the three files, stores nothing (storage
  is unchanged before and after), and its SVG registers back onto the bitmap through
  the reader.
* **Scoring:** a submission built from `tests/lessons/cross-t2-w8-a35` gives the same
  outcome and metrics as `run_lessons` (using the Stage 2 runner); timeout handling
  with a forced sleep.
* **Limits:** per-IP and global limits, quota persistence across a simulated restart,
  the `X-Forwarded-For` hop logic, and spoofed left-most entries being ignored.
* **Admin:** HTTPS enforcement, lockout, `compare_digest` being used (patched to
  record calls), no cookies set, and `Cache-Control: no-store`.
* **Visibility:** unreviewed items expose only the number, status, dates and the
  sanitised title; the sanitiser removes URLs, e-mail addresses, control and bidi
  characters and truncates to 80; hiding a title removes it from the index, the
  duplicate check and alerts; opt-in rules; the manage token in the fragment works;
  deletion leaves a tombstone.
* **Retention:** the sweep honours the 6-hour guard and the lock (two concurrent
  calls delete once), deletes at most 200 per run, never touches accepted items, and
  "Delete private copies" works only after a merge is recorded.
* **Export:** the exported folder passes `validate_meta`, is picked up by
  `discover_lessons`, and gives the same result in `run_lessons`.
* **Notifications:** the ntfy request contains only titles, statuses and links, with
  fixed ASCII headers; the debounce works; the URL never appears in logs.

## 19. Uncertainties and decisions needed

**Decisions (made by the owner, revision 2):**
1. Shared code: option A, move to `lessonkit/` with shims, as the first Stage 5
   commit with tests green before and after (§3.1).
2. Submission licence: CC0 1.0 (§13.1). The repo itself still has no LICENSE file;
   that is separate from Teach.
3. Unreviewed items show a sanitised title (no URLs or control characters, at most
   80 characters) straight away; the admin can hide it with one click. Descriptions,
   thumbnails and bitmaps stay private until review (§11).
4. All five optional environment variables are approved.
   `TEACH_ALERT_WEBHOOK_URL` is a secret (§4).
5. A missing or short admin token disables all of Teach; `/teach` returns 404 and the
   log line says why (§4).
6. The limits in §12 are approved, with `/teach/api/validate` raised to 60/hour per
   IP and 600/hour globally.
7. Retention uses a sweep at startup and on admin summary requests, guarded by a lock
   and a last-swept timestamp; deleting private copies after a merge is an admin
   action (§13.3).
8. Alerts go to ntfy, debounced, with title, status and link only (§15.5).
9. Registration from the `<image>` placement frame, and transforms and units, are
   required before Stage 5 starts (§9.2; implemented in Stage 2b).
10. A stateless "Download trace template" is offered after a bitmap is chosen or
    validated (§5, §14).

**Unverified (check first in Stage 5):**
11. **[unverified]** Railway sets `X-Forwarded-Proto` and appends exactly one
    `X-Forwarded-For` hop, and exposes `RAILWAY_GIT_COMMIT_SHA`.
12. **[unverified]** The Railway plan's memory can hold the live app plus one scoring
    subprocess, about 300–500 MB for the worker with scikit-image loaded. The 1.5 GB
    `RLIMIT_AS` is a ceiling, not an expectation.
13. **[unverified]** R2 conditional writes (`If-Match`) through boto3 for the counter.
    The fallback is in-process locking only, which is safe with one worker.
14. **[unverified]** ntfy's `auth` query parameter for protected topics (§15.5). It is
    not needed if the topic is an unguessable ntfy.sh topic.
15. **[unverified]** How LightBurn imports the trace template SVG (embedded image) and
    the faded PNG; the owner tests this by hand (Stage 2c).

**Assumptions and other notes:**
16. **Single process:** the design assumes one gunicorn worker and one instance, as in
    `railway.json`. `app.yaml` (App Engine, up to 10 instances) appears unused. If it
    is used, or Railway is scaled out, rate limits, locks, the retention lock and the
    counter must move to D1 or KV first (§7.4).
17. **Worker startup:** spawning a subprocess per job costs about 2–3 s of import time.
    A persistent worker process is an optimisation for later.
18. **Scoring only fits stroke-crossing cases well today.** For other kinds of problem
    (noise, gaps, wrong threshold), single-path coverage plus p95 may not express what
    is wrong. The admin can still accept these as `draft` lessons with notes.
19. **SVG transforms, units and `<image>` registration (§9.2)** are required before
    Stage 5 starts. Stage 2b implements them and tests them on the owner's LightBurn
    export; exports from other programs (Inkscape, CAD) have not been tested yet.
20. **Deletion after acceptance** cannot remove a lesson from git history. The consent
    text says so.
21. **Public titles before review** can still carry unwanted words; the sanitiser
    only removes links and invisible characters. One-click hiding is the remedy.

## 20. Rollout

0. Before Stage 5: Stage 2b (SVG transforms, units, `<image>` registration) and
   Stage 2c (trace template) are merged, and the `lessonkit/` move is its own commit.
1. Stage 5 lands with `TEACH_ENABLED` unset. Deploy, and confirm the url_map is
   unchanged and `/teach` returns 404.
2. Create a separate Railway environment (staging) with its own R2 bucket and
   `TEACH_ENABLED=1`. Run the checks in §19 items 11–14 there.
3. Stage 6 (admin review and export) is completed and tested on staging.
4. Enable on production. Keep the global caps low for the first week.

---

## Appendix A. GitHub Actions workflow

Added as `.github/workflows/lessons.yml` in a separate commit after this revision;
that file is authoritative and this copy is for reference. Action versions were checked
against the actions' release pages on 2026-11-09 (checkout v7, setup-python v7,
upload-artifact v7, download-artifact v8). It runs the two new suites under
Python 3.11 (pinned to `runtime.txt`) and 3.12, writes the lesson table as JSON for
each, and then compares the two versions. Lesson differences are only reported and do
not fail the job, but the compare job fails if either run produced no `lessons.json`.
The 3.11.13 job runs on `ubuntu-24.04` because setup-python has no 3.11.13 build for
Ubuntu 26.04 (`ubuntu-latest`); revisit when that image is retired. It never runs
`--mark-failing-as-baseline` or `--record`.

```yaml
name: lessons

on:
  push:
    branches: [main, test-suite-migration]
  pull_request:
  workflow_dispatch:

permissions:
  contents: read

concurrency:
  group: lessons-${{ github.ref }}
  cancel-in-progress: true

jobs:
  test:
    runs-on: ${{ matrix.os }}
    timeout-minutes: 20
    strategy:
      fail-fast: false
      matrix:
        include:
          # 3.11.13 matches runtime.txt (the deployed runtime). setup-python has no 3.11.13
          # build for Ubuntu 26.04 (ubuntu-latest), so this job is pinned to ubuntu-24.04.
          # Revisit when the ubuntu-24.04 runner image is retired or runtime.txt changes.
          - python-version: "3.11.13"
            os: ubuntu-24.04
          - python-version: "3.12"
            os: ubuntu-latest
    env:
      PYTHONWARNINGS: "ignore::FutureWarning"
      TEACH_ENABLED: "0"
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v7
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
      - name: Install
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt "pytest>=8,<10"
          pip freeze > pip-freeze.txt
      - name: pytest tests/pipeline tests/light_port
        run: >
          python -m pytest tests/pipeline tests/light_port -q -rxX -p no:cacheprovider
          --junitxml=junit.xml
      - name: Lesson table
        # exits 1 only if an *active* lesson fails; baseline lessons are reported, not failed
        run: python -m tests.pipeline.run_lessons --json lessons.json
      - uses: actions/upload-artifact@v7
        if: always()
        with:
          name: lessons-py${{ matrix.python-version }}
          path: |
            lessons.json
            junit.xml
            pip-freeze.txt

  compare-python-versions:
    needs: test
    if: always()
    runs-on: ubuntu-latest
    steps:
      - uses: actions/download-artifact@v8
        with:
          path: artifacts
      - name: Compare lesson results 3.11 vs 3.12 (report only)
        run: |
          python - <<'PY'
          import json, pathlib, os
          def load(v):
              p = pathlib.Path(f"artifacts/lessons-py{v}/lessons.json")
              return {r["id"]: r for r in json.loads(p.read_text())} if p.exists() else None
          def norm(x):
              if isinstance(x, dict):
                  return {k: norm(v) for k, v in x.items() if k != "elapsed_ms"}
              if isinstance(x, list):
                  return [norm(v) for v in x]
              return round(x, 3) if isinstance(x, float) else x
          a, b = load("3.11.13"), load("3.12")
          out = ["## Lesson results: Python 3.11 vs 3.12", ""]
          missing = [v for v, r in (("3.11.13", a), ("3.12", b)) if r is None]
          if missing:
              out.append(f"No lessons.json from Python {', '.join(missing)}; see the test job.")
          else:
              diffs = [i for i in sorted(set(a) | set(b)) if norm(a.get(i)) != norm(b.get(i))]
              out.append(f"{len(a)} lessons compared; {len(diffs)} differ (timings ignored).")
              out += [f"- `{i}`: {a.get(i, {}).get('outcome')} vs {b.get(i, {}).get('outcome')}" for i in diffs]
          with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
              f.write("\n".join(out) + "\n")
          print("\n".join(out))
          if missing:
              raise SystemExit(1)
          PY
      - name: Compare installed versions
        run: diff artifacts/lessons-py3.11.13/pip-freeze.txt artifacts/lessons-py3.12/pip-freeze.txt || true
```

Notes on the workflow:
* It runs only the new suites. The legacy `tests/` suite has 7 known failures on a
  clean clone (Stage 0) and would make the job red. It can be added as a separate,
  `continue-on-error` job if wanted.
* `requirements.txt` uses ranges, not pins, so CI may resolve newer scipy and
  scikit-image than the local venv (scipy 1.17.1, scikit-image 0.26.0). The
  `pip freeze` artifacts and diff show exactly what ran. A constraints file would make
  this reproducible, but that is a separate decision.
* `TEACH_ENABLED: "0"` is set so that CI never enables Teach, even after Stage 5.
* No secrets are used. The live-R2 smoke test (§18) stays skipped in CI.
