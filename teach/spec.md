# Teach: submitting a hard tracing case

Teach collects cases where the centerline tracer gets it wrong: a **bitmap** plus a
**reference SVG** with the centerlines you expected, drawn by hand. Each case is scored
against the live pipeline. Cases are reviewed by the maintainer and may become
permanent regression tests in the project's repository.

Machine-readable versions: `/teach/openapi.json` (OpenAPI 3.1) and
`/teach/schema/submission.json` (JSON Schema for the `meta` field).

## Rules

1. One problem per submission.
2. Search the queue first (`/teach/queue`). If your case is already there, use "me too".
3. The SVG must be registered to the bitmap. Best: keep the traced image in the export
   (LightBurn does this as long as the image is not hidden or deleted), or start from
   the trace template. Otherwise match the bitmap's pixel size or include a border
   rect. Draw one open path per stroke, along the ideal centre.
4. Crop to the problem area with some context. No confidential or personal content.
5. You cannot set the scoring tolerance; the maintainer does.
6. Run `/teach/api/validate` first and fix every error before submitting.

## Requests

All `POST` endpoints take `multipart/form-data` with:

| part | required | notes |
|---|---|---|
| `image` | yes | PNG or JPEG, at most 4 MB, 32–2048 px per side, at most 4 megapixels, single frame |
| `reference` | yes | SVG, at most 512 KB **including any embedded image**, at most 200 shapes |
| `meta` | yes | JSON object, fields below |

`meta` fields:

| field | required | limit / values |
|---|---|---|
| `title` | yes | 120 characters. A sanitised version (no links, at most 80 characters) is public straight away. |
| `what_is_wrong` | yes | 2000 characters |
| `what_is_expected` | yes | 2000 characters |
| `notes` | no | 4000 characters |
| `focus_box` | no | `{"x0","y0","x1","y1"}` in bitmap pixels (x right, y down, pixel edges); the scoring region. Default: whole image. |
| `settings` | no | `dark_threshold` 0.05–0.8 (0.2), `min_path_length` 1–20 (2), `enable_pruning` (false), `normalization_mode` auto/on/off (auto), `normalization_sensitivity` low/medium/high (medium) |
| `public_thumbnail` | no | `true` lets the overlay thumbnail be public once reviewed (default false) |
| `share_bitmap` | no | `true` lets the full bitmap be public once reviewed (default false) |
| `not_duplicate_reason` | no | 200 characters; one line if `/teach/api/similar` found a similar item |
| `consent` | yes | must be `true`: you accept the consent text below |

Consent text: *I made this image and drawing, or have the right to share them. I have
cropped them to the problem area and they contain no personal or confidential
information. I license them under CC0 1.0 (public domain dedication) so they can be
used as test data for this project. If the case is accepted, it may be published in
the project's public repository. After that it can be removed from future versions but
not from the repository's history. Until then I can delete it at any time with my
manage link.*

## Endpoints

| method and path | purpose |
|---|---|
| `POST /teach/api/validate` | dry run: validate and score; stores nothing |
| `POST /teach/api/trace-template` | `image` only; returns a zip with a faded PNG, a template SVG and a README; stores nothing |
| `POST /teach/api/similar` | duplicate check only |
| `POST /teach/api/submissions` | submit (201) |
| `POST /teach/api/submissions/<number>/me-too` | `meta` = `{"note": "...", "consent": true}`, optionally with a variant `image` and `reference` |
| `GET /teach/api/queue`, `GET /teach/api/submissions/<number>` | public data |
| `GET`/`PATCH`/`DELETE /teach/api/manage/<id>` | your own item; send the token from your manage link in `X-Teach-Manage-Token` |

## The SVG

* SVG coordinates are **pixel edges**: the centre of pixel column 10 is x = 10.5.
* Registration (how the SVG is placed on the bitmap), in order:
  1. **An embedded `<image>`** (preferred): exactly one `<image>`, the bitmap you traced
     over, embedded as base64 PNG or JPEG. Its placed rectangle is mapped onto the
     bitmap. Its stored pixel size must have the bitmap's aspect ratio (to within 1 px).
     Flips are fine; rotation, skew and stretching are not.
  2. **A border rect**: a `rect` with `id`, `class` or `data-role` = `border` spanning the image.
  3. **A size match**: `width`/`height`/`viewBox` equal to the bitmap's pixel size.
* Draw one open `<path>` (or `line`/`polyline`) per stroke along the ideal centre.
  Transforms (`matrix`, `translate`, `scale`, `rotate`, `skewX`, `skewY`) and units
  (`px`, `mm`, `cm`, `in`, `pt`, `pc`) are supported.
* Not supported: `<use>`, DOCTYPE/ENTITY declarations, linked (non-embedded) images.
  Styles, text and other non-geometry elements are ignored.
* **LightBurn:** keep the image in the export (do not hide or delete it) and leave the
  traced paths ungrouped. Document units mm or px are both fine.

## Trace template (recommended for agents)

When you have a bitmap but no SVG yet, `POST` the bitmap alone to
`/teach/api/trace-template`. The zip contains:

* `<name>-faded.png`: the bitmap faded towards white, same pixel size, 96 dpi;
* `<name>-template.svg`: `width`/`height` in px equal to the bitmap, `viewBox="0 0 W H"`,
  the faded PNG embedded at `0,0 W×H`, and nothing else;
* `<name>-README.txt`.

Add one open `<path>` per stroke to the template SVG, in its pixel coordinates (pixel
edges), keep the `<image>`, and submit that file as `reference`. This is the most
reliable way to produce a correctly registered SVG.

## Worked example

Files: `/teach/example/image.png`, `/teach/example/reference.svg`,
`/teach/example/request.json` (a 2 px line crossing an 8 px line at 35 degrees).

```bash
curl -sS -X POST https://<host>/teach/api/validate \
  -F image=@image.png -F reference=@reference.svg \
  -F 'meta={"title":"Thin line kinks where it crosses a thick line","what_is_wrong":"...","what_is_expected":"...","focus_box":{"x0":103.28,"y0":103.28,"x1":152.72,"y1":152.72},"consent":true}'
```

Expected response (abridged): the case is valid (`ok: true`) and the engine fails it,
which is exactly the kind of case Teach is for:

```json
{"ok": true, "errors": [], "warnings": [],
 "result": {"outcome": "fail", "passed": false, "variant": "prune_off",
            "summary": {"passed": false, "min_single_coverage": 0.386, "max_p95_px": 32.631,
                        "path_count": 5, "junctions_in_region": 2, "...": "..."},
            "registration": "size_match", "engine_commit": "...",
            "trace_template_url": "/teach/api/trace-template",
            "overlay_png": "data:image/png;base64,..."}}
```

## Responses

Every response is JSON with the same shape:

```json
{"ok": false,
 "errors": [{"field": "reference", "code": "svg_no_registration", "message": "...", "hint": "..."}],
 "warnings": [{"field": "image", "code": "image_large", "message": "...", "hint": "..."}],
 "result": null}
```

`validate` returns 200 whether or not the case is valid; check `ok`. `submissions`
returns 201 with `number`, `status`, `queue_url` and `manage_url`. **The manage link is
shown once and cannot be recovered**; it lets you view and delete your submission.
Other statuses: 409 `duplicate_exact`, 413 `request_too_large`, 422 validation errors,
429 `rate_limited` / `quota_exceeded` (with `Retry-After`), 503 `busy` (with
`Retry-After: 30`).

## Error codes

* Bitmap: `image_too_large`, `image_dimensions`, `image_format`, `image_multiframe`.
* SVG: `svg_too_large`, `svg_doctype`, `svg_parse`, `svg_use_unsupported`,
  `svg_transform_invalid`, `svg_units_unsupported`, `svg_no_registration`,
  `svg_multiple_images`, `svg_image_rotated`, `svg_image_aspect`, `svg_bitmap_mismatch`,
  `svg_image_dimensions`, `svg_image_unreadable`, `svg_no_geometry`, `svg_too_many_elements`.
* Fields: `field_missing`, `field_too_long`, `param_out_of_range`, `consent_required`, `meta_invalid`.
* Limits and state: `duplicate_exact`, `rate_limited`, `quota_exceeded`, `busy`,
  `scoring_timeout` (a warning: the case is still stored), `request_too_large`,
  `invalid_state`, `not_found`, `manage_token_invalid`.
* Warnings: `image_no_dark`, `image_mostly_dark`, `image_large`, `svg_warning`, `similar`.

## Limits

* Validate and trace template together: 60 per hour per IP, 600 per hour overall.
* Similar: 60 per hour per IP.
* Submissions and "me too" together: 5 per hour and 20 per day per IP, 200 per day overall;
  20 MB per day per IP and 500 MB per day overall.
* Manage-link actions: 30 per hour per IP.
* Scoring runs one case at a time with at most 3 waiting, and at most one per client.
  Each case has 30 s.
