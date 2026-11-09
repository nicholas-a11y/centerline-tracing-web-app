# Fixture ground-truth audit (Stage 2)

Question: is each legacy fixture's `ideal_sample` (including `inset_px`) a valid
ground truth for **centerline placement**? If it is, the fixture can become a
placement lesson.

Method (`python -m tests.pipeline.fixture_audit`): render the fixture in memory
with `render_fixture`. `load_fixture` is not used because it writes into
`tests/fixtures/data`. For each `ideal_sample` point, scan along the normal to
both stroke edges and measure the signed offset of the sample from the true
stroke midline. This is independent of any engine.

A fixture is valid if p95 |offset| ≤ 0.5 px, the share of samples off the stroke
is ≤ 2 %, and the mismatch between the ends of the ideal sample and the stroke
ends is ≤ 1.5 px.

## 64 px, stroke width 3 (used for the lessons)

| fixture | valid | bias px | mean abs px | p95 abs px | max abs px | off-stroke | end mismatch px | reasons |
|---|---|---|---|---|---|---|---|---|
| curve-quarter-arc | yes | -0.02 | 0.14 | 0.32 | 0.50 | 0.0% | 0.49 | - |
| curve-half-arc | yes | 0.02 | 0.13 | 0.31 | 0.50 | 0.0% | 0.49 | - |
| curve-true-circle | yes | 0.01 | 0.15 | 0.35 | 0.42 | 0.0% | - | - |
| curve-s-cubic | yes | -0.00 | 0.22 | 0.48 | 0.61 | 0.0% | 0.57 | - |
| curve-gentle-convex | yes | -0.06 | 0.15 | 0.29 | 0.40 | 0.0% | 0.31 | - |
| curve-tight-180 | yes | -0.02 | 0.15 | 0.33 | 0.53 | 0.0% | 0.49 | - |
| curve-line-curve-line | **no** | 0.73 | 0.73 | 0.92 | 0.97 | 0.0% | 0.68 | p95 offset 0.92px > 0.5px |
| curve-to-straight-to-curve | **no** | 0.27 | 0.42 | 0.89 | 0.98 | 0.7% | 0.91 | p95 offset 0.89px > 0.5px |
| curve-perfect-circle | yes | -0.02 | 0.13 | 0.31 | 0.37 | 0.0% | - | - |
| square-opposing-curves | **no** | -0.14 | 0.55 | 1.18 | 2.64 | 0.0% | - | p95 offset 1.18px > 0.5px |
| line-horizontal | yes | 0.01 | 0.01 | 0.01 | 0.01 | 0.3% | 0.60 | - |
| line-diagonal-45 | yes | -0.00 | 0.00 | 0.00 | 0.00 | 0.7% | 0.85 | - |
| line-horizontal-r3 | **no** | 0.53 | 0.53 | 0.92 | 1.48 | 0.0% | 0.48 | p95 offset 0.92px > 0.5px |
| square-perfect | **no** | 0.17 | 0.59 | 1.60 | 2.42 | 0.0% | - | p95 offset 1.60px > 0.5px |
| square-perfect-r3 | **no** | -0.13 | 0.62 | 1.94 | 2.73 | 0.0% | - | p95 offset 1.94px > 0.5px |
| corner-l-shape | **no** | -0.01 | 0.55 | 0.59 | 2.53 | 0.0% | 0.60 | p95 offset 0.59px > 0.5px |
| corner-l-shape-r45 | yes | -0.01 | 0.38 | 0.42 | 2.10 | 0.2% | 0.42 | - |

## 64 px, stroke width 2

| fixture | valid | bias px | mean abs px | p95 abs px | max abs px | off-stroke | end mismatch px | reasons |
|---|---|---|---|---|---|---|---|---|
| curve-quarter-arc | **no** | 0.48 | 0.48 | 0.73 | 0.83 | 0.0% | 0.49 | p95 offset 0.73px > 0.5px |
| curve-half-arc | **no** | 0.52 | 0.52 | 0.78 | 0.88 | 0.0% | 0.49 | p95 offset 0.78px > 0.5px |
| curve-true-circle | **no** | 0.52 | 0.52 | 0.78 | 0.89 | 0.0% | - | p95 offset 0.78px > 0.5px |
| curve-s-cubic | **no** | -0.14 | 0.34 | 0.93 | 0.98 | 0.0% | 0.57 | p95 offset 0.93px > 0.5px |
| curve-gentle-convex | **no** | -0.41 | 0.41 | 0.78 | 0.79 | 0.0% | 0.82 | p95 offset 0.78px > 0.5px |
| curve-tight-180 | **no** | 0.47 | 0.47 | 0.74 | 0.79 | 0.0% | 0.49 | p95 offset 0.74px > 0.5px |
| curve-line-curve-line | **no** | 0.79 | 0.79 | 1.18 | 1.26 | 0.0% | 0.68 | p95 offset 1.18px > 0.5px |
| curve-to-straight-to-curve | **no** | 0.19 | 0.35 | 1.02 | 1.47 | 0.7% | 0.91 | p95 offset 1.02px > 0.5px |
| curve-perfect-circle | **no** | 0.48 | 0.48 | 0.73 | 0.78 | 0.0% | - | p95 offset 0.73px > 0.5px |
| square-opposing-curves | **no** | -0.03 | 0.56 | 1.23 | 2.57 | 0.0% | - | p95 offset 1.23px > 0.5px |
| line-horizontal | yes | -0.49 | 0.49 | 0.49 | 0.50 | 0.3% | 0.60 | - |
| line-diagonal-45 | yes | -0.00 | 0.00 | 0.00 | 0.00 | 0.0% | 0.14 | - |
| line-horizontal-r3 | yes | 0.06 | 0.25 | 0.47 | 0.52 | 0.2% | 0.53 | - |
| square-perfect | **no** | 0.23 | 0.69 | 1.31 | 2.38 | 0.0% | - | p95 offset 1.31px > 0.5px |
| square-perfect-r3 | **no** | -0.08 | 0.67 | 1.43 | 3.48 | 0.0% | - | p95 offset 1.43px > 0.5px |
| corner-l-shape | yes | -0.04 | 0.13 | 0.09 | 2.03 | 0.0% | 0.60 | - |
| corner-l-shape-r45 | yes | -0.06 | 0.35 | 0.42 | 2.63 | 0.0% | 0.99 | - |

## Conclusions

* The `ideal_sample` curves are fitting targets. They are not reliable placement
  ground truth. Validity depends on stroke width, because `inset_px` is fixed
  (1.0) while the rasterised stroke centre moves by half a pixel between odd and
  even widths. At sw2, almost every curve has a bias of about ±0.5 px.
* **Invalid at sw3, and kept as `draft` lessons (not run in pytest):**
  curve-line-curve-line, curve-to-straight-to-curve, square-opposing-curves,
  line-horizontal-r3, square-perfect, square-perfect-r3, corner-l-shape.
  Excluding the corner regions still leaves p95 at 0.59–0.92 px, so these are
  real offsets, not artefacts of the corner scan.
* **Valid at sw3, and converted to lessons:** quarter-arc, half-arc, true-circle,
  s-cubic, gentle-convex, tight-180, perfect-circle, line-horizontal,
  line-diagonal-45, corner-l-shape-r45. Of these, corner-l-shape-r45 currently
  fails on the engine (the corner is split, with spurs), so it is `baseline`.
* The originals in `tests/fixtures/` were not modified. The lessons were
  generated once from the current definitions by
  `tests/pipeline/generate_lessons.py`.
