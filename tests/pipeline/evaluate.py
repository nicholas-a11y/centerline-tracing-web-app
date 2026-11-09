"""Evaluate a lesson (or user image) through the live pipeline and score it."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .runner import PipelineResult, run_live_both, run_process
from .scoring import LessonScore, graph_metrics, score_paths

PRIMARY_VARIANT = "prune_off"  # UI default: pruning checkbox off


@dataclass
class VariantOutcome:
    result: PipelineResult
    score: LessonScore | None
    graph: dict


@dataclass
class LessonEvaluation:
    id: str
    status: str
    variants: dict[str, VariantOutcome] = field(default_factory=dict)
    reference: list[np.ndarray] | None = None

    @property
    def primary(self) -> VariantOutcome:
        return self.variants[PRIMARY_VARIANT]

    @property
    def scored(self) -> bool:
        return self.status != "draft" and self.reference is not None

    @property
    def passed(self) -> bool | None:
        if not self.scored:
            return None
        sc = self.primary.score
        return bool(sc and sc.passed and not self.primary.result.error)

    @property
    def outcome(self) -> str:
        if not self.scored:
            return "not scored"
        if self.primary.result.error:
            return "ERROR"
        if self.status == "baseline":
            return "UNEXPECTED PASS" if self.passed else "xfail"
        return "pass" if self.passed else "FAIL"


def evaluate(item, include_process: bool = False, process_timeout_s: float = 60.0) -> LessonEvaluation:
    """``item`` is a Lesson or UserImage (needs id, status, params, scoring, image_path, load_reference())."""
    reference = item.load_reference()
    scoring = item.scoring
    runs: dict[str, PipelineResult] = dict(run_live_both(item.image_path, item.params))
    if include_process:
        runs.update(run_process(item.image_path, item.params, timeout_s=process_timeout_s))
    ev = LessonEvaluation(id=item.id, status=item.status, reference=reference)
    for name, res in runs.items():
        if res.error:
            ev.variants[name] = VariantOutcome(res, None, {})
            continue
        sc = score_paths(res.paths, reference, scoring, res.image_shape) if reference else None
        graph = sc.graph if sc else graph_metrics(res.paths, scoring.get("region"), res.image_shape)
        ev.variants[name] = VariantOutcome(res, sc, graph)
    return ev


def variant_cell(v: VariantOutcome | None) -> str:
    if v is None:
        return "-"
    if v.result.timed_out:
        return "TIMEOUT"
    if v.result.error:
        return "ERROR"
    g = v.graph
    base = f"paths={g.get('path_count')} J={g.get('junctions_in_region')} short={g.get('short_paths_in_region')}"
    if v.score is None:
        return base
    s = v.score.summary()
    p95 = "-" if s["max_p95_px"] is None else f"{s['max_p95_px']:.2f}"
    return f"{'PASS' if s['passed'] else 'fail'} cov={s['min_single_coverage']:.2f} p95={p95} {base}"


def evaluation_record(ev: LessonEvaluation) -> dict:
    out = {"id": ev.id, "status": ev.status, "outcome": ev.outcome, "variants": {}}
    for name, v in ev.variants.items():
        rec = {
            "error": v.result.error,
            "timed_out": v.result.timed_out,
            "elapsed_ms": round(v.result.elapsed_ms, 1),
            "counts": v.result.counts,
            "graph": {k: v.graph.get(k) for k in ("path_count", "paths_in_region", "endpoints_in_region",
                                                  "junctions_in_region", "short_paths_in_region",
                                                  "short_path_lengths")},
        }
        if v.score is not None:
            rec["summary"] = v.score.summary()
            rec["lines"] = [{
                "index": ln.index, "best_path": ln.best_path,
                "best_single_coverage": round(ln.best_single_coverage, 4),
                "any_coverage": round(ln.any_coverage, 4), "fragments": ln.fragments,
                "ref_to_best": ln.ref_to_best, "best_to_ref": ln.best_to_ref, "ref_to_any": ln.ref_to_any,
                "passed": ln.passed, "reason": ln.reason,
            } for ln in v.score.lines]
        out["variants"][name] = rec
    return out
