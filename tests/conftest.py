"""
tests/conftest.py

Shared pytest infrastructure for curve-fitting golden master tests.

Key responsibilities
--------------------
1. Category-grouped fixture ID lists (parametrize helpers).
2. --update-goldens CLI flag + fixture.
3. Analytical deviation helpers used by TestGoldenMaster and TestFitCurvesContract.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

from tests.fixtures import FIXTURES, load_fixture, LoadedFixture, fixture_stem

# ─── Category lists ───────────────────────────────────────────────────────────

def _ids_for(category: str) -> list[str]:
    return [f.id for f in FIXTURES if f.category == category]


CURVE_FIXTURES   = _ids_for("curve")
LINE_FIXTURES    = _ids_for("line")
CORNER_FIXTURES  = _ids_for("corner")
ALL_FIXTURE_IDS  = [f.id for f in FIXTURES]

# ─── Golden file location ─────────────────────────────────────────────────────

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "golden"

# ─── pytest CLI option ────────────────────────────────────────────────────────

def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-goldens",
        action="store_true",
        default=False,
        help="Overwrite golden snapshot files with current results.",
    )


@pytest.fixture(scope="session")
def update_goldens(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--update-goldens"))


# ─── Analytical deviation helpers ─────────────────────────────────────────────

def sample_fitted_segments(
    start_rc: list[int],
    segments: list[dict],
    n_per_segment: int = 60,
) -> np.ndarray:
    """Densely sample a list of fit_curve_segments() output dicts.

    Returns an (N, 2) float32 array of [row, col] points along the fitted
    curve.  Uses the implicit start point + each segment's end_point /
    control points to reconstruct the geometry analytically.
    """
    if not segments:
        return np.array([start_rc], dtype=np.float32)

    cursor = np.array(start_rc, dtype=np.float64)
    out = [cursor.copy()]

    for seg in segments:
        end = np.array(seg["end_point"], dtype=np.float64)
        if seg["type"] == "line":
            t = np.linspace(0.0, 1.0, n_per_segment + 1)[1:]
            pts = cursor[None, :] + t[:, None] * (end - cursor)[None, :]
            out.append(pts)
        elif seg["type"] == "cubic":
            c1 = np.array(seg["control1"], dtype=np.float64)
            c2 = np.array(seg["control2"], dtype=np.float64)
            t = np.linspace(0.0, 1.0, n_per_segment + 1)[1:]
            omt = 1.0 - t
            pts = (
                (omt**3)[:, None] * cursor
                + (3 * omt**2 * t)[:, None] * c1
                + (3 * omt * t**2)[:, None] * c2
                + (t**3)[:, None] * end
            )
            out.append(pts)
        cursor = end

    return np.vstack(out).astype(np.float32)


def analytical_deviation(
    fitted_pts: np.ndarray,    # [[row, col]], shape (N, 2)
    ideal_pts: np.ndarray,     # [[row, col]], shape (M, 2)
) -> dict:
    """Compute analytical deviation between a fitted curve and an ideal curve.

    Both arrays are in (row, col) space (same as extract_skeleton_paths output).

    Returns
    -------
    mean_deviation_px  : float — mean nearest-neighbour distance, fitted → ideal
    max_deviation_px   : float — max  nearest-neighbour distance (Hausdorff-like)
    coverage           : float — fraction of ideal pts within tolerance_px=3
    """
    if len(fitted_pts) == 0 or len(ideal_pts) == 0:
        return {
            "mean_deviation_px": float("inf"),
            "max_deviation_px": float("inf"),
            "coverage": 0.0,
        }

    tree_ideal = cKDTree(ideal_pts)
    dists_f2i, _ = tree_ideal.query(fitted_pts, workers=-1)

    tree_fitted = cKDTree(fitted_pts)
    dists_i2f, _ = tree_fitted.query(ideal_pts, workers=-1)

    # Coverage: fraction of ideal pts with a fitted pt closer than 3 px
    coverage_threshold = 3.0
    coverage = float(np.mean(dists_i2f <= coverage_threshold))

    return {
        "mean_deviation_px": float(np.mean(dists_f2i)),
        "max_deviation_px":  float(np.max(dists_f2i)),
        "coverage":          coverage,
    }


def flatten_all_fitted_paths(
    fitted_paths: list[list[dict]],
    raw_paths: list[list],
) -> np.ndarray:
    """Flatten all fitted paths to a single (N, 2) array for global metrics.

    fitted_paths — output of fit_curve_segments(raw_paths)
    raw_paths    — original [[row,col]] paths (supplies the start point of each)
    """
    all_pts: list[np.ndarray] = []
    for raw_path, fitted in zip(raw_paths, fitted_paths):
        if not fitted:
            continue
        pts = sample_fitted_segments(raw_path[0], fitted)
        all_pts.append(pts)
    if not all_pts:
        return np.zeros((0, 2), dtype=np.float32)
    return np.vstack(all_pts)
