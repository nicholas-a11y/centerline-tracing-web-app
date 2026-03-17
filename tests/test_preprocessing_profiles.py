from __future__ import annotations

import numpy as np

from centerline_core import _normalize_background_if_needed, resolve_extraction_profile


def _make_faint_line_art_image() -> np.ndarray:
    image = np.ones((96, 96), dtype=np.float32)
    rows = np.arange(12, 84)
    cols = np.rint(48 + 11.0 * np.sin(rows / 7.5)).astype(int)

    for row, col in zip(rows, cols):
        for offset, value in ((0, 0.78), (-1, 0.87), (1, 0.87)):
            cur_col = col + offset
            if 0 <= cur_col < image.shape[1]:
                image[row, cur_col] = min(image[row, cur_col], value)

    return image


def _make_dark_line_art_image() -> np.ndarray:
    image = np.ones((96, 96), dtype=np.float32)
    rows = np.arange(12, 84)
    cols = np.rint(48 + 10.0 * np.sin(rows / 8.0)).astype(int)

    for row, col in zip(rows, cols):
        for offset, value in ((0, 0.12), (-1, 0.34), (1, 0.34)):
            cur_col = col + offset
            if 0 <= cur_col < image.shape[1]:
                image[row, cur_col] = min(image[row, cur_col], value)

    return image


def _make_ultra_faint_uniform_line_art_image() -> np.ndarray:
    image = np.ones((96, 96), dtype=np.float32)
    rows = np.arange(14, 82)
    cols = np.rint(48 + 9.0 * np.sin(rows / 9.0)).astype(int)

    for row, col in zip(rows, cols):
        for offset, value in ((0, 0.90), (-1, 0.95), (1, 0.95)):
            cur_col = col + offset
            if 0 <= cur_col < image.shape[1]:
                image[row, cur_col] = min(image[row, cur_col], value)

    return image


def test_faint_line_art_gets_profile_gated_enhancement():
    gray = _make_faint_line_art_image()

    processed, metadata = _normalize_background_if_needed(
        gray,
        normalization_mode='auto',
        normalization_sensitivity='high',
    )

    assert metadata['faint_line_art_detected'] is True
    assert metadata['profile_name'] == 'faint_line_art'
    assert metadata['enhancement_applied'] is True

    line_mask = gray < 0.95
    background_mask = gray > 0.99
    assert float(processed[line_mask].mean()) < float(gray[line_mask].mean()) - 0.01
    assert abs(float(processed[background_mask].mean()) - float(gray[background_mask].mean())) < 0.02

    extraction_profile = resolve_extraction_profile(metadata)
    assert extraction_profile['profile_name'] == 'faint_line_art'
    assert extraction_profile['force_merge_preview'] is True
    assert extraction_profile['max_min_path_length'] == 2


def test_dark_line_art_stays_on_standard_profile():
    gray = _make_dark_line_art_image()

    processed, metadata = _normalize_background_if_needed(
        gray,
        normalization_mode='auto',
        normalization_sensitivity='medium',
    )

    assert metadata['faint_line_art_detected'] is False
    assert metadata['profile_name'] == 'standard'
    assert metadata['enhancement_applied'] is False
    assert processed.shape == gray.shape

    extraction_profile = resolve_extraction_profile(metadata)
    assert extraction_profile['profile_name'] == 'standard'
    assert extraction_profile['force_merge_preview'] is False
    assert extraction_profile['max_min_path_length'] is None


def test_ultra_faint_uniform_line_art_forces_faint_profile():
    gray = _make_ultra_faint_uniform_line_art_image()

    processed, metadata = _normalize_background_if_needed(
        gray,
        normalization_mode='auto',
        normalization_sensitivity='medium',
    )

    assert metadata['faint_line_art_detected'] is True
    assert metadata['applied'] is True
    assert metadata['reason'] == 'faint line-art on uniform background'
    assert float(processed[gray < 0.98].mean()) < float(gray[gray < 0.98].mean()) - 0.005