#!/usr/bin/env python3
"""
Shared preprocessing and auto-tune helpers for the centerline web app.
"""

import time
from io import BytesIO
from pathlib import Path
import os
import tempfile

import numpy as np
from PIL import Image

from centerline_engine import extract_skeleton_paths, merge_nearby_paths, prune_extracted_paths


AUTO_TUNE_TIME_BUDGET_SEC = 60.0
AUTO_TUNE_CONFIDENCE_TARGET = 0.90
AUTO_TUNE_RANDOM_TILE_DIM = 144
AUTO_TUNE_RANDOM_TILE_COUNT = 12
AUTO_TUNE_MIN_TILE_COUNT = 4
AUTO_TUNE_RANDOM_TILE_DIM_MAX = 320
AUTO_TUNE_TARGET_COVERAGE_RATIO = 0.06
AUTO_TUNE_TARGET_COVERAGE_RATIO_MIN = 0.03
AUTO_TUNE_HOTSPOT_TILE_RATIO = 0.40
RESOLUTION_SCALE_REFERENCE_LONG_SIDE_PX = 1600.0
RESOLUTION_SCALE_MAX = 4.0


def resolve_parameter_scale(image_shape, reference_long_side=RESOLUTION_SCALE_REFERENCE_LONG_SIDE_PX,
                            max_scale=RESOLUTION_SCALE_MAX):
    """Return a conservative resolution scale for pixel-based extraction settings."""
    if image_shape is None or len(image_shape) < 2:
        return 1.0

    try:
        height = max(1, int(image_shape[0]))
        width = max(1, int(image_shape[1]))
    except Exception:
        return 1.0

    long_side = float(max(height, width))
    if long_side <= float(reference_long_side):
        return 1.0

    return max(1.0, min(float(max_scale), long_side / float(reference_long_side)))


def scale_length_parameter(value, image_shape=None, parameter_scale=None, minimum=1):
    """Scale a pixel-length parameter for very large images while keeping defaults stable."""
    scale = float(parameter_scale) if parameter_scale is not None else resolve_parameter_scale(image_shape)
    scale = max(1.0, scale)
    return max(int(minimum), int(round(float(value) * scale)))


def _resolve_auto_tune_tile_dim(image_shape, base_tile_dim=AUTO_TUNE_RANDOM_TILE_DIM,
                                max_tile_dim=AUTO_TUNE_RANDOM_TILE_DIM_MAX):
    """Increase autotune tile size moderately on very large images."""
    scale = max(1.0, resolve_parameter_scale(image_shape))
    adaptive_dim = int(round(float(base_tile_dim) * np.sqrt(scale)))
    return max(
        int(base_tile_dim),
        min(
            int(max_tile_dim),
            int(max(1, image_shape[0])),
            int(max(1, image_shape[1])),
            adaptive_dim,
        ),
    )


def resolve_auto_tune_sampling_plan(image_shape, base_tile_dim=AUTO_TUNE_RANDOM_TILE_DIM,
                                    min_tile_count=AUTO_TUNE_MIN_TILE_COUNT,
                                    max_tile_count=AUTO_TUNE_RANDOM_TILE_COUNT):
    """Return a deterministic autotune sampling plan for large images."""
    if image_shape is None or len(image_shape) < 2:
        return {
            'tile_dim': int(base_tile_dim),
            'tile_count': int(min_tile_count),
            'hotspot_tile_target': max(1, int(round(min_tile_count * AUTO_TUNE_HOTSPOT_TILE_RATIO))),
            'coverage_tile_target': max(1, int(min_tile_count) - max(1, int(round(min_tile_count * AUTO_TUNE_HOTSPOT_TILE_RATIO)))),
            'parameter_scale': 1.0,
            'target_coverage_ratio': float(AUTO_TUNE_TARGET_COVERAGE_RATIO),
            'target_sample_area': int(base_tile_dim * base_tile_dim * min_tile_count),
            'plan_version': 'coverage_hotspot_v2',
        }

    height = max(1, int(image_shape[0]))
    width = max(1, int(image_shape[1]))
    image_area = max(1, height * width)
    tile_dim = _resolve_auto_tune_tile_dim(image_shape, base_tile_dim=base_tile_dim)
    tile_area = max(1, int(tile_dim) * int(tile_dim))
    parameter_scale = max(1.0, resolve_parameter_scale(image_shape))

    target_coverage_ratio = float(np.clip(
        AUTO_TUNE_TARGET_COVERAGE_RATIO / np.sqrt(parameter_scale),
        AUTO_TUNE_TARGET_COVERAGE_RATIO_MIN,
        AUTO_TUNE_TARGET_COVERAGE_RATIO,
    ))
    target_sample_area = max(tile_area, int(round(image_area * target_coverage_ratio)))
    coverage_count_from_area = int(np.ceil(target_sample_area / float(tile_area)))

    aspect_ratio = float(width) / float(height)
    spatial_cols = int(np.ceil(np.sqrt(max(1.0, aspect_ratio) * 2.0)))
    spatial_rows = int(np.ceil(np.sqrt(max(1.0, 1.0 / max(aspect_ratio, 1e-6))) * 2.0))
    spatial_goal = max(2, spatial_cols + spatial_rows)

    tile_count = max(int(min_tile_count), coverage_count_from_area, spatial_goal)
    tile_count = min(int(max_tile_count), tile_count)

    hotspot_tile_target = min(
        max(1, int(round(tile_count * AUTO_TUNE_HOTSPOT_TILE_RATIO))),
        max(1, tile_count - 1),
    )
    coverage_tile_target = max(1, tile_count - hotspot_tile_target)

    return {
        'tile_dim': int(tile_dim),
        'tile_count': int(tile_count),
        'hotspot_tile_target': int(hotspot_tile_target),
        'coverage_tile_target': int(coverage_tile_target),
        'parameter_scale': float(parameter_scale),
        'target_coverage_ratio': float(target_coverage_ratio),
        'target_sample_area': int(target_sample_area),
        'source_area': int(image_area),
        'plan_version': 'coverage_hotspot_v2',
    }


def _generate_coverage_tile_origins(image_shape, tile_w, tile_h, target_count):
    """Generate deterministic, scattered coverage origins using jittered spatial cells."""
    height = max(1, int(image_shape[0]))
    width = max(1, int(image_shape[1]))
    max_y = max(0, height - int(tile_h))
    max_x = max(0, width - int(tile_w))
    if target_count <= 0:
        return []

    aspect_ratio = float(width) / float(height)
    cols = max(1, int(np.ceil(np.sqrt(float(target_count) * max(aspect_ratio, 1e-6)))))
    rows = max(1, int(np.ceil(float(target_count) / float(cols))))

    rng_seed = int((height << 16) ^ (width << 2) ^ (int(tile_w) << 9) ^ int(tile_h) ^ (int(target_count) << 5)) & 0xFFFFFFFF
    rng = np.random.default_rng(rng_seed)

    x_edges = np.linspace(0, max_x, num=cols + 1)
    y_edges = np.linspace(0, max_y, num=rows + 1)
    cell_indices = [(row_index, col_index) for row_index in range(rows) for col_index in range(cols)]
    rng.shuffle(cell_indices)

    origins = []
    for row_index, col_index in cell_indices:
        if len(origins) >= int(target_count):
            break

        x0 = int(round(x_edges[col_index]))
        x1 = int(round(x_edges[col_index + 1]))
        y0 = int(round(y_edges[row_index]))
        y1 = int(round(y_edges[row_index + 1]))

        cell_w = max(1, x1 - x0)
        cell_h = max(1, y1 - y0)
        x_margin = min(max(0, cell_w // 6), max(0, (x1 - x0) // 2))
        y_margin = min(max(0, cell_h // 6), max(0, (y1 - y0) // 2))

        x_min = min(max_x, x0 + x_margin)
        x_max = min(max_x, max(x_min, x1 - x_margin))
        y_min = min(max_y, y0 + y_margin)
        y_max = min(max_y, max(y_min, y1 - y_margin))

        x = int(rng.integers(x_min, x_max + 1)) if x_max > x_min else int(x_min)
        y = int(rng.integers(y_min, y_max + 1)) if y_max > y_min else int(y_min)
        origins.append((x, y))

    return origins


def _window_sum(integral_image, x, y, width, height):
    x1 = int(x)
    y1 = int(y)
    x2 = int(x + width)
    y2 = int(y + height)
    return float(
        integral_image[y2, x2]
        - integral_image[y1, x2]
        - integral_image[y2, x1]
        + integral_image[y1, x1]
    )


def _compute_hotspot_tile_candidates(gray_image, tile_w, tile_h, max_candidates=96):
    """Rank coarse tile origins by likely line activity."""
    gray = np.clip(gray_image.astype(np.float32), 0.0, 1.0)
    darkness = np.clip(0.92 - gray, 0.0, 1.0)
    grad_y, grad_x = np.gradient(gray)
    edge_strength = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    edge_scale = max(float(np.percentile(edge_strength, 99.5)), 1e-4)
    edge_strength = np.clip(edge_strength / edge_scale, 0.0, 1.0)

    activity = np.clip(0.72 * darkness + 0.28 * edge_strength, 0.0, 1.0)
    activity_peak = float(activity.max()) if activity.size else 0.0
    activity_mean = float(activity.mean()) if activity.size else 0.0
    if activity_peak < 0.035:
        return [], {
            'activity_peak': activity_peak,
            'activity_mean': activity_mean,
        }

    padded_integral = np.pad(activity, ((1, 0), (1, 0)), mode='constant').cumsum(axis=0).cumsum(axis=1)
    image_h, image_w = gray.shape
    max_y = max(0, int(image_h - tile_h))
    max_x = max(0, int(image_w - tile_w))
    step_y = max(1, int(tile_h // 2))
    step_x = max(1, int(tile_w // 2))

    y_candidates = list(range(0, max_y + 1, step_y))
    x_candidates = list(range(0, max_x + 1, step_x))
    if y_candidates[-1] != max_y:
        y_candidates.append(max_y)
    if x_candidates[-1] != max_x:
        x_candidates.append(max_x)

    candidates = []
    tile_area = float(max(1, tile_h * tile_w))
    for y in y_candidates:
        for x in x_candidates:
            score = _window_sum(padded_integral, x, y, tile_w, tile_h) / tile_area
            candidates.append((float(score), int(x), int(y)))

    if not candidates:
        return [], {
            'activity_peak': activity_peak,
            'activity_mean': activity_mean,
        }

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score = float(candidates[0][0])
    if best_score < max(0.040, activity_mean * 1.35):
        return [], {
            'activity_peak': activity_peak,
            'activity_mean': activity_mean,
        }

    return candidates[:max(1, int(max_candidates))], {
        'activity_peak': activity_peak,
        'activity_mean': activity_mean,
    }


def resolve_extraction_profile(preprocessing_info=None):
    """Return extraction hints derived from upload-time preprocessing metadata."""
    info = preprocessing_info or {}
    profile_name = str(info.get('profile_name') or '').strip().lower()
    faint_line_art = bool(
        info.get('faint_line_art_detected', False)
        or info.get('enhancement_applied', False)
        or profile_name == 'faint_line_art'
    )

    if faint_line_art:
        return {
            'profile_name': 'faint_line_art',
            'preview_min_object_size': 1,
            'full_min_object_size': 2,
            'auto_tune_min_object_size': 1,
            'max_min_path_length': 2,
            'force_merge_preview': True,
        }

    return {
        'profile_name': 'standard',
        'preview_min_object_size': 2,
        'full_min_object_size': 5,
        'auto_tune_min_object_size': 3,
        'max_min_path_length': None,
        'force_merge_preview': False,
    }


def _enhance_faint_line_art(gray_image):
    """Darken likely ink regions without broadly amplifying paper noise."""
    from skimage import filters

    gray = np.clip(gray_image.astype(np.float32), 0.0, 1.0)
    local_background = filters.gaussian(gray, sigma=2.2, preserve_range=True)
    local_darkness = np.clip(local_background - gray, 0.0, 1.0)
    local_darkness_scale = max(float(np.percentile(local_darkness, 99.2)), 1e-4)
    local_darkness = np.clip(local_darkness / local_darkness_scale, 0.0, 1.0)

    edge_strength = np.abs(filters.sobel(gray))
    edge_scale = max(float(np.percentile(edge_strength, 99.5)), 1e-4)
    edge_strength = np.clip(edge_strength / edge_scale, 0.0, 1.0)

    darkness_mask = np.clip((0.95 - gray) / 0.28, 0.0, 1.0)
    stroke_mask = np.clip(0.70 * darkness_mask + 0.20 * local_darkness + 0.10 * edge_strength, 0.0, 1.0)
    stroke_mask = np.power(stroke_mask, 1.45)

    tone_mapped = np.power(gray, 1.18)
    local_target = np.clip(gray - 0.12 * local_darkness, 0.0, 1.0)
    target = np.minimum(tone_mapped, local_target)
    blend = 0.65 * stroke_mask
    enhanced = gray * (1.0 - blend) + target * blend

    return np.clip(enhanced.astype(np.float32), 0.0, 1.0)


def _build_random_tile_mosaic(gray_image, tile_dim=AUTO_TUNE_RANDOM_TILE_DIM, tile_count=None):
    """Create a multi-tile mosaic for auto-tune scoring on large images."""
    height, width = gray_image.shape
    plan = resolve_auto_tune_sampling_plan(gray_image.shape, base_tile_dim=tile_dim)
    requested_count = int(tile_count) if tile_count is not None else int(plan['tile_count'])
    requested_count = max(2, requested_count)
    resolved_tile_dim = int(plan['tile_dim'])
    tile_h = min(int(resolved_tile_dim), int(height))
    tile_w = min(int(resolved_tile_dim), int(width))

    if tile_h >= height and tile_w >= width:
        return gray_image, {
            'sampled': False,
            'sample_shape': gray_image.shape,
            'sample_origin': [0, 0],
            'source_shape': gray_image.shape,
            'sampling_mode': 'full_image',
            'overlay_supported': True,
        }

    max_y = max(0, height - tile_h)
    max_x = max(0, width - tile_w)
    tiles = []
    tile_origins = []
    tile_selection_modes = []
    used_origins = set()

    def _overlaps_existing(x, y, allowed_overlap_ratio=0.0):
        for ox, oy in used_origins:
            overlap_w = max(0, min(x + tile_w, ox + tile_w) - max(x, ox))
            overlap_h = max(0, min(y + tile_h, oy + tile_h) - max(y, oy))
            overlap_area = overlap_w * overlap_h
            if overlap_area > int(round(tile_w * tile_h * float(allowed_overlap_ratio))):
                return True
        return False

    def _append_tile(x, y, mode, allowed_overlap_ratio=0.0):
        key = (int(x), int(y))
        if key in used_origins or _overlaps_existing(int(x), int(y), allowed_overlap_ratio=allowed_overlap_ratio):
            return False
        tile = gray_image[int(y):int(y) + tile_h, int(x):int(x) + tile_w]
        if tile.shape[0] != tile_h or tile.shape[1] != tile_w:
            return False
        used_origins.add(key)
        tiles.append(tile)
        tile_origins.append([int(x), int(y)])
        tile_selection_modes.append(str(mode))
        return True

    hotspot_candidates, activity_summary = _compute_hotspot_tile_candidates(gray_image, tile_w, tile_h)
    hotspot_target = 0
    if hotspot_candidates:
        hotspot_target = min(int(plan['hotspot_tile_target']), max(1, requested_count - 1))

    hotspot_selected = 0
    for _, x, y in hotspot_candidates:
        if hotspot_selected >= hotspot_target or len(tiles) >= requested_count:
            break
        if _append_tile(x, y, mode='hotspot', allowed_overlap_ratio=0.18):
            hotspot_selected += 1

    coverage_target = max(0, min(int(plan['coverage_tile_target']), requested_count - len(tiles)))
    for x, y in _generate_coverage_tile_origins(gray_image.shape, tile_w, tile_h, coverage_target):
        if len(tiles) >= requested_count:
            break
        _append_tile(x, y, mode='coverage', allowed_overlap_ratio=0.10)

    fallback_origins = [
        (0, 0),
        (max_x, 0),
        (0, max_y),
        (max_x, max_y),
        (max_x // 2, max_y // 2),
    ]
    for x, y in fallback_origins:
        if len(tiles) >= requested_count:
            break
        _append_tile(int(x), int(y), mode='fallback')

    if not tiles:
        return gray_image, {
            'sampled': False,
            'sample_shape': gray_image.shape,
            'sample_origin': [0, 0],
            'source_shape': gray_image.shape,
            'sampling_mode': 'full_image_fallback',
            'overlay_supported': True,
        }

    count = len(tiles)
    cols = int(np.ceil(np.sqrt(count)))
    rows = int(np.ceil(count / float(cols)))
    mosaic = np.ones((rows * tile_h, cols * tile_w), dtype=np.float32)

    for idx, tile in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        y0 = row * tile_h
        x0 = col * tile_w
        mosaic[y0:y0 + tile_h, x0:x0 + tile_w] = tile

    return mosaic, {
        'sampled': True,
        'sample_shape': mosaic.shape,
        'sample_origin': [0, 0],
        'source_shape': gray_image.shape,
        'sampling_mode': 'random_tiles_mosaic',
        'tile_count': int(count),
        'tile_shape': [int(tile_h), int(tile_w)],
        'tile_origins': tile_origins,
        'tile_selection_modes': tile_selection_modes,
        'selected_hotspot_tiles': int(sum(1 for mode in tile_selection_modes if mode == 'hotspot')),
        'selected_coverage_tiles': int(sum(1 for mode in tile_selection_modes if mode == 'coverage')),
        'selected_random_tiles': 0,
        'selected_fallback_tiles': int(sum(1 for mode in tile_selection_modes if mode == 'fallback')),
        'requested_hotspot_tiles': int(hotspot_target),
        'requested_coverage_tiles': int(coverage_target),
        'adaptive_tile_dim': int(resolved_tile_dim),
        'tile_strategy': str(plan['plan_version']),
        'parameter_scale': float(plan['parameter_scale']),
        'target_coverage_ratio': round(float(plan['target_coverage_ratio']), 5),
        'target_sample_area': int(plan['target_sample_area']),
        'source_area': int(plan['source_area']),
        'activity_peak': round(float(activity_summary.get('activity_peak', 0.0)), 5),
        'activity_mean': round(float(activity_summary.get('activity_mean', 0.0)), 5),
        'overlay_supported': False,
    }


def _resolve_crop_bounds(image_shape, crop_region):
    """Convert normalized crop values to pixel bounds within image limits."""
    if crop_region is None or not isinstance(crop_region, dict):
        return None
    try:
        h, w = int(image_shape[0]), int(image_shape[1])
        left = float(crop_region.get('left', 0.0))
        top = float(crop_region.get('top', 0.0))
        right = float(crop_region.get('right', 1.0))
        bottom = float(crop_region.get('bottom', 1.0))
    except Exception:
        return None

    left = max(0.0, min(1.0, left))
    top = max(0.0, min(1.0, top))
    right = max(0.0, min(1.0, right))
    bottom = max(0.0, min(1.0, bottom))

    x1 = max(0, min(w, int(round(left * w))))
    y1 = max(0, min(h, int(round(top * h))))
    x2 = max(0, min(w, int(round(right * w))))
    y2 = max(0, min(h, int(round(bottom * h))))

    if x2 <= x1 + 10 or y2 <= y1 + 10:
        return None

    if x1 == 0 and y1 == 0 and x2 == w and y2 == h:
        return None

    return {
        'x1': int(x1),
        'y1': int(y1),
        'x2': int(x2),
        'y2': int(y2),
        'width': int(x2 - x1),
        'height': int(y2 - y1),
    }


def _offset_sample_metadata_to_source(sample_metadata, offset_x, offset_y, source_shape):
    """Shift sample metadata from crop-local coordinates back into full image space."""
    if sample_metadata is None:
        return {
            'sampled': False,
            'sample_shape': list(source_shape),
            'sample_origin': [0, 0],
            'source_shape': list(source_shape),
            'sampling_mode': 'full_image',
            'overlay_supported': True,
        }

    meta = dict(sample_metadata)
    sample_origin = meta.get('sample_origin', [0, 0])
    if isinstance(sample_origin, (list, tuple)) and len(sample_origin) >= 2:
        meta['sample_origin'] = [int(sample_origin[0]) + int(offset_x), int(sample_origin[1]) + int(offset_y)]

    tile_origins = meta.get('tile_origins', [])
    if isinstance(tile_origins, list):
        shifted = []
        for origin in tile_origins:
            if not isinstance(origin, (list, tuple)) or len(origin) < 2:
                continue
            shifted.append([int(origin[0]) + int(offset_x), int(origin[1]) + int(offset_y)])
        meta['tile_origins'] = shifted

    meta['source_shape'] = [int(source_shape[0]), int(source_shape[1])]
    return meta


def auto_detect_min_path_length(gray_image, dark_threshold, test_lengths=None, min_object_size=3, max_gap=25,
                                parameter_scale=1.0):
    """Automatically detect the best minimum path length by analyzing path distribution."""
    if test_lengths is None:
        test_lengths = [1, 3, 5, 8, 12, 16, 20]

    scaled_test_lengths = []
    seen_scaled_lengths = set()
    for logical_length in test_lengths:
        effective_length = scale_length_parameter(logical_length, parameter_scale=parameter_scale)
        if effective_length in seen_scaled_lengths:
            continue
        seen_scaled_lengths.add(effective_length)
        scaled_test_lengths.append((int(logical_length), int(effective_length)))

    fallback_logical_length = int(test_lengths[0]) if test_lengths else 3
    fallback_effective_length = scale_length_parameter(fallback_logical_length, parameter_scale=parameter_scale)

    print(f"Auto-detecting min path length using threshold {dark_threshold:.3f}...")

    try:
        initial_paths = extract_skeleton_paths(gray_image, dark_threshold, min_object_size=min_object_size)
        if len(initial_paths) == 0:
            return {
                'best_min_length': fallback_logical_length,
                'effective_best_min_length': fallback_effective_length,
                'best_score': 0,
                'recommendation': 'no paths found - try adjusting threshold first'
            }

        merged_paths = merge_nearby_paths(initial_paths, max_gap=max_gap)
    except Exception as e:
        return {
            'best_min_length': fallback_logical_length,
            'effective_best_min_length': fallback_effective_length,
            'best_score': 0,
            'recommendation': f'error in path extraction: {str(e)}'
        }

    path_lengths = [len(path) for path in merged_paths]
    path_lengths.sort(reverse=True)

    if len(path_lengths) == 0:
        return {
            'best_min_length': fallback_logical_length,
            'effective_best_min_length': fallback_effective_length,
            'best_score': 0,
            'recommendation': 'no merged paths found'
        }

    total_paths = len(path_lengths)
    median_length = path_lengths[len(path_lengths) // 2] if path_lengths else 0

    print(f"  Found {total_paths} merged paths, lengths: {min(path_lengths)}-{max(path_lengths)}, median: {median_length}")

    best_min_length = fallback_logical_length
    best_effective_min_length = fallback_effective_length
    best_score = 0
    length_results = []

    for logical_min_length, effective_min_length in scaled_test_lengths:
        valid_paths = [path for path in merged_paths if len(path) >= effective_min_length]

        if len(valid_paths) == 0:
            score = 0
        else:
            path_count = len(valid_paths)
            avg_length = sum(len(path) for path in valid_paths) / len(valid_paths)
            total_length = sum(len(path) for path in valid_paths)

            count_score = min(path_count / max(total_paths * 0.3, 1), 1.0)
            length_score = min(avg_length / max(median_length * 1.5, 10), 1.0)
            coverage_score = min(total_length / max(sum(path_lengths) * 0.7, 100), 1.0)

            penalty = 0.5 if path_count < max(total_paths * 0.1, 2) else 1.0
            score = (count_score * 0.4 + length_score * 0.3 + coverage_score * 0.3) * penalty

        length_results.append({
            'min_length': logical_min_length,
            'effective_min_length': effective_min_length,
            'valid_paths': len(valid_paths) if 'valid_paths' in locals() else 0,
            'score': score
        })

        if score > best_score:
            best_score = score
            best_min_length = logical_min_length
            best_effective_min_length = effective_min_length

        print(
            f"    Min length {logical_min_length} (effective {effective_min_length}): "
            f"{len(valid_paths) if 'valid_paths' in locals() else 0} paths, score: {score:.3f}"
        )

    recommendation = 'auto-detected' if best_score > 0.3 else ('low confidence - consider manual adjustment' if best_score > 0.1 else 'manual adjustment recommended')

    print(f"  Best min path length: {best_min_length} (score: {best_score:.3f})")

    return {
        'best_min_length': best_min_length,
        'effective_best_min_length': best_effective_min_length,
        'best_score': best_score,
        'all_results': length_results,
        'recommendation': recommendation,
        'path_stats': {
            'total_paths': total_paths,
            'median_length': median_length,
            'length_range': [min(path_lengths), max(path_lengths)] if path_lengths else [0, 0]
        }
    }


def auto_detect_dark_threshold(gray_image, sample_size=1000, threshold_range=(0.05, 0.8), num_thresholds=8,
                               min_object_size=3, max_gap=25, parameter_scale=1.0,
                               preview_max_dim=900, confidence_target=0.92,
                               should_continue=None, on_progress=None):
    """Automatically detect the best dark threshold on a reduced preview or tile mosaic."""
    started_at = time.perf_counter()
    height, width = gray_image.shape
    cancelled = False

    if should_continue is None:
        def should_continue():
            return True

    sample_metadata = {
        'sampled': False,
        'sample_shape': gray_image.shape,
        'sample_origin': [0, 0],
        'source_shape': gray_image.shape,
        'sampling_mode': 'full_image',
        'overlay_supported': True,
    }
    sample_image = gray_image
    if max(height, width) > int(max(1, preview_max_dim)):
        sample_image, sample_metadata = _build_random_tile_mosaic(gray_image)

    scale = 1.0
    preview_image = sample_image
    if max(sample_image.shape) > int(max(1, preview_max_dim)):
        scale = preview_max_dim / float(max(sample_image.shape))
        preview_size = (
            max(1, int(round(sample_image.shape[1] * scale))),
            max(1, int(round(sample_image.shape[0] * scale))),
        )
        resample_filter = Image.Resampling.BILINEAR if hasattr(Image, 'Resampling') else Image.BILINEAR
        preview_pil = Image.fromarray((sample_image * 255).astype(np.uint8)).resize(preview_size, resample_filter)
        preview_image = np.asarray(preview_pil).astype(np.float32) / 255.0

    preview_values = preview_image.reshape(-1)
    if preview_values.size > int(sample_size):
        rng_seed = int((height << 16) ^ width ^ int(preview_values.size)) & 0xFFFFFFFF
        rng = np.random.default_rng(rng_seed)
        preview_values = preview_values[rng.choice(preview_values.size, size=int(sample_size), replace=False)]

    effective_min_path_length = scale_length_parameter(3, parameter_scale=parameter_scale)
    preview_min_path_length = max(1, int(round(effective_min_path_length * scale)))
    preview_merge_gap = max(6, int(round(float(max_gap) * max(1.0, float(parameter_scale)) * scale)))
    threshold_min = float(min(threshold_range[0], threshold_range[1]))
    threshold_max = float(max(threshold_range[0], threshold_range[1]))
    max_threshold_evals = max(3, int(num_thresholds))

    best_result = None
    second_best_score = 0.0
    threshold_results = []
    tested_thresholds = []
    tested_threshold_keys = set()

    mean_intensity = float(np.mean(preview_values)) if preview_values.size else float(np.mean(preview_image))
    dark_ratio_high = float(np.mean(preview_image < 0.80))

    print(
        f"Auto-detecting dark threshold on {preview_image.shape[1]}x{preview_image.shape[0]} preview "
        f"({sample_metadata.get('sampling_mode', 'full_image')}, max {max_threshold_evals} thresholds)..."
    )

    def _threshold_key(value):
        return int(round(float(value) * 1_000_000))

    def _pick_next_threshold():
        mid = (threshold_min + threshold_max) * 0.5
        seed_thresholds = [mid, (threshold_min + mid) * 0.5, (mid + threshold_max) * 0.5]
        for candidate in seed_thresholds:
            if _threshold_key(candidate) not in tested_threshold_keys:
                return float(candidate)

        if best_result is not None and threshold_results:
            best_t = float(best_result['threshold'])
            sorted_results = sorted(threshold_results, key=lambda item: float(item.get('threshold', 0.0)))
            left_neighbor = None
            right_neighbor = None
            for item in sorted_results:
                threshold_value = float(item.get('threshold', 0.0))
                if threshold_value < best_t:
                    left_neighbor = threshold_value
                elif threshold_value > best_t and right_neighbor is None:
                    right_neighbor = threshold_value
                    break

            local_candidates = []
            if left_neighbor is not None:
                local_candidates.append((abs(best_t - left_neighbor), (best_t + left_neighbor) * 0.5))
            else:
                local_candidates.append((abs(best_t - threshold_min), (best_t + threshold_min) * 0.5))
            if right_neighbor is not None:
                local_candidates.append((abs(right_neighbor - best_t), (best_t + right_neighbor) * 0.5))
            else:
                local_candidates.append((abs(threshold_max - best_t), (best_t + threshold_max) * 0.5))

            local_candidates.sort(key=lambda item: item[0], reverse=True)
            for _, candidate in local_candidates:
                if _threshold_key(candidate) not in tested_threshold_keys:
                    return float(candidate)

        anchors = [threshold_min] + sorted(tested_thresholds) + [threshold_max]
        widest_gap = None
        widest_candidate = None
        for left, right in zip(anchors[:-1], anchors[1:]):
            if right <= left:
                continue
            candidate = (left + right) * 0.5
            if _threshold_key(candidate) in tested_threshold_keys:
                continue
            gap = right - left
            if widest_gap is None or gap > widest_gap:
                widest_gap = gap
                widest_candidate = candidate

        if widest_candidate is not None:
            return float(widest_candidate)
        return None

    def _current_confidence_score():
        if best_result is None or float(best_result.get('score', 0.0)) <= 0.0:
            return 0.0
        return max(
            0.0,
            min(1.0, (float(best_result['score']) - second_best_score) / max(float(best_result['score']), 1e-6)),
        )

    def _emit_progress(current_threshold=None, current_score=0.0):
        if on_progress is None:
            return
        on_progress({
            'elapsed_sec': float(time.perf_counter() - started_at),
            'thresholds_evaluated': int(len(threshold_results)),
            'thresholds_planned': int(max_threshold_evals),
            'current_threshold': float(current_threshold) if current_threshold is not None else None,
            'current_score': float(current_score),
            'best_threshold': float(best_result['threshold']) if best_result is not None else None,
            'best_score': float(best_result['score']) if best_result is not None else 0.0,
            'confidence_score': float(_current_confidence_score()),
            'preview_shape': preview_image.shape,
            'preview_scale': float(scale),
            'sample_metadata': sample_metadata,
            'cancelled': bool(cancelled),
        })

    while len(threshold_results) < max_threshold_evals:
        if not should_continue():
            cancelled = True
            break

        threshold = _pick_next_threshold()
        if threshold is None:
            break

        threshold_key = _threshold_key(threshold)
        tested_threshold_keys.add(threshold_key)
        tested_thresholds.append(float(threshold))

        initial_paths = []
        valid_paths = []
        dark_ratio = float(np.mean(preview_image < threshold)) if preview_image.size else 0.0
        score = 0.0

        try:
            initial_paths = extract_skeleton_paths(preview_image, threshold, min_object_size=min_object_size)
            if initial_paths:
                merged_paths = merge_nearby_paths(initial_paths, max_gap=preview_merge_gap)
                valid_paths = [path for path in merged_paths if len(path) >= preview_min_path_length]
                if valid_paths:
                    total_length = float(sum(len(path) for path in valid_paths))
                    path_count_score = min(len(valid_paths) / 10.0, 1.0)
                    length_score = min(total_length / max(250.0, float(preview_image.size) * 0.012), 1.0)
                    density_penalty = 1.0 - float(np.clip((dark_ratio - max(dark_ratio_high * 1.7, 0.72)) / 0.22, 0.0, 0.55))
                    sparse_bonus = 1.0 + float(np.clip((mean_intensity - 0.70) / 0.18, 0.0, 0.18))
                    score = path_count_score * length_score * density_penalty * sparse_bonus
                    if len(valid_paths) <= 1 and total_length < float(preview_min_path_length * 4):
                        score *= 0.5

            result = {
                'threshold': float(threshold),
                'score': float(score),
                'path_count': int(len(initial_paths)),
                'valid_count': int(len(valid_paths)),
                'dark_ratio': float(dark_ratio),
            }
            threshold_results.append(result)

            if best_result is None or score > best_result['score']:
                if best_result is not None:
                    second_best_score = max(second_best_score, float(best_result['score']))
                best_result = result
            else:
                second_best_score = max(second_best_score, float(score))

            print(f"  Threshold {threshold:.3f}: {len(valid_paths)} valid paths, score: {score:.3f}")
            _emit_progress(current_threshold=threshold, current_score=score)

            if best_result is not None and len(threshold_results) >= 4:
                confidence_score = _current_confidence_score()
                if confidence_score >= float(confidence_target):
                    break

            if len(threshold_results) >= 6:
                top_results = sorted(threshold_results, key=lambda item: float(item.get('score', 0.0)), reverse=True)[:3]
                if len(top_results) == 3 and float(top_results[0]['score']) > 0.0:
                    top_score = float(top_results[0]['score'])
                    third_score = float(top_results[-1]['score'])
                    top_thresholds = [float(item['threshold']) for item in top_results]
                    threshold_span = max(top_thresholds) - min(top_thresholds)
                    if third_score >= top_score * 0.96 and threshold_span <= max(0.08, (threshold_max - threshold_min) * 0.18):
                        break
        except Exception as e:
            print(f"  Threshold {threshold:.3f}: Failed - {e}")
            threshold_results.append({
                'threshold': float(threshold),
                'score': 0.0,
                'path_count': 0,
                'valid_count': 0,
                'dark_ratio': float(dark_ratio),
            })
            _emit_progress(current_threshold=threshold, current_score=0.0)

    if cancelled:
        return {
            'best_threshold': float(best_result['threshold']) if best_result is not None else float(threshold_min),
            'best_score': float(best_result['score']) if best_result is not None else 0.0,
            'all_results': threshold_results,
            'effective_min_path_length': int(effective_min_path_length),
            'recommendation': 'cancelled',
            'preview_shape': preview_image.shape,
            'preview_scale': float(scale),
            'thresholds_evaluated': int(len(threshold_results)),
            'elapsed_sec': float(time.perf_counter() - started_at),
            'sample_metadata': sample_metadata,
            'cancelled': True,
        }

    if best_result is None or best_result['score'] <= 0:
        return {
            'best_threshold': float(threshold_min),
            'best_score': 0.0,
            'all_results': threshold_results,
            'effective_min_path_length': int(effective_min_path_length),
            'recommendation': 'manual adjustment needed',
            'preview_shape': preview_image.shape,
            'preview_scale': float(scale),
            'thresholds_evaluated': int(len(threshold_results)),
            'elapsed_sec': float(time.perf_counter() - started_at),
            'sample_metadata': sample_metadata,
            'cancelled': False,
        }

    confidence_score = _current_confidence_score()
    if confidence_score >= 0.18:
        recommendation = 'high confidence'
    elif confidence_score >= 0.08:
        recommendation = 'moderate confidence'
    else:
        recommendation = 'low confidence - manual adjustment may still help'

    print(f"Best threshold: {best_result['threshold']:.3f} (score: {best_result['score']:.3f})")

    return {
        'best_threshold': float(best_result['threshold']),
        'best_score': float(best_result['score']),
        'all_results': threshold_results,
        'effective_min_path_length': int(effective_min_path_length),
        'recommendation': recommendation,
        'preview_shape': preview_image.shape,
        'preview_scale': float(scale),
        'thresholds_evaluated': int(len(threshold_results)),
        'elapsed_sec': float(time.perf_counter() - started_at),
        'sample_metadata': sample_metadata,
        'cancelled': False,
    }


def auto_tune_extraction_parameters(gray_image, threshold_range=(0.05, 0.8), num_thresholds=10,
                                    base_min_lengths=None, preview_max_dim=900,
                                    time_budget_sec=AUTO_TUNE_TIME_BUDGET_SEC,
                                    sample_metadata=None,
                                    should_continue=None,
                                    on_best_result=None,
                                    on_progress=None,
                                    confidence_target=0.95,
                                    min_object_size=3,
                                    parameter_scale=1.0):
    """Jointly tune threshold and min path length on a preview image for a strong first extraction."""
    if base_min_lengths is None:
        base_min_lengths = [1, 3, 5, 8, 12, 16, 20]

    started_at = time.perf_counter()
    timed_out = False
    cancelled = False
    high_confidence_reached = False

    if should_continue is None:
        def should_continue():
            return True

    height, width = gray_image.shape
    scale = 1.0
    preview_image = gray_image

    if max(height, width) > preview_max_dim:
        scale = preview_max_dim / float(max(height, width))
        preview_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        resample_filter = Image.Resampling.BILINEAR if hasattr(Image, 'Resampling') else Image.BILINEAR
        preview_pil = Image.fromarray((gray_image * 255).astype(np.uint8)).resize(preview_size, resample_filter)
        preview_image = np.asarray(preview_pil).astype(np.float32) / 255.0

    mean_intensity = float(np.mean(preview_image))
    dark_ratio_mid = float(np.mean(preview_image < 0.55))
    dark_ratio_high = float(np.mean(preview_image < 0.80))
    bright_score = float(np.clip((mean_intensity - 0.72) / 0.20, 0.0, 1.0))
    sparse_mid_score = float(np.clip((0.45 - dark_ratio_high) / 0.45, 0.0, 1.0))
    sparse_dark_score = float(np.clip((0.18 - dark_ratio_mid) / 0.18, 0.0, 1.0))
    line_art_bias_strength = float(np.clip(
        bright_score * 0.45 + sparse_mid_score * 0.35 + sparse_dark_score * 0.20,
        0.0,
        1.0
    ))

    candidate_lengths = []
    seen_candidate_keys = set()
    for logical_length in base_min_lengths:
        effective_full_length = scale_length_parameter(logical_length, parameter_scale=parameter_scale)
        preview_length = max(1, int(round(effective_full_length * scale)))
        candidate_key = (int(effective_full_length), int(preview_length))
        if candidate_key in seen_candidate_keys:
            continue
        seen_candidate_keys.add(candidate_key)
        candidate_lengths.append((int(logical_length), int(effective_full_length), int(preview_length)))

    threshold_min = float(min(threshold_range[0], threshold_range[1]))
    threshold_max = float(max(threshold_range[0], threshold_range[1]))
    max_threshold_evals = max(2, int(num_thresholds))
    merge_gap = max(6, int(round(25 * max(1.0, float(parameter_scale)) * scale)))

    best_result = None
    second_best_score = 0.0
    all_results = []

    best_paths_for_progress = []
    tested_thresholds = []
    tested_threshold_keys = set()
    overlay_supported = True
    if sample_metadata is not None:
        overlay_supported = bool(sample_metadata.get('overlay_supported', True))

    sampling_mode = 'full_image'
    tile_count = 1
    if sample_metadata is not None:
        sampling_mode = str(sample_metadata.get('sampling_mode', sampling_mode))
        tile_count = max(1, int(sample_metadata.get('tile_count', 1)))

    def _sampling_progress_fields():
        if sampling_mode == 'random_tiles_mosaic':
            return {
                'sampling_mode': sampling_mode,
                'current_tile_index': 0,
                'current_tile_total': int(tile_count),
            }
        return {
            'sampling_mode': sampling_mode,
            'current_tile_index': 1,
            'current_tile_total': 1,
        }

    def _emit_progress(phase='evaluating', threshold=None, threshold_index=None,
                       current_min_length=0, threshold_best=None,
                       current_paths_for_progress=None):
        if on_progress is None:
            return

        current_confidence = 0.0
        if best_result is not None:
            current_confidence = max(0.0, min(1.0, (best_result['score'] - second_best_score) / max(best_result['score'], 1e-6)))

        payload = {
            'elapsed_sec': float(time.perf_counter() - started_at),
            'timed_out': False,
            'cancelled': False,
            'high_confidence_reached': False,
            'best_threshold': float(best_result['threshold']) if best_result is not None else 0.20,
            'best_min_length': int(best_result['best_min_length']) if best_result is not None else 3,
            'effective_min_path_length': int(best_result['effective_min_path_length']) if best_result is not None else scale_length_parameter(3, parameter_scale=parameter_scale),
            'quality_score': float(best_result['score']) if best_result is not None else 0.0,
            'confidence_score': float(current_confidence),
            'sample_metadata': sample_metadata or {'sampled': False},
            'live_paths': list(best_paths_for_progress),
            'live_paths_current': list(current_paths_for_progress) if current_paths_for_progress is not None else [],
            'live_paths_current_threshold': float(threshold) if threshold is not None else 0.0,
            'live_paths_current_min_length': int(current_min_length if current_min_length else (best_result['best_min_length'] if best_result is not None else 3)),
            'live_paths_current_score': float(threshold_best.get('score', 0.0)) if threshold_best is not None else 0.0,
            'live_paths_frame_id': int(len(all_results)),
            'live_paths_scale': float(scale),
            'live_paths_sample_origin': list(sample_metadata.get('sample_origin', [0, 0])) if sample_metadata else [0, 0],
            'live_paths_source_shape': list(sample_metadata.get('source_shape', list(preview_image.shape))) if sample_metadata else list(preview_image.shape),
            'iterations_done': len(all_results),
            'iterations_total': int(max_threshold_evals),
            'current_phase': str(phase),
            'current_threshold_index': int(threshold_index) if threshold_index is not None else int(len(all_results) + 1),
        }
        payload.update(_sampling_progress_fields())
        on_progress(payload)

    def _encode_overlay_paths(source_paths, preview_min_length, max_paths=16):
        filtered_paths = [p for p in source_paths if len(p) >= int(preview_min_length)]
        ranked_paths = sorted(filtered_paths, key=len, reverse=True)[:max_paths]
        return [
            [[int(pt[0]), int(pt[1])] for pt in p[::max(1, len(p) // 25)]]
            for p in ranked_paths if len(p) >= 2
        ]

    def _threshold_key(value):
        return int(round(float(value) * 1_000_000))

    def _pick_next_threshold():
        mid = (threshold_min + threshold_max) * 0.5
        seed_thresholds = [mid, (threshold_min + mid) * 0.5, (mid + threshold_max) * 0.5]
        for candidate in seed_thresholds:
            if _threshold_key(candidate) not in tested_threshold_keys:
                return float(candidate)

        if best_result is not None and all_results:
            best_t = float(best_result['threshold'])
            sorted_results = sorted(all_results, key=lambda item: float(item.get('threshold', 0.0)))
            left_neighbor = None
            right_neighbor = None
            for item in sorted_results:
                threshold_value = float(item.get('threshold', 0.0))
                if threshold_value < best_t:
                    left_neighbor = threshold_value
                elif threshold_value > best_t and right_neighbor is None:
                    right_neighbor = threshold_value
                    break

            local_candidates = []
            if left_neighbor is not None:
                local_candidates.append((abs(best_t - left_neighbor), (best_t + left_neighbor) * 0.5))
            else:
                local_candidates.append((abs(best_t - threshold_min), (best_t + threshold_min) * 0.5))
            if right_neighbor is not None:
                local_candidates.append((abs(right_neighbor - best_t), (best_t + right_neighbor) * 0.5))
            else:
                local_candidates.append((abs(threshold_max - best_t), (best_t + threshold_max) * 0.5))

            local_candidates.sort(key=lambda item: item[0], reverse=True)
            for _, candidate in local_candidates:
                if _threshold_key(candidate) not in tested_threshold_keys:
                    return float(candidate)

        anchors = [threshold_min] + sorted(tested_thresholds) + [threshold_max]
        best_gap = 0.0
        best_candidate = None
        for idx in range(len(anchors) - 1):
            gap_start = float(anchors[idx])
            gap_end = float(anchors[idx + 1])
            gap = gap_end - gap_start
            if gap <= 1e-4:
                continue
            candidate = (gap_start + gap_end) * 0.5
            if _threshold_key(candidate) in tested_threshold_keys:
                continue
            if gap > best_gap:
                best_gap = gap
                best_candidate = candidate

        return float(best_candidate) if best_candidate is not None else None

    def _evaluate_threshold(threshold):
        nonlocal timed_out, cancelled
        try:
            threshold_index = int(len(all_results) + 1)
            _emit_progress(phase='extracting_paths', threshold=threshold, threshold_index=threshold_index, current_min_length=0)
            initial_paths = extract_skeleton_paths(preview_image, float(threshold), min_object_size=min_object_size)
            if not initial_paths:
                return ({
                    'threshold': float(threshold),
                    'best_min_length': 3,
                    'effective_min_path_length': scale_length_parameter(3, parameter_scale=parameter_scale),
                    'score': 0.0,
                    'valid_paths': 0,
                    'longest_path': 0,
                }, [])

            _emit_progress(phase='merging_paths', threshold=threshold, threshold_index=threshold_index, current_min_length=0)
            merged_paths = merge_nearby_paths(initial_paths, max_gap=merge_gap, verbose=False, should_continue=should_continue)
            threshold_best = None

            for logical_min_length, effective_full_length, preview_min_length in candidate_lengths:
                _emit_progress(phase='scoring_min_length', threshold=threshold, threshold_index=threshold_index, current_min_length=int(logical_min_length), threshold_best=threshold_best)
                if not should_continue():
                    cancelled = True
                    print('Auto-tune cancelled while evaluating candidates')
                    break
                if (time.perf_counter() - started_at) > time_budget_sec:
                    timed_out = True
                    print(f"Auto-tune timed out at {time_budget_sec:.1f}s while evaluating min lengths")
                    break

                valid_paths = [path for path in merged_paths if len(path) >= preview_min_length]

                if not valid_paths:
                    candidate = {
                        'threshold': float(threshold),
                        'best_min_length': int(logical_min_length),
                        'effective_min_path_length': int(effective_full_length),
                        'preview_min_length': int(preview_min_length),
                        'score': 0.0,
                        'valid_paths': 0,
                        'longest_path': 0,
                        'top_total_length': 0.0,
                        'median_top_length': 0.0
                    }
                else:
                    lengths = sorted((len(path) for path in valid_paths), reverse=True)
                    top_lengths = lengths[:min(6, len(lengths))]
                    longest_path = float(top_lengths[0])
                    top_total_length = float(sum(top_lengths))
                    median_top_length = float(np.median(top_lengths))
                    valid_count = len(valid_paths)

                    fragment_penalty = 1.0 / (1.0 + max(valid_count - 60, 0) / 60.0)
                    path_count_bonus = 1.0 + min(valid_count, 12) / 60.0
                    high_threshold_bonus = 1.0 + line_art_bias_strength * max(0.0, (float(threshold) - 0.55) / 0.25) * 0.50
                    low_threshold_penalty = 1.0 - line_art_bias_strength * max(0.0, (0.45 - float(threshold)) / 0.40) * 0.25
                    threshold_bias = max(0.50, high_threshold_bonus * low_threshold_penalty)
                    score = (
                        longest_path * 0.55 +
                        top_total_length * 0.30 +
                        median_top_length * 0.15
                    ) * fragment_penalty * path_count_bonus * threshold_bias

                    candidate = {
                        'threshold': float(threshold),
                        'best_min_length': int(logical_min_length),
                        'effective_min_path_length': int(effective_full_length),
                        'preview_min_length': int(preview_min_length),
                        'score': float(score),
                        'threshold_bias': float(threshold_bias),
                        'valid_paths': int(valid_count),
                        'longest_path': int(round(longest_path / max(scale, 1e-6))),
                        'top_total_length': float(top_total_length / max(scale, 1e-6)),
                        'median_top_length': float(median_top_length / max(scale, 1e-6))
                    }

                if threshold_best is None:
                    threshold_best = candidate
                else:
                    candidate_score = float(candidate.get('score', 0.0))
                    current_best_score = float(threshold_best.get('score', 0.0))
                    score_delta = candidate_score - current_best_score
                    if score_delta > 1e-6 or (
                        abs(score_delta) <= 1e-6 and
                        int(candidate.get('best_min_length', 0)) > int(threshold_best.get('best_min_length', 0))
                    ):
                        threshold_best = candidate

            if threshold_best is None:
                return ({
                    'threshold': float(threshold),
                    'best_min_length': 3,
                    'effective_min_path_length': scale_length_parameter(3, parameter_scale=parameter_scale),
                    'preview_min_length': 3,
                    'score': 0.0,
                    'valid_paths': 0,
                    'longest_path': 0,
                    'top_total_length': 0.0,
                    'median_top_length': 0.0
                }, merged_paths)

            return threshold_best, merged_paths
        except Exception as e:
            print(f"  Threshold {threshold:.3f}: auto-tune failed - {e}")
            return ({
                'threshold': float(threshold),
                'best_min_length': 3,
                'preview_min_length': 3,
                'score': 0.0,
                'valid_paths': 0,
                'longest_path': 0,
                'error': str(e)
            }, [])

    print(f"Auto-tuning extraction parameters on preview {preview_image.shape} (scale {scale:.3f})...")
    print(
        '  Line-art bias: '
        f"strength={line_art_bias_strength:.2f}, "
        f"mean={mean_intensity:.3f}, dark<0.80={dark_ratio_high:.3f}, dark<0.55={dark_ratio_mid:.3f}"
    )

    while len(all_results) < max_threshold_evals:
        threshold = _pick_next_threshold()
        if threshold is None:
            break

        threshold_key = _threshold_key(threshold)
        tested_threshold_keys.add(threshold_key)
        tested_thresholds.append(float(threshold))

        if not should_continue():
            cancelled = True
            print('Auto-tune cancelled by client request')
            break
        if (time.perf_counter() - started_at) > time_budget_sec:
            timed_out = True
            print(f"Auto-tune timed out at {time_budget_sec:.1f}s before threshold {threshold:.3f}")
            break
        threshold_best, merged_paths = _evaluate_threshold(threshold)
        all_results.append(threshold_best)

        current_paths_for_progress = []
        if overlay_supported:
            try:
                preview_min_length = int(threshold_best.get('preview_min_length', threshold_best['best_min_length']))
                current_paths_for_progress = _encode_overlay_paths(merged_paths, preview_min_length)
            except Exception:
                current_paths_for_progress = []

        if best_result is None or threshold_best['score'] > best_result['score']:
            if best_result is not None:
                second_best_score = best_result['score']
            best_result = threshold_best
            if overlay_supported:
                try:
                    best_paths_for_progress = list(current_paths_for_progress)
                except Exception:
                    pass
            else:
                best_paths_for_progress = []
            if on_best_result is not None:
                on_best_result({
                    'best_threshold': float(best_result['threshold']),
                    'best_min_length': int(best_result['best_min_length']),
                    'effective_min_path_length': int(best_result['effective_min_path_length']),
                    'quality_score': float(best_result['score']),
                    'confidence_score': float(max(0.0, min(1.0, (best_result['score'] - second_best_score) / max(best_result['score'], 1e-6)))),
                    'recommendation': 'partial best-so-far',
                    'preview_shape': preview_image.shape,
                    'preview_scale': float(scale),
                    'longest_path': int(best_result.get('longest_path', 0)),
                    'valid_paths': int(best_result.get('valid_paths', 0)),
                    'timed_out': False,
                    'cancelled': False,
                    'elapsed_sec': float(time.perf_counter() - started_at),
                    'sample_metadata': sample_metadata or {'sampled': False}
                })
        elif threshold_best['score'] > second_best_score:
            second_best_score = threshold_best['score']

        print(
            f"  Threshold {threshold:.3f}: best min length {threshold_best['best_min_length']}, "
            f"score {threshold_best['score']:.2f}, longest {threshold_best['longest_path']}"
        )

        current_confidence = 0.0
        if best_result is not None:
            current_confidence = max(0.0, min(1.0, (best_result['score'] - second_best_score) / max(best_result['score'], 1e-6)))

        if on_progress is not None:
            _emit_progress(
                phase='threshold_complete',
                threshold=threshold,
                threshold_index=len(all_results),
                current_min_length=int(threshold_best.get('best_min_length', 3)),
                threshold_best=threshold_best,
                current_paths_for_progress=current_paths_for_progress,
            )

        if len(all_results) >= 2 and current_confidence >= confidence_target:
            high_confidence_reached = True
            print(f"Auto-tune reached confidence target {confidence_target:.2f} at {current_confidence:.3f}; stopping early")
            break

        if cancelled or timed_out:
            break

    if best_result is None or best_result['score'] <= 0:
        return {
            'best_threshold': 0.20,
            'best_min_length': 3,
            'effective_min_path_length': scale_length_parameter(3, parameter_scale=parameter_scale),
            'quality_score': 0.0,
            'confidence_score': 0.0,
            'recommendation': 'failed to detect best settings',
            'preview_shape': preview_image.shape,
            'preview_scale': scale,
            'longest_path': 0,
            'valid_paths': 0,
            'all_results': all_results,
            'timed_out': timed_out,
            'cancelled': cancelled,
            'high_confidence_reached': high_confidence_reached,
            'elapsed_sec': float(time.perf_counter() - started_at),
            'sample_metadata': sample_metadata or {'sampled': False}
        }

    confidence_score = max(0.0, min(1.0, (best_result['score'] - second_best_score) / max(best_result['score'], 1e-6)))
    if confidence_score >= 0.18:
        recommendation = 'high confidence'
    elif confidence_score >= 0.08:
        recommendation = 'moderate confidence'
    else:
        recommendation = 'low confidence - manual adjustment may still help'

    return {
        'best_threshold': float(best_result['threshold']),
        'best_min_length': int(best_result['best_min_length']),
        'effective_min_path_length': int(best_result['effective_min_path_length']),
        'quality_score': float(best_result['score']),
        'confidence_score': float(confidence_score),
        'recommendation': recommendation,
        'preview_shape': preview_image.shape,
        'preview_scale': float(scale),
        'longest_path': int(best_result['longest_path']),
        'valid_paths': int(best_result['valid_paths']),
        'all_results': all_results,
        'timed_out': timed_out,
        'cancelled': cancelled,
        'high_confidence_reached': high_confidence_reached,
        'parameter_scale': float(max(1.0, parameter_scale)),
        'elapsed_sec': float(time.perf_counter() - started_at),
        'sample_metadata': sample_metadata or {'sampled': False}
    }


def _normalize_background_if_needed(gray_image, normalization_mode='auto', normalization_sensitivity='medium'):
    """Conditionally flatten uneven backgrounds while preserving dark strokes."""
    from skimage import exposure, filters

    requested_mode = str(normalization_mode or 'auto').strip().lower()
    if requested_mode not in ('auto', 'on', 'off'):
        requested_mode = 'auto'

    requested_sensitivity = str(normalization_sensitivity or 'medium').strip().lower()
    if requested_sensitivity not in ('low', 'medium', 'high'):
        requested_sensitivity = 'medium'

    sensitivity_scale = {'low': 1.20, 'medium': 1.00, 'high': 0.82}[requested_sensitivity]
    gray = np.clip(gray_image.astype(np.float32), 0.0, 1.0)
    background = filters.gaussian(gray, sigma=24.0, preserve_range=True)

    edge_strength = np.abs(filters.sobel(gray))
    bright_cutoff = float(np.quantile(gray, 0.70))
    edge_cutoff = float(np.quantile(edge_strength, 0.60))
    bright_mask = (gray >= bright_cutoff) & (edge_strength <= edge_cutoff)
    bright_background = background[bright_mask]

    if bright_background.size >= 500:
        bg_p05, bg_p95 = np.percentile(bright_background, [5, 95])
        bg_range = float(bg_p95 - bg_p05)
        bg_std = float(np.std(bright_background))
    else:
        bg_p05, bg_p95 = np.percentile(background, [5, 95])
        bg_range = float(bg_p95 - bg_p05)
        bg_std = float(np.std(background))

    illumination_gradient = float(np.mean(np.abs(filters.sobel(background))))
    dark_ratio = float(np.mean(gray < 0.65))
    paper_level = float(np.percentile(gray, 90))
    ink_candidates = gray[gray <= (paper_level - 0.08)]
    ink_level = float(np.percentile(ink_candidates, 25)) if ink_candidates.size >= 200 else float(np.percentile(gray, 2))
    ink_contrast = max(0.0, paper_level - ink_level)
    edge_peak = float(np.percentile(edge_strength, 99.5))
    edge_density = float(np.mean((edge_strength > max(edge_cutoff * 1.15, 0.018)) & (gray < min(0.97, paper_level + 0.06))))

    local_trend = filters.gaussian(gray, sigma=1.6, preserve_range=True)
    texture_residual = gray - local_trend
    bright_texture = texture_residual[bright_mask]
    background_texture = float(np.std(bright_texture)) if bright_texture.size >= 500 else float(np.std(texture_residual))

    range_score = float(np.clip((bg_range - 0.035) / 0.040, 0.0, 1.0))
    std_score = float(np.clip((bg_std - 0.008) / 0.020, 0.0, 1.0))
    gradient_score = float(np.clip((illumination_gradient - 0.004) / 0.015, 0.0, 1.0))
    normalization_score = 0.55 * range_score + 0.25 * std_score + 0.20 * gradient_score
    if dark_ratio > 0.62:
        normalization_score *= 0.70

    decision_score = float(np.round(normalization_score, 3))
    decision_threshold = 0.35 * sensitivity_scale
    bg_range_threshold = 0.075 * sensitivity_scale
    should_normalize = bool(decision_score >= decision_threshold or bg_range > bg_range_threshold)

    low_contrast_shaded = bool(ink_contrast < 0.24 and (bg_range > 0.040 or illumination_gradient > 0.0060 or background_texture > 0.0060))
    force_normalize = bool(low_contrast_shaded and (decision_score >= 0.28 or bg_range > 0.055 or background_texture > 0.0065))
    faint_sparse_ink = bool(dark_ratio < 0.030 and ink_contrast < 0.40 and bg_range > 0.030 and background_texture > 0.0010)
    faint_line_art = bool(
        paper_level > 0.84 and
        dark_ratio < 0.12 and
        ink_contrast < 0.34 and
        background_texture < 0.012 and
        bg_std < 0.035 and
        illumination_gradient < 0.014 and
        (edge_density > 0.0015 or edge_peak > 0.030)
    )
    faint_uniform_line_art = bool(
        paper_level > 0.92 and
        dark_ratio < 0.025 and
        ink_contrast < 0.46 and
        bg_range < 0.040 and
        bg_std < 0.012 and
        illumination_gradient < 0.0035 and
        background_texture < 0.0025 and
        (edge_density > 0.0005 or edge_peak > 0.014)
    )
    if faint_sparse_ink:
        force_normalize = True
        faint_line_art = True
    if faint_uniform_line_art:
        force_normalize = True
        faint_line_art = True
    if force_normalize:
        should_normalize = True

    high_contrast_clean_line_art = bool(ink_contrast >= 0.42 and dark_ratio < 0.30 and background_texture < 0.0058 and bg_std < 0.024 and illumination_gradient < 0.011)
    if high_contrast_clean_line_art and not force_normalize:
        should_normalize = False

    inflated_range_clean_art = bool(ink_contrast >= 0.70 and dark_ratio < 0.16 and illumination_gradient < 0.0032 and background_texture < 0.020)
    if inflated_range_clean_art and not force_normalize:
        should_normalize = False

    perfect_contrast_no_gradient = bool(ink_contrast >= 0.80 and illumination_gradient < 0.010)
    if perfect_contrast_no_gradient and not force_normalize:
        should_normalize = False

    line_art_guard = bool(
        dark_ratio < 0.22 and
        bg_std < 0.017 and
        illumination_gradient < 0.0085 and
        bg_range < 0.080 and
        background_texture < 0.0050 and
        (decision_score < 0.32 or ink_contrast >= 0.15)
    )
    if line_art_guard and not force_normalize and not faint_line_art:
        should_normalize = False

    if requested_mode == 'off':
        should_normalize = False
    elif requested_mode == 'on':
        should_normalize = True

    if requested_mode == 'off':
        reason = 'disabled by user'
    elif requested_mode == 'on':
        reason = 'forced by user'
    elif force_normalize:
        if faint_uniform_line_art:
            reason = 'faint line-art on uniform background'
        else:
            reason = 'faint sparse ink detail' if faint_sparse_ink else 'low-contrast ink with background shading'
    elif should_normalize and bg_range > bg_range_threshold:
        reason = 'strong background shading range'
    elif should_normalize:
        reason = 'combined background variance score'
    elif inflated_range_clean_art:
        reason = 'clean high-contrast drawing (range inflated by line structure)'
    elif high_contrast_clean_line_art:
        reason = 'high-contrast clean line-art'
    elif line_art_guard:
        reason = 'sparse line-art on uniform background'
    else:
        reason = 'background uniform enough'

    metadata = {
        'applied': should_normalize,
        'background_range': round(bg_range, 4),
        'background_std': round(bg_std, 4),
        'illumination_gradient': round(illumination_gradient, 4),
        'normalization_score': decision_score,
        'dark_ratio': round(dark_ratio, 4),
        'ink_contrast': round(ink_contrast, 4),
        'background_texture': round(background_texture, 4),
        'edge_peak': round(edge_peak, 4),
        'edge_density': round(edge_density, 5),
        'faint_line_art_detected': faint_line_art,
        'profile_name': 'faint_line_art' if faint_line_art else 'standard',
        'enhancement_applied': False,
        'reason': reason,
        'mode': 'none',
        'requested_mode': requested_mode,
        'requested_sensitivity': requested_sensitivity,
    }

    if should_normalize:
        safe_background = np.clip(background, 1e-3, 1.0)
        output_gray = (gray / safe_background) * float(np.mean(safe_background))
        output_gray = np.clip(output_gray, 0.0, 1.0)

        low, high = np.percentile(output_gray, [2, 98])
        if high - low > 1e-6:
            output_gray = exposure.rescale_intensity(output_gray, in_range=(low, high), out_range=(0.0, 1.0))

        metadata['mode'] = 'divide_and_rescale'
        metadata['post_low'] = round(float(low), 4)
        metadata['post_high'] = round(float(high), 4)
    else:
        output_gray = gray

    enhancement_allowed = requested_mode != 'off'
    if faint_line_art and enhancement_allowed:
        enhanced = _enhance_faint_line_art(output_gray)
        enhancement_delta = float(np.mean(np.abs(enhanced - output_gray)))
        if enhancement_delta > 1e-4:
            output_gray = enhanced
            metadata['enhancement_applied'] = True
            metadata['enhancement_delta'] = round(enhancement_delta, 5)
            metadata['mode'] = 'dark_stroke_enhanced' if metadata['mode'] == 'none' else f"{metadata['mode']}_plus_dark_stroke_enhancement"

    return np.clip(output_gray.astype(np.float32), 0.0, 1.0), metadata


def load_and_process_image(image_source, normalization_mode='auto', normalization_sensitivity='medium', source_name=None):
    """Load an image/PDF and return raw+extraction grayscale images in 0-1 range."""
    from skimage import color
    from PIL import Image as PILImage

    extension_source = source_name if source_name is not None else image_source
    extension = Path(str(extension_source)).suffix.lower()

    in_memory_bytes = None
    if isinstance(image_source, (bytes, bytearray, memoryview)):
        in_memory_bytes = bytes(image_source)

    def _open_pil_image():
        if in_memory_bytes is not None:
            return PILImage.open(BytesIO(in_memory_bytes))
        return PILImage.open(image_source)

    if extension == '.pdf':
        try:
            import importlib as _il
            pdfium = _il.import_module('pypdfium2')
        except Exception as exc:
            raise ValueError('PDF upload requires pypdfium2. Install dependencies and retry.') from exc
        pdf_path = None
        try:
            if in_memory_bytes is not None:
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
                    temp_pdf.write(in_memory_bytes)
                    pdf_path = temp_pdf.name
            else:
                pdf_path = image_source

            pdf = pdfium.PdfDocument(pdf_path)
            if len(pdf) == 0:
                raise ValueError('PDF has no pages')
            page = pdf[0]
            bitmap = page.render(scale=2.0)
            pil_page = bitmap.to_pil().convert('RGB')
            img = np.asarray(pil_page).astype(np.float32) / 255.0
            pdf.close()
            gray = color.rgb2gray(img)
        except Exception as exc:
            raise ValueError(f'Failed to render PDF: {exc}') from exc
        finally:
            if in_memory_bytes is not None and pdf_path and os.path.exists(pdf_path):
                os.remove(pdf_path)
    elif extension == '.png':
        with _open_pil_image() as pil_src:
            png_gamma_val = pil_src.info.get('gamma', None)
            print(f"[load] PNG mode={pil_src.mode} size={pil_src.size} gamma={png_gamma_val}")
            pil_rgba = pil_src.convert('RGBA')
            white_bg = PILImage.new('RGBA', pil_rgba.size, (255, 255, 255, 255))
            white_bg.alpha_composite(pil_rgba)
            rgb_arr = np.asarray(white_bg.convert('RGB')).astype(np.float32) / 255.0
            print(f"[load] after composite: min={rgb_arr.min():.4f} max={rgb_arr.max():.4f} mean={rgb_arr.mean():.4f}")

        if png_gamma_val is not None and 0.0 < png_gamma_val < 0.5:
            rgb_arr = np.clip(np.power(rgb_arr, png_gamma_val), 0.0, 1.0)
            print(f"[load] gamma {png_gamma_val:.5f} applied -> min={rgb_arr.min():.4f} max={rgb_arr.max():.4f}")

        gray = color.rgb2gray(rgb_arr)
    else:
        with _open_pil_image() as pil_src:
            print(f"[load] PIL mode={pil_src.mode} size={pil_src.size} format={pil_src.format}")

            if pil_src.mode in ('1', 'L', 'I', 'I;16', 'I;16B', 'I;16L', 'F'):
                raw_arr = np.asarray(pil_src)
                gray = raw_arr.astype(np.float32)
                if np.issubdtype(raw_arr.dtype, np.integer):
                    gray /= max(float(np.iinfo(raw_arr.dtype).max), 1.0)
                elif gray.max() > 1.0:
                    gray /= float(gray.max())
            else:
                pil_rgba = pil_src.convert('RGBA')
                white_bg = PILImage.new('RGBA', pil_rgba.size, (255, 255, 255, 255))
                white_bg.alpha_composite(pil_rgba)
                rgb_arr = np.asarray(white_bg.convert('RGB')).astype(np.float32) / 255.0
                gray = color.rgb2gray(rgb_arr)

    raw_gray = np.clip(gray.astype(np.float32), 0.0, 1.0)
    print(f"[load] raw_gray  min={raw_gray.min():.4f} max={raw_gray.max():.4f} mean={raw_gray.mean():.4f} std={raw_gray.std():.4f}")

    gray_range = float(raw_gray.max() - raw_gray.min())
    if gray_range < 0.05 and raw_gray.mean() < 0.1:
        p_lo = float(np.percentile(raw_gray, 1))
        p_hi = float(np.percentile(raw_gray, 99))
        if p_hi > p_lo:
            raw_gray = np.clip((raw_gray - p_lo) / (p_hi - p_lo), 0.0, 1.0)
            print(f"[load] near-black flat image (range={gray_range:.4f}); contrast stretched -> min={raw_gray.min():.4f} max={raw_gray.max():.4f}")

    extraction_gray, preprocessing_info = _normalize_background_if_needed(
        raw_gray,
        normalization_mode=normalization_mode,
        normalization_sensitivity=normalization_sensitivity,
    )
    return raw_gray, extraction_gray, preprocessing_info


def _extract_tile_preview_paths(
    gray_image,
    sample_metadata,
    dark_threshold,
    min_path_length,
    merge_gap=None,
    min_object_size=3,
    full_resolution=False,
):
    """Extract the kept detection paths per sampled tile with quality stats."""
    if gray_image is None or sample_metadata is None:
        return [], {}

    image_h, image_w = gray_image.shape
    sampling_mode = str(sample_metadata.get('sampling_mode', 'full_image'))
    tile_shape = sample_metadata.get('tile_shape') or [0, 0]
    tile_origins = sample_metadata.get('tile_origins') or []

    if sampling_mode == 'random_tiles_mosaic' and tile_origins:
        tile_h = int(tile_shape[0]) if len(tile_shape) >= 1 else 0
        tile_w = int(tile_shape[1]) if len(tile_shape) >= 2 else 0
        if tile_h <= 0 or tile_w <= 0:
            return [], {}
    else:
        tile_h, tile_w = int(image_h), int(image_w)
        tile_origins = [[0, 0]]

    out_tiles = []

    for tile_index, origin in enumerate(tile_origins, start=1):
        if not isinstance(origin, (list, tuple)) or len(origin) < 2:
            continue
        ox = int(origin[0])
        oy = int(origin[1])
        tile_entry = _extract_single_tile_preview_entry(
            gray_image,
            tile_index=tile_index,
            origin_x=ox,
            origin_y=oy,
            tile_width=tile_w,
            tile_height=tile_h,
            dark_threshold=dark_threshold,
            min_path_length=min_path_length,
            merge_gap=merge_gap,
            min_object_size=min_object_size,
            full_resolution=full_resolution,
        )
        if tile_entry is not None:
            out_tiles.append(tile_entry)

    total_valid = sum(tile['valid_count'] for tile in out_tiles)
    total_orphan = sum(tile['orphan_count'] for tile in out_tiles)
    total_length = sum(tile['total_length'] for tile in out_tiles)
    agg_stats = {
        'total_valid_paths': total_valid,
        'total_orphan_paths': total_orphan,
        'total_path_length': total_length,
        'tile_count': len(out_tiles),
        'noise_ratio': round(total_orphan / max(1, total_valid + total_orphan), 3),
    }

    return out_tiles, agg_stats


def _extract_single_tile_preview_entry(
    gray_image,
    *,
    tile_index,
    origin_x,
    origin_y,
    tile_width,
    tile_height,
    dark_threshold,
    min_path_length,
    merge_gap=None,
    min_object_size=3,
    full_resolution=False,
):
    """Extract a faithful kept-path payload for one sampled tile."""
    if gray_image is None:
        return None

    image_h, image_w = gray_image.shape
    ox = int(origin_x)
    oy = int(origin_y)
    tile_w = int(tile_width)
    tile_h = int(tile_height)
    if tile_w <= 0 or tile_h <= 0:
        return None
    if ox < 0 or oy < 0 or ox + tile_w > image_w or oy + tile_h > image_h:
        return None

    resolved_merge_gap = int(merge_gap) if merge_gap is not None else max(4, int(round(min(tile_h, tile_w) * 0.17)))
    resolved_min_object_size = max(1, int(min_object_size or 1))

    tile = gray_image[oy:oy + tile_h, ox:ox + tile_w]
    initial_paths = extract_skeleton_paths(tile, float(dark_threshold), min_object_size=resolved_min_object_size)
    if not initial_paths:
        return None

    tile_merge_gap = max(1, min(int(min(tile_h, tile_w)), int(resolved_merge_gap)))
    merged_paths = merge_nearby_paths(initial_paths, max_gap=tile_merge_gap, verbose=False)
    valid_paths = [path for path in merged_paths if len(path) >= int(min_path_length)]
    orphan_paths = [path for path in merged_paths if len(path) < int(min_path_length)]
    kept_paths = prune_extracted_paths(valid_paths, min_output_points=2)
    tile_total_length = sum(len(path) for path in kept_paths)
    tile_id = f"tile-{int(tile_index)}-{ox}-{oy}-{tile_w}-{tile_h}"

    tile_entry = {
        'tile_id': tile_id,
        'tile_index': int(tile_index),
        'valid_count': len(kept_paths),
        'orphan_count': len(orphan_paths),
        'total_length': tile_total_length,
        'paths': [],
    }

    for path_index, path in enumerate(kept_paths):
        encoded = []
        for pt in path:
            py = int(pt[0]) + oy
            px = int(pt[1]) + ox
            encoded.append([py, px])
        if len(encoded) >= 2:
            tile_entry['paths'].append({
                'path_id': f"{tile_id}-path-{int(path_index)}-{len(path)}-{encoded[0][0]}-{encoded[0][1]}-{encoded[-1][0]}-{encoded[-1][1]}",
                'length': int(len(path)),
                'points': encoded,
            })

    return tile_entry