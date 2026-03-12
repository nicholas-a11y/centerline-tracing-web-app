#!/usr/bin/env python3
"""
Web-based Centerline Extraction Tool
====================================

A Flask web application for interactive centerline extraction with real-time parameter adjustment.
"""

import threading
import time
from queue import Queue
from flask import Flask, render_template, request, jsonify, send_file
import os
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import tempfile
import uuid
from pathlib import Path
import importlib

# Import our existing centerline extraction functions
from worksok3_optimized import (
    CircleEvaluationSystem, extract_skeleton_paths, merge_nearby_paths,
    optimize_path_with_circles, create_svg_output, remove_overlapping_paths,
    optimize_path_with_custom_params
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global storage for session data
sessions = {}

AUTO_TUNE_TIME_BUDGET_SEC = 90.0
AUTO_TUNE_CONFIDENCE_TARGET = 0.95
AUTO_TUNE_RANDOM_TILE_DIM = 144
AUTO_TUNE_RANDOM_TILE_COUNT = 12

class CenterlineSession:
    def __init__(self, session_id):
        self.session_id = session_id
        self.image = None
        self.display_image = None
        self.image_path = None
        self.image_file_size = 0
        self.original_filename = None
        self.preprocessing_info = {}
        self.results = None
        self.initial_paths = None  # Store raw magenta paths
        self.progress_queue = Queue()  # For progress updates
        self.optimization_thread = None
        self.optimization_complete = False
        self.optimization_stopped = False
        self.optimization_generation = 0
        self.auto_tune_generation = 0
        self.auto_tune_active = False
        self.auto_tune_best = None
        self.auto_tune_crop_region = None
        self.auto_tune_thread = None
        self.auto_tune_progress = {
            'running': False,
            'started_at': 0.0,
            'elapsed_sec': 0.0,
            'confidence_score': 0.0,
            'quality_score': 0.0,
            'best_threshold': 0.20,
            'best_min_length': 3,
            'timed_out': False,
            'cancelled': False,
            'high_confidence_reached': False,
            'finished': False,
            'success': False,
            'message': 'idle',
            'sample_metadata': {'sampled': False},
        }
        self.partial_optimized_paths = []  # Store partially optimized paths
        self.lock = threading.Lock()  # Lock for thread-safe access to partial_optimized_paths
        self.parameters = {
            'dark_threshold': 0.20,
            'merge_gap': 25,  # Endpoint reach-out distance (pixels) for path merging
            'merge_angle_priority': 30.0,  # % weight for angle continuity vs distance
            'rdp_tolerance': 2.5,      # Balanced simplification for speed and smoothness
            'smoothing_factor': 0.006,  # Balanced smoothing that keeps runtime responsive
            'simplification_strength': 50.0,  # Moderate vertex reduction target
            'arc_fit_strength': 72.0,  # Favor curves without over-smoothing
            'line_fit_strength': 22.0,  # Moderate straight segment fitting
            'short_path_protection': 65.0,  # Preserve detail on shorter paths
            'mean_closeness_px': 1.8,  # Average allowed distance from blue path to magenta path
            'peak_closeness_px': 4.5,  # 95th percentile allowed distance from blue path to magenta path
            'score_preservation': 80.0,  # Minimum retained circle score percentage for accepted fits
            'min_path_length': 3,      # Increase to 8-15 for longer segments
            'enable_optimization': True,   # Enable path optimization and circle evaluation
            'show_pre_optimization': False,  # Show unoptimized paths in SVG
            'include_image': False,    # Include original image in SVG background
            'normalization_mode': 'auto',  # auto|on|off preprocessing normalization
            'normalization_sensitivity': 'medium',  # low|medium|high
        }


def _build_random_tile_mosaic(gray_image, tile_dim=AUTO_TUNE_RANDOM_TILE_DIM, tile_count=AUTO_TUNE_RANDOM_TILE_COUNT):
    """Create a multi-tile random mosaic for auto-tune scoring on large images."""
    height, width = gray_image.shape
    tile_h = min(int(tile_dim), int(height))
    tile_w = min(int(tile_dim), int(width))

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
    requested_count = max(2, int(tile_count))

    rng_seed = int((time.perf_counter_ns() ^ (height << 11) ^ (width << 3)) & 0xFFFFFFFF)
    rng = np.random.default_rng(rng_seed)

    tiles = []
    tile_origins = []
    used_origins = set()

    def _overlaps_existing(x, y):
        # Strict no-overlap policy for sampled tiles.
        for ox, oy in used_origins:
            separated = (
                (x + tile_w) <= ox or
                (ox + tile_w) <= x or
                (y + tile_h) <= oy or
                (oy + tile_h) <= y
            )
            if not separated:
                return True
        return False

    attempts = max(24, requested_count * 10)
    for _ in range(attempts):
        if len(tiles) >= requested_count:
            break
        y = int(rng.integers(0, max_y + 1)) if max_y > 0 else 0
        x = int(rng.integers(0, max_x + 1)) if max_x > 0 else 0
        key = (x, y)
        if key in used_origins:
            continue
        if _overlaps_existing(x, y):
            continue
        tile = gray_image[y:y + tile_h, x:x + tile_w]
        if tile.shape[0] != tile_h or tile.shape[1] != tile_w:
            continue
        used_origins.add(key)
        tiles.append(tile)
        tile_origins.append([int(x), int(y)])

    # Ensure deterministic coverage of corners/center if random picks were sparse.
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
        key = (int(x), int(y))
        if key in used_origins:
            continue
        if _overlaps_existing(int(x), int(y)):
            continue
        tile = gray_image[int(y):int(y) + tile_h, int(x):int(x) + tile_w]
        if tile.shape[0] != tile_h or tile.shape[1] != tile_w:
            continue
        used_origins.add(key)
        tiles.append(tile)
        tile_origins.append([int(x), int(y)])

    if not tiles:
        return gray_image, {
            'sampled': False,
            'sample_shape': gray_image.shape,
            'sample_origin': [0, 0],
            'source_shape': gray_image.shape,
            'sampling_mode': 'full_image_fallback',
            'overlay_supported': True,
        }

    # Arrange tiles in a near-square mosaic for joint threshold scoring.
    count = len(tiles)
    cols = int(np.ceil(np.sqrt(count)))
    rows = int(np.ceil(count / float(cols)))
    mosaic = np.ones((rows * tile_h, cols * tile_w), dtype=np.float32)

    for idx, tile in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        y0 = r * tile_h
        x0 = c * tile_w
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
        # Paths extracted from a mosaic do not map 1:1 to source coordinates.
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

    # Avoid tiny/invalid crops that would destabilize sampling.
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


def auto_detect_min_path_length(gray_image, dark_threshold, test_lengths=None):
    """
    Automatically detect the best minimum path length by analyzing path distribution.
    
    Args:
        gray_image: Grayscale image (0-1 range)
        dark_threshold: Dark threshold to use for path extraction
        test_lengths: List of path lengths to test (default: [1, 3, 5, 8, 12, 16, 20])
    
    Returns:
        Best min path length and analysis results
    """
    if test_lengths is None:
        test_lengths = [1, 3, 5, 8, 12, 16, 20]
    
    print(f"Auto-detecting min path length using threshold {dark_threshold:.3f}...")
    
    # Extract initial skeleton paths
    try:
        initial_paths = extract_skeleton_paths(gray_image, dark_threshold, min_object_size=3)
        if len(initial_paths) == 0:
            return {
                'best_min_length': 3,
                'best_score': 0,
                'recommendation': 'no paths found - try adjusting threshold first'
            }
        
        # Merge nearby paths
        merged_paths = merge_nearby_paths(initial_paths, max_gap=25)
        
    except Exception as e:
        return {
            'best_min_length': 3,
            'best_score': 0,
            'recommendation': f'error in path extraction: {str(e)}'
        }
    
    # Analyze path length distribution
    path_lengths = [len(path) for path in merged_paths]
    path_lengths.sort(reverse=True)  # Longest first
    
    if len(path_lengths) == 0:
        return {
            'best_min_length': 3,
            'best_score': 0,
            'recommendation': 'no merged paths found'
        }
    
    # Calculate statistics
    total_paths = len(path_lengths)
    median_length = path_lengths[len(path_lengths) // 2] if path_lengths else 0
    
    print(f"  Found {total_paths} merged paths, lengths: {min(path_lengths)}-{max(path_lengths)}, median: {median_length}")
    
    best_min_length = test_lengths[0]
    best_score = 0
    length_results = []
    
    for min_length in test_lengths:
        # Filter paths by this minimum length
        valid_paths = [path for path in merged_paths if len(path) >= min_length]
        
        if len(valid_paths) == 0:
            score = 0
        else:
            # Score based on:
            # 1. Reasonable number of paths (not too few, not too many)
            # 2. Good average length of remaining paths
            # 3. Good coverage (total length of all paths)
            
            path_count = len(valid_paths)
            avg_length = sum(len(path) for path in valid_paths) / len(valid_paths)
            total_length = sum(len(path) for path in valid_paths)
            
            # Normalize scores
            count_score = min(path_count / max(total_paths * 0.3, 1), 1.0)  # Prefer keeping ~30% of paths
            length_score = min(avg_length / max(median_length * 1.5, 10), 1.0)  # Prefer good average length
            coverage_score = min(total_length / max(sum(path_lengths) * 0.7, 100), 1.0)  # Prefer good coverage
            
            # Penalty for being too strict (keeping too few paths)
            if path_count < max(total_paths * 0.1, 2):  # Less than 10% or less than 2 paths
                penalty = 0.5
            else:
                penalty = 1.0
            
            score = (count_score * 0.4 + length_score * 0.3 + coverage_score * 0.3) * penalty
        
        length_results.append({
            'min_length': min_length,
            'valid_paths': len(valid_paths) if 'valid_paths' in locals() else 0,
            'score': score
        })
        
        if score > best_score:
            best_score = score
            best_min_length = min_length
        
        print(f"    Min length {min_length}: {len(valid_paths) if 'valid_paths' in locals() else 0} paths, score: {score:.3f}")
    
    # Generate recommendation
    if best_score > 0.3:
        recommendation = 'auto-detected'
    elif best_score > 0.1:
        recommendation = 'low confidence - consider manual adjustment'
    else:
        recommendation = 'manual adjustment recommended'
    
    print(f"  Best min path length: {best_min_length} (score: {best_score:.3f})")
    
    return {
        'best_min_length': best_min_length,
        'best_score': best_score,
        'all_results': length_results,
        'recommendation': recommendation,
        'path_stats': {
            'total_paths': total_paths,
            'median_length': median_length,
            'length_range': [min(path_lengths), max(path_lengths)] if path_lengths else [0, 0]
        }
    }

def auto_detect_dark_threshold(gray_image, sample_size=1000, threshold_range=(0.05, 0.8), num_thresholds=15):
    """
    Automatically detect the best dark threshold by sampling the image.
    
    Args:
        gray_image: Grayscale image (0-1 range)
        sample_size: Number of random pixels to sample for analysis
        threshold_range: (min, max) range of thresholds to test
        num_thresholds: Number of threshold values to test
    
    Returns:
        Best threshold value and analysis results
    """
    height, width = gray_image.shape
    
    # Sample random pixels from the image
    sample_indices = np.random.choice(height * width, min(sample_size, height * width), replace=False)
    sample_coords = [(idx // width, idx % width) for idx in sample_indices]
    sample_values = [gray_image[y, x] for y, x in sample_coords]
    
    # Test different threshold values
    thresholds = np.linspace(threshold_range[0], threshold_range[1], num_thresholds)
    best_threshold = threshold_range[0]
    best_score = 0
    threshold_results = []
    
    print(f"Auto-detecting dark threshold from {len(sample_values)} sample pixels...")
    
    for threshold in thresholds:
        try:
            # Extract skeleton paths with this threshold
            initial_paths = extract_skeleton_paths(gray_image, threshold, min_object_size=3)
            
            if len(initial_paths) == 0:
                score = 0
            else:
                # Merge nearby paths
                merged_paths = merge_nearby_paths(initial_paths, max_gap=25)
                
                # Filter by minimum length
                valid_paths = [path for path in merged_paths if len(path) >= 3]
                
                # Score based on number of valid paths and total path length
                if len(valid_paths) == 0:
                    score = 0
                else:
                    total_length = sum(len(path) for path in valid_paths)
                    # Favor moderate number of paths with good total length
                    # Penalize too few paths (missing details) or too many paths (noise)
                    path_count_score = min(len(valid_paths) / 10.0, 1.0)  # Normalize to 0-1
                    length_score = min(total_length / 1000.0, 1.0)  # Normalize to 0-1
                    score = path_count_score * length_score
            
            threshold_results.append({
                'threshold': threshold,
                'score': score,
                'path_count': len(initial_paths) if 'initial_paths' in locals() else 0,
                'valid_count': len(valid_paths) if 'valid_paths' in locals() else 0
            })
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
                
            print(f"  Threshold {threshold:.3f}: {len(initial_paths) if 'initial_paths' in locals() else 0} paths, score: {score:.3f}")
            
        except Exception as e:
            print(f"  Threshold {threshold:.3f}: Failed - {e}")
            threshold_results.append({
                'threshold': threshold,
                'score': 0,
                'path_count': 0,
                'valid_count': 0
            })
    
    print(f"Best threshold: {best_threshold:.3f} (score: {best_score:.3f})")
    
    return {
        'best_threshold': best_threshold,
        'best_score': best_score,
        'all_results': threshold_results,
        'recommendation': 'auto-detected' if best_score > 0 else 'manual adjustment needed'
    }

def auto_tune_extraction_parameters(gray_image, threshold_range=(0.05, 0.8), num_thresholds=10,
                                    base_min_lengths=None, preview_max_dim=900,
                                    time_budget_sec=AUTO_TUNE_TIME_BUDGET_SEC,
                                    sample_metadata=None,
                                    should_continue=None,
                                    on_best_result=None,
                                    on_progress=None,
                                    confidence_target=0.95):
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
        preview_size = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale)))
        )
        resample_filter = Image.Resampling.BILINEAR if hasattr(Image, 'Resampling') else Image.BILINEAR
        preview_pil = Image.fromarray((gray_image * 255).astype(np.uint8)).resize(preview_size, resample_filter)
        preview_image = np.asarray(preview_pil).astype(np.float32) / 255.0

    # Heuristic: mostly-white technical scans with sparse ink should prefer higher thresholds.
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
    seen_preview_lengths = set()
    for full_length in base_min_lengths:
        preview_length = max(1, int(round(full_length * scale)))
        if preview_length in seen_preview_lengths:
            continue
        seen_preview_lengths.add(preview_length)
        candidate_lengths.append((int(full_length), preview_length))

    threshold_min = float(min(threshold_range[0], threshold_range[1]))
    threshold_max = float(max(threshold_range[0], threshold_range[1]))
    max_threshold_evals = max(2, int(num_thresholds))
    merge_gap = max(6, int(round(25 * scale)))

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
                # 0 means "all sampled tiles are evaluated together".
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
            current_confidence = max(
                0.0,
                min(1.0, (best_result['score'] - second_best_score) / max(best_result['score'], 1e-6))
            )

        payload = {
            'elapsed_sec': float(time.perf_counter() - started_at),
            'timed_out': False,
            'cancelled': False,
            'high_confidence_reached': False,
            'best_threshold': float(best_result['threshold']) if best_result is not None else 0.20,
            'best_min_length': int(best_result['best_min_length']) if best_result is not None else 3,
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
        """Compact path payload for live overlay updates."""
        filtered_paths = [p for p in source_paths if len(p) >= int(preview_min_length)]
        ranked_paths = sorted(filtered_paths, key=len, reverse=True)[:max_paths]
        return [
            [[int(pt[0]), int(pt[1])] for pt in p[::max(1, len(p) // 25)]]
            for p in ranked_paths if len(p) >= 2
        ]

    def _threshold_key(value):
        return int(round(float(value) * 1_000_000))

    def _pick_next_threshold():
        """Pick the next threshold adaptively to reduce redundant evaluations."""
        mid = (threshold_min + threshold_max) * 0.5
        seed_thresholds = [
            mid,
            (threshold_min + mid) * 0.5,
            (mid + threshold_max) * 0.5,
        ]
        for candidate in seed_thresholds:
            if _threshold_key(candidate) not in tested_threshold_keys:
                return float(candidate)

        if best_result is not None and all_results:
            best_t = float(best_result['threshold'])
            sorted_results = sorted(all_results, key=lambda item: float(item.get('threshold', 0.0)))
            left_neighbor = None
            right_neighbor = None
            for item in sorted_results:
                t = float(item.get('threshold', 0.0))
                if t < best_t:
                    left_neighbor = t
                elif t > best_t and right_neighbor is None:
                    right_neighbor = t
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
            _emit_progress(
                phase='extracting_paths',
                threshold=threshold,
                threshold_index=threshold_index,
                current_min_length=0,
            )
            initial_paths = extract_skeleton_paths(preview_image, float(threshold), min_object_size=3)
            if not initial_paths:
                return ({
                    'threshold': float(threshold),
                    'best_min_length': 3,
                    'score': 0.0,
                    'valid_paths': 0,
                    'longest_path': 0
                }, [])

            _emit_progress(
                phase='merging_paths',
                threshold=threshold,
                threshold_index=threshold_index,
                current_min_length=0,
            )
            merged_paths = merge_nearby_paths(initial_paths, max_gap=merge_gap, verbose=False, should_continue=should_continue)
            threshold_best = None

            for full_min_length, preview_min_length in candidate_lengths:
                _emit_progress(
                    phase='scoring_min_length',
                    threshold=threshold,
                    threshold_index=threshold_index,
                    current_min_length=int(full_min_length),
                    threshold_best=threshold_best,
                )
                if not should_continue():
                    cancelled = True
                    print("Auto-tune cancelled while evaluating candidates")
                    break
                if (time.perf_counter() - started_at) > time_budget_sec:
                    timed_out = True
                    print(f"Auto-tune timed out at {time_budget_sec:.1f}s while evaluating min lengths")
                    break

                valid_paths = [path for path in merged_paths if len(path) >= preview_min_length]

                if not valid_paths:
                    candidate = {
                        'threshold': float(threshold),
                        'best_min_length': int(full_min_length),
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
                        'best_min_length': int(full_min_length),
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

                    # If scores are effectively tied, prefer the stricter minimum path length.
                    # This avoids defaulting to length 1 when several thresholds keep the same
                    # long paths and only differ in how aggressively they suppress short fragments.
                    if score_delta > 1e-6 or (
                        abs(score_delta) <= 1e-6 and
                        int(candidate.get('best_min_length', 0)) > int(threshold_best.get('best_min_length', 0))
                    ):
                        threshold_best = candidate

            if threshold_best is None:
                return ({
                    'threshold': float(threshold),
                    'best_min_length': 3,
                    'preview_min_length': 3,
                    'score': 0.0,
                    'valid_paths': 0,
                    'longest_path': 0,
                    'top_total_length': 0.0,
                    'median_top_length': 0.0
                }, merged_paths)

            return (threshold_best, merged_paths)
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
        "  Line-art bias: "
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
            print("Auto-tune cancelled by client request")
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
                _cpmin = int(threshold_best.get('preview_min_length', threshold_best['best_min_length']))
                current_paths_for_progress = _encode_overlay_paths(merged_paths, _cpmin)
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
            current_confidence = max(
                0.0,
                min(1.0, (best_result['score'] - second_best_score) / max(best_result['score'], 1e-6))
            )

        if on_progress is not None:
            _emit_progress(
                phase='threshold_complete',
                threshold=threshold,
                threshold_index=len(all_results),
                current_min_length=int(threshold_best.get('best_min_length', 3)),
                threshold_best=threshold_best,
                current_paths_for_progress=current_paths_for_progress,
            )

        # Only test the confidence target once we have at least two thresholds
        # evaluated — with only one result second_best_score is 0 and the
        # formula always yields 1.0, which would stop the search after a
        # single threshold and give meaningless "instant" results.
        if len(all_results) >= 2 and current_confidence >= confidence_target:
            high_confidence_reached = True
            print(
                f"Auto-tune reached confidence target {confidence_target:.2f} "
                f"at {current_confidence:.3f}; stopping early"
            )
            break

        if cancelled or timed_out:
                break

    if best_result is None or best_result['score'] <= 0:
        return {
            'best_threshold': 0.20,
            'best_min_length': 3,
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
        'elapsed_sec': float(time.perf_counter() - started_at),
        'sample_metadata': sample_metadata or {'sampled': False}
    }

def _normalize_background_if_needed(gray_image, normalization_mode='auto', normalization_sensitivity='medium'):
    """Conditionally flatten uneven backgrounds while preserving dark strokes.

    Returns:
        (processed_image, metadata)
    """
    from skimage import exposure, filters

    requested_mode = str(normalization_mode or 'auto').strip().lower()
    if requested_mode not in ('auto', 'on', 'off'):
        requested_mode = 'auto'

    requested_sensitivity = str(normalization_sensitivity or 'medium').strip().lower()
    if requested_sensitivity not in ('low', 'medium', 'high'):
        requested_sensitivity = 'medium'

    sensitivity_scale = {
        'low': 1.20,
        'medium': 1.00,
        'high': 0.82,
    }[requested_sensitivity]

    gray = np.clip(gray_image.astype(np.float32), 0.0, 1.0)

    # Estimate slow background shading using a large blur.
    background = filters.gaussian(gray, sigma=24.0, preserve_range=True)

    # Measure background metrics on bright, low-edge pixels.
    # Restricting to non-edge regions avoids dark strokes and anti-aliased
    # contours inflating shading estimates on already-clean line art.
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

    # Robust ink-vs-paper contrast estimate for sparse line drawings.
    # Percentile-15 can miss sparse ink entirely; use dark-pixel candidates
    # with a fallback to a low global percentile.
    paper_level = float(np.percentile(gray, 90))
    ink_candidates = gray[gray <= (paper_level - 0.08)]
    if ink_candidates.size >= 200:
        ink_level = float(np.percentile(ink_candidates, 25))
    else:
        ink_level = float(np.percentile(gray, 2))
    ink_contrast = max(0.0, paper_level - ink_level)

    # Estimate fine-grained background texture/noise in bright paper regions.
    # Phone-camera captures tend to have more texture than clean digital line art.
    local_trend = filters.gaussian(gray, sigma=1.6, preserve_range=True)
    texture_residual = gray - local_trend
    bright_texture = texture_residual[bright_mask]
    if bright_texture.size >= 500:
        background_texture = float(np.std(bright_texture))
    else:
        background_texture = float(np.std(texture_residual))

    # Weighted trigger to reduce false positives on already-uniform images.
    # Tuned for scanned drawings with mostly bright paper and sparse dark ink.
    range_score = float(np.clip((bg_range - 0.035) / 0.040, 0.0, 1.0))
    std_score = float(np.clip((bg_std - 0.008) / 0.020, 0.0, 1.0))
    gradient_score = float(np.clip((illumination_gradient - 0.004) / 0.015, 0.0, 1.0))
    normalization_score = 0.55 * range_score + 0.25 * std_score + 0.20 * gradient_score

    # Dense dark images are less likely to be paper-background sketches.
    if dark_ratio > 0.62:
        normalization_score *= 0.70

    # Use the same precision for decision + UI display to avoid confusing edge cases
    # where an internal value like 0.3496 is shown as 0.350 but does not trigger.
    decision_score = float(np.round(normalization_score, 3))

    decision_threshold = 0.35 * sensitivity_scale
    bg_range_threshold = 0.075 * sensitivity_scale
    should_normalize = bool(decision_score >= decision_threshold or bg_range > bg_range_threshold)

    # Low-contrast ink plus noticeable shading usually benefits from normalization,
    # even if the drawing is sparse.
    low_contrast_shaded = bool(
        ink_contrast < 0.24 and
        (bg_range > 0.040 or illumination_gradient > 0.0060 or background_texture > 0.0060)
    )
    force_normalize = bool(
        low_contrast_shaded and
        (decision_score >= 0.28 or bg_range > 0.055 or background_texture > 0.0065)
    )

    # Very sparse, faint ink can still benefit from normalization even when
    # global shading metrics look small. This captures low-coverage sketches
    # that otherwise get skipped by the sparse line-art guard.
    faint_sparse_ink = bool(
        dark_ratio < 0.030 and
        ink_contrast < 0.40 and
        bg_range > 0.030 and
        background_texture > 0.0010
    )
    if faint_sparse_ink:
        force_normalize = True

    if force_normalize:
        should_normalize = True

    # Explicit skip for crisp, high-contrast digital-style line art.
    # Mild global shading in the background should not force normalization here.
    high_contrast_clean_line_art = bool(
        ink_contrast >= 0.42 and
        dark_ratio < 0.30 and
        background_texture < 0.0058 and
        bg_std < 0.024 and
        illumination_gradient < 0.011
    )
    if high_contrast_clean_line_art and not force_normalize:
        should_normalize = False

    # Some clean vector-like drawings produce a large blurred "range" simply
    # because thick dark strokes span big regions, not because paper shading is
    # uneven. Detect that pattern and skip normalization.
    inflated_range_clean_art = bool(
        ink_contrast >= 0.70 and
        dark_ratio < 0.16 and
        illumination_gradient < 0.0032 and
        background_texture < 0.020
    )
    if inflated_range_clean_art and not force_normalize:
        should_normalize = False

    # Dense or bordered images (maps, technical drawings with thick frames) can produce
    # very high bg_range/bg_std purely from dark content mass being smeared by the large
    # Gaussian blur — not from actual background unevenness.  When ink contrast is very
    # high (paper vs. ink is clean) AND there is no real illumination gradient, the
    # inflated range/std values are artefacts of the content, not shading problems.
    # Normalizing such images would degrade already-perfect contrast.
    perfect_contrast_no_gradient = bool(
        ink_contrast >= 0.80 and
        illumination_gradient < 0.010
    )
    if perfect_contrast_no_gradient and not force_normalize:
        should_normalize = False

    # Safety guard: sparse line art on a truly uniform page should not be normalized.
    # In those cases, normalization can increase fragmentation and make extraction slower.
    line_art_guard = bool(
        dark_ratio < 0.22 and
        bg_std < 0.017 and
        illumination_gradient < 0.0085 and
        bg_range < 0.080 and
        background_texture < 0.0050 and
        (decision_score < 0.32 or ink_contrast >= 0.15)
    )
    if line_art_guard and not force_normalize:
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
        if faint_sparse_ink:
            reason = 'faint sparse ink detail'
        else:
            reason = 'low-contrast ink with background shading'
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
        'reason': reason,
        'mode': 'none'
    }
    metadata['requested_mode'] = requested_mode
    metadata['requested_sensitivity'] = requested_sensitivity

    if not should_normalize:
        return gray, metadata

    safe_background = np.clip(background, 1e-3, 1.0)
    normalized = (gray / safe_background) * float(np.mean(safe_background))
    normalized = np.clip(normalized, 0.0, 1.0)

    # Stretch contrast after flattening so a global threshold behaves more predictably.
    low, high = np.percentile(normalized, [2, 98])
    if high - low > 1e-6:
        normalized = exposure.rescale_intensity(normalized, in_range=(low, high), out_range=(0.0, 1.0))

    metadata['mode'] = 'divide_and_rescale'
    metadata['post_low'] = round(float(low), 4)
    metadata['post_high'] = round(float(high), 4)

    return np.clip(normalized.astype(np.float32), 0.0, 1.0), metadata


def load_and_process_image(image_path, normalization_mode='auto', normalization_sensitivity='medium'):
    """Load an image/PDF and return raw+extraction grayscale images in 0-1 range.

    - PDF files are rendered from page 1 using pypdfium2.
    - PNG files are loaded via Pillow, which correctly handles all transparency
      types (RGBA, LA, palette+tRNS, grayscale+tRNS).  A white background is
      always composited so transparent areas become white.  If the PNG gAMA chunk
      reports a gamma < 0.5 the correction curve is applied manually.
    - Other formats (jpg, tiff, bmp, …) use skimage with manual alpha handling.

    Returns:
        raw_gray: Grayscale image in [0, 1].
        extraction_gray: Normalised version used for path extraction.
        preprocessing_info: Metadata about normalisation decisions.
    """
    from skimage import color, io
    from PIL import Image as PILImage

    extension = Path(image_path).suffix.lower()

    if extension == '.pdf':
        try:
            import importlib as _il
            pdfium = _il.import_module('pypdfium2')
        except Exception as exc:
            raise ValueError(
                "PDF upload requires pypdfium2. Install dependencies and retry."
            ) from exc
        try:
            pdf = pdfium.PdfDocument(image_path)
            if len(pdf) == 0:
                raise ValueError("PDF has no pages")
            page = pdf[0]
            bitmap = page.render(scale=2.0)
            pil_page = bitmap.to_pil().convert('RGB')
            img = np.asarray(pil_page).astype(np.float32) / 255.0
            pdf.close()
            gray = color.rgb2gray(img)
        except Exception as exc:
            raise ValueError(f"Failed to render PDF: {exc}") from exc

    elif extension == '.png':
        with PILImage.open(image_path) as pil_src:
            png_gamma_val = pil_src.info.get('gamma', None)
            print(f"[load] PNG mode={pil_src.mode} size={pil_src.size} "
                  f"gamma={png_gamma_val}")

            # Composite over solid white so transparent/semi-transparent areas
            # become white rather than mapping to black or undefined values.
            pil_rgba = pil_src.convert('RGBA')
            white_bg = PILImage.new('RGBA', pil_rgba.size, (255, 255, 255, 255))
            white_bg.alpha_composite(pil_rgba)
            rgb_arr = np.asarray(white_bg.convert('RGB')).astype(np.float32) / 255.0
            print(f"[load] after composite: min={rgb_arr.min():.4f} "
                  f"max={rgb_arr.max():.4f} mean={rgb_arr.mean():.4f}")

        # Pillow stores gAMA info but does NOT apply the encode curve on load.
        # Apply it manually for dark-encoded images (gamma < 0.5).
        if png_gamma_val is not None and 0.0 < png_gamma_val < 0.5:
            rgb_arr = np.clip(np.power(rgb_arr, png_gamma_val), 0.0, 1.0)
            print(f"[load] gamma {png_gamma_val:.5f} applied -> "
                  f"min={rgb_arr.min():.4f} max={rgb_arr.max():.4f}")

        gray = color.rgb2gray(rgb_arr)

    else:
        img = io.imread(image_path)
        print(f"[load] skimage dtype={img.dtype} shape={img.shape} "
              f"min={int(np.min(img))} max={int(np.max(img))}")

        if len(img.shape) == 3:
            channels = img.shape[2]
            if channels == 4:
                rgb   = img[:, :, :3].astype(np.float32)
                alpha = img[:, :,  3].astype(np.float32)
                if rgb.max()   > 1.0: rgb   /= 255.0
                if alpha.max() > 1.0: alpha /= 255.0
                composited_rgb = rgb * alpha[:, :, None] + (1.0 - alpha[:, :, None])
                gray = color.rgb2gray(composited_rgb)
            elif channels == 3:
                rgb = img.astype(np.float32)
                if rgb.max() > 1.0: rgb /= 255.0
                gray = color.rgb2gray(rgb)
            else:
                raise ValueError(f"Unsupported image format with {channels} channels")
        elif len(img.shape) == 2:
            gray = img.astype(np.float32)
            if gray.max() > 1.0: gray /= 255.0
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")

    raw_gray = np.clip(gray.astype(np.float32), 0.0, 1.0)
    print(f"[load] raw_gray  min={raw_gray.min():.4f} max={raw_gray.max():.4f} "
          f"mean={raw_gray.mean():.4f} std={raw_gray.std():.4f}")

    # Last-resort contrast stretch: if the image arrived near-black with no
    # meaningful range, stretch whatever exists so structure becomes visible.
    gray_range = float(raw_gray.max() - raw_gray.min())
    if gray_range < 0.05 and raw_gray.mean() < 0.1:
        p_lo = float(np.percentile(raw_gray, 1))
        p_hi = float(np.percentile(raw_gray, 99))
        if p_hi > p_lo:
            raw_gray = np.clip((raw_gray - p_lo) / (p_hi - p_lo), 0.0, 1.0)
            print(f"[load] near-black flat image (range={gray_range:.4f}); "
                  f"contrast stretched -> min={raw_gray.min():.4f} max={raw_gray.max():.4f}")

    extraction_gray, preprocessing_info = _normalize_background_if_needed(
        raw_gray,
        normalization_mode=normalization_mode,
        normalization_sensitivity=normalization_sensitivity,
    )
    return raw_gray, extraction_gray, preprocessing_info


def process_centerlines(session):
    """Process centerlines with current parameters."""
    if session.image is None:
        return None
    
    params = session.parameters
    gray = session.image
    
    # Check if optimization is enabled to determine processing level
    optimization_enabled = params.get('enable_optimization', True)
    
    if optimization_enabled:
        # Full processing mode: use standard extraction
        initial_paths = extract_skeleton_paths(gray, params['dark_threshold'], min_object_size=5)
    else:
        # Fast mode: use the fast extraction function directly
        print("Fast extraction mode for unoptimized processing...")
        # Import the fast function from worksok3_optimized
        from worksok3_optimized import create_fast_paths
        import numpy as np
        from skimage import morphology
        
        # Quick binary conversion and skeletonization with minimal filtering
        binary = gray < params['dark_threshold']
        binary = morphology.remove_small_objects(binary, 2)  # Reduced for speed
        skeleton = morphology.skeletonize(binary)
        initial_paths = create_fast_paths(skeleton)
    
    if len(initial_paths) == 0:
        return {'error': 'No skeleton paths found. Try adjusting the dark threshold.'}
    
    if optimization_enabled:
        # Full processing mode: merge paths and handle overlaps
        merge_gap = max(1, int(params.get('merge_gap', 25)))
        merge_angle_priority = float(params.get('merge_angle_priority', 30.0)) / 100.0

        # Merge nearby paths
        merged_paths = merge_nearby_paths(
            initial_paths,
            max_gap=merge_gap,
            angle_priority=merge_angle_priority,
        )
        
        # Filter paths by minimum length
        valid_paths = [path for path in merged_paths if len(path) >= params['min_path_length']]
        
        # Remove overlapping paths if enabled
        if params.get('remove_overlaps', True):
            valid_paths = remove_overlapping_paths(
                valid_paths, 
                overlap_threshold=params.get('overlap_threshold', 0.3),
                min_distance=params.get('min_overlap_distance', 8)
            )
        
        merged_paths_count = len(merged_paths)
    else:
        # Raw mode: no merging, no overlap processing - just filter by length
        print("Raw skeleton mode: skipping path merging and overlap processing...")
        
        # Filter paths by minimum length only
        valid_paths = [path for path in initial_paths if len(path) >= params['min_path_length']]
        merged_paths_count = len(initial_paths)  # Use initial count since no merging
    
    if len(valid_paths) == 0:
        return {'error': 'No valid paths after filtering. Try reducing minimum path length.'}
    
    # Check if optimization is enabled
    if not optimization_enabled:
        # Fast processing mode - skip circle evaluation and optimization
        print(f"Fast processing mode: returning {len(valid_paths)} unoptimized paths...")
        
        # Return unoptimized paths directly
        results = {
            'initial_paths_count': len(initial_paths),
            'merged_paths_count': merged_paths_count,
            'valid_paths_count': len(valid_paths),
            'stats': {'total_paths_evaluated': len(valid_paths)},
            'best_score': 0,  # No scoring in fast mode
            'optimized_paths': valid_paths,  # Use original paths as "optimized"
            'optimized_scores': [1.0] * len(valid_paths),  # Dummy scores
            'pre_optimization_paths': valid_paths,
            'circle_system': None  # No circle system in fast mode
        }
        
        return results

    # Initialize circle evaluation system with fixed parameters
    circle_system = CircleEvaluationSystem(
        gray, 
        params['dark_threshold'],
        3,    # max_circle_radius (fixed)
        0.2,  # min_circle_radius (fixed)
        2.0   # circle_intersection_bonus (fixed)
    )
    
    # Evaluate all paths
    path_scores, stats = circle_system.evaluate_all_paths(valid_paths)
    
    if len(path_scores) == 0:
        return {'error': 'No path scores computed.'}
    
    # Optimize ALL valid paths instead of limiting to top N
    sorted_indices = np.argsort(path_scores)[::-1]  # Best first
    
    optimized_paths = []
    optimized_scores = []
    
    print(f"Optimizing all {len(valid_paths)} valid paths...")
    
    for i, idx in enumerate(sorted_indices):
        path = valid_paths[idx]
        original_score = path_scores[idx]
        
        print(f"  Path {i+1}/{len(valid_paths)}:")
        # Apply custom parameters for optimization
        optimized_path, optimized_score = optimize_path_with_custom_params(
            path, circle_system, params
        )
        
        optimized_paths.append(optimized_path)
        optimized_scores.append(optimized_score)
    
    # Get corresponding pre-optimization paths for visualization (in same order)
    top_pre_optimization_paths = [valid_paths[idx] for idx in sorted_indices]
    
    # Prepare results
    results = {
        'initial_paths_count': len(initial_paths),
        'merged_paths_count': len(merged_paths),
        'valid_paths_count': len(valid_paths),
        'stats': stats,
        'best_score': optimized_scores[0] if optimized_scores else 0,
        'optimized_paths': optimized_paths,
        'optimized_scores': optimized_scores,
        'pre_optimization_paths': top_pre_optimization_paths,
        'circle_system': circle_system
    }
    
    return results

def optimize_path_with_custom_params(path, circle_system, params, initial_score=None):
    """
    Optimize path with a conservative RDP pre-simplification followed by a high-quality spline fit.
    This approach prioritizes smoothness and fidelity over aggressive point reduction.
    """
    from worksok3_optimized import rdp_simplify, smooth_path_spline, fit_curve_to_path
    
    current_path = path
    best_score = float(initial_score) if initial_score is not None else circle_system.evaluate_path(current_path)
    
    print(f"    Optimizing path: {len(path)} points, initial score: {best_score:.2f}")
    
    # 1. Adaptive pre-simplification with curvature awareness
    import numpy as np

    def _remove_duplicate_points(points):
        """Drop consecutive duplicates so downstream math stays stable."""
        if not points:
            return points
        cleaned = [points[0]]
        for pt in points[1:]:
            if pt != cleaned[-1]:
                cleaned.append(pt)
        return cleaned

    def _resample_even_spacing(points):
        """Resample to even arc-length spacing to reduce pixel jitter."""
        if len(points) < 4:
            return points
        pts = np.asarray(points, dtype=float)
        seg = np.diff(pts, axis=0)
        seg_len = np.linalg.norm(seg, axis=1)
        total = seg_len.sum()
        if total == 0:
            return points
        cumulative = np.concatenate(([0.0], np.cumsum(seg_len)))
        targets = np.linspace(0.0, total, len(points))
        resampled = []
        j = 1
        for dist in targets:
            while j < len(cumulative) and cumulative[j] < dist:
                j += 1
            if j == len(cumulative):
                interp = pts[-1]
            else:
                span = max(cumulative[j] - cumulative[j-1], 1e-6)
                ratio = (dist - cumulative[j-1]) / span
                interp = pts[j-1] + ratio * (pts[j] - pts[j-1])
            resampled.append((float(interp[0]), float(interp[1])))
        rounded = [(int(round(p[0])), int(round(p[1]))) for p in resampled]
        return _remove_duplicate_points(rounded)

    def _reduce_vertices_evenly(points, target_count):
        """Keep endpoints while reducing control vertices for display simplicity."""
        if len(points) <= target_count:
            return points
        target_count = max(2, int(target_count))
        if target_count == 2:
            return [points[0], points[-1]]

        interior = points[1:-1]
        keep_interior = max(0, target_count - 2)
        if len(interior) <= keep_interior:
            return points

        indices = np.linspace(0, len(interior) - 1, keep_interior).astype(int)
        reduced = [points[0]] + [interior[i] for i in indices] + [points[-1]]
        return _remove_duplicate_points(reduced)

    def _point_to_segment_distance(point, start, end):
        point = np.asarray(point, dtype=float)
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        segment = end - start
        denom = float(np.dot(segment, segment))
        if denom <= 1e-9:
            return float(np.linalg.norm(point - start))
        t = float(np.dot(point - start, segment) / denom)
        t = max(0.0, min(1.0, t))
        projection = start + t * segment
        return float(np.linalg.norm(point - projection))

    def _reference_to_candidate_metrics(reference, candidate, max_samples=120):
        """Measure how closely the simplified blue path still follows the magenta source."""
        if len(reference) < 2 or len(candidate) < 2:
            return float('inf'), float('inf')

        stride = max(1, len(reference) // max_samples)
        sampled_reference = reference[::stride]
        distances = []

        for ref_point in sampled_reference:
            best_dist = float('inf')
            for idx in range(1, len(candidate)):
                dist = _point_to_segment_distance(ref_point, candidate[idx - 1], candidate[idx])
                if dist < best_dist:
                    best_dist = dist
            distances.append(best_dist)

        if not distances:
            return float('inf'), float('inf')

        return float(np.mean(distances)), float(np.percentile(distances, 95))

    requested_tolerance = float(params.get('rdp_tolerance', 2.5))
    smoothing_factor = float(params.get('smoothing_factor', 0.006))
    simplification_strength = max(0.0, min(100.0, float(params.get('simplification_strength', 50.0))))
    simplification_ratio = simplification_strength / 100.0
    arc_fit_strength = max(0.0, min(100.0, float(params.get('arc_fit_strength', 72.0))))
    line_fit_strength = max(0.0, min(100.0, float(params.get('line_fit_strength', 22.0))))
    short_path_protection = max(0.0, min(100.0, float(params.get('short_path_protection', 65.0))))
    arc_fit_ratio = arc_fit_strength / 100.0
    line_fit_ratio = line_fit_strength / 100.0
    short_path_protection_ratio = short_path_protection / 100.0
    mean_closeness_px = max(0.25, float(params.get('mean_closeness_px', 1.8)))
    peak_closeness_px = max(mean_closeness_px, float(params.get('peak_closeness_px', 4.5)))
    score_preservation_ratio = max(0.70, min(0.999, float(params.get('score_preservation', 80.0)) / 100.0))
    min_path_length = int(params.get('min_path_length', 3))
    reference_path = _remove_duplicate_points(path)
    path_length = max(1.0, float(params.get('path_length', len(reference_path))))
    max_path_length = max(path_length, float(params.get('max_path_length', path_length)))

    # Shorter paths should keep proportionally more vertices than long paths.
    relative_length = max(0.0, min(1.0, path_length / max_path_length))
    short_weight = (1.0 - relative_length) ** (0.6 + 0.9 * short_path_protection_ratio)
    min_short_factor = 0.85 - (0.65 * short_path_protection_ratio)
    min_short_factor = max(0.20, min(0.95, min_short_factor))
    length_protection = 1.0 - (1.0 - min_short_factor) * short_weight
    effective_simplification_ratio = simplification_ratio * length_protection

    def _curvature_profile(points):
        """Return mean turning angle (deg) and sharp-turn ratio for adaptive tolerance."""
        if len(points) < 3:
            return 0.0, 0.0
        pts = np.asarray(points, dtype=float)
        v1 = pts[1:-1] - pts[:-2]
        v2 = pts[2:] - pts[1:-1]
        norm1 = np.linalg.norm(v1, axis=1)
        norm2 = np.linalg.norm(v2, axis=1)
        valid = (norm1 > 1e-6) & (norm2 > 1e-6)
        if not np.any(valid):
            return 0.0, 0.0
        v1 = v1[valid]
        v2 = v2[valid]
        norm1 = norm1[valid]
        norm2 = norm2[valid]
        cosang = np.sum(v1 * v2, axis=1) / (norm1 * norm2)
        cosang = np.clip(cosang, -1.0, 1.0)
        angles = np.degrees(np.arccos(cosang))
        mean_angle = float(np.mean(angles)) if len(angles) else 0.0
        sharp_ratio = float(np.mean(angles > 35.0)) if len(angles) else 0.0
        return mean_angle, sharp_ratio

    cleaned_path = _remove_duplicate_points(current_path)
    even_path = _resample_even_spacing(cleaned_path)

    # Estimate geometric complexity to tailor tolerance
    diffs = np.diff(np.asarray(even_path, dtype=float), axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1) if len(diffs) else np.array([0.0])
    total_length = float(seg_lengths.sum())
    mean_angle, sharp_ratio = _curvature_profile(even_path)

    base_tolerance = requested_tolerance * (0.7 + min(total_length / 220.0, 0.7) + 0.35 * line_fit_ratio)
    if sharp_ratio > 0.25 or mean_angle > 25.0:
        base_tolerance *= 0.65  # protect detailed curves
    adaptive_tolerance = max(0.85, min(base_tolerance, requested_tolerance + 2.5))

    if line_fit_ratio > 0.05:
        try:
            simplified_path = rdp_simplify(even_path, adaptive_tolerance)
            if len(simplified_path) >= min_path_length:
                rdp_score = circle_system.evaluate_path(simplified_path)
                print(
                    f"      Adaptive RDP: {len(current_path)} -> {len(simplified_path)} points, "
                    f"score: {rdp_score:.2f}, tolerance: {adaptive_tolerance:.2f}, "
                    f"mean angle: {mean_angle:.1f}°, sharp ratio: {sharp_ratio:.2f}"
                )

                if rdp_score >= best_score * max(0.96, score_preservation_ratio + 0.02):
                    current_path = simplified_path
                    best_score = rdp_score
                    print("      ✓ Accepted adaptive pre-simplification.")
                else:
                    print("      ✗ Rejected RDP due to score drop.")
        except Exception as e:
            print(f"      RDP failed: {e}")
    
    # 2. Primary Optimization: Arc-heavy spline smoothing
    try:
        if len(current_path) >= 4:
            smoothed_path = smooth_path_spline(
                current_path,
                smoothing_factor * (1.0 + 3.0 * arc_fit_ratio),
            )
            if len(smoothed_path) >= min_path_length:
                smooth_score = circle_system.evaluate_path(smoothed_path)
                print(f"      Spline smooth: {len(current_path)} -> {len(smoothed_path)} points, score: {smooth_score:.2f}")

                if smooth_score > best_score * max(0.92, score_preservation_ratio):
                    current_path = smoothed_path
                    best_score = smooth_score
                    print("      ✓ Accepted high-quality spline fit.")
    except Exception as e:
        print(f"      Spline smoothing failed: {e}")

    def _consider_candidate(candidate, label):
        nonlocal current_path, best_score

        candidate = _remove_duplicate_points(candidate)
        if len(candidate) < min_path_length or len(candidate) >= len(current_path):
            return

        candidate_score = circle_system.evaluate_path(candidate)
        mean_dist, p95_dist = _reference_to_candidate_metrics(reference_path, candidate)
        max_mean_dist = mean_closeness_px
        max_p95_dist = peak_closeness_px
        min_score_ratio = score_preservation_ratio

        print(
            f"      {label}: {len(current_path)} -> {len(candidate)} points, "
            f"score: {candidate_score:.2f}, mean drift: {mean_dist:.2f}px, p95 drift: {p95_dist:.2f}px"
        )

        if candidate_score >= best_score * min_score_ratio and mean_dist <= max_mean_dist and p95_dist <= max_p95_dist:
            current_path = candidate
            best_score = candidate_score
            print(f"      ✓ Accepted {label.lower()}.")
        else:
            print(f"      ✗ Rejected {label.lower()} (too much drift or score loss).")

    if simplification_ratio > 0 and len(current_path) >= max(4, min_path_length + 1):
        target_count = max(
            min_path_length,
            int(round(len(current_path) * (1.0 - 0.78 * effective_simplification_ratio))),
        )

        try:
            curve_seed = smooth_path_spline(
                current_path,
                smoothing_factor * (1.0 + 4.2 * effective_simplification_ratio + 2.2 * arc_fit_ratio),
            )
            curve_fit_candidate = _reduce_vertices_evenly(curve_seed, target_count)
            _consider_candidate(curve_fit_candidate, "Arc-fit simplification")
        except Exception as e:
            print(f"      Arc-fit simplification failed: {e}")

        try:
            spline_curve_candidate = fit_curve_to_path(current_path, 'spline')
            spline_curve_candidate = _reduce_vertices_evenly(spline_curve_candidate, target_count)
            _consider_candidate(spline_curve_candidate, "Spline arc fit")
        except Exception as e:
            print(f"      Spline arc fit failed: {e}")

        try:
            hybrid_seed = smooth_path_spline(
                current_path,
                smoothing_factor * (1.0 + 2.2 * effective_simplification_ratio + 1.3 * arc_fit_ratio),
            )
            hybrid_candidate = rdp_simplify(
                hybrid_seed,
                max(0.85, adaptive_tolerance * (0.6 + 0.6 * effective_simplification_ratio + 0.45 * line_fit_ratio)),
            )
            hybrid_candidate = _reduce_vertices_evenly(hybrid_candidate, target_count)
            _consider_candidate(hybrid_candidate, "Arc-first hybrid fit")
        except Exception as e:
            print(f"      Arc-first hybrid fit failed: {e}")

        if line_fit_ratio > 0.05:
            try:
                line_fit_candidate = rdp_simplify(
                    current_path,
                    adaptive_tolerance * (0.75 + 2.0 * effective_simplification_ratio * line_fit_ratio),
                )
                _consider_candidate(line_fit_candidate, "Line-fit simplification")
            except Exception as e:
                print(f"      Line-fit simplification failed: {e}")
    
    # 3. Polynomial fitting is removed for consistency and speed.
    
    print(f"    Final: {len(current_path)} points, score: {best_score:.2f}")
    return current_path, best_score

def create_dense_spline(path, target_points, smoothing_factor=0.005):
    """Create a dense spline with specified number of points for better circle alignment."""
    if len(path) < 4:
        return path
    
    from scipy import interpolate
    import numpy as np
    
    points = np.array(path)
    y, x = points[:, 0], points[:, 1]
    
    try:
        # Fit spline with fine smoothing
        tck, _ = interpolate.splprep([x, y], s=len(x) * smoothing_factor)
        
        # Generate many smooth points for better circle alignment
        u_new = np.linspace(0, 1, target_points)
        x_new, y_new = interpolate.splev(u_new, tck)
        
        # Round to pixel coordinates
        result = [(round(y), round(x)) for y, x in zip(y_new, x_new)]
        
        # Remove consecutive duplicates
        filtered_result = [result[0]]
        for point in result[1:]:
            if point != filtered_result[-1]:
                filtered_result.append(point)
        
        return filtered_result
    except:
        return path

@app.route('/auto_detect_min_path_length', methods=['POST'])
def auto_detect_min_path_length_route():
    """Auto-detect the best minimum path length for the current session."""
    data = request.json
    session_id = data.get('session_id')
    
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session'})
    
    session = sessions[session_id]
    
    if session.image is None:
        return jsonify({'error': 'No image loaded'})
    
    try:
        # Use current dark threshold for detection
        current_threshold = session.parameters.get('dark_threshold', 0.20)
        
        # Run auto-detection
        detection_result = auto_detect_min_path_length(session.image, current_threshold)
        
        if detection_result['best_score'] > 0:
            # Update session parameters with detected min path length
            session.parameters['min_path_length'] = detection_result['best_min_length']
            
            return jsonify({
                'success': True,
                'detected_min_length': detection_result['best_min_length'],
                'confidence_score': detection_result['best_score'],
                'recommendation': detection_result['recommendation'],
                'path_stats': detection_result.get('path_stats', {}),
                'updated_parameters': session.parameters
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Could not detect suitable min path length. Manual adjustment recommended.',
                'detected_min_length': detection_result['best_min_length'],
                'recommendation': detection_result['recommendation']
            })
            
    except Exception as e:
        return jsonify({'error': f'Auto-detection error: {str(e)}'})

@app.route('/auto_detect_threshold', methods=['POST'])
def auto_detect_threshold():
    """Auto-detect the best dark threshold for the current session."""
    data = request.json
    session_id = data.get('session_id')
    
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session'})
    
    session = sessions[session_id]
    
    if session.image is None:
        return jsonify({'error': 'No image loaded'})
    
    try:
        # Run auto-detection
        detection_result = auto_detect_dark_threshold(session.image)
        
        if detection_result['best_score'] > 0:
            # Update session parameters with detected threshold
            session.parameters['dark_threshold'] = detection_result['best_threshold']
            
            return jsonify({
                'success': True,
                'detected_threshold': detection_result['best_threshold'],
                'confidence_score': detection_result['best_score'],
                'recommendation': detection_result['recommendation'],
                'updated_parameters': session.parameters
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Could not detect suitable threshold. Manual adjustment recommended.',
                'detected_threshold': detection_result['best_threshold'],
                'recommendation': detection_result['recommendation']
            })
            
    except Exception as e:
        return jsonify({'error': f'Auto-detection error: {str(e)}'})

@app.route('/auto_tune_extraction', methods=['POST'])
def auto_tune_extraction():
    """Jointly auto-tune threshold and minimum path length for a better first extraction."""
    data = request.json
    session_id = data.get('session_id')

    if session_id not in sessions:
        return jsonify({'error': 'Invalid session'})

    session = sessions[session_id]

    if session.image is None:
        return jsonify({'error': 'No image loaded'})

    try:
        with session.lock:
            session.auto_tune_generation += 1
            auto_tune_generation = session.auto_tune_generation
            session.auto_tune_active = True
            session.auto_tune_best = None

        def _should_continue_auto_tune():
            with session.lock:
                return session.auto_tune_generation == auto_tune_generation

        def _record_best_so_far(best_payload):
            with session.lock:
                if session.auto_tune_generation != auto_tune_generation:
                    return
                session.auto_tune_best = dict(best_payload)

        tuning_image = session.image
        sample_metadata = {
            'sampled': False,
            'sample_shape': session.image.shape,
            'sample_origin': [0, 0],
            'source_shape': session.image.shape,
            'sampling_mode': 'full_image',
            'overlay_supported': True,
        }
        image_height, image_width = session.image.shape
        if max(image_height, image_width) > AUTO_TUNE_RANDOM_TILE_DIM:
            _tile_count = 6 if session.image_file_size > 300 * 1024 else AUTO_TUNE_RANDOM_TILE_COUNT
            tuning_image, sample_metadata = _build_random_tile_mosaic(session.image, tile_count=_tile_count)

        tuning_result = auto_tune_extraction_parameters(
            tuning_image,
            threshold_range=(0.05, 0.8),
            num_thresholds=8,
            base_min_lengths=[1, 3, 5, 8, 12, 16],
            preview_max_dim=700,
            time_budget_sec=AUTO_TUNE_TIME_BUDGET_SEC,
            sample_metadata=sample_metadata,
            should_continue=_should_continue_auto_tune,
            on_best_result=_record_best_so_far,
        )

        with session.lock:
            if session.auto_tune_generation == auto_tune_generation:
                session.auto_tune_active = False
                session.auto_tune_best = dict(tuning_result)

        if tuning_result.get('timed_out'):
            print(
                f"Auto-tune timed out after {tuning_result.get('elapsed_sec', 0.0):.1f}s; "
                "falling back to defaults with optimization disabled"
            )

        if tuning_result['quality_score'] <= 0 or tuning_result.get('timed_out'):
            session.parameters['dark_threshold'] = 0.20
            session.parameters['min_path_length'] = 3
            session.parameters['enable_optimization'] = False
            session.parameters['show_pre_optimization'] = True

            return jsonify({
                'success': False,
                'error': 'Failed to detect best settings. Using defaults with optimization turned off.',
                'recommendation': tuning_result.get('recommendation', 'failed to detect best settings'),
                'timed_out': bool(tuning_result.get('timed_out', False)),
                'elapsed_sec': float(tuning_result.get('elapsed_sec', 0.0)),
                'sample_metadata': tuning_result.get('sample_metadata', sample_metadata),
                'updated_parameters': session.parameters,
            })

        session.parameters['dark_threshold'] = tuning_result['best_threshold']
        session.parameters['min_path_length'] = tuning_result['best_min_length']

        return jsonify({
            'success': True,
            'detected_threshold': tuning_result['best_threshold'],
            'detected_min_length': tuning_result['best_min_length'],
            'confidence_score': tuning_result['confidence_score'],
            'quality_score': tuning_result['quality_score'],
            'recommendation': tuning_result['recommendation'],
            'preview_shape': tuning_result['preview_shape'],
            'preview_scale': tuning_result['preview_scale'],
            'longest_path': tuning_result['longest_path'],
            'valid_paths': tuning_result['valid_paths'],
            'timed_out': bool(tuning_result.get('timed_out', False)),
            'elapsed_sec': float(tuning_result.get('elapsed_sec', 0.0)),
            'sample_metadata': tuning_result.get('sample_metadata', sample_metadata),
            'updated_parameters': session.parameters
        })
    except Exception as e:
        with session.lock:
            if session.auto_tune_generation == auto_tune_generation:
                session.auto_tune_active = False
        session.parameters['dark_threshold'] = 0.20
        session.parameters['min_path_length'] = 3
        session.parameters['enable_optimization'] = False
        session.parameters['show_pre_optimization'] = True
        return jsonify({
            'success': False,
            'error': f'Failed to detect best settings ({str(e)}). Using defaults with optimization turned off.',
            'timed_out': False,
            'updated_parameters': session.parameters,
        })


@app.route('/auto_tune_sample_region', methods=['POST'])
def auto_tune_sample_region():
    """Return auto-tune sampling metadata for UI preview."""
    data = request.json or {}
    session_id = data.get('session_id')

    if session_id not in sessions:
        return jsonify({'error': 'Invalid session'})

    session = sessions[session_id]

    if session.image is None:
        return jsonify({'error': 'No image loaded'})

    crop_bounds = _resolve_crop_bounds(session.image.shape, data.get('crop_region'))
    sample_source = session.image
    offset_x = 0
    offset_y = 0
    if crop_bounds is not None:
        sample_source = session.image[crop_bounds['y1']:crop_bounds['y2'], crop_bounds['x1']:crop_bounds['x2']]
        offset_x = int(crop_bounds['x1'])
        offset_y = int(crop_bounds['y1'])

    sample_metadata = {
        'sampled': bool(crop_bounds is not None),
        'sample_shape': list(sample_source.shape),
        'sample_origin': [int(offset_x), int(offset_y)],
        'source_shape': list(session.image.shape),
        'sampling_mode': 'full_image',
        'overlay_supported': True,
    }

    image_height, image_width = sample_source.shape
    if max(image_height, image_width) > AUTO_TUNE_RANDOM_TILE_DIM:
        _tile_count = 6 if session.image_file_size > 300 * 1024 else AUTO_TUNE_RANDOM_TILE_COUNT
        _, sample_metadata = _build_random_tile_mosaic(sample_source, tile_count=_tile_count)
        sample_metadata = _offset_sample_metadata_to_source(
            sample_metadata,
            offset_x=offset_x,
            offset_y=offset_y,
            source_shape=session.image.shape,
        )

    return jsonify({
        'success': True,
        'sample_metadata': sample_metadata,
    })


def _extract_tile_preview_paths(gray_image, sample_metadata, dark_threshold, min_path_length, per_tile_limit=3, full_resolution=False):
    """Extract path overlays per sampled tile with quality stats (coverage and noise ratio)."""
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
        # Full-image preview mode: evaluate one region covering the complete source.
        tile_h, tile_w = int(image_h), int(image_w)
        tile_origins = [[0, 0]]

    max_gap = max(4, int(round(min(tile_h, tile_w) * 0.17)))
    out_tiles = []

    for tile_index, origin in enumerate(tile_origins, start=1):
        if not isinstance(origin, (list, tuple)) or len(origin) < 2:
            continue
        ox = int(origin[0])
        oy = int(origin[1])
        if ox < 0 or oy < 0:
            continue
        if ox + tile_w > image_w or oy + tile_h > image_h:
            continue

        tile = gray_image[oy:oy + tile_h, ox:ox + tile_w]
        initial_paths = extract_skeleton_paths(tile, float(dark_threshold), min_object_size=3)
        if not initial_paths:
            continue

        merged_paths = merge_nearby_paths(initial_paths, max_gap=max_gap, verbose=False)
        valid_paths = [path for path in merged_paths if len(path) >= int(min_path_length)]
        orphan_paths = [path for path in merged_paths if len(path) < int(min_path_length)]
        tile_total_length = sum(len(p) for p in valid_paths)

        tile_entry = {
            'tile_index': int(tile_index),
            'valid_count': len(valid_paths),
            'orphan_count': len(orphan_paths),
            'total_length': tile_total_length,
            'paths': [],
        }

        for rank, path in enumerate(sorted(valid_paths, key=len, reverse=True)):
            # Full-image review mode should match final magenta path fidelity.
            stride = 1 if full_resolution else max(1, len(path) // 30)
            encoded = []
            for pt in path[::stride]:
                py = int(pt[0]) + oy
                px = int(pt[1]) + ox
                encoded.append([py, px])
            if len(encoded) >= 2:
                tile_entry['paths'].append({
                    'rank': rank,
                    'length': int(len(path)),
                    'points': encoded,
                })

        out_tiles.append(tile_entry)

    total_valid = sum(t['valid_count'] for t in out_tiles)
    total_orphan = sum(t['orphan_count'] for t in out_tiles)
    total_length = sum(t['total_length'] for t in out_tiles)
    agg_stats = {
        'total_valid_paths': total_valid,
        'total_orphan_paths': total_orphan,
        'total_path_length': total_length,
        'tile_count': len(out_tiles),
        'noise_ratio': round(total_orphan / max(1, total_valid + total_orphan), 3),
    }

    return out_tiles, agg_stats


@app.route('/auto_tune_tile_preview', methods=['POST'])
def auto_tune_tile_preview():
    """Return tile-only preview overlays for manual threshold/min-length tuning."""
    data = request.json or {}
    session_id = data.get('session_id')

    if session_id not in sessions:
        return jsonify({'error': 'Invalid session'})

    session = sessions[session_id]
    if session.image is None:
        return jsonify({'error': 'No image loaded'})

    try:
        dark_threshold = float(data.get('dark_threshold', session.parameters.get('dark_threshold', 0.20)))
        min_path_length = int(data.get('min_path_length', session.parameters.get('min_path_length', 3)))
    except Exception:
        return jsonify({'error': 'Invalid threshold or min path length'})

    sample_metadata = data.get('sample_metadata')
    if not isinstance(sample_metadata, dict):
        with session.lock:
            sample_metadata = dict(session.auto_tune_progress.get('sample_metadata', {'sampled': False}))

    crop_bounds = _resolve_crop_bounds(session.image.shape, data.get('crop_region'))

    if bool(data.get('full_image_preview', False)):
        if crop_bounds is not None:
            sample_metadata = {
                'sampled': True,
                'sample_shape': [int(crop_bounds['height']), int(crop_bounds['width'])],
                'sample_origin': [int(crop_bounds['x1']), int(crop_bounds['y1'])],
                'source_shape': list(session.image.shape),
                # Use one explicit tile so extraction stays inside crop bounds.
                'sampling_mode': 'random_tiles_mosaic',
                'overlay_supported': True,
                'tile_count': 1,
                'tile_shape': [int(crop_bounds['height']), int(crop_bounds['width'])],
                'tile_origins': [[int(crop_bounds['x1']), int(crop_bounds['y1'])]],
            }
        else:
            sample_metadata = {
                'sampled': False,
                'sample_shape': list(session.image.shape),
                'sample_origin': [0, 0],
                'source_shape': list(session.image.shape),
                'sampling_mode': 'full_image',
                'overlay_supported': True,
                'tile_count': 1,
                'tile_shape': list(session.image.shape),
                'tile_origins': [[0, 0]],
            }

    preview_started_at = time.perf_counter()
    full_image_preview = bool(data.get('full_image_preview', False))
    tile_data, stats = _extract_tile_preview_paths(
        session.image,
        sample_metadata,
        dark_threshold=dark_threshold,
        min_path_length=min_path_length,
        full_resolution=full_image_preview,
    )
    preview_runtime_ms = int(round((time.perf_counter() - preview_started_at) * 1000.0))

    return jsonify({
        'success': True,
        'sample_metadata': sample_metadata,
        'tile_data': tile_data,
        'tile_path_count': sum(len(t.get('paths', [])) for t in tile_data),
        'stats': stats,
        'preview_runtime_ms': preview_runtime_ms,
        'source_shape': list(session.image.shape),
        'dark_threshold': float(dark_threshold),
        'min_path_length': int(min_path_length),
    })


def _run_auto_tune_job(session, auto_tune_generation):
    """Background auto-tune job that continuously updates session.auto_tune_progress."""
    try:
        source_image = session.image
        crop_bounds = _resolve_crop_bounds(source_image.shape, session.auto_tune_crop_region)
        tuning_image = source_image
        offset_x = 0
        offset_y = 0
        if crop_bounds is not None:
            tuning_image = source_image[crop_bounds['y1']:crop_bounds['y2'], crop_bounds['x1']:crop_bounds['x2']]
            offset_x = int(crop_bounds['x1'])
            offset_y = int(crop_bounds['y1'])

        sample_metadata = {
            'sampled': bool(crop_bounds is not None),
            'sample_shape': list(tuning_image.shape),
            'sample_origin': [int(offset_x), int(offset_y)],
            'source_shape': list(source_image.shape),
            'sampling_mode': 'full_image',
            'overlay_supported': True,
        }
        image_height, image_width = tuning_image.shape
        if max(image_height, image_width) > AUTO_TUNE_RANDOM_TILE_DIM:
            _tile_count = 6 if session.image_file_size > 300 * 1024 else AUTO_TUNE_RANDOM_TILE_COUNT
            tuning_image, sample_metadata = _build_random_tile_mosaic(tuning_image, tile_count=_tile_count)
            sample_metadata = _offset_sample_metadata_to_source(
                sample_metadata,
                offset_x=offset_x,
                offset_y=offset_y,
                source_shape=source_image.shape,
            )

        def _should_continue_auto_tune():
            with session.lock:
                return session.auto_tune_generation == auto_tune_generation

        def _record_best_so_far(best_payload):
            with session.lock:
                if session.auto_tune_generation != auto_tune_generation:
                    return
                session.auto_tune_best = dict(best_payload)

        def _record_progress(progress_payload):
            with session.lock:
                if session.auto_tune_generation != auto_tune_generation:
                    return
                session.auto_tune_progress.update({
                    'running': True,
                    'finished': False,
                    'elapsed_sec': float(progress_payload.get('elapsed_sec', 0.0)),
                    'confidence_score': float(progress_payload.get('confidence_score', 0.0)),
                    'quality_score': float(progress_payload.get('quality_score', 0.0)),
                    'best_threshold': float(progress_payload.get('best_threshold', 0.20)),
                    'best_min_length': int(progress_payload.get('best_min_length', 3)),
                    'timed_out': bool(progress_payload.get('timed_out', False)),
                    'cancelled': bool(progress_payload.get('cancelled', False)),
                    'high_confidence_reached': bool(progress_payload.get('high_confidence_reached', False)),
                    'sample_metadata': progress_payload.get('sample_metadata', sample_metadata),
                    'message': (
                        f"Live best: threshold {float(progress_payload.get('best_threshold', 0.20)):.3f}, "
                        f"min length {int(progress_payload.get('best_min_length', 3))}, "
                        f"confidence {float(progress_payload.get('confidence_score', 0.0)) * 100.0:.1f}%"
                    ),
                    'live_paths': progress_payload.get('live_paths', []),
                    'live_paths_current': progress_payload.get('live_paths_current', []),
                    'live_paths_current_threshold': float(progress_payload.get('live_paths_current_threshold', 0.0)),
                    'live_paths_current_min_length': int(progress_payload.get('live_paths_current_min_length', 3)),
                    'live_paths_current_score': float(progress_payload.get('live_paths_current_score', 0.0)),
                    'live_paths_frame_id': int(progress_payload.get('live_paths_frame_id', 0)),
                    'live_paths_scale': float(progress_payload.get('live_paths_scale', 1.0)),
                    'live_paths_sample_origin': progress_payload.get('live_paths_sample_origin', [0, 0]),
                    'live_paths_source_shape': progress_payload.get('live_paths_source_shape', []),
                    'iterations_done': int(progress_payload.get('iterations_done', 0)),
                    'iterations_total': int(progress_payload.get('iterations_total', 6)),
                    'current_phase': str(progress_payload.get('current_phase', 'evaluating')),
                    'current_threshold_index': int(progress_payload.get('current_threshold_index', 1)),
                    'sampling_mode': str(progress_payload.get('sampling_mode', 'full_image')),
                    'current_tile_index': int(progress_payload.get('current_tile_index', 1)),
                    'current_tile_total': int(progress_payload.get('current_tile_total', 1)),
                })

        tuning_result = auto_tune_extraction_parameters(
            tuning_image,
            threshold_range=(0.05, 0.8),
            num_thresholds=6,
            base_min_lengths=[1, 3, 5, 8, 12],
            preview_max_dim=550,
            time_budget_sec=AUTO_TUNE_TIME_BUDGET_SEC,
            sample_metadata=sample_metadata,
            should_continue=_should_continue_auto_tune,
            on_best_result=_record_best_so_far,
            on_progress=_record_progress,
            confidence_target=AUTO_TUNE_CONFIDENCE_TARGET,
        )

        with session.lock:
            if session.auto_tune_generation != auto_tune_generation:
                return
            session.auto_tune_active = False
            session.auto_tune_best = dict(tuning_result)

        if tuning_result['quality_score'] <= 0 or tuning_result.get('timed_out'):
            session.parameters['dark_threshold'] = 0.20
            session.parameters['min_path_length'] = 3
            session.parameters['enable_optimization'] = False
            session.parameters['show_pre_optimization'] = True
            success = False
            status_message = 'Failed to detect best settings. Using defaults with optimization turned off.'
        else:
            session.parameters['dark_threshold'] = tuning_result['best_threshold']
            session.parameters['min_path_length'] = tuning_result['best_min_length']
            success = True
            if tuning_result.get('high_confidence_reached'):
                status_message = 'Reached 95% confidence early.'
            else:
                status_message = 'Auto-tune completed within time budget.'

        with session.lock:
            if session.auto_tune_generation != auto_tune_generation:
                return
            session.auto_tune_progress.update({
                'running': False,
                'finished': True,
                'success': bool(success),
                'elapsed_sec': float(tuning_result.get('elapsed_sec', 0.0)),
                'confidence_score': float(tuning_result.get('confidence_score', 0.0)),
                'quality_score': float(tuning_result.get('quality_score', 0.0)),
                'best_threshold': float(tuning_result.get('best_threshold', session.parameters['dark_threshold'])),
                'best_min_length': int(tuning_result.get('best_min_length', session.parameters['min_path_length'])),
                'timed_out': bool(tuning_result.get('timed_out', False)),
                'cancelled': bool(tuning_result.get('cancelled', False)),
                'high_confidence_reached': bool(tuning_result.get('high_confidence_reached', False)),
                'sample_metadata': tuning_result.get('sample_metadata', sample_metadata),
                'message': status_message,
                'updated_parameters': dict(session.parameters),
            })
    except Exception as e:
        with session.lock:
            if session.auto_tune_generation != auto_tune_generation:
                return
            session.auto_tune_active = False
            session.parameters['dark_threshold'] = 0.20
            session.parameters['min_path_length'] = 3
            session.parameters['enable_optimization'] = False
            session.parameters['show_pre_optimization'] = True
            session.auto_tune_progress.update({
                'running': False,
                'finished': True,
                'success': False,
                'timed_out': False,
                'cancelled': False,
                'high_confidence_reached': False,
                'message': f'Auto-tune failed: {str(e)}. Using defaults with optimization turned off.',
                'updated_parameters': dict(session.parameters),
            })


@app.route('/auto_tune_extraction_start', methods=['POST'])
def auto_tune_extraction_start():
    """Start background auto-tune and return immediately for realtime polling."""
    data = request.json or {}
    session_id = data.get('session_id')

    if session_id not in sessions:
        return jsonify({'error': 'Invalid session'})

    session = sessions[session_id]

    if session.image is None:
        return jsonify({'error': 'No image loaded'})

    with session.lock:
        session.auto_tune_generation += 1
        auto_tune_generation = session.auto_tune_generation
        session.auto_tune_active = True
        session.auto_tune_best = None
        session.auto_tune_crop_region = data.get('crop_region') if isinstance(data.get('crop_region'), dict) else None
        session.auto_tune_progress = {
            'running': True,
            'started_at': float(time.perf_counter()),
            'finished': False,
            'success': False,
            'elapsed_sec': 0.0,
            'confidence_score': 0.0,
            'quality_score': 0.0,
            'best_threshold': float(session.parameters.get('dark_threshold', 0.20)),
            'best_min_length': int(session.parameters.get('min_path_length', 3)),
            'timed_out': False,
            'cancelled': False,
            'high_confidence_reached': False,
            'message': 'Auto-tune started. Gathering candidates...',
            'sample_metadata': {'sampled': False},
            'live_paths': [],
            'live_paths_current': [],
            'live_paths_current_threshold': 0.0,
            'live_paths_current_min_length': int(session.parameters.get('min_path_length', 3)),
            'live_paths_current_score': 0.0,
            'live_paths_frame_id': 0,
            'live_paths_scale': 1.0,
            'live_paths_sample_origin': [0, 0],
            'live_paths_source_shape': [],
            'iterations_done': 0,
            'iterations_total': 6,
            'current_phase': 'starting',
            'current_threshold_index': 1,
            'sampling_mode': 'full_image',
            'current_tile_index': 1,
            'current_tile_total': 1,
            'updated_parameters': dict(session.parameters),
        }

        worker = threading.Thread(
            target=_run_auto_tune_job,
            args=(session, auto_tune_generation),
            daemon=True,
        )
        session.auto_tune_thread = worker
        worker.start()

    return jsonify({'success': True, 'started': True})


@app.route('/auto_tune_progress/<session_id>', methods=['GET'])
def auto_tune_progress(session_id):
    """Return live auto-tune progress and best-so-far candidate."""
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session'})

    session = sessions[session_id]
    with session.lock:
        progress = dict(session.auto_tune_progress)
        progress['active'] = bool(session.auto_tune_active)

    if progress.get('running'):
        started_at = float(progress.get('started_at', 0.0) or 0.0)
        if started_at > 0.0:
            progress['elapsed_sec'] = max(float(progress.get('elapsed_sec', 0.0)), float(time.perf_counter() - started_at))

    return jsonify(progress)

@app.route('/')
def index():
    """Main page with the interface."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    # Create session
    session_id = str(uuid.uuid4())
    session = CenterlineSession(session_id)

    requested_mode = request.form.get('normalization_mode', 'auto')
    requested_sensitivity = request.form.get('normalization_sensitivity', 'medium')
    session.parameters['normalization_mode'] = requested_mode
    session.parameters['normalization_sensitivity'] = requested_sensitivity
    
    # Store original filename
    session.original_filename = file.filename
    
    # Save uploaded file
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"centerline_{session_id}_{file.filename}")
    file.save(temp_path)
    session.image_file_size = os.path.getsize(temp_path)
    
    try:
        # Load and preprocess image for extraction, while keeping original for display.
        raw_gray, extraction_gray, preprocessing_info = load_and_process_image(
            temp_path,
            normalization_mode=requested_mode,
            normalization_sensitivity=requested_sensitivity,
        )
        session.image = extraction_gray
        session.display_image = raw_gray
        session.preprocessing_info = preprocessing_info
        session.image_path = temp_path
        
        # Store session
        sessions[session_id] = session
        
        # Convert raw and normalized images to base64 for UI preview toggling.
        raw_img = Image.fromarray((raw_gray * 255).astype(np.uint8))
        raw_buf = BytesIO()
        raw_img.save(raw_buf, format='PNG')
        raw_buf.seek(0)
        raw_img_data = base64.b64encode(raw_buf.read()).decode('utf-8')

        normalized_img = Image.fromarray((extraction_gray * 255).astype(np.uint8))
        normalized_buf = BytesIO()
        normalized_img.save(normalized_buf, format='PNG')
        normalized_buf.seek(0)
        normalized_img_data = base64.b64encode(normalized_buf.read()).decode('utf-8')

        normalization_applied = bool(preprocessing_info.get('applied', False))
        default_img_data = normalized_img_data if normalization_applied else raw_img_data
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'image_data': default_img_data,
            'raw_image_data': raw_img_data,
            'normalized_image_data': normalized_img_data,
            'normalization_applied': normalization_applied,
            'image_shape': raw_gray.shape,
            'parameters': session.parameters,
            'preprocessing': preprocessing_info
        })
        
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'})

@app.route('/process_immediate', methods=['POST'])
def process_immediate():
    """Immediately extract and display raw magenta paths, then start optimization in background."""
    data = request.json
    session_id = data.get('session_id')
    
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session'})
    
    session = sessions[session_id]
    
    # Update parameters
    if 'parameters' in data:
        session.parameters.update(data['parameters'])
    
    try:
        # Quick raw path extraction
        params = session.parameters
        gray = session.image

        # Apply optional crop region (fractions 0-1 of image dimensions)
        crop_offset_row = 0
        crop_offset_col = 0
        crop = data.get('crop_region')
        if crop and isinstance(crop, dict):
            h, w = gray.shape[:2]
            y1 = max(0, int(round(float(crop.get('top', 0)) * h)))
            y2 = min(h, int(round(float(crop.get('bottom', 1)) * h)))
            x1 = max(0, int(round(float(crop.get('left', 0)) * w)))
            x2 = min(w, int(round(float(crop.get('right', 1)) * w)))
            if y2 > y1 + 10 and x2 > x1 + 10:
                gray = gray[y1:y2, x1:x2]
                crop_offset_row = y1
                crop_offset_col = x1

        # Always use fast extraction for immediate display
        print("Immediate extraction mode...")
        from worksok3_optimized import create_fast_paths
        import numpy as np
        from skimage import morphology
        
        # Quick binary conversion and skeletonization
        binary = gray < params['dark_threshold']
        binary = morphology.remove_small_objects(binary, 2)
        skeleton = morphology.skeletonize(binary)
        initial_paths = create_fast_paths(skeleton)
        
        if len(initial_paths) == 0:
            return jsonify({'error': 'No skeleton paths found. Try adjusting the dark threshold.'})
        
        merge_gap = max(1, int(params.get('merge_gap', 25)))
        merge_angle_priority = float(params.get('merge_angle_priority', 30.0)) / 100.0

        # Preserve legacy default behavior in progressive mode unless users
        # intentionally change merge controls from defaults.
        # If either merge_gap or merge_angle_priority changes, apply merging.
        merged_paths = initial_paths
        merge_applied = False
        angle_priority_changed = abs(merge_angle_priority - 0.30) > 1e-9
        merge_controls_changed = (merge_gap != 25) or angle_priority_changed
        if params.get('enable_optimization', True) and merge_controls_changed:
            merged_paths = merge_nearby_paths(
                initial_paths,
                max_gap=merge_gap,
                angle_priority=merge_angle_priority,
            )
            merge_applied = True

        # Filter by minimum length
        valid_paths = [path for path in merged_paths if len(path) >= params['min_path_length']]
        
        if len(valid_paths) == 0:
            return jsonify({'error': 'No valid paths after filtering. Try reducing minimum path length.'})
        
        # Convert numpy coordinates to regular Python lists for JSON serialization.
        # If a crop was applied, offset coordinates back to full-image space so
        # the frontend can render paths correctly over the original image.
        json_serializable_paths = []
        for path in valid_paths:
            serializable_path = [
                [int(point[0]) + crop_offset_row, int(point[1]) + crop_offset_col]
                for point in path
            ]
            json_serializable_paths.append(serializable_path)

        # Store paths in full-image coordinates for subsequent SVG generation and optimization.
        if crop_offset_row != 0 or crop_offset_col != 0:
            session.initial_paths = [
                [[pt[0] + crop_offset_row, pt[1] + crop_offset_col] for pt in path]
                for path in valid_paths
            ]
        else:
            session.initial_paths = valid_paths  # Keep original for processing
        session.partial_optimized_paths = []
        session.optimization_complete = False
        session.optimization_stopped = False
        session.optimization_generation += 1
        current_generation = session.optimization_generation
        
        # Clear progress queue
        while not session.progress_queue.empty():
            session.progress_queue.get()
        
        # Create immediate results for magenta display
        immediate_results = {
            'initial_paths_count': len(initial_paths),
            'merged_paths_count': len(merged_paths),
            'merge_applied': merge_applied,
            'valid_paths_count': len(valid_paths),
            'paths': json_serializable_paths,  # Use JSON-serializable version
            'optimization_started': False
        }
        
        # Start optimization in background if enabled
        if params.get('enable_optimization', True):
            session.optimization_thread = threading.Thread(
                target=background_optimization,
                args=(session, current_generation)
            )
            session.optimization_thread.start()
            immediate_results['optimization_started'] = True
        
        return jsonify({
            'success': True,
            'immediate_display': True,
            **immediate_results
        })
        
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'})

def background_optimization(session, generation):
    """Run optimization in background thread."""
    try:
        start_time = time.perf_counter()

        if generation != session.optimization_generation:
            return

        params = session.parameters
        valid_paths = session.initial_paths
        original_total_segments = sum(max(len(path) - 1, 0) for path in valid_paths)
        optimized_total_segments = 0
        
        session.progress_queue.put("Starting optimization process...")
        
        # Initialize circle evaluation system
        circle_system = CircleEvaluationSystem(
            session.image, 
            params['dark_threshold'],
            3,    # max_circle_radius
            0.2,  # min_circle_radius
            2.0   # circle_intersection_bonus
        )
        
        session.progress_queue.put("Circle evaluation system initialized...")

        # Avoid a full upfront scoring pass so the first path starts sooner.
        # Longer paths tend to matter most visually, so prioritize them first.
        sorted_indices = sorted(
            range(len(valid_paths)),
            key=lambda idx: len(valid_paths[idx]),
            reverse=True,
        )
        max_path_length = max((len(p) for p in valid_paths), default=1)
        session.progress_queue.put("Prioritizing longer paths first...")
        session.progress_queue.put(f"Optimizing {len(valid_paths)} paths...")
        
        # Optimize paths one by one with progress updates
        for i, idx in enumerate(sorted_indices):
            if session.optimization_stopped or generation != session.optimization_generation:
                session.progress_queue.put("Optimization stopped by user")
                break
                
            path = valid_paths[idx]
            original_score = circle_system.evaluate_path(path)
            
            session.progress_queue.put(f"Optimizing path {i+1}/{len(valid_paths)} ({len(path)} points, score: {original_score:.3f})")
            
            # Apply optimization using current UI parameters, including guarded simplification.
            balanced_params = {
                'rdp_tolerance': params.get('rdp_tolerance', 2.5),
                'smoothing_factor': params.get('smoothing_factor', 0.006),
                'simplification_strength': params.get('simplification_strength', 50.0),
                'arc_fit_strength': params.get('arc_fit_strength', 72.0),
                'line_fit_strength': params.get('line_fit_strength', 22.0),
                'short_path_protection': params.get('short_path_protection', 65.0),
                'mean_closeness_px': params.get('mean_closeness_px', 1.8),
                'peak_closeness_px': params.get('peak_closeness_px', 4.5),
                'score_preservation': params.get('score_preservation', 80.0),
                'path_length': len(path),
                'max_path_length': max_path_length,
                'min_path_length': params.get('min_path_length', 3)
            }
            
            optimized_path, optimized_score = optimize_path_with_custom_params(
                path, circle_system, balanced_params, initial_score=original_score
            )

            if session.optimization_stopped or generation != session.optimization_generation:
                session.progress_queue.put("Optimization stopped by user")
                break
            
            # Add to partial results - always add, even if not dramatically different
            with session.lock:
                if generation != session.optimization_generation:
                    break
                session.partial_optimized_paths.append(optimized_path)

            optimized_total_segments += max(len(optimized_path) - 1, 0)
            
            # Show optimization results with dramatic reduction emphasis
            if len(optimized_path) != len(path):
                reduction_percent = ((len(path) - len(optimized_path)) / len(path)) * 100
                session.progress_queue.put(f"  → OPTIMIZED: {len(path)} → {len(optimized_path)} points ({reduction_percent:.1f}% reduction, score: {optimized_score:.3f})")
            else:
                session.progress_queue.put(f"  → Refined: {len(path)} points (score: {optimized_score:.3f})")
            
            # Update progress
            progress_percent = ((i + 1) / len(valid_paths)) * 100
            session.progress_queue.put(f"Progress: {progress_percent:.1f}% - {i+1}/{len(valid_paths)} paths completed")
            
            # Add small delay to make progress visible (optional)
            # No artificial delay: keep optimization throughput high.
        
        if generation != session.optimization_generation:
            return

        session.optimization_complete = True
        session.progress_queue.put("Optimization complete!")

        elapsed_seconds = time.perf_counter() - start_time
        if original_total_segments > 0:
            segment_reduction = ((original_total_segments - optimized_total_segments) / original_total_segments) * 100.0
        else:
            segment_reduction = 0.0

        session.progress_queue.put(
            (
                f"Summary: {elapsed_seconds:.2f}s total | "
                f"Segments (magenta -> optimized): {original_total_segments} -> {optimized_total_segments} "
                f"({segment_reduction:.1f}% reduction)"
            )
        )
        
    except Exception as e:
        session.progress_queue.put(f"Optimization error: {str(e)}")

@app.route('/progress/<session_id>')
def get_progress(session_id):
    """Get optimization progress updates."""
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session'})
    
    session = sessions[session_id]
    messages = []
    
    # Get all available progress messages
    while not session.progress_queue.empty():
        messages.append(session.progress_queue.get())
    
    return jsonify({
        'messages': messages,
        'optimization_complete': session.optimization_complete,
        'optimized_count': len(session.partial_optimized_paths),
        'total_paths': len(session.initial_paths) if session.initial_paths else 0
    })

@app.route('/stop_optimization', methods=['POST'])
def stop_optimization():
    """Stop optimization process early."""
    data = request.json
    session_id = data.get('session_id')
    
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session'})
    
    session = sessions[session_id]
    session.optimization_stopped = True
    session.optimization_generation += 1
    
    return jsonify({'success': True, 'message': 'Optimization stopping...'})


@app.route('/stop_auto_tune', methods=['POST'])
def stop_auto_tune():
    """Cancel auto-tune and apply the best-so-far detection settings if available."""
    data = request.json
    session_id = data.get('session_id')

    if session_id not in sessions:
        return jsonify({'error': 'Invalid session'})

    session = sessions[session_id]

    with session.lock:
        session.auto_tune_generation += 1
        session.auto_tune_active = False
        best_so_far = dict(session.auto_tune_best) if session.auto_tune_best else None

    if best_so_far and float(best_so_far.get('quality_score', 0.0)) > 0.0:
        session.parameters['dark_threshold'] = float(best_so_far.get('best_threshold', 0.20))
        session.parameters['min_path_length'] = int(best_so_far.get('best_min_length', 3))
        session.parameters['enable_optimization'] = False
        session.parameters['show_pre_optimization'] = True

        with session.lock:
            session.auto_tune_progress.update({
                'running': False,
                'finished': True,
                'success': True,
                'timed_out': False,
                'cancelled': True,
                'high_confidence_reached': bool(best_so_far.get('confidence_score', 0.0) >= AUTO_TUNE_CONFIDENCE_TARGET),
                'elapsed_sec': float(best_so_far.get('elapsed_sec', 0.0)),
                'confidence_score': float(best_so_far.get('confidence_score', 0.0)),
                'quality_score': float(best_so_far.get('quality_score', 0.0)),
                'best_threshold': float(session.parameters['dark_threshold']),
                'best_min_length': int(session.parameters['min_path_length']),
                'message': 'Auto-tune cancelled by user. Applied best-so-far settings with optimization off.',
                'updated_parameters': dict(session.parameters),
            })

        return jsonify({
            'success': True,
            'used_best_so_far': True,
            'detected_threshold': session.parameters['dark_threshold'],
            'detected_min_length': session.parameters['min_path_length'],
            'confidence_score': float(best_so_far.get('confidence_score', 0.0)),
            'quality_score': float(best_so_far.get('quality_score', 0.0)),
            'recommendation': 'Auto-tune stopped. Applied best-so-far detection settings with optimization turned off.',
            'updated_parameters': session.parameters,
        })

    session.parameters['dark_threshold'] = 0.20
    session.parameters['min_path_length'] = 3
    session.parameters['enable_optimization'] = False
    session.parameters['show_pre_optimization'] = True

    with session.lock:
        session.auto_tune_progress.update({
            'running': False,
            'finished': True,
            'success': False,
            'timed_out': False,
            'cancelled': True,
            'high_confidence_reached': False,
            'message': 'Auto-tune cancelled before a reliable candidate. Using safe defaults with optimization off.',
            'updated_parameters': dict(session.parameters),
        })

    return jsonify({
        'success': False,
        'used_best_so_far': False,
        'error': 'Auto-tune stopped before a reliable candidate was found. Using defaults with optimization turned off.',
        'updated_parameters': session.parameters,
    })

@app.route('/process', methods=['POST'])
def process():
    """Process centerlines with given parameters."""
    data = request.json
    session_id = data.get('session_id')
    
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session'})
    
    session = sessions[session_id]
    
    # Update parameters
    if 'parameters' in data:
        session.parameters.update(data['parameters'])
        
        # Force show_pre-optimization to True when optimization is disabled
        if not session.parameters.get('enable_optimization', True):
            session.parameters['show_pre_optimization'] = True
    
    try:
        # Process centerlines
        results = process_centerlines(session)
        
        if results and 'error' not in results:
            session.results = results
            
            # Calculate point reduction statistics
            original_points = sum(len(path) for path in results.get('pre_optimization_paths', []))
            optimized_points = sum(len(path) for path in results.get('optimized_paths', []))
            reduction_percentage = ((original_points - optimized_points) / max(original_points, 1)) * 100 if original_points > 0 else 0
            
            # Prepare response data
            response = {
                'success': True,
                'stats': results['stats'],
                'initial_paths_count': results['initial_paths_count'],
                'merged_paths_count': results['merged_paths_count'],
                'valid_paths_count': results['valid_paths_count'],
                'best_score': results['best_score'],
                'paths_count': len(results['optimized_paths']),
                'original_points': original_points,
                'optimized_points': optimized_points,
                'point_reduction_percentage': round(reduction_percentage, 1)
            }
            
            return jsonify(response)
        else:
            return jsonify(results or {'error': 'Unknown processing error'})
            
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'})

@app.route('/test_svg/<session_id>')
def test_svg(session_id):
    """Test SVG generation route."""
    print(f"Test SVG route called for session: {session_id}")
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session'})
    
    session = sessions[session_id]
    print(f"Session found. Has initial_paths: {bool(session.initial_paths)}")
    if session.initial_paths:
        print(f"Number of initial paths: {len(session.initial_paths)}")
    
    return jsonify({'status': 'test successful', 'has_paths': bool(session.initial_paths)})

@app.route('/generate_svg', methods=['POST'])
def generate_svg():
    """Generate and return SVG visualization with support for progressive display."""
    data = request.json
    session_id = data.get('session_id')
    display_mode = data.get('mode', 'final')  # 'immediate', 'progressive', or 'final'
    
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session'})

    session = sessions[session_id]
    
    # Update session parameters if provided (especially important for include_image setting)
    if 'parameters' in data:
        session.parameters.update(data['parameters'])

    show_pre = session.parameters.get('show_pre_optimization', False)
    
    # Check what data we have available
    if display_mode == 'immediate' and session.initial_paths:
        # Show only magenta paths immediately
        print(f"Generating immediate SVG with {len(session.initial_paths)} paths")
        paths_to_show = session.initial_paths
        optimized_paths = []
        circle_system = None
        pre_optimization_paths = paths_to_show if show_pre else []
        
    elif display_mode == 'progressive':
        # Show magenta + current blue progress
        with session.lock:
            # Make a thread-safe copy of the paths to avoid issues during iteration
            optimized_paths = list(session.partial_optimized_paths)
        
        print(f"Generating progressive SVG with {len(optimized_paths)} optimized paths")
        paths_to_show = session.initial_paths
        circle_system = None  # Could add this later
        pre_optimization_paths = paths_to_show if show_pre else []
        
    elif session.results:
        # Show final results
        print("Generating final SVG from session.results")
        results = session.results
        paths_to_show = results.get('pre_optimization_paths', [])
        optimized_paths = results.get('optimized_paths', [])
        circle_system = results.get('circle_system')
        pre_optimization_paths = paths_to_show if show_pre else []
        
    else:
        return jsonify({'error': 'No results available yet'})

    try:
        temp_svg = f"/tmp/centerline_result_{session_id}.svg"
        
        # Set global variables for SVG generation
        import worksok3_optimized as wo
        original_output = wo.OUTPUT_PATH
        original_show_bitmap = wo.SHOW_BITMAP
        original_show_pre = wo.SHOW_PRE_OPTIMIZATION_PATHS
        
        wo.OUTPUT_PATH = temp_svg
        wo.SHOW_BITMAP = session.parameters.get('include_image', False)
        
        # Respect user toggle for showing magenta pre-optimization paths in all modes.
        wo.SHOW_PRE_OPTIMIZATION_PATHS = show_pre
        
        background_image = session.display_image if session.display_image is not None else session.image

        # Generate SVG based on mode
        if display_mode == 'immediate':
            # Only magenta paths, no blue
            print("Creating immediate SVG")
            create_svg_output(
                background_image,
                None,  # No circle system
                [],    # No optimized paths (no blue lines)
                [],    # No scores
                pre_optimization_paths  # Show magenta paths
            )
        else:
            # Progressive or final - show both magenta and blue
            print(f"Creating {display_mode} SVG with {len(optimized_paths)} blue paths and {len(pre_optimization_paths)} magenta paths")
            create_svg_output(
                background_image,
                circle_system,
                optimized_paths,  # Blue paths
                [1.0] * len(optimized_paths),  # Dummy scores
                pre_optimization_paths  # Magenta paths
            )
        
        # Restore original settings
        wo.OUTPUT_PATH = original_output
        wo.SHOW_BITMAP = original_show_bitmap
        wo.SHOW_PRE_OPTIMIZATION_PATHS = original_show_pre
        
        # Read and return SVG content
        if os.path.exists(temp_svg):
            with open(temp_svg, 'r') as f:
                svg_content = f.read()
            print(f"SVG file created successfully, size: {len(svg_content)} characters")
            os.remove(temp_svg)
            return jsonify({'svg': svg_content})
        else:
            print(f"SVG file not found at: {temp_svg}")
            return jsonify({'error': 'SVG generation failed - file not created'})
            
    except Exception as e:
        return jsonify({'error': f'SVG generation error: {str(e)}'})

@app.route('/download_svg/<session_id>')
def download_svg(session_id):
    """Download SVG file."""
    if session_id not in sessions:
        return "Invalid session", 404
    
    session = sessions[session_id]
    
    # Check for progressive data or traditional results
    has_progressive_data = session.initial_paths and len(session.initial_paths) > 0
    has_traditional_results = session.results and 'error' not in session.results
    
    if not has_progressive_data and not has_traditional_results:
        return "No valid results", 404
    
    try:
        # Create filename based on original upload
        original_filename = session.original_filename or 'centerline_extraction'
        # Remove extension and add _centerline.svg
        name_without_ext = os.path.splitext(original_filename)[0]
        download_filename = f"{name_without_ext}_centerline.svg"
        
        # Create temporary SVG file
        temp_svg = os.path.join(tempfile.gettempdir(), f"centerline_download_{session_id}.svg")
        
        # Generate SVG with user settings
        import worksok3_optimized as wo
        original_output = wo.OUTPUT_PATH
        original_show_bitmap = wo.SHOW_BITMAP
        original_show_pre = wo.SHOW_PRE_OPTIMIZATION_PATHS
        
        wo.OUTPUT_PATH = temp_svg
        wo.SHOW_BITMAP = session.parameters.get('include_image', False)  # User-controlled
        background_image = session.display_image if session.display_image is not None else session.image
        
        # Determine what to show based on available data
        show_pre = session.parameters.get('show_pre_optimization', False)
        if has_progressive_data:
            # Use progressive data
            print(f"Generating download SVG with progressive data: {len(session.initial_paths)} initial paths, {len(session.partial_optimized_paths)} optimized paths")
            wo.SHOW_PRE_OPTIMIZATION_PATHS = show_pre
            
            create_svg_output(
                background_image,
                None,  # No circle system in progressive mode
                session.partial_optimized_paths,  # Blue optimized paths (may be empty if just started)
                [1.0] * len(session.partial_optimized_paths),  # Dummy scores
                session.initial_paths if show_pre else []  # Magenta initial paths
            )
        else:
            # Use traditional results
            results = session.results
            optimization_enabled = session.parameters.get('enable_optimization', True)
            if not optimization_enabled:
                show_pre = True
            wo.SHOW_PRE_OPTIMIZATION_PATHS = show_pre
            
            if results['circle_system'] is not None:
                create_svg_output(
                    background_image,
                    results['circle_system'],
                    results['optimized_paths'],
                    results['optimized_scores'],
                    results['pre_optimization_paths']
                )
            else:
                create_svg_output(
                    background_image,
                    None,
                    results['optimized_paths'],
                    results['optimized_scores'],
                    results['pre_optimization_paths']
                )
        
        # Restore original settings
        wo.OUTPUT_PATH = original_output
        wo.SHOW_BITMAP = original_show_bitmap
        wo.SHOW_PRE_OPTIMIZATION_PATHS = original_show_pre
        
        return send_file(temp_svg, as_attachment=True, download_name=download_filename)
        
    except Exception as e:
        return f"Error generating download: {str(e)}", 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Get port from environment variable for deployment
    port = int(os.environ.get('PORT', 5002))
    
    print("Starting Centerline Extraction Web App...")
    print(f"Access the app at: http://localhost:{port}")
    app.run(debug=False, host='0.0.0.0', port=port)
