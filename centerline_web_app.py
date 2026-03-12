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

class CenterlineSession:
    def __init__(self, session_id):
        self.session_id = session_id
        self.image = None
        self.display_image = None
        self.image_path = None
        self.original_filename = None
        self.preprocessing_info = {}
        self.results = None
        self.initial_paths = None  # Store raw magenta paths
        self.progress_queue = Queue()  # For progress updates
        self.optimization_thread = None
        self.optimization_complete = False
        self.optimization_stopped = False
        self.optimization_generation = 0
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
                                    base_min_lengths=None, preview_max_dim=900):
    """Jointly tune threshold and min path length on a preview image for a strong first extraction."""
    if base_min_lengths is None:
        base_min_lengths = [1, 3, 5, 8, 12, 16, 20]

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

    thresholds = np.linspace(threshold_range[0], threshold_range[1], num_thresholds)
    merge_gap = max(6, int(round(25 * scale)))

    best_result = None
    second_best_score = 0.0
    all_results = []

    print(f"Auto-tuning extraction parameters on preview {preview_image.shape} (scale {scale:.3f})...")
    print(
        "  Line-art bias: "
        f"strength={line_art_bias_strength:.2f}, "
        f"mean={mean_intensity:.3f}, dark<0.80={dark_ratio_high:.3f}, dark<0.55={dark_ratio_mid:.3f}"
    )

    for threshold in thresholds:
        try:
            initial_paths = extract_skeleton_paths(preview_image, float(threshold), min_object_size=3)
            if not initial_paths:
                all_results.append({
                    'threshold': float(threshold),
                    'best_min_length': 3,
                    'score': 0.0,
                    'valid_paths': 0,
                    'longest_path': 0
                })
                continue

            merged_paths = merge_nearby_paths(initial_paths, max_gap=merge_gap)
            threshold_best = None

            for full_min_length, preview_min_length in candidate_lengths:
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
                threshold_best = {
                    'threshold': float(threshold),
                    'best_min_length': 3,
                    'preview_min_length': 3,
                    'score': 0.0,
                    'valid_paths': 0,
                    'longest_path': 0,
                    'top_total_length': 0.0,
                    'median_top_length': 0.0
                }

            all_results.append(threshold_best)

            if best_result is None or threshold_best['score'] > best_result['score']:
                if best_result is not None:
                    second_best_score = best_result['score']
                best_result = threshold_best
            elif threshold_best['score'] > second_best_score:
                second_best_score = threshold_best['score']

            print(
                f"  Threshold {threshold:.3f}: best min length {threshold_best['best_min_length']}, "
                f"score {threshold_best['score']:.2f}, longest {threshold_best['longest_path']}"
            )
        except Exception as e:
            print(f"  Threshold {threshold:.3f}: auto-tune failed - {e}")
            all_results.append({
                'threshold': float(threshold),
                'best_min_length': 3,
                'preview_min_length': 3,
                'score': 0.0,
                'valid_paths': 0,
                'longest_path': 0,
                'error': str(e)
            })

    if best_result is None or best_result['score'] <= 0:
        return {
            'best_threshold': 0.20,
            'best_min_length': 3,
            'quality_score': 0.0,
            'confidence_score': 0.0,
            'recommendation': 'manual adjustment recommended',
            'preview_shape': preview_image.shape,
            'preview_scale': scale,
            'longest_path': 0,
            'valid_paths': 0,
            'all_results': all_results
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
        'all_results': all_results
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

    - Transparent images are composited over white to avoid losing alpha semantics.
    - PDF files are rendered from page 1 using pypdfium2.

    Returns:
        raw_gray: Original grayscale image in 0-1 range.
        extraction_gray: Grayscale image used for path extraction.
        preprocessing_info: Metadata about normalization decisions.
    """
    from skimage import color, io

    extension = Path(image_path).suffix.lower()

    if extension == '.pdf':
        try:
            pdfium = importlib.import_module('pypdfium2')
        except Exception as exc:
            raise ValueError(
                "PDF upload requires pypdfium2. Install dependencies and retry."
            ) from exc

        try:
            pdf = pdfium.PdfDocument(image_path)
            if len(pdf) == 0:
                raise ValueError("PDF has no pages")

            page = pdf[0]
            # Render first page at a readable resolution for tracing.
            bitmap = page.render(scale=2.0)
            pil_page = bitmap.to_pil().convert('RGB')
            img = np.asarray(pil_page).astype(np.float32) / 255.0
            pdf.close()
        except Exception as exc:
            raise ValueError(f"Failed to render PDF: {exc}") from exc
    else:
        img = io.imread(image_path)

    if len(img.shape) == 3:
        channels = img.shape[2]

        if channels == 4:
            rgb = img[:, :, :3].astype(np.float32)
            alpha = img[:, :, 3].astype(np.float32)

            if rgb.max() > 1.0:
                rgb /= 255.0
            if alpha.max() > 1.0:
                alpha /= 255.0

            # Composite transparent pixels over white background.
            composited_rgb = rgb * alpha[:, :, None] + (1.0 - alpha[:, :, None])
            gray = color.rgb2gray(composited_rgb)
        elif channels == 3:
            rgb = img.astype(np.float32)
            if rgb.max() > 1.0:
                rgb /= 255.0
            gray = color.rgb2gray(rgb)
        else:
            raise ValueError(f"Unsupported image format with {channels} channels")
    elif len(img.shape) == 2:
        gray = img.astype(np.float32)
        if gray.max() > 1.0:
            gray /= 255.0
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    raw_gray = np.clip(gray.astype(np.float32), 0.0, 1.0)
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
        tuning_result = auto_tune_extraction_parameters(session.image)

        if tuning_result['quality_score'] <= 0:
            return jsonify({
                'success': False,
                'error': 'Could not auto-tune extraction parameters. Current settings left unchanged.',
                'recommendation': tuning_result['recommendation']
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
            'updated_parameters': session.parameters
        })
    except Exception as e:
        return jsonify({'error': f'Auto-tune error: {str(e)}'})

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
        
        # Convert image to base64 for display
        pil_img = Image.fromarray((raw_gray * 255).astype(np.uint8))
        buf = BytesIO()
        pil_img.save(buf, format='PNG')
        buf.seek(0)
        img_data = base64.b64encode(buf.read()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'image_data': img_data,
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

        # Preserve legacy default behavior in progressive mode: with the default
        # gap (25), keep the original unmerged path flow. Only apply merging when
        # users intentionally change merge_gap from its default.
        merged_paths = initial_paths
        merge_applied = False
        if params.get('enable_optimization', True) and merge_gap != 25:
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
        
        # Convert numpy coordinates to regular Python lists for JSON serialization
        json_serializable_paths = []
        for path in valid_paths:
            serializable_path = [[int(point[0]), int(point[1])] for point in path]
            json_serializable_paths.append(serializable_path)
        
        # Store raw paths for immediate display
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
