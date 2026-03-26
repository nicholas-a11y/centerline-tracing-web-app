#!/usr/bin/env python3
"""
Web-based Centerline Extraction Tool
====================================

A Flask web application for interactive centerline extraction with real-time parameter adjustment.
"""

import threading
import time
import logging
import hashlib
import mimetypes
from html.parser import HTMLParser
from queue import Queue
from flask import Flask, render_template, request, jsonify, send_file
import os
import numpy as np
import base64
from scipy import ndimage as ndi
from io import BytesIO
from PIL import Image
import uuid
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urljoin, urlsplit
from urllib.request import Request, urlopen

# Import the current UI engine surface from the lightweight module.
from centerline_engine_light import (
    build_line_span_hints,
    create_svg_output,
    extract_skeleton_paths,
    prune_extracted_paths,
)

# Keep legacy optimization functionality on the original engine during migration.
from centerline_engine import (
    CircleEvaluationSystem,
    merge_nearby_paths,
    optimize_path_with_circles,
    optimize_path_with_custom_params,
    remove_overlapping_paths,
)
from centerline_core import (
    AUTO_TUNE_CONFIDENCE_TARGET,
    AUTO_TUNE_RANDOM_TILE_DIM,
    AUTO_TUNE_TIME_BUDGET_SEC,
    _build_random_tile_mosaic,
    _extract_single_tile_preview_entry,
    _offset_sample_metadata_to_source,
    _resolve_crop_bounds,
    auto_detect_dark_threshold,
    auto_detect_min_path_length,
    auto_tune_extraction_parameters,
    load_and_process_image,
    resolve_parameter_scale,
    resolve_extraction_profile,
)


TILE_PREVIEW_CACHE_LIMIT = 512
IMMEDIATE_PROGRESS_BAR_PATH_THRESHOLD = 200
TEST_UI_ENABLED = False
TEST_UI_IMPORT_ERROR = None

try:
    from test_ui_backend import (
        ALL_FIXTURE_IDS,
        DEFAULT_FITTING_PARAMETERS,
        create_run,
        get_fixture_detail,
        get_run_progress,
        get_run_summary,
        list_runs,
    )
    TEST_UI_ENABLED = True
except Exception as exc:
    # Production images may omit pytest/test dependencies. Keep core app alive.
    ALL_FIXTURE_IDS = []
    DEFAULT_FITTING_PARAMETERS = {}
    TEST_UI_IMPORT_ERROR = str(exc)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

MERGE_GAP_REFERENCE_STROKE_WIDTH_PX = 3.0
REMOTE_UPLOAD_TIMEOUT_SEC = 12
DEFAULT_MIN_PATH_LENGTH = 2
DEFAULT_ENABLE_PRUNING = False

IS_RAILWAY_DEPLOYMENT = any(key.startswith('RAILWAY_') for key in os.environ)
QUIET_HTTP_LOGS = os.environ.get('QUIET_HTTP_LOGS', '1' if IS_RAILWAY_DEPLOYMENT else '0') == '1'
VERBOSE_SERVER_LOGS = os.environ.get('VERBOSE_SERVER_LOGS', '0') == '1'


def _configure_runtime_logging():
    if not QUIET_HTTP_LOGS:
        return

    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    app.logger.setLevel(logging.ERROR)


def _log_debug(message):
    if VERBOSE_SERVER_LOGS:
        print(message)


_configure_runtime_logging()


def _test_ui_unavailable_response():
    detail = TEST_UI_IMPORT_ERROR or "test_ui_backend unavailable"
    return jsonify({
        'error': 'Test UI is unavailable in this deployment environment',
        'detail': detail,
    }), 503

# Global storage for session data
sessions = {}
SVG_VIEW_PATH_RENDER_LIMIT = 5000
SVG_PROGRESSIVE_PREVIEW_PATH_LIMIT = 1200
BENCHMARK_FILENAME_PREFIXES = ('bench__', 'metrics__')


def _sample_paths_for_progressive_preview(paths, max_paths):
    """Return an evenly spread subset of paths while keeping first and last."""
    paths = list(paths or [])
    limit = max(0, int(max_paths or 0))
    if limit <= 0 or len(paths) <= limit:
        return paths

    if limit == 1:
        return [paths[0]]

    sampled_indices = []
    last_index = len(paths) - 1
    for slot in range(limit):
        sampled_index = int(round(slot * last_index / max(limit - 1, 1)))
        if sampled_indices and sampled_index == sampled_indices[-1]:
            continue
        sampled_indices.append(sampled_index)

    if sampled_indices[0] != 0:
        sampled_indices[0] = 0
    if sampled_indices[-1] != last_index:
        sampled_indices[-1] = last_index

    return [paths[index] for index in sampled_indices]


def _sample_paths_and_hints_for_preview(paths, path_hints, max_paths):
    """Sample paths and keep path hints aligned with the sampled subset."""
    sampled_paths = _sample_paths_for_progressive_preview(paths, max_paths)
    original_paths = list(paths or [])
    original_hints = list(path_hints or [])

    if not sampled_paths:
        return [], []

    if len(sampled_paths) == len(original_paths):
        if len(original_hints) == len(original_paths):
            return sampled_paths, original_hints
        return sampled_paths, [build_line_span_hints(path) for path in sampled_paths]

    sampled_lookup = {id(path): index for index, path in enumerate(original_paths)}
    sampled_hints = []
    for path in sampled_paths:
        original_index = sampled_lookup.get(id(path))
        if original_index is None or original_index >= len(original_hints):
            sampled_hints.append(build_line_span_hints(path))
        else:
            sampled_hints.append(original_hints[original_index])

    return sampled_paths, sampled_hints

class CenterlineSession:
    def __init__(self, session_id):
        self.session_id = session_id
        self.created_at = time.time()
        self.image = None
        self.display_image = None
        self.image_path = None
        self.image_file_size = 0
        self.original_filename = None
        self.benchmark_enabled = False
        self.benchmark_metrics = None
        self.preprocessing_info = {}
        self.results = None
        self.initial_paths = None  # Store raw magenta paths
        self.progress_queue = Queue()  # For progress updates
        self.optimization_thread = None
        self.optimization_complete = False
        self.optimization_stopped = False
        self.optimization_generation = 0
        self.auto_detect_generation = 0
        self.auto_detect_active = False
        self.auto_detect_thread = None
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
        self.initial_path_hints = []
        self.partial_optimized_path_hints = []
        self.pruning_debug_payload = None
        self.live_preview_paths = []
        self.live_preview_kind = None
        self.live_preview_frame_id = 0
        self.stroke_width_estimate_px = None
        self.merge_progress = {
            'active': False,
            'phase': 'idle',
            'processed': 0,
            'total': 0,
            'percent': 0.0,
        }
        self.immediate_progress = {
            'active': False,
            'phase': 'idle',
            'percent': 0.0,
            'show_bar': False,
            'path_count': 0,
            'message': 'idle',
            'detail': '',
            'elapsed_sec': 0.0,
        }
        self.svg_cache = {}
        self.tile_preview_cache = {}
        self.lock = threading.Lock()  # Lock for thread-safe access to partial_optimized_paths
        self.parameters = {
            'dark_threshold': 0.20,
            'merge_gap': 25,  # Endpoint reach-out distance (pixels) for path merging
            'merge_angle_priority': 30.0,  # % weight for angle continuity vs distance
            'enable_long_path_merging': False,  # Optional slower second pass that can join already-long fragments
            'rdp_tolerance': 5.0,       # Default simplification tolerance
            'smoothing_factor': 0.006,  # Balanced smoothing that keeps runtime responsive
            'simplification_strength': 100.0,  # Maximum vertex reduction target
            'arc_fit_strength': 72.0,  # Favor curves without over-smoothing
            'line_fit_strength': 0.0,  # Disable line fitting by default
            'enable_post_fit_export': False,  # Opt-in export fitting so fast preview stays fast
            'source_smoothing': 70.0,  # Export-only pre-fit smoothing strength
            'short_path_protection': 0.0,  # No extra protection for shorter paths by default
            'mean_closeness_px': 1.8,  # Average allowed distance from blue path to magenta path
            'peak_closeness_px': 4.5,  # 95th percentile allowed distance from blue path to magenta path
            'score_preservation': 99.0,  # Keep accepted fits as close as possible to the source path
            'cubic_fit_tolerance': 0.35,  # SVG cubic fitting tolerance in px (lower = tighter, more segments)
            'endpoint_tangent_strictness': 85.0,  # Strength of start/end handle alignment to extracted path direction (not fixture/golden data)
            'force_orthogonal_as_lines': False,  # Optionally force axis-aligned/corner paths to use line segments only
            'enable_curve_fitting': True,  # Fit optimized paths to cubic segments only when explicitly enabled
            'min_path_length': DEFAULT_MIN_PATH_LENGTH,      # Increase to 8-15 for longer segments
            'enable_pruning': DEFAULT_ENABLE_PRUNING,    # Enable the path simplification/pruning pass after length filtering
            'enable_optimization': False,   # Keep the simplified UI on the direct pruned-skeleton path
            'show_pre_optimization': True,  # Always expose the direct skeleton output in simplified mode
            'show_pruning_debug_grid': False,  # Debug-only pixel-grid inspector for pruning
            'include_image': False,    # Include original image in SVG background
            'normalization_mode': 'auto',  # auto|on|off preprocessing normalization
            'normalization_sensitivity': 'medium',  # low|medium|high
        }
        self.auto_detect_progress = {
            'running': False,
            'started_at': 0.0,
            'finished': False,
            'success': False,
            'cancelled': False,
            'elapsed_sec': 0.0,
            'confidence_score': 0.0,
            'detected_threshold': float(self.parameters['dark_threshold']),
            'best_threshold': float(self.parameters['dark_threshold']),
            'recommendation': 'idle',
            'thresholds_evaluated': 0,
            'thresholds_planned': 0,
            'preview_shape': [],
            'preview_scale': 1.0,
            'sample_metadata': {'sampled': False},
            'message': 'idle',
            'updated_parameters': dict(self.parameters),
        }


def _clear_live_preview(session, generation=None):
    with session.lock:
        if generation is not None and generation != session.optimization_generation:
            return
        session.live_preview_paths = []
        session.live_preview_kind = None


def _publish_live_preview(session, paths, kind, generation=None, max_paths=SVG_PROGRESSIVE_PREVIEW_PATH_LIMIT):
    sampled_paths = _sample_paths_for_progressive_preview(paths, max_paths)
    with session.lock:
        if generation is not None and generation != session.optimization_generation:
            return False
        session.live_preview_paths = list(sampled_paths)
        session.live_preview_kind = str(kind or 'progressive')
        session.live_preview_frame_id += 1
        return True


def _mark_live_preview_frame(session, kind, generation=None):
    with session.lock:
        if generation is not None and generation != session.optimization_generation:
            return False
        session.live_preview_kind = str(kind or 'progressive')
        session.live_preview_frame_id += 1
        return True


def _set_merge_progress(session, *, active, phase, processed=0, total=0, percent=0.0, generation=None):
    with session.lock:
        if generation is not None and generation != session.optimization_generation:
            return False
        session.merge_progress = {
            'active': bool(active),
            'phase': str(phase or 'idle'),
            'processed': max(0, int(processed or 0)),
            'total': max(0, int(total or 0)),
            'percent': max(0.0, min(100.0, float(percent or 0.0))),
        }
        return True


def _set_immediate_progress(
    session,
    *,
    active,
    phase,
    percent=0.0,
    show_bar=False,
    path_count=0,
    message='',
    detail='',
    elapsed_sec=0.0,
    generation=None,
):
    with session.lock:
        if generation is not None and generation != session.optimization_generation:
            return False
        session.immediate_progress = {
            'active': bool(active),
            'phase': str(phase or 'idle'),
            'percent': max(0.0, min(100.0, float(percent or 0.0))),
            'show_bar': bool(show_bar),
            'path_count': max(0, int(path_count or 0)),
            'message': str(message or ''),
            'detail': str(detail or ''),
            'elapsed_sec': max(0.0, float(elapsed_sec or 0.0)),
        }
        return True


def _get_session_extraction_profile(session):
    return resolve_extraction_profile(session.preprocessing_info)


def _logical_min_path_length(params, extraction_profile):
    requested_min_length = max(1, int(params.get('min_path_length', DEFAULT_MIN_PATH_LENGTH)))
    max_min_path_length = extraction_profile.get('max_min_path_length')
    if max_min_path_length is None:
        return requested_min_length
    return max(1, min(requested_min_length, int(max_min_path_length)))


def _effective_min_object_size(base_min_object_size, params, extraction_profile, image_shape=None):
    resolved_min_object_size = max(0, int(base_min_object_size or 0))
    if resolved_min_object_size <= 0:
        return 0

    logical_min_length = _logical_min_path_length(params, extraction_profile)
    parameter_scale = max(1.0, float(resolve_parameter_scale(image_shape)))
    scale_preserving_cap = max(1, int(round(float(resolved_min_object_size) / parameter_scale)))
    loop_preserving_cap = max(1, int(logical_min_length) - 1)

    return max(1, min(resolved_min_object_size, loop_preserving_cap, scale_preserving_cap))


def _effective_min_path_length(params, extraction_profile, image_shape=None):
    logical_min_length = _logical_min_path_length(params, extraction_profile)
    parameter_scale = resolve_parameter_scale(image_shape)
    return max(1, int(round(logical_min_length * parameter_scale)))


def _effective_merge_gap(params, image_shape=None):
    logical_merge_gap = max(1, int(params.get('merge_gap', 25)))
    parameter_scale = resolve_parameter_scale(image_shape)
    return max(1, int(round(logical_merge_gap * parameter_scale)))


def _estimate_median_stroke_width_from_masks(binary_mask, skeleton_mask):
    if binary_mask is None or skeleton_mask is None:
        return None

    try:
        binary = np.asarray(binary_mask, dtype=bool)
        skeleton = np.asarray(skeleton_mask, dtype=bool)
    except Exception:
        return None

    if binary.shape != skeleton.shape or not np.any(binary) or not np.any(skeleton):
        return None

    distances = ndi.distance_transform_edt(binary)
    radii = distances[skeleton]
    radii = radii[np.isfinite(radii) & (radii > 0.0)]
    if radii.size == 0:
        return None

    return float(np.median(radii) * 2.0)


def _estimate_median_stroke_width(gray_image, dark_threshold, min_object_size=1):
    if gray_image is None:
        return None

    from skimage import morphology

    binary = np.asarray(gray_image) < float(dark_threshold)
    min_object_size = max(1, int(min_object_size or 1))
    if min_object_size > 1:
        binary = morphology.remove_small_objects(binary, min_object_size)
    if not np.any(binary):
        return None

    skeleton = morphology.skeletonize(binary)
    return _estimate_median_stroke_width_from_masks(binary, skeleton)


def _effective_merge_gap_from_stroke(params, image_shape=None, median_stroke_width_px=None):
    merge_gap_mode = str(params.get('merge_gap_mode', '') or '').strip().lower()
    logical_merge_gap = max(1, int(params.get('merge_gap', 25)))
    if merge_gap_mode == 'stroke_factor':
        if median_stroke_width_px is None or not np.isfinite(median_stroke_width_px) or median_stroke_width_px <= 0.0:
            return logical_merge_gap
        return max(1, int(round(float(logical_merge_gap) * float(median_stroke_width_px))))

    if median_stroke_width_px is None or not np.isfinite(median_stroke_width_px) or median_stroke_width_px <= 0.0:
        return _effective_merge_gap(params, image_shape=image_shape)

    stroke_scaled_gap = float(logical_merge_gap) * (float(median_stroke_width_px) / float(MERGE_GAP_REFERENCE_STROKE_WIDTH_PX))
    return max(1, int(round(stroke_scaled_gap)))


def _apply_auto_tuned_merge_gap(session, dark_threshold, gray_image, min_object_size=1):
    median_stroke_width_px = _estimate_median_stroke_width(
        gray_image,
        dark_threshold,
        min_object_size=min_object_size,
    )
    session.stroke_width_estimate_px = median_stroke_width_px
    session.parameters['merge_gap'] = 2
    session.parameters['merge_gap_mode'] = 'stroke_factor'
    if median_stroke_width_px is not None and np.isfinite(median_stroke_width_px) and median_stroke_width_px > 0.0:
        session.preprocessing_info['median_stroke_width_px'] = float(median_stroke_width_px)
    session.preprocessing_info['effective_merge_gap'] = int(_effective_merge_gap_from_stroke(
        session.parameters,
        image_shape=gray_image.shape if gray_image is not None else None,
        median_stroke_width_px=median_stroke_width_px,
    ))
    return median_stroke_width_px


def _augment_sample_metadata(sample_metadata, image_shape, logical_min_path_length=None, logical_merge_gap=None):
    metadata = dict(sample_metadata or {})
    parameter_scale = resolve_parameter_scale(image_shape)
    metadata['parameter_scale'] = float(parameter_scale)
    if logical_min_path_length is not None:
        metadata['logical_min_path_length'] = int(max(1, logical_min_path_length))
        metadata['effective_min_path_length'] = int(max(1, round(float(logical_min_path_length) * parameter_scale)))
    if logical_merge_gap is not None:
        metadata['logical_merge_gap'] = int(max(1, logical_merge_gap))
        metadata['effective_merge_gap'] = int(max(1, round(float(logical_merge_gap) * parameter_scale)))
    return metadata


def _attach_merge_gap_metadata(metadata, logical_merge_gap, effective_merge_gap, median_stroke_width_px=None):
    enriched = dict(metadata or {})
    enriched['logical_merge_gap'] = int(max(1, logical_merge_gap))
    enriched['effective_merge_gap'] = int(max(1, effective_merge_gap))
    if median_stroke_width_px is not None and np.isfinite(median_stroke_width_px) and median_stroke_width_px > 0.0:
        enriched['median_stroke_width_px'] = float(median_stroke_width_px)
    return enriched


def _serialize_point_list(points):
    serialized = []
    for point in list(points or []):
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        serialized.append([float(point[0]), float(point[1])])
    return serialized


def _serialize_line_span_hints(hints):
    serialized = []
    for item in list(hints or []):
        if not isinstance(item, dict):
            continue
        start_point = item.get('start_point')
        end_point = item.get('end_point')
        serialized.append({
            'start_idx': int(item.get('start_idx', 0)),
            'end_idx': int(item.get('end_idx', 0)),
            'kind': str(item.get('kind', 'line') or 'line'),
            'confidence': float(item.get('confidence', 0.0)),
            'source': str(item.get('source', 'unknown') or 'unknown'),
            'start_point': _serialize_point_list([start_point])[0] if isinstance(start_point, (list, tuple)) and len(start_point) >= 2 else None,
            'end_point': _serialize_point_list([end_point])[0] if isinstance(end_point, (list, tuple)) and len(end_point) >= 2 else None,
        })
    return serialized


def _offset_serialized_point_list(points, row_offset=0, col_offset=0):
    adjusted = []
    for point in list(points or []):
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        adjusted.append([
            float(point[0]) + float(row_offset),
            float(point[1]) + float(col_offset),
        ])
    return adjusted


def _offset_pruning_debug_payload(debug_payload, row_offset=0, col_offset=0):
    if not isinstance(debug_payload, dict) or (row_offset == 0 and col_offset == 0):
        return debug_payload

    adjusted = {
        'summary': dict(debug_payload.get('summary') or {}),
        'paths': [],
        'path_hints': [],
    }

    for hint_list in list(debug_payload.get('path_hints') or []):
        adjusted_hints = []
        for hint in list(hint_list or []):
            if not isinstance(hint, dict):
                continue
            adjusted_hint = dict(hint)
            adjusted_hint['start_point'] = _offset_serialized_point_list([hint.get('start_point')], row_offset, col_offset)[0] if isinstance(hint.get('start_point'), (list, tuple)) and len(hint.get('start_point')) >= 2 else None
            adjusted_hint['end_point'] = _offset_serialized_point_list([hint.get('end_point')], row_offset, col_offset)[0] if isinstance(hint.get('end_point'), (list, tuple)) and len(hint.get('end_point')) >= 2 else None
            adjusted_hints.append(adjusted_hint)
        adjusted['path_hints'].append(adjusted_hints)

    for path_payload in list(debug_payload.get('paths') or []):
        if not isinstance(path_payload, dict):
            continue
        adjusted_path = dict(path_payload)
        adjusted_path['input_points'] = _offset_serialized_point_list(path_payload.get('input_points'), row_offset, col_offset)
        adjusted_path['output_points'] = _offset_serialized_point_list(path_payload.get('output_points'), row_offset, col_offset)
        adjusted_path['dropped_input_points'] = _offset_serialized_point_list(path_payload.get('dropped_input_points'), row_offset, col_offset)

        adjusted_details = []
        for detail in list(path_payload.get('dropped_input_point_details') or []):
            if not isinstance(detail, dict):
                continue
            adjusted_detail = dict(detail)
            adjusted_detail['point'] = _offset_serialized_point_list([detail.get('point')], row_offset, col_offset)[0] if isinstance(detail.get('point'), (list, tuple)) and len(detail.get('point')) >= 2 else None
            adjusted_details.append(adjusted_detail)
        adjusted_path['dropped_input_point_details'] = adjusted_details

        adjusted_guards = []
        for detail in list(path_payload.get('kept_vertex_guards') or []):
            if not isinstance(detail, dict):
                continue
            adjusted_detail = dict(detail)
            adjusted_detail['point'] = _offset_serialized_point_list([detail.get('point')], row_offset, col_offset)[0] if isinstance(detail.get('point'), (list, tuple)) and len(detail.get('point')) >= 2 else None
            adjusted_guards.append(adjusted_detail)
        adjusted_path['kept_vertex_guards'] = adjusted_guards

        adjusted_line_hints = []
        for hint in list(path_payload.get('line_span_hints') or []):
            if not isinstance(hint, dict):
                continue
            adjusted_hint = dict(hint)
            adjusted_hint['start_point'] = _offset_serialized_point_list([hint.get('start_point')], row_offset, col_offset)[0] if isinstance(hint.get('start_point'), (list, tuple)) and len(hint.get('start_point')) >= 2 else None
            adjusted_hint['end_point'] = _offset_serialized_point_list([hint.get('end_point')], row_offset, col_offset)[0] if isinstance(hint.get('end_point'), (list, tuple)) and len(hint.get('end_point')) >= 2 else None
            adjusted_line_hints.append(adjusted_hint)
        adjusted_path['line_span_hints'] = adjusted_line_hints

        bbox = path_payload.get('bbox') if isinstance(path_payload.get('bbox'), dict) else None
        if bbox:
            adjusted_path['bbox'] = {
                'min_row': float(bbox.get('min_row', 0.0)) + float(row_offset),
                'max_row': float(bbox.get('max_row', 0.0)) + float(row_offset),
                'min_col': float(bbox.get('min_col', 0.0)) + float(col_offset),
                'max_col': float(bbox.get('max_col', 0.0)) + float(col_offset),
            }

        adjusted['paths'].append(adjusted_path)

    return adjusted


def _serialize_pruning_debug_payload(debug_payload):
    if not isinstance(debug_payload, dict):
        return None

    def _serialize_point_detail(detail):
        if not isinstance(detail, dict):
            return None
        point = detail.get('point')
        point_pair = None
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            point_pair = [float(point[0]), float(point[1])]
        return {
            'point_index': int(detail.get('point_index', 0)),
            'point': point_pair,
            'reason': str(detail.get('reason', 'pruned')),
            'source_stage': str(detail.get('source_stage', 'finalize')),
        }

    def _serialize_kept_vertex_guard(detail):
        if not isinstance(detail, dict):
            return None
        point = detail.get('point')
        point_pair = None
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            point_pair = [float(point[0]), float(point[1])]
        return {
            'point_index': int(detail.get('point_index', 0)),
            'point': point_pair,
            'provenance': 'inferred' if str(detail.get('provenance', 'preserved')).lower() == 'inferred' else 'preserved',
            'working_index': int(detail.get('working_index')) if detail.get('working_index') is not None else None,
            'guards': [str(item) for item in list(detail.get('guards') or [])],
            'criteria_explanations': [str(item) for item in list(detail.get('criteria_explanations') or [])],
            'turn_angle': float(detail.get('turn_angle', 0.0)),
            'local_deviation': float(detail.get('local_deviation', 0.0)),
            'turn_sign': int(detail.get('turn_sign', 0)),
        }

    summary = dict(debug_payload.get('summary') or {})
    serialized_paths = []
    for path_payload in list(debug_payload.get('paths') or []):
        if not isinstance(path_payload, dict):
            continue
        bbox = path_payload.get('bbox') if isinstance(path_payload.get('bbox'), dict) else None
        dropped_input_point_details = []
        for detail in list(path_payload.get('dropped_input_point_details') or []):
            serialized_detail = _serialize_point_detail(detail)
            if serialized_detail is not None:
                dropped_input_point_details.append(serialized_detail)
        kept_vertex_guards = []
        for detail in list(path_payload.get('kept_vertex_guards') or []):
            serialized_detail = _serialize_kept_vertex_guard(detail)
            if serialized_detail is not None:
                kept_vertex_guards.append(serialized_detail)
        serialized_paths.append({
            'path_index': int(path_payload.get('path_index', 0)),
            'original_path_index': int(path_payload.get('original_path_index', path_payload.get('path_index', 0))),
            'input_count': int(path_payload.get('input_count', 0)),
            'output_count': int(path_payload.get('output_count', 0)),
            'dropped_input_count': int(path_payload.get('dropped_input_count', 0)),
            'inferred_output_count': int(path_payload.get('inferred_output_count', 0)),
            'spur_trimmed_count': int(path_payload.get('spur_trimmed_count', 0)),
            'critical_points_count': int(path_payload.get('critical_points_count', 0)),
            'closed': bool(path_payload.get('closed', False)),
            'path_state': str(path_payload.get('path_state', 'unknown') or 'unknown'),
            'changed': bool(path_payload.get('changed', False)),
            'state_reasons': [str(item) for item in list(path_payload.get('state_reasons') or [])],
            'unchanged_reasons': [str(item) for item in list(path_payload.get('unchanged_reasons') or [])],
            'pruning_config': {
                'min_output_points': int(dict(path_payload.get('pruning_config') or {}).get('min_output_points', 0)),
                'deviation_tol': float(dict(path_payload.get('pruning_config') or {}).get('deviation_tol', 0.0)),
                'rdp_epsilon': float(dict(path_payload.get('pruning_config') or {}).get('rdp_epsilon', 0.0)),
            },
            'input_points': _serialize_point_list(path_payload.get('input_points')),
            'output_points': _serialize_point_list(path_payload.get('output_points')),
            'output_point_provenance': [
                'inferred' if str(item).lower() == 'inferred' else 'preserved'
                for item in list(path_payload.get('output_point_provenance') or [])
            ],
            'dropped_input_points': _serialize_point_list(path_payload.get('dropped_input_points')),
            'dropped_input_point_details': dropped_input_point_details,
            'dropped_counts_by_reason': {
                str(key): int(value)
                for key, value in dict(path_payload.get('dropped_counts_by_reason') or {}).items()
            },
            'kept_vertex_guards': kept_vertex_guards,
            'line_span_hints': _serialize_line_span_hints(path_payload.get('line_span_hints')),
            'rejection_reason': str(path_payload.get('rejection_reason')) if path_payload.get('rejection_reason') else None,
            'bbox': {
                'min_row': float(bbox.get('min_row', 0.0)),
                'max_row': float(bbox.get('max_row', 0.0)),
                'min_col': float(bbox.get('min_col', 0.0)),
                'max_col': float(bbox.get('max_col', 0.0)),
            } if bbox else None,
        })

    return {
        'summary': {
            'path_count': int(summary.get('path_count', 0)),
            'changed_path_count': int(summary.get('changed_path_count', 0)),
            'unchanged_path_count': int(summary.get('unchanged_path_count', 0)),
            'rejected_path_count': int(summary.get('rejected_path_count', 0)),
            'input_points': int(summary.get('input_points', 0)),
            'output_points': int(summary.get('output_points', 0)),
            'dropped_input_points': int(summary.get('dropped_input_points', 0)),
            'line_span_hint_count': int(summary.get('line_span_hint_count', 0)),
            'inferred_output_points': int(summary.get('inferred_output_points', 0)),
            'dropped_counts_by_reason': {
                str(key): int(value)
                for key, value in dict(summary.get('dropped_counts_by_reason') or {}).items()
            },
            'path_state_counts': {
                str(key): int(value)
                for key, value in dict(summary.get('path_state_counts') or {}).items()
            },
        },
        'paths': serialized_paths,
        'path_hints': [
            _serialize_line_span_hints(path_hints)
            for path_hints in list(debug_payload.get('path_hints') or [])
        ],
    }


def _should_capture_benchmark_metrics(filename):
    cleaned = os.path.basename(str(filename or '')).lower()
    return any(cleaned.startswith(prefix) for prefix in BENCHMARK_FILENAME_PREFIXES)


def _should_force_pruning_debug(filename):
    cleaned = os.path.basename(str(filename or '')).lower()
    return '_debug' in cleaned


def _ensure_benchmark_metrics(session):
    if not session.benchmark_enabled:
        return None
    if session.benchmark_metrics is None:
        session.benchmark_metrics = {
            'session_id': session.session_id,
            'filename': session.original_filename,
            'enabled': True,
            'created_at_epoch': float(session.created_at),
            'created_at_ms': int(round(session.created_at * 1000.0)),
            'upload_size_bytes': int(session.image_file_size or 0),
            'preprocessing': {},
            'stages': {},
            'latest_svg': {},
            'latest_download': {},
            'optimization': {},
        }
    return session.benchmark_metrics


def _record_benchmark_stage(session, stage_name, elapsed_ms, extra=None):
    metrics = _ensure_benchmark_metrics(session)
    if metrics is None:
        return
    payload = {
        'elapsed_ms': round(float(elapsed_ms), 3),
        'recorded_at_ms': int(round(time.time() * 1000.0)),
    }
    if extra:
        payload.update(extra)
    metrics['stages'][stage_name] = payload


def _merge_benchmark_numeric_dict(target, updates):
    for key, value in updates.items():
        target[key] = float(target.get(key, 0.0)) + float(value)


def _round_benchmark_numeric_dict(values):
    return {
        str(key): round(float(value), 3)
        for key, value in values.items()
        if isinstance(value, (int, float))
    }


def _update_benchmark_bucket(session, bucket_name, payload):
    metrics = _ensure_benchmark_metrics(session)
    if metrics is None:
        return
    bucket = dict(metrics.get(bucket_name, {}))
    bucket.update(payload)
    bucket['recorded_at_ms'] = int(round(time.time() * 1000.0))
    metrics[bucket_name] = bucket


def _record_benchmark_svg(session, display_mode, elapsed_ms, svg_content, suppressed, cache_hit=False):
    _update_benchmark_bucket(session, 'latest_svg', {
        'mode': str(display_mode),
        'elapsed_ms': round(float(elapsed_ms), 3),
        'svg_chars': int(len(svg_content)),
        'svg_bytes_utf8': int(len(svg_content.encode('utf-8'))),
        'preview_paths_suppressed': bool(suppressed),
        'cache_hit': bool(cache_hit),
    })


def _record_benchmark_download(session, elapsed_ms, file_size, cache_hit=False):
    _update_benchmark_bucket(session, 'latest_download', {
        'elapsed_ms': round(float(elapsed_ms), 3),
        'svg_file_bytes': int(file_size),
        'cache_hit': bool(cache_hit),
    })


def _latest_benchmark_recorded_at_ms(metrics):
    latest_recorded_at_ms = int(metrics.get('created_at_ms', 0) or 0)

    for stage_payload in metrics.get('stages', {}).values():
        latest_recorded_at_ms = max(latest_recorded_at_ms, int(stage_payload.get('recorded_at_ms', 0) or 0))

    for bucket_name in ('latest_svg', 'latest_download', 'optimization'):
        bucket_payload = metrics.get(bucket_name, {})
        latest_recorded_at_ms = max(latest_recorded_at_ms, int(bucket_payload.get('recorded_at_ms', 0) or 0))

    return latest_recorded_at_ms


def _build_benchmark_summary(session, metrics, payload):
    latest_recorded_at_ms = _latest_benchmark_recorded_at_ms(metrics)
    if latest_recorded_at_ms <= 0:
        latest_recorded_at_ms = int(round(time.time() * 1000.0))

    stage_elapsed_ms = 0.0
    for stage_payload in metrics.get('stages', {}).values():
        elapsed_ms = stage_payload.get('elapsed_ms')
        if isinstance(elapsed_ms, (int, float)):
            stage_elapsed_ms += float(elapsed_ms)

    optimization_elapsed_ms = metrics.get('optimization', {}).get('elapsed_ms')

    if payload.get('optimization_complete'):
        current_phase = 'optimization_complete'
    elif payload.get('has_results'):
        current_phase = 'results_ready'
    elif metrics.get('stages', {}).get('process_immediate'):
        current_phase = 'optimization_running'
    else:
        current_phase = 'uploaded'

    wall_clock_ms = max(0.0, float(latest_recorded_at_ms) - float(metrics.get('created_at_ms', 0) or 0))

    summary = {
        'current_phase': current_phase,
        'wall_clock_ms': round(wall_clock_ms, 3),
        'wall_clock_sec': round(wall_clock_ms / 1000.0, 4),
        'recorded_stage_ms': round(stage_elapsed_ms, 3),
    }
    if isinstance(optimization_elapsed_ms, (int, float)):
        summary['optimization_ms'] = round(float(optimization_elapsed_ms), 3)

    return summary


def _path_snapshot_stats(paths):
    paths = paths or []
    return len(paths), sum(len(path) for path in paths)


def _path_snapshot_signature(paths):
    digest = hashlib.blake2b(digest_size=16)
    for path in list(paths or []):
        digest.update(f"{len(path)}:".encode('ascii'))
        for point in list(path or []):
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                digest.update(b'?,?;')
                continue
            digest.update(f"{float(point[0]):.4f},{float(point[1]):.4f};".encode('ascii'))
        digest.update(b'|')
    return digest.hexdigest()


def _path_hint_signature(path_hints):
    digest = hashlib.blake2b(digest_size=16)
    for hints in list(path_hints or []):
        digest.update(f"{len(list(hints or []))}:".encode('ascii'))
        for hint in list(hints or []):
            if not isinstance(hint, dict):
                digest.update(b'bad|')
                continue
            digest.update(
                (
                    f"{int(hint.get('start_idx', 0))},{int(hint.get('end_idx', 0))},"
                    f"{str(hint.get('kind', 'line'))},{float(hint.get('confidence', 0.0)):.4f}|"
                ).encode('ascii')
            )
    return digest.hexdigest()


def _svg_render_signature(
    render_variant,
    include_image,
    show_pre,
    source_smoothing,
    curve_fit_tolerance,
    endpoint_tangent_strictness,
    force_orthogonal_as_lines,
    enable_curve_fitting,
    enable_post_fit_export,
    fit_optimized_paths,
    combine_optimized_paths,
    combine_pre_optimization_paths,
    coordinate_precision,
    pre_paths,
    optimized_paths,
    optimized_path_hints,
    suppress_paths_in_view,
    has_circle_system,
    optimization_generation,
    optimization_complete,
):
    pre_count, pre_points = _path_snapshot_stats(pre_paths)
    optimized_count, optimized_points = _path_snapshot_stats(optimized_paths)
    pre_signature = _path_snapshot_signature(pre_paths)
    optimized_signature = _path_snapshot_signature(optimized_paths)
    optimized_hint_signature = _path_hint_signature(optimized_path_hints)
    return (
        str(render_variant),
        bool(include_image),
        bool(show_pre),
        round(float(source_smoothing), 4),
        round(float(curve_fit_tolerance), 4),
        round(float(endpoint_tangent_strictness), 4),
        bool(force_orthogonal_as_lines),
        bool(enable_curve_fitting),
        bool(enable_post_fit_export),
        bool(fit_optimized_paths),
        bool(combine_optimized_paths),
        bool(combine_pre_optimization_paths),
        int(coordinate_precision),
        bool(suppress_paths_in_view),
        bool(has_circle_system),
        int(optimization_generation),
        bool(optimization_complete),
        int(pre_count),
        int(pre_points),
        str(pre_signature),
        int(optimized_count),
        int(optimized_points),
        str(optimized_signature),
        str(optimized_hint_signature),
    )


def _get_cached_svg_render(session, cache_key):
    return session.svg_cache.get(cache_key)


def _store_cached_svg_render(session, cache_key, svg_content, suppressed, render_metadata=None):
    if len(session.svg_cache) >= 8:
        oldest_key = next(iter(session.svg_cache))
        session.svg_cache.pop(oldest_key, None)
    session.svg_cache[cache_key] = {
        'svg': svg_content,
        'preview_paths_suppressed': bool(suppressed),
        'render_metadata': dict(render_metadata or {}),
    }


def _build_svg_optimization_summary(
    render_metadata,
    *,
    enable_post_fit_export,
    fit_optimized_paths,
    enable_curve_fitting,
    force_orthogonal_as_lines,
    elapsed_ms,
    svg_content,
    cache_hit,
):
    metadata = dict(render_metadata or {})
    source_path_count = int(metadata.get('source_path_count', 0))
    source_point_count = int(metadata.get('source_point_count', 0))
    source_segment_count = int(metadata.get('source_segment_count', 0))
    output_path_count = int(metadata.get('output_path_count', source_path_count))
    output_segment_count = int(metadata.get('output_segment_count', source_segment_count))
    line_segment_count = int(metadata.get('line_segment_count', output_segment_count))
    bezier_segment_count = int(metadata.get('bezier_segment_count', 0))

    segment_delta = output_segment_count - source_segment_count
    segment_reduction_pct = 0.0
    if source_segment_count > 0:
        segment_reduction_pct = ((source_segment_count - output_segment_count) / source_segment_count) * 100.0

    if not enable_post_fit_export or not fit_optimized_paths:
        work_label = 'Refreshing direct path preview'
    elif enable_curve_fitting:
        work_label = 'Re-fitting cubic export geometry'
    elif force_orthogonal_as_lines:
        work_label = 'Refreshing line-only export geometry'
    else:
        work_label = 'Refreshing optimized path preview'

    geometry_summary = (
        f"{source_segment_count} source segments -> {output_segment_count} final segments"
        if source_segment_count > 0 or output_segment_count > 0
        else 'No vector segments generated'
    )

    return {
        'work_label': work_label,
        'geometry_summary': geometry_summary,
        'source_path_count': source_path_count,
        'output_path_count': output_path_count,
        'source_point_count': source_point_count,
        'source_segment_count': source_segment_count,
        'output_segment_count': output_segment_count,
        'line_segment_count': line_segment_count,
        'bezier_segment_count': bezier_segment_count,
        'segment_delta': int(segment_delta),
        'segment_reduction_pct': float(segment_reduction_pct),
        'elapsed_ms': float(elapsed_ms),
        'elapsed_sec': float(elapsed_ms / 1000.0),
        'svg_size_bytes': int(len(svg_content.encode('utf-8'))),
        'cache_hit': bool(cache_hit),
        'fit_optimized_paths': bool(fit_optimized_paths),
        'enable_curve_fitting': bool(enable_curve_fitting),
        'force_orthogonal_as_lines': bool(force_orthogonal_as_lines),
        'enable_post_fit_export': bool(enable_post_fit_export),
    }


def _render_svg_content(
    session,
    render_variant,
    background_image,
    circle_system,
    optimized_paths,
    pre_optimization_paths,
    optimized_path_hints,
    source_smoothing,
    curve_fit_tolerance,
    endpoint_tangent_strictness,
    force_orthogonal_as_lines,
    enable_curve_fitting,
    enable_post_fit_export,
    include_image,
    show_pre,
    suppress_paths_in_view=False,
    fit_optimized_paths=True,
    combine_optimized_paths=False,
    combine_pre_optimization_paths=False,
    coordinate_precision=2,
):
    cache_key = _svg_render_signature(
        render_variant,
        include_image,
        show_pre,
        source_smoothing,
        curve_fit_tolerance,
        endpoint_tangent_strictness,
        force_orthogonal_as_lines,
        enable_curve_fitting,
        enable_post_fit_export,
        fit_optimized_paths,
        combine_optimized_paths,
        combine_pre_optimization_paths,
        coordinate_precision,
        pre_optimization_paths,
        optimized_paths,
        optimized_path_hints,
        suppress_paths_in_view,
        circle_system is not None,
        session.optimization_generation,
        session.optimization_complete,
    )
    cached_render = _get_cached_svg_render(session, cache_key)
    if cached_render is not None:
        return (
            cached_render['svg'],
            cached_render['preview_paths_suppressed'],
            True,
            dict(cached_render.get('render_metadata') or {}),
        )

    import centerline_engine_light as wo
    original_show_bitmap = wo.SHOW_BITMAP
    original_show_pre = wo.SHOW_PRE_OPTIMIZATION_PATHS

    try:
        wo.SHOW_BITMAP = include_image
        wo.SHOW_PRE_OPTIMIZATION_PATHS = show_pre
        render_result = create_svg_output(
            background_image,
            circle_system,
            optimized_paths,
            [1.0] * len(optimized_paths),
            pre_optimization_paths,
            source_smoothing=source_smoothing,
            curve_fit_tolerance=curve_fit_tolerance,
            endpoint_tangent_strictness=endpoint_tangent_strictness,
            force_orthogonal_as_lines=force_orthogonal_as_lines,
            enable_curve_fitting=enable_curve_fitting,
            output_path=None,
            fit_optimized_paths=fit_optimized_paths,
            combine_optimized_paths=combine_optimized_paths,
            combine_pre_optimization_paths=combine_pre_optimization_paths,
            coordinate_precision=coordinate_precision,
            optimized_path_hints=optimized_path_hints,
            return_metadata=True,
        )
        if isinstance(render_result, tuple) and len(render_result) == 2:
            svg_content, render_metadata = render_result
        else:
            svg_content = render_result
            render_metadata = {}
    finally:
        wo.SHOW_BITMAP = original_show_bitmap
        wo.SHOW_PRE_OPTIMIZATION_PATHS = original_show_pre

    _store_cached_svg_render(session, cache_key, svg_content, suppress_paths_in_view, render_metadata)
    return svg_content, suppress_paths_in_view, False, render_metadata


def _resolve_svg_render_settings(parameters):
    render_parameters = dict(parameters or {})
    optimization_enabled = bool(render_parameters.get('enable_optimization', True))
    enable_post_fit_export = bool(render_parameters.get('enable_post_fit_export', False))
    return {
        'source_smoothing': max(
            0.0,
            min(100.0, float(render_parameters.get('source_smoothing', 0.0))),
        ),
        'curve_fit_tolerance': max(
            0.35,
            min(8.0, float(render_parameters.get('cubic_fit_tolerance', 1.0))),
        ),
        'endpoint_tangent_strictness': max(
            0.0,
            min(100.0, float(render_parameters.get('endpoint_tangent_strictness', 85.0))),
        ),
        'force_orthogonal_as_lines': bool(render_parameters.get('force_orthogonal_as_lines', False)),
        'enable_curve_fitting': bool(render_parameters.get('enable_curve_fitting', False)),
        'enable_post_fit_export': enable_post_fit_export,
        'include_image': bool(render_parameters.get('include_image', False)),
        'show_pre': optimization_enabled and bool(render_parameters.get('show_pre_optimization', False)),
        'fit_optimized_paths': optimization_enabled or enable_post_fit_export,
        'combine_optimized_paths': False,
        'combine_pre_optimization_paths': False,
        'coordinate_precision': 2,
    }


def _get_request_bool_arg(name, default=None):
    raw_value = request.args.get(name)
    if raw_value is None:
        return default
    cleaned = str(raw_value).strip().lower()
    if cleaned in {'1', 'true', 'yes', 'on'}:
        return True
    if cleaned in {'0', 'false', 'no', 'off'}:
        return False
    return default


def _benchmark_metrics_response(session):
    metrics = _ensure_benchmark_metrics(session)
    if metrics is None:
        return None

    payload = dict(metrics)
    payload['original_filename'] = session.original_filename
    payload['image_shape'] = list(session.image.shape) if session.image is not None else None
    payload['optimization_complete'] = bool(session.optimization_complete)
    payload['optimized_path_count'] = len(session.partial_optimized_paths)
    payload['has_results'] = bool(session.results and 'error' not in session.results)
    payload['summary'] = _build_benchmark_summary(session, metrics, payload)
    return payload


def process_centerlines(session):
    """Process centerlines with current parameters."""
    if session.image is None:
        return None
    
    params = session.parameters
    gray = session.image
    extraction_profile = _get_session_extraction_profile(session)
    effective_min_path_length = _effective_min_path_length(params, extraction_profile, image_shape=gray.shape)
    effective_full_min_object_size = _effective_min_object_size(
        extraction_profile['full_min_object_size'],
        params,
        extraction_profile,
        image_shape=gray.shape,
    )
    effective_preview_min_object_size = _effective_min_object_size(
        extraction_profile['preview_min_object_size'],
        params,
        extraction_profile,
        image_shape=gray.shape,
    )
    simplification_enabled = bool(params.get('enable_pruning', DEFAULT_ENABLE_PRUNING))
    pruning_debug_enabled = simplification_enabled and (
        bool(params.get('show_pruning_debug_grid', False)) or _should_force_pruning_debug(session.original_filename)
    )
    
    # Check if optimization is enabled to determine processing level
    optimization_enabled = params.get('enable_optimization', True)
    
    if optimization_enabled:
        # Full processing mode: use standard extraction
        initial_paths = extract_skeleton_paths(
            gray,
            params['dark_threshold'],
            min_object_size=effective_full_min_object_size,
            min_path_length=max(1, min(8, int(round(effective_min_path_length * 0.5)))),
        )
    else:
        # Fast mode: use the fast extraction function directly
        print("Fast extraction mode for unoptimized processing...")
        initial_paths = extract_skeleton_paths(
            gray,
            params['dark_threshold'],
            min_object_size=effective_preview_min_object_size,
            min_path_length=effective_min_path_length,
        )
    
    if len(initial_paths) == 0:
        return {'error': 'No skeleton paths found. Try adjusting the dark threshold.'}
    
    if optimization_enabled:
        # Full processing mode: merge paths and handle overlaps
        median_stroke_width_px = _estimate_median_stroke_width(
            gray,
            params['dark_threshold'],
            min_object_size=effective_full_min_object_size,
        )
        session.stroke_width_estimate_px = median_stroke_width_px
        merge_gap = _effective_merge_gap_from_stroke(
            params,
            image_shape=gray.shape,
            median_stroke_width_px=median_stroke_width_px,
        )
        merge_angle_priority = float(params.get('merge_angle_priority', 30.0)) / 100.0

        # Merge nearby paths
        merged_paths = merge_nearby_paths(
            initial_paths,
            max_gap=merge_gap,
            angle_priority=merge_angle_priority,
        )
        
        # Filter paths by minimum length
        valid_paths = [path for path in merged_paths if len(path) >= effective_min_path_length]
        
        # Remove overlapping paths if enabled
        if params.get('remove_overlaps', True):
            valid_paths = remove_overlapping_paths(
                valid_paths, 
                overlap_threshold=params.get('overlap_threshold', 0.3),
                min_distance=params.get('min_overlap_distance', 8)
            )

        if pruning_debug_enabled:
            valid_paths, pruning_debug_payload = prune_extracted_paths(
                valid_paths,
                min_output_points=2,
                return_diagnostics=True,
            )
        elif simplification_enabled:
            valid_paths = prune_extracted_paths(
                valid_paths,
                min_output_points=2,
            )
            pruning_debug_payload = None
        else:
            pruning_debug_payload = None
        
        merged_paths_count = len(merged_paths)
    else:
        # Raw mode: no merging, no overlap processing - just filter by length
        print("Raw skeleton mode: skipping path merging and overlap processing...")
        
        # Filter paths by minimum length only
        valid_paths = [path for path in initial_paths if len(path) >= effective_min_path_length]
        if pruning_debug_enabled:
            valid_paths, pruning_debug_payload = prune_extracted_paths(
                valid_paths,
                min_output_points=2,
                return_diagnostics=True,
            )
        elif simplification_enabled:
            valid_paths = prune_extracted_paths(
                valid_paths,
                min_output_points=2,
            )
            pruning_debug_payload = None
        else:
            pruning_debug_payload = None
        merged_paths_count = len(initial_paths)  # Use initial count since no merging

    valid_path_hints = list((pruning_debug_payload or {}).get('path_hints') or [build_line_span_hints(path) for path in valid_paths])
    
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
            'optimized_path_hints': valid_path_hints,
            'pre_optimization_path_hints': valid_path_hints,
            'circle_system': None,  # No circle system in fast mode
            'pruning_debug': _serialize_pruning_debug_payload(pruning_debug_payload),
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
    optimized_path_hints = []
    
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
        optimized_path_hints.append(build_line_span_hints(optimized_path))
    
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
        'optimized_path_hints': optimized_path_hints,
        'pre_optimization_path_hints': [valid_path_hints[idx] for idx in sorted_indices],
        'circle_system': circle_system,
        'pruning_debug': _serialize_pruning_debug_payload(pruning_debug_payload),
    }
    
    return results

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
        extraction_profile = _get_session_extraction_profile(session)
        effective_max_min_path_length = extraction_profile.get('max_min_path_length')
        parameter_scale = resolve_parameter_scale(session.image.shape)
        effective_auto_tune_min_object_size = _effective_min_object_size(
            extraction_profile['auto_tune_min_object_size'],
            session.parameters,
            extraction_profile,
            image_shape=session.image.shape,
        )
        
        # Run auto-detection
        detection_result = auto_detect_min_path_length(
            session.image,
            current_threshold,
            min_object_size=effective_auto_tune_min_object_size,
            parameter_scale=parameter_scale,
        )
        
        if detection_result['best_score'] > 0:
            # Update session parameters with detected min path length
            detected_min_length = int(detection_result['best_min_length'])
            if effective_max_min_path_length is not None:
                detected_min_length = min(detected_min_length, int(effective_max_min_path_length))
            session.parameters['min_path_length'] = detected_min_length
            
            return jsonify({
                'success': True,
                'detected_min_length': detected_min_length,
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
        extraction_profile = _get_session_extraction_profile(session)
        parameter_scale = resolve_parameter_scale(session.image.shape)
        effective_auto_tune_min_object_size = _effective_min_object_size(
            extraction_profile['auto_tune_min_object_size'],
            session.parameters,
            extraction_profile,
            image_shape=session.image.shape,
        )
        # Run auto-detection
        detection_result = auto_detect_dark_threshold(
            session.image,
            min_object_size=effective_auto_tune_min_object_size,
            parameter_scale=parameter_scale,
        )
        
        if detection_result['best_score'] > 0:
            # Update session parameters with detected threshold
            session.parameters['dark_threshold'] = detection_result['best_threshold']
            session.parameters['enable_optimization'] = False
            session.parameters['show_pre_optimization'] = True
            sample_metadata = _augment_sample_metadata(
                detection_result.get('sample_metadata'),
                session.image.shape,
                logical_min_path_length=3,
            )
            
            return jsonify({
                'success': True,
                'detected_threshold': detection_result['best_threshold'],
                'confidence_score': detection_result['best_score'],
                'recommendation': detection_result['recommendation'],
                'elapsed_sec': float(detection_result.get('elapsed_sec', 0.0)),
                'thresholds_evaluated': int(detection_result.get('thresholds_evaluated', 0)),
                'preview_shape': detection_result.get('preview_shape'),
                'preview_scale': float(detection_result.get('preview_scale', 1.0)),
                'sample_metadata': sample_metadata,
                'updated_parameters': session.parameters
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Could not detect suitable threshold. Manual adjustment recommended.',
                'detected_threshold': detection_result['best_threshold'],
                'recommendation': detection_result['recommendation'],
                'elapsed_sec': float(detection_result.get('elapsed_sec', 0.0)),
                'thresholds_evaluated': int(detection_result.get('thresholds_evaluated', 0)),
                'preview_shape': detection_result.get('preview_shape'),
                'preview_scale': float(detection_result.get('preview_scale', 1.0)),
                'sample_metadata': _augment_sample_metadata(
                    detection_result.get('sample_metadata'),
                    session.image.shape,
                    logical_min_path_length=3,
                ),
            })
            
    except Exception as e:
        return jsonify({'error': f'Auto-detection error: {str(e)}'})


def _run_auto_detect_threshold_job(session, auto_detect_generation):
    """Background threshold auto-detect job with cancellable progress state."""
    try:
        source_image = session.image
        extraction_profile = _get_session_extraction_profile(session)
        parameter_scale = resolve_parameter_scale(source_image.shape)

        def _should_continue_auto_detect():
            with session.lock:
                return session.auto_detect_generation == auto_detect_generation

        def _record_progress(progress_payload):
            with session.lock:
                if session.auto_detect_generation != auto_detect_generation:
                    return

                sample_metadata = _augment_sample_metadata(
                    progress_payload.get('sample_metadata'),
                    source_image.shape,
                    logical_min_path_length=3,
                )
                best_threshold = float(progress_payload.get('best_threshold') or session.parameters.get('dark_threshold', 0.20))
                thresholds_evaluated = int(progress_payload.get('thresholds_evaluated', 0))
                thresholds_planned = int(progress_payload.get('thresholds_planned', 0))
                elapsed_sec = float(progress_payload.get('elapsed_sec', 0.0))
                session.auto_detect_progress.update({
                    'running': True,
                    'finished': False,
                    'success': False,
                    'cancelled': bool(progress_payload.get('cancelled', False)),
                    'elapsed_sec': elapsed_sec,
                    'confidence_score': float(progress_payload.get('confidence_score', 0.0)),
                    'detected_threshold': best_threshold,
                    'best_threshold': best_threshold,
                    'recommendation': 'evaluating thresholds',
                    'thresholds_evaluated': thresholds_evaluated,
                    'thresholds_planned': thresholds_planned,
                    'preview_shape': progress_payload.get('preview_shape'),
                    'preview_scale': float(progress_payload.get('preview_scale', 1.0)),
                    'sample_metadata': sample_metadata,
                    'message': (
                        f"Auto-detect running: tested {thresholds_evaluated}"
                        + (f" / {thresholds_planned}" if thresholds_planned > 0 else '')
                        + f" thresholds in {elapsed_sec:.1f}s. Best so far {best_threshold:.3f}."
                    ),
                })

        detection_result = auto_detect_dark_threshold(
            source_image,
            min_object_size=_effective_min_object_size(
                extraction_profile['auto_tune_min_object_size'],
                session.parameters,
                extraction_profile,
                image_shape=source_image.shape,
            ),
            parameter_scale=parameter_scale,
            should_continue=_should_continue_auto_detect,
            on_progress=_record_progress,
        )

        sample_metadata = _augment_sample_metadata(
            detection_result.get('sample_metadata'),
            source_image.shape,
            logical_min_path_length=3,
        )

        with session.lock:
            if session.auto_detect_generation != auto_detect_generation:
                return
            session.auto_detect_active = False

        if bool(detection_result.get('cancelled', False)):
            with session.lock:
                if session.auto_detect_generation != auto_detect_generation:
                    return
                session.auto_detect_progress.update({
                    'running': False,
                    'finished': True,
                    'success': False,
                    'cancelled': True,
                    'elapsed_sec': float(detection_result.get('elapsed_sec', 0.0)),
                    'confidence_score': float(detection_result.get('best_score', 0.0)),
                    'detected_threshold': float(session.parameters.get('dark_threshold', 0.20)),
                    'best_threshold': float(detection_result.get('best_threshold', session.parameters.get('dark_threshold', 0.20))),
                    'recommendation': 'cancelled',
                    'thresholds_evaluated': int(detection_result.get('thresholds_evaluated', 0)),
                    'preview_shape': detection_result.get('preview_shape'),
                    'preview_scale': float(detection_result.get('preview_scale', 1.0)),
                    'sample_metadata': sample_metadata,
                    'message': 'Auto-detect cancelled. Previous threshold kept.',
                    'updated_parameters': dict(session.parameters),
                })
            return

        if detection_result['best_score'] > 0:
            session.parameters['dark_threshold'] = float(detection_result['best_threshold'])
            session.parameters['enable_optimization'] = False
            session.parameters['show_pre_optimization'] = True
            success = True
            status_message = f"Auto-detect complete. Threshold set to {float(detection_result['best_threshold']):.3f}."
        else:
            success = False
            status_message = 'Could not detect a suitable threshold. Previous value kept.'

        with session.lock:
            if session.auto_detect_generation != auto_detect_generation:
                return
            session.auto_detect_progress.update({
                'running': False,
                'finished': True,
                'success': bool(success),
                'cancelled': False,
                'elapsed_sec': float(detection_result.get('elapsed_sec', 0.0)),
                'confidence_score': float(detection_result.get('best_score', 0.0)),
                'detected_threshold': float(detection_result.get('best_threshold', session.parameters.get('dark_threshold', 0.20))),
                'best_threshold': float(detection_result.get('best_threshold', session.parameters.get('dark_threshold', 0.20))),
                'recommendation': str(detection_result.get('recommendation', 'manual adjustment recommended')),
                'thresholds_evaluated': int(detection_result.get('thresholds_evaluated', 0)),
                'thresholds_planned': int(max(session.auto_detect_progress.get('thresholds_planned', 0), detection_result.get('thresholds_evaluated', 0))),
                'preview_shape': detection_result.get('preview_shape'),
                'preview_scale': float(detection_result.get('preview_scale', 1.0)),
                'sample_metadata': sample_metadata,
                'message': status_message,
                'updated_parameters': dict(session.parameters),
            })
    except Exception as e:
        with session.lock:
            if session.auto_detect_generation != auto_detect_generation:
                return
            session.auto_detect_active = False
            session.auto_detect_progress.update({
                'running': False,
                'finished': True,
                'success': False,
                'cancelled': False,
                'message': f'Auto-detect failed: {str(e)}',
                'updated_parameters': dict(session.parameters),
            })


@app.route('/auto_detect_threshold_start', methods=['POST'])
def auto_detect_threshold_start():
    """Start threshold auto-detect in the background for cancellable UI polling."""
    data = request.json or {}
    session_id = data.get('session_id')

    if session_id not in sessions:
        return jsonify({'error': 'Invalid session'})

    session = sessions[session_id]

    if session.image is None:
        return jsonify({'error': 'No image loaded'})

    with session.lock:
        session.auto_detect_generation += 1
        auto_detect_generation = session.auto_detect_generation
        session.auto_detect_active = True
        session.auto_detect_progress = {
            'running': True,
            'started_at': float(time.perf_counter()),
            'finished': False,
            'success': False,
            'cancelled': False,
            'elapsed_sec': 0.0,
            'confidence_score': 0.0,
            'detected_threshold': float(session.parameters.get('dark_threshold', 0.20)),
            'best_threshold': float(session.parameters.get('dark_threshold', 0.20)),
            'recommendation': 'evaluating thresholds',
            'thresholds_evaluated': 0,
            'thresholds_planned': 8,
            'preview_shape': [],
            'preview_scale': 1.0,
            'sample_metadata': {'sampled': False},
            'message': 'Auto-detect started. Evaluating threshold candidates...',
            'updated_parameters': dict(session.parameters),
        }

        worker = threading.Thread(
            target=_run_auto_detect_threshold_job,
            args=(session, auto_detect_generation),
            daemon=True,
        )
        session.auto_detect_thread = worker

    worker.start()

    return jsonify({'success': True, 'started': True})


@app.route('/auto_detect_threshold_progress/<session_id>', methods=['GET'])
def auto_detect_threshold_progress(session_id):
    """Return live threshold auto-detect progress state."""
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session'})

    session = sessions[session_id]
    with session.lock:
        progress = dict(session.auto_detect_progress)
        progress['active'] = bool(session.auto_detect_active)

    if progress.get('running'):
        started_at = float(progress.get('started_at', 0.0) or 0.0)
        if started_at > 0.0:
            progress['elapsed_sec'] = max(float(progress.get('elapsed_sec', 0.0)), float(time.perf_counter() - started_at))

    return jsonify(progress)

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

    extraction_profile = _get_session_extraction_profile(session)

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
        parameter_scale = resolve_parameter_scale(session.image.shape)
        if max(image_height, image_width) > AUTO_TUNE_RANDOM_TILE_DIM:
            tuning_image, sample_metadata = _build_random_tile_mosaic(session.image)
        sample_metadata = _augment_sample_metadata(
            sample_metadata,
            session.image.shape,
            logical_min_path_length=int(session.parameters.get('min_path_length', DEFAULT_MIN_PATH_LENGTH)),
            logical_merge_gap=int(session.parameters.get('merge_gap', 25)),
        )

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
            parameter_scale=parameter_scale,
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
            session.parameters['min_path_length'] = DEFAULT_MIN_PATH_LENGTH
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
        session.parameters['enable_optimization'] = False
        session.parameters['show_pre_optimization'] = True
        _apply_auto_tuned_merge_gap(
            session,
            dark_threshold=tuning_result['best_threshold'],
            gray_image=session.image,
            min_object_size=_effective_min_object_size(
                extraction_profile['auto_tune_min_object_size'],
                session.parameters,
                extraction_profile,
                image_shape=session.image.shape,
            ),
        )

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
        session.parameters['min_path_length'] = DEFAULT_MIN_PATH_LENGTH
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
        _, sample_metadata = _build_random_tile_mosaic(sample_source)
        sample_metadata = _offset_sample_metadata_to_source(
            sample_metadata,
            offset_x=offset_x,
            offset_y=offset_y,
            source_shape=session.image.shape,
        )
    sample_metadata = _augment_sample_metadata(
        sample_metadata,
        session.image.shape,
        logical_min_path_length=int(session.parameters.get('min_path_length', DEFAULT_MIN_PATH_LENGTH)),
        logical_merge_gap=int(session.parameters.get('merge_gap', 25)),
    )

    return jsonify({
        'success': True,
        'sample_metadata': sample_metadata,
    })


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
    extraction_profile = _get_session_extraction_profile(session)

    try:
        dark_threshold = float(data.get('dark_threshold', session.parameters.get('dark_threshold', 0.20)))
        min_path_length = int(data.get('min_path_length', session.parameters.get('min_path_length', DEFAULT_MIN_PATH_LENGTH)))
    except Exception:
        return jsonify({'error': 'Invalid threshold or min path length'})

    effective_merge_gap = _effective_merge_gap_from_stroke(
        session.parameters,
        image_shape=session.image.shape,
        median_stroke_width_px=session.stroke_width_estimate_px,
    )

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
    sample_metadata = _augment_sample_metadata(
        sample_metadata,
        session.image.shape,
        logical_min_path_length=min_path_length,
        logical_merge_gap=int(session.parameters.get('merge_gap', 25)),
    )
    sample_metadata = _attach_merge_gap_metadata(
        sample_metadata,
        logical_merge_gap=int(session.parameters.get('merge_gap', 25)),
        effective_merge_gap=effective_merge_gap,
        median_stroke_width_px=session.stroke_width_estimate_px,
    )

    preview_started_at = time.perf_counter()
    full_image_preview = bool(data.get('full_image_preview', False))
    effective_min_path_length = _effective_min_path_length(
        {'min_path_length': min_path_length},
        extraction_profile,
        image_shape=session.image.shape,
    )

    tile_shape = sample_metadata.get('tile_shape')
    tile_origins = sample_metadata.get('tile_origins') or [[0, 0]]
    if not (isinstance(tile_shape, (list, tuple)) and len(tile_shape) >= 2):
        tile_shape = sample_metadata.get('sample_shape') or list(session.image.shape)
    tile_height = int(tile_shape[0] or session.image.shape[0])
    tile_width = int(tile_shape[1] or session.image.shape[1])
    resolved_min_object_size = _effective_min_object_size(
        extraction_profile['auto_tune_min_object_size'],
        {'min_path_length': min_path_length},
        extraction_profile,
        image_shape=session.image.shape,
    )

    tile_data = []
    cache_hits = 0
    cache_misses = 0

    for tile_index, origin in enumerate(tile_origins):
        if not isinstance(origin, (list, tuple)) or len(origin) < 2:
            continue
        origin_x = int(origin[0])
        origin_y = int(origin[1])
        cache_key = (
            int(origin_x),
            int(origin_y),
            int(tile_width),
            int(tile_height),
            round(float(dark_threshold), 6),
            int(effective_min_path_length),
            int(effective_merge_gap),
            int(resolved_min_object_size),
        )

        with session.lock:
            cache_has_key = cache_key in session.tile_preview_cache
            cached_tile_entry = session.tile_preview_cache.get(cache_key)

        if cache_has_key:
            cache_hits += 1
            tile_entry = cached_tile_entry
        else:
            cache_misses += 1
            tile_entry = _extract_single_tile_preview_entry(
                session.image,
                tile_index=tile_index,
                origin_x=origin_x,
                origin_y=origin_y,
                tile_width=tile_width,
                tile_height=tile_height,
                dark_threshold=dark_threshold,
                min_path_length=effective_min_path_length,
                merge_gap=effective_merge_gap,
                min_object_size=resolved_min_object_size,
                full_resolution=full_image_preview,
            )
            with session.lock:
                if len(session.tile_preview_cache) >= TILE_PREVIEW_CACHE_LIMIT:
                    session.tile_preview_cache.clear()
                session.tile_preview_cache[cache_key] = tile_entry

        if tile_entry is not None:
            tile_data.append(tile_entry)

    total_valid = sum(int(tile.get('valid_count', 0)) for tile in tile_data)
    total_orphan = sum(int(tile.get('orphan_count', 0)) for tile in tile_data)
    total_length = sum(float(tile.get('total_length', 0)) for tile in tile_data)
    stats = {
        'total_valid_paths': total_valid,
        'total_orphan_paths': total_orphan,
        'total_path_length': total_length,
        'tile_count': len(tile_data),
        'noise_ratio': round(total_orphan / max(1, total_valid + total_orphan), 3),
        'cached_tile_count': int(cache_hits),
        'computed_tile_count': int(cache_misses),
    }
    preview_runtime_ms = int(round((time.perf_counter() - preview_started_at) * 1000.0))

    return jsonify({
        'success': True,
        'sample_metadata': sample_metadata,
        'tile_data': tile_data,
        'tile_path_count': sum(len(t.get('paths', [])) for t in tile_data),
        'stats': stats,
        'cached_tile_count': int(cache_hits),
        'computed_tile_count': int(cache_misses),
        'preview_runtime_ms': preview_runtime_ms,
        'source_shape': list(session.image.shape),
        'dark_threshold': float(dark_threshold),
        'min_path_length': int(min_path_length),
        'effective_min_path_length': int(effective_min_path_length),
        'effective_merge_gap': int(effective_merge_gap),
        'median_stroke_width_px': float(session.stroke_width_estimate_px) if session.stroke_width_estimate_px is not None else None,
    })


def _run_auto_tune_job(session, auto_tune_generation):
    """Background auto-tune job that continuously updates session.auto_tune_progress."""
    try:
        source_image = session.image
        parameter_scale = resolve_parameter_scale(source_image.shape)
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
            tuning_image, sample_metadata = _build_random_tile_mosaic(tuning_image)
            sample_metadata = _offset_sample_metadata_to_source(
                sample_metadata,
                offset_x=offset_x,
                offset_y=offset_y,
                source_shape=source_image.shape,
            )
        sample_metadata = _augment_sample_metadata(
            sample_metadata,
            source_image.shape,
            logical_min_path_length=int(session.parameters.get('min_path_length', DEFAULT_MIN_PATH_LENGTH)),
            logical_merge_gap=int(session.parameters.get('merge_gap', 25)),
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

        extraction_profile = _get_session_extraction_profile(session)
        effective_max_min_path_length = extraction_profile.get('max_min_path_length')
        base_min_lengths = [1, 3, 5, 8, 12]
        if effective_max_min_path_length is not None:
            base_min_lengths = [length for length in base_min_lengths if length <= int(effective_max_min_path_length)]
            if not base_min_lengths:
                base_min_lengths = [max(1, int(effective_max_min_path_length))]

        tuning_result = auto_tune_extraction_parameters(
            tuning_image,
            threshold_range=(0.05, 0.8),
            num_thresholds=6,
            base_min_lengths=base_min_lengths,
            preview_max_dim=550,
            time_budget_sec=AUTO_TUNE_TIME_BUDGET_SEC,
            sample_metadata=sample_metadata,
            should_continue=_should_continue_auto_tune,
            on_best_result=_record_best_so_far,
            on_progress=_record_progress,
            confidence_target=AUTO_TUNE_CONFIDENCE_TARGET,
            min_object_size=_effective_min_object_size(
                extraction_profile['auto_tune_min_object_size'],
                session.parameters,
                extraction_profile,
                image_shape=tuning_image.shape,
            ),
            parameter_scale=parameter_scale,
        )

        with session.lock:
            if session.auto_tune_generation != auto_tune_generation:
                return
            session.auto_tune_active = False
            session.auto_tune_best = dict(tuning_result)

        if tuning_result['quality_score'] <= 0 or tuning_result.get('timed_out'):
            session.parameters['dark_threshold'] = 0.20
            session.parameters['min_path_length'] = DEFAULT_MIN_PATH_LENGTH
            session.parameters['enable_optimization'] = False
            session.parameters['show_pre_optimization'] = True
            success = False
            status_message = 'Failed to detect best settings. Using defaults with optimization turned off.'
        else:
            session.parameters['dark_threshold'] = tuning_result['best_threshold']
            detected_min_length = int(tuning_result['best_min_length'])
            if effective_max_min_path_length is not None:
                detected_min_length = min(detected_min_length, int(effective_max_min_path_length))
            session.parameters['min_path_length'] = detected_min_length
            session.parameters['enable_optimization'] = False
            session.parameters['show_pre_optimization'] = True
            stroke_measure_image = source_image
            if crop_bounds is not None:
                stroke_measure_image = source_image[crop_bounds['y1']:crop_bounds['y2'], crop_bounds['x1']:crop_bounds['x2']]
            _apply_auto_tuned_merge_gap(
                session,
                dark_threshold=tuning_result['best_threshold'],
                gray_image=stroke_measure_image,
                min_object_size=_effective_min_object_size(
                    extraction_profile['auto_tune_min_object_size'],
                    session.parameters,
                    extraction_profile,
                    image_shape=stroke_measure_image.shape,
                ),
            )
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
            session.parameters['min_path_length'] = DEFAULT_MIN_PATH_LENGTH
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
            'best_min_length': int(session.parameters.get('min_path_length', DEFAULT_MIN_PATH_LENGTH)),
            'timed_out': False,
            'cancelled': False,
            'high_confidence_reached': False,
            'message': 'Auto-tune started. Gathering candidates...',
            'sample_metadata': {'sampled': False},
            'live_paths': [],
            'live_paths_current': [],
            'live_paths_current_threshold': 0.0,
            'live_paths_current_min_length': int(session.parameters.get('min_path_length', DEFAULT_MIN_PATH_LENGTH)),
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


@app.route('/test-ui')
def test_ui():
    """Dedicated UI for pytest run management and fixture-layer inspection."""
    if not TEST_UI_ENABLED:
        return render_template('index.html')
    return render_template(
        'test_management.html',
        fixture_ids=ALL_FIXTURE_IDS,
        fitting_defaults=DEFAULT_FITTING_PARAMETERS,
    )


@app.route('/api/test-runs', methods=['GET'])
def api_list_test_runs():
    """Return known persisted and in-memory test runs."""
    if not TEST_UI_ENABLED:
        return _test_ui_unavailable_response()
    return jsonify({'runs': list_runs()})


@app.route('/api/test-runs/execute', methods=['POST'])
def api_execute_test_run():
    """Start a new background pytest run with visualization artifacts."""
    if not TEST_UI_ENABLED:
        return _test_ui_unavailable_response()
    data = request.json or {}
    update_goldens = bool(data.get('update_goldens', False))
    fixture_ids = data.get('fixture_ids', None)
    fitting_parameters = data.get('fitting_parameters', None)
    if fixture_ids is not None and not isinstance(fixture_ids, list):
        return jsonify({'error': 'fixture_ids must be a list of fixture IDs'}), 400
    if fitting_parameters is not None and not isinstance(fitting_parameters, dict):
        return jsonify({'error': 'fitting_parameters must be an object'}), 400

    try:
        started = create_run(
            update_goldens=update_goldens,
            fixture_ids=fixture_ids,
            fitting_parameters=fitting_parameters,
        )
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    return jsonify({'success': True, **started})


@app.route('/api/test-runs/<run_id>', methods=['GET'])
def api_get_test_run(run_id):
    """Fetch run summary and optional fixture detail payload."""
    if not TEST_UI_ENABLED:
        return _test_ui_unavailable_response()
    summary = get_run_summary(run_id)
    if summary is None:
        return jsonify({'error': 'Unknown run_id'}), 404

    fixture_id = request.args.get('fixture_id', '').strip()
    if fixture_id:
        detail = get_fixture_detail(run_id, fixture_id)
        if detail is None:
            return jsonify({'error': f'No detail found for fixture {fixture_id!r}'}), 404
        return jsonify({'summary': summary, 'fixture': detail})

    return jsonify({'summary': summary})


@app.route('/api/test-runs/<run_id>/progress', methods=['GET'])
def api_get_test_run_progress(run_id):
    """Poll progress/log messages for a specific run."""
    if not TEST_UI_ENABLED:
        return _test_ui_unavailable_response()
    progress = get_run_progress(run_id)
    if progress is None:
        return jsonify({'error': 'Unknown run_id'}), 404
    return jsonify(progress)


def _is_supported_upload_source(mime_type=None, source_name=None):
    cleaned_mime_type = str(mime_type or '').split(';', 1)[0].strip().lower()
    cleaned_name = os.path.basename(str(source_name or '')).lower()
    return (
        cleaned_mime_type.startswith('image/')
        or cleaned_mime_type == 'application/pdf'
        or cleaned_name.endswith('.pdf')
    )


def _guess_remote_source_name(remote_url, content_type=None):
    parsed = urlsplit(str(remote_url or ''))
    candidate = os.path.basename(unquote(parsed.path or '')).strip()
    if candidate and candidate not in ('.', '..'):
        return candidate

    cleaned_mime_type = str(content_type or '').split(';', 1)[0].strip().lower()
    extension = mimetypes.guess_extension(cleaned_mime_type) or ''
    if extension == '.jpe':
        extension = '.jpg'
    if not extension and cleaned_mime_type.startswith('image/'):
        extension = '.png'

    return f'dropped-image{extension}'


class _RemoteFetchError(Exception):
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.status_code = int(status_code)


class _RemoteMediaCandidateParser(HTMLParser):
    def __init__(self, base_url):
        super().__init__()
        self.base_url = str(base_url or '')
        self.candidates = []
        self._seen = set()

    def _append_candidate(self, candidate):
        normalized = str(candidate or '').strip()
        if not normalized:
            return
        resolved = urljoin(self.base_url, normalized)
        parsed = urlsplit(resolved)
        if parsed.scheme.lower() not in {'http', 'https'}:
            return
        if resolved in self._seen:
            return
        self._seen.add(resolved)
        self.candidates.append(resolved)

    def _append_srcset(self, srcset_value):
        for item in str(srcset_value or '').split(','):
            candidate = item.strip().split(' ', 1)[0].strip()
            if candidate:
                self._append_candidate(candidate)

    def handle_starttag(self, tag, attrs):
        attr_map = {str(key).lower(): value for key, value in attrs}
        lowered_tag = str(tag).lower()

        if lowered_tag == 'meta':
            marker = (
                attr_map.get('property')
                or attr_map.get('name')
                or attr_map.get('itemprop')
                or ''
            ).strip().lower()
            if marker in {'og:image', 'og:image:url', 'twitter:image', 'twitter:image:src', 'image'}:
                self._append_candidate(attr_map.get('content'))
            return

        if lowered_tag == 'link':
            rel_tokens = {token.strip().lower() for token in str(attr_map.get('rel') or '').split() if token.strip()}
            if 'image_src' in rel_tokens or ('preload' in rel_tokens and str(attr_map.get('as') or '').strip().lower() == 'image'):
                self._append_candidate(attr_map.get('href'))
            return

        if lowered_tag in {'img', 'source'}:
            for key in ('src', 'data-src', 'data-original', 'data-lazy-src', 'data-url'):
                self._append_candidate(attr_map.get(key))
            self._append_srcset(attr_map.get('srcset'))


def _looks_like_html_payload(content_type, payload):
    cleaned_content_type = str(content_type or '').split(';', 1)[0].strip().lower()
    if cleaned_content_type in {'text/html', 'application/xhtml+xml'}:
        return True
    snippet = bytes(payload[:256] if payload else b'').lstrip().lower()
    return snippet.startswith(b'<!doctype html') or snippet.startswith(b'<html')


def _extract_remote_media_candidate_urls(html_bytes, base_url):
    try:
        html_text = bytes(html_bytes or b'').decode('utf-8', errors='ignore')
    except Exception:
        return []

    parser = _RemoteMediaCandidateParser(base_url)
    try:
        parser.feed(html_text)
        parser.close()
    except Exception:
        return list(parser.candidates)
    return list(parser.candidates)


def _fetch_remote_payload(remote_url, referer=None):
    try:
        headers = {
            'User-Agent': 'CenterlineTracingWebApp/1.0',
            'Accept': 'image/*,application/pdf,text/html;q=0.8,*/*;q=0.1',
        }
        if referer:
            headers['Referer'] = str(referer)
        remote_request = Request(str(remote_url), headers=headers)
        with urlopen(remote_request, timeout=REMOTE_UPLOAD_TIMEOUT_SEC) as response:
            response_headers = getattr(response, 'headers', {}) or {}
            content_type = response_headers.get('Content-Type', '')
            content_length = response_headers.get('Content-Length')
            if content_length is not None:
                try:
                    if int(content_length) > int(app.config['MAX_CONTENT_LENGTH']):
                        raise _RemoteFetchError('Remote file exceeds the 16MB upload limit', status_code=413)
                except (TypeError, ValueError):
                    pass

            upload_bytes = response.read(int(app.config['MAX_CONTENT_LENGTH']) + 1)
            final_url = response.geturl() or str(remote_url)

        if len(upload_bytes) > int(app.config['MAX_CONTENT_LENGTH']):
            raise _RemoteFetchError('Remote file exceeds the 16MB upload limit', status_code=413)

        return upload_bytes, final_url, content_type
    except HTTPError as exc:
        raise _RemoteFetchError(f'Remote server returned HTTP {exc.code}') from exc
    except URLError as exc:
        raise _RemoteFetchError(f'Could not fetch dropped URL: {exc.reason}') from exc


def _build_upload_response(upload_bytes, source_name, normalization_mode, normalization_sensitivity):
    cleaned_source_name = os.path.basename(str(source_name or '')).strip() or 'uploaded-file'
    if not upload_bytes:
        return jsonify({'error': 'Uploaded file is empty'})

    session_id = str(uuid.uuid4())
    session = CenterlineSession(session_id)
    session.parameters['normalization_mode'] = normalization_mode
    session.parameters['normalization_sensitivity'] = normalization_sensitivity
    session.original_filename = cleaned_source_name
    session.benchmark_enabled = _should_capture_benchmark_metrics(cleaned_source_name)
    session.image_file_size = len(upload_bytes)
    _ensure_benchmark_metrics(session)

    try:
        load_started_at = time.perf_counter()
        raw_gray, extraction_gray, preprocessing_info = load_and_process_image(
            upload_bytes,
            normalization_mode=normalization_mode,
            normalization_sensitivity=normalization_sensitivity,
            source_name=cleaned_source_name,
        )
        load_elapsed_ms = (time.perf_counter() - load_started_at) * 1000.0
        session.image = extraction_gray
        session.display_image = raw_gray
        session.preprocessing_info = preprocessing_info
        session.image_path = None
        extraction_profile = _get_session_extraction_profile(session)
        session.stroke_width_estimate_px = _estimate_median_stroke_width(
            extraction_gray,
            session.parameters['dark_threshold'],
            min_object_size=_effective_min_object_size(
                extraction_profile['preview_min_object_size'],
                session.parameters,
                extraction_profile,
                image_shape=extraction_gray.shape,
            ),
        )
        session.preprocessing_info = dict(preprocessing_info)
        if session.stroke_width_estimate_px is not None:
            session.preprocessing_info['median_stroke_width_px'] = float(session.stroke_width_estimate_px)
            session.preprocessing_info['effective_merge_gap'] = int(_effective_merge_gap_from_stroke(
                session.parameters,
                image_shape=extraction_gray.shape,
                median_stroke_width_px=session.stroke_width_estimate_px,
            ))
        _record_benchmark_stage(session, 'load_and_preprocess', load_elapsed_ms, {
            'raw_shape': list(raw_gray.shape),
            'normalization_applied': bool(session.preprocessing_info.get('applied', False)),
            'loaded_from_memory': True,
        })
        metrics = _ensure_benchmark_metrics(session)
        if metrics is not None:
            metrics['preprocessing'] = dict(session.preprocessing_info)

        sessions[session_id] = session

        preview_encode_started_at = time.perf_counter()
        raw_img = Image.fromarray((raw_gray * 255).astype(np.uint8))
        raw_buf = BytesIO()
        raw_img.save(raw_buf, format='PNG')
        raw_buf.seek(0)
        raw_img_data = base64.b64encode(raw_buf.read()).decode('utf-8')

        normalization_applied = bool(session.preprocessing_info.get('applied', False))
        if normalization_applied:
            normalized_img = Image.fromarray((extraction_gray * 255).astype(np.uint8))
            normalized_buf = BytesIO()
            normalized_img.save(normalized_buf, format='PNG')
            normalized_buf.seek(0)
            normalized_img_data = base64.b64encode(normalized_buf.read()).decode('utf-8')
        else:
            normalized_img_data = raw_img_data

        preview_encode_elapsed_ms = (time.perf_counter() - preview_encode_started_at) * 1000.0
        _record_benchmark_stage(session, 'preview_image_encoding', preview_encode_elapsed_ms, {
            'raw_image_b64_chars': len(raw_img_data),
            'normalized_image_b64_chars': len(normalized_img_data),
            'encoded_variant_count': 2 if normalization_applied else 1,
            'reused_preview_image': not normalization_applied,
        })

        default_img_data = normalized_img_data if normalization_applied else raw_img_data

        response_payload = {
            'success': True,
            'session_id': session_id,
            'image_data': default_img_data,
            'normalization_applied': normalization_applied,
            'image_shape': raw_gray.shape,
            'parameters': session.parameters,
            'preprocessing': session.preprocessing_info,
        }
        if normalization_applied:
            response_payload['raw_image_data'] = raw_img_data
        if session.benchmark_enabled:
            response_payload['benchmark_enabled'] = True
            response_payload['benchmark_metrics_url'] = f"/benchmark_metrics/{session_id}"

        return jsonify(response_payload)
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'})

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    requested_mode = request.form.get('normalization_mode', 'auto')
    requested_sensitivity = request.form.get('normalization_sensitivity', 'medium')
    upload_bytes = file.read()
    if not _is_supported_upload_source(getattr(file, 'mimetype', None), file.filename):
        return jsonify({'error': 'Please upload an image or PDF file.'}), 400

    return _build_upload_response(upload_bytes, file.filename, requested_mode, requested_sensitivity)


@app.route('/upload_remote', methods=['POST'])
def upload_remote_file():
    """Handle remote image or PDF upload from a dropped webpage URL."""
    data = request.json or {}
    remote_url = str(data.get('url', '')).strip()
    if not remote_url:
        return jsonify({'error': 'No remote URL provided'}), 400

    parsed_url = urlsplit(remote_url)
    if parsed_url.scheme.lower() not in {'http', 'https'}:
        return jsonify({'error': 'Only http and https URLs are supported'}), 400

    requested_mode = str(data.get('normalization_mode', 'auto'))
    requested_sensitivity = str(data.get('normalization_sensitivity', 'medium'))

    try:
        upload_bytes, final_url, content_type = _fetch_remote_payload(remote_url)

        source_name = _guess_remote_source_name(final_url, content_type=content_type)
        if _is_supported_upload_source(content_type, source_name):
            return _build_upload_response(upload_bytes, source_name, requested_mode, requested_sensitivity)

        if _looks_like_html_payload(content_type, upload_bytes):
            candidate_urls = _extract_remote_media_candidate_urls(upload_bytes, final_url)
            for candidate_url in candidate_urls[:8]:
                try:
                    candidate_bytes, candidate_final_url, candidate_content_type = _fetch_remote_payload(candidate_url, referer=final_url)
                except _RemoteFetchError:
                    continue

                candidate_name = _guess_remote_source_name(candidate_final_url, content_type=candidate_content_type)
                if _is_supported_upload_source(candidate_content_type, candidate_name):
                    return _build_upload_response(candidate_bytes, candidate_name, requested_mode, requested_sensitivity)

            return jsonify({'error': 'Dropped webpage did not expose a fetchable image or PDF file'}), 400

        return jsonify({'error': 'Dropped URL did not resolve to an image or PDF file'}), 400
    except _RemoteFetchError as exc:
        return jsonify({'error': str(exc)}), exc.status_code
    except Exception as exc:
        return jsonify({'error': f'Could not fetch dropped URL: {str(exc)}'}), 400

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
        immediate_started_at = time.perf_counter()
        extraction_elapsed_ms = 0.0
        post_extraction_elapsed_ms = 0.0
        response_assembly_elapsed_ms = 0.0
        initial_preview_publish_elapsed_ms = 0.0
        length_filter_elapsed_ms = 0.0
        pruning_elapsed_ms = 0.0
        final_path_serialization_elapsed_ms = 0.0
        hint_generation_elapsed_ms = 0.0
        final_preview_publish_elapsed_ms = 0.0
        progress_queue_clear_elapsed_ms = 0.0
        while not session.progress_queue.empty():
            session.progress_queue.get()

        session.results = None
        session.optimization_complete = False
        session.optimization_stopped = False
        session.optimization_generation += 1
        current_generation = session.optimization_generation
        _clear_live_preview(session, generation=current_generation)
        _set_merge_progress(
            session,
            active=False,
            phase='idle',
            processed=0,
            total=0,
            percent=0.0,
            generation=current_generation,
        )
        _set_immediate_progress(
            session,
            active=True,
            phase='preparing',
            percent=10.0,
            show_bar=False,
            path_count=0,
            message='Detecting centerline candidates...',
            detail='Thresholding the image, extracting the skeleton, and building the first overlay.',
            elapsed_sec=0.0,
            generation=current_generation,
        )

        # Quick raw path extraction
        params = session.parameters
        gray = session.image
        extraction_profile = _get_session_extraction_profile(session)
        simplification_enabled = bool(params.get('enable_pruning', DEFAULT_ENABLE_PRUNING))
        requested_min_path_length = max(1, int(params.get('min_path_length', DEFAULT_MIN_PATH_LENGTH)))
        max_min_path_length = extraction_profile.get('max_min_path_length')
        if max_min_path_length is None:
            logical_min_path_length = requested_min_path_length
        else:
            logical_min_path_length = max(1, min(requested_min_path_length, int(max_min_path_length)))
        parameter_scale = resolve_parameter_scale(session.image.shape)
        effective_min_path_length = _effective_min_path_length(params, extraction_profile, image_shape=session.image.shape)

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

        session.progress_queue.put(
            f"Preparing immediate extraction on {gray.shape[1]}x{gray.shape[0]} pixels "
            f"(threshold: {float(params['dark_threshold']):.3f})..."
        )

        # Always use fast extraction for immediate display
        print("Immediate extraction mode...")
        from skimage import morphology
        effective_preview_min_object_size = _effective_min_object_size(
            extraction_profile['preview_min_object_size'],
            params,
            extraction_profile,
            image_shape=gray.shape,
        )
        
        # Quick binary conversion and skeletonization
        session.progress_queue.put(
            f"Skeletonizing binary mask (min object size: {int(effective_preview_min_object_size)})..."
        )
        binary = gray < params['dark_threshold']
        if effective_preview_min_object_size > 0:
            binary = morphology.remove_small_objects(binary, effective_preview_min_object_size)
        skeleton = morphology.skeletonize(binary)
        session.stroke_width_estimate_px = _estimate_median_stroke_width_from_masks(binary, skeleton)
        skeleton_pixels = int(np.count_nonzero(skeleton))
        extraction_min_length = max(1, min(8, int(round(effective_min_path_length * 0.5))))
        extracting_detail = (
            f"Skeletonizing the thresholded mask and extracting fast centerline paths "
            f"from {skeleton_pixels:,} skeleton pixels..."
        )
        _set_immediate_progress(
            session,
            active=True,
            phase='extracting',
            percent=40.0,
            show_bar=False,
            path_count=0,
            message='Detecting centerline candidates...',
            detail=extracting_detail,
            elapsed_sec=time.perf_counter() - immediate_started_at,
            generation=current_generation,
        )
        session.progress_queue.put(f"Starting fast path extraction on {skeleton_pixels:,} skeleton pixels...")
        initial_paths = extract_skeleton_paths(
            gray,
            params['dark_threshold'],
            min_object_size=effective_preview_min_object_size,
            min_path_length=extraction_min_length,
        )
        extraction_elapsed_ms = max(0.0, (time.perf_counter() - immediate_started_at) * 1000.0)
        session.progress_queue.put(
            f"Fast path extraction complete: {len(initial_paths)} raw paths kept "
            f"(min length: {int(extraction_min_length)})"
        )

        preview_paths = [
            [
                [int(point[0]) + crop_offset_row, int(point[1]) + crop_offset_col]
                for point in path
            ]
            for path in initial_paths
            if path
        ]
        initial_preview_publish_started_at = time.perf_counter()
        _publish_live_preview(session, preview_paths, kind='initial', generation=current_generation)
        initial_preview_publish_elapsed_ms = max(0.0, (time.perf_counter() - initial_preview_publish_started_at) * 1000.0)
        session.progress_queue.put(
            f"Showing a sampled first-pass overlay from {len(preview_paths)} raw paths while filtering continues..."
        )
        show_bar = len(initial_paths) >= IMMEDIATE_PROGRESS_BAR_PATH_THRESHOLD
        filtering_detail = (
            f"Filtering extracted paths with an effective minimum length of {int(effective_min_path_length)} px "
            f"(slider: {int(logical_min_path_length)} px, scale: {float(parameter_scale):.2f}x)..."
        )
        _set_immediate_progress(
            session,
            active=True,
            phase='filtering',
            percent=78.0,
            show_bar=show_bar,
            path_count=len(initial_paths),
            message='Detecting centerline candidates...',
            detail=filtering_detail,
            elapsed_sec=time.perf_counter() - immediate_started_at,
            generation=current_generation,
        )
        
        if len(initial_paths) == 0:
            _set_immediate_progress(
                session,
                active=False,
                phase='error',
                percent=72.0,
                show_bar=show_bar,
                path_count=len(initial_paths),
                message='Detecting centerline candidates...',
                detail='No skeleton paths found. Try adjusting the dark threshold.',
                elapsed_sec=time.perf_counter() - immediate_started_at,
                generation=current_generation,
            )
            return jsonify({'error': 'No skeleton paths found. Try adjusting the dark threshold.'})
        
        merge_gap = _effective_merge_gap_from_stroke(
            params,
            image_shape=session.image.shape,
            median_stroke_width_px=session.stroke_width_estimate_px,
        )
        merge_angle_priority = float(params.get('merge_angle_priority', 30.0)) / 100.0
        merged_paths = initial_paths
        merge_applied = False
        merge_deferred = bool(params.get('enable_optimization', True))
        if merge_deferred:
            session.progress_queue.put(
                f"Deferring nearby path merging to background optimization "
                f"(max gap: {merge_gap} pixels, angle priority: {merge_angle_priority:.2f})..."
            )
        else:
            session.progress_queue.put("Skipping nearby path merging because optimization is disabled.")

        # Filter by minimum length
        session.progress_queue.put(filtering_detail)
        length_filter_started_at = time.perf_counter()
        valid_paths = [path for path in merged_paths if len(path) >= effective_min_path_length]
        length_filter_elapsed_ms = max(0.0, (time.perf_counter() - length_filter_started_at) * 1000.0)
        pruning_debug_enabled = simplification_enabled and (
            bool(params.get('show_pruning_debug_grid', False)) or _should_force_pruning_debug(session.original_filename)
        )
        pruning_debug_payload = None
        pruning_timing_payload = None
        pruning_started_at = time.perf_counter()
        if pruning_debug_enabled:
            valid_paths, pruning_debug_payload, pruning_timing_payload = prune_extracted_paths(
                valid_paths,
                min_output_points=2,
                return_diagnostics=True,
                max_debug_paths=None,
                return_timing=True,
            )
            pruning_summary = dict(pruning_debug_payload.get('summary') or {})
            session.progress_queue.put(
                "Pruning debug: "
                f"{int(pruning_summary.get('input_points', 0))} input points -> "
                f"{int(pruning_summary.get('output_points', 0))} output points across "
                f"{int(pruning_summary.get('changed_path_count', 0))}/{int(pruning_summary.get('path_count', 0))} paths"
            )
        elif simplification_enabled:
            valid_paths, pruning_timing_payload = prune_extracted_paths(
                valid_paths,
                min_output_points=2,
                return_timing=True,
            )
        else:
            session.progress_queue.put("Path simplification disabled: keeping length-filtered centerlines without pruning.")
        pruning_elapsed_ms = max(0.0, (time.perf_counter() - pruning_started_at) * 1000.0)
        _set_immediate_progress(
            session,
            active=True,
            phase='finalizing',
            percent=94.0,
            show_bar=show_bar,
            path_count=len(initial_paths),
            message='Detecting centerline candidates...',
            detail='Finalizing the first simplified overlay...',
            elapsed_sec=time.perf_counter() - immediate_started_at,
            generation=current_generation,
        )
        ready_detail = (
            f"Immediate overlay ready: {len(valid_paths)} paths kept after length filtering "
            f"(effective min length: {int(effective_min_path_length)}; "
            f"path simplification {'on' if simplification_enabled else 'off'})"
        )
        post_extraction_elapsed_ms = max(0.0, (time.perf_counter() - immediate_started_at) * 1000.0) - float(extraction_elapsed_ms)
        session.progress_queue.put(ready_detail)
        _set_immediate_progress(
            session,
            active=True,
            phase='ready',
            percent=100.0,
            show_bar=show_bar,
            path_count=len(initial_paths),
            message='Detecting centerline candidates...',
            detail=ready_detail,
            elapsed_sec=time.perf_counter() - immediate_started_at,
            generation=current_generation,
        )
        
        if len(valid_paths) == 0:
            _set_immediate_progress(
                session,
                active=False,
                phase='error',
                percent=100.0,
                show_bar=show_bar,
                path_count=len(initial_paths),
                message='Detecting centerline candidates...',
                detail='No valid paths after filtering. Try reducing minimum path length.',
                elapsed_sec=time.perf_counter() - immediate_started_at,
                generation=current_generation,
            )
            return jsonify({'error': 'No valid paths after filtering. Try reducing minimum path length.'})
        
        # Convert numpy coordinates to regular Python lists for JSON serialization.
        # If a crop was applied, offset coordinates back to full-image space so
        # the frontend can render paths correctly over the original image.
        response_assembly_started_at = time.perf_counter()
        final_path_serialization_started_at = time.perf_counter()
        json_serializable_paths = []
        for path in valid_paths:
            serializable_path = [
                [int(point[0]) + crop_offset_row, int(point[1]) + crop_offset_col]
                for point in path
            ]
            json_serializable_paths.append(serializable_path)
        final_path_serialization_elapsed_ms = max(0.0, (time.perf_counter() - final_path_serialization_started_at) * 1000.0)

        # Store paths in full-image coordinates for subsequent SVG generation and optimization.
        if crop_offset_row != 0 or crop_offset_col != 0:
            session.initial_paths = [
                [[pt[0] + crop_offset_row, pt[1] + crop_offset_col] for pt in path]
                for path in valid_paths
            ]
        else:
            session.initial_paths = valid_paths  # Keep original for processing
        session.partial_optimized_paths = []
        hint_generation_started_at = time.perf_counter()
        session.initial_path_hints = [build_line_span_hints(path) for path in session.initial_paths]
        hint_generation_elapsed_ms = max(0.0, (time.perf_counter() - hint_generation_started_at) * 1000.0)
        session.partial_optimized_path_hints = []
        session.pruning_debug_payload = _serialize_pruning_debug_payload(
            _offset_pruning_debug_payload(pruning_debug_payload, crop_offset_row, crop_offset_col)
        )
        final_preview_publish_started_at = time.perf_counter()
        _publish_live_preview(session, json_serializable_paths, kind='initial', generation=current_generation)
        final_preview_publish_elapsed_ms = max(0.0, (time.perf_counter() - final_preview_publish_started_at) * 1000.0)
        
        # Clear progress queue
        progress_queue_clear_started_at = time.perf_counter()
        while not session.progress_queue.empty():
            session.progress_queue.get()
        progress_queue_clear_elapsed_ms = max(0.0, (time.perf_counter() - progress_queue_clear_started_at) * 1000.0)

        processing_elapsed_sec = time.perf_counter() - immediate_started_at
        processing_elapsed_ms = processing_elapsed_sec * 1000.0
        response_assembly_elapsed_ms = max(0.0, (time.perf_counter() - response_assembly_started_at) * 1000.0)
        timing_breakdown_ms = {
            'extraction': round(float(extraction_elapsed_ms), 3),
            'initial_preview_publish': round(float(initial_preview_publish_elapsed_ms), 3),
            'length_filter': round(float(length_filter_elapsed_ms), 3),
            'pruning': round(float(pruning_elapsed_ms), 3),
            'final_path_serialization': round(float(final_path_serialization_elapsed_ms), 3),
            'hint_generation': round(float(hint_generation_elapsed_ms), 3),
            'final_preview_publish': round(float(final_preview_publish_elapsed_ms), 3),
            'progress_queue_clear': round(float(progress_queue_clear_elapsed_ms), 3),
            'post_extraction': round(float(max(0.0, post_extraction_elapsed_ms)), 3),
            'response_assembly': round(float(response_assembly_elapsed_ms), 3),
            'total_server': round(float(processing_elapsed_ms), 3),
        }
        
        # Create immediate results for magenta display
        immediate_results = {
            'initial_paths_count': len(initial_paths),
            'merged_paths_count': len(merged_paths),
            'merge_applied': merge_applied,
            'merge_deferred': merge_deferred,
            'extraction_profile': extraction_profile['profile_name'],
            'valid_paths_count': len(valid_paths),
            'paths': json_serializable_paths,  # Use JSON-serializable version
            'optimization_enabled': bool(params.get('enable_optimization', True)),
            'optimization_started': False,
            'logical_min_path_length': int(logical_min_path_length),
            'effective_min_path_length': int(effective_min_path_length),
            'parameter_scale': float(parameter_scale),
            'processing_elapsed_sec': float(processing_elapsed_sec),
            'processing_elapsed_ms': float(processing_elapsed_ms),
            'timing_breakdown_ms': timing_breakdown_ms,
            'pruning_timing_breakdown_ms': pruning_timing_payload,
            'effective_merge_gap': int(merge_gap),
            'median_stroke_width_px': float(session.stroke_width_estimate_px) if session.stroke_width_estimate_px is not None else None,
            'path_simplification_enabled': bool(simplification_enabled),
            'pruning_debug': session.pruning_debug_payload,
        }
        
        # Start optimization in background if enabled
        if params.get('enable_optimization', True):
            session.optimization_thread = threading.Thread(
                target=background_optimization,
                args=(session, current_generation)
            )
            session.optimization_thread.start()
            immediate_results['optimization_started'] = True

        _record_benchmark_stage(session, 'process_immediate', (time.perf_counter() - immediate_started_at) * 1000.0, {
            'initial_paths_count': len(initial_paths),
            'merged_paths_count': len(merged_paths),
            'valid_paths_count': len(valid_paths),
            'optimization_started': bool(immediate_results['optimization_started']),
            'crop_applied': bool(crop_offset_row != 0 or crop_offset_col != 0),
            'timing_breakdown_ms': timing_breakdown_ms,
        })
        _set_immediate_progress(
            session,
            active=False,
            phase='complete',
            percent=100.0,
            show_bar=show_bar,
            path_count=len(initial_paths),
            message='Detecting centerline candidates...',
            detail=ready_detail,
            elapsed_sec=processing_elapsed_sec,
            generation=current_generation,
        )
        
        return jsonify({
            'success': True,
            'immediate_display': True,
            **immediate_results
        })
        
    except Exception as e:
        _set_immediate_progress(
            session,
            active=False,
            phase='error',
            percent=0.0,
            show_bar=False,
            path_count=0,
            message='Detecting centerline candidates...',
            detail=f'Processing error: {str(e)}',
            elapsed_sec=(time.perf_counter() - immediate_started_at) if 'immediate_started_at' in locals() else 0.0,
            generation=current_generation if 'current_generation' in locals() else None,
        )
        return jsonify({'error': f'Processing error: {str(e)}'})

def background_optimization(session, generation):
    """Run optimization in background thread."""
    try:
        start_time = time.perf_counter()

        if generation != session.optimization_generation:
            return

        params = session.parameters
        valid_paths = list(session.initial_paths or [])
        valid_path_hints = list(session.initial_path_hints or [])
        extraction_profile = _get_session_extraction_profile(session)
        effective_min_path_length = _effective_min_path_length(params, extraction_profile, image_shape=session.image.shape)
        merge_gap = _effective_merge_gap_from_stroke(
            params,
            image_shape=session.image.shape,
            median_stroke_width_px=session.stroke_width_estimate_px,
        )
        merge_angle_priority = float(params.get('merge_angle_priority', 30.0)) / 100.0
        merge_metrics = {}
        merge_progress_state = {
            'last_seed_paths_processed': -1,
            'last_emit_at': 0.0,
        }

        def _report_merge_progress(progress_metrics):
            now = time.perf_counter()
            processed = int(progress_metrics.get('seed_paths_processed', 0))
            total = max(1, int(progress_metrics.get('seed_paths_total', 0)))
            merges = int(progress_metrics.get('merged_pairs', 0))
            candidates = int(progress_metrics.get('candidate_paths_scanned', 0))
            merge_scope = str(progress_metrics.get('merge_scope', 'cheap_only') or 'cheap_only')
            merge_phase = 'long_merge' if merge_scope == 'long_only' else 'cheap_merge'
            _set_merge_progress(
                session,
                active=True,
                phase=merge_phase,
                processed=processed,
                total=total,
                percent=(processed / total) * 100.0,
                generation=generation,
            )
            if (
                processed == merge_progress_state['last_seed_paths_processed']
                and (now - merge_progress_state['last_emit_at']) < 0.9
            ):
                return
            if (
                processed < total
                and processed > 0
                and (processed - merge_progress_state['last_seed_paths_processed']) < 25
                and (now - merge_progress_state['last_emit_at']) < 0.9
            ):
                return

            merge_progress_state['last_seed_paths_processed'] = processed
            merge_progress_state['last_emit_at'] = now
            session.progress_queue.put(
                f"Merge scan: {processed}/{total} seed paths, {candidates:,} nearby candidates checked, "
                f"{merges} merges accepted ({float(progress_metrics.get('elapsed_sec', 0.0)):.1f}s)"
            )

        def _report_merge_preview(progress_metrics, preview_paths):
            _publish_live_preview(
                session,
                preview_paths,
                kind='merge',
                generation=generation,
            )

        def _summarize_merge_pass(label, pass_metrics):
            session.progress_queue.put(
                f"{label} summary: {int(pass_metrics.get('candidate_paths_scanned', 0)):,} nearby candidates, "
                f"{int(pass_metrics.get('cheap_candidates_ranked', 0)):,} ranked bridges, "
                f"{int(pass_metrics.get('long_fragment_candidates_skipped', 0)):,} long-fragment skips, "
                f"{int(pass_metrics.get('endpoint_distance_checks', 0)):,} endpoint checks, "
                f"{int(pass_metrics.get('safety_checks', 0)):,} geometry checks, "
                f"{int(pass_metrics.get('merged_pairs', 0))} accepted, "
                f"{int(pass_metrics.get('distance_rejections', 0)):,} distance rejects, "
                f"{int(pass_metrics.get('angle_rejections', 0)):,} angle rejects, "
                f"{int(pass_metrics.get('safety_rejections', 0)):,} geometry rejects."
            )

        session.progress_queue.put(
            f"Cheap nearby-path merge pass (max gap: {merge_gap} pixels, angle priority: {merge_angle_priority:.2f})..."
        )
        _set_merge_progress(
            session,
            active=True,
            phase='cheap_merge',
            processed=0,
            total=len(valid_paths),
            percent=0.0,
            generation=generation,
        )
        cheap_merge_started_at = time.perf_counter()
        merged_paths = merge_nearby_paths(
            valid_paths,
            max_gap=merge_gap,
            angle_priority=merge_angle_priority,
            verbose=False,
            should_continue=lambda: (not session.optimization_stopped) and generation == session.optimization_generation,
            metrics=merge_metrics,
            progress_callback=_report_merge_progress,
            preview_callback=_report_merge_preview,
            allow_long_path_merges=False,
        )
        merge_elapsed_sec = time.perf_counter() - cheap_merge_started_at
        _summarize_merge_pass("Cheap merge pass", merge_metrics)

        if bool(params.get('enable_long_path_merging', False)) and len(merged_paths) > 1:
            session.progress_queue.put(
                f"Optional long-fragment merge pass enabled (threshold: {int(merge_metrics.get('long_path_threshold', 0))} points)..."
            )
            _set_merge_progress(
                session,
                active=True,
                phase='long_merge',
                processed=0,
                total=len(merged_paths),
                percent=0.0,
                generation=generation,
            )
            long_merge_metrics = {}
            long_merge_started_at = time.perf_counter()
            merged_paths = merge_nearby_paths(
                merged_paths,
                max_gap=merge_gap,
                angle_priority=merge_angle_priority,
                verbose=False,
                should_continue=lambda: (not session.optimization_stopped) and generation == session.optimization_generation,
                metrics=long_merge_metrics,
                progress_callback=_report_merge_progress,
                preview_callback=_report_merge_preview,
                allow_long_path_merges=True,
                only_long_path_merges=True,
                long_path_threshold=int(merge_metrics.get('long_path_threshold', 0) or max(256, int(round(float(merge_gap) * 4.0)))),
            )
            merge_elapsed_sec += (time.perf_counter() - long_merge_started_at)
            _summarize_merge_pass("Long-fragment pass", long_merge_metrics)

        merged_valid_paths = [path for path in merged_paths if len(path) >= effective_min_path_length]
        if merged_valid_paths:
            valid_paths = merged_valid_paths
            valid_path_hints = [build_line_span_hints(path) for path in valid_paths]
            session.progress_queue.put(
                f"Path merging complete: {len(session.initial_paths or [])} raw paths -> {len(merged_paths)} merged paths in {merge_elapsed_sec:.2f}s; "
                f"{len(valid_paths)} paths kept (min length: {int(effective_min_path_length)})"
            )
        else:
            valid_paths = [path for path in valid_paths if len(path) >= effective_min_path_length]
            valid_path_hints = [build_line_span_hints(path) for path in valid_paths]
            session.progress_queue.put(
                "Merged result was empty after filtering; falling back to unmerged paths for optimization."
            )

        with session.lock:
            if generation != session.optimization_generation:
                return
            session.initial_paths = valid_paths
            session.initial_path_hints = valid_path_hints
        _publish_live_preview(session, valid_paths, kind='merge', generation=generation)
        _set_merge_progress(
            session,
            active=False,
            phase='merge_complete',
            processed=len(valid_paths),
            total=len(valid_paths),
            percent=100.0,
            generation=generation,
        )

        original_total_segments = sum(max(len(path) - 1, 0) for path in valid_paths)
        optimized_total_segments = 0
        benchmark_enabled = bool(session.benchmark_enabled)
        optimizer_phase_totals_ms = {}
        optimizer_counts = {}
        initial_scoring_ms = 0.0
        path_optimization_ms = 0.0
        optimized_scores = []
        processed_sorted_indices = []
        
        session.progress_queue.put("Starting optimization process...")
        _set_merge_progress(
            session,
            active=False,
            phase='optimizing',
            processed=len(valid_paths),
            total=len(valid_paths),
            percent=100.0,
            generation=generation,
        )
        
        # Initialize circle evaluation system
        circle_init_started_at = time.perf_counter()
        circle_system = CircleEvaluationSystem(
            session.image, 
            params['dark_threshold'],
            3,    # max_circle_radius
            0.2,  # min_circle_radius
            2.0   # circle_intersection_bonus
        )
        circle_system_init_ms = (time.perf_counter() - circle_init_started_at) * 1000.0
        
        session.progress_queue.put("Circle evaluation system initialized...")

        # Avoid a full upfront scoring pass so the first path starts sooner.
        # Longer paths tend to matter most visually, so prioritize them first.
        prioritization_started_at = time.perf_counter()
        sorted_indices = sorted(
            range(len(valid_paths)),
            key=lambda idx: len(valid_paths[idx]),
            reverse=True,
        )
        path_prioritization_ms = (time.perf_counter() - prioritization_started_at) * 1000.0
        max_path_length = max((len(p) for p in valid_paths), default=1)
        session.progress_queue.put("Prioritizing longer paths first...")
        session.progress_queue.put(f"Optimizing {len(valid_paths)} paths...")
        
        # Optimize paths one by one with progress updates
        for i, idx in enumerate(sorted_indices):
            if session.optimization_stopped or generation != session.optimization_generation:
                session.progress_queue.put("Optimization stopped by user")
                break
                
            path = valid_paths[idx]
            score_started_at = time.perf_counter()
            original_score = circle_system.evaluate_path(path)
            initial_scoring_ms += (time.perf_counter() - score_started_at) * 1000.0
            
            session.progress_queue.put(f"Optimizing path {i+1}/{len(valid_paths)} ({len(path)} points, score: {original_score:.3f})")
            
            # Apply optimization using current UI parameters, including guarded simplification.
            balanced_params = {
                'rdp_tolerance': params.get('rdp_tolerance', 5.0),
                'smoothing_factor': params.get('smoothing_factor', 0.006),
                'simplification_strength': params.get('simplification_strength', 100.0),
                'arc_fit_strength': params.get('arc_fit_strength', 72.0),
                'line_fit_strength': params.get('line_fit_strength', 0.0),
                'short_path_protection': params.get('short_path_protection', 0.0),
                'mean_closeness_px': params.get('mean_closeness_px', 1.8),
                'peak_closeness_px': params.get('peak_closeness_px', 4.5),
                'score_preservation': params.get('score_preservation', 99.0),
                'path_length': len(path),
                'max_path_length': max_path_length,
                'min_path_length': effective_min_path_length
            }
            
            optimization_started_at = time.perf_counter()
            if benchmark_enabled:
                optimized_path, optimized_score, optimizer_diagnostics = optimize_path_with_custom_params(
                    path,
                    circle_system,
                    balanced_params,
                    initial_score=original_score,
                    return_diagnostics=True,
                )
                _merge_benchmark_numeric_dict(optimizer_phase_totals_ms, optimizer_diagnostics.get('phase_ms', {}))
                _merge_benchmark_numeric_dict(optimizer_counts, optimizer_diagnostics.get('counts', {}))
            else:
                optimized_path, optimized_score = optimize_path_with_custom_params(
                    path, circle_system, balanced_params, initial_score=original_score
                )
            path_optimization_ms += (time.perf_counter() - optimization_started_at) * 1000.0

            if session.optimization_stopped or generation != session.optimization_generation:
                session.progress_queue.put("Optimization stopped by user")
                break
            
            # Add to partial results - always add, even if not dramatically different
            with session.lock:
                if generation != session.optimization_generation:
                    break
                session.partial_optimized_paths.append(optimized_path)
                session.partial_optimized_path_hints.append(build_line_span_hints(optimized_path))
                session.live_preview_kind = 'optimization'
                session.live_preview_frame_id += 1

            optimized_scores.append(float(optimized_score))
            processed_sorted_indices.append(int(idx))

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

        finalized_optimized_paths = list(session.partial_optimized_paths)
        finalized_optimized_path_hints = list(session.partial_optimized_path_hints)
        finalized_pre_optimization_paths = [
            valid_paths[idx]
            for idx in processed_sorted_indices[:len(finalized_optimized_paths)]
            if 0 <= int(idx) < len(valid_paths)
        ]
        finalized_pre_optimization_path_hints = [
            valid_path_hints[idx]
            for idx in processed_sorted_indices[:len(finalized_optimized_paths)]
            if 0 <= int(idx) < len(valid_path_hints)
        ]
        session.results = {
            'initial_paths_count': len(session.initial_paths or []),
            'merged_paths_count': len(valid_paths),
            'valid_paths_count': len(valid_paths),
            'stats': {},
            'best_score': optimized_scores[0] if optimized_scores else 0.0,
            'optimized_paths': finalized_optimized_paths,
            'optimized_scores': list(optimized_scores[:len(finalized_optimized_paths)]),
            'pre_optimization_paths': finalized_pre_optimization_paths,
            'optimized_path_hints': finalized_optimized_path_hints,
            'pre_optimization_path_hints': finalized_pre_optimization_path_hints,
            'circle_system': circle_system,
        }

        session.optimization_complete = True
        _mark_live_preview_frame(session, kind='final', generation=generation)
        _set_merge_progress(
            session,
            active=False,
            phase='complete',
            processed=len(valid_paths),
            total=len(valid_paths),
            percent=100.0,
            generation=generation,
        )
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
        _update_benchmark_bucket(session, 'optimization', {
            'status': 'complete',
            'elapsed_ms': round(elapsed_seconds * 1000.0, 3),
            'elapsed_sec': round(elapsed_seconds, 4),
            'input_path_count': int(len(valid_paths)),
            'optimized_path_count': int(len(session.partial_optimized_paths)),
            'original_total_segments': int(original_total_segments),
            'optimized_total_segments': int(optimized_total_segments),
            'segment_reduction_pct': round(float(segment_reduction), 3),
            'circle_system_init_ms': round(circle_system_init_ms, 3),
            'path_prioritization_ms': round(path_prioritization_ms, 3),
            'initial_scoring_ms': round(initial_scoring_ms, 3),
            'path_optimization_ms': round(path_optimization_ms, 3),
            'avg_path_optimization_ms': round(path_optimization_ms / max(len(valid_paths), 1), 3),
            'optimizer_phase_ms': _round_benchmark_numeric_dict(optimizer_phase_totals_ms),
            'optimizer_counts': {
                str(key): int(round(float(value)))
                for key, value in optimizer_counts.items()
                if isinstance(value, (int, float))
            },
        })
        
    except Exception as e:
        _update_benchmark_bucket(session, 'optimization', {
            'status': 'error',
            'error': str(e),
        })
        _set_merge_progress(
            session,
            active=False,
            phase='error',
            processed=0,
            total=0,
            percent=0.0,
        )
        session.progress_queue.put(f"Optimization error: {str(e)}")


@app.route('/benchmark_metrics/<session_id>')
def benchmark_metrics(session_id):
    """Return session benchmark metrics when filename-triggered metrics capture is enabled."""
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session'}), 404

    session = sessions[session_id]
    payload = _benchmark_metrics_response(session)
    if payload is None:
        return jsonify({'error': 'Benchmark metrics not enabled for this session'}), 404
    return jsonify(payload)

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

    include_preview_paths = _get_request_bool_arg('include_preview_paths', default=True)

    with session.lock:
        optimized_count = len(session.partial_optimized_paths)
        live_preview_kind = session.live_preview_kind
        live_preview_frame_id = int(session.live_preview_frame_id)
        live_preview_count = optimized_count if optimized_count > 0 else len(session.live_preview_paths)
        live_preview_paths = []
        if include_preview_paths:
            live_preview_paths = [
                _serialize_point_list(path)
                for path in list(session.live_preview_paths)
            ]
        merge_progress = dict(session.merge_progress)
        immediate_progress = dict(session.immediate_progress)
    
    return jsonify({
        'messages': messages,
        'optimization_enabled': bool(session.parameters.get('enable_optimization', True)),
        'optimization_complete': session.optimization_complete,
        'optimized_count': optimized_count,
        'total_paths': len(session.initial_paths) if session.initial_paths else 0,
        'live_preview_kind': live_preview_kind,
        'live_preview_frame_id': live_preview_frame_id,
        'live_preview_count': live_preview_count,
        'live_preview_paths': live_preview_paths,
        'merge_progress': merge_progress,
        'immediate_progress': immediate_progress,
        'optimization_running': bool(
            session.optimization_thread and session.optimization_thread.is_alive() and not session.optimization_complete
        ),
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


@app.route('/stop_auto_detect', methods=['POST'])
def stop_auto_detect():
    """Cancel threshold auto-detect and keep the previous threshold unchanged."""
    data = request.json or {}
    session_id = data.get('session_id')

    if session_id not in sessions:
        return jsonify({'error': 'Invalid session'})

    session = sessions[session_id]

    with session.lock:
        session.auto_detect_generation += 1
        session.auto_detect_active = False
        started_at = float(session.auto_detect_progress.get('started_at', 0.0) or 0.0)
        elapsed_sec = float(time.perf_counter() - started_at) if started_at > 0.0 else float(session.auto_detect_progress.get('elapsed_sec', 0.0))
        session.auto_detect_progress.update({
            'running': False,
            'finished': True,
            'success': False,
            'cancelled': True,
            'elapsed_sec': elapsed_sec,
            'detected_threshold': float(session.parameters.get('dark_threshold', 0.20)),
            'best_threshold': float(session.parameters.get('dark_threshold', 0.20)),
            'message': 'Auto-detect cancelled. Previous threshold kept.',
            'updated_parameters': dict(session.parameters),
        })

    return jsonify({'success': True, 'message': 'Auto-detect stopping...'})


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
    session.parameters['min_path_length'] = DEFAULT_MIN_PATH_LENGTH
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
        process_started_at = time.perf_counter()
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

            _record_benchmark_stage(session, 'process', (time.perf_counter() - process_started_at) * 1000.0, {
                'paths_count': len(results['optimized_paths']),
                'original_points': int(original_points),
                'optimized_points': int(optimized_points),
                'point_reduction_percentage': round(float(reduction_percentage), 3),
            })
            
            return jsonify(response)
        else:
            return jsonify(results or {'error': 'Unknown processing error'})
            
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'})

@app.route('/test_svg/<session_id>')
def test_svg(session_id):
    """Test SVG generation route."""
    _log_debug(f"Test SVG route called for session: {session_id}")
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session'})
    
    session = sessions[session_id]
    _log_debug(f"Session found. Has initial_paths: {bool(session.initial_paths)}")
    if session.initial_paths:
        _log_debug(f"Number of initial paths: {len(session.initial_paths)}")
    
    return jsonify({'status': 'test successful', 'has_paths': bool(session.initial_paths)})

@app.route('/generate_svg', methods=['POST'])
def generate_svg():
    """Generate and return SVG visualization with support for progressive display."""
    data = request.json
    session_id = data.get('session_id')
    display_mode = data.get('mode', 'final')  # 'immediate', 'progressive', or 'final'
    persist_parameters = bool(data.get('persist_parameters', True))
    
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session'})

    session = sessions[session_id]
    
    # Update session parameters if provided (especially important for include_image setting)
    if persist_parameters and 'parameters' in data:
        session.parameters.update(data['parameters'])

    render_parameters = dict(session.parameters)
    if 'parameters' in data and isinstance(data['parameters'], dict):
        render_parameters.update(data['parameters'])
    svg_render_settings = _resolve_svg_render_settings(render_parameters)
    show_pre = bool(svg_render_settings['show_pre'])
    optimization_enabled = bool(render_parameters.get('enable_optimization', True))
    
    detected_paths_count = len(session.initial_paths) if session.initial_paths else 0

    # Check what data we have available
    if display_mode == 'immediate' and session.initial_paths:
        # Show only magenta paths immediately
        _log_debug(f"Generating immediate SVG with {len(session.initial_paths)} paths")
        paths_to_show = session.initial_paths
        if optimization_enabled:
            optimized_paths = []
            optimized_path_hints = []
            pre_optimization_paths = paths_to_show if show_pre else []
        else:
            optimized_paths = paths_to_show
            optimized_path_hints = list(session.initial_path_hints) or [build_line_span_hints(path) for path in optimized_paths]
            pre_optimization_paths = []
        circle_system = None

    elif display_mode == 'progressive':
        # Show magenta + current blue progress
        with session.lock:
            # Make a thread-safe copy of the paths to avoid issues during iteration
            optimized_paths = list(session.partial_optimized_paths)
            optimized_path_hints = list(session.partial_optimized_path_hints)
            live_preview_paths = list(session.live_preview_paths)

        _log_debug(f"Generating progressive SVG with {len(optimized_paths)} optimized paths")
        paths_to_show = session.initial_paths
        if not optimization_enabled:
            optimized_paths = list(session.initial_paths or [])
            optimized_path_hints = list(session.initial_path_hints) or [build_line_span_hints(path) for path in optimized_paths]
        elif live_preview_paths:
            stitched_progressive_paths = list(live_preview_paths)
            optimized_prefix_count = min(len(optimized_paths), len(stitched_progressive_paths))
            if optimized_prefix_count > 0:
                stitched_progressive_paths[:optimized_prefix_count] = optimized_paths[:optimized_prefix_count]
            if len(optimized_paths) > len(stitched_progressive_paths):
                stitched_progressive_paths.extend(optimized_paths[optimized_prefix_count:])
            optimized_paths = stitched_progressive_paths
            optimized_path_hints = [build_line_span_hints(path) for path in optimized_paths]
        circle_system = None  # Could add this later
        pre_optimization_paths = paths_to_show if show_pre else []
        
    elif session.results:
        # Show final results
        _log_debug("Generating final SVG from session.results")
        results = session.results
        paths_to_show = results.get('pre_optimization_paths', [])
        optimized_paths = results.get('optimized_paths', [])
        optimized_path_hints = results.get('optimized_path_hints') or [build_line_span_hints(path) for path in optimized_paths]
        circle_system = results.get('circle_system')
        pre_optimization_paths = paths_to_show if show_pre else []
        
    else:
        return jsonify({'error': 'No results available yet'})

    preview_paths_suppressed = detected_paths_count > SVG_VIEW_PATH_RENDER_LIMIT
    if preview_paths_suppressed and (optimized_paths or pre_optimization_paths):
        _log_debug(
            f"Sampling SVG preview paths because detected path count "
            f"{detected_paths_count} exceeds limit {SVG_VIEW_PATH_RENDER_LIMIT}"
        )
        optimized_paths, optimized_path_hints = _sample_paths_and_hints_for_preview(
            optimized_paths,
            optimized_path_hints,
            SVG_VIEW_PATH_RENDER_LIMIT,
        )
        pre_optimization_paths, _ = _sample_paths_and_hints_for_preview(
            pre_optimization_paths,
            None,
            SVG_VIEW_PATH_RENDER_LIMIT,
        )

    try:
        svg_started_at = time.perf_counter()
        background_image = session.display_image if session.display_image is not None else session.image
        render_optimized_paths = [] if display_mode == 'immediate' and optimization_enabled else optimized_paths
        render_optimized_path_hints = [] if display_mode == 'immediate' and optimization_enabled else optimized_path_hints
        svg_content, suppressed, cache_hit, render_metadata = _render_svg_content(
            session,
            f"view:{display_mode}",
            background_image,
            None if display_mode == 'immediate' else circle_system,
            render_optimized_paths,
            pre_optimization_paths,
            render_optimized_path_hints,
            svg_render_settings['source_smoothing'],
            svg_render_settings['curve_fit_tolerance'],
            svg_render_settings['endpoint_tangent_strictness'],
            svg_render_settings['force_orthogonal_as_lines'],
            svg_render_settings['enable_curve_fitting'],
            svg_render_settings['enable_post_fit_export'],
            svg_render_settings['include_image'],
            show_pre,
            suppress_paths_in_view=preview_paths_suppressed,
            fit_optimized_paths=svg_render_settings['fit_optimized_paths'],
            combine_optimized_paths=svg_render_settings['combine_optimized_paths'],
            combine_pre_optimization_paths=svg_render_settings['combine_pre_optimization_paths'],
            coordinate_precision=svg_render_settings['coordinate_precision'],
        )
        svg_elapsed_ms = (time.perf_counter() - svg_started_at) * 1000.0
        optimization_summary = _build_svg_optimization_summary(
            render_metadata,
            enable_post_fit_export=svg_render_settings['enable_post_fit_export'],
            fit_optimized_paths=svg_render_settings['fit_optimized_paths'],
            enable_curve_fitting=svg_render_settings['enable_curve_fitting'],
            force_orthogonal_as_lines=svg_render_settings['force_orthogonal_as_lines'],
            elapsed_ms=svg_elapsed_ms,
            svg_content=svg_content,
            cache_hit=cache_hit,
        )
        _log_debug(f"SVG content ready, size: {len(svg_content)} characters (cache_hit={cache_hit})")
        _record_benchmark_svg(
            session,
            display_mode,
            svg_elapsed_ms,
            svg_content,
            suppressed,
            cache_hit=cache_hit,
        )
        return jsonify({
            'svg': svg_content,
            'preview_paths_suppressed': suppressed,
            'detected_paths_count': int(detected_paths_count),
            'rendered_path_count': int(len(render_optimized_paths) + len(pre_optimization_paths)),
            'optimization_summary': optimization_summary,
        })
            
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
        download_started_at = time.perf_counter()
        render_parameters = dict(session.parameters)
        include_image_override = _get_request_bool_arg('include_image', default=None)
        if include_image_override is not None:
            render_parameters['include_image'] = bool(include_image_override)

        svg_render_settings = _resolve_svg_render_settings(render_parameters)
        optimization_enabled = bool(session.parameters.get('enable_optimization', True))

        # Create filename based on original upload
        original_filename = session.original_filename or 'centerline_extraction'
        # Remove extension and add _centerline.svg
        name_without_ext = os.path.splitext(original_filename)[0]
        download_filename = f"{name_without_ext}_centerline.svg"
        
        background_image = session.display_image if session.display_image is not None else session.image
        
        # Determine what to show based on available data
        show_pre = bool(svg_render_settings['show_pre'])
        prefer_final_results = bool(
            has_traditional_results and (
                session.optimization_complete
                or not has_progressive_data
                or (
                    optimization_enabled
                    and not session.partial_optimized_paths
                    and len(session.results.get('optimized_paths', []) or []) > 0
                )
            )
        )

        if has_progressive_data and not prefer_final_results:
            # Use progressive data
            _log_debug(f"Generating download SVG with progressive data: {len(session.initial_paths)} initial paths, {len(session.partial_optimized_paths)} optimized paths")
            if optimization_enabled:
                optimized_paths = list(session.partial_optimized_paths)
                optimized_path_hints = list(session.partial_optimized_path_hints)
                pre_optimization_paths = session.initial_paths if show_pre else []
            else:
                optimized_paths = list(session.initial_paths or [])
                optimized_path_hints = list(session.initial_path_hints) or [build_line_span_hints(path) for path in optimized_paths]
                pre_optimization_paths = []
            circle_system = None
            render_variant = 'download:progressive'
            export_source = 'progressive'
        else:
            # Use traditional results
            results = session.results
            optimization_enabled = session.parameters.get('enable_optimization', True)
            optimized_paths = results['optimized_paths']
            optimized_path_hints = results.get('optimized_path_hints') or [build_line_span_hints(path) for path in optimized_paths]
            pre_optimization_paths = results['pre_optimization_paths'] if optimization_enabled else []
            circle_system = results['circle_system']
            render_variant = 'download:final'
            export_source = 'final'

        if not optimization_enabled and not svg_render_settings['enable_post_fit_export']:
            pre_optimization_paths = list(session.initial_paths or optimized_paths or [])
            optimized_paths = []
            optimized_path_hints = []
            show_pre = True

        _log_debug(
            "Download SVG export source="
            f"{export_source}, optimization_enabled={bool(optimization_enabled)}, "
            f"optimized_paths={len(optimized_paths)}, pre_paths={len(pre_optimization_paths)}, "
            f"include_image={bool(svg_render_settings['include_image'])}"
        )

        svg_content, _, cache_hit, _ = _render_svg_content(
            session,
            render_variant,
            background_image,
            circle_system,
            optimized_paths,
            pre_optimization_paths,
            optimized_path_hints,
            svg_render_settings['source_smoothing'],
            svg_render_settings['curve_fit_tolerance'],
            svg_render_settings['endpoint_tangent_strictness'],
            svg_render_settings['force_orthogonal_as_lines'],
            svg_render_settings['enable_curve_fitting'],
            svg_render_settings['enable_post_fit_export'],
            svg_render_settings['include_image'],
            show_pre,
            suppress_paths_in_view=False,
            fit_optimized_paths=svg_render_settings['fit_optimized_paths'],
            combine_optimized_paths=svg_render_settings['combine_optimized_paths'],
            combine_pre_optimization_paths=svg_render_settings['combine_pre_optimization_paths'],
            coordinate_precision=svg_render_settings['coordinate_precision'],
        )
        svg_bytes = svg_content.encode('utf-8')

        _record_benchmark_download(
            session,
            (time.perf_counter() - download_started_at) * 1000.0,
            len(svg_bytes),
            cache_hit=cache_hit,
        )

        return send_file(
            BytesIO(svg_bytes),
            mimetype='image/svg+xml',
            as_attachment=True,
            download_name=download_filename,
        )
        
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
