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

# Import our existing centerline extraction functions
from centerline_engine import (
    CircleEvaluationSystem, extract_skeleton_paths, merge_nearby_paths,
    optimize_path_with_circles, create_svg_output, remove_overlapping_paths,
    optimize_path_with_custom_params
)
from centerline_core import (
    AUTO_TUNE_CONFIDENCE_TARGET,
    AUTO_TUNE_RANDOM_TILE_COUNT,
    AUTO_TUNE_RANDOM_TILE_DIM,
    AUTO_TUNE_TIME_BUDGET_SEC,
    _build_random_tile_mosaic,
    _extract_tile_preview_paths,
    _offset_sample_metadata_to_source,
    _resolve_crop_bounds,
    auto_detect_dark_threshold,
    auto_detect_min_path_length,
    auto_tune_extraction_parameters,
    load_and_process_image,
)
TEST_UI_ENABLED = False
TEST_UI_IMPORT_ERROR = None

try:
    from test_ui_backend import (
        ALL_FIXTURE_IDS,
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
    TEST_UI_IMPORT_ERROR = str(exc)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size


def _test_ui_unavailable_response():
    detail = TEST_UI_IMPORT_ERROR or "test_ui_backend unavailable"
    return jsonify({
        'error': 'Test UI is unavailable in this deployment environment',
        'detail': detail,
    }), 503

# Global storage for session data
sessions = {}

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
            'cubic_fit_tolerance': 1.0,  # SVG cubic fitting tolerance in px (lower = tighter, more segments)
            'endpoint_tangent_strictness': 85.0,  # Strength of start/end handle alignment to extracted path direction (not fixture/golden data)
            'force_orthogonal_as_lines': True,  # Force axis-aligned/corner paths to use line segments only
            'min_path_length': 3,      # Increase to 8-15 for longer segments
            'enable_optimization': True,   # Enable path optimization and circle evaluation
            'show_pre_optimization': False,  # Show unoptimized paths in SVG
            'include_image': False,    # Include original image in SVG background
            'normalization_mode': 'auto',  # auto|on|off preprocessing normalization
            'normalization_sensitivity': 'medium',  # low|medium|high
        }


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
        # Import the fast function from the shared engine module.
        from centerline_engine import create_fast_paths
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


@app.route('/test-ui')
def test_ui():
    """Dedicated UI for pytest run management and fixture-layer inspection."""
    if not TEST_UI_ENABLED:
        return render_template('index.html')
    return render_template('test_management.html', fixture_ids=ALL_FIXTURE_IDS)


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
    if fixture_ids is not None and not isinstance(fixture_ids, list):
        return jsonify({'error': 'fixture_ids must be a list of fixture IDs'}), 400

    try:
        started = create_run(update_goldens=update_goldens, fixture_ids=fixture_ids)
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
        from centerline_engine import create_fast_paths
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

    curve_fit_tolerance = max(
        0.35,
        min(8.0, float(session.parameters.get('cubic_fit_tolerance', 1.0))),
    )
    endpoint_tangent_strictness = max(
        0.0,
        min(100.0, float(session.parameters.get('endpoint_tangent_strictness', 85.0))),
    )
    force_orthogonal_as_lines = bool(session.parameters.get('force_orthogonal_as_lines', True))

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
        import centerline_engine as wo
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
                pre_optimization_paths,  # Show magenta paths
                curve_fit_tolerance=curve_fit_tolerance,
                endpoint_tangent_strictness=endpoint_tangent_strictness,
                force_orthogonal_as_lines=force_orthogonal_as_lines,
            )
        else:
            # Progressive or final - show both magenta and blue
            print(f"Creating {display_mode} SVG with {len(optimized_paths)} blue paths and {len(pre_optimization_paths)} magenta paths")
            create_svg_output(
                background_image,
                circle_system,
                optimized_paths,  # Blue paths
                [1.0] * len(optimized_paths),  # Dummy scores
                pre_optimization_paths,  # Magenta paths
                curve_fit_tolerance=curve_fit_tolerance,
                endpoint_tangent_strictness=endpoint_tangent_strictness,
                force_orthogonal_as_lines=force_orthogonal_as_lines,
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
        curve_fit_tolerance = max(
            0.35,
            min(8.0, float(session.parameters.get('cubic_fit_tolerance', 1.0))),
        )
        endpoint_tangent_strictness = max(
            0.0,
            min(100.0, float(session.parameters.get('endpoint_tangent_strictness', 85.0))),
        )
        force_orthogonal_as_lines = bool(session.parameters.get('force_orthogonal_as_lines', True))

        # Create filename based on original upload
        original_filename = session.original_filename or 'centerline_extraction'
        # Remove extension and add _centerline.svg
        name_without_ext = os.path.splitext(original_filename)[0]
        download_filename = f"{name_without_ext}_centerline.svg"
        
        # Create temporary SVG file
        temp_svg = os.path.join(tempfile.gettempdir(), f"centerline_download_{session_id}.svg")
        
        # Generate SVG with user settings
        import centerline_engine as wo
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
                session.initial_paths if show_pre else [],  # Magenta initial paths
                curve_fit_tolerance=curve_fit_tolerance,
                endpoint_tangent_strictness=endpoint_tangent_strictness,
                force_orthogonal_as_lines=force_orthogonal_as_lines,
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
                    results['pre_optimization_paths'],
                    curve_fit_tolerance=curve_fit_tolerance,
                    endpoint_tangent_strictness=endpoint_tangent_strictness,
                    force_orthogonal_as_lines=force_orthogonal_as_lines,
                )
            else:
                create_svg_output(
                    background_image,
                    None,
                    results['optimized_paths'],
                    results['optimized_scores'],
                    results['pre_optimization_paths'],
                    curve_fit_tolerance=curve_fit_tolerance,
                    endpoint_tangent_strictness=endpoint_tangent_strictness,
                    force_orthogonal_as_lines=force_orthogonal_as_lines,
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
