#!/usr/bin/env python3
"""
Web-based Centerline Extraction Tool
====================================

A Flask web application for interactive centerline extraction with real-time parameter adjustment.
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import tempfile
import uuid

# Import our existing centerline extraction functions
from worksok3_optimized import (
    CircleEvaluationSystem, extract_skeleton_paths, merge_nearby_paths,
    optimize_path_with_circles, create_svg_output
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global storage for session data
sessions = {}

class CenterlineSession:
    def __init__(self, session_id):
        self.session_id = session_id
        self.image = None
        self.image_path = None
        self.results = None
        self.parameters = {
            'dark_threshold': 0.20,
            'rdp_tolerance': 2.0,      # Increase to 5.0-10.0 for fewer points
            'smoothing_factor': 0.005,  # Increase to 0.02-0.05 for fewer points
            'min_path_length': 3       # Increase to 8-15 for longer segments
        }

def load_and_process_image(image_path):
    """Load and convert image to grayscale."""
    from skimage import io, color
    
    img = io.imread(image_path)
    if len(img.shape) == 3:
        gray = color.rgb2gray(img)
    else:
        gray = img.astype(float) / 255.0 if img.dtype != float else img
    
    return gray

def process_centerlines(session):
    """Process centerlines with current parameters."""
    if session.image is None:
        return None
    
    params = session.parameters
    gray = session.image
    
    # Extract initial skeleton paths
    initial_paths = extract_skeleton_paths(gray, params['dark_threshold'], min_object_size=5)
    
    if len(initial_paths) == 0:
        return {'error': 'No skeleton paths found. Try adjusting the dark threshold.'}
    
    # Merge nearby paths
    merged_paths = merge_nearby_paths(initial_paths, max_gap=25)
    
    # Filter paths by minimum length
    valid_paths = [path for path in merged_paths if len(path) >= params['min_path_length']]
    
    if len(valid_paths) == 0:
        return {'error': 'No valid paths after filtering. Try reducing minimum path length.'}
    
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

def optimize_path_with_custom_params(path, circle_system, params):
    """Optimize path using custom parameters from the frontend."""
    from worksok3_optimized import rdp_simplify, smooth_path_spline, fit_curve_to_path
    
    current_path = path
    best_score = circle_system.evaluate_path(current_path)
    
    print(f"    Optimizing path: {len(path)} points, initial score: {best_score:.2f}")
    
    # Apply RDP simplification with user tolerance
    rdp_tolerance = params.get('rdp_tolerance', 2.0)
    try:
        simplified_path = rdp_simplify(current_path, rdp_tolerance)
        if len(simplified_path) >= params.get('min_path_length', 3):
            rdp_score = circle_system.evaluate_path(simplified_path)
            print(f"      RDP: {len(current_path)} -> {len(simplified_path)} points, score: {rdp_score:.2f}")
            if rdp_score > best_score:
                current_path = simplified_path
                best_score = rdp_score
    except Exception as e:
        print(f"      RDP failed: {e}")
    
    # Apply smoothing with user factor - try multiple approaches
    smoothing_factor = params.get('smoothing_factor', 0.005)
    
    # Method 1: Spline smoothing
    try:
        smoothed_path = smooth_path_spline(current_path, smoothing_factor)
        if len(smoothed_path) >= params.get('min_path_length', 3):
            smooth_score = circle_system.evaluate_path(smoothed_path)
            print(f"      Spline smooth: {len(current_path)} -> {len(smoothed_path)} points, score: {smooth_score:.2f}")
            if smooth_score > best_score * 0.95:  # Accept if within 95% of best score
                current_path = smoothed_path
                best_score = smooth_score
    except Exception as e:
        print(f"      Spline smoothing failed: {e}")
    
    # Method 2: Polynomial curve fitting for longer paths
    if len(current_path) >= 6:
        try:
            poly_path = fit_curve_to_path(current_path, 'polynomial', degree=3)
            if len(poly_path) >= params.get('min_path_length', 3):
                poly_score = circle_system.evaluate_path(poly_path)
                print(f"      Polynomial: {len(current_path)} -> {len(poly_path)} points, score: {poly_score:.2f}")
                if poly_score > best_score * 0.95:  # Accept if within 95% of best score
                    current_path = poly_path
                    best_score = poly_score
        except Exception as e:
            print(f"      Polynomial fitting failed: {e}")
    
    print(f"    Final: {len(current_path)} points, score: {best_score:.2f}")
    return current_path, best_score

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
    
    # Save uploaded file
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"centerline_{session_id}_{file.filename}")
    file.save(temp_path)
    
    try:
        # Load and process image
        gray = load_and_process_image(temp_path)
        session.image = gray
        session.image_path = temp_path
        
        # Store session
        sessions[session_id] = session
        
        # Convert image to base64 for display
        pil_img = Image.fromarray((gray * 255).astype(np.uint8))
        buf = BytesIO()
        pil_img.save(buf, format='PNG')
        buf.seek(0)
        img_data = base64.b64encode(buf.read()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'image_data': img_data,
            'image_shape': gray.shape,
            'parameters': session.parameters
        })
        
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'})

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

@app.route('/generate_svg', methods=['POST'])
def generate_svg():
    """Generate and return SVG visualization."""
    data = request.json
    session_id = data.get('session_id')
    
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session'})
    
    session = sessions[session_id]
    results = session.results
    
    if not results or 'error' in results:
        return jsonify({'error': 'No valid results to visualize'})
    
    try:
        # Create temporary SVG file
        temp_svg = os.path.join(tempfile.gettempdir(), f"centerline_{session_id}.svg")
        
        # Update global configuration temporarily
        import worksok3_optimized as wo
        original_output = wo.OUTPUT_PATH
        original_show_bitmap = wo.SHOW_BITMAP
        original_show_pre = wo.SHOW_PRE_OPTIMIZATION_PATHS
        
        wo.OUTPUT_PATH = temp_svg
        wo.SHOW_BITMAP = True  # Always show bitmap
        wo.SHOW_PRE_OPTIMIZATION_PATHS = True  # Always show pre-optimization paths
        
        # Generate SVG
        create_svg_output(
            session.image,
            results['circle_system'],
            results['optimized_paths'],
            results['optimized_scores'],
            results['pre_optimization_paths']
        )
        
        # Restore original configuration
        wo.OUTPUT_PATH = original_output
        wo.SHOW_BITMAP = original_show_bitmap
        wo.SHOW_PRE_OPTIMIZATION_PATHS = original_show_pre
        
        # Read SVG content
        with open(temp_svg, 'r') as f:
            svg_content = f.read()
        
        # Clean up
        os.remove(temp_svg)
        
        return jsonify({
            'success': True,
            'svg_content': svg_content
        })
        
    except Exception as e:
        return jsonify({'error': f'SVG generation error: {str(e)}'})

@app.route('/download_svg/<session_id>')
def download_svg(session_id):
    """Download SVG file."""
    if session_id not in sessions:
        return "Invalid session", 404
    
    session = sessions[session_id]
    results = session.results
    
    if not results or 'error' in results:
        return "No valid results", 404
    
    try:
        # Create temporary SVG file
        temp_svg = os.path.join(tempfile.gettempdir(), f"centerline_download_{session_id}.svg")
        
        # Generate SVG with fixed settings
        import worksok3_optimized as wo
        original_output = wo.OUTPUT_PATH
        wo.OUTPUT_PATH = temp_svg
        
        create_svg_output(
            session.image,
            results['circle_system'],
            results['optimized_paths'],
            results['optimized_scores'],
            results['pre_optimization_paths']
        )
        
        wo.OUTPUT_PATH = original_output
        
        return send_file(temp_svg, as_attachment=True, download_name='centerline_extraction.svg')
        
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
