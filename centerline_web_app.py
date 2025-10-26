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
import json
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
            'rdp_tolerance': 2.0,
            'smoothing_factor': 0.005,
            'min_path_length': 3,
            'optimize_top_n_paths': 10
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
    
    # Fast optimization: Only optimize top N paths
    fast_optimize_count = min(5, len(valid_paths))
    sorted_indices = np.argsort(path_scores)[::-1]  # Best first
    top_indices = sorted_indices[:fast_optimize_count]
    
    optimized_paths = []
    optimized_scores = []
    
    for i, idx in enumerate(top_indices):
        path = valid_paths[idx]
        original_score = path_scores[idx]
        
        optimized_path, optimized_score = optimize_path_with_circles(path, circle_system)
        
        optimized_paths.append(optimized_path)
        optimized_scores.append(optimized_score)
    
    # Add remaining unoptimized paths for visualization
    remaining_count = min(params['optimize_top_n_paths'] - len(optimized_paths), 
                         len(valid_paths) - len(optimized_paths))
    if remaining_count > 0:
        remaining_indices = sorted_indices[len(optimized_paths):len(optimized_paths) + remaining_count]
        for idx in remaining_indices:
            optimized_paths.append(valid_paths[idx])
            optimized_scores.append(path_scores[idx])
    
    # Get corresponding pre-optimization paths for visualization
    all_indices = sorted_indices[:len(optimized_paths)]
    top_pre_optimization_paths = [valid_paths[idx] for idx in all_indices]
    
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
            
            # Prepare response data
            response = {
                'success': True,
                'stats': results['stats'],
                'initial_paths_count': results['initial_paths_count'],
                'merged_paths_count': results['merged_paths_count'],
                'valid_paths_count': results['valid_paths_count'],
                'best_score': results['best_score'],
                'paths_count': len(results['optimized_paths'])
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
