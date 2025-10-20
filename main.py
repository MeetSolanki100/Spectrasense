# main.py - Updated with OCR and Room Description

import os
import cv2
import torch
import numpy as np
from PIL import Image
from flask import Flask, render_template, Response, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import time
import threading
import platform
import json
import re
from pathlib import Path
import pytesseract
from transformers import BlipProcessor, BlipForConditionalGeneration

# Initialize Flask app with static folder
app = Flask(__name__, static_folder='static', static_url_path='')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configuration
YOLO_MODEL = 'yolov8s.pt'
BLIP_MODEL = "Salesforce/blip-image-captioning-base"
TEXT_DETECTION_CONFIDENCE = 0.25  # Lowered further to catch more text

# Initialize models and device
yolo_model = None
blip_processor = None
blip_model = None
device = None
is_jetson = platform.machine() == 'aarch64' and 'jetson' in platform.platform().lower()

def get_optimal_device():
    """Determine the best available device for inference."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def initialize_models():
    """Initialize all required models with error handling."""
    global yolo_model, blip_processor, blip_model, device
    
    try:
        print("Initializing models...")
        device = get_optimal_device()
        print(f"Using device: {device}")
        
        # Initialize YOLO
        print("Loading YOLO model...")
        yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        yolo_model.to(device).eval()
        
        # Initialize BLIP for image captioning
        print("Loading BLIP model...")
        blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL)
        blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL).to(device)
        
        # Check if pytesseract is available
        print("Checking pytesseract...")
        try:
            pytesseract.get_tesseract_version()
            print(f"Pytesseract version: {pytesseract.get_tesseract_version()}")
        except Exception as e:
            print(f"Warning: Pytesseract not found or not configured properly: {e}")
            print("OCR functionality may not work. Install tesseract: brew install tesseract (macOS)")
        
        print("All models initialized successfully!")
        
    except Exception as e:
        print(f"Error initializing models: {e}")
        raise

# Initialize models
initialize_models()

# Global variables for real-time analysis
last_analysis = {
    "objects": [],
    "caption": "Initializing...",
    "text_blocks": [],
    "room_description": "No description available yet"
}
analysis_lock = threading.Lock()
frame_counter = 0
video_capture = None
frame_generator_thread = None
stop_event = threading.Event()

# --- Background Thread for Frame Generation ---
def frame_generator_task():
    """A background task that reads frames from the camera and analyzes them."""
    global video_capture, frame_counter
    
    while not stop_event.is_set():
        if video_capture is None or not video_capture.isOpened():
            time.sleep(0.1)
            continue
            
        ret, frame = video_capture.read()
        if not ret:
            time.sleep(0.1)
            continue
            
        frame_counter += 1
        analyze_frame(frame.copy())

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}

def get_room_description(image):
    """Generate a room description using BLIP model."""
    try:
        inputs = blip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            caption_ids = blip_model.generate(**inputs, max_length=50)
        return blip_processor.decode(caption_ids[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error generating room description: {e}")
        return "Could not generate room description"

def preprocess_image_for_ocr(image, method='adaptive'):
    """Preprocess image to improve OCR accuracy with multiple methods."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if method == 'adaptive':
        # Adaptive thresholding - best for varying lighting
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    elif method == 'otsu':
        # Otsu's thresholding - best for bimodal images
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'mean':
        # Mean adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
        )
    else:
        binary = gray
    
    # Denoise the image
    denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
    
    # Apply slight dilation to make text more readable
    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.dilate(denoised, kernel, iterations=1)
    
    return processed

def upscale_image(image, scale=2):
    """Upscale image for better OCR on small text."""
    height, width = image.shape[:2]
    return cv2.resize(image, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)

def sharpen_image(image):
    """Sharpen image to enhance text edges."""
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    return cv2.filter2D(image, -1, kernel)

def deskew_image(image):
    """Deskew image to correct text angle."""
    coords = np.column_stack(np.where(image > 0))
    if len(coords) == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    # Only deskew if angle is significant
    if abs(angle) > 0.5:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    return image

def extract_text_fast(image):
    """Fast OCR for real-time analysis - optimized for speed."""
    try:
        # Quick upscale
        upscaled = upscale_image(image, scale=2)
        
        # Single best method: Adaptive threshold
        processed = preprocess_image_for_ocr(upscaled, method='adaptive')
        pil_processed = Image.fromarray(processed)
        
        # Single OCR pass with PSM 3 (fastest general mode)
        text = pytesseract.image_to_string(pil_processed, config='--psm 3 --oem 3')
        
        # Clean up
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        cleaned_text = '\n'.join(lines)
        
        return cleaned_text if cleaned_text else "No text detected"
        
    except Exception as e:
        print(f"Error in fast text extraction: {e}")
        return ""

def extract_text_with_pytesseract(image):
    """Extract text from image using Pytesseract OCR with advanced preprocessing."""
    try:
        results = []
        
        # Upscale image for better small text recognition
        upscaled = upscale_image(image, scale=2)
        
        # 1. Original image (RGB)
        rgb_image = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        text1 = pytesseract.image_to_string(pil_image, config='--psm 3 --oem 3')
        results.append(text1)
        
        # 2. Sharpened image for better edge detection
        sharpened = sharpen_image(upscaled)
        rgb_sharp = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)
        pil_sharp = Image.fromarray(rgb_sharp)
        text2 = pytesseract.image_to_string(pil_sharp, config='--psm 3 --oem 3')
        results.append(text2)
        
        # 3. Adaptive thresholding (best for varying lighting)
        processed_adaptive = preprocess_image_for_ocr(upscaled, method='adaptive')
        processed_adaptive = deskew_image(processed_adaptive)
        pil_adaptive = Image.fromarray(processed_adaptive)
        text3 = pytesseract.image_to_string(pil_adaptive, config='--psm 3 --oem 3')
        results.append(text3)
        
        # 4. Otsu's thresholding (best for clear backgrounds)
        processed_otsu = preprocess_image_for_ocr(upscaled, method='otsu')
        pil_otsu = Image.fromarray(processed_otsu)
        text4 = pytesseract.image_to_string(pil_otsu, config='--psm 3 --oem 3')
        results.append(text4)
        
        # 5. Mean thresholding
        processed_mean = preprocess_image_for_ocr(upscaled, method='mean')
        pil_mean = Image.fromarray(processed_mean)
        text5 = pytesseract.image_to_string(pil_mean, config='--psm 3 --oem 3')
        results.append(text5)
        
        # 6. PSM 6: Single uniform block of text
        text6 = pytesseract.image_to_string(pil_adaptive, config='--psm 6 --oem 3')
        results.append(text6)
        
        # 7. PSM 11: Sparse text detection
        text7 = pytesseract.image_to_string(pil_image, config='--psm 11 --oem 3')
        results.append(text7)
        
        # 8. PSM 4: Single column of text
        text8 = pytesseract.image_to_string(pil_adaptive, config='--psm 4 --oem 3')
        results.append(text8)
        
        # Find the result with the most content
        # Filter out very short results (likely noise)
        valid_results = [r for r in results if len(r.strip()) > 3]
        
        if not valid_results:
            return "No text detected"
        
        # Return the longest result (usually most complete)
        combined_text = max(valid_results, key=len)
        
        # Clean up the text
        lines = [line.strip() for line in combined_text.split('\n') if line.strip()]
        cleaned_text = '\n'.join(lines)
        
        return cleaned_text if cleaned_text else "No text detected"
        
    except Exception as e:
        print(f"Error in Pytesseract text extraction: {e}")
        return ""

def detect_text_fast(image):
    """Fast text detection for real-time analysis."""
    try:
        # Moderate upscale for speed
        upscaled = upscale_image(image, scale=2)
        scale_factor = 2
        
        # Single best preprocessing method
        processed = preprocess_image_for_ocr(upscaled, method='adaptive')
        pil_processed = Image.fromarray(processed)
        
        # Single OCR pass
        data = pytesseract.image_to_data(pil_processed, output_type=pytesseract.Output.DICT, config='--psm 3 --oem 3')
        
        text_blocks = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            if int(data['conf'][i]) > 0 and data['text'][i].strip():
                confidence = int(data['conf'][i]) / 100.0
                if confidence > TEXT_DETECTION_CONFIDENCE:
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    
                    # Scale back to original coordinates
                    x, y, w, h = x // scale_factor, y // scale_factor, w // scale_factor, h // scale_factor
                    
                    bbox = [x, y, x+w, y, x+w, y+h, x, y+h]
                    text_blocks.append({
                        'text': data['text'][i],
                        'confidence': float(confidence),
                        'bbox': bbox
                    })
        
        # Sort by position
        text_blocks.sort(key=lambda b: (b['bbox'][1], b['bbox'][0]))
        
        return text_blocks
    except Exception as e:
        print(f"Error in fast text detection: {e}")
        return []

def detect_text(image):
    """Detect text in the image using Pytesseract with bounding boxes."""
    try:
        # Upscale for better detection
        upscaled = upscale_image(image, scale=2)
        scale_factor = 2
        
        # Convert to RGB for better text detection
        rgb_image = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Try multiple preprocessing methods
        processed_adaptive = preprocess_image_for_ocr(upscaled, method='adaptive')
        processed_adaptive = deskew_image(processed_adaptive)
        pil_adaptive = Image.fromarray(processed_adaptive)
        
        processed_otsu = preprocess_image_for_ocr(upscaled, method='otsu')
        pil_otsu = Image.fromarray(processed_otsu)
        
        # Get detailed data including bounding boxes from multiple approaches
        data1 = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT, config='--psm 3 --oem 3')
        data2 = pytesseract.image_to_data(pil_adaptive, output_type=pytesseract.Output.DICT, config='--psm 3 --oem 3')
        data3 = pytesseract.image_to_data(pil_otsu, output_type=pytesseract.Output.DICT, config='--psm 3 --oem 3')
        
        # Combine results from all approaches
        text_blocks = []
        seen_text_positions = {}
        
        for data in [data1, data2, data3]:
            n_boxes = len(data['text'])
            
            for i in range(n_boxes):
                # Filter out empty text and low confidence
                if int(data['conf'][i]) > 0 and data['text'][i].strip():
                    confidence = int(data['conf'][i]) / 100.0  # Convert to 0-1 range
                    if confidence > TEXT_DETECTION_CONFIDENCE:
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                        
                        # Scale back to original coordinates
                        x, y, w, h = x // scale_factor, y // scale_factor, w // scale_factor, h // scale_factor
                        
                        # Use position and text as key to avoid duplicates but keep best confidence
                        pos_key = (x // 5, y // 5, data['text'][i].strip())
                        
                        if pos_key not in seen_text_positions or confidence > seen_text_positions[pos_key]['confidence']:
                            bbox = [x, y, x+w, y, x+w, y+h, x, y+h]
                            seen_text_positions[pos_key] = {
                                'text': data['text'][i],
                                'confidence': float(confidence),
                                'bbox': bbox
                            }
        
        # Convert to list
        text_blocks = list(seen_text_positions.values())
        
        # Sort by vertical position (top to bottom) then horizontal (left to right)
        text_blocks.sort(key=lambda b: (b['bbox'][1], b['bbox'][0]))
        
        return text_blocks
    except Exception as e:
        print(f"Error in text detection: {e}")
        return []

def analyze_frame(frame, fast_mode=True):
    """Analyze a single frame for objects, text, and generate room description.
    
    Args:
        frame: Input frame to analyze
        fast_mode: If True, use fast OCR (1-2s). If False, use comprehensive OCR (3-5s)
    """
    global last_analysis
    
    try:
        # Convert frame to RGB (BLIP expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # 1. Object detection with YOLO
        results = yolo_model(frame)
        detections = results.pandas().xyxy[0]
        
        # Process detections
        objects = []
        for _, det in detections.iterrows():
            if det['confidence'] > 0.5:  # Confidence threshold
                objects.append({
                    'class': det['name'],
                    'confidence': float(det['confidence']),
                    'bbox': [int(x) for x in det[['xmin', 'ymin', 'xmax', 'ymax']].values]
                })
        
        # 2. Text Detection and Extraction (Pytesseract)
        # Use fast mode for real-time, comprehensive for uploads
        if fast_mode:
            text_blocks = detect_text_fast(frame)
            full_text = extract_text_fast(frame)
        else:
            text_blocks = detect_text(frame)
            full_text = extract_text_with_pytesseract(frame)
        
        # 3. Generate Room Description
        room_description = get_room_description(pil_image)
        
        # 4. Update last_analysis with all results
        with analysis_lock:
            last_analysis = {
                "objects": objects,
                "text_blocks": text_blocks,
                "full_text": full_text,
                "room_description": room_description,
                "caption": room_description,  # For backward compatibility
                "timestamp": time.time()
            }
            
        return last_analysis
        
    except Exception as e:
        print(f"Error in frame analysis: {e}")
        return {"error": str(e)}

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/real_time')
def real_time():
    return render_template('real_time.html')

@app.route('/upload_page')
def upload_page():
    return render_template('upload.html')

def generate_frames_for_feed():
    """Generates frames for the video feed by drawing on the latest captured frame."""
    while True:
        if video_capture is None or not video_capture.isOpened():
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            _, buffer = cv2.imencode('.jpg', blank)
            frame_bytes = buffer.tobytes()
        else:
            ret, frame = video_capture.read()
            if not ret:
                time.sleep(0.1)
                continue

            with analysis_lock:
                analysis = last_analysis.copy()

            for obj in analysis.get('objects', []):
                x1, y1, x2, y2 = obj['bbox']
                label = f"{obj['class']} {obj['confidence']:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            for text_block in analysis.get('text_blocks', []):
                points = np.array(text_block['bbox']).reshape(-1, 2).astype(np.int32)
                cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 255), thickness=2)

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames_for_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera')
def start_camera():
    global video_capture, frame_counter, last_analysis, frame_generator_thread, stop_event
    
    try:
        # Release existing capture if exists
        if video_capture is not None:
            video_capture.release()
            
        # Try to initialize camera
        video_capture = cv2.VideoCapture(0)
        
        # Check if camera opened successfully
        if not video_capture.isOpened():
            return jsonify({
                "status": "error",
                "message": "Failed to open camera. Please ensure it's connected and not in use by another application."
            }), 500
            
        # Set camera properties
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Reset analysis data
        with analysis_lock:
            last_analysis = {
                "objects": [],
                "caption": "Initializing...",
                "room_description": "Initializing...",
                "text_blocks": [],
                "timestamp": time.time()
            }
        
        # Start frame generation in a separate thread if not already running
        if frame_generator_thread is None or not frame_generator_thread.is_alive():
            stop_event.clear()
            frame_generator_thread = threading.Thread(target=frame_generator_task, name='frame_generator')
            frame_generator_thread.daemon = True
            frame_generator_thread.start()
        
        return jsonify({
            "status": "success",
            "message": "Camera initialized successfully",
            "resolution": {
                "width": int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            }
        })
        
    except Exception as e:
        error_msg = f"Camera initialization error: {str(e)}"
        print(error_msg)
        if video_capture is not None:
            video_capture.release()
            video_capture = None
        return jsonify({
            "status": "error",
            "message": error_msg,
            "error_type": str(type(e).__name__)
        }), 500

@app.route('/stop_camera')
def stop_camera():
    global video_capture
    global frame_generator_thread
    if video_capture is not None:
        video_capture.release()
        video_capture = None
    
    stop_event.set()
    if frame_generator_thread is not None:
        frame_generator_thread.join(timeout=2)
        frame_generator_thread = None
    return jsonify({"status": "success", "message": "Camera stopped"})

@app.route('/analysis_data')
def analysis_data():
    try:
        if video_capture is None or not video_capture.isOpened():
            return jsonify({
                "error": "Camera not initialized",
                "status": "error",
                "message": "Camera is not ready. Please ensure the camera is connected and accessible.",
                "objects": [],
                "caption": "Camera not available",
                "room_description": "Camera not available",
                "text_blocks": [],
                "horizontal_text": [],
                "timestamp": time.time()
            }), 503  # Service Unavailable

        with analysis_lock:
            # Create a deep copy to prevent race conditions during iteration
            current_analysis = last_analysis.copy()

            # Format the response with all analysis data
            response = {
                "status": "success",
                "objects": [obj.get('class', 'unknown') for obj in current_analysis.get('objects', [])],
                "caption": current_analysis.get('room_description', 'Analyzing...'),
                "room_description": current_analysis.get('room_description', 'Analyzing...'),
                "caption_confidence": current_analysis.get('caption_confidence', 0.0),
                "text_blocks": current_analysis.get('text_blocks', []),
                "full_text": current_analysis.get('full_text', ''),
                "timestamp": current_analysis.get('timestamp', time.time())
            }
            return jsonify(response)
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "message": "Error processing analysis data",
            "objects": [],
            "caption": "Analysis error",
            "room_description": "Analysis error",
            "text_blocks": [],
            "horizontal_text": [],
            "timestamp": time.time()
        }), 500

@app.route('/analyze_media', methods=['POST'])
def analyze_media():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read and process the image
            if filename.lower().endswith(('png', 'jpg', 'jpeg')):
                img = cv2.imread(filepath)
                if img is None:
                    return jsonify({'error': 'Could not read image'}), 400
                
                # Analyze the image with comprehensive mode (not real-time)
                analysis = analyze_frame(img, fast_mode=False)
                
                # Save annotated image
                annotated_img = img.copy()
                
                # Draw object detections
                for obj in analysis.get('objects', []):
                    x1, y1, x2, y2 = obj['bbox']
                    label = f"{obj['class']} {obj['confidence']:.2f}"
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(annotated_img, (x1, y1-20), (x1 + len(label)*8, y1), (0, 255, 0), -1)
                    cv2.putText(annotated_img, label, (x1, y1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # Draw text detections
                for text_block in analysis.get('text_blocks', []):
                    points = np.array(text_block['bbox']).reshape(-1, 2).astype(np.int32)
                    cv2.polylines(annotated_img, [points], isClosed=True, color=(0, 255, 255), thickness=2)
                    # Add text label
                    cv2.putText(annotated_img, text_block['text'], 
                               (points[0][0], points[0][1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                annotated_filename = f"annotated_{filename}"
                annotated_path = os.path.join(app.config['UPLOAD_FOLDER'], annotated_filename)
                cv2.imwrite(annotated_path, annotated_img)
                
                # Prepare response
                response = {
                    'type': 'image',
                    'original': url_for('static', filename=f'uploads/{filename}'),
                    'annotated': url_for('static', filename=f'uploads/{annotated_filename}'),
                    'analysis': {
                        'objects': [{'class': obj['class'], 'confidence': obj['confidence']} 
                                  for obj in analysis.get('objects', [])],
                        'room_description': analysis.get('room_description', 'No description available'),
                        'text_blocks': [{'text': tb['text'], 'confidence': tb['confidence']} 
                                      for tb in analysis.get('text_blocks', [])],
                        'full_text': analysis.get('full_text', '')
                    }
                }
                
                return jsonify(response)
                
            else:
                return jsonify({'error': 'Video processing not implemented yet'}), 400
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Run the app
    try:
        app.run(host='0.0.0.0', port=5005, debug=True, threaded=True)
    finally:
        # Cleanup
        if 'video_capture' in globals() and video_capture is not None:
            video_capture.release()