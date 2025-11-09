import sys
import os

# Add mock transformers_stream_generator to path
sys.path.insert(0, '/tmp/mock_transformers_stream_generator')

from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
from PIL import Image
import json
import time
from ultralytics import YOLO
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor
try:
    from transformers.generation.beam_search import BeamSearchScorer
except ImportError:
    # Fallback for newer versions of transformers
    try:
        from transformers.generation import BeamSearchScorer
    except ImportError:
        # If still not found, we'll define a dummy class to prevent import errors
        class BeamSearchScorer:
            def __init__(self, *args, **kwargs):
                raise ImportError("BeamSearchScorer not available. Please check your transformers installation.")
import os
from io import BytesIO
import base64
import tempfile
import logging

# Global variables
yolo_model = None
qwen_model = None
qwen_processor = None

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize global variables for models
def init_globals():
    global yolo_model, qwen_model, qwen_processor
    yolo_model = None
    qwen_model = None
    qwen_processor = None

app = Flask(__name__)

# Initialize models
yolo_model = None
qwen_model = None
qwen_processor = None

def get_optimal_device():
    """Get the best available device with fallback to CPU if needed"""
    try:
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    except Exception as e:
        logger.warning(f"Error getting optimal device: {e}. Falling back to CPU.")
        return 'cpu'

def initialize_models():
    """Initialize all required models with better error handling and 4-bit quantization"""
    global qwen_model, qwen_processor, yolo_model
    
    # Initialize globals
    init_globals()
    
    device = get_optimal_device()
    logger.info(f"Using device: {device}")
    
    try:
        # Initialize YOLO with yolov8x
        global yolo_model
        yolo_model = YOLO('yolov8x.pt').to(device)
        logger.info("YOLOv8x model loaded successfully")
        
        # Try to import transformers_stream_generator
        try:
            from transformers_stream_generator import init_stream_support
            init_stream_support()
            logger.info("Initialized stream support")
        except ImportError:
            logger.warning("transformers_stream_generator not available, continuing without streaming support")
        
        # Using a simpler image captioning model
        model_id = "Salesforce/blip-image-captioning-base"
        
        logger.info(f"Loading image captioning model {model_id}...")
        
        # Load processor and model
        qwen_processor = BlipProcessor.from_pretrained(model_id)
        qwen_model = BlipForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        ).eval()
        
        logger.info("Qwen-VL model loaded successfully with 4-bit quantization")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}", exc_info=True)
        return False

def detect_objects(image_np):
    """Detect objects in the image using YOLO with error handling"""
    try:
        results = yolo_model(image_np)
        detections = []
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = yolo_model.names[cls]
                
                detections.append({
                    'label': label,
                    'confidence': float(conf),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                })
        
        return detections
    except Exception as e:
        logger.error(f"Error in object detection: {e}")
        return []

def analyze_with_qwen(image_pil, query="What is in this image?"):
    """Generate a description of the image"""
    global qwen_model, qwen_processor
    
    if qwen_model is None or qwen_processor is None:
        return "Error: Model not initialized. Please restart the application."
        
    try:
        # Convert PIL Image to RGB if it's not
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')
        
        # Process the image
        inputs = qwen_processor(images=image_pil, return_tensors="pt").to(qwen_model.device)
        
        # Generate caption
        with torch.no_grad():
            outputs = qwen_model.generate(
                **inputs,
                max_length=50,
                num_beams=5,
                temperature=0.7
            )
        
        # Decode the response
        response = qwen_processor.decode(outputs[0], skip_special_tokens=True)
        return response.strip()
        
        # Generate response
        inputs = qwen_tokenizer(prompt, return_tensors="pt", return_token_type_ids=False)
        
        with torch.no_grad():
            outputs = qwen_model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=qwen_tokenizer.eos_token_id
            )
        
        response = qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
        
    except Exception as e:
        logger.error(f"Error in analyze_with_qwen: {str(e)}", exc_info=True)
        return f"Error processing image: {str(e)}"

def analyze_scene(image_pil, detections=None):
    """Generate a description of the scene with fallback mechanisms"""
    try:
        if not isinstance(image_pil, Image.Image):
            image_pil = Image.fromarray(cv2.cvtColor(image_pil, cv2.COLOR_BGR2RGB))
        
        try:
            return analyze_with_qwen(image_pil)
        except Exception as e:
            logger.warning(f"Qwen-VL analysis failed, falling back to YOLO: {str(e)}")
            
        if detections is None:
            detections = detect_objects(np.array(image_pil))
            
        if not detections:
            return "I don't see any objects in this image."
        
        class_counts = {}
        for det in detections:
            label = det.get('label', 'object')
            class_counts[label] = class_counts.get(label, 0) + 1
        
        items = [f"{count} {label}{'s' if count > 1 else ''}" 
                for label, count in class_counts.items()]
        
        if not items:
            return "I can't identify any objects in this image."
        elif len(items) == 1:
            return f"I can see {items[0]} in the image."
        else:
            return f"I can see {', '.join(items[:-1])} and {items[-1]} in the image."
            
    except Exception as e:
        logger.error(f"Error in scene analysis: {str(e)}", exc_info=True)
        return "I'm having trouble analyzing this image."

@app.route('/')
def index():
    return render_template('real_time.html')

@app.route('/stop_camera')
def stop_camera():
    """Endpoint to stop the camera"""
    return jsonify({'status': 'success', 'message': 'Camera stopped'})

@app.route('/start_camera')
def start_camera():
    """Endpoint to initialize the camera"""
    try:
        # Test if we can open the camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return jsonify({
                'status': 'error',
                'message': 'Could not open camera. Make sure it is connected and not in use by another application.'
            }), 500
        
        # Release the camera immediately after testing
        cap.release()
        return jsonify({
            'status': 'success',
            'message': 'Camera initialized successfully'
        })
    except Exception as e:
        logger.error(f"Error initializing camera: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error initializing camera: {str(e)}'
        }), 500

def gen_frames():
    """Video streaming generator function with real-time object detection."""
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        logger.error("Error: Could not open video device")
        return

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break
                
            # Get object detections
            detections = detect_objects(frame)
            
            # Draw bounding boxes
            for d in detections:
                if d.get('confidence', 0) > 0.5:  # Only draw high confidence detections
                    x1, y1, x2, y2 = map(int, d.get('bbox', [0, 0, 0, 0]))
                    label = f"{d.get('label', 'object')} {d.get('confidence', 0):.1f}"
                    
                    # Draw rectangle and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Convert the frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    except Exception as e:
        logger.error(f"Error in video feed: {str(e)}")
    finally:
        camera.release()

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/analyze_frame', methods=['POST'])
def analyze_frame():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'status': 'error', 'message': 'No image data provided'}), 400
        
        try:
            img_data = data['image']
            image_bytes = base64.b64decode(img_data.split(',')[-1])
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
            
            # Get object detections
            image_np = np.array(image)
            detections = detect_objects(image_np)
            
            # Process detections for response
            objects_dict = {}
            objects_count = {}
            
            for i, obj in enumerate(detections):
                obj_id = f"obj_{i}_{int(time.time() * 1000) % 10000}"
                bbox = obj.get('bbox', [])
                label = obj.get('label', 'unknown')
                
                objects_dict[obj_id] = {
                    'label': label,
                    'confidence': float(obj.get('confidence', 0)),
                    'bbox': [float(x) for x in bbox]
                }
                
                objects_count[label] = objects_count.get(label, 0) + 1
            
            # Generate scene description
            analysis = analyze_scene(image, detections)
            
            return jsonify({
                'status': 'success',
                'detection_count': len(detections),
                'objects': objects_dict,
                'objects_summary': objects_count,
                'analysis': analysis
            })
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}", exc_info=True)
            return jsonify({
                'status': 'error',
                'message': f'Error processing image: {str(e)}'
            }), 500
            
    except Exception as e:
        logger.error(f"Unexpected error in analyze_frame: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': 'An unexpected error occurred'
        }), 500

if __name__ == '__main__':
    try:
        if initialize_models():
            logger.info("Starting server on http://0.0.0.0:5008")
            app.run(debug=True, host='0.0.0.0', port=5008)
        else:
            logger.error("Failed to initialize models. Exiting...")
    except Exception as e:
        logger.critical(f"Failed to start server: {str(e)}", exc_info=True)