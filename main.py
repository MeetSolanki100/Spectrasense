import torch
from ultralytics import YOLO
import easyocr
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import cv2
import os
from flask import Flask, request, render_template, url_for, Response, jsonify, redirect
from werkzeug.utils import secure_filename
import time
import threading
import numpy as np
import platform

# --- Configuration ---
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- 1. Initialize Models ---
def get_optimal_device():
    """Determines the optimal device for inference on Jetson Nano."""
    print("Detecting optimal device...")
    
    # Check if we're on Jetson Nano
    is_jetson = platform.machine() == 'aarch64' and 'jetson' in platform.platform().lower()
    
    if is_jetson:
        print("Jetson Nano detected!")
        if torch.cuda.is_available():
            # Check CUDA memory availability on Jetson
            try:
                torch.cuda.empty_cache()
                # Test CUDA allocation
                test_tensor = torch.zeros(1).cuda()
                del test_tensor
                torch.cuda.empty_cache()
                device = "cuda"
                print(f"Using CUDA on Jetson Nano. GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            except RuntimeError as e:
                print(f"CUDA allocation failed: {e}. Falling back to CPU.")
                device = "cpu"
        else:
            device = "cpu"
    else:
        # Non-Jetson system
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    
    print(f"Selected device: {device}")
    return device

def initialize_models():
    """Loads all the required models and returns them."""
    print("Initializing models...")
    
    # Use smaller YOLO model for Jetson Nano to save memory
    device = get_optimal_device()
    if device == "cpu" or platform.machine() == 'aarch64':
        print("Using YOLOv8s (smaller model) for better performance on Jetson Nano")
        yolo_model = YOLO('yolov8s.pt')
    else:
        yolo_model = YOLO('yolov8x.pt')
    
    # Initialize BLIP model with memory optimization
    blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
    
    # Load BLIP model with memory optimization for Jetson
    if device == "cuda" and platform.machine() == 'aarch64':
        # Jetson Nano CUDA optimization
        blip_model = BlipForConditionalGeneration.from_pretrained(
            BLIP_MODEL_NAME,
            torch_dtype=torch.float16,  # Use half precision to save memory
            low_cpu_mem_usage=True
        ).to(device)
        # Enable memory efficient attention if available
        if hasattr(blip_model, 'gradient_checkpointing_enable'):
            blip_model.gradient_checkpointing_enable()
    else:
        blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME).to(device)

    print("Models initialized successfully.")
    return yolo_model, blip_processor, blip_model, device

yolo_model, blip_processor, blip_model, device = initialize_models()

# --- Global variables for real-time analysis ---
video_camera = None # Initialize as None
last_analysis = {"objects": [], "description": "Initializing..."}
analysis_lock = threading.Lock()

# Optimize frame analysis interval for Jetson Nano
is_jetson = platform.machine() == 'aarch64' and 'jetson' in platform.platform().lower()
if is_jetson:
    FRAME_INTERVAL_FOR_ANALYSIS = 30  # Analyze less frequently on Jetson to save resources
    print("Jetson Nano detected - using optimized frame analysis interval")
else:
    FRAME_INTERVAL_FOR_ANALYSIS = 20

frame_counter = 0

# Jetson-specific performance settings
if is_jetson:
    # Set optimal thread count for Jetson Nano
    torch.set_num_threads(4)
    # Enable memory efficient mode
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

# --- 2. Image Processing Functions ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_frame(frame):
    """Runs the full analysis pipeline on a single video frame."""
    global last_analysis
    print("Analyzing a new frame...")

    try:
        # Clear CUDA cache before processing (important for Jetson)
        if device == "cuda":
            torch.cuda.empty_cache()

        # --- Detailed Object Detection ---
        results = yolo_model(frame, verbose=False)
        detected_objects = []
        for result in results:
            for box in result.boxes:
                confidence = float(box.conf)
                if confidence > 0.4:
                    class_id = int(box.cls)
                    class_name = yolo_model.names[class_id]
                    detected_objects.append(f"{class_name} (Confidence: {confidence:.2f})")

        # --- Image Captioning with BLIP (with memory optimization) ---
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Resize image for Jetson Nano to save memory
        if is_jetson:
            pil_image = pil_image.resize((320, 240), Image.Resampling.LANCZOS)
        
        inputs = blip_processor(pil_image, return_tensors="pt").to(blip_model.device)
        
        # Use optimized generation for Jetson
        with torch.no_grad():  # Disable gradients to save memory
            if is_jetson and device == "cuda":
                # Use half precision on Jetson for memory efficiency
                inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
                blip_model.half()
            
            out = blip_model.generate(**inputs, max_length=50, num_beams=3)  # Limit generation for speed
            caption = blip_processor.decode(out[0], skip_special_tokens=True)

        # --- Format the analysis into a detailed description ---
        description = format_analysis_as_description(caption, detected_objects)

        with analysis_lock:
            last_analysis["description"] = description
            last_analysis["objects"] = detected_objects # Also update the objects list
        print(f"Detailed analysis updated.")

        # Clear CUDA cache after processing
        if device == "cuda":
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error during frame analysis: {e}")
        # Clear cache on error
        if device == "cuda":
            torch.cuda.empty_cache()

def format_analysis_as_description(caption, objects):
    """Formats the analysis into a descriptive list of observations."""
    description_lines = [
        f"- General scene: {caption.capitalize()}."
    ]

    # Extract just the object name, removing confidence score for cleaner sentences
    plain_objects = [obj.split(' (')[0] for obj in objects]
    unique_objects = sorted(list(set(plain_objects)))

    if unique_objects:
        description_lines.append("") # Add a newline for spacing
        for obj in unique_objects:
            article = "An" if obj[0].lower() in "aeiou" else "A"
            description_lines.append(f"- {article} {obj} is visible.")
            
    return "\n".join(description_lines)

def analyze_image(image_path):
    """Runs the full analysis pipeline on a single image."""
    print(f"Analyzing image: {image_path}")
    
    # Object Detection
    results = yolo_model(image_path)
    img = cv2.imread(image_path)
    detected_objects = []
    for result in results:
        for box in result.boxes:
            confidence = float(box.conf)
            if confidence > 0.25:
                class_id = int(box.cls)
                class_name = yolo_model.names[class_id]
                if class_name == 'Wardrobe':
                    class_name = 'Cupboard'
                detected_objects.append(class_name)
            # Draw bounding boxes on the image
            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save the annotated image
    annotated_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'annotated_' + os.path.basename(image_path))
    cv2.imwrite(annotated_image_path, img)

    # Image Captioning with BLIP
    pil_image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(pil_image, return_tensors="pt").to(blip_model.device)
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)

    # Combine results
    analysis = {
        "objects": list(set(detected_objects)),
        "description": caption
    }
    
    print(f"Analysis complete: {analysis}")
    return analysis, annotated_image_path

# --- 3. Flask Routes ---
@app.route('/')
def welcome():
    """Render the welcome page."""
    return render_template('welcome.html')

@app.route('/real_time')
def real_time():
    """Render the real-time analysis page."""
    return render_template('real_time.html')

@app.route('/upload_page')
def upload_page():
    """Render the video upload page."""
    return render_template('upload.html')

def generate_frames():
    global frame_counter, video_camera
    while True:
        if video_camera is None or not video_camera.isOpened():
            # If camera is off, send a blank frame
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            ret, buffer = cv2.imencode('.jpg', blank_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(1) # Avoid busy-waiting
            continue

        success, frame = video_camera.read()
        if not success:
            break
        else:
            # Periodically run the heavy analysis in a separate thread
            frame_counter += 1
            if frame_counter % FRAME_INTERVAL_FOR_ANALYSIS == 0:
                threading.Thread(target=analyze_frame, args=(frame.copy(),)).start()

            # Run lightweight object detection on every frame
            results = yolo_model(frame, verbose=False)
            detected_objects_in_frame = []
            for result in results:
                for box in result.boxes:
                    confidence = float(box.conf)
                    if confidence > 0.4:
                        class_id = int(box.cls)
                        class_name = yolo_model.names[class_id]
                        detected_objects_in_frame.append(class_name)
                        # Draw bounding boxes
                        x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            with analysis_lock:
                last_analysis["objects"] = list(set(detected_objects_in_frame))

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def get_camera_config():
    """Get optimal camera configuration for Jetson Nano."""
    is_jetson = platform.machine() == 'aarch64' and 'jetson' in platform.platform().lower()
    
    if is_jetson:
        # Jetson Nano camera configuration
        # Try CSI camera first (common on Jetson), then USB
        camera_configs = [
            {'index': 0, 'width': 640, 'height': 480, 'fps': 30},  # CSI camera
            {'index': 1, 'width': 640, 'height': 480, 'fps': 30},  # USB camera
        ]
    else:
        # Standard configuration for other systems
        camera_configs = [
            {'index': 0, 'width': 640, 'height': 480, 'fps': 30}
        ]
    
    return camera_configs

@app.route('/start_camera')
def start_camera():
    global video_camera
    if video_camera is None or not video_camera.isOpened():
        camera_configs = get_camera_config()
        
        for config in camera_configs:
            try:
                video_camera = cv2.VideoCapture(config['index'])
                if video_camera.isOpened():
                    # Set camera properties for Jetson Nano optimization
                    video_camera.set(cv2.CAP_PROP_FRAME_WIDTH, config['width'])
                    video_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config['height'])
                    video_camera.set(cv2.CAP_PROP_FPS, config['fps'])
                    
                    # Jetson-specific optimizations
                    if platform.machine() == 'aarch64':
                        video_camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size
                        video_camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
                    
                    print(f"Camera started with config: {config}")
                    return jsonify(status="Camera started", config=config)
                else:
                    video_camera.release()
                    video_camera = None
            except Exception as e:
                print(f"Failed to start camera {config['index']}: {e}")
                continue
        
        return jsonify(status="Failed to start camera", error="No working camera found")
    
    return jsonify(status="Camera already running")

@app.route('/stop_camera')
def stop_camera():
    global video_camera
    if video_camera is not None:
        video_camera.release()
        video_camera = None
    return jsonify(status="Camera stopped")

@app.route('/analysis_data')
def analysis_data():
    """Endpoint to get the latest analysis data."""
    with analysis_lock:
        return jsonify(last_analysis)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return redirect(request.url)
    file = request.files['video']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)

        # Start analysis
        processed_video_path, analysis_results = analyze_video(video_path)

        return render_template('video_analysis.html', 
                               video_path=processed_video_path, 
                               analysis=analysis_results)
    return redirect(url_for('index'))

def analyze_video(video_path):
    """Runs analysis on a pre-recorded video."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the codec and create VideoWriter object
    output_filename = 'processed_' + os.path.basename(video_path)
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or 'XVID', 'MJPG', etc.
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    all_detected_objects = []
    video_descriptions = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Analyze one frame per second
        if frame_count % int(fps) == 0:
            # Object Detection
            results = yolo_model(frame, verbose=False)
            for result in results:
                for box in result.boxes:
                    if float(box.conf) > 0.3:
                        class_id = int(box.cls)
                        class_name = yolo_model.names[class_id]
                        all_detected_objects.append(class_name)
                        x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Generate description for this frame
            if frame_count % (int(fps) * 5) == 0: # Description every 5 seconds
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                inputs = blip_processor(pil_image, return_tensors="pt").to(blip_model.device)
                summary_out = blip_model.generate(**inputs)
                caption = blip_processor.decode(summary_out[0], skip_special_tokens=True)
                video_descriptions.append(caption)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    analysis = {
        "objects": list(set(all_detected_objects)),
        "description": video_descriptions
    }

    return output_filename, analysis

# --- Main Execution ---
if __name__ == "__main__":
    # Jetson Nano optimized Flask configuration
    if is_jetson:
        print("Starting Flask app optimized for Jetson Nano...")
        # Use threaded mode for better performance on Jetson
        app.run(host='0.0.0.0', port=5002, threaded=True, debug=False)
    else:
        app.run(debug=True, port=5002)
