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
def initialize_models():
    """Loads all the required models and returns them."""
    print("Initializing models...")
    yolo_model = YOLO('yolov8x.pt') # This will automatically download the model if not present
    
    # Initialize BLIP model
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
    blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME).to(device)

    print("Models initialized successfully.")
    return yolo_model, blip_processor, blip_model

yolo_model, blip_processor, blip_model = initialize_models()

# --- Global variables for real-time analysis ---
video_camera = None # Initialize as None
last_analysis = {"objects": [], "description": "Initializing..."}
analysis_lock = threading.Lock()
FRAME_INTERVAL_FOR_ANALYSIS = 20 # Analyze one frame every 20 frames for more frequent updates
frame_counter = 0

# --- 2. Image Processing Functions ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_frame(frame):
    """Runs the full analysis pipeline on a single video frame."""
    global last_analysis
    print("Analyzing a new frame...")

    try:
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

        # --- Image Captioning with BLIP ---
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        inputs = blip_processor(pil_image, return_tensors="pt").to(blip_model.device)
        out = blip_model.generate(**inputs)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)

        # --- Format the analysis into a detailed description ---
        description = format_analysis_as_description(caption, detected_objects)

        with analysis_lock:
            last_analysis["description"] = description
            last_analysis["objects"] = detected_objects # Also update the objects list
        print(f"Detailed analysis updated.")

    except Exception as e:
        print(f"Error during frame analysis: {e}")

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

@app.route('/start_camera')
def start_camera():
    global video_camera
    if video_camera is None or not video_camera.isOpened():
        video_camera = cv2.VideoCapture(0)
    return jsonify(status="Camera started")

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
    app.run(debug=True, port=5002)
