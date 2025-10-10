from flask import Flask, render_template, Response, jsonify, request
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np # Import numpy
import os
import datetime
import face_recognition # Import face_recognition
import pickle # Import pickle for caching
import random # Import random for selecting subset of images
import geocoder # Import geocoder for location services
import json
import time
import shutil

app = Flask(__name__)

# Configuration
KNOWN_FACES_DIR = "known_faces"
METADATA_FILE = os.path.join(KNOWN_FACES_DIR, "metadata.json")
CACHE_FILE = os.path.join(KNOWN_FACES_DIR, "face_encodings_cache.pkl")
OUTPUT_FRAMES_DIR = "/Users/kabirmathur/Documents/a_s/Kabir_Mathur"
IDENTIFIED_PERSONS_DIR = "/Users/kabirmathur/Documents/a_s/identified_persons"
os.makedirs(IDENTIFIED_PERSONS_DIR, exist_ok=True)
os.makedirs(OUTPUT_FRAMES_DIR, exist_ok=True)

# Determine device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Global variable to control monitoring state
monitoring_active = False
latest_identification = {
    "name": None,
    "time": None,
    "location": None,
    "face_encoding": None,
    "face_location": None,
    "is_unknown": False
}
ANTI_SPOOF_THRESHOLD = 0.95 # Adjusted threshold for anti-spoofing (increased for stricter classification)

# Face Recognition variables
known_face_encodings = []
known_face_names = []

def save_face_encoding(name, encoding, image):
    """Save a new face encoding and its image to the known faces directory."""
    # Create output directory if it doesn't exist
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    
    # Load or create metadata
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {"known_faces": []}
    
    # Create person's directory if it doesn't exist
    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    
    # Generate a unique filename
    timestamp = int(time.time())
    filename = f"{timestamp}.jpg"
    filepath = os.path.join(person_dir, filename)
    
    # Save the face image
    cv2.imwrite(filepath, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    # Load or create encodings for this person
    encodings_file = os.path.join(person_dir, 'encodings.pkl')
    if os.path.exists(encodings_file):
        with open(encodings_file, 'rb') as f:
            person_encodings = pickle.load(f)
    else:
        person_encodings = []
        metadata['known_faces'].append(name)
    
    # Add new encoding
    person_encodings.append(encoding)
    
    # Save updated encodings
    with open(encodings_file, 'wb') as f:
        pickle.dump(person_encodings, f)
    
    # Save updated metadata
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f)
    
    # Reload known faces
    global known_face_encodings, known_face_names
    known_face_encodings, known_face_names = load_known_faces()
    
    print(f"Saved new face: {name} ({filename})")
    return True

def load_known_faces():
    known_face_encodings = []
    known_face_names = []
    
    # Create the known_faces directory if it doesn't exist
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    
    # If metadata file doesn't exist, create it
    if not os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'w') as f:
            json.dump({"known_faces": []}, f)
        return known_face_encodings, known_face_names
    
    # Load metadata
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)
    
    # Process each known person
    for person_name in metadata['known_faces']:
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        if not os.path.exists(person_dir):
            continue
            
        # Load encodings if they exist
        encodings_file = os.path.join(person_dir, 'encodings.pkl')
        if os.path.exists(encodings_file):
            try:
                with open(encodings_file, 'rb') as f:
                    person_encodings = pickle.load(f)
                    known_face_encodings.extend(person_encodings)
                    known_face_names.extend([person_name] * len(person_encodings))
            except Exception as e:
                print(f"Error loading encodings for {person_name}: {e}")
    
    # Save to cache
    if known_face_encodings:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump((known_face_encodings, known_face_names), f)
    
    print(f"Loaded {len(known_face_encodings)} known faces from {len(set(known_face_names))} different people.")
    return known_face_encodings, known_face_names

# Load known faces on startup
with app.app_context(): # Run this within the Flask app context
    load_known_faces()  # Load known faces from the known_faces directory

# Load the anti-spoofing model
model_path = '/Users/kabirmathur/Documents/a_s/antispoof_vit_traced.pt'

anti_spoof_model = torch.jit.load(model_path, map_location=torch.device('cpu'))
anti_spoof_model.eval()
anti_spoof_model = anti_spoof_model.to(torch.float32).to(device) # Convert to float32 and then move to device

# Define preprocessing for the anti-spoofing model
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), # Reverted to 0.5 mean/std
])

# Initialize YOLO face detector with explicit configuration
try:
    # Try loading with the latest API first
    yolo_model = YOLO('yolov8n.pt')
    # Test the model with a dummy input to ensure it's properly loaded
    yolo_model.predict(torch.zeros((1, 3, 640, 640), device=device), verbose=False)
    print("YOLO model loaded successfully with default settings")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print("Trying alternative loading method...")
    try:
        # Fallback to explicit model loading with CPU first
        yolo_model = YOLO('yolov8n.pt', task='detect')
        yolo_model = yolo_model.cpu()
        # Test with a small image
        yolo_model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
        print("YOLO model loaded successfully with CPU-first method")
    except Exception as e2:
        print(f"Failed to load YOLO model: {e2}")
        raise RuntimeError("Could not initialize YOLO model. Please check your installation and try again.")

# Move model to the appropriate device
yolo_model.to(device)
if device.type == 'cuda':
    yolo_model = yolo_model.half()  # Use half precision for CUDA
    print("Using half precision (FP16) for YOLO model on CUDA")
else:
    yolo_model = yolo_model.float()  # Use full precision for CPU/MPS
    print("Using full precision (FP32) for YOLO model")

# Get class names from the YOLO model
class_names = yolo_model.names
face_class_id = None
for class_id, name in class_names.items():
    if name == 'person': # YOLOv8n typically detects 'person' for faces
        face_class_id = class_id
        break

if face_class_id is None:
    print("Warning: 'person' class not found in YOLO model. Face detection might not work as expected.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_monitoring')
def start_monitoring():
    global monitoring_active
    monitoring_active = True
    print("Monitoring started via Flask route.")
    return "Monitoring started"

@app.route('/stop_monitoring')
def stop_monitoring():
    global monitoring_active
    monitoring_active = False
    print("Monitoring stopped via Flask route.")
    return "Monitoring stopped"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/latest_identification')
def latest_identification_data():
    # Create a new dictionary with only the data we want to send
    response_data = {
        'name': latest_identification.get('name'),
        'time': latest_identification.get('time'),
        'location': latest_identification.get('location'),
        'is_unknown': latest_identification.get('is_unknown', False)
    }
    return jsonify(response_data)

def clear_known_faces():
    """Clear all known face data and reload from disk."""
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []
    # Reload known faces from disk
    load_known_faces()

@app.route('/clear_faces', methods=['POST'])
def clear_faces():
    """Clear all known faces and reset the system."""
    try:
        # Remove the entire known_faces directory
        if os.path.exists(KNOWN_FACES_DIR):
            shutil.rmtree(KNOWN_FACES_DIR)
        
        # Recreate the directory structure
        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
        
        # Reset global variables
        global known_face_encodings, known_face_names
        known_face_encodings = []
        known_face_names = []
        
        # Create fresh metadata file
        with open(METADATA_FILE, 'w') as f:
            json.dump({"known_faces": []}, f)
        
        return jsonify({
            "status": "success", 
            "message": "Successfully cleared all known faces. The system has been reset."
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to clear known faces: {str(e)}"
        }), 500

@app.route('/register_face', methods=['POST'])
def register_face():
    global latest_identification, known_face_encodings, known_face_names
    
    data = request.get_json()
    name = data.get('name')
    
    if not name or not latest_identification.get('is_unknown') or not latest_identification.get('face_encoding'):
        return jsonify({
            "status": "error", 
            "message": "Invalid request or no face to register. Make sure a face is detected and marked as unknown."
        }), 400
    
    # Get the face encoding and image
    face_encoding = np.array(latest_identification['face_encoding'])
    face_location = latest_identification['face_location']
    
    # Save the new face encoding
    face_encoding = latest_identification['face_encoding']
    face_image = latest_identification.get('last_frame')
    
    if face_image is None:
        return jsonify({"status": "error", "message": "No face image available"}), 400
        
    # Convert face_encoding from list back to numpy array if needed
    if isinstance(face_encoding, list):
        face_encoding = np.array(face_encoding)
    
    # Extract face from frame using face_location
    top, right, bottom, left = face_location
    face_image = face_image[top:bottom, left:right]
    
    # Clear and reload known faces to ensure consistency
    clear_known_faces()
    
    # Save the new face encoding and image to disk
    save_face_encoding(name, face_encoding, face_image)
    
    # Reload known faces to include the newly added one
    global known_face_encodings, known_face_names
    known_face_encodings, known_face_names = load_known_faces()
    
    # Update the latest identification with the new face data
    latest_identification.update({
        'name': name,
        'is_unknown': False,
        'face_encoding': face_encoding.tolist(),
        'face_location': face_location,
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })
    
    print(f"Successfully registered new face: {name}")
    print(f"Total known faces after registration: {len(known_face_encodings)}")
    
    return jsonify({"status": "success", "message": f"Face registered as {name}"})

def open_camera_jetson():
    # GStreamer pipeline for Jetson Nano CSI camera
    pipeline = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! "
        "nvvidconv flip-method=0 ! "
        "video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print("Successfully opened camera using GStreamer pipeline.")
        return cap
    else:
        print("Failed to open camera with GStreamer, falling back to standard method.")
        # Fallback to standard USB camera detection
        for camera_index in [0, 1, 2]:
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                print(f"Successfully opened camera at index {camera_index}")
                return cap
    return None

def generate_frames():
    global monitoring_active
    frame_count = 0  # Initialize frame counter
    PROCESS_EVERY_N_FRAMES = 1  # Process anti-spoofing and recognition every 1st frame

    cap = open_camera_jetson()

    if not cap or not cap.isOpened():
        print("Error: Could not open any video stream. Please check webcam permissions or if another application is using it.")
        # Generate a blank frame with an error message
        font = cv2.FONT_HERSHEY_SIMPLEX
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        text = "Webcam Error: Check permissions or if in use."
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        text_x = (img.shape[1] - text_size[0]) // 2
        text_y = (img.shape[0] + text_size[1]) // 2
        cv2.putText(img, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        while True:
            yield (b'--frame\n' 
                   b'Content-Type: image/jpeg\n\n' + frame + b'\n')

    # Set camera resolution (adjust these values based on your camera's capabilities)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Warm-up the camera (some cameras need a few frames to adjust)
    for _ in range(5):
        cap.read()

    while True:
        if not monitoring_active:
            # Create a more informative waiting frame
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            text = "Monitoring Paused - Click 'Start Monitoring' to begin"
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
            text_x = (blank_frame.shape[1] - text_size[0]) // 2
            text_y = (blank_frame.shape[0] + text_size[1]) // 2
            cv2.putText(blank_frame, text, (text_x, text_y), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            ret, buffer = cv2.imencode('.jpg', blank_frame)
            frame = buffer.tobytes()
            yield (b'--frame\n' 
                   b'Content-Type: image/jpeg\n\n' + frame + b'\n')
            continue  # Skip processing if not active

        success, frame = cap.read()
        if not success:
            print("Failed to grab frame from camera")
            # Create an error frame
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, "Failed to grab frame from camera", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', error_frame)
            frame = buffer.tobytes()
            yield (b'--frame\n' 
                   b'Content-Type: image/jpeg\n\n' + frame + b'\n')
            continue

        # Convert frame to RGB for face recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_identified_this_frame = False
        
        # Store the current frame for potential registration
        latest_identification['last_frame'] = rgb_frame.copy()

        results = yolo_model(frame)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Filter for 'person' class if identified
                if face_class_id is not None and int(box.cls[0]) != face_class_id:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                # Only perform anti-spoofing and recognition on every Nth frame
                if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                    # Preprocess the face for the anti-spoofing model
                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face_pil = Image.fromarray(face_rgb)
                    input_tensor = preprocess(face_pil).unsqueeze(0)

                    # Predict using the anti-spoofing model
                    with torch.no_grad():
                        output = anti_spoof_model(input_tensor.to(device))
                        # Corrected: Assuming output is [logit_real, logit_fake] and 'real' is index 0.
                        probability = torch.sigmoid(output[:, 0]).item() # Take the first logit (index 0) and apply sigmoid
                        is_real = probability > ANTI_SPOOF_THRESHOLD # Use configurable threshold

                    print(f"Face detected: Probability = {probability:.4f}, Is Real = {is_real}") # Debugging output

                    if is_real:
                        color = (0, 255, 0) # Green for real faces

                        # Perform face recognition for the detected face
                        face_image_for_recognition = frame[y1:y2, x1:x2]
                        # Convert from BGR to RGB (face_recognition expects RGB)
                        face_image_for_recognition_rgb = cv2.cvtColor(face_image_for_recognition, cv2.COLOR_BGR2RGB)
                        
                        face_locations = face_recognition.face_locations(face_image_for_recognition_rgb)
                        if face_locations:
                            # Assuming only one face per crop for recognition purposes
                            face_encoding = face_recognition.face_encodings(face_image_for_recognition_rgb, face_locations)[0]

                            # Compare with known faces
                            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                            best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else -1
                            
                            # Use a threshold to determine if it's a match
                            if best_match_index != -1 and face_distances[best_match_index] < 0.6:  # Lower is better match
                                identified_name = known_face_names[best_match_index]
                                is_unknown = False
                            else:
                                identified_name = "Unknown"
                                is_unknown = True
                            
                            # Store the face encoding and location for potential registration
                            latest_identification["face_encoding"] = face_encoding.tolist()
                            latest_identification["face_location"] = face_locations[0]  # Store the face location
                            latest_identification["is_unknown"] = is_unknown
                            
                            # Print the identified name
                            print(f"Identified Face: {identified_name} (Distance: {face_distances[best_match_index] if best_match_index != -1 else 'N/A'})")
                            
                            # Draw bounding box and text with different colors for known/unknown
                            if is_unknown:
                                # Yellow for unknown faces
                                box_color = (0, 255, 255)  # Yellow in BGR
                                text_color = (0, 0, 0)     # Black text for better visibility on yellow
                                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                                cv2.putText(frame, "Unknown - Click to Register", (x1, y1 - 10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                            else:
                                # Green for known faces
                                box_color = (0, 255, 0)    # Green in BGR
                                text_color = (255, 255, 255)  # White text
                                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                                cv2.putText(frame, identified_name, (x1, y1 - 10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

                            # Get current time and location
                            now = datetime.datetime.now()
                            current_time = now.strftime("%Y-%m-%d %H:%M:%S")
                            g = geocoder.ip('me')
                            location = g.city if g.city else "Unknown Location"

                            # Update latest identification
                            latest_identification["name"] = identified_name
                            latest_identification["time"] = current_time
                            latest_identification["location"] = location

                            if not is_unknown:  # Only save frames for known faces
                                # Save the frame with the identified person
                                filename = f"{identified_name},{current_time},{location}.jpg".replace(" ", "_").replace(":", "-")
                                filepath = os.path.join(IDENTIFIED_PERSONS_DIR, filename)
                                cv2.imwrite(filepath, frame)
                                face_identified_this_frame = True

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                else:
                    # If not processing this frame for anti-spoofing/recognition, just draw YOLO box if needed
                    # No box is drawn for fake faces as per user request
                    pass

        # Increment frame count
        frame_count += 1

        if not face_identified_this_frame:
            latest_identification["name"] = None
            latest_identification["time"] = None
            latest_identification["location"] = None

        # # Save the processed frame (Disabled for performance)
        # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # frame_filename = os.path.join(OUTPUT_FRAMES_DIR, f"frame_{timestamp}.jpg")
        # cv2.imwrite(frame_filename, frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\n'
               b'Content-Type: image/jpeg\n\n' + frame + b'\n')

    cap.release()

if __name__ == "__main__":
    app.run(debug=True, port=5001)
