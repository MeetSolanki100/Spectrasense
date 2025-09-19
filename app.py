
from flask import Flask, render_template, Response
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

app = Flask(__name__)

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
ANTI_SPOOF_THRESHOLD = 0.95 # Adjusted threshold for anti-spoofing (increased for stricter classification)

# Directory to store frames
OUTPUT_FRAMES_DIR = "/Users/kabirmathur/Documents/a_s/Kabir_Mathur"
CACHE_FILE = os.path.join(OUTPUT_FRAMES_DIR, "face_encodings_cache.pkl") # Define cache file path
os.makedirs(OUTPUT_FRAMES_DIR, exist_ok=True)

# Face Recognition variables
known_face_encodings = []
known_face_names = []

def load_known_faces(known_faces_dir):
    global known_face_encodings, known_face_names

    if os.path.exists(CACHE_FILE):
        print("Loading known faces from cache...")
        with open(CACHE_FILE, "rb") as f:
            known_face_encodings, known_face_names = pickle.load(f)
        print(f"Loaded {len(known_face_encodings)} known faces from cache.")
        return

    print("No cache found. Loading a random subset of known faces from directory...")
    all_image_files = [f for f in os.listdir(known_faces_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
    # Select a random subset of up to 50 images, or fewer if less than 50 are available
    selected_image_files = random.sample(all_image_files, min(len(all_image_files), 50))

    for filename in selected_image_files:
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)
        if face_encodings:
            known_face_encodings.append(face_encodings[0])
            known_face_names.append(os.path.basename(known_faces_dir)) # Storing folder name as person's name
    print(f"Loaded {len(known_face_encodings)} known faces from directory.")

    # Save to cache
    with open(CACHE_FILE, "wb") as f:
        pickle.dump((known_face_encodings, known_face_names), f)
    print(f"Saved {len(known_face_encodings)} known faces to cache.")

# Load known faces from the Kabir_Mathur directory on startup
with app.app_context(): # Run this within the Flask app context
    load_known_faces(OUTPUT_FRAMES_DIR) # Using OUTPUT_FRAMES_DIR as the known faces directory

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

# Initialize YOLO face detector
yolo_model = YOLO('yolov8n.pt') # Reverted to generic YOLOv8n model for stability
yolo_model.to(device)
yolo_model = yolo_model.to(torch.float32) # Ensure YOLO model is also float32

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

def generate_frames():
    global monitoring_active
    frame_count = 0 # Initialize frame counter
    PROCESS_EVERY_N_FRAMES = 1 # Process anti-spoofing and recognition every 1st frame

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream. Please check webcam permissions or if another application is using it.")
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

    while True:
        if not monitoring_active:
            # If monitoring is not active, yield a blank frame or a static image
            # For now, we'll yield a blank gray frame
            blank_frame = np.full((480, 640, 3), 128, dtype=np.uint8) # Gray frame
            ret, buffer = cv2.imencode('.jpg', blank_frame)
            frame = buffer.tobytes()
            yield (b'--frame\n'
                   b'Content-Type: image/jpeg\n\n' + frame + b'\n')
            continue # Skip processing if not active

        success, frame = cap.read()
        if not success:
            break

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
                            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                            identified_name = "Unknown" # Initialize identified_name

                            if True in matches:
                                first_match_index = matches.index(True)
                                identified_name = known_face_names[first_match_index]
                            
                            # Print the identified name
                            print(f"Identified Face: {identified_name}")
                            cv2.putText(frame, identified_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                else:
                    # If not processing this frame for anti-spoofing/recognition, just draw YOLO box if needed
                    # No box is drawn for fake faces as per user request
                    pass

        # Increment frame count
        frame_count += 1

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
    app.run(debug=True)
