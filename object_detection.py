import cv2
from ultralytics import YOLO
import numpy as np

def load_yolo_model():
    """Initialize and return YOLO model"""
    print("Loading YOLOv8 model...")
    model = YOLO('yolov8s.pt')  # Using the existing YOLOv8s model
    return model

def detect_objects(frame, yolo_model):
    """Detect objects in frame using YOLOv8"""
    # Convert frame to RGB (YOLOv8 expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run YOLOv8 inference
    results = yolo_model(rgb_frame)
    
    # Process results
    detections = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = f"{yolo_model.names[cls_id]} {conf:.2f}"
            detections.append({
                'box': (x1, y1, x2, y2),
                'label': label,
                'class_id': cls_id,
                'confidence': float(conf)
            })
    
    return detections

def draw_detections(frame, detections):
    """Draw bounding boxes and labels on frame"""
    for det in detections:
        x1, y1, x2, y2 = det['box']
        label = det['label']
        
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
        
        # Draw label text
        cv2.putText(frame, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    return frame

def main():
    # Load YOLO model
    print("Loading YOLOv8 model...")
    yolo_model = load_yolo_model()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Starting webcam feed. Press 'q' to quit.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame. Exiting...")
                break
            
            # Detect objects
            detections = detect_objects(frame, yolo_model)
            
            # Draw detections on frame
            frame_with_detections = draw_detections(frame.copy(), detections)
            
            # Show the frame with FPS
            cv2.imshow('YOLOv8 Object Detection', frame_with_detections)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam released and windows closed.")

if __name__ == "__main__":
    main()
