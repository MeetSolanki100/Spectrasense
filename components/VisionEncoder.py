"""
Vision Encoder Module - Object Detection, OCR, and Room Description
Integrates YOLO, BLIP, EasyOCR, and TrOCR models for comprehensive image analysis
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
import platform
import easyocr
from transformers import BlipProcessor, BlipForConditionalGeneration, TrOCRProcessor, VisionEncoderDecoderModel
from typing import Dict, List, Any, Optional
import base64
from io import BytesIO


class VisionEncoder:
    """Vision encoder for object detection, OCR, and scene understanding"""
    
    def __init__(self):
        self.device = self._get_optimal_device()
        self.yolo_model = None
        self.blip_processor = None
        self.blip_model = None
        self.easyocr_reader = None
        self.trocr_processor = None
        self.trocr_model = None
        self.is_jetson = platform.machine() == 'aarch64' and 'jetson' in platform.platform().lower()
        self.text_detection_confidence = 0.5
        self.initialized = False
        
    def _get_optimal_device(self) -> str:
        """Determine the best available device for inference."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def initialize_models(self):
        """Initialize all required models with error handling."""
        if self.initialized:
            return
            
        try:
            print(f"🔧 Initializing Vision Encoder models on {self.device}...")
            
            # Initialize YOLO
            print("📦 Loading YOLO model...")
            self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.yolo_model.to(self.device).eval()
            
            # Initialize BLIP for image captioning
            print("📦 Loading BLIP model...")
            blip_model_name = "Salesforce/blip-image-captioning-base"
            self.blip_processor = BlipProcessor.from_pretrained(blip_model_name)
            self.blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_name).to(self.device)
            
            # Initialize EasyOCR for text detection
            print("📦 Initializing EasyOCR...")
            self.easyocr_reader = easyocr.Reader(['en'], gpu=(self.device == 'cuda'))

            # Initialize TrOCR for handwritten text
            print("📦 Loading TrOCR model...")
            self.trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten').to(self.device)
            
            self.initialized = True
            print("✅ All Vision Encoder models initialized successfully!")
            
        except Exception as e:
            print(f"❌ Error initializing Vision Encoder models: {e}")
            raise
    
    def get_room_description(self, image: Image.Image) -> str:
        """Generate a room description using BLIP model."""
        try:
            inputs = self.blip_processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                caption_ids = self.blip_model.generate(**inputs, max_length=50)
            return self.blip_processor.decode(caption_ids[0], skip_special_tokens=True)
        except Exception as e:
            print(f"❌ Error generating room description: {e}")
            return "Could not generate room description"
    
    def extract_text_with_trocr(self, image: np.ndarray) -> str:
        """Recognize handwritten text using TrOCR."""
        try:
            # Convert image to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)

            # Process image and generate text
            pixel_values = self.trocr_processor(images=pil_image, return_tensors="pt").pixel_values.to(self.device)
            with torch.no_grad():
                generated_ids = self.trocr_model.generate(pixel_values)
            
            generated_text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return generated_text
        except Exception as e:
            print(f"❌ Error in TrOCR text extraction: {e}")
            return ""
    
    def detect_text(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect text in the image using EasyOCR."""
        try:
            # Convert to RGB for better text detection
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.easyocr_reader.readtext(rgb_image)
            
            text_blocks = []
            for (bbox, text, prob) in results:
                if prob > self.text_detection_confidence:
                    text_blocks.append({
                        'text': text,
                        'confidence': float(prob),
                        'bbox': [int(x) for point in bbox for x in point]  # Flatten bbox points
                    })
            return text_blocks
        except Exception as e:
            print(f"❌ Error in text detection: {e}")
            return []
    
    def detect_objects(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects in the image using YOLO."""
        try:
            results = self.yolo_model(image)
            detections = results.pandas().xyxy[0]
            
            objects = []
            for _, det in detections.iterrows():
                if det['confidence'] > 0.5:  # Confidence threshold
                    objects.append({
                        'class': det['name'],
                        'confidence': float(det['confidence']),
                        'bbox': [int(x) for x in det[['xmin', 'ymin', 'xmax', 'ymax']].values]
                    })
            return objects
        except Exception as e:
            print(f"❌ Error in object detection: {e}")
            return []
    
    def analyze_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive image analysis including object detection, text detection, 
        handwritten text recognition, and room description.
        """
        if not self.initialized:
            self.initialize_models()
        
        try:
            # Convert frame to RGB (BLIP expects RGB)
            rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # 1. Object detection with YOLO
            objects = self.detect_objects(image)
            
            # 2. Text Detection (EasyOCR)
            text_blocks = self.detect_text(image)

            # 3. Handwritten Text Recognition (TrOCR)
            handwritten_text = self.extract_text_with_trocr(image)
            
            # 4. Generate Room Description
            room_description = self.get_room_description(pil_image)
            
            # 5. Return comprehensive analysis
            return {
                "objects": objects,
                "text_blocks": text_blocks,
                "handwritten_text": handwritten_text,
                "room_description": room_description,
                "status": "success"
            }
            
        except Exception as e:
            print(f"❌ Error in image analysis: {e}")
            return {
                "error": str(e),
                "status": "error"
            }
    
    def annotate_image(self, image: np.ndarray, analysis: Dict[str, Any]) -> np.ndarray:
        """Draw annotations on the image based on analysis results."""
        annotated_img = image.copy()
        
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
        
        return annotated_img
    
    def image_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy image to base64 string."""
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')
    
    def base64_to_image(self, base64_string: str) -> np.ndarray:
        """Convert base64 string to numpy image."""
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
