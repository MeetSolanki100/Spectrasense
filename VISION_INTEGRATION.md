# Vision Encoder Integration

## Overview
The Vision Encoder module has been successfully integrated into the Spectrasense application, adding comprehensive computer vision capabilities including:

- **Object Detection** using YOLOv5
- **Text Detection** using EasyOCR
- **Handwritten Text Recognition** using TrOCR
- **Scene Understanding** using BLIP (image captioning)

## Features

### 1. Real-Time Camera Analysis
- Start/stop webcam feed
- Capture and analyze frames in real-time
- Display detected objects, text, and scene descriptions

### 2. Image Upload Analysis
- Upload images for comprehensive analysis
- View original and annotated images side-by-side
- Get detailed results including:
  - Detected objects with confidence scores
  - Extracted text with confidence scores
  - Handwritten text recognition
  - Room/scene descriptions

## Architecture

### Backend Components

#### VisionEncoder.py (`components/VisionEncoder.py`)
Core module that handles:
- Model initialization (YOLO, BLIP, EasyOCR, TrOCR)
- Image analysis pipeline
- Object detection
- Text detection and OCR
- Image annotation

#### API Endpoints (`backend/api.py`)

1. **POST `/api/vision/initialize`**
   - Initialize vision encoder models
   - Returns: status, device info

2. **POST `/api/vision/analyze`**
   - Upload and analyze an image file
   - Returns: analysis results + base64 encoded images (original & annotated)

3. **POST `/api/vision/analyze-frame`**
   - Analyze a base64 encoded frame from webcam
   - Returns: analysis results (objects, text, description)

4. **GET `/api/vision/status`**
   - Check vision encoder initialization status
   - Returns: initialization status, device info

### Frontend Components

#### Vision Tab (`frontend/src/App.jsx`)
New "Vision" tab added to the navigation with:

1. **Vision Status Panel**
   - Shows initialization status
   - Initialize models button
   - Device information

2. **Real-Time Camera Section**
   - Start/Stop camera controls
   - Live video feed
   - Capture and analyze button

3. **Image Upload Section**
   - File upload button
   - Side-by-side view of original and annotated images

4. **Analysis Results Panel**
   - Room description
   - Detected objects (with confidence)
   - Detected text (with confidence)
   - Handwritten text recognition

## Installation

### 1. Install Dependencies

```bash
cd /Users/kabirmathur/Documents/spectra_GUI/Spectrasense
pip install -r requirements.txt
```

New dependencies added:
- opencv-python
- easyocr
- ultralytics
- yolov5

### 2. Start Backend

```bash
cd backend
python api.py
```

The backend will automatically initialize the Vision Encoder on startup.

### 3. Start Frontend

```bash
cd frontend
npm install  # if not already installed
npm run dev
```

## Usage

### Using the Vision Tab

1. **Navigate to Vision Tab**
   - Click on the "Vision" tab in the navigation

2. **Initialize Models** (First Time)
   - Click "Initialize Models" button
   - Wait for models to download and load
   - Status will change to "Models Loaded"

3. **Real-Time Camera Analysis**
   - Click "Start Camera" to activate webcam
   - Click "Analyze Frame" to analyze current view
   - Results appear in the Analysis Results panel

4. **Upload Image Analysis**
   - Click "Upload Image"
   - Select an image file
   - View original and annotated images
   - Review detailed analysis results

## Model Information

### YOLOv5s
- **Purpose**: Object detection
- **Size**: ~28MB
- **Speed**: Fast inference
- **Classes**: 80 COCO classes

### BLIP (Salesforce/blip-image-captioning-base)
- **Purpose**: Image captioning and scene understanding
- **Size**: ~990MB
- **Output**: Natural language descriptions

### EasyOCR
- **Purpose**: Text detection and recognition
- **Languages**: English (configurable)
- **Accuracy**: High for printed text

### TrOCR (microsoft/trocr-base-handwritten)
- **Purpose**: Handwritten text recognition
- **Size**: ~334MB
- **Specialty**: Handwritten text

## Device Support

The Vision Encoder automatically detects and uses the best available device:
- **CUDA** (NVIDIA GPU) - Fastest
- **MPS** (Apple Silicon) - Fast on M1/M2/M3 Macs
- **CPU** - Fallback option

## API Response Examples

### Analyze Image Response
```json
{
  "status": "success",
  "analysis": {
    "objects": [
      {"class": "person", "confidence": 0.95},
      {"class": "laptop", "confidence": 0.87}
    ],
    "text_blocks": [
      {"text": "Hello World", "confidence": 0.92}
    ],
    "handwritten_text": "Sample handwritten text",
    "room_description": "a person sitting at a desk with a laptop"
  },
  "images": {
    "original": "base64_encoded_image...",
    "annotated": "base64_encoded_image..."
  }
}
```

### Analyze Frame Response
```json
{
  "status": "success",
  "analysis": {
    "objects": ["person", "laptop", "chair"],
    "text_blocks": ["Hello World"],
    "handwritten_text": "",
    "room_description": "a person working at a desk"
  }
}
```

## Troubleshooting

### Models Not Loading
- Ensure you have sufficient disk space (~2GB for all models)
- Check internet connection for first-time model downloads
- Verify Python dependencies are installed correctly

### Camera Not Working
- Grant browser camera permissions
- Ensure no other application is using the camera
- Try refreshing the page

### Slow Performance
- Models will download on first use (one-time)
- CPU inference is slower than GPU
- Consider using a machine with CUDA or MPS support

## Future Enhancements

Potential improvements:
- Video file upload and analysis
- Batch image processing
- Custom object detection training
- Multi-language OCR support
- Real-time continuous analysis mode
- Export analysis results to PDF/JSON

## Integration with Existing Features

The Vision Encoder integrates seamlessly with existing Spectrasense features:
- Can be used alongside voice chat
- Analysis results can be stored in vector database
- Compatible with translation features
- Shares the same modern UI/UX

## File Structure

```
Spectrasense/
├── components/
│   └── VisionEncoder.py          # Core vision module
├── backend/
│   └── api.py                     # Updated with vision endpoints
├── frontend/
│   └── src/
│       └── App.jsx                # Updated with Vision tab
├── requirements.txt               # Updated with vision dependencies
└── VISION_INTEGRATION.md          # This file
```

## Credits

This integration combines:
- Original vision encoder from `/Users/kabirmathur/Documents/spectra_GUI/vision_encoder/main.py`
- Spectrasense voice assistant framework
- Modern React frontend with Tailwind CSS
