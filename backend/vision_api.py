"""
Minimal Vision API - Standalone server for vision features only
This avoids the heavy dependencies of the main API
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
import cv2
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.VisionEncoder import VisionEncoder

app = FastAPI(title="Vision API", description="Computer Vision Analysis API")

# CORS configuration - Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=False,  # Must be False when allow_origins is "*"
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Global vision encoder instance
vision_encoder = None

class AnalyzeFrameRequest(BaseModel):
    image_base64: str

@app.on_event("startup")
async def startup():
    """Initialize vision encoder on startup"""
    global vision_encoder
    print("🚀 Vision API starting...")
    vision_encoder = VisionEncoder()
    print("✅ Vision API ready!")

@app.options("/{full_path:path}")
async def options_handler(full_path: str):
    """Handle CORS preflight requests"""
    return {}

@app.get("/")
async def root():
    return {
        "message": "Vision API",
        "status": "running",
        "endpoints": {
            "status": "/api/vision/status",
            "initialize": "/api/vision/initialize",
            "analyze": "/api/vision/analyze",
            "analyze_frame": "/api/vision/analyze-frame"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "vision_encoder_initialized": vision_encoder is not None,
        "models_loaded": vision_encoder.initialized if vision_encoder else False
    }

@app.post("/api/vision/initialize")
async def initialize_vision_encoder():
    """Initialize the vision encoder models"""
    global vision_encoder
    try:
        if vision_encoder is None:
            print("🔧 Creating VisionEncoder instance...")
            vision_encoder = VisionEncoder()
        
        if not vision_encoder.initialized:
            print("🔧 Initializing vision models (this may take 2-5 minutes)...")
            print("📥 Downloading models if not cached...")
            vision_encoder.initialize_models()
            print("✅ Vision models loaded successfully!")
        else:
            print("ℹ️ Models already initialized")
        
        return {
            "status": "success",
            "message": "Vision Encoder initialized successfully",
            "device": vision_encoder.device,
            "models_loaded": vision_encoder.initialized
        }
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Initialization Error: {e}")
        print(f"❌ Traceback:\n{error_details}")
        raise HTTPException(
            status_code=500, 
            detail=f"Vision Encoder initialization failed: {str(e)}"
        )

@app.post("/api/vision/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """Analyze an uploaded image for objects, text, and scene description"""
    if not vision_encoder:
        raise HTTPException(status_code=503, detail="Vision Encoder not initialized")
    
    if not vision_encoder.initialized:
        print("🔧 Auto-initializing models...")
        vision_encoder.initialize_models()
    
    try:
        # Read the uploaded file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Could not read image")
        
        # Analyze the image
        analysis = vision_encoder.analyze_image(image)
        
        # Annotate the image
        annotated_image = vision_encoder.annotate_image(image, analysis)
        
        # Convert images to base64
        original_base64 = vision_encoder.image_to_base64(image)
        annotated_base64 = vision_encoder.image_to_base64(annotated_image)
        
        return {
            "status": "success",
            "analysis": {
                "objects": [{"class": obj["class"], "confidence": obj["confidence"]} 
                           for obj in analysis.get("objects", [])],
                "text_blocks": [{"text": tb["text"], "confidence": tb["confidence"]} 
                               for tb in analysis.get("text_blocks", [])],
                "handwritten_text": analysis.get("handwritten_text", ""),
                "room_description": analysis.get("room_description", "No description available")
            },
            "images": {
                "original": original_base64,
                "annotated": annotated_base64
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")

@app.get("/api/vision/status")
async def get_vision_status():
    """Get the status of the vision encoder"""
    return {
        "status": "success",
        "vision_encoder_initialized": vision_encoder is not None,
        "models_loaded": vision_encoder.initialized if vision_encoder else False,
        "device": vision_encoder.device if vision_encoder else "N/A"
    }

@app.post("/api/vision/analyze-frame")
async def analyze_frame(request: AnalyzeFrameRequest):
    """Analyze a frame from webcam (base64 encoded)"""
    if not vision_encoder:
        raise HTTPException(status_code=503, detail="Vision Encoder not initialized")
    
    if not vision_encoder.initialized:
        print("🔧 Auto-initializing models...")
        vision_encoder.initialize_models()
    
    try:
        # Decode base64 image
        image = vision_encoder.base64_to_image(request.image_base64)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # Analyze the image
        analysis = vision_encoder.analyze_image(image)
        
        return {
            "status": "success",
            "analysis": {
                "objects": [obj["class"] for obj in analysis.get("objects", [])],
                "text_blocks": [tb["text"] for tb in analysis.get("text_blocks", [])],
                "handwritten_text": analysis.get("handwritten_text", ""),
                "room_description": analysis.get("room_description", "Analyzing...")
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=f"Frame analysis failed: {str(e)}")

# Stub endpoints for compatibility with main API
@app.get("/api/chats")
async def get_chats():
    return {"status": "success", "total": 0, "returned": 0, "chats": []}

@app.get("/api/stats")
async def get_stats():
    return {
        "status": "success",
        "stats": {
            "total_conversations": 0,
            "chatbot_active": False,
            "database_path": "./chroma_db",
            "collection_name": "conversations"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Vision API on port 8000...")
    uvicorn.run("vision_api:app", host="0.0.0.0", port=8000, reload=True)
