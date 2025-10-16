from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from contextlib import asynccontextmanager
import asyncio
import json
import sys
import os

# Add parent directory to path to import components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.SpeechChatbot import SpeechChatbot
from components.VectorDB import RAGChatbot
from components.TextToSpeech import TextToSpeech
from components.Translate import Translate

# Global instances
chatbot_instance = None
vector_db = RAGChatbot()
tts = TextToSpeech()
translator = Translate()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global chatbot_instance
    try:
        chatbot_instance = SpeechChatbot(
            whisper_model="base",
            llm_model="llama3.1:8b",
            glasses_device=None
        )
        print("✅ Chatbot initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing chatbot: {e}")
    
    yield
    
    # Shutdown (if needed)
    print("Shutting down...")

app = FastAPI(title="Voice Assistant API", lifespan=lifespan)

# CORS configuration - MUST BE RIGHT AFTER app creation
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)
class ChatbotConfig(BaseModel):
    whisper_model: str = "base"
    llm_model: str = "llama3.1:8b"
    glasses_device: Optional[str] = None

class MessageRequest(BaseModel):
    message: str
    translate: bool = False
    target_lang: str = "hi"

class ChatMessage(BaseModel):
    id: str
    user_message: str
    bot_response: str
    timestamp: str

class DeleteChatRequest(BaseModel):
    chat_ids: List[str]

@app.get("/")
async def root():
    return {
        "message": "Voice Assistant API",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "chat": "/api/chat",
            "voice": "/api/voice",
            "chats": "/api/chats",
            "delete": "/api/chats/delete"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "chatbot_initialized": chatbot_instance is not None,
        "vector_db_count": vector_db.collection.count()
    }

@app.post("/api/initialize")
async def initialize_chatbot(config: ChatbotConfig):
    """Initialize or reconfigure the chatbot"""
    global chatbot_instance
    try:
        chatbot_instance = SpeechChatbot(
            whisper_model=config.whisper_model,
            llm_model=config.llm_model,
            glasses_device=config.glasses_device
        )
        return {
            "status": "success",
            "message": "Chatbot initialized",
            "config": config.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")

@app.post("/api/chat")
async def chat(request: MessageRequest):
    """Process text message and get response"""
    if not chatbot_instance:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    try:
        # Process conversation
        response = chatbot_instance.process_conversation(request.message)
        
        # Translate if requested
        final_response = response
        if request.translate:
            final_response = translator.translate_and_transliterate(
                response, 
                target_lang_code=request.target_lang
            )
        
        # Generate audio and play it
        import uuid
        audio_filename = f"output_{uuid.uuid4().hex}.mp3"
        audio_file = tts.synthesize_speech(final_response, output_file=audio_filename)
        
        # Play audio through speakers/glasses
        from components.SmartGlassesAudio import SmartGlassesAudio
        glasses = SmartGlassesAudio()
        glasses.play_audio_to_glasses(audio_file)
        
        # Clean up audio file
        try:
            import os
            os.remove(audio_file)
        except:
            pass
        
        return {
            "status": "success",
            "user_message": request.message,
            "bot_response": response,
            "translated_response": final_response if request.translate else None,
            "timestamp": vector_db.collection.get(limit=1)["metadatas"][0]["timestamp"] if vector_db.collection.count() > 0 else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@app.post("/api/voice/start")
async def start_voice_session():
    """Start voice interaction session"""
    if not chatbot_instance:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    return {
        "status": "success",
        "message": "Voice session ready. Use WebSocket at /ws/voice for real-time interaction"
    }

@app.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket):
    """WebSocket endpoint for voice interactions"""
    await websocket.accept()
    
    if not chatbot_instance:
        await websocket.send_json({"error": "Chatbot not initialized"})
        await websocket.close()
        return
    
    try:
        while True:
            # Receive audio data or commands
            data = await websocket.receive_json()
            
            if data.get("action") == "record":
                # Record audio
                await websocket.send_json({"status": "recording", "message": "Speak now..."})
                
                audio_data = chatbot_instance.stt.record_audio(duration=data.get("duration", 5))
                user_input = chatbot_instance.stt.transcribe_realtime(audio_data)
                
                if user_input.strip() and len(user_input.strip()) > 3:
                    # Process conversation
                    response = chatbot_instance.process_conversation(user_input)
                    
                    # Translate if needed
                    final_response = response
                    if data.get("translate", False):
                        final_response = translator.translate_and_transliterate(
                            response, 
                            target_lang_code=data.get("target_lang", "hi")
                        )
                    
                    # Generate and play audio
                    import uuid
                    audio_filename = f"output_{uuid.uuid4().hex}.mp3"
                    audio_file = tts.synthesize_speech(final_response, output_file=audio_filename)
                    
                    from components.SmartGlassesAudio import SmartGlassesAudio
                    glasses = SmartGlassesAudio()
                    glasses.play_audio_to_glasses(audio_file)
                    
                    # Clean up
                    try:
                        import os
                        os.remove(audio_file)
                    except:
                        pass
                    
                    await websocket.send_json({
                        "status": "success",
                        "user_message": user_input,
                        "bot_response": response
                    })
                else:
                    await websocket.send_json({
                        "status": "no_speech",
                        "message": "No speech detected"
                    })
            
            elif data.get("action") == "stop":
                await websocket.send_json({"status": "stopped"})
                break
                
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        await websocket.send_json({"error": str(e)})

@app.get("/api/chats")
async def get_all_chats(limit: int = 50, offset: int = 0):
    """Retrieve all chat history from vector database"""
    try:
        # Get all documents from ChromaDB
        results = vector_db.collection.get(
            limit=limit,
            offset=offset,
            include=["metadatas", "documents"]
        )
        
        chats = []
        if results['ids']:
            for i, chat_id in enumerate(results['ids']):
                metadata = results['metadatas'][i]
                chats.append({
                    "id": chat_id,
                    "user_message": metadata.get("user_message", ""),
                    "bot_response": metadata.get("bot_response", ""),
                    "timestamp": metadata.get("timestamp", ""),
                    "document": results['documents'][i]
                })
        
        return {
            "status": "success",
            "total": vector_db.collection.count(),
            "returned": len(chats),
            "chats": chats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chats: {str(e)}")

@app.get("/api/chats/{chat_id}")
async def get_chat_by_id(chat_id: str):
    """Retrieve specific chat by ID"""
    try:
        result = vector_db.collection.get(
            ids=[chat_id],
            include=["metadatas", "documents"]
        )
        
        if not result['ids']:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        metadata = result['metadatas'][0]
        return {
            "status": "success",
            "chat": {
                "id": result['ids'][0],
                "user_message": metadata.get("user_message", ""),
                "bot_response": metadata.get("bot_response", ""),
                "timestamp": metadata.get("timestamp", ""),
                "document": result['documents'][0]
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chat: {str(e)}")

@app.delete("/api/chats/delete")
async def delete_chats(request: DeleteChatRequest):
    """Delete specific chats by IDs"""
    try:
        # Verify all IDs exist
        existing = vector_db.collection.get(ids=request.chat_ids)
        
        if len(existing['ids']) == 0:
            raise HTTPException(status_code=404, detail="No matching chats found")
        
        # Delete from ChromaDB
        vector_db.collection.delete(ids=request.chat_ids)
        
        return {
            "status": "success",
            "message": f"Deleted {len(existing['ids'])} chat(s)",
            "deleted_ids": existing['ids']
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete chats: {str(e)}")

@app.delete("/api/chats/clear")
async def clear_all_chats():
    """Clear all chat history"""
    try:
        count = vector_db.collection.count()
        
        # Get all IDs and delete
        all_data = vector_db.collection.get()
        if all_data['ids']:
            vector_db.collection.delete(ids=all_data['ids'])
        
        return {
            "status": "success",
            "message": f"Cleared {count} chat(s)",
            "deleted_count": count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear chats: {str(e)}")

@app.get("/api/stats")
async def get_stats():
    """Get statistics about the chatbot and database"""
    try:
        return {
            "status": "success",
            "stats": {
                "total_conversations": vector_db.collection.count(),
                "chatbot_active": chatbot_instance is not None,
                "database_path": str(vector_db.client._settings.persist_directory) if hasattr(vector_db.client._settings, 'persist_directory') else "./chroma_db",
                "collection_name": vector_db.collection.name
            }
        }
    except Exception as e:
        print(f"Stats error: {e}")
        return {
            "status": "success",
            "stats": {
                "total_conversations": 0,
                "chatbot_active": chatbot_instance is not None,
                "database_path": "./chroma_db",
                "collection_name": "conversations"
            }
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)