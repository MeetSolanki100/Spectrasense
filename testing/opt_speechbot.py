import time
from datetime import datetime
import uuid
import os
import threading
from queue import Queue
from components.SpeechToText import SpeechToText
from components.LocalLLM import LocalLLM
from components.SmartGlassesAudio import SmartGlassesAudio
from components.TextToSpeech import TextToSpeech
from components.VectorDB import RAGChatbot

class SpeechChatbot:
    def __init__(self, 
                 whisper_model="base",
                 llm_model="mistral:7b",
                 glasses_device=None):
        
        self.stt = SpeechToText(whisper_model)
        self.llm = LocalLLM(llm_model)
        self.tts = TextToSpeech()
        self.glasses = SmartGlassesAudio(glasses_device)
        self.vectordb = RAGChatbot()
        self.is_listening = False
        
        # Pre-warm models
        self._warmup_models()
    
    def _warmup_models(self):
        """Warm up models to avoid first-call latency"""
        try:
            # Warm up Whisper
            import numpy as np
            dummy_audio = np.zeros(16000, dtype=np.float32)
            self.stt.transcribe_realtime(dummy_audio)
            
            # Warm up LLM
            self.llm.generate_response("Hi", max_tokens=5)
            
            print("Models warmed up successfully")
        except Exception as e:
            print(f"Warmup warning: {e}")
    
    def listen_continuously(self):
        """Continuous listening mode with parallel processing"""
        
        while True:
            try:
                print("speak now [lmao]")  
                audio_data = self.stt.record_audio(duration=5)
                
                # Use threading to process STT while potentially preparing for next input
                user_input = self.stt.transcribe_realtime(audio_data)
                
                if user_input.strip() and len(user_input.strip()) > 3:
                    print(f"User: {user_input}")
                    
                    # Process and respond
                    self._process_and_respond(user_input)
                
                time.sleep(0.5)  # Reduced from 2 seconds
                
            except KeyboardInterrupt:
                print("Stopping chatbot...")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
                continue
    
    def _process_and_respond(self, user_input):
        """Process conversation and respond with parallel TTS generation"""
        # Get context and generate response
        system_prompt = "You are a helpful voice assistant. Be concise and clear in your responses."
        prompt, contexts = self.vectordb.build_prompt_with_context(user_input, system_prompt)
        
        print("Prompt sent to LLM:")
        print(prompt)
        print("context given:")
        for ctx in contexts:
            print(f"- {ctx['content'][:50]}... (Distance: {ctx.get('distance', 'N/A')})")
        
        # Generate response
        response = self.llm.generate_response(prompt, max_tokens=100)
        print(f"Assistant: {response}")
        
        # Store in vector DB asynchronously (non-blocking)
        threading.Thread(
            target=self.vectordb.store_conversation,
            args=(user_input, response),
            daemon=True
        ).start()
        
        # Generate and play audio
        audio_filename = f"output_{uuid.uuid4().hex}.mp3"
        audio_file = self.tts.synthesize_speech(response, output_file=audio_filename)
        self.glasses.play_audio_to_glasses(audio_file)
        
        # Clean up asynchronously
        threading.Thread(
            target=self._cleanup_file,
            args=(audio_file,),
            daemon=True
        ).start()
    
    def _cleanup_file(self, filepath):
        """Async file cleanup"""
        try:
            time.sleep(0.5)  # Brief delay to ensure playback started
            os.remove(filepath)
        except Exception:
            pass
    
    def process_conversation(self, user_input):
        """Process user input and generate response using vector database for context"""
        system_prompt = "You are a helpful voice assistant. Be concise and clear in your responses."
        prompt, contexts = self.vectordb.build_prompt_with_context(user_input, system_prompt)
        
        print("Prompt sent to LLM:")
        print(prompt)
        print("context given:")
        for ctx in contexts:
            print(f"- {ctx['content']} (Distance: {ctx.get('distance', 'N/A')})")
        
        response = self.llm.generate_response(prompt, max_tokens=100)
        
        # Store asynchronously
        threading.Thread(
            target=self.vectordb.store_conversation,
            args=(user_input, response),
            daemon=True
        ).start()
        
        return response
    
    def single_interaction(self):
        """Single question-answer interaction"""
        print("Recording... Speak now!")
        
        audio_data = self.stt.record_audio(duration=5)
        user_input = self.stt.transcribe_realtime(audio_data)
        print(f"You said: {user_input}")
        
        if user_input.strip():
            response = self.process_conversation(user_input)
            print(f"Response: {response}")
            
            audio_filename = f"output_{uuid.uuid4().hex}.mp3"
            audio_file = self.tts.synthesize_speech(response, output_file=audio_filename)
            self.glasses.play_audio_to_glasses(audio_filename)
            
            threading.Thread(
                target=self._cleanup_file,
                args=(audio_file,),
                daemon=True
            ).start()
            
            return user_input, response
        
        return user_input, ""