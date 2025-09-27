import time
from datetime import datetime
import uuid
import os
from components.SpeechToText import SpeechToText
from components.LocalLLM import LocalLLM
from components.SmartGlassesAudio import SmartGlassesAudio
from components.TextToSpeech import TextToSpeech
class SpeechChatbot:
    def __init__(self, 
                 whisper_model="base",
                 llm_model="mistral:7b",
                 glasses_device=None):
        
        self.stt = SpeechToText(whisper_model)
        self.llm = LocalLLM(llm_model)
        self.tts = TextToSpeech()
        self.glasses = SmartGlassesAudio(glasses_device)
        
        self.conversation_history = []
        self.is_listening = False
    
    def listen_continuously(self):
        """Continuous listening mode"""
        print("Starting continuous listening mode...")
        
        while True:
            try:
                # Record audio
                audio_data = self.stt.record_audio(duration=3)
                
                # Convert to text
                user_input = self.stt.transcribe_realtime(audio_data)
                
                if user_input.strip() and len(user_input.strip()) > 3:
                    print(f"User: {user_input}")
                    # Generate LLM response
                    response = self.process_conversation(user_input)
                    print(f"Assistant: {response}")
                    # Use a unique filename for each response
                    audio_filename = f"output_{uuid.uuid4().hex}.wav"
                    print(audio_filename)
                    audio_file = self.tts.synthesize_speech(response, output_file=audio_filename)
                    self.glasses.play_audio_to_glasses(audio_file)
                    # Optionally, remove the file after playback
                    try:
                        os.remove(audio_file)
                    except Exception:
                        pass
                time.sleep(1.5)
                  # Brief pause
                
            except KeyboardInterrupt:
                print("Stopping chatbot...")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
                continue
    
    def process_conversation(self, user_input):
        """Process user input and generate response"""
        # Add context from conversation history
        context = ""
        if self.conversation_history:
            recent_history = self.conversation_history[-3:]  # Last 3 exchanges
            for exchange in recent_history:
                context += f"User: {exchange['user']}\nAssistant: {exchange['assistant']}\n"
        
        # Create prompt with context
        prompt = f"{context}User: {user_input}\nAssistant:"
        
        # Generate response
        response = self.llm.generate_response(prompt, max_tokens=100)
        
        # Store in conversation history
        self.conversation_history.append({
            "user": user_input,
            "assistant": response,
            "timestamp": datetime.now().isoformat()
        })
        
        return response
    
    def single_interaction(self):
        """Single question-answer interaction"""
        print("Recording... Speak now!")
        
        # Record audio
        audio_data = self.stt.record_audio(duration=5)
        
        # Convert to text
        user_input = self.stt.transcribe_realtime(audio_data)
        print(f"You said: {user_input}")
        
        if user_input.strip():
            # Generate response
            response = self.process_conversation(user_input)
            print(f"Response: {response}")
            # Use a unique filename for each response
            audio_filename = f"output_{uuid.uuid4().hex}.wav"
            audio_file = self.tts.synthesize_speech(response, output_file=audio_filename)
            self.glasses.play_audio_to_glasses(audio_file)
            try:
                os.remove(audio_file)
            except Exception:
                pass
        return user_input, response