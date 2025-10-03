import time
from datetime import datetime
import uuid
import os
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
    
    def listen_continuously(self):
        """Continuous listening mode"""
        
        while True:
            try:
                # Record audio
                print("speak now [lmao]")  
                audio_data = self.stt.record_audio(duration=5)
                
                # Convert to text
                user_input = self.stt.transcribe_realtime(audio_data)
                
                if user_input.strip() and len(user_input.strip()) > 3:
                    print(f"User: {user_input}")
                    # Generate LLM response
                    response = self.process_conversation(user_input)
                    print(f"Assistant: {response}")
                    # Use a unique filename for each response
                    audio_filename = f"output_{uuid.uuid4().hex}.mp3"
                    audio_file = self.tts.synthesize_speech(response, output_file=audio_filename)
                    self.glasses.play_audio_to_glasses(audio_file)
                    # Optionally, remove the file after playback
                    try:
                        os.remove(audio_file)
                    except Exception:
                        pass
                time.sleep(2)
                # Brief pause
                
            except KeyboardInterrupt:
                print("Stopping chatbot...")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
                continue
    
    def process_conversation(self, user_input):
        """Process user input and generate response using vector database for context"""
        # Get prompt with relevant context from vector DB
        system_prompt = "You are a helpful voice assistant. Be concise and clear in your responses."
        prompt, contexts = self.vectordb.build_prompt_with_context(user_input, system_prompt)
        print("Prompt sent to LLM:")
        print(prompt)
        # print()
        print("context given:")
        for ctx in contexts:
            print(f"- {ctx['content']} (Distance: {ctx.get('distance', 'N/A')})")
        # Generate response
        response = self.llm.generate_response(prompt, max_tokens=100)
        
        # Store conversation in vector DB
        self.vectordb.store_conversation(user_input, response)
        
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
            audio_filename = f"output_{uuid.uuid4().hex}.mp3"
            audio_file = self.tts.synthesize_speech(response, output_file=audio_filename)
            self.glasses.play_audio_to_glasses(audio_filename)
            try:
                os.remove(audio_file)
            except Exception:
                pass
        return user_input, response