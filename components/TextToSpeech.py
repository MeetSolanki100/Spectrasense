import pyttsx3
from elevenlabs import stream
from elevenlabs.client import ElevenLabs
import dotenv
import os
dotenv.load_dotenv()

class TextToSpeech:
    def __init__(self, lang="en"):
        self.lang = lang

    def synthesize_speech(self, response, output_file="output.mp3"):
        """Convert text to speech and save as WAV file"""
        # engine = pyttsx3.init()
        # # Set language/voice again for the new engine
        # voices = engine.getProperty("voices")
        # for voice in voices:
        #     if self.lang in voice.languages[0].lower():
        #         engine.setProperty("voice", voice.id)
        #         break
        # engine.save_to_file(text, output_file)
        # engine.runAndWait()
        # engine.stop()
        # return output_file
        try:
                from gtts import gTTS
                tts = gTTS(text=response, slow=False)
                tts.save(output_file)
                # os.system("start output.mp3") # For playing on Windows
                return output_file
        except Exception as e:
                print(f"Google TTS error: {e}")
                