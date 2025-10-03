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
        client = ElevenLabs(  base_url="https://api.elevenlabs.io",
                        api_key=os.getenv("ELEVENLABS_API_KEY") )


        audio = client.text_to_speech.convert(
            voice_id="JBFqnCBsd6RMkjVDRZzb",
            output_format="mp3_44100_128",
            text=response,
            model_id="eleven_multilingual_v2"
        )

        with open(output_file, "wb") as f:
            for chunk in audio:
                if isinstance(chunk, bytes):
                    f.write(chunk)
        return output_file

