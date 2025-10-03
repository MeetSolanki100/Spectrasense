from elevenlabs import stream
from elevenlabs.client import ElevenLabs
import dotenv
import os
from playsound import playsound
from pydub import AudioSegment
dotenv.load_dotenv()
client = ElevenLabs(  base_url="https://api.elevenlabs.io",
                        api_key=os.getenv("ELEVENLABS_API_KEY") )


audio = client.text_to_speech.convert(
    voice_id="JBFqnCBsd6RMkjVDRZzb",
    output_format="mp3_44100_128",
    text="The first move is what sets everything in motion.",
    model_id="eleven_multilingual_v2"
)

with open("output11.mp3", "wb") as f:
    for chunk in audio:
        if isinstance(chunk, bytes):
            f.write(chunk)
playsound("output11.mp3")
print("Played using playsound")
