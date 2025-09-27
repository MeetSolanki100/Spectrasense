import pyttsx3


class TextToSpeech:
    def __init__(self, lang="en"):
        self.lang = lang

    def synthesize_speech(self, text, output_file="output.wav"):
        """Convert text to speech and save as WAV file"""
        engine = pyttsx3.init()
        # Set language/voice again for the new engine
        voices = engine.getProperty("voices")
        for voice in voices:
            if self.lang in voice.languages[0].lower():
                engine.setProperty("voice", voice.id)
                break
        engine.save_to_file(text, output_file)
        engine.runAndWait()
        engine.stop()
        return output_file

