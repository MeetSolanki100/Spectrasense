import whisper
import pyaudio
import wave
import numpy as np

class SpeechToText:
    def __init__(self, model_size="base"):
        # Load Whisper model (base, small, medium, large)
        self.model = whisper.load_model(model_size)
    
    def record_audio(self, duration=5, sample_rate=16000):
        """Record audio from microphone"""
        p = pyaudio.PyAudio()
        
        stream = p.open(format=pyaudio.paInt16,
                       channels=1,
                       rate=sample_rate,
                       input=True,
                       frames_per_buffer=1024)
        
        frames = []
        for _ in range(0, int(sample_rate / 1024 * duration)):
            data = stream.read(1024)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Convert to numpy array
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        return audio_data.astype(np.float32) / 32768.0
    
    def transcribe(self, audio_file_path):
        """Convert audio to text"""
        result = self.model.transcribe(audio_file_path)
        return result["text"]
    
    def transcribe_realtime(self, audio_data):
        """Convert audio array to text"""
        result = self.model.transcribe(audio_data)
        return result["text"]