import pyaudio
import wave
from playsound import playsound
class SmartGlassesAudio:
    def __init__(self, device_name=None):
        self.p = pyaudio.PyAudio()
        self.device_index = self._find_device(device_name)
    
    def _find_device(self, device_name):
        """Find audio device by name"""
        if device_name is None:
            return None
        
        for i in range(self.p.get_device_count()):
            info = self.p.get_device_info_by_index(i)
            if device_name.lower() in info['name'].lower():
                return i
        return None
    
    def play_audio_to_glasses(self, wav_file):
        """Play audio file to smart glasses"""
        playsound(wav_file)
        # wf = wave.open(wav_file, 'rb')
        
        # stream = self.p.open(
        #     format=self.p.get_format_from_width(wf.getsampwidth()),
        #     channels=wf.getnchannels(),
        #     rate=wf.getframerate(),
        #     output=True,
        #     output_device_index=self.device_index
        # )
        
        # chunk = 1024
        # data = wf.readframes(chunk)
        
        # while data:
        #     stream.write(data)
        #     data = wf.readframes(chunk)
        
        # stream.stop_stream()
        # stream.close()
        # wf.close()