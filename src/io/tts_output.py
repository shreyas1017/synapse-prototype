"""
Text-to-speech output module using pyttsx3.
"""

import pyttsx3
import threading
from typing import Optional


class TTSOutput:
    """Offline text-to-speech engine."""
    
    def __init__(self, rate: int = 150, volume: float = 0.9):
        """
        Initialize TTS engine.
        
        Args:
            rate: Speech rate in words per minute
            volume: Volume level (0.0 to 1.0)
        """
        self.rate = rate
        self.volume = volume
        
        # Thread lock to prevent simultaneous speech
        self.lock = threading.Lock()
        
        print(f"[TTS] Initialized: rate={rate}, volume={volume}")
    
    def speak(self, text: str, blocking: bool = True):
        """
        Speak text aloud.
        
        Args:
            text: Text to speak
            blocking: If True, wait for speech to complete. If False, speak in background.
        """
        if not text or len(text.strip()) == 0:
            return
        
        def _speak():
            with self.lock:
                # Reinitialize engine each time (fixes pyttsx3 one-time issue)
                engine = pyttsx3.init()
                engine.setProperty('rate', self.rate)
                engine.setProperty('volume', self.volume)
                
                print(f"[TTS] Speaking: {text}")
                engine.say(text)
                engine.runAndWait()
                engine.stop()
        
        if blocking:
            _speak()
        else:
            thread = threading.Thread(target=_speak, daemon=True)
            thread.start()
    
    def stop(self):
        """Stop current speech."""
        pass  # Not needed with per-call init
    
    def set_rate(self, rate: int):
        """Change speech rate."""
        self.rate = rate
    
    def set_volume(self, volume: float):
        """Change volume (0.0 to 1.0)."""
        self.volume = volume
