"""
Text-to-speech output module using pyttsx3.
"""
import pyttsx3
import threading
from typing import Optional


class TTSOutput:
    """Offline text-to-speech engine."""

    def __init__(self, rate: int = 150, volume: float = 0.9):
        self.rate = rate
        self.volume = volume
        self.lock = threading.Lock()
        self._engine = None
        self._init_engine()
        print(f"[TTS] Initialized: rate={rate}, volume={volume}")

    def _init_engine(self):
        try:
            self._engine = pyttsx3.init()
            self._engine.setProperty('rate', self.rate)
            self._engine.setProperty('volume', self.volume)
        except Exception as e:
            print(f"[TTS] Engine init failed: {e}")
            self._engine = None

    def speak(self, text: str, blocking: bool = True):
        if not text or not text.strip():
            return

        def _speak():
            with self.lock:
                try:
                    if self._engine is None:
                        self._init_engine()
                    print(f"[TTS] Speaking: {text}")
                    self._engine.say(text)
                    self._engine.runAndWait()
                except Exception:
                    # Engine crashed — reinit for next call
                    self._init_engine()

        if blocking:
            _speak()
        else:
            threading.Thread(target=_speak, daemon=True).start()

    def set_rate(self, rate: int):
        self.rate = rate
        if self._engine:
            self._engine.setProperty('rate', rate)

    def set_volume(self, volume: float):
        self.volume = volume
        if self._engine:
            self._engine.setProperty('volume', volume)

    def stop(self):
        if self._engine:
            try:
                self._engine.stop()
            except Exception:
                pass
