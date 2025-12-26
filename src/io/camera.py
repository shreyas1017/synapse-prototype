"""
Threaded camera capture module for real-time video processing.
Prevents blocking by running frame capture in a separate thread.
"""

import cv2
import threading
import time
from typing import Optional, Tuple


class CameraCapture:
    """
    Threaded camera capture class that continuously reads frames in background.
    Ensures main thread can process frames without waiting for camera I/O.
    """
    
    def __init__(self, device_id: int = 0, width: int = 640, height: int = 480):
        """
        Initialize camera capture.
        
        Args:
            device_id: Camera device index (0 for default webcam)
            width: Frame width in pixels
            height: Frame height in pixels
        """
        self.device_id = device_id
        self.width = width
        self.height = height
        
        # Initialize camera
        self.cap = cv2.VideoCapture(self.device_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # Check if camera opened successfully
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera with device_id={device_id}")
        
        # Frame storage
        self.frame = None
        self.grabbed = False
        
        # Thread control
        self.stopped = False
        self.thread = None
        
        print(f"[CAMERA] Initialized: Device {device_id}, Resolution {width}x{height}")
    
    def start(self):
        """Start the background thread for frame capture."""
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        
        # Wait a moment for first frame
        time.sleep(0.5)
        
        print("[CAMERA] Background capture thread started")
        return self
    
    def _update(self):
        """
        Internal method that runs in background thread.
        Continuously reads frames from camera.
        """
        while not self.stopped:
            self.grabbed, self.frame = self.cap.read()
            
            if not self.grabbed:
                print("[CAMERA WARNING] Failed to grab frame")
                self.stop()
                break
    
    def read(self) -> Optional[cv2.Mat]:
        """
        Get the most recent frame.
        
        Returns:
            Latest frame as numpy array, or None if not available
        """
        return self.frame
    
    def stop(self):
        """Stop the capture thread and release camera."""
        self.stopped = True
        
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        
        if self.cap is not None:
            self.cap.release()
        
        print("[CAMERA] Stopped and released")
    
    def get_resolution(self) -> Tuple[int, int]:
        """Get actual camera resolution."""
        return (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
    
    def get_fps(self) -> float:
        """Get camera's reported FPS (not actual processing FPS)."""
        return self.cap.get(cv2.CAP_PROP_FPS)
