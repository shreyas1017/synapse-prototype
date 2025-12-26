"""
Simple FPS counter for performance monitoring.
"""

import time


class FPSCounter:
    """Calculates frames per second over a rolling window."""
    
    def __init__(self, avg_over_frames: int = 30):
        """
        Initialize FPS counter.
        
        Args:
            avg_over_frames: Number of frames to average FPS over
        """
        self.avg_over_frames = avg_over_frames
        self.frame_times = []
        self.fps = 0.0
    
    def update(self):
        """Call this once per frame to update FPS calculation."""
        current_time = time.time()
        self.frame_times.append(current_time)
        
        # Keep only recent frames
        if len(self.frame_times) > self.avg_over_frames:
            self.frame_times.pop(0)
        
        # Calculate FPS
        if len(self.frame_times) >= 2:
            elapsed = self.frame_times[-1] - self.frame_times[0]
            self.fps = (len(self.frame_times) - 1) / elapsed if elapsed > 0 else 0.0
    
    def get_fps(self) -> float:
        """Get current FPS value."""
        return self.fps
