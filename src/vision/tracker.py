"""
SORT-based object tracker with trajectory prediction.
"""

from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
from typing import List, Dict, Tuple, Optional


class ObjectTracker:
    """Wrapper for DeepSORT tracking with trajectory analysis."""
    
    def __init__(self, max_age: int = 5, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        Initialize tracker.
        
        Args:
            max_age: Frames to keep track alive without detection
            min_hits: Minimum detections before track is confirmed
            iou_threshold: IoU threshold for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        # Initialize DeepSORT
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=min_hits,
            nms_max_overlap=1.0,
            max_cosine_distance=0.3,
            nn_budget=None,
            override_track_class=None,
            embedder="mobilenet",
            half=False,
            bgr=True,
            embedder_gpu=False
        )
        
        # Track history for trajectory prediction
        self.track_history = {}  # track_id -> list of (frame_num, x_center, y_center)
        self.frame_count = 0
        
        print(f"[TRACKER] Initialized with max_age={max_age}, min_hits={min_hits}")
    
    def update(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detections from YOLODetector.detect()
            frame: Current frame (for embedding extraction)
            
        Returns:
            List of tracks with IDs and trajectory info
        """
        self.frame_count += 1
        
        # Convert detections to DeepSORT format
        # Format: [[x1, y1, x2, y2, confidence, class_id], ...]
        raw_detections = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            cls = det['class_id']
            raw_detections.append(([x1, y1, x2, y2], conf, cls))
        
        # Update tracker
        tracks = self.tracker.update_tracks(raw_detections, frame=frame)
        
        # Process tracks
        confirmed_tracks = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb()  # [left, top, right, bottom]
            
            # Calculate center
            x_center = int((ltrb[0] + ltrb[2]) / 2)
            y_center = int((ltrb[1] + ltrb[3]) / 2)
            
            # Update history
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            
            self.track_history[track_id].append((self.frame_count, x_center, y_center))
            
            # Keep only last 10 positions
            if len(self.track_history[track_id]) > 10:
                self.track_history[track_id].pop(0)
            
            # Predict direction
            direction = self._predict_direction(track_id, x_center, frame.shape[1])
            
            # Get class name from original detection
            class_name = "unknown"
            for det in detections:
                det_center_x = (det['bbox'][0] + det['bbox'][2]) / 2
                det_center_y = (det['bbox'][1] + det['bbox'][3]) / 2
                
                # Match by center proximity
                if abs(det_center_x - x_center) < 50 and abs(det_center_y - y_center) < 50:
                    class_name = det['class_name']
                    break
            
            confirmed_tracks.append({
                'track_id': track_id,
                'bbox': [int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])],
                'center': (x_center, y_center),
                'class_name': class_name,
                'direction': direction,
                'age': len(self.track_history[track_id])
            })
        
        return confirmed_tracks
    
    def _predict_direction(self, track_id: int, current_x: int, frame_width: int) -> str:
        """
        Predict movement direction based on track history.
        
        Args:
            track_id: Track ID
            current_x: Current x-center position
            frame_width: Frame width for relative position
            
        Returns:
            Direction string: "approaching from left", "approaching from right", 
                             "moving away left", "moving away right", "stationary"
        """
        if track_id not in self.track_history or len(self.track_history[track_id]) < 3:
            return "tracking"
        
        history = self.track_history[track_id]
        
        # Get positions from 3 frames ago and current
        old_frame, old_x, old_y = history[0]
        new_frame, new_x, new_y = history[-1]
        
        # Calculate displacement
        dx = new_x - old_x
        frame_center = frame_width / 2
        
        # Determine relative position
        position = "left" if current_x < frame_center else "right"
        
        # Movement threshold (in pixels)
        movement_threshold = 20
        
        if abs(dx) < movement_threshold:
            return "stationary"
        elif dx > 0:  # Moving right
            if current_x < frame_center:
                return f"approaching from left"
            else:
                return f"moving away right"
        else:  # Moving left
            if current_x > frame_center:
                return f"approaching from right"
            else:
                return f"moving away left"
    
    def get_track_count(self) -> int:
        """Get number of active tracks."""
        return len(self.track_history)
    
    def reset(self):
        """Reset tracker state."""
        self.track_history = {}
        self.frame_count = 0
        print("[TRACKER] Reset")
