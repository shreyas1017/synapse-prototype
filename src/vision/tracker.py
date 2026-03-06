"""
ByteTrack-based object tracker using Ultralytics built-in tracking.
Replaces SimpleTracker and DeepSORT for better ID stability.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict


class ByteTracker:
    """
    Wrapper around Ultralytics ByteTrack for stable multi-object tracking.
    Uses model.track() instead of model.predict() for built-in tracking.
    """

    def __init__(self, model_path: str = "yolo11n.pt", device: str = "cpu",
                 confidence: float = 0.35, iou: float = 0.4,
                 max_age: int = 30):
        """
        Initialize ByteTracker.

        Args:
            model_path: Path to YOLO model weights
            device: 'cpu' or 'cuda'
            confidence: Detection confidence threshold
            iou: IoU threshold for NMS
            max_age: Frames to keep track alive without detection
        """
        self.model_path = model_path
        self.device = device
        self.confidence = confidence
        self.iou = iou
        self.max_age = max_age
        self.direction_history = {}

        print(f"[BYTETRACK] Loading model: {model_path}")
        self.model = YOLO(model_path)

        # Track history for direction prediction
        self.track_history = {}  # track_id -> list of (frame_num, cx, cy)
        self.frame_count = 0

        print(f"[BYTETRACK] Initialized on {device}")

    def track(self, frame: np.ndarray) -> List[Dict]:
        """
        Run detection + tracking on a frame.

        Args:
            frame: Input image (BGR format from OpenCV)

        Returns:
            List of confirmed tracks, each containing:
                - track_id, bbox, center, class_name, confidence, direction
        """
        self.frame_count += 1

        # Run ByteTrack via Ultralytics
        results = self.model.track(
            frame,
            conf=self.confidence,
            iou=self.iou,
            tracker="bytetrack.yaml",
            persist=True,        # Crucial: maintains track IDs across frames
            verbose=False,
            device=self.device
        )

        confirmed_tracks = []

        if results and results[0].boxes is not None:
            boxes = results[0].boxes

            # Only process if tracking IDs exist
            if boxes.id is not None:
                for i in range(len(boxes)):
                    # Extract data
                    track_id = int(boxes.id[i].item())
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                    conf = float(boxes.conf[i].item())
                    cls_id = int(boxes.cls[i].item())
                    class_name = self.model.names[cls_id]

                    # Calculate center
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    # Update history
                    if track_id not in self.track_history:
                        self.track_history[track_id] = []

                    self.track_history[track_id].append(
                        (self.frame_count, cx, cy)
                    )

                    # Keep last 15 positions
                    if len(self.track_history[track_id]) > 15:
                        self.track_history[track_id].pop(0)

                    # Predict direction
                    direction = self._predict_direction(
                        track_id, cx, frame.shape[1]
                    )
                    if track_id not in self.direction_history:
                        self.direction_history[track_id] = []
                    self.direction_history[track_id].append(direction)
                    if len(self.direction_history[track_id]) > 8:
                        self.direction_history[track_id].pop(0)
                        
                    stable_direction = self._get_stable_direction(track_id, direction)

                    confirmed_tracks.append({
                        'track_id': track_id,
                        'bbox': [x1, y1, x2, y2],
                        'center': (cx, cy),
                        'class_name': class_name,
                        'confidence': conf,
                        'direction': stable_direction
                    })

        # Cleanup old tracks
        self._cleanup_old_tracks(confirmed_tracks)

        return confirmed_tracks

    def _get_stable_direction(self, track_id: int, current_direction: str) -> str:
        """
        Return direction only if it has been consistent for majority of recent frames.
        Prevents single-frame direction flips from triggering warnings.
        """
        history = self.direction_history.get(track_id, [])

        if len(history) < 5:
            return current_direction

        # Count occurrences of each direction in recent history
        from collections import Counter
        counts = Counter(history)
        most_common, count = counts.most_common(1)[0]

        # Only return a direction if it dominates (>60% of recent frames)
        if count / len(history) >= 0.6:
            return most_common
        else:
            return "stationary"  # Treat inconsistent movement as stationary


    def _predict_direction(self, track_id: int, current_x: int,
                           frame_width: int) -> str:
        """
        Predict movement direction from track history.

        Returns:
            One of: 'approaching from left', 'approaching from right',
                    'moving away left', 'moving away right',
                    'stationary', 'tracking'
        """
        history = self.track_history.get(track_id, [])

        if len(history) < 5:  # Need at least 5 frames for reliable prediction
            return "tracking"

        # Use first and last position in history window
        _, old_x, _ = history[0]
        _, new_x, _ = history[-1]

        dx = new_x - old_x
        frame_center = frame_width / 2
        threshold = 15  # Minimum pixel movement to count as motion

        if abs(dx) < threshold:
            return "stationary"
        elif dx > 0:  # Moving right
            return "approaching from left" if current_x < frame_center else "moving away right"
        else:  # Moving left
            return "approaching from right" if current_x > frame_center else "moving away left"

    def _cleanup_old_tracks(self, current_tracks):
        active_ids = {t['track_id'] for t in current_tracks}
        stale_ids = [
            tid for tid in self.track_history
            if tid not in active_ids
        ]
        for tid in stale_ids:
            history = self.track_history[tid]
            if self.frame_count - history[-1][0] > self.max_age:
                del self.track_history[tid]
                # Also clean direction history
                self.direction_history.pop(tid, None)


    def get_track_count(self) -> int:
        """Get number of active tracks."""
        return len(self.track_history)

    def reset(self):
        """Reset tracker state."""
        self.track_history = {}
        self.frame_count = 0
        print("[BYTETRACK] Reset")
