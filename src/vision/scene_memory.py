"""
Short-term scene memory for SYNAPSE.
Remembers recent detections and detects what changed between scans.
"""

import time
from src.utils.logger import logger


class SceneMemory:
    def __init__(self, memory_duration: float = 5.0):
        """
        Args:
            memory_duration: How many seconds to remember objects (default 5s)
        """
        self.memory_duration = memory_duration
        self.last_snapshot = {}   # class_name -> {zone, distance, timestamp}
        self.last_scan_time = 0
        logger.info("[MEMORY] Scene memory initialized")

    def update(self, tracks: list, frame_width: int, estimator) -> None:
        """
        Save current tracks into memory snapshot.
        Called every time W is pressed.
        """
        from src.vision.spatial_layout import get_zone

        now = time.time()
        new_snapshot = {}

        for track in tracks:
            label = track['class_name']
            zone = get_zone(track['bbox'], frame_width)
            distance = estimator.estimate(label, track['bbox'])
            new_snapshot[label] = {
                "zone": zone,
                "distance": distance,
                "timestamp": now
            }

        self.last_snapshot = new_snapshot
        self.last_scan_time = now

    def get_changes(self, tracks: list, frame_width: int, estimator) -> dict:
        """
        Compare current tracks against last snapshot.

        Returns:
            dict with keys:
                new      → objects not seen before
                gone     → objects that disappeared
                closer   → objects that moved significantly closer
                same     → objects unchanged
        """
        from src.vision.spatial_layout import get_zone

        now = time.time()

        # Build current state
        current = {}
        for track in tracks:
            label = track['class_name']
            zone = get_zone(track['bbox'], frame_width)
            distance = estimator.estimate(label, track['bbox'])
            current[label] = {"zone": zone, "distance": distance}

        # Expire old memory entries
        valid_last = {
            k: v for k, v in self.last_snapshot.items()
            if now - v["timestamp"] < self.memory_duration
        }

        new    = []
        gone   = []
        closer = []
        same   = []

        # What's new or closer?
        for label, data in current.items():
            if label not in valid_last:
                new.append((label, data["zone"], data["distance"]))
            else:
                prev_dist = valid_last[label]["distance"]
                curr_dist = data["distance"]
                # "Significantly closer" = moved more than 1 metre closer
                if prev_dist and curr_dist and (prev_dist - curr_dist) > 1.0:
                    closer.append((label, data["zone"], curr_dist))
                else:
                    same.append((label, data["zone"], data["distance"]))

        # What's gone?
        for label in valid_last:
            if label not in current:
                gone.append((label, valid_last[label]["zone"]))

        return {"new": new, "gone": gone, "closer": closer, "same": same}
