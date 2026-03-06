"""
Estimates real-world distance of detected objects using
known average object heights and camera focal length.

Formula: distance = (known_height * focal_length) / pixel_height
"""

from src.utils.logger import logger


# Known average real-world heights in metres
KNOWN_HEIGHTS = {
    "person":     1.70,
    "bicycle":    1.00,
    "car":        1.50,
    "motorcycle": 1.10,
    "bus":        3.20,
    "truck":      2.80,
    "chair":      0.90,
    "dining table": 0.75,
    "bottle":     0.25,
    "cup":        0.12,
    "laptop":     0.30,
    "dog":        0.50,
    "cat":        0.25,
}

# Focal length in pixels (calibrated for 640x480 webcam ~70° FOV)
# Formula: f = (frame_height / 2) / tan(FOV/2)
# For 480px height, ~60° vertical FOV → ~415px
FOCAL_LENGTH_PX = 415.0


class DistanceEstimator:
    def __init__(self, focal_length: float = FOCAL_LENGTH_PX):
        self.focal_length = focal_length
        logger.info("[DISTANCE] Estimator initialized")

    def estimate(self, class_name: str, bbox: list) -> float | None:
        """
        Estimate distance in metres for a detected object.

        Args:
            class_name: YOLO class name
            bbox: [x1, y1, x2, y2] in pixels

        Returns:
            Distance in metres, or None if class unknown
        """
        known_height = KNOWN_HEIGHTS.get(class_name)
        if known_height is None:
            return None

        x1, y1, x2, y2 = bbox
        pixel_height = y2 - y1

        if pixel_height <= 0:
            return None

        distance = (known_height * self.focal_length) / pixel_height

        # Clamp to reasonable range (0.3m to 15m)
        distance = max(0.3, min(15.0, distance))

        return round(distance, 1)

    def format_distance(self, distance: float | None) -> str:
        """
        Convert distance float to a natural spoken string.

        Examples:
            0.5  → "very close"
            1.2  → "about 1 metre away"
            3.4  → "about 3 metres away"
            8.0  → "far away"
        """
        if distance is None:
            return ""

        if distance < 0.8:
            return "very close"
        elif distance < 2.0:
            return f"about {distance:.0f} metre away"
        elif distance < 6.0:
            return f"about {distance:.0f} metres away"
        else:
            return "far away"
