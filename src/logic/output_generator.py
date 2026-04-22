"""
output_generator.py
Converts detection/tracking/scene results to natural language for TTS.
All string-building logic lives here. Nothing in main.py or command_processor.py
should construct speech strings directly.
"""

from typing import List, Dict, Optional


class OutputGenerator:
    """Converts system outputs to natural language descriptions."""

    def __init__(self):
        print("OUTPUT GENERATOR: Initialized")

    # ------------------------------------------------------------------
    # Detection / Tracking
    # ------------------------------------------------------------------

    def describe_detections(self, detections: List[Dict]) -> str:
        """Generate natural language from raw YOLO detections.

        Args:
            detections: List of dicts from YOLODetector.detect()
        Returns:
            Speech-ready string, e.g. "I see one person and two cars."
        """
        if len(detections) == 0:
            return "No objects detected."

        class_counts: Dict[str, int] = {}
        for det in detections:
            name = det["class_name"]
            class_counts[name] = class_counts.get(name, 0) + 1

        parts = []
        for name, count in class_counts.items():
            if count == 1:
                parts.append(f"one {name}")
            else:
                parts.append(f"{count} {name}s")

        if len(parts) == 1:
            return f"I see {parts[0]}."
        elif len(parts) == 2:
            return f"I see {parts[0]} and {parts[1]}."
        else:
            return f"I see {', '.join(parts[:-1])}, and {parts[-1]}."

    def describe_tracks(self, tracks: List[Dict]) -> str:
        """Generate natural language from tracked objects.

        Args:
            tracks: List of track dicts from ObjectTracker.update()
        Returns:
            Speech-ready string prioritising approaching objects.
        """
        if len(tracks) == 0:
            return "No objects being tracked."

        approaching = [t for t in tracks if "approaching" in t["direction"]]

        if approaching:
            warnings = []
            for track in approaching:
                name = track["class_name"]
                if "left" in track["direction"]:
                    warnings.append(f"{name} approaching from your left")
                else:
                    warnings.append(f"{name} approaching from your right")
            if len(warnings) == 1:
                return f"Caution! {warnings[0]}."
            else:
                return f"Caution! {', '.join(warnings)}."

        return self.describe_detections([{"class_name": t["class_name"]} for t in tracks])

    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------

    def format_ocr_result(self, text: str) -> str:
        """Format OCR result for speech.

        Args:
            text: Raw extracted text string (may be empty / None).
        Returns:
            Speech-ready string.
        """
        if not text or len(text.strip()) == 0:
            return "No text detected."
        return f"The text reads: {text.strip()}"

    # ------------------------------------------------------------------
    # Scene caption (BLIP)
    # ------------------------------------------------------------------

    def format_caption(self, caption: str) -> str:
        """Format BLIP scene caption for speech.

        Args:
            caption: Raw caption string from SceneCaptioner.
        Returns:
            Capitalised, period-terminated speech string.
        """
        if not caption or len(caption.strip()) == 0:
            return "Unable to describe the scene."
        formatted = caption[0].upper() + caption[1:]
        if not formatted.endswith("."):
            formatted += "."
        return formatted

    # ------------------------------------------------------------------
    # Scene memory changes  ← NEW
    # ------------------------------------------------------------------

    def format_scene_changes(
        self,
        changes: Dict,
        distance_estimator=None,
    ) -> str:
        """Build a speech string from a scene_memory.get_changes() result.

        The changes dict has four keys (all lists):
            - 'new'    : [(label, zone, dist), ...]
            - 'closer' : [(label, zone, dist), ...]
            - 'same'   : [(label, zone, dist), ...]
            - 'gone'   : [(label,), ...]   ← only label, no zone/dist

        Priority order for speech:
            1. New objects  (most important — user may not know)
            2. Closer objects
            3. Gone objects (path now clearer)
            4. Same objects (only if nothing else to report)

        Args:
            changes:            Dict returned by SceneMemory.get_changes()
            distance_estimator: Optional DistanceEstimator for formatting
                                distances. If None, distances are omitted.
        Returns:
            Speech-ready string.
        """
        # Helper: format a single (label, zone, dist) entry
        def _entry(label: str, zone: str, dist: Optional[float]) -> str:
            dist_str = (
                distance_estimator.format_distance(dist)
                if distance_estimator and dist is not None
                else None
            )
            zone_str = "ahead" if zone == "ahead" else f"to your {zone}"
            if dist_str:
                return f"{label} {zone_str}, {dist_str}"
            return f"{label} {zone_str}"

        parts: List[str] = []

        # 1. New objects
        for label, zone, dist in changes.get("new", []):
            parts.append(f"New {_entry(label, zone, dist)}")

        # 2. Closer objects
        for label, zone, dist in changes.get("closer", []):
            dist_str = (
                distance_estimator.format_distance(dist)
                if distance_estimator and dist is not None
                else None
            )
            zone_str = "ahead" if zone == "ahead" else f"to your {zone}"
            if dist_str:
                parts.append(f"{label} moving closer, now {dist_str} {zone_str}")
            else:
                parts.append(f"{label} moving closer {zone_str}")

        # 3. Gone objects (append after new/closer so path-clear info comes last)
        gone_labels = [g[0] if isinstance(g, (list, tuple)) else g
                       for g in changes.get("gone", [])]

        # 4. Same objects — only if nothing else to report
        if not parts:
            for label, zone, dist in changes.get("same", []):
                parts.append(_entry(label, zone, dist))

        # Append gone after everything else
        for label in gone_labels:
            parts.append(f"{label} no longer detected")

        if not parts:
            return "Path appears clear."

        return ". ".join(parts) + "."

    def format_scene_changes_empty(self, changes: Dict) -> str:
        """Speech string for the special case where tracks are empty.

        When the user presses W with no live tracks, we still check
        scene_memory for recently disappeared objects.

        Args:
            changes: Dict returned by SceneMemory.get_changes([], ...)
        Returns:
            Speech-ready string.
        """
        gone = changes.get("gone", [])
        if gone:
            gone_labels = [g[0] if isinstance(g, (list, tuple)) else g for g in gone]
            label_str = ", ".join(gone_labels)
            return f"Path now clear. {label_str} no longer detected."
        return "Path appears clear."