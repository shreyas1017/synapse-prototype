"""
Output generator - converts detection/tracking results to natural language.
"""

from typing import List, Dict


class OutputGenerator:
    """Converts system outputs to natural language descriptions."""
    
    def __init__(self):
        print("[OUTPUT GENERATOR] Initialized")
    
    def describe_detections(self, detections: List[Dict]) -> str:
        """
        Generate natural language from detections.
        
        Args:
            detections: List from YOLODetector.detect()
            
        Returns:
            Natural language description
        """
        if len(detections) == 0:
            return "No objects detected"
        
        # Count objects by class
        class_counts = {}
        for det in detections:
            class_name = det['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Build description
        parts = []
        for class_name, count in class_counts.items():
            if count == 1:
                parts.append(f"one {class_name}")
            else:
                parts.append(f"{count} {class_name}s")
        
        if len(parts) == 1:
            return f"I see {parts[0]}"
        elif len(parts) == 2:
            return f"I see {parts[0]} and {parts[1]}"
        else:
            return f"I see {', '.join(parts[:-1])}, and {parts[-1]}"
    
    def describe_tracks(self, tracks: List[Dict]) -> str:
        """
        Generate natural language from tracked objects.
        
        Args:
            tracks: List from ObjectTracker.update()
            
        Returns:
            Natural language description with warnings
        """
        if len(tracks) == 0:
            return "No objects being tracked"
        
        # Check for approaching objects
        approaching = [t for t in tracks if "approaching" in t['direction']]
        
        if len(approaching) > 0:
            # Priority: warn about approaching objects
            warnings = []
            for track in approaching:
                class_name = track['class_name']
                direction = track['direction']
                
                if "left" in direction:
                    warnings.append(f"{class_name} approaching from your left")
                else:
                    warnings.append(f"{class_name} approaching from your right")
            
            if len(warnings) == 1:
                return f"Caution: {warnings[0]}"
            else:
                return f"Caution: {', '.join(warnings)}"
        
        # Otherwise, general description
        return self.describe_detections([{'class_name': t['class_name']} for t in tracks])
    
    def format_ocr_result(self, text: str) -> str:
        """
        Format OCR result for speech.
        
        Args:
            text: Extracted text
            
        Returns:
            Speech-ready text
        """
        if not text or len(text.strip()) == 0:
            return "No text detected"
        
        return f"The text reads: {text}"
    
    def format_caption(self, caption: str) -> str:
        """
        Format scene caption for speech.
        
        Args:
            caption: Generated caption
            
        Returns:
            Speech-ready caption
        """
        if not caption or len(caption.strip()) == 0:
            return "Unable to describe the scene"
        
        # Capitalize first letter
        formatted = caption[0].upper() + caption[1:]
        
        # Ensure it ends with period
        if not formatted.endswith('.'):
            formatted += '.'
        
        return formatted
