"""
YOLOv11-Nano detector wrapper for real-time object detection.
"""

from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Dict, Tuple


class YOLODetector:
    """Wrapper for YOLOv11 object detection."""
    
    def __init__(self, model_path: str = "yolo11n.pt", device: str = "cpu", 
                 confidence: float = 0.5, iou: float = 0.45):
        """
        Initialize YOLO detector.
        
        Args:
            model_path: Path to YOLO model weights
            device: 'cpu' or 'cuda'
            confidence: Confidence threshold (0-1)
            iou: IoU threshold for NMS (0-1)
        """
        self.model_path = model_path
        self.device = device
        self.confidence = confidence
        self.iou = iou
        
        print(f"[DETECTOR] Loading YOLOv11-Nano from {model_path}...")
        self.model = YOLO(model_path)
        
        # Move to device
        if device == "cuda":
            self.model.to("cuda")
        
        print(f"[DETECTOR] Model loaded on {device}")
        print(f"[DETECTOR] Confidence threshold: {confidence}, IoU: {iou}")
    
    def detect(self, frame: np.ndarray, verbose: bool = False) -> List[Dict]:
        """
        Run detection on a frame.
        
        Args:
            frame: Input image (BGR format from OpenCV)
            verbose: Print detection info
            
        Returns:
            List of detections, each containing:
                - bbox: [x1, y1, x2, y2]
                - confidence: float
                - class_id: int
                - class_name: str
        """
        # Run inference
        results = self.model.predict(
            frame,
            conf=self.confidence,
            iou=self.iou,
            verbose=False,
            device=self.device
        )
        
        # Parse results
        detections = []
        
        if len(results) > 0:
            result = results[0]  # Single image
            
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for box, conf, cls_id in zip(boxes, confidences, class_ids):
                    detection = {
                        'bbox': box.astype(int).tolist(),
                        'confidence': float(conf),
                        'class_id': int(cls_id),
                        'class_name': self.model.names[cls_id]
                    }
                    detections.append(detection)
        
        if verbose and len(detections) > 0:
            print(f"[DETECTOR] Found {len(detections)} objects")
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame.
        
        Args:
            frame: Input image
            detections: List of detections from detect()
            
        Returns:
            Frame with drawn boxes
        """
        annotated_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            label = det['class_name']
            
            # Draw box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label background
            label_text = f"{label} {conf:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - label_height - 10),
                (x1 + label_width, y1),
                (0, 255, 0),
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated_frame,
                label_text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
        
        return annotated_frame
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            'model': self.model_path,
            'device': self.device,
            'names': self.model.names,
            'num_classes': len(self.model.names)
        }
