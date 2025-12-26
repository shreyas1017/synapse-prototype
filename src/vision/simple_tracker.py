"""
Simplified object tracker using IoU matching (lighter than DeepSORT).
"""

import numpy as np
from typing import List, Dict


class SimpleTracker:
    """Basic IoU-based tracker for faster performance."""
    
    def __init__(self, max_age: int = 15, min_hits: int = 1, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.next_id = 1
        self.tracks = {}  # track_id -> {bbox, class_name, age, hits, history}
        self.frame_count = 0
        
        print(f"[SIMPLE TRACKER] Initialized")
    
    def update(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """Update with new detections."""
        self.frame_count += 1
        
        # Match detections to existing tracks
        matched_tracks = set()
        matched_detections = set()
        
        for track_id, track_data in list(self.tracks.items()):
            best_iou = 0
            best_det_idx = None
            
            for i, det in enumerate(detections):
                if i in matched_detections:
                    continue
                
                iou = self._calculate_iou(track_data['bbox'], det['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = i
            
            if best_det_idx is not None:
                # Update track
                det = detections[best_det_idx]
                track_data['bbox'] = det['bbox']
                track_data['class_name'] = det['class_name']
                track_data['age'] = 0
                track_data['hits'] += 1
                
                # Update history
                cx = (det['bbox'][0] + det['bbox'][2]) // 2
                cy = (det['bbox'][1] + det['bbox'][3]) // 2
                track_data['history'].append((self.frame_count, cx, cy))
                if len(track_data['history']) > 10:
                    track_data['history'].pop(0)
                
                matched_tracks.add(track_id)
                matched_detections.add(best_det_idx)
        
        # Create new tracks for unmatched detections
        for i, det in enumerate(detections):
            if i not in matched_detections:
                cx = (det['bbox'][0] + det['bbox'][2]) // 2
                cy = (det['bbox'][1] + det['bbox'][3]) // 2
                
                self.tracks[self.next_id] = {
                    'bbox': det['bbox'],
                    'class_name': det['class_name'],
                    'age': 0,
                    'hits': 1,
                    'history': [(self.frame_count, cx, cy)]
                }
                self.next_id += 1
        
        # Age unmatched tracks and remove old ones
        for track_id in list(self.tracks.keys()):
            if track_id not in matched_tracks:
                self.tracks[track_id]['age'] += 1
                if self.tracks[track_id]['age'] > self.max_age:
                    del self.tracks[track_id]
        
        # Return confirmed tracks
        confirmed = []
        for track_id, track_data in self.tracks.items():
            if track_data['hits'] >= self.min_hits:
                cx = (track_data['bbox'][0] + track_data['bbox'][2]) // 2
                cy = (track_data['bbox'][1] + track_data['bbox'][3]) // 2
                
                direction = self._predict_direction(track_data['history'], cx, frame.shape[1])
                
                confirmed.append({
                    'track_id': track_id,
                    'bbox': track_data['bbox'],
                    'center': (cx, cy),
                    'class_name': track_data['class_name'],
                    'direction': direction,
                    'age': track_data['hits']
                })
        
        return confirmed
    
    def _calculate_iou(self, box1, box2):
        """Calculate IoU."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _predict_direction(self, history, current_x, frame_width):
        """Predict direction from history."""
        if len(history) < 3:
            return "tracking"
        
        old_x = history[0][1]
        new_x = history[-1][1]
        dx = new_x - old_x
        
        frame_center = frame_width / 2
        threshold = 20
        
        if abs(dx) < threshold:
            return "stationary"
        elif dx > 0:
            return "approaching from left" if current_x < frame_center else "moving away right"
        else:
            return "approaching from right" if current_x > frame_center else "moving away left"
    
    def get_track_count(self):
        return len(self.tracks)
    
    def reset(self):
        self.tracks = {}
        self.next_id = 1
        self.frame_count = 0
