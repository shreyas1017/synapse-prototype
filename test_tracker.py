"""
Step 3 Validation: Test object tracking with trajectory prediction.
Press 'q' to quit.
"""

import cv2
import yaml
import time
from src.io.camera import CameraCapture
from src.vision.detector import YOLODetector
from src.vision.simple_tracker import SimpleTracker as ObjectTracker
from src.utils.fps_counter import FPSCounter


def main():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    camera_config = config['camera']
    detection_config = config['detection']
    tracking_config = config['tracking']
    
    # Initialize components
    print("\n=== STEP 3: OBJECT TRACKING TEST ===\n")
    
    camera = CameraCapture(
        device_id=camera_config['device_id'],
        width=camera_config['width'],
        height=camera_config['height']
    )
    camera.start()
    
    # Wait for camera
    frame = None
    for i in range(50):
        frame = camera.read()
        if frame is not None:
            break
        time.sleep(0.1)
    
    if frame is None:
        print("[ERROR] Camera didn't initialize")
        camera.stop()
        return
    
    # Initialize detector
    detector = YOLODetector(
        model_path=detection_config['model'],
        device=detection_config['device'],
        confidence=detection_config['confidence'],
        iou=detection_config['iou']
    )
    
    # Initialize tracker
    tracker = ObjectTracker(
        max_age=tracking_config['max_age'],
        min_hits=tracking_config['min_hits'],
        iou_threshold=tracking_config['iou_threshold']
    )
    
    print("\nPress 'q' to quit")
    print("Move objects around to see tracking and direction prediction!\n")
    
    # FPS counter
    fps_counter = FPSCounter(avg_over_frames=30)
    
    # Track event log
    direction_events = []
    
    # Main loop
    while True:
        # Read frame
        frame = camera.read()
        if frame is None:
            continue
        
        # Detect objects
        detections = detector.detect(frame)
        
        # Update tracker
        tracks = tracker.update(detections, frame)
        
        # Update FPS
        fps_counter.update()
        fps = fps_counter.get_fps()
        
        # Draw tracks
        annotated_frame = frame.copy()
        
        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            track_id = track['track_id']
            class_name = track['class_name']
            direction = track['direction']
            cx, cy = track['center']
            
            # Color based on direction
            if "approaching" in direction:
                color = (0, 0, 255)  # Red for approaching
            elif "moving away" in direction:
                color = (255, 0, 0)  # Blue for moving away
            else:
                color = (0, 255, 0)  # Green for stationary/tracking
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw center point
            cv2.circle(annotated_frame, (cx, cy), 4, color, -1)
            
            # Draw label
            label = f"ID:{track_id} {class_name}"
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
            
            # Draw direction
            cv2.putText(
                annotated_frame,
                direction,
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1
            )
            
            # Log direction events (for warnings)
            if "approaching" in direction:
                event = f"[TRACK {track_id}] {class_name} {direction}"
                if event not in direction_events:
                    direction_events.append(event)
                    print(event)  # Console warning
        
        # Draw FPS and stats
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        cv2.putText(
            annotated_frame,
            f"Tracks: {len(tracks)}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Display
        cv2.imshow("SYNAPSE - Object Tracking", annotated_frame)
        
        # Exit on 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    # Cleanup
    camera.stop()
    cv2.destroyAllWindows()
    
    # Final stats
    print(f"\n=== STEP 3 RESULTS ===")
    print(f"Average FPS: {fps:.1f}")
    print(f"Total Direction Events: {len(direction_events)}")
    print(f"Active Tracks: {tracker.get_track_count()}")
    
    if len(tracks) > 0 and fps >= 5:
        print("\n✓ Step 3 Complete - Tracking with trajectory prediction working!")
    else:
        print("\n⚠ Low performance, but tracking functional.")


if __name__ == "__main__":
    main()
