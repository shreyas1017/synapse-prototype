"""
Step 2 Validation: Test YOLOv11 object detection on live camera.
Press 'q' to quit.
"""

import cv2
import yaml
import time
from src.io.camera import CameraCapture
from src.vision.detector import YOLODetector
from src.utils.fps_counter import FPSCounter


def main():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    camera_config = config['camera']
    detection_config = config['detection']
    
    # Initialize camera
    print("\n=== STEP 2: OBJECT DETECTION TEST ===\n")
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
    
    # Print model info
    info = detector.get_model_info()
    print(f"\n[INFO] Loaded {info['num_classes']} object classes")
    print(f"[INFO] Device: {info['device']}")
    print("\nPress 'q' to quit\n")
    
    # FPS counters
    fps_counter = FPSCounter(avg_over_frames=30)
    detection_times = []
    
    # Main loop
    while True:
        # Read frame
        frame = camera.read()
        if frame is None:
            continue
        
        # Run detection (measure time)
        start_time = time.time()
        detections = detector.detect(frame)
        detection_time = time.time() - start_time
        detection_times.append(detection_time)
        
        # Keep last 30 measurements
        if len(detection_times) > 30:
            detection_times.pop(0)
        
        # Draw detections
        annotated_frame = detector.draw_detections(frame, detections)
        
        # Update FPS
        fps_counter.update()
        fps = fps_counter.get_fps()
        
        # Calculate average detection time
        avg_detection_ms = (sum(detection_times) / len(detection_times)) * 1000
        
        # Draw stats
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
            f"Detection: {avg_detection_ms:.0f}ms",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        cv2.putText(
            annotated_frame,
            f"Objects: {len(detections)}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Display
        cv2.imshow("SYNAPSE - Object Detection", annotated_frame)
        
        # Exit on 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    # Cleanup
    camera.stop()
    cv2.destroyAllWindows()
    
    # Final stats
    print(f"\n=== STEP 2 RESULTS ===")
    print(f"Average FPS: {fps:.1f}")
    print(f"Average Detection Time: {avg_detection_ms:.0f}ms")
    print(f"Total Objects Detected: {len(detections)} (last frame)")
    
    if fps >= 10:
        print("\n✓ Step 2 Complete - Detection working!")
    else:
        print("\n⚠ FPS low but functional. Consider lowering resolution.")


if __name__ == "__main__":
    main()
