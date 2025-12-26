"""
Step 1 Validation: Test camera capture with FPS counter.
Press 'q' to quit.
"""

import cv2
import yaml
import time
from src.io.camera import CameraCapture
from src.utils.fps_counter import FPSCounter


def main():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    camera_config = config['camera']
    
    # Initialize camera
    camera = CameraCapture(
        device_id=camera_config['device_id'],
        width=camera_config['width'],
        height=camera_config['height']
    )
    camera.start()
    
    # Initialize FPS counter
    fps_counter = FPSCounter(avg_over_frames=30)
    
    print("\n=== STEP 1: CAMERA TEST ===")
    print("Waiting for camera to warm up...")
    
    # Wait for first valid frame
    frame = None
    for i in range(50):  # Try for ~5 seconds
        frame = camera.read()
        if frame is not None:
            print("✓ Camera ready!")
            break
        time.sleep(0.1)
    
    if frame is None:
        print("[ERROR] Camera didn't produce frames after 5 seconds")
        camera.stop()
        return
    
    print("Press 'q' to quit\n")
    
    # Main loop
    while True:
        # Read frame
        frame = camera.read()
        
        if frame is None:
            print("[WARNING] Frame dropped, continuing...")
            continue
        
        # Update FPS
        fps_counter.update()
        fps = fps_counter.get_fps()
        
        # Draw FPS on frame
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )
        
        # Display
        cv2.imshow("SYNAPSE - Camera Test", frame)
        
        # Exit on 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    # Cleanup
    camera.stop()
    cv2.destroyAllWindows()
    
    print(f"\n[RESULT] Average FPS: {fps:.1f}")
    print("✓ Step 1 Complete" if fps >= 20 else "✗ FPS too low, check camera")


if __name__ == "__main__":
    main()
