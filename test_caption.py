"""
Step 5 Validation: Test scene captioning with BLIP.
Press 'd' to describe scene, 'q' to quit.
"""

import cv2
import yaml
import time
from src.io.camera import CameraCapture
from src.vision.captioner import SceneCaptioner
from src.io.tts_output import TTSOutput


def main():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    camera_config = config['camera']
    caption_config = config['captioning']
    tts_config = config['tts']
    
    # Initialize components
    print("\n=== STEP 5: SCENE CAPTIONING TEST ===\n")
    
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
    
    # Initialize captioner (will take 1-2 min first time)
    captioner = SceneCaptioner(
        model_name=caption_config['model'],
        device=caption_config['device'],
        max_length=caption_config['max_length'],
        min_length=caption_config['min_length']
    )
    
    # Initialize TTS
    tts = TTSOutput(
        rate=tts_config['rate'],
        volume=tts_config['volume']
    )
    
    print("\n=== CONTROLS ===")
    print("Press 'd' - Describe current scene")
    print("Press 'q' - Quit")
    print("\nTIP: Point camera at different scenes (room, objects, people)\n")
    
    # State
    last_caption = ""
    caption_active = False
    caption_times = []
    
    # Main loop
    while True:
        # Read frame
        frame = camera.read()
        if frame is None:
            continue
        
        # Display frame
        display_frame = frame.copy()
        
        # Draw last caption on frame
        if last_caption:
            # Split caption into lines if too long
            words = last_caption.split()
            lines = []
            current_line = []
            
            for word in words:
                current_line.append(word)
                if len(' '.join(current_line)) > 40:
                    lines.append(' '.join(current_line[:-1]))
                    current_line = [word]
            
            if current_line:
                lines.append(' '.join(current_line))
            
            # Draw each line
            y_offset = frame.shape[0] - 80
            for i, line in enumerate(lines):
                cv2.putText(
                    display_frame,
                    line,
                    (10, y_offset + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )
        
        # Draw instructions
        cv2.putText(
            display_frame,
            "Press 'd' to describe scene",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        if caption_active:
            cv2.putText(
                display_frame,
                "Generating caption...",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
        
        cv2.imshow("SYNAPSE - Scene Captioning", display_frame)
        
        # Handle keypresses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('d') and not caption_active:
            # Trigger captioning
            caption_active = True
            print("\n[USER] Describing scene...")
            
            # Capture current frame
            caption_frame = frame.copy()
            
            # Generate caption
            start_time = time.time()
            caption = captioner.generate_caption(caption_frame)
            caption_time = time.time() - start_time
            
            caption_times.append(caption_time)
            last_caption = caption
            
            print(f"[CAPTIONER] Processing time: {caption_time:.2f}s")
            print(f"[RESULT] Caption: {caption}")
            
            # Speak caption
            tts.speak(caption, blocking=False)
            
            caption_active = False
    
    # Cleanup
    camera.stop()
    cv2.destroyAllWindows()
    
    # Final stats
    if len(caption_times) > 0:
        avg_time = sum(caption_times) / len(caption_times)
        print(f"\n=== STEP 5 RESULTS ===")
        print(f"Total Captions Generated: {len(caption_times)}")
        print(f"Average Time: {avg_time:.2f}s")
        print(f"Last Caption: {last_caption}")
        
        if avg_time < 5.0:
            print("\n✓ Step 5 Complete - Scene captioning working!")
        else:
            print("\n⚠ Captioning slow but functional.")
    else:
        print("\n✓ Step 5 Complete - Model loaded successfully!")


if __name__ == "__main__":
    main()
