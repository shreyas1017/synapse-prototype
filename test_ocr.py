"""
Step 4 Validation: Test OCR text extraction with TTS output.
Press 'r' to read text, 'q' to quit.
"""

import cv2
import yaml
import time
from src.io.camera import CameraCapture
from src.vision.ocr import OCRModule
from src.io.tts_output import TTSOutput


def main():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    camera_config = config['camera']
    ocr_config = config['ocr']
    tts_config = config['tts']
    
    # Initialize components
    print("\n=== STEP 4: OCR + TTS TEST ===\n")
    
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
    
    # Initialize OCR (will take 1-2 min first time)
    ocr = OCRModule(
        languages=ocr_config['languages'],
        gpu=ocr_config['gpu'],
        min_confidence=ocr_config['min_confidence']
    )
    
    # Initialize TTS
    tts = TTSOutput(
        rate=tts_config['rate'],
        volume=tts_config['volume']
    )
    
    print("\n=== CONTROLS ===")
    print("Press 'r' - Read text from current frame")
    print("Press 'q' - Quit")
    print("\nTIP: Point camera at printed text (signs, labels, books)\n")
    
    # State
    last_detections = []
    ocr_active = False
    
    # Main loop
    while True:
        # Read frame
        frame = camera.read()
        if frame is None:
            continue
        
        # Display frame (with last OCR results if any)
        display_frame = frame.copy()
        
        if len(last_detections) > 0:
            display_frame = ocr.draw_text_boxes(display_frame, last_detections)
        
        # Draw instructions
        cv2.putText(
            display_frame,
            "Press 'r' to read text",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        if ocr_active:
            cv2.putText(
                display_frame,
                "Processing OCR...",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
        
        cv2.imshow("SYNAPSE - OCR Test", display_frame)
        
        # Handle keypresses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r') and not ocr_active:
            # Trigger OCR
            ocr_active = True
            print("\n[USER] Reading text...")
            
            # Capture current frame
            ocr_frame = frame.copy()
            
            # Run OCR
            start_time = time.time()
            combined_text, detections = ocr.extract_text(ocr_frame)
            ocr_time = time.time() - start_time
            
            last_detections = detections
            
            print(f"[OCR] Processing time: {ocr_time:.2f}s")
            
            # Speak result
            if combined_text:
                print(f"[RESULT] Text: {combined_text}")
                tts.speak(combined_text, blocking=False)
            else:
                print("[RESULT] No text detected")
                tts.speak("No text detected", blocking=False)
            
            ocr_active = False
    
    # Cleanup
    camera.stop()
    cv2.destroyAllWindows()
    
    print("\n✓ Step 4 Complete - OCR + TTS working!")


if __name__ == "__main__":
    main()
