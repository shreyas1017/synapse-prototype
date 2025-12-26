"""
Step 4 Validation: Test OCR text extraction with TTS output (no GUI).
Type 'r' and press Enter to read text, 'q' to quit.
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
    print("\n=== STEP 4: OCR + TTS TEST (No GUI) ===\n")
    
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
    
    # Initialize OCR
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
    print("Type 'r' and press Enter - Read text from current frame")
    print("Type 'q' and press Enter - Quit")
    print("\nTIP: Point camera at printed text (signs, labels, books)\n")
    
    ocr_count = 0
    
    # Main loop
    while True:
        # Get user input
        user_input = input("\nCommand (r/q): ").strip().lower()
        
        if user_input == 'q':
            break
        elif user_input == 'r':
            # Capture current frame
            frame = camera.read()
            if frame is None:
                print("[ERROR] No frame available")
                continue
            
            ocr_count += 1
            
            # Save frame
            filename = f"ocr_capture_{ocr_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"[SAVED] Frame saved as {filename}")
            
            print("\n[USER] Reading text...")
            
            # Run OCR
            start_time = time.time()
            combined_text, detections = ocr.extract_text(frame)
            ocr_time = time.time() - start_time
            
            # Draw boxes and save
            if len(detections) > 0:
                annotated = ocr.draw_text_boxes(frame, detections)
                annotated_filename = f"ocr_result_{ocr_count}.jpg"
                cv2.imwrite(annotated_filename, annotated)
                print(f"[SAVED] Result saved as {annotated_filename}")
            
            print(f"[OCR] Processing time: {ocr_time:.2f}s")
            print(f"[OCR] Found {len(detections)} text regions")
            
            # Speak result
            if combined_text:
                print(f"[RESULT] Text: {combined_text}")
                tts.speak(combined_text, blocking=True)
            else:
                print("[RESULT] No text detected")
                tts.speak("No text detected", blocking=True)
    
    # Cleanup
    camera.stop()
    
    print(f"\n✓ Step 4 Complete - OCR + TTS working!")
    print(f"✓ Total OCR operations: {ocr_count}")


if __name__ == "__main__":
    main()
