"""
SYNAPSE: Real-Time Embedded Vision System for Assistive Navigation
Main orchestrator integrating all modules.

CONTROLS:
- 'w' : What's ahead? (Detection summary with audio)
- 'd' : Describe scene (BLIP captioning)
- 'r' : Read text (OCR)
- 't' : Toggle tracking warnings (continuous vs on-demand)
- 'q' : Quit

Author: Your Team
Date: December 2025
"""

import cv2
import yaml
import time
from src.io.camera import CameraCapture
from src.vision.detector import YOLODetector
from src.vision.simple_tracker import SimpleTracker as ObjectTracker
from src.vision.ocr import OCRModule
from src.vision.captioner import SceneCaptioner
from src.io.tts_output import TTSOutput
from src.logic.output_generator import OutputGenerator
from src.utils.fps_counter import FPSCounter


class SynapseSystem:
    """Main SYNAPSE system orchestrator."""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize all system components."""
        print("\n" + "="*60)
        print("PROJECT SYNAPSE - Assistive Vision System")
        print("="*60 + "\n")
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self._init_camera()
        self._init_detector()
        self._init_tracker()
        self._init_ocr()
        self._init_captioner()
        self._init_tts()
        self._init_output_generator()
        
        # System state
        self.tracking_warnings_enabled = True
        self.last_warning_time = 0
        self.warning_cooldown = 3.0  # seconds between warnings
        
        # Performance tracking
        self.fps_counter = FPSCounter(avg_over_frames=30)
        
        print("\n" + "="*60)
        print("SYNAPSE SYSTEM READY")
        print("="*60 + "\n")
    
    def _init_camera(self):
        """Initialize camera."""
        config = self.config['camera']
        self.camera = CameraCapture(
            device_id=config['device_id'],
            width=config['width'],
            height=config['height']
        )
        self.camera.start()
        
        # Wait for first frame
        for _ in range(50):
            if self.camera.read() is not None:
                break
            time.sleep(0.1)
    
    def _init_detector(self):
        """Initialize object detector."""
        config = self.config['detection']
        self.detector = YOLODetector(
            model_path=config['model'],
            device=config['device'],
            confidence=config['confidence'],
            iou=config['iou']
        )
    
    def _init_tracker(self):
        """Initialize object tracker."""
        config = self.config['tracking']
        self.tracker = ObjectTracker(
            max_age=config['max_age'],
            min_hits=config['min_hits'],
            iou_threshold=config['iou_threshold']
        )
    
    def _init_ocr(self):
        """Initialize OCR module."""
        config = self.config['ocr']
        self.ocr = OCRModule(
            languages=config['languages'],
            gpu=config['gpu'],
            min_confidence=config['min_confidence']
        )
    
    def _init_captioner(self):
        """Initialize scene captioner."""
        config = self.config['captioning']
        self.captioner = SceneCaptioner(
            model_name=config['model'],
            device=config['device'],
            max_length=config['max_length'],
            min_length=config['min_length']
        )
    
    def _init_tts(self):
        """Initialize text-to-speech."""
        config = self.config['tts']
        self.tts = TTSOutput(
            rate=config['rate'],
            volume=config['volume']
        )
    
    def _init_output_generator(self):
        """Initialize output generator."""
        self.output_gen = OutputGenerator()
    
    def draw_interface(self, frame, detections, tracks):
        """Draw UI overlay on frame."""
        display = frame.copy()
        
        # Draw tracks
        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            track_id = track['track_id']
            class_name = track['class_name']
            direction = track['direction']
            
            # Color based on direction
            if "approaching" in direction:
                color = (0, 0, 255)  # Red
            elif "moving away" in direction:
                color = (255, 0, 0)  # Blue
            else:
                color = (0, 255, 0)  # Green
            
            # Draw box
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            
            # Draw ID
            label = f"ID:{track_id} {class_name}"
            cv2.putText(display, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw FPS
        fps = self.fps_counter.get_fps()
        cv2.putText(display, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw object count
        cv2.putText(display, f"Objects: {len(tracks)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw controls
        y_offset = frame.shape[0] - 120
        controls = [
            "CONTROLS:",
            "W - What's ahead?",
            "D - Describe scene",
            "R - Read text",
            "T - Toggle warnings",
            "Q - Quit"
        ]
        
        for i, text in enumerate(controls):
            cv2.putText(display, text, (10, y_offset + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Warning status
        warning_status = "ON" if self.tracking_warnings_enabled else "OFF"
        cv2.putText(display, f"Warnings: {warning_status}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return display
    
    def check_approaching_warnings(self, tracks):
        """Check for approaching objects and issue warnings."""
        if not self.tracking_warnings_enabled:
            return
        
        current_time = time.time()
        if current_time - self.last_warning_time < self.warning_cooldown:
            return
        
        # Find approaching objects
        approaching = [t for t in tracks if "approaching" in t['direction']]
        
        if len(approaching) > 0:
            warning = self.output_gen.describe_tracks(tracks)
            print(f"[WARNING] {warning}")
            self.tts.speak(warning, blocking=False)
            self.last_warning_time = current_time
    
    def run(self):
        """Main system loop."""
        print("\n=== SYNAPSE CONTROLS ===")
        print("W - What's ahead? (Detection summary)")
        print("D - Describe scene (BLIP captioning)")
        print("R - Read text (OCR)")
        print("T - Toggle tracking warnings")
        print("Q - Quit")
        print("="*40 + "\n")
        
        while True:
            # Read frame
            frame = self.camera.read()
            if frame is None:
                continue
            
            # Run detection
            detections = self.detector.detect(frame, verbose=False)
            
            # Update tracker
            tracks = self.tracker.update(detections, frame)
            
            # Check for warnings
            self.check_approaching_warnings(tracks)
            
            # Update FPS
            self.fps_counter.update()
            
            # Draw interface
            display = self.draw_interface(frame, detections, tracks)
            
            # Show
            cv2.imshow("SYNAPSE - Assistive Vision System", display)
            
            # Handle commands
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n[SYSTEM] Shutting down...")
                break
            
            elif key == ord('w'):
                # What's ahead?
                print("\n[USER] What's ahead?")
                description = self.output_gen.describe_tracks(tracks)
                print(f"[SYSTEM] {description}")
                self.tts.speak(description, blocking=False)
            
            elif key == ord('d'):
                # Describe scene
                print("\n[USER] Describe scene")
                self.tts.speak("Analyzing scene", blocking=False)
                
                caption = self.captioner.generate_caption(frame, verbose=False)
                formatted = self.output_gen.format_caption(caption)
                
                print(f"[SYSTEM] {formatted}")
                self.tts.speak(formatted, blocking=False)
            
            elif key == ord('r'):
                # Read text
                print("\n[USER] Read text")
                self.tts.speak("Reading text", blocking=False)
                
                text, _ = self.ocr.extract_text(frame, verbose=False)
                formatted = self.output_gen.format_ocr_result(text)
                
                print(f"[SYSTEM] {formatted}")
                self.tts.speak(formatted, blocking=False)
            
            elif key == ord('t'):
                # Toggle warnings
                self.tracking_warnings_enabled = not self.tracking_warnings_enabled
                status = "enabled" if self.tracking_warnings_enabled else "disabled"
                print(f"\n[SYSTEM] Tracking warnings {status}")
                self.tts.speak(f"Warnings {status}", blocking=False)
        
        # Cleanup
        self.camera.stop()
        cv2.destroyAllWindows()
        
        print("\n[SYSTEM] SYNAPSE shutdown complete")
        print("="*60 + "\n")


def main():
    """Entry point."""
    try:
        system = SynapseSystem()
        system.run()
    except KeyboardInterrupt:
        print("\n\n[SYSTEM] Interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
