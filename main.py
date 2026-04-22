"""
SYNAPSE: Real-Time Embedded Vision System for Assistive Navigation
Main orchestrator integrating all modules.


CONTROLS:
- 'w' : What's ahead? (Detection summary with audio)
- 'd' : Describe scene (BLIP captioning)
- 'r' : Read text (OCR)
- 't' : Toggle tracking warnings
- 'q' : Quit
"""


import cv2
import yaml
import time
import suppress_warnings
from src.io.camera import CameraCapture
from src.vision.tracker import ByteTracker
from src.vision.ocr import OCRModule
from src.vision.captioner import SceneCaptioner
from src.vision.distance_estimator import DistanceEstimator
from src.vision.spatial_layout import build_spatial_summary
from src.io.tts_output import TTSOutput
from src.logic.output_generator import OutputGenerator
from src.logic.command_processor import CommandProcessor, CommandContext
from src.utils.fps_counter import FPSCounter
from src.utils.logger import logger
from src.vision.scene_memory import SceneMemory
from src.utils.async_processor import AsyncProcessor


class SynapseSystem:
    """Main SYNAPSE system orchestrator."""


    def __init__(self, config_path: str = 'config.yaml'):
        print("\n" + "="*60)
        print("  PROJECT SYNAPSE - Assistive Vision System")
        print("="*60 + "\n")


        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)


        self._init_camera()
        self._init_tracker()
        self._init_ocr()
        self._init_captioner()
        self._init_tts()
        self._init_output_generator()
        self._init_command_processor()


        # Warning system state
        warn_cfg = self.config['warnings']
        self.warnings_enabled = warn_cfg['enabled']
        self.global_cooldown = warn_cfg['global_cooldown']
        self.per_track_cooldown = warn_cfg['per_track_cooldown']
        self.warn_min_confidence = warn_cfg['min_confidence']
        self.risk_classes = set(warn_cfg['risk_classes'])


        self.last_global_warning_time = 0
        self.last_warning_per_track = {}
        self.distance_estimator = DistanceEstimator()
        self.scene_memory = SceneMemory(memory_duration=5.0)
        self.fps_counter = FPSCounter(avg_over_frames=30)
        self.frame_count = 0
        self.last_tracks = []
        self.async_processor = AsyncProcessor()
        self.warning_count = 0


        print("\n" + "="*60)
        print("  SYNAPSE READY")
        print("="*60 + "\n")


    # ------------------------------------------------------------------ init


    def _init_camera(self):
        cfg = self.config['camera']
        self.camera = CameraCapture(
            device_id=cfg['device_id'],
            width=cfg['width'],
            height=cfg['height']
        )
        self.camera.start()
        for _ in range(50):
            if self.camera.read() is not None:
                break
            time.sleep(0.1)


    def _init_tracker(self):
        det = self.config['detection']
        trk = self.config['tracking']
        self.tracker = ByteTracker(
            model_path=det['model'],
            device=det['device'],
            confidence=det['confidence'],
            iou=det['iou'],
            max_age=trk['max_age']
        )


    def _init_ocr(self):
        cfg = self.config['ocr']
        self.ocr = OCRModule(
            languages=cfg['languages'],
            gpu=cfg['gpu'],
            min_confidence=cfg['min_confidence']
        )


    def _init_captioner(self):
        cfg = self.config['captioning']
        self.captioner = SceneCaptioner(
            model_name=cfg['model'],
            device=cfg['device'],
            max_length=cfg['max_length'],
            min_length=cfg['min_length']
        )


    def _init_tts(self):
        cfg = self.config['tts']
        self.tts = TTSOutput(
            rate=cfg['rate'],
            volume=cfg['volume']
        )


    def _init_output_generator(self):
        self.output_gen = OutputGenerator()


    def _init_command_processor(self):
        self.cmd_processor = CommandProcessor()


    # -------------------------------------------------------- warning system


    def check_warnings(self, tracks):
        """
        Issue warning only when:
        1. Warnings enabled
        2. Risk class with high confidence
        3. Object is approaching
        4. Object is NOT too close/large (avoids hand false positives)
        5. Cooldowns respected
        """


        if not self.warnings_enabled:
            return


        now = time.time()
        if now - self.last_global_warning_time < self.global_cooldown:
            return


        warn_cfg = self.config['warnings']
        frame_area = self.config['camera']['width'] * self.config['camera']['height']
        min_ratio = warn_cfg.get('min_bbox_ratio', 0.02)
        max_ratio = warn_cfg.get('max_bbox_ratio', 0.35)


        candidates = []
        for track in tracks:
            class_name = track['class_name']
            direction = track['direction']
            confidence = track['confidence']
            track_id = track['track_id']
            x1, y1, x2, y2 = track['bbox']


            # Risk class check
            if class_name not in self.risk_classes:
                continue


            # Must be consistently approaching
            if "approaching" not in direction:
                continue


            # Confidence check
            if confidence < self.warn_min_confidence:
                continue


            # Bbox size filter
            bbox_area = (x2 - x1) * (y2 - y1)
            bbox_ratio = bbox_area / frame_area
            if not (min_ratio <= bbox_ratio <= max_ratio):
                continue


            # Per-track cooldown
            last_warned = self.last_warning_per_track.get(track_id, 0)
            if now - last_warned < self.per_track_cooldown:
                continue


            candidates.append(track)


        if not candidates:
            return


        # Score = confidence + proximity bonus (closer = higher priority)
        def priority_score(track):
            distance = self.distance_estimator.estimate(track['class_name'], track['bbox'])
            proximity_bonus = (1.0 / distance) if distance and distance > 0 else 0
            return track['confidence'] + proximity_bonus


        best = max(candidates, key=priority_score)
        direction = best['direction']
        class_name = best['class_name']


        distance = self.distance_estimator.estimate(class_name, best['bbox'])
        dist_str = self.distance_estimator.format_distance(distance)


        if "left" in direction:
            message = f"Caution! {class_name} approaching from your left"
        else:
            message = f"Caution! {class_name} approaching from your right"


        if dist_str:
            message = f"{message}, {dist_str}"


        logger.info(f"[WARNING] {message}")
        self.tts.speak(message, blocking=False)
        self.warning_count += 1


        self.last_global_warning_time = now
        self.last_warning_per_track[best['track_id']] = now


        self.last_warning_per_track = {
            tid: t for tid, t in self.last_warning_per_track.items()
            if now - t < self.per_track_cooldown * 2
        }


    # --------------------------------------------------------------- display


    def draw_interface(self, frame, tracks):
        """Draw UI overlay on frame."""
        import psutil
        display = frame.copy()


        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            tid = track['track_id']
            class_name = track['class_name']
            direction = track['direction']
            conf = track['confidence']


            # Color by direction
            if "approaching" in direction:
                color = (0, 0, 255)      # Red
            elif "moving away" in direction:
                color = (255, 100, 0)    # Blue
            else:
                color = (0, 255, 0)      # Green


            # Bounding box
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)


            # Label: ID + class + confidence
            label = f"ID:{tid} {class_name} {conf:.2f}"
            cv2.putText(display, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


            # Direction below box
            cv2.putText(display, direction, (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


        # Stats
        fps = self.fps_counter.get_fps()
        cpu = psutil.cpu_percent()
        warn_status = "ON" if self.warnings_enabled else "OFF"


        cv2.putText(display, f"FPS: {fps:.1f}  CPU: {cpu:.0f}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f"Objects: {len(tracks)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f"Warnings: {warn_status}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


        # Async processing indicator
        if self.async_processor.is_running:
            cv2.putText(display, "Processing...", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)


        # Controls
        controls = ["W-Whats ahead", "D-Describe", "R-Read text",
                    "T-Toggle warns", "Q-Quit"]
        for i, ctrl in enumerate(controls):
            cv2.putText(display, ctrl,
                        (10, frame.shape[0] - 110 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1)


        return display


    # --------------------------------------------------------- context build


    def _build_context(self, key: int, frame, tracks) -> CommandContext:
        """Package all live state into a CommandContext for the processor."""
        return CommandContext(
            key=key,
            frame=frame,
            tracks=tracks,
            tts=self.tts,
            output_gen=self.output_gen,
            async_proc=self.async_processor,
            scene_memory=self.scene_memory,
            captioner=self.captioner,
            ocr=self.ocr,
            distance_est=self.distance_estimator,
            frame_width=self.config['camera']['width'],
            warnings_enabled=self.warnings_enabled,
        )


    # -------------------------------------------------------- session summary


    def print_session_summary(self, session_start: float):
        """Print a summary of what was detected during the session."""
        import time
        duration = time.time() - session_start
        minutes = int(duration // 60)
        seconds = int(duration % 60)


        # Collect all unique classes ever tracked
        seen_classes = set()
        for tid, history in self.tracker.track_history.items():
            pass  # track_history stores positions, not classes


        logger.info("\n" + "=" * 60)
        logger.info("  SESSION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  Duration        : {minutes}m {seconds}s")
        logger.info(f"  Total warnings  : {self.warning_count}")
        logger.info(f"  Log saved to    : logs/")
        logger.info("=" * 60 + "\n")


    # ------------------------------------------------------------------ run


    def run(self):
        """Main system loop."""
        print("=== CONTROLS ===")
        print("W - What's ahead?")
        print("D - Describe scene")
        print("R - Read text")
        print("T - Toggle warnings")
        print("Q - Quit\n")

        session_start = time.time()
        while True:
            frame = self.camera.read()
            if frame is None:
                continue


            # Frame skipping — run YOLO every Nth frame
            self.frame_count += 1
            frame_skip = self.config['camera'].get('frame_skip', 2)


            if self.frame_count % frame_skip == 0:
                self.last_tracks = self.tracker.track(frame)


            tracks = self.last_tracks


            # Check warnings
            self.check_warnings(tracks)


            # FPS
            self.fps_counter.update()


            # Draw
            display = self.draw_interface(frame, tracks)
            cv2.imshow("SYNAPSE - Assistive Vision System", display)


            # Commands
            key = cv2.waitKey(1) & 0xFF
            if key == 0xFF:
                continue


            ctx = self._build_context(key, frame, tracks)
            result = self.cmd_processor.handle(ctx)

            # Sync warnings toggle back from result
            self.warnings_enabled = result.warnings_enabled

            if result.should_quit:
                self.print_session_summary(session_start)
                break


        # Cleanup
        self.camera.stop()
        cv2.destroyAllWindows()
        print("\n[SYSTEM] SYNAPSE shutdown complete\n")



def main():
    try:
        system = SynapseSystem()
        system.run()
    except KeyboardInterrupt:
        print("\n[SYSTEM] Interrupted")
    except Exception as e:
        logger.info(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()



if __name__ == "__main__":
    main()