"""
Test: CommandProcessor + OutputGenerator validation.
Runs the full command dispatch pipeline against live camera + real modules.

CONTROLS:
- 'w' : What's ahead?        (tests W command + scene memory + OutputGenerator)
- 'd' : Describe scene        (tests D command + async BLIP captioning)
- 'r' : Read text             (tests R command + async OCR)
- 't' : Toggle warnings       (tests T command + state sync)
- 'q' : Quit
"""


import cv2
import yaml
import time
from src.io.camera import CameraCapture
from src.vision.tracker import ByteTracker
from src.vision.ocr import OCRModule
from src.vision.captioner import SceneCaptioner
from src.vision.distance_estimator import DistanceEstimator
from src.io.tts_output import TTSOutput
from src.logic.output_generator import OutputGenerator
from src.logic.command_processor import CommandProcessor, CommandContext
from src.vision.scene_memory import SceneMemory
from src.utils.async_processor import AsyncProcessor
from src.utils.logger import logger


def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print("\n=== COMMAND PROCESSOR TEST ===\n")

    # --- Init camera ---
    cam_cfg = config['camera']
    camera = CameraCapture(
        device_id=cam_cfg['device_id'],
        width=cam_cfg['width'],
        height=cam_cfg['height']
    )
    camera.start()

    frame = None
    for _ in range(50):
        frame = camera.read()
        if frame is not None:
            break
        time.sleep(0.1)

    if frame is None:
        print("[ERROR] Camera failed to initialise")
        camera.stop()
        return

    # --- Init modules ---
    det  = config['detection']
    trk  = config['tracking']
    tracker = ByteTracker(
        model_path=det['model'],
        device=det['device'],
        confidence=det['confidence'],
        iou=det['iou'],
        max_age=trk['max_age']
    )

    ocr_cfg = config['ocr']
    ocr = OCRModule(
        languages=ocr_cfg['languages'],
        gpu=ocr_cfg['gpu'],
        min_confidence=ocr_cfg['min_confidence']
    )

    cap_cfg = config['captioning']
    captioner = SceneCaptioner(
        model_name=cap_cfg['model'],
        device=cap_cfg['device'],
        max_length=cap_cfg['max_length'],
        min_length=cap_cfg['min_length']
    )

    tts_cfg = config['tts']
    tts = TTSOutput(
        rate=tts_cfg['rate'],
        volume=tts_cfg['volume']
    )

    output_gen     = OutputGenerator()
    cmd_processor  = CommandProcessor()
    scene_memory   = SceneMemory(memory_duration=5.0)
    async_proc     = AsyncProcessor()
    distance_est   = DistanceEstimator()

    # --- Test tracking ---
    print("\n[CHECK] Running one detection pass...")
    test_tracks = tracker.track(frame)
    print(f"  Tracker OK — {len(test_tracks)} objects detected in test frame")

    # --- Test OutputGenerator directly ---
    print("\n[CHECK] OutputGenerator.format_scene_changes()...")
    changes_test = {"new": [("person", "ahead", 2.0)], "closer": [], "same": [], "gone": []}
    result_str = output_gen.format_scene_changes(changes_test)
    assert "person" in result_str, f"FAIL: expected 'person' in output, got: {result_str}"
    print(f"  format_scene_changes OK — '{result_str}'")

    changes_empty = {"new": [], "closer": [], "same": [], "gone": []}
    result_clear = output_gen.format_scene_changes(changes_empty)
    assert "clear" in result_clear.lower(), f"FAIL: expected 'clear', got: {result_clear}"
    print(f"  format_scene_changes (empty) OK — '{result_clear}'")

    result_ocr = output_gen.format_ocr_result("EXIT")
    assert "EXIT" in result_ocr, f"FAIL: expected 'EXIT' in result, got: {result_ocr}"
    print(f"  format_ocr_result OK — '{result_ocr}'")

    result_cap = output_gen.format_caption("a chair near the window")
    assert result_cap[0].isupper() and result_cap.endswith("."), f"FAIL: bad caption format: {result_cap}"
    print(f"  format_caption OK — '{result_cap}'")

    # --- Test CommandProcessor dispatch (no camera, mocked context) ---
    print("\n[CHECK] CommandProcessor dispatch (key logic)...")

    class _FakeTTS:
        def __init__(self): self.last = None
        def speak(self, text, blocking=True): self.last = text; print(f"    TTS: {text}")

    class _FakeAsync:
        is_running = False
        def run(self, fn, callback=None): callback(fn()) if callback else fn()

    class _FakeMemory:
        def get_changes(self, *a): return {"new": [], "closer": [], "same": [], "gone": []}
        def update(self, *a): pass

    fake_tts   = _FakeTTS()
    fake_async = _FakeAsync()
    fake_mem   = _FakeMemory()

    import numpy as np
    blank_frame = np.zeros((480, 640, 3), dtype="uint8")

    def _ctx(key, warnings_enabled=True):
        return CommandContext(
            key=key,
            frame=blank_frame.copy(),
            tracks=[],
            tts=fake_tts,
            output_gen=output_gen,
            async_proc=fake_async,
            scene_memory=fake_mem,
            captioner=captioner,
            ocr=ocr,
            distance_est=distance_est,
            frame_width=640,
            warnings_enabled=warnings_enabled,
        )

    # Q → should_quit
    res = cmd_processor.handle(_ctx(ord('q')))
    assert res.should_quit is True, "FAIL: Q key did not set should_quit"
    print("  Q → should_quit=True  OK")

    # T → toggle off
    res = cmd_processor.handle(_ctx(ord('t'), warnings_enabled=True))
    assert res.warnings_enabled is False, "FAIL: T key did not disable warnings"
    print("  T → warnings toggled off  OK")

    # T → toggle on
    res = cmd_processor.handle(_ctx(ord('t'), warnings_enabled=False))
    assert res.warnings_enabled is True, "FAIL: T key did not enable warnings"
    print("  T → warnings toggled on  OK")

    # W → handled, TTS spoken
    res = cmd_processor.handle(_ctx(ord('w')))
    assert res.handled is True, "FAIL: W key not handled"
    assert fake_tts.last is not None, "FAIL: W key did not call TTS"
    print(f"  W → handled, TTS spoken  OK")

    # Unknown key → not handled
    res = cmd_processor.handle(_ctx(ord('z')))
    assert res.handled is False, "FAIL: unknown key should not be handled"
    print("  Unknown key → no-op  OK")

    print("\n[ALL CHECKS PASSED] CommandProcessor and OutputGenerator are working correctly.")
    print("\n=== LIVE TEST (camera) ===")
    print("Commands now tested against live camera + real modules.")
    print("\nCONTROLS:")
    print("  W - What\'s ahead?")
    print("  D - Describe scene")
    print("  R - Read text")
    print("  T - Toggle warnings")
    print("  Q - Quit\n")

    # --- Live test state ---
    warnings_enabled = config['warnings']['enabled']
    frame_count = 0
    last_tracks = []

    while True:
        frame = camera.read()
        if frame is None:
            continue

        frame_count += 1
        if frame_count % cam_cfg.get('frame_skip', 2) == 0:
            last_tracks = tracker.track(frame)
        tracks = last_tracks

        # HUD
        display = frame.copy()
        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            color = (0, 0, 255) if "approaching" in track['direction'] else (0, 255, 0)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, f"{track['class_name']} {track['confidence']:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        warn_status = "ON" if warnings_enabled else "OFF"
        cv2.putText(display, f"Objects: {len(tracks)}  Warnings: {warn_status}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if async_proc.is_running:
            cv2.putText(display, "Processing...", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        controls = ["W-Whats ahead", "D-Describe", "R-Read text",
                    "T-Toggle warns", "Q-Quit"]
        for i, ctrl in enumerate(controls):
            cv2.putText(display, ctrl, (10, display.shape[0] - 110 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        cv2.imshow("SYNAPSE - CommandProcessor Test", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 0xFF:
            continue

        ctx = CommandContext(
            key=key,
            frame=frame,
            tracks=tracks,
            tts=tts,
            output_gen=output_gen,
            async_proc=async_proc,
            scene_memory=scene_memory,
            captioner=captioner,
            ocr=ocr,
            distance_est=distance_est,
            frame_width=cam_cfg['width'],
            warnings_enabled=warnings_enabled,
        )

        result = cmd_processor.handle(ctx)
        warnings_enabled = result.warnings_enabled

        if result.should_quit:
            break

    camera.stop()
    cv2.destroyAllWindows()
    print("\n✓ CommandProcessor test complete.")


if __name__ == "__main__":
    main()