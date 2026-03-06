"""
Step 1.1 Validation: Test ByteTrack stability (improved metrics).
Move objects around - IDs should stay consistent.
Press 'q' to quit.
"""

import cv2
import yaml
import time
from src.vision.tracker import ByteTracker
from src.io.camera import CameraCapture
from src.utils.fps_counter import FPSCounter


def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print("\n=== STEP 1.1: BYTETRACK TEST (Improved Metrics) ===\n")
    print("Move objects around - watch for stable IDs")
    print("Press 'q' to quit\n")

    camera = CameraCapture(
        device_id=config['camera']['device_id'],
        width=config['camera']['width'],
        height=config['camera']['height']
    )
    camera.start()

    frame = None
    for _ in range(50):
        frame = camera.read()
        if frame is not None:
            break
        time.sleep(0.1)

    tracker = ByteTracker(
        model_path=config['detection']['model'],
        device=config['detection']['device'],
        confidence=config['detection']['confidence'],
        iou=config['detection']['iou'],
        max_age=config['tracking']['max_age']
    )

    fps_counter = FPSCounter(avg_over_frames=30)

    # Smarter tracking metrics
    true_id_switches = 0       # ID changed for same spatial position
    new_object_entries = 0     # Brand new object entered frame
    max_track_age = {}         # track_id -> frame count alive
    last_frame_tracks = {}     # position_key -> track_id (previous frame)
    frame_num = 0
    start_time = time.time()

    while True:
        frame = camera.read()
        if frame is None:
            continue

        frame_num += 1
        tracks = tracker.track(frame)

        fps_counter.update()
        fps = fps_counter.get_fps()

        # --- Smart ID switch detection ---
        current_frame_tracks = {}
        for track in tracks:
            tid = track['track_id']
            cx, cy = track['center']

            # Quantize position into a rough grid cell (80x80 px zones)
            zone_x = cx // 80
            zone_y = cy // 80
            zone_key = (zone_x, zone_y)

            current_frame_tracks[zone_key] = tid

            # Update max age
            if tid not in max_track_age:
                max_track_age[tid] = 0
                new_object_entries += 1
            max_track_age[tid] += 1

        # Check for ID switches: same zone, different ID
        for zone_key, prev_id in last_frame_tracks.items():
            if zone_key in current_frame_tracks:
                curr_id = current_frame_tracks[zone_key]
                if curr_id != prev_id:
                    true_id_switches += 1

        last_frame_tracks = current_frame_tracks
        elapsed = time.time() - start_time

        # --- Draw ---
        display = frame.copy()

        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            tid = track['track_id']
            label = track['class_name']
            direction = track['direction']
            age = max_track_age.get(tid, 0)

            if "approaching" in direction:
                color = (0, 0, 255)
            elif "moving away" in direction:
                color = (255, 0, 0)
            else:
                color = (0, 255, 0)

            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, f"ID:{tid} {label} [{age}f]",
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, color, 2)
            cv2.putText(display, direction,
                       (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                       0.4, color, 1)

        # Stats overlay
        stats = [
            f"FPS: {fps:.1f}",
            f"Tracks: {len(tracks)}",
            f"True Switches: {true_id_switches}",
            f"New Entries: {new_object_entries}",
            f"Elapsed: {elapsed:.0f}s"
        ]
        for i, stat in enumerate(stats):
            cv2.putText(display, stat, (10, 30 + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                       (0, 255, 0) if i < 2 else (0, 255, 255), 2)

        cv2.imshow("SYNAPSE - ByteTrack Test", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.stop()
    cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    longest_track = max(max_track_age.values()) if max_track_age else 0

    print(f"\n=== BYTETRACK RESULTS ===")
    print(f"Duration: {elapsed:.1f}s")
    print(f"Average FPS: {fps:.1f}")
    print(f"Total Objects Seen: {len(max_track_age)}")
    print(f"New Object Entries: {new_object_entries}")
    print(f"True ID Switches: {true_id_switches}")
    print(f"Longest Track: {longest_track} frames ({longest_track/fps:.1f}s)")

    # Verdict
    switch_rate = true_id_switches / max(elapsed, 1)
    print(f"\nSwitch Rate: {switch_rate:.2f} switches/sec")

    if true_id_switches == 0:
        print("✓ PERFECT - Zero ID switches!")
    elif switch_rate < 0.5:
        print("✓ Step 1.1 Complete - ByteTrack stable!")
    elif switch_rate < 1.5:
        print("⚠ Acceptable stability for prototype")
    else:
        print("✗ Still unstable - needs tuning")


if __name__ == "__main__":
    main()
