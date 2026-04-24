"""
D1 Benchmark — No-camera detector speed test for Pi.
Tests YOLO26n inference speed on static frames without needing a camera.
"""

import time
import numpy as np
import psutil
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.vision.detector import YOLODetector

def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except:
        return None

def run_benchmark():
    print("=" * 50)
    print("   SYNAPSE D1 — YOLO26n Benchmark (No Camera)")
    print("=" * 50)

    mem = psutil.virtual_memory()
    print(f"\n[SYSTEM] RAM: {mem.total // (1024**2)}MB total, {mem.available // (1024**2)}MB available")
    print(f"[SYSTEM] CPU cores: {psutil.cpu_count()}")
    temp = get_cpu_temp()
    if temp:
        print(f"[SYSTEM] CPU temp: {temp:.1f}C")

    print("\n[STEP 1] Loading YOLO26n model...")
    t_load = time.time()
    detector = YOLODetector(
        model_path="yolo26n_ncnn_256",
        device="cpu",
        confidence=0.35,
        iou=0.4,
        detection_size=256,
        num_threads=4
    )
    load_time = time.time() - t_load
    print(f"[STEP 1] Model loaded in {load_time:.2f}s")

    print("\n[STEP 2] Warming up (5 frames)...")
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    for _ in range(5):
        detector.detect(test_frame)
    print("[STEP 2] Warmup done")

    print("\n[STEP 3] Benchmarking 30 frames...")
    times = []
    cpu_samples = []

    for i in range(30):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        t0 = time.perf_counter()
        detections = detector.detect(frame)
        t1 = time.perf_counter()
        cpu_samples.append(psutil.cpu_percent(interval=None))
        times.append(t1 - t0)

        if (i + 1) % 10 == 0:
            recent_fps = 1.0 / (sum(times[-10:]) / 10)
            print(f"  Frame {i+1}/30 — {recent_fps:.1f} FPS")

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    avg_fps = 1.0 / avg_time
    peak_fps = 1.0 / min_time
    avg_cpu = sum(cpu_samples) / len(cpu_samples)

    mem_after = psutil.virtual_memory()
    mem_used = (mem.available - mem_after.available) // (1024**2)
    temp_after = get_cpu_temp()

    print("\n" + "=" * 50)
    print("   BENCHMARK RESULTS")
    print("=" * 50)
    print(f"  Avg inference time : {avg_time*1000:.1f} ms")
    print(f"  Min inference time : {min_time*1000:.1f} ms")
    print(f"  Max inference time : {max_time*1000:.1f} ms")
    print(f"  Avg FPS            : {avg_fps:.1f}")
    print(f"  Peak FPS           : {peak_fps:.1f}")
    print(f"  Avg CPU usage      : {avg_cpu:.1f}%")
    print(f"  RAM used by model  : ~{mem_used}MB")
    if temp_after:
        print(f"  CPU temp after     : {temp_after:.1f}C")
    print("=" * 50)

    print("\n  Target (Pi): 15-20 FPS")
    if avg_fps >= 15:
        print(f"  Status: TARGET MET ({avg_fps:.1f} FPS)")
    elif avg_fps >= 10:
        print(f"  Status: CLOSE ({avg_fps:.1f} FPS) — optimisation needed")
    else:
        print(f"  Status: BELOW TARGET ({avg_fps:.1f} FPS) — quantization required")
    print("=" * 50)

if __name__ == "__main__":
    run_benchmark()
