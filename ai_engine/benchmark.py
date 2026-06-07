"""
benchmark.py
------------
Measures inference speed and accuracy for the EdgeAuth AI pipeline.
Run on the target device after setup.

Usage:
    python benchmark.py [--camera 0] [--runs 100]

Outputs:
    - Face detection time (ms)
    - Embedding extraction time (ms)
    - Total pipeline time (ms)
    - TFLite vs dlib comparison (if tflite model present)

Author: EdgeAuth Offline Verification Platform
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2        # type: ignore[import-untyped]
import numpy as np  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------

_AI_ENGINE_DIR = Path(__file__).resolve().parent
if str(_AI_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(_AI_ENGINE_DIR))

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from face_recognizer import (  # type: ignore[import]
    extract_embedding,
    get_face_bounding_box,
)

try:
    from tflite_recognizer import (  # type: ignore[import]
        extract_embedding_tflite,
        TFLITE_AVAILABLE,
    )
except ImportError:
    TFLITE_AVAILABLE = False
    extract_embedding_tflite = None

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _time_fn(fn, *args, runs: int = 10) -> Tuple[float, float, float]:
    """
    Call fn(*args) `runs` times and return (min_ms, avg_ms, max_ms).
    """
    times: List[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args)
        times.append((time.perf_counter() - t0) * 1000.0)
    return min(times), sum(times) / len(times), max(times)


def _generate_random_frame(height: int = 720, width: int = 1280) -> np.ndarray:
    """Generate a random BGR frame that simulates a 720p camera image."""
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

_COL_WIDTHS = (31, 10, 10, 10)


def _table_header() -> None:
    w1, w2, w3, w4 = _COL_WIDTHS
    print(f"┌{'─' * w1}┬{'─' * w2}┬{'─' * w3}┬{'─' * w4}┐")
    print(f"│ {'Operation':<{w1-2}} │ {'Min ms':>{w2-2}} │ {'Avg ms':>{w3-2}} │ {'Max ms':>{w4-2}} │")
    print(f"├{'─' * w1}┼{'─' * w2}┼{'─' * w3}┼{'─' * w4}┤")


def _table_row(label: str, min_ms: float, avg_ms: float, max_ms: float) -> None:
    w1, w2, w3, w4 = _COL_WIDTHS
    print(
        f"│ {label:<{w1-2}} │ {min_ms:>{w2-2}.1f} │ {avg_ms:>{w3-2}.1f} │ {max_ms:>{w4-2}.1f} │"
    )


def _table_footer() -> None:
    w1, w2, w3, w4 = _COL_WIDTHS
    print(f"└{'─' * w1}┴{'─' * w2}┴{'─' * w3}┴{'─' * w4}┘")


# ---------------------------------------------------------------------------
# Benchmark runs
# ---------------------------------------------------------------------------

def run_benchmark(runs: int = 100, camera: int = 0) -> float:
    """
    Execute all benchmark measurements and print the results table.
    Returns the average total pipeline time in milliseconds.
    """
    print(f"\n{'='*60}")
    print(f"  EdgeAuth AI Pipeline Benchmark — {runs} runs")
    print(f"{'='*60}\n")

    # -----------------------------------------------------------------------
    # Generate synthetic frames (no real camera needed for benchmarking)
    # -----------------------------------------------------------------------
    print(f"Generating {runs} synthetic 720p frames...")
    frames = [_generate_random_frame() for _ in range(min(runs, 20))]
    # Re-use frames in round-robin for larger run counts
    def _get_frame(i: int) -> np.ndarray:
        return frames[i % len(frames)]

    # -----------------------------------------------------------------------
    # 1. Face detection
    # -----------------------------------------------------------------------
    print("Benchmarking face detection (dlib HOG)...")
    det_times: List[float] = []
    for i in range(runs):
        t0 = time.perf_counter()
        get_face_bounding_box(_get_frame(i))
        det_times.append((time.perf_counter() - t0) * 1000.0)
    det_min, det_avg, det_max = min(det_times), sum(det_times)/len(det_times), max(det_times)

    # -----------------------------------------------------------------------
    # 2. Embedding extraction — dlib
    # -----------------------------------------------------------------------
    print("Benchmarking embedding extraction (dlib ResNet)...")
    emb_dlib_times: List[float] = []
    for i in range(runs):
        t0 = time.perf_counter()
        extract_embedding(_get_frame(i))
        emb_dlib_times.append((time.perf_counter() - t0) * 1000.0)
    dlib_min = min(emb_dlib_times)
    dlib_avg = sum(emb_dlib_times) / len(emb_dlib_times)
    dlib_max = max(emb_dlib_times)

    # -----------------------------------------------------------------------
    # 3. Embedding extraction — TFLite (optional)
    # -----------------------------------------------------------------------
    tfl_times: Optional[List[float]] = None
    if TFLITE_AVAILABLE and extract_embedding_tflite is not None:
        print("Benchmarking embedding extraction (TFLite)...")
        tfl_times = []
        for i in range(runs):
            t0 = time.perf_counter()
            extract_embedding_tflite(_get_frame(i))
            tfl_times.append((time.perf_counter() - t0) * 1000.0)
    else:
        print("TFLite model not available — skipping TFLite benchmark.")
        print("Run convert_to_tflite.py to generate the model.")

    # -----------------------------------------------------------------------
    # Print table
    # -----------------------------------------------------------------------
    _table_header()
    _table_row("Face detection",              det_min,  det_avg,  det_max)
    _table_row("Embedding extraction (dlib)", dlib_min, dlib_avg, dlib_max)
    if tfl_times:
        tfl_min = min(tfl_times)
        tfl_avg = sum(tfl_times) / len(tfl_times)
        tfl_max = max(tfl_times)
        _table_row("Embedding extraction (TFLite)", tfl_min, tfl_avg, tfl_max)
    _table_footer()

    # Total pipeline avg = detection + embedding (dlib baseline)
    total_avg = det_avg + dlib_avg
    tfl_total = (det_avg + (sum(tfl_times)/len(tfl_times))) if tfl_times else None

    print(f"\nPipeline totals:")
    print(f"  dlib baseline:  {total_avg:.1f} ms avg")
    if tfl_total is not None:
        speedup = total_avg / tfl_total if tfl_total > 0 else float("inf")
        print(f"  TFLite total:   {tfl_total:.1f} ms avg  ({speedup:.1f}× faster than dlib)")
    print(f"\n  Target: all operations < 1000 ms total")
    if total_avg < 1000.0:
        print(f"  ✓ PASS — avg total {total_avg:.0f} ms is within target.")
    else:
        print(f"  ✗ FAIL — avg total {total_avg:.0f} ms exceeds 1000 ms target.")

    return total_avg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="EdgeAuth AI Pipeline Performance Benchmark"
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera index (default: 0). Reserved for future live-camera benchmarks.",
    )
    parser.add_argument(
        "--runs", type=int, default=100,
        help="Number of benchmark iterations (default: 100).",
    )
    args = parser.parse_args()

    avg_total_ms = run_benchmark(runs=args.runs, camera=args.camera)
    return 0 if avg_total_ms < 1000.0 else 1


if __name__ == "__main__":
    sys.exit(main())
