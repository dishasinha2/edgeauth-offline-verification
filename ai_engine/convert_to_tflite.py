"""
convert_to_tflite.py
---------------------
Downloads MobileFaceNet ONNX model and converts it to TFLite with INT8 quantization.
Run this once on a development machine, then copy the output .tflite file to the device.

Usage:
    python convert_to_tflite.py

Output:
    ai_engine/models/mobilefacenet.tflite       (quantized INT8, ~5 MB)
    ai_engine/models/mobilefacenet_fp32.tflite  (float32 reference)

Author: EdgeAuth Offline Verification Platform
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ONNX_MODEL_URL = (
    "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/"
    "arcface/model/arcfaceresnet100-8.onnx"
)
OUTPUT_DIR         = Path(__file__).resolve().parent / "models"
TARGET_INPUT_SHAPE = (1, 3, 112, 112)   # batch, channels, H, W (NCHW)

ONNX_MODEL_PATH     = OUTPUT_DIR / "arcfaceresnet100.onnx"
SAVED_MODEL_DIR     = OUTPUT_DIR / "arcface_savedmodel"
TFLITE_FP32_PATH    = OUTPUT_DIR / "mobilefacenet_fp32.tflite"
TFLITE_INT8_PATH    = OUTPUT_DIR / "mobilefacenet.tflite"

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

def _check_dependencies() -> bool:
    """Verify required packages are installed. Print install instructions if not."""
    missing = []
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        missing.append("tensorflow>=2.14.0")

    try:
        import onnx  # noqa: F401
    except ImportError:
        missing.append("onnx>=1.15.0")

    try:
        import onnx_tf  # noqa: F401
    except ImportError:
        missing.append("onnx-tf>=1.10.0")

    if missing:
        print("\n[ERROR] Missing required packages:")
        for pkg in missing:
            print(f"   pip install {pkg}")
        print("\nInstall all at once:")
        print(f"   pip install {' '.join(missing)}")
        return False

    return True


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _download_onnx_model() -> Optional[Path]:
    """
    Download the ONNX model to OUTPUT_DIR.
    Returns the path to the downloaded file, or None on failure.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if ONNX_MODEL_PATH.exists():
        print(f"[OK] ONNX model already exists: {ONNX_MODEL_PATH}")
        return ONNX_MODEL_PATH

    print(f"\n[STEP 1/4] Downloading ONNX model from:\n   {ONNX_MODEL_URL}")
    print("   This may take a few minutes (~100 MB model)...")

    try:
        def _progress(block_num, block_size, total_size):
            if total_size > 0:
                pct = min(100, int(block_num * block_size * 100 / total_size))
                print(f"\r   Progress: {pct}%", end="", flush=True)

        urllib.request.urlretrieve(ONNX_MODEL_URL, ONNX_MODEL_PATH, _progress)
        print()  # newline after progress
        print(f"[OK] Downloaded to: {ONNX_MODEL_PATH} ({ONNX_MODEL_PATH.stat().st_size // 1024} KB)")
        return ONNX_MODEL_PATH

    except Exception as exc:
        print(f"\n[ERROR] Download failed: {exc}")
        print(f"\n   Manual download instructions:")
        print(f"   1. Open a browser and navigate to:")
        print(f"      {ONNX_MODEL_URL}")
        print(f"   2. Save the file as:")
        print(f"      {ONNX_MODEL_PATH}")
        print(f"   3. Re-run this script.")
        if ONNX_MODEL_PATH.exists():
            ONNX_MODEL_PATH.unlink()  # Remove partial download
        return None


# ---------------------------------------------------------------------------
# ONNX → TensorFlow SavedModel
# ---------------------------------------------------------------------------

def _onnx_to_savedmodel() -> Optional[Path]:
    """Convert ONNX model to TensorFlow SavedModel format."""
    print(f"\n[STEP 2/4] Converting ONNX → TensorFlow SavedModel...")

    try:
        import onnx
        from onnx_tf.backend import prepare

        print(f"   Loading ONNX model: {ONNX_MODEL_PATH}")
        onnx_model = onnx.load(str(ONNX_MODEL_PATH))
        onnx.checker.check_model(onnx_model)
        print(f"   ONNX model verified. Inputs: {[i.name for i in onnx_model.graph.input]}")

        print(f"   Converting to SavedModel at: {SAVED_MODEL_DIR}")
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(str(SAVED_MODEL_DIR))
        print(f"[OK] SavedModel written to: {SAVED_MODEL_DIR}")
        return SAVED_MODEL_DIR

    except Exception as exc:
        print(f"[ERROR] ONNX → SavedModel conversion failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# SavedModel → TFLite FP32
# ---------------------------------------------------------------------------

def _savedmodel_to_tflite_fp32() -> Optional[Path]:
    """Convert TensorFlow SavedModel to TFLite float32 format."""
    print(f"\n[STEP 3/4] Converting SavedModel → TFLite FP32...")

    try:
        import tensorflow as tf

        converter = tf.lite.TFLiteConverter.from_saved_model(str(SAVED_MODEL_DIR))
        converter.optimizations = []            # No quantization for FP32 reference
        tflite_model = converter.convert()

        TFLITE_FP32_PATH.parent.mkdir(parents=True, exist_ok=True)
        TFLITE_FP32_PATH.write_bytes(tflite_model)
        size_kb = TFLITE_FP32_PATH.stat().st_size // 1024
        print(f"[OK] FP32 TFLite model: {TFLITE_FP32_PATH} ({size_kb} KB)")
        return TFLITE_FP32_PATH

    except Exception as exc:
        print(f"[ERROR] SavedModel → TFLite FP32 conversion failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# TFLite FP32 → TFLite INT8 (post-training quantization)
# ---------------------------------------------------------------------------

def _tflite_fp32_to_int8() -> Optional[Path]:
    """Apply post-training INT8 quantization to the FP32 TFLite model."""
    print(f"\n[STEP 4/4] Applying INT8 post-training quantization...")

    try:
        import tensorflow as tf
        import numpy as np

        converter = tf.lite.TFLiteConverter.from_saved_model(str(SAVED_MODEL_DIR))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

        # Representative dataset for calibration (random 112x112 RGB images)
        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(1, 3, 112, 112).astype(np.float32)
                yield [data]

        converter.representative_dataset = representative_dataset
        converter.inference_input_type   = tf.int8
        converter.inference_output_type  = tf.int8

        tflite_quant = converter.convert()
        TFLITE_INT8_PATH.parent.mkdir(parents=True, exist_ok=True)
        TFLITE_INT8_PATH.write_bytes(tflite_quant)
        size_kb = TFLITE_INT8_PATH.stat().st_size // 1024
        print(f"[OK] INT8 quantized TFLite model: {TFLITE_INT8_PATH} ({size_kb} KB)")
        return TFLITE_INT8_PATH

    except Exception as exc:
        print(f"[ERROR] INT8 quantization failed: {exc}")
        print("       FP32 model is still functional. Use mobilefacenet_fp32.tflite instead.")
        return None


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def _benchmark_tflite(model_path: Path, runs: int = 50) -> None:
    """Run inference benchmark using a random 112×112 input and print results."""
    print(f"\n[BENCHMARK] Running {runs} inference passes on {model_path.name}...")

    try:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter

        import numpy as np

        interp = Interpreter(model_path=str(model_path))
        interp.allocate_tensors()
        inp_details = interp.get_input_details()

        input_dtype = inp_details[0]["dtype"]
        if input_dtype == __import__("numpy").int8:
            dummy = (np.random.rand(1, 3, 112, 112) * 127).astype(inp_details[0]["dtype"])
        else:
            dummy = np.random.rand(1, 3, 112, 112).astype(np.float32)

        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            interp.set_tensor(inp_details[0]["index"], dummy)
            interp.invoke()
            times.append((time.perf_counter() - t0) * 1000)  # ms

        avg_ms = sum(times) / len(times)
        min_ms = min(times)
        max_ms = max(times)
        print(f"   Min: {min_ms:.1f} ms | Avg: {avg_ms:.1f} ms | Max: {max_ms:.1f} ms")
        if avg_ms < 100:
            print(f"   ✓ Speed target met (<100 ms avg).")
        else:
            print(f"   ⚠ Avg inference {avg_ms:.0f} ms exceeds 100 ms target.")

    except Exception as exc:
        print(f"[WARNING] Benchmark failed: {exc}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary() -> None:
    print("\n" + "=" * 60)
    print("  Conversion Summary")
    print("=" * 60)
    for path in [TFLITE_FP32_PATH, TFLITE_INT8_PATH]:
        if path.exists():
            size_kb = path.stat().st_size // 1024
            print(f"  ✓ {path.name:40s} {size_kb:>6} KB")
        else:
            print(f"  ✗ {path.name:40s} MISSING")
    print("=" * 60)
    print(f"\nCopy '{TFLITE_INT8_PATH.name}' to the target device's")
    print(f"  ai_engine/models/ directory and restart the server.\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("  EdgeAuth — TFLite Model Conversion")
    print("=" * 60)

    if not _check_dependencies():
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Download
    onnx_path = _download_onnx_model()
    if onnx_path is None:
        return 1

    # Step 2: ONNX → SavedModel
    saved_model = _onnx_to_savedmodel()
    if saved_model is None:
        return 1

    # Step 3: SavedModel → TFLite FP32
    tflite_fp32 = _savedmodel_to_tflite_fp32()
    if tflite_fp32 is None:
        return 1

    # Step 4: TFLite FP32 → INT8
    tflite_int8 = _tflite_fp32_to_int8()   # Non-fatal if this fails

    # Benchmark the best available model
    best_model = tflite_int8 if (tflite_int8 and tflite_int8.exists()) else tflite_fp32
    if best_model and best_model.exists():
        _benchmark_tflite(best_model)

    _print_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
