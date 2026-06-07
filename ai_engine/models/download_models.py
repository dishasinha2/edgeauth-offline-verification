"""
download_models.py
------------------
Downloads the MobileFaceNet TFLite model required for face recognition.
Run this once before starting the server:

    cd ai_engine/models && python download_models.py
    # or from the project root:
    python ai_engine/models/download_models.py

Note: The face_landmarker.task file is already present in ai_engine/ root.
This script downloads only the face recognition TFLite model used by
tflite_recognizer.py.

Output:
    ai_engine/models/mobilefacenet.tflite
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# MediaPipe Face Embedder (MobileFaceNet backbone, float16 precision).
# This model produces 128-dimensional L2-normalized face embeddings.
# Source: https://ai.google.dev/edge/mediapipe/solutions/vision/face_embedder
# ---------------------------------------------------------------------------

MODELS = [
    {
        "filename": "mobilefacenet.tflite",
        "url": (
            "https://storage.googleapis.com/mediapipe-models/"
            "face_embedder/face_embedder/float16/1/face_embedder.tflite"
        ),
        "description": "MediaPipe Face Embedder (MobileFaceNet) — 128-d float16 TFLite",
    },
]


# ---------------------------------------------------------------------------
# Progress reporter
# ---------------------------------------------------------------------------

def _progress(block_num: int, block_size: int, total_size: int) -> None:
    if total_size > 0:
        pct = min(100, int(block_num * block_size * 100 / total_size))
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r  [{bar}] {pct}%", end="", flush=True)


# ---------------------------------------------------------------------------
# Main download loop
# ---------------------------------------------------------------------------

def download_models() -> bool:
    """
    Download all models listed in MODELS to MODELS_DIR.
    Returns True if all downloads succeeded, False if any failed.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    all_ok = True

    for model in MODELS:
        dest = MODELS_DIR / model["filename"]

        if dest.exists():
            size_kb = dest.stat().st_size / 1024
            print(f"[SKIP] {model['filename']} already present ({size_kb:.1f} KB)")
            continue

        print(f"\n[DOWNLOAD] {model['description']}")
        print(f"  URL : {model['url']}")
        print(f"  Dest: {dest}")

        try:
            urllib.request.urlretrieve(model["url"], dest, _progress)
            print()  # newline after progress bar
            size_kb = dest.stat().st_size / 1024
            print(f"  Done — {size_kb:.1f} KB saved to {dest.name}")
        except Exception as exc:
            print(f"\n  [ERROR] Download failed: {exc}")
            print(f"  Manually download from:")
            print(f"    {model['url']}")
            print(f"  and save as:")
            print(f"    {dest}")
            if dest.exists():
                dest.unlink()  # Remove partial file
            all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("  EdgeAuth — Model Downloader")
    print(f"  Target directory: {MODELS_DIR}")
    print("=" * 55)

    success = download_models()

    if success:
        print("\n✓ All models ready.")
        print("  Start the server with:")
        print("    python ai_engine/server.py")
        sys.exit(0)
    else:
        print("\n✗ One or more downloads failed. See errors above.")
        sys.exit(1)
