"""
tflite_recognizer.py
---------------------
TFLite-based face recognition — drop-in replacement for face_recognizer.py
for use when a .tflite model is available on the device.

Falls back to face_recognizer.py automatically if no .tflite model is found.
This ensures the system works during development before TFLite conversion.

Public functions match face_recognizer.py signatures exactly:
  extract_embedding_tflite(frame_bgr) -> np.ndarray | None
  match_face_tflite(live_embedding, enrolled_embedding, threshold) -> (bool, float)

Author: EdgeAuth Offline Verification Platform
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2        # type: ignore[import-untyped]
import numpy as np  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------

_AI_ENGINE_DIR = Path(__file__).resolve().parent
if str(_AI_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(_AI_ENGINE_DIR))

logger = logging.getLogger("tflite_recognizer")

# ---------------------------------------------------------------------------
# TFLite model path
# ---------------------------------------------------------------------------

_MODEL_PATH = _AI_ENGINE_DIR / "models" / "mobilefacenet.tflite"

# ---------------------------------------------------------------------------
# TFLite interpreter — attempt to load on import
# ---------------------------------------------------------------------------

TFLITE_AVAILABLE: bool = False
_interpreter = None
_input_details  = None
_output_details = None

if _MODEL_PATH.exists():
    # Try tflite_runtime first (lightweight, recommended for edge devices)
    _loaded = False
    try:
        from tflite_runtime.interpreter import Interpreter as _TFLiteInterpreter  # type: ignore[import]
        _interpreter    = _TFLiteInterpreter(model_path=str(_MODEL_PATH))
        _interpreter.allocate_tensors()
        _input_details  = _interpreter.get_input_details()
        _output_details = _interpreter.get_output_details()
        TFLITE_AVAILABLE = True
        _loaded = True
        logger.info("TFLite model loaded via tflite_runtime: %s", _MODEL_PATH)
    except ImportError:
        pass

    if not _loaded:
        try:
            import tensorflow as tf  # type: ignore[import]
            _interpreter    = tf.lite.Interpreter(model_path=str(_MODEL_PATH))
            _interpreter.allocate_tensors()
            _input_details  = _interpreter.get_input_details()
            _output_details = _interpreter.get_output_details()
            TFLITE_AVAILABLE = True
            logger.info("TFLite model loaded via TensorFlow: %s", _MODEL_PATH)
        except ImportError:
            logger.warning(
                "Neither tflite_runtime nor tensorflow is installed. "
                "TFLite inference unavailable. Falling back to face_recognizer."
            )
        except Exception as exc:
            logger.warning("Failed to load TFLite model: %s — falling back to face_recognizer.", exc)
else:
    logger.warning(
        "TFLite model not found at '%s'. "
        "Run convert_to_tflite.py to generate it. "
        "Falling back to face_recognizer (dlib).",
        _MODEL_PATH,
    )


# ---------------------------------------------------------------------------
# Fallback import from face_recognizer
# ---------------------------------------------------------------------------

from face_recognizer import (  # type: ignore[import]
    extract_embedding as _dlib_extract_embedding,
    match_face,
)


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _preprocess_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Resize frame to 112×112, convert BGR→RGB, normalize to [-1, 1],
    reshape to (1, 3, 112, 112) channels-first format for MobileFaceNet.
    """
    resized    = cv2.resize(frame_bgr, (112, 112))
    rgb        = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = (rgb.astype(np.float32) / 127.5) - 1.0          # [-1, 1]
    chw        = np.transpose(normalized, (2, 0, 1))              # HWC → CHW
    batch      = np.expand_dims(chw, axis=0)                      # (1, 3, 112, 112)
    return batch


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a 1-D vector."""
    norm = np.linalg.norm(vec)
    if norm < 1e-10:
        return vec
    return vec / norm


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_embedding_tflite(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract a face embedding using the TFLite MobileFaceNet model.

    If TFLite is not available (model missing or runtime not installed),
    transparently falls back to the dlib-backed face_recognizer.extract_embedding().

    Args:
        frame_bgr: OpenCV BGR image (H × W × 3, uint8).

    Returns:
        L2-normalized numpy float64 embedding array, or None if no face detected.
    """
    if not TFLITE_AVAILABLE:
        logger.debug("TFLite unavailable — delegating to face_recognizer.extract_embedding().")
        return _dlib_extract_embedding(frame_bgr)

    try:
        inp = _preprocess_frame(frame_bgr)

        # Check whether the model expects float32 or int8 input
        input_dtype = _input_details[0]["dtype"]
        if input_dtype == np.int8:
            scale, zero_point = _input_details[0]["quantization"]
            inp = (inp / scale + zero_point).astype(np.int8)
        else:
            inp = inp.astype(np.float32)

        _interpreter.set_tensor(_input_details[0]["index"], inp)
        _interpreter.invoke()

        raw_output = _interpreter.get_tensor(_output_details[0]["index"])  # (1, embedding_dim)
        embedding  = raw_output[0].astype(np.float64)
        return _l2_normalize(embedding)

    except Exception as exc:
        logger.warning("TFLite inference failed (%s), falling back to dlib.", exc)
        return _dlib_extract_embedding(frame_bgr)


def match_face_tflite(
    live_emb: np.ndarray,
    enrolled_emb: np.ndarray,
    threshold: float = 0.55,
) -> Tuple[bool, float]:
    """
    Compare two face embeddings using cosine similarity.

    Delegates directly to face_recognizer.match_face() — identical logic,
    no duplication. The threshold default here is 0.55 to match server.py.

    Args:
        live_emb:     Embedding from the live camera frame.
        enrolled_emb: Stored enrollment embedding.
        threshold:    Minimum cosine similarity for a positive match.

    Returns:
        (matched: bool, similarity: float)
    """
    return match_face(live_emb, enrolled_emb, threshold)
