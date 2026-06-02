import base64
import os
import sys
from typing import Any, Dict, Optional

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ai_engine.liveness_api import verify_liveness


def decode_base64_image(image_b64: str):
    """
    Decode a base64 image string into an OpenCV BGR image.
    Supports raw base64 and data URL payloads.
    """
    if not image_b64 or not isinstance(image_b64, str):
        return None

    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(image_b64, validate=True)
    except Exception:
        return None

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)


def verify_liveness_request(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Decode a backend request body and run the liveness challenge verifier.

    Expected payload:
        {
            "image": "<base64>",
            "challenge": "Blink Twice"
        }
    """
    if not isinstance(payload, dict):
        return {
            "success": False,
            "challenge": None,
            "status": "Invalid request"
        }

    challenge = payload.get("challenge")
    image_b64 = payload.get("image")

    if not challenge:
        return {
            "success": False,
            "challenge": challenge,
            "status": "Missing challenge"
        }

    image_bgr = decode_base64_image(image_b64)
    if image_bgr is None:
        return {
            "success": False,
            "challenge": challenge,
            "status": "Invalid image"
        }

    return verify_liveness(image_bgr, challenge)
