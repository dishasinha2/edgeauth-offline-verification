try:
    from .blink_detection import verify_blink as _verify_blink
    from .headpose_detection import verify_headpose as _verify_headpose
    from .liveness_challenge import verify_liveness as _verify_liveness
except ImportError:
    from blink_detection import verify_blink as _verify_blink
    from headpose_detection import verify_headpose as _verify_headpose
    from liveness_challenge import verify_liveness as _verify_liveness


def verify_blink(image_bgr):
    """
    API-ready blink verification.
    Accepts an OpenCV BGR image and returns a JSON-compatible dictionary.
    """
    return _verify_blink(image_bgr)


def verify_headpose(image_bgr):
    """
    API-ready head-pose verification.
    Accepts an OpenCV BGR image and returns a JSON-compatible dictionary.
    """
    return _verify_headpose(image_bgr)


def verify_liveness(image_bgr, challenge):
    """
    API-ready liveness challenge verification.
    Accepts an OpenCV BGR image and returns a JSON-compatible dictionary.
    """
    return _verify_liveness(image_bgr, challenge)
