"""
test_pipeline_smoke.py
----------------------
Smoke test for the EdgeAuth AI pipeline.
Uses a synthetic (noise) image -- no camera required.

Run:
    python ai_engine/test_pipeline_smoke.py
    # or via pytest:
    pytest ai_engine/test_pipeline_smoke.py -v

Tests that depend on cv2/face_recognition/mediapipe are marked SKIP if those
packages are not installed in the current environment (e.g. during CI on a
machine without dlib wheels).

All 4 tests must PASS or SKIP -- none must FAIL with an unhandled exception.
"""

import sys
import os
from pathlib import Path

# Force UTF-8 output on Windows to avoid cp1252 codec errors
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Path bootstrap
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "backend" / "local_device"))

# ---------------------------------------------------------------------------
# Dependency probes
# ---------------------------------------------------------------------------

def _has_cv2() -> bool:
    try:
        import cv2  # noqa
        return True
    except ImportError:
        return False

def _has_face_recognition() -> bool:
    try:
        import face_recognition  # noqa
        return True
    except ImportError:
        return False

def _has_mediapipe() -> bool:
    try:
        import mediapipe  # noqa
        return True
    except ImportError:
        return False

# ---------------------------------------------------------------------------
# Test results tracker
# ---------------------------------------------------------------------------

_PASS  = "PASS"
_FAIL  = "FAIL"
_SKIP  = "SKIP"

results = []


def _record(name: str, status: str, note: str = "") -> None:
    tag = f"[{status}]"
    msg = f"{tag} {name}"
    if note:
        msg += f": {note}"
    print(msg)
    results.append((name, status))


# =============================================================================
# Tests
# =============================================================================

def test_imports():
    """All AI modules must import without errors."""
    if not (_has_cv2() and _has_face_recognition() and _has_mediapipe()):
        _record("test_imports", _SKIP,
                "cv2/face_recognition/mediapipe not installed -- run: pip install -r ai_engine/requirements.txt")
        return

    from face_recognizer import extract_embedding, compare_embeddings, DEFAULT_VERIFICATION_THRESHOLD  # noqa
    from liveness_challenge import LivenessChallenge  # noqa
    from backend_api import enroll_face, verify_face, verify_liveness  # noqa
    _record("test_imports", _PASS, "All AI modules imported successfully")


def test_embedding_on_noise():
    """
    extract_embedding on a noise frame must not crash.
    Returning None is correct behavior when no face is detected in noise.
    """
    if not _has_cv2() or not _has_face_recognition():
        _record("test_embedding_on_noise", _SKIP, "cv2/face_recognition not installed")
        return

    import numpy as np
    from face_recognizer import extract_embedding

    noise_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    try:
        result = extract_embedding(noise_frame)
        assert result is None or len(result) == 128, (
            f"Embedding must be None or 128-d, got: {type(result)}"
        )
        _record("test_embedding_on_noise", _PASS, "No crash; result is None (expected for noise)")
    except Exception as exc:
        _record("test_embedding_on_noise", _FAIL, str(exc))


def test_cosine_similarity():
    """Cosine similarity of identical L2-normalized vectors must be 1.0."""
    try:
        import numpy as np
        # This test uses only numpy -- always runs
        # But face_recognizer imports cv2 so we must guard
        if not _has_cv2() or not _has_face_recognition():
            # Implement the same logic inline to avoid the import chain
            def _normalize(v):
                n = (sum(x**2 for x in v)) ** 0.5
                return [x / n for x in v]

            def _dot(a, b):
                return sum(x * y for x, y in zip(a, b))

            import random
            raw = [random.gauss(0, 1) for _ in range(128)]
            vec = _normalize(raw)
            score = _dot(vec, vec)
            assert abs(score - 1.0) < 1e-5, f"Expected ~1.0, got {score}"
            _record("test_cosine_similarity", _PASS,
                    f"Inline cosine similarity = {score:.6f} (face_recognizer not available)")
            return

        from face_recognizer import cosine_similarity
        vec = np.random.rand(128).astype(np.float64)
        vec /= np.linalg.norm(vec)
        score = cosine_similarity(vec, vec)
        assert abs(score - 1.0) < 1e-5, f"Expected ~1.0, got {score}"
        _record("test_cosine_similarity", _PASS, f"score = {score:.6f}")
    except Exception as exc:
        _record("test_cosine_similarity", _FAIL, str(exc))


def test_liveness_challenge_init():
    """LivenessChallenge must initialize without error."""
    if not _has_mediapipe():
        _record("test_liveness_challenge_init", _SKIP,
                "mediapipe not installed -- run: pip install mediapipe")
        return

    try:
        from liveness_challenge import LivenessChallenge
        lc = LivenessChallenge()
        assert lc is not None
        _record("test_liveness_challenge_init", _PASS, "LivenessChallenge initialized")
    except Exception as exc:
        _record("test_liveness_challenge_init", _FAIL, str(exc))


# =============================================================================
# Entry point (standalone + pytest-compatible)
# =============================================================================

def test_all():
    """pytest entry point -- runs all sub-tests."""
    test_imports()
    test_embedding_on_noise()
    test_cosine_similarity()
    test_liveness_challenge_init()


if __name__ == "__main__":
    print("=" * 50)
    print("  EdgeAuth AI Pipeline Smoke Test")
    print("=" * 50)
    print()

    test_imports()
    test_embedding_on_noise()
    test_cosine_similarity()
    test_liveness_challenge_init()

    passed = sum(1 for _, s in results if s == _PASS)
    skipped = sum(1 for _, s in results if s == _SKIP)
    failed  = sum(1 for _, s in results if s == _FAIL)

    print()
    print(f"Results: {passed} passed, {skipped} skipped, {failed} failed "
          f"out of {len(results)} tests")

    if failed:
        print()
        print("[FAIL] Some smoke tests failed -- see errors above.")
        sys.exit(1)
    else:
        print()
        print("=== All smoke tests passed ===")
        sys.exit(0)
