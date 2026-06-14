"""
backend_api.py
--------------
EdgeAuth Offline AI — Backend-Ready Public API Functions.

These three functions are the ONLY interface the backend (Flask / FastAPI /
Lambda handler) needs to call into the AI engine.  They are:

  enroll_face(...)     → capture embedding and persist to storage
  verify_face(...)     → compare a live frame against a stored embedding
  verify_liveness(...) → run the sequential liveness challenge sequence

All functions:
  * Return structured JSON-serialisable dicts
  * Never raise on expected failures (missing face, camera error, etc.)
  * Are fully OFFLINE — no cloud calls whatsoever
  * Integrate with the existing database.py layer (optional; degrades gracefully)

Usage example (from a Flask route):
    from backend_api import enroll_face, verify_face, verify_liveness

    @app.route("/enroll", methods=["POST"])
    def enroll_route():
        result = enroll_face(employee_id="EMP-001", full_name="Priya Nair")
        return jsonify(result)

Author: EdgeAuth Offline Verification Platform
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2  # type: ignore[import-untyped]
import numpy as np  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------

_AI_ENGINE_DIR = Path(__file__).resolve().parent
_REPO_ROOT     = _AI_ENGINE_DIR.parent
_BACKEND_PATH  = _REPO_ROOT / "backend" / "local_device"

for _p in [str(_AI_ENGINE_DIR), str(_BACKEND_PATH)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Internal AI imports
# ---------------------------------------------------------------------------

from face_recognizer import (  # type: ignore[import]
    compare_embeddings,
    extract_embedding,
    extract_embedding_from_file,
    embedding_to_list,
    list_to_embedding,
    match_face,
    DEFAULT_VERIFICATION_THRESHOLD,
)
from liveness_challenge import LivenessChallenge  # type: ignore[import]
from face_verification_engine import (  # type: ignore[import]
    LivenessSequenceState,
    DEFAULT_CHALLENGE_SEQUENCE,
    run_face_recognition,
    _make_result,
)

# ---------------------------------------------------------------------------
# Optional database import
# ---------------------------------------------------------------------------

try:
    from database import (
        initialize_database,
        insert_employee,
        insert_organization,
        list_organizations,
        get_employees_by_organization,
    )
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False

logger = logging.getLogger("backend_api")

# ---------------------------------------------------------------------------
# In-process embedding store (used when DB is unavailable)
#   Key:   employee_id  (str)
#   Value: list[float]  (128-d normalized embedding)
# ---------------------------------------------------------------------------

_EMBEDDING_STORE: Dict[str, List[float]] = {}

# ---------------------------------------------------------------------------
# Enrollment photos directory
# ---------------------------------------------------------------------------

ENROLLMENTS_DIR = _AI_ENGINE_DIR / "enrollments"
ENROLLMENTS_DIR.mkdir(exist_ok=True)


# ===========================================================================
# FUNCTION 1 — enroll_face()
# ===========================================================================


def enroll_face(
    employee_id: Optional[str] = None,
    full_name: str = "Unknown",
    organization_id: str = "00000000-0000-0000-0000-000000000000",
    department: str = "",
    role: str = "",
    frame_bgr: Optional[np.ndarray] = None,
    image_path: Optional[str] = None,
    camera_index: int = 0,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Enroll a face: extract embedding and persist to storage.

    Exactly ONE of frame_bgr / image_path / camera_index must supply the face.
      - frame_bgr:    Use a pre-captured OpenCV BGR numpy array.
      - image_path:   Use a JPEG/PNG file path.
      - camera_index: Open webcam, wait for a stable face, capture and close.

    Args:
        employee_id:     Unique identifier (auto-generated UUID if omitted).
        full_name:       Employee display name.
        organization_id: Organisation UUID for DB association.
        department:      Optional department label.
        role:            Optional role label.
        frame_bgr:       Pre-captured BGR frame (highest priority).
        image_path:      Path to an image file.
        camera_index:    Webcam index used if no frame/image provided.
        db_path:         SQLite DB path.

    Returns:
        {
            "success":     bool,
            "employee_id": str,
            "full_name":   str,
            "embedding_stored": bool,
            "error":       str | None
        }
    """
    employee_id = employee_id or str(uuid.uuid4())

    # ------------------------------------------------------------------
    # 1. Obtain the BGR frame
    # ------------------------------------------------------------------
    bgr_frame: Optional[np.ndarray] = None

    if frame_bgr is not None:
        bgr_frame = frame_bgr

    elif image_path is not None:
        if not os.path.isfile(image_path):
            return _enroll_error(employee_id, full_name, f"Image file not found: {image_path}")
        bgr_frame = cv2.imread(image_path)
        if bgr_frame is None:
            return _enroll_error(employee_id, full_name, f"Could not decode image: {image_path}")

    else:
        # Webcam capture — open, grab one usable face frame, close
        bgr_frame = _capture_face_from_camera(camera_index)
        if bgr_frame is None:
            return _enroll_error(employee_id, full_name, "No face detected via camera")

    # ------------------------------------------------------------------
    # 2. Extract embedding
    # ------------------------------------------------------------------
    embedding = extract_embedding(bgr_frame)

    if embedding is None:
        return _enroll_error(employee_id, full_name, "Could not extract face embedding — no face detected")

    embedding_list = embedding_to_list(embedding)

    # ------------------------------------------------------------------
    # 3. Save enrollment photo
    # ------------------------------------------------------------------
    photo_path = ENROLLMENTS_DIR / f"{employee_id}.jpg"
    try:
        cv2.imwrite(str(photo_path), bgr_frame)
    except Exception as exc:
        logger.warning("Could not save enrollment photo: %s", exc)

    # ------------------------------------------------------------------
    # 4a. Persist to DB if available
    # ------------------------------------------------------------------
    embedding_stored = False
    db_error: Optional[str] = None

    if _DB_AVAILABLE and db_path:
        try:
            initialize_database(db_path=db_path)
            insert_employee(
                organization_id=organization_id,
                full_name=full_name,
                face_embedding=embedding_list,
                department=department,
                role=role,
                embedding_model="face_recognition_dlib_resnet_v1",
                employee_id=employee_id,
                db_path=db_path,
            )
            embedding_stored = True
            logger.info("Enrolled '%s' → DB (id=%s)", full_name, employee_id[:8])
        except Exception as exc:
            db_error = str(exc)
            logger.error("DB insert failed: %s", exc)

    # ------------------------------------------------------------------
    # 4b. Always store in-process (for same-session verify_face calls)
    # ------------------------------------------------------------------
    _EMBEDDING_STORE[employee_id] = embedding_list
    if not embedding_stored:
        embedding_stored = True  # stored in memory

    result: Dict[str, Any] = {
        "success":          True,
        "employee_id":      employee_id,
        "full_name":        full_name,
        "embedding_stored": embedding_stored,
        "error":            db_error,
    }
    return result


def _enroll_error(employee_id: str, full_name: str, message: str) -> Dict[str, Any]:
    logger.error("enroll_face failed: %s", message)
    return {
        "success":          False,
        "employee_id":      employee_id,
        "full_name":        full_name,
        "embedding_stored": False,
        "error":            message,
    }


def _capture_face_from_camera(
    camera_index: int = 0,
    max_wait_seconds: float = 10.0,
) -> Optional[np.ndarray]:
    """
    Open webcam, wait until a face is detected, capture the frame, close camera.

    Returns BGR frame or None on failure.
    """
    from face_recognizer import get_face_bounding_box

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error("Camera %d could not be opened.", camera_index)
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    deadline = time.monotonic() + max_wait_seconds
    captured: Optional[np.ndarray] = None

    try:
        while time.monotonic() < deadline:
            ret, frame = cap.read()
            if not ret:
                continue
            if get_face_bounding_box(frame) is not None:
                captured = frame.copy()
                break
    finally:
        cap.release()

    return captured


# ===========================================================================
# FUNCTION 2 — verify_face()
# ===========================================================================


def verify_face(
    employee_id: str,
    frame_bgr: Optional[np.ndarray] = None,
    image_path: Optional[str] = None,
    camera_index: int = 0,
    enrolled_embedding: Optional[np.ndarray] = None,
    enrolled_embedding_list: Optional[List[float]] = None,
    db_path: Optional[str] = None,
    organization_id: Optional[str] = None,
    threshold: float = DEFAULT_VERIFICATION_THRESHOLD,
) -> Dict[str, Any]:
    """
    Verify whether a face matches the enrolled embedding for employee_id.

    The enrolled embedding is resolved in this priority order:
      1. ``enrolled_embedding``      (numpy array — highest priority, e.g. in-memory)
      2. ``enrolled_embedding_list`` (list of floats — from JSON / database row)
      3. In-process _EMBEDDING_STORE[employee_id] (same-session enrollment)
      4. SQLite database lookup (if db_path and organization_id provided)

    Args:
        employee_id:             Employee identifier to verify against.
        frame_bgr:               Live BGR frame (optional).
        image_path:              Path to a verification image (optional).
        camera_index:            Webcam index if no frame/image given.
        enrolled_embedding:      Pre-loaded numpy embedding.
        enrolled_embedding_list: Embedding as plain Python list.
        db_path:                 SQLite path for DB lookup.
        organization_id:         Org UUID for DB lookup.
        threshold:               Cosine similarity threshold.

    Returns:
        {
            "matched":    bool,
            "similarity": float,
            "employee_id": str,
            "error":      str | None
        }
    """
    # ------------------------------------------------------------------
    # 1. Resolve enrolled embedding
    # ------------------------------------------------------------------
    emb_enrolled: Optional[np.ndarray] = None

    if enrolled_embedding is not None:
        emb_enrolled = np.array(enrolled_embedding, dtype=np.float64)

    elif enrolled_embedding_list is not None:
        emb_enrolled = list_to_embedding(enrolled_embedding_list)

    elif employee_id in _EMBEDDING_STORE:
        emb_enrolled = list_to_embedding(_EMBEDDING_STORE[employee_id])

    elif _DB_AVAILABLE and db_path and organization_id:
        try:
            employees = get_employees_by_organization(organization_id, db_path=db_path)
            for emp in employees:
                if emp.get("employee_id") == employee_id:
                    emb_enrolled = list_to_embedding(emp["face_embedding"])
                    break
        except Exception as exc:
            logger.error("DB lookup failed: %s", exc)
            return _verify_error(employee_id, f"DB lookup failed: {exc}")

    if emb_enrolled is None:
        return _verify_error(employee_id, f"No enrolled embedding found for employee '{employee_id}'")

    # ------------------------------------------------------------------
    # 2. Obtain the live verification frame
    # ------------------------------------------------------------------
    bgr_frame: Optional[np.ndarray] = None

    if frame_bgr is not None:
        bgr_frame = frame_bgr
    elif image_path is not None:
        if not os.path.isfile(image_path):
            return _verify_error(employee_id, f"Image file not found: {image_path}")
        bgr_frame = cv2.imread(image_path)
        if bgr_frame is None:
            return _verify_error(employee_id, f"Could not decode image: {image_path}")
    else:
        bgr_frame = _capture_face_from_camera(camera_index)
        if bgr_frame is None:
            return _verify_error(employee_id, "No face detected via camera")

    # ------------------------------------------------------------------
    # 3. Extract live embedding
    # ------------------------------------------------------------------
    live_embedding = extract_embedding(bgr_frame)
    if live_embedding is None:
        return _verify_error(employee_id, "Could not extract embedding from live frame — no face detected")

    # ------------------------------------------------------------------
    # 4. Compare embeddings
    # ------------------------------------------------------------------
    matched, similarity = match_face(live_embedding, emb_enrolled, threshold)

    return {
        "matched":     matched,
        "similarity":  round(float(similarity), 6),
        "employee_id": employee_id,
        "error":       None,
    }


def _verify_error(employee_id: str, message: str) -> Dict[str, Any]:
    logger.error("verify_face failed: %s", message)
    return {
        "matched":     False,
        "similarity":  0.0,
        "employee_id": employee_id,
        "error":       message,
    }


# ===========================================================================
# FUNCTION 3 — verify_liveness()
# ===========================================================================


def verify_liveness(
    camera_index: int = 0,
    challenges: Sequence[str] = DEFAULT_CHALLENGE_SEQUENCE,
    timeout_per_challenge: float = 15.0,
    headless: bool = False,
) -> Dict[str, Any]:
    """
    Run the full sequential liveness challenge sequence.

    Challenges are presented in order.  Each must be completed before the
    next appears. The function blocks until all challenges pass or the user
    aborts (press 'q' in the display window).

    Args:
        camera_index:          Webcam index.
        challenges:            Ordered list of challenge names.
        timeout_per_challenge: Per-challenge timeout in seconds.
        headless:              If True, no cv2.imshow() window is shown
                               (suitable for server / background use).

    Returns:
        {
            "liveness_verified":   bool,
            "challenges_passed":   list[str],
            "challenges_remaining": list[str],
            "error":               str | None
        }
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return _liveness_error("Camera not available", challenges)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    try:
        liveness_engine = LivenessChallenge()
    except Exception as exc:
        cap.release()
        return _liveness_error(f"Failed to initialize LivenessChallenge: {exc}", challenges)

    state = LivenessSequenceState(challenges)
    # Override per-challenge timeout if caller specified one
    import face_verification_engine as _fve
    orig_timeout = _fve.CHALLENGE_TIMEOUT_SECONDS

    try:
        _fve.CHALLENGE_TIMEOUT_SECONDS = timeout_per_challenge

        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            complete, status_msg = state.process_frame(frame_rgb, liveness_engine)

            if not headless:
                _draw_liveness_hud(frame_bgr, state, status_msg)
                key = cv2.waitKey(5) & 0xFF
                if key == ord('q'):
                    break

            if complete:
                break

    finally:
        _fve.CHALLENGE_TIMEOUT_SECONDS = orig_timeout
        cap.release()
        if not headless:
            cv2.destroyAllWindows()

    passed   = state.get_passed_challenges()
    complete = state.is_complete()
    remaining = [c for c in challenges if c not in passed]

    return {
        "liveness_verified":    complete,
        "challenges_passed":    passed,
        "challenges_remaining": remaining,
        "error":                None if complete else "Liveness not completed",
    }


def _liveness_error(message: str, challenges: Sequence[str]) -> Dict[str, Any]:
    logger.error("verify_liveness failed: %s", message)
    return {
        "liveness_verified":    False,
        "challenges_passed":    [],
        "challenges_remaining": list(challenges),
        "error":                message,
    }


def _draw_liveness_hud(
    frame_bgr: np.ndarray,
    state: LivenessSequenceState,
    status_msg: str,
) -> None:
    """Overlay current challenge progress on the frame and show it."""
    display = cv2.flip(frame_bgr, 1)
    h, w = display.shape[:2]

    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (w, 75), (10, 15, 25), -1)
    cv2.addWeighted(overlay, 0.75, display, 0.25, 0, display)

    cv2.putText(display, "EdgeAuth  |  Liveness Verification",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 118), 2)

    progress = f"Challenge {state.completed_count + 1}/{state.total_count}"
    cv2.putText(display, progress,
                (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

    if state.is_complete():
        cv2.putText(display, "Liveness Verified ✓",
                    (15, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 128), 2)
    elif state.current_challenge_passed():
        cv2.putText(display, "Challenge Passed ✓",
                    (15, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 128), 2)
    else:
        challenge_label = state.get_display_status()
        cv2.putText(display, challenge_label,
                    (15, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 230, 118), 2)

        remaining = max(0, 15.0 - state.elapsed_for_current())
        timer_color = (0, 200, 255) if remaining > 5 else (0, 80, 255)
        cv2.putText(display, f"Time: {remaining:.1f}s",
                    (15, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, timer_color, 2)

        cv2.putText(display, status_msg,
                    (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

    cv2.imshow("EdgeAuth — Liveness Challenge", display)


# ===========================================================================
# Convenience: structured JSON serialisation
# ===========================================================================


def to_json(result: Dict[str, Any], indent: int = 2) -> str:
    """Serialize a result dict to a JSON string."""
    return json.dumps(result, indent=indent, default=str)
