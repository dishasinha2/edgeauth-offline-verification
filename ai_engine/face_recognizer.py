"""
face_recognizer.py
------------------
Offline Face Recognition Engine for EdgeAuth.

Uses the `face_recognition` library (dlib-backed ResNet pretrained model)
to extract 128-dimensional face embeddings from images. All inference runs
fully OFFLINE — no cloud API calls, no internet required.

Pipeline:
  1. Detect face bounding boxes in a BGR frame.
  2. Extract 128-d embedding for each detected face.
  3. Compare embeddings using cosine similarity.
  4. Threshold comparison → Verified / Not Verified.

Author: EdgeAuth Offline Verification Platform
"""

import logging
import math
import os
from typing import List, Optional, Tuple

import cv2  # type: ignore[import-untyped]
import face_recognition  # type: ignore[import-untyped]
import numpy as np  # type: ignore[import-untyped]

logger = logging.getLogger("face_recognizer")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default cosine similarity threshold for a positive match.
# 0.75 is a conservative cutoff; adjust per-org via DB liveness_threshold field.
DEFAULT_VERIFICATION_THRESHOLD = 0.75

# face_recognition uses HOG (fast, CPU-friendly) or CNN (accurate, slower).
# For edge devices, HOG is preferred. Set to "cnn" for GPU-enabled devices.
FACE_DETECTION_MODEL = os.environ.get("EDGEAUTH_FACE_MODEL", "hog")


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------


def _normalize(vector: np.ndarray) -> np.ndarray:
    """L2-normalize a vector so cosine similarity = dot product."""
    norm = np.linalg.norm(vector)
    if norm < 1e-10:
        return vector
    return vector / norm


def extract_embedding(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract a 128-d face embedding from a BGR image frame.

    Args:
        frame_bgr: OpenCV BGR image (H x W x 3 uint8).

    Returns:
        A normalized 128-d numpy float64 array, or None if no face detected.

    Notes:
        - Only the FIRST detected face is used (single-person verification).
        - The embedding is L2-normalized so dot product equals cosine similarity.
        - `face_recognition` uses a pretrained ResNet model embedded in dlib.
    """
    # face_recognition expects RGB, OpenCV gives BGR
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Detect face locations using HOG (fast) or CNN (accurate)
    face_locations = face_recognition.face_locations(
        frame_rgb, model=FACE_DETECTION_MODEL
    )

    if not face_locations:
        logger.debug("extract_embedding: No face detected in frame.")
        return None

    # Use only the first face (single-user verification context)
    face_loc = [face_locations[0]]

    # Extract 128-d embedding
    encodings = face_recognition.face_encodings(frame_rgb, known_face_locations=face_loc)

    if not encodings:
        logger.debug("extract_embedding: Could not encode face.")
        return None

    embedding = np.array(encodings[0], dtype=np.float64)
    return _normalize(embedding)


def extract_embedding_from_file(image_path: str) -> Optional[np.ndarray]:
    """
    Extract a 128-d face embedding directly from an image file path.

    Args:
        image_path: Absolute or relative path to a JPEG/PNG image.

    Returns:
        Normalized 128-d embedding or None if no face found.
    """
    if not os.path.isfile(image_path):
        logger.error("Image file not found: %s", image_path)
        return None

    image = face_recognition.load_image_file(image_path)  # Returns RGB
    face_locations = face_recognition.face_locations(image, model=FACE_DETECTION_MODEL)

    if not face_locations:
        logger.warning("No face found in image: %s", image_path)
        return None

    encodings = face_recognition.face_encodings(image, known_face_locations=[face_locations[0]])

    if not encodings:
        return None

    return _normalize(np.array(encodings[0], dtype=np.float64))


def cosine_similarity(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two L2-normalized embedding vectors.

    Since both vectors are pre-normalized, this is equivalent to the dot product.

    Args:
        embedding_a: First 128-d normalized embedding.
        embedding_b: Second 128-d normalized embedding.

    Returns:
        Similarity score in [-1.0, 1.0]. Higher = more similar.
        Identical faces → ~1.0; different people → ~0.0 to 0.5.
    """
    a = np.array(embedding_a, dtype=np.float64)
    b = np.array(embedding_b, dtype=np.float64)

    dot = float(np.dot(a, b))
    # Clamp to [-1, 1] to guard against floating point drift
    return max(-1.0, min(1.0, dot))


def verify(
    embedding_a: np.ndarray,
    embedding_b: np.ndarray,
    threshold: float = DEFAULT_VERIFICATION_THRESHOLD,
) -> Tuple[bool, float]:
    """
    Verify whether two face embeddings belong to the same person.

    Args:
        embedding_a: Query embedding (from live camera frame).
        embedding_b: Enrolled embedding (from SQLite database).
        threshold:   Minimum cosine similarity to accept as a match.

    Returns:
        (is_match: bool, similarity_score: float)
    """
    score = cosine_similarity(embedding_a, embedding_b)
    return score >= threshold, round(score, 6)


def compare_embeddings(
    embedding_a: np.ndarray,
    embedding_b: np.ndarray,
    threshold: float = DEFAULT_VERIFICATION_THRESHOLD,
) -> Tuple[bool, float]:
    """
    Public API alias for verify().

    Compare two face embeddings and return a match decision plus score.

    Args:
        embedding_a: Query embedding (live frame).
        embedding_b: Enrolled embedding (from database or storage).
        threshold:   Cosine similarity threshold for a positive match.

    Returns:
        (matched: bool, similarity_score: float)

    Example output::

        {"matched": True, "similarity": 0.91}
    """
    return verify(embedding_a, embedding_b, threshold)


def match_face(
    live_embedding: np.ndarray,
    enrolled_embedding: np.ndarray,
    threshold: float = DEFAULT_VERIFICATION_THRESHOLD,
) -> Tuple[bool, float]:
    """
    Determine whether a live face matches an enrolled face.

    This is the primary entry point for single-user 1-to-1 verification.
    Both embeddings must be L2-normalized 128-d vectors (as returned by
    ``extract_embedding()``).

    Args:
        live_embedding:     128-d embedding from the current camera frame.
        enrolled_embedding: 128-d embedding stored during enrollment.
        threshold:          Minimum cosine similarity to accept as a match.
                            Default is 0.75 (conservative; raise to 0.80 for
                            higher security sites).

    Returns:
        (matched: bool, similarity_score: float)
        ``matched`` is True when score >= threshold.
        ``similarity_score`` is rounded to 6 decimal places.

    Example::

        emb_live     = extract_embedding(camera_frame)
        emb_enrolled = list_to_embedding(stored_list)
        matched, score = match_face(emb_live, emb_enrolled)
        # → (True, 0.912345)
    """
    return verify(live_embedding, enrolled_embedding, threshold)


def find_best_match(
    query_embedding: np.ndarray,
    enrolled_employees: List[dict],
    threshold: float = DEFAULT_VERIFICATION_THRESHOLD,
) -> Tuple[Optional[dict], float]:
    """
    Find the best-matching employee from a list of enrolled employees.

    Args:
        query_embedding:    128-d normalized embedding from the live frame.
        enrolled_employees: List of employee dicts from database.get_employees_by_organization().
                            Each dict must have 'face_embedding' (list of floats) and 'employee_id'.
        threshold:          Minimum score for a positive match.

    Returns:
        (best_employee: dict | None, best_score: float)
        Returns (None, best_score) if no employee meets the threshold.
    """
    best_employee = None
    best_score = -1.0

    for employee in enrolled_employees:
        enrolled_emb = np.array(employee["face_embedding"], dtype=np.float64)
        enrolled_emb = _normalize(enrolled_emb)
        score = cosine_similarity(query_embedding, enrolled_emb)

        if score > best_score:
            best_score = score
            best_employee = employee

    if best_score >= threshold:
        logger.info(
            "Best match: '%s' (score=%.4f, threshold=%.4f)",
            best_employee.get("full_name", "?"),
            best_score,
            threshold,
        )
        return best_employee, round(best_score, 6)

    logger.info(
        "No match found. Best score=%.4f below threshold=%.4f.", best_score, threshold
    )
    return None, round(best_score, 6)


def get_face_bounding_box(frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Get the bounding box of the first detected face in a BGR frame.

    Returns:
        (top, right, bottom, left) in pixel coordinates, or None.
        Note: face_recognition returns boxes as (top, right, bottom, left).
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(frame_rgb, model=FACE_DETECTION_MODEL)
    return locations[0] if locations else None


def crop_face(frame_bgr: np.ndarray, padding: int = 20) -> Optional[np.ndarray]:
    """
    Crop the face region from a BGR frame with optional padding.

    Args:
        frame_bgr: Full resolution camera frame.
        padding:   Extra pixels added around the face box.

    Returns:
        Cropped BGR face image, or None if no face detected.
    """
    box = get_face_bounding_box(frame_bgr)
    if box is None:
        return None

    top, right, bottom, left = box
    h, w = frame_bgr.shape[:2]

    # Apply padding clamped to image dimensions
    top    = max(0, top    - padding)
    left   = max(0, left   - padding)
    bottom = min(h, bottom + padding)
    right  = min(w, right  + padding)

    return frame_bgr[top:bottom, left:right]


# ---------------------------------------------------------------------------
# Embedding Storage Helpers (list ↔ numpy)
# ---------------------------------------------------------------------------


def embedding_to_list(embedding: np.ndarray) -> List[float]:
    """Convert numpy embedding to a plain Python list for SQLite JSON storage."""
    return [round(float(v), 8) for v in embedding]


def list_to_embedding(embedding_list: List[float]) -> np.ndarray:
    """Convert stored list back to a normalized numpy embedding."""
    return _normalize(np.array(embedding_list, dtype=np.float64))


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    print("\n=== face_recognizer.py — Self Test ===")
    print("Opening webcam (press 'q' to quit)...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        sys.exit(1)

    enrolled_embedding = None

    print("Press 'e' to ENROLL current frame, 'v' to VERIFY, 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        display = cv2.flip(frame, 1)

        # Draw face box
        box = get_face_bounding_box(frame)
        if box:
            top, right, bottom, left = box
            h, w = frame.shape[:2]
            # Flip coordinates for the mirrored display
            flipped_left  = w - right
            flipped_right = w - left
            cv2.rectangle(display, (flipped_left, top), (flipped_right, bottom), (0, 255, 0), 2)

        status_text = "No face" if not box else "Face detected"
        cv2.putText(display, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        if enrolled_embedding is not None:
            cv2.putText(display, "Enrolled: YES", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("EdgeAuth — Face Recognizer Test", display)

        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e'):
            emb = extract_embedding(frame)
            if emb is not None:
                enrolled_embedding = emb
                print("[OK] Enrollment embedding extracted!")
            else:
                print("[WARN] No face found — try again.")
        elif key == ord('v'):
            if enrolled_embedding is None:
                print("[WARN] Enroll a face first with 'e'.")
            else:
                emb = extract_embedding(frame)
                if emb is not None:
                    matched, score = verify(emb, enrolled_embedding)
                    status = "VERIFIED ✓" if matched else "REJECTED ✗"
                    print(f"[{status}] Cosine Similarity: {score:.4f}")
                else:
                    print("[WARN] No face detected for verification.")

    cap.release()
    cv2.destroyAllWindows()
