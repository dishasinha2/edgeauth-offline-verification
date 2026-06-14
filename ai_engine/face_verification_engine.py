"""
face_verification_engine.py
---------------------------
EdgeAuth Offline AI Verification Engine — Single Orchestrator Module.

Pipeline (in order):
  1. Face Detection        → is a face present?
  2. Face Mesh             → facial landmarks via MediaPipe
  3. Blink Verification    → anti-spoofing blink challenge
  4. Head Pose Verification → anti-spoofing head-turn challenge
  5. Liveness Challenge    → sequential multi-challenge liveness gate
  6. Face Recognition      → cosine similarity match against enrolled embedding
  7. Final Decision        → structured JSON verdict

This module is the SINGLE integration point for all AI capabilities.
All sub-modules are reused; NO logic is duplicated.

Public API:
  run_verification_pipeline(frame_bgr, enrolled_embedding, ...)
  run_liveness_sequence(frames_iter, challenges, ...)

Backend-ready functions (see backend_api.py):
  enroll_face()   — generate & store embedding
  verify_face()   — compare against stored embedding
  verify_liveness() — run liveness challenge sequence

Output format:
  {
      "face_detected":      bool,
      "blink_verified":     bool,
      "head_pose_verified": bool,
      "liveness_verified":  bool,
      "face_matched":       bool,
      "similarity_score":   float,
      "verification_passed": bool
  }

Author: EdgeAuth Offline Verification Platform
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import cv2  # type: ignore[import-untyped]
import numpy as np  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# Path bootstrap — make sibling ai_engine modules importable when run from
# outside the ai_engine directory.
# ---------------------------------------------------------------------------

_AI_ENGINE_DIR = Path(__file__).resolve().parent
if str(_AI_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(_AI_ENGINE_DIR))

# ---------------------------------------------------------------------------
# Import stable, production-ready sub-modules (DO NOT modify them)
# ---------------------------------------------------------------------------

from liveness_challenge import LivenessChallenge  # type: ignore[import]  # noqa: E402
from face_recognizer import (  # type: ignore[import]  # noqa: E402
    DEFAULT_VERIFICATION_THRESHOLD,
    compare_embeddings,
    extract_embedding,
    get_face_bounding_box,
    match_face,
    embedding_to_list,
    list_to_embedding,
)

logger = logging.getLogger("face_verification_engine")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default ordered challenge sequence for liveness verification
DEFAULT_CHALLENGE_SEQUENCE: List[str] = [
    "Blink Twice",
    "Turn Head Left",
    "Turn Head Right",
]

# Frames to collect before declaring a blink counted (debounce)
BLINK_DEBOUNCE_FRAMES = 15

# Per-challenge timeout in seconds
CHALLENGE_TIMEOUT_SECONDS = 15.0

# ---------------------------------------------------------------------------
# Result Helpers
# ---------------------------------------------------------------------------


def _make_result(
    face_detected: bool = False,
    blink_verified: bool = False,
    head_pose_verified: bool = False,
    liveness_verified: bool = False,
    face_matched: bool = False,
    similarity_score: float = 0.0,
    verification_passed: bool = False,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the canonical pipeline result dictionary."""
    result: Dict[str, Any] = {
        "face_detected": face_detected,
        "blink_verified": blink_verified,
        "head_pose_verified": head_pose_verified,
        "liveness_verified": liveness_verified,
        "face_matched": face_matched,
        "similarity_score": round(float(similarity_score), 6),
        "verification_passed": verification_passed,
    }
    if error:
        result["error"] = error
    return result


# ---------------------------------------------------------------------------
# Step 1 — Face Detection
# ---------------------------------------------------------------------------


def check_face_detected(frame_bgr: np.ndarray) -> bool:
    """
    Step 1: Check whether at least one face is visible in the frame.

    Args:
        frame_bgr: OpenCV BGR image.

    Returns:
        True if a face bounding box is found, False otherwise.
    """
    try:
        box = get_face_bounding_box(frame_bgr)
        return box is not None
    except Exception as exc:
        logger.warning("check_face_detected error: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Step 2 — Face Mesh (via LivenessChallenge internal MediaPipe detector)
# ---------------------------------------------------------------------------


def check_face_mesh(
    frame_rgb: np.ndarray,
    liveness_engine: LivenessChallenge,
) -> bool:
    """
    Step 2: Confirm face mesh landmarks are extractable via MediaPipe.

    Re-uses the LivenessChallenge detector so no extra model is loaded.

    Args:
        frame_rgb:       RGB image frame.
        liveness_engine: Initialized LivenessChallenge instance.

    Returns:
        True if MediaPipe returns valid facial landmarks.
    """
    try:
        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = liveness_engine.detector.detect(mp_image)
        return bool(result.face_landmarks)
    except Exception as exc:
        logger.warning("check_face_mesh error: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Step 3 — Blink Verification (single blink anti-spoof check)
# ---------------------------------------------------------------------------


def check_blink_verified(
    frame_rgb: np.ndarray,
    liveness_engine: LivenessChallenge,
) -> bool:
    """
    Step 3: Verify that at least one blink is detectable (EAR threshold).

    This is a single-frame probe — to count blinks over time use the full
    liveness sequence runner.

    Args:
        frame_rgb:       RGB image frame during a blink moment.
        liveness_engine: Initialized LivenessChallenge instance.

    Returns:
        True if EAR is below blink threshold.
    """
    try:
        passed, _ = liveness_engine.check_liveness(frame_rgb, "Blink Twice")
        return passed
    except Exception as exc:
        logger.warning("check_blink_verified error: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Step 4 — Head Pose Verification
# ---------------------------------------------------------------------------


def check_head_pose_verified(
    frame_rgb: np.ndarray,
    liveness_engine: LivenessChallenge,
    direction: str = "Turn Head Left",
) -> bool:
    """
    Step 4: Verify that a head turn can be detected.

    Args:
        frame_rgb:       RGB frame showing a head turn.
        liveness_engine: Initialized LivenessChallenge instance.
        direction:       "Turn Head Left" or "Turn Head Right".

    Returns:
        True if the head turn matches the requested direction.
    """
    try:
        passed, _ = liveness_engine.check_liveness(frame_rgb, direction)
        return passed
    except Exception as exc:
        logger.warning("check_head_pose_verified error: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Step 5 — Sequential Liveness Challenge (improved flow with state tracking)
# ---------------------------------------------------------------------------


class LivenessSequenceState:
    """
    Tracks progress through a sequential liveness challenge sequence.

    Usage:
        state = LivenessSequenceState(["Blink Twice", "Turn Head Left"])
        while not state.is_complete():
            passed, status_msg = state.process_frame(frame_rgb, liveness_engine)
            if state.current_challenge_passed():
                print("Challenge Passed!")
    """

    def __init__(self, challenges: Sequence[str] = DEFAULT_CHALLENGE_SEQUENCE):
        self.challenges: List[str] = list(challenges)
        self._index: int = 0
        self._blink_count: int = 0
        self._blink_cooldown: int = 0
        self._challenge_start: float = time.monotonic()
        self._passed_challenges: List[str] = []
        self._last_challenge_passed: bool = False
        self._pass_display_frames: int = 0  # frames left to display "Challenge Passed"
        self._PASS_DISPLAY_DURATION = 30    # ~1 second at 30fps

    # ------------------------------------------------------------------
    @property
    def current_challenge(self) -> Optional[str]:
        """The challenge currently being presented, or None if complete."""
        if self._index < len(self.challenges):
            return self.challenges[self._index]
        return None

    @property
    def completed_count(self) -> int:
        return len(self._passed_challenges)

    @property
    def total_count(self) -> int:
        return len(self.challenges)

    def is_complete(self) -> bool:
        return self._index >= len(self.challenges)

    def current_challenge_passed(self) -> bool:
        """Returns True for the frame window immediately after a challenge passes."""
        return self._last_challenge_passed

    def get_passed_challenges(self) -> List[str]:
        return list(self._passed_challenges)

    def elapsed_for_current(self) -> float:
        return time.monotonic() - self._challenge_start

    def timed_out(self) -> bool:
        return self.elapsed_for_current() > CHALLENGE_TIMEOUT_SECONDS

    # ------------------------------------------------------------------
    def _advance(self) -> None:
        """Move to the next challenge."""
        if self.current_challenge:
            self._passed_challenges.append(self.current_challenge)
        self._index += 1
        self._blink_count = 0
        self._blink_cooldown = 0
        self._challenge_start = time.monotonic()
        self._last_challenge_passed = True
        self._pass_display_frames = self._PASS_DISPLAY_DURATION

    def reset_current(self) -> None:
        """Reset the current challenge timer (called on timeout)."""
        self._blink_count = 0
        self._blink_cooldown = 0
        self._challenge_start = time.monotonic()
        self._last_challenge_passed = False

    # ------------------------------------------------------------------
    def process_frame(
        self,
        frame_rgb: np.ndarray,
        liveness_engine: LivenessChallenge,
    ) -> Tuple[bool, str]:
        """
        Process one camera frame against the current challenge.

        Returns:
            (challenge_sequence_complete, status_message)
        """
        # Tick down the "Challenge Passed" display window
        if self._pass_display_frames > 0:
            self._pass_display_frames -= 1
            if self._pass_display_frames == 0:
                self._last_challenge_passed = False

        if self.is_complete():
            return True, "Liveness Verified"

        challenge = self.current_challenge

        # Per-challenge timeout → reset current (no penalty, no skip)
        if self.timed_out():
            logger.debug("Challenge '%s' timed out — resetting.", challenge)
            self.reset_current()
            return False, f"Timeout — retry: {challenge}"

        # Poll blink cooldown
        if self._blink_cooldown > 0:
            self._blink_cooldown -= 1

        try:
            passed, raw_status = liveness_engine.check_liveness(frame_rgb, challenge)
        except Exception as exc:
            logger.warning("liveness.check_liveness error: %s", exc)
            return False, "Sensor error"

        if challenge == "Blink Twice":
            if passed and self._blink_cooldown == 0:
                self._blink_count += 1
                self._blink_cooldown = BLINK_DEBOUNCE_FRAMES
                logger.debug("Blink %d/2 detected.", self._blink_count)
            if self._blink_count >= 2:
                self._advance()
                return self.is_complete(), "Challenge Passed ✓"
            return False, f"Blink {self._blink_count}/2"

        else:
            if passed:
                self._advance()
                return self.is_complete(), "Challenge Passed ✓"
            return False, raw_status

    # ------------------------------------------------------------------
    def get_display_status(self) -> str:
        """Human-readable status string for HUD overlay."""
        if self.is_complete():
            return "Liveness Verified ✓"
        if self._last_challenge_passed:
            return "Challenge Passed ✓"
        challenge = self.current_challenge
        if challenge == "Blink Twice":
            return f"Blink Twice ({self._blink_count}/2)"
        return challenge or "Complete"


# ---------------------------------------------------------------------------
# Step 6 — Face Recognition (compare_embeddings / match_face wrappers)
# ---------------------------------------------------------------------------
# Note: compare_embeddings() and match_face() are defined in face_recognizer.py
# and imported above. They are re-exported here for convenience.


def run_face_recognition(
    frame_bgr: np.ndarray,
    enrolled_embedding: np.ndarray,
    threshold: float = DEFAULT_VERIFICATION_THRESHOLD,
) -> Dict[str, Any]:
    """
    Step 6: Extract embedding from live frame and compare against enrolled.

    Args:
        frame_bgr:          Live camera frame (BGR).
        enrolled_embedding: 128-d numpy float64 vector (L2-normalized).
        threshold:          Cosine similarity threshold for a positive match.

    Returns:
        {
            "face_detected": bool,
            "face_matched":  bool,
            "similarity":    float,
            "error":         str | None
        }
    """
    live_embedding = extract_embedding(frame_bgr)

    if live_embedding is None:
        return {
            "face_detected": False,
            "face_matched": False,
            "similarity": 0.0,
            "error": "No face detected in frame",
        }

    matched, similarity = match_face(live_embedding, enrolled_embedding, threshold)

    return {
        "face_detected": True,
        "face_matched": matched,
        "similarity": round(float(similarity), 6),
        "error": None,
    }


# ---------------------------------------------------------------------------
# Step 7 — Full Pipeline (single-frame mode for programmatic use)
# ---------------------------------------------------------------------------


def run_verification_pipeline(
    frame_bgr: np.ndarray,
    enrolled_embedding: np.ndarray,
    liveness_sequence_state: LivenessSequenceState,
    liveness_engine: Optional[LivenessChallenge] = None,
    threshold: float = DEFAULT_VERIFICATION_THRESHOLD,
) -> Dict[str, Any]:
    """
    Run the complete EdgeAuth AI verification pipeline on a single frame.

    This is the MASTER function that orchestrates all steps in order:
      Face Detection → Face Mesh → Blink → Head Pose → Liveness → Recognition

    The liveness steps rely on cumulative state tracked in
    ``liveness_sequence_state`` across multiple frames.

    Args:
        frame_bgr:               OpenCV BGR frame.
        enrolled_embedding:      Enrolled 128-d face vector.
        liveness_sequence_state: Persistent state for challenge progress.
        liveness_engine:         Pre-initialized LivenessChallenge (created
                                 if None; avoid re-creating each frame).
        threshold:               Cosine similarity threshold.

    Returns:
        Canonical result dict (see module docstring).
    """
    if liveness_engine is None:
        liveness_engine = LivenessChallenge()

    # ------------------------------------------------------------------
    # Step 1 — Face Detection
    # ------------------------------------------------------------------
    face_detected = check_face_detected(frame_bgr)
    if not face_detected:
        return _make_result(error="No face detected")

    # ------------------------------------------------------------------
    # Step 2 — Face Mesh
    # ------------------------------------------------------------------
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    face_mesh_ok = check_face_mesh(frame_rgb, liveness_engine)
    if not face_mesh_ok:
        return _make_result(face_detected=True, error="Face mesh landmarks unavailable")

    # ------------------------------------------------------------------
    # Steps 3-5 — Liveness Sequence
    # ------------------------------------------------------------------
    liveness_complete, liveness_status = liveness_sequence_state.process_frame(
        frame_rgb, liveness_engine
    )

    blink_verified     = "Blink Twice"      in liveness_sequence_state.get_passed_challenges()
    head_pose_verified = (
        "Turn Head Left"  in liveness_sequence_state.get_passed_challenges()
        or "Turn Head Right" in liveness_sequence_state.get_passed_challenges()
    )
    liveness_verified  = liveness_complete

    if not liveness_verified:
        return _make_result(
            face_detected=True,
            blink_verified=blink_verified,
            head_pose_verified=head_pose_verified,
            liveness_verified=False,
            error=f"Liveness in progress: {liveness_status}",
        )

    # ------------------------------------------------------------------
    # Step 6 — Face Recognition
    # ------------------------------------------------------------------
    recognition = run_face_recognition(frame_bgr, enrolled_embedding, threshold)
    face_matched     = recognition["face_matched"]
    similarity_score = recognition["similarity"]

    # ------------------------------------------------------------------
    # Step 7 — Final Decision
    # ------------------------------------------------------------------
    verification_passed = liveness_verified and face_matched

    return _make_result(
        face_detected=True,
        blink_verified=blink_verified,
        head_pose_verified=head_pose_verified,
        liveness_verified=liveness_verified,
        face_matched=face_matched,
        similarity_score=similarity_score,
        verification_passed=verification_passed,
    )


# ---------------------------------------------------------------------------
# Convenience runner for headless / test use
# ---------------------------------------------------------------------------


def headless_pipeline_test(
    enrolled_embedding: np.ndarray,
    camera_index: int = 0,
    threshold: float = DEFAULT_VERIFICATION_THRESHOLD,
    challenges: Sequence[str] = DEFAULT_CHALLENGE_SEQUENCE,
) -> Dict[str, Any]:
    """
    Run the full pipeline against a live webcam feed until completion.

    Blocks until all liveness challenges pass and face recognition runs,
    then returns the final result dict. Intended for CLI / integration tests.

    Args:
        enrolled_embedding: Pre-enrolled 128-d embedding.
        camera_index:       OpenCV camera index.
        threshold:          Cosine similarity threshold.
        challenges:         Ordered list of liveness challenges.

    Returns:
        Final pipeline result dict.
    """
    liveness_engine = LivenessChallenge()
    state           = LivenessSequenceState(challenges)
    last_result     = _make_result(error="Pipeline not started")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return _make_result(error="Camera not available")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    logger.info("Headless pipeline started. Challenges: %s", challenges)

    try:
        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret:
                continue

            last_result = run_verification_pipeline(
                frame_bgr,
                enrolled_embedding,
                state,
                liveness_engine,
                threshold,
            )

            if last_result["verification_passed"] or last_result.get("face_matched") is True:
                break

            # Abort early if liveness is complete but no enrolled embedding provided
            if last_result["liveness_verified"] and enrolled_embedding is None:
                logger.warning("Liveness verified but no enrolled embedding to compare.")
                break

    finally:
        cap.release()

    return last_result
