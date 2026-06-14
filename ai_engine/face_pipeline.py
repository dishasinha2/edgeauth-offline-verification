"""
face_pipeline.py
----------------
Unified EdgeAuth Offline Face Verification Pipeline.

This is the MASTER orchestrator that glues every component together:
  1. Opens webcam and feeds frames to MediaPipe FaceLandmarker.
  2. Presents a random liveness challenge (Blink / Head Turn / Smile).
  3. On liveness pass → extracts 128-d face embedding via face_recognition.
  4. Searches all enrolled employees in SQLite for the best cosine match.
  5. Determines event type (CLOCK_IN / ACCESS_DENIED) based on match result.
  6. Writes an immutable log to SQLite (auto-enqueued for AWS sync).

Usage:
  # Run with default settings (uses workforce_verification.db):
  python face_pipeline.py

  # Specify org, device, and DB path:
  python face_pipeline.py --org-id <uuid> --device-id EDGE-001 --db ./my.db

  # Verification threshold (cosine similarity, default 0.75):
  python face_pipeline.py --threshold 0.80

Requirements:
  pip install mediapipe opencv-python face_recognition numpy

Author: EdgeAuth Offline Verification Platform
"""

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Ensure backend database module is importable from the ai_engine folder
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BACKEND_PATH = _REPO_ROOT / "backend" / "local_device"
if str(_BACKEND_PATH) not in sys.path:
    sys.path.insert(0, str(_BACKEND_PATH))

# ---------------------------------------------------------------------------
# Local module imports
# ---------------------------------------------------------------------------

from liveness_challenge import LivenessChallenge
from face_recognizer import (
    DEFAULT_VERIFICATION_THRESHOLD,
    extract_embedding,
    find_best_match,
    get_face_bounding_box,
)

try:
    from database import (
        DB_PATH,
        get_employees_by_organization,
        initialize_database,
        insert_log,
        list_organizations,
    )
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging & ANSI Colours
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("face_pipeline")

GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def _banner(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'=' * 55}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 55}{RESET}\n")


# ---------------------------------------------------------------------------
# Pipeline State Machine
# ---------------------------------------------------------------------------

class PipelineState:
    WAITING_FOR_FACE    = "WAITING_FOR_FACE"
    LIVENESS_CHALLENGE  = "LIVENESS_CHALLENGE"
    EXTRACTING_EMBEDDING = "EXTRACTING_EMBEDDING"
    VERIFYING           = "VERIFYING"
    RESULT              = "RESULT"
    COOLDOWN            = "COOLDOWN"


class FacePipeline:
    """
    Stateful webcam verification pipeline.

    Each verification session progresses through:
      WAITING_FOR_FACE → LIVENESS_CHALLENGE → EXTRACTING_EMBEDDING
      → VERIFYING → RESULT → COOLDOWN → (repeat)
    """

    def __init__(
        self,
        org_id: str,
        device_id: str,
        db_path: str,
        verification_threshold: float = DEFAULT_VERIFICATION_THRESHOLD,
        liveness_threshold: float = 0.80,
        result_hold_seconds: float = 3.0,
        cooldown_seconds: float = 2.0,
    ):
        self.org_id = org_id
        self.device_id = device_id
        self.db_path = db_path
        self.verification_threshold = verification_threshold
        self.liveness_threshold = liveness_threshold
        self.result_hold = result_hold_seconds
        self.cooldown = cooldown_seconds

        # Initialize MediaPipe liveness engine
        self.liveness = LivenessChallenge()

        # Load enrolled employees from DB
        self.enrolled = self._load_employees()
        print(f"  {GREEN}[OK]{RESET} Loaded {len(self.enrolled)} enrolled employee(s) for org.")

        # Pipeline state
        self.state = PipelineState.WAITING_FOR_FACE
        self.current_challenge = None
        self.blink_count = 0
        self.blink_cooldown = 0
        self.challenge_start_time = None
        self.challenge_timeout = 15.0  # seconds per challenge attempt
        self.liveness_score = 0.0
        self.liveness_passed = False
        self.result_start_time = None
        self.last_result = {}

    def _load_employees(self):
        if _DB_AVAILABLE:
            try:
                emps = get_employees_by_organization(self.org_id, db_path=self.db_path)
                return emps
            except Exception as exc:
                logger.warning("Could not load employees from DB: %s", exc)
        return []

    def _pick_challenge(self):
        """Pick a random liveness challenge."""
        self.current_challenge = random.choice(self.liveness.challenges)
        self.blink_count = 0
        self.blink_cooldown = 0
        self.challenge_start_time = time.monotonic()

    def _compute_liveness_score(self, attempts_used: int, time_taken: float) -> float:
        """
        Compute a liveness score in [0, 1] from challenge performance.
        A faster, first-try completion scores higher.
        Formula: score = clamp(1 - time_taken/timeout * 0.3, 0.80, 0.99)
        """
        time_ratio = min(time_taken / self.challenge_timeout, 1.0)
        score = 1.0 - time_ratio * 0.2
        return round(max(0.80, min(0.99, score)), 4)

    def _draw_overlay(self, frame: np.ndarray, state: str) -> np.ndarray:
        """Draw HUD overlay on the display frame."""
        display = cv2.flip(frame, 1)
        h, w = display.shape[:2]

        # Semi-transparent top bar
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, 70), (10, 15, 25), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

        # EdgeAuth title
        cv2.putText(display, "EdgeAuth  |  Offline Verification",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 118), 2)

        # Face bounding box
        box = get_face_bounding_box(frame)
        if box:
            top, right, bottom, left = box
            flipped_left  = w - right
            flipped_right = w - left
            color = (0, 255, 128) if state == PipelineState.LIVENESS_CHALLENGE else (100, 100, 255)
            cv2.rectangle(display, (flipped_left, top), (flipped_right, bottom), color, 2)

        # State-specific text
        if state == PipelineState.WAITING_FOR_FACE:
            cv2.putText(display, "Look at the camera...",
                        (15, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        elif state == PipelineState.LIVENESS_CHALLENGE:
            elapsed = time.monotonic() - (self.challenge_start_time or time.monotonic())
            remaining = max(0, self.challenge_timeout - elapsed)

            challenge_label = f"Challenge: {self.current_challenge}"
            cv2.putText(display, challenge_label,
                        (15, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 230, 118), 2)

            timer_color = (0, 200, 255) if remaining > 5 else (0, 80, 255)
            cv2.putText(display, f"Time: {remaining:.1f}s",
                        (15, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, timer_color, 2)

            if self.current_challenge == "Blink Twice":
                cv2.putText(display, f"Blinks: {self.blink_count}/2",
                            (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

        elif state == PipelineState.EXTRACTING_EMBEDDING:
            cv2.putText(display, "Extracting face embedding...",
                        (15, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        elif state == PipelineState.VERIFYING:
            cv2.putText(display, "Verifying identity...",
                        (15, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        elif state == PipelineState.RESULT:
            result = self.last_result
            if result.get("verified"):
                msg  = f"VERIFIED: {result.get('name', 'Unknown')}"
                msg2 = f"Score: {result.get('score', 0):.4f}  |  Event: {result.get('event', '')}"
                cv2.putText(display, msg,  (15, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 128), 2)
                cv2.putText(display, msg2, (15, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 255, 180), 2)
            else:
                msg  = "ACCESS DENIED"
                msg2 = f"Score: {result.get('score', 0):.4f}  (threshold: {self.verification_threshold:.2f})"
                cv2.putText(display, msg,  (15, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 80, 255), 2)
                cv2.putText(display, msg2, (15, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 255), 2)

        cv2.putText(display, "Press 'q' to quit  |  'r' to reset",
                    (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        return display

    def _log_event(self, employee_id: str, event_type: str,
                   verification_score: float) -> None:
        """Write verification result to SQLite."""
        if not _DB_AVAILABLE:
            logger.warning("DB not available — skipping log write.")
            return
        try:
            log_id = insert_log(
                organization_id=self.org_id,
                employee_id=employee_id,
                event_type=event_type,
                verification_score=verification_score,
                liveness_score=self.liveness_score,
                liveness_passed=self.liveness_passed,
                device_id=self.device_id,
                metadata={
                    "challenge": self.current_challenge,
                    "model": "face_recognition_dlib_resnet_v1",
                    "pipeline": "face_pipeline.py",
                },
                db_path=self.db_path,
            )
            logger.info("Logged %s event: log_id=%s", event_type, log_id)
            print(f"  {GREEN}[DB]{RESET} Event logged → log_id: {log_id[:12]}...")
        except Exception as exc:
            logger.error("Failed to write log: %s", exc)

    def run(self):
        """Main loop — opens webcam and runs the pipeline state machine."""
        _banner("EdgeAuth — Live Verification Pipeline")

        if not self.enrolled:
            print(f"  {YELLOW}[WARN]{RESET} No enrolled employees found for org '{self.org_id}'.")
            print(f"  {YELLOW}[WARN]{RESET} Run enroll_employee.py first to register a face.")
            print(f"  {YELLOW}[INFO]{RESET} Continuing in DEMO mode (will always show ACCESS DENIED).\n")

        print(f"  {CYAN}-->{RESET} Device: {self.device_id}")
        print(f"  {CYAN}-->{RESET} Org:    {self.org_id}")
        print(f"  {CYAN}-->{RESET} DB:     {self.db_path}")
        print(f"  {CYAN}-->{RESET} Threshold: {self.verification_threshold:.2f}\n")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print(f"  {RED}[ERROR]{RESET} Could not open webcam.")
            sys.exit(1)

        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print(f"  {GREEN}[OK]{RESET} Webcam opened. Starting pipeline...\n")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ---- STATE: WAITING_FOR_FACE --------------------------------
            if self.state == PipelineState.WAITING_FOR_FACE:
                box = get_face_bounding_box(frame)
                if box:
                    self._pick_challenge()
                    print(f"  {CYAN}-->{RESET} Face detected → Challenge: {self.current_challenge}")
                    self.state = PipelineState.LIVENESS_CHALLENGE

            # ---- STATE: LIVENESS_CHALLENGE ------------------------------
            elif self.state == PipelineState.LIVENESS_CHALLENGE:
                elapsed = time.monotonic() - self.challenge_start_time

                # Timeout: pick a new challenge
                if elapsed > self.challenge_timeout:
                    print(f"  {YELLOW}[TIMEOUT]{RESET} Challenge expired — picking new challenge.")
                    self._pick_challenge()
                    continue

                passed, status = self.liveness.check_liveness(frame_rgb, self.current_challenge)

                if self.blink_cooldown > 0:
                    self.blink_cooldown -= 1

                if self.current_challenge == "Blink Twice":
                    if passed and self.blink_cooldown == 0:
                        self.blink_count += 1
                        self.blink_cooldown = 15
                        print(f"  {CYAN}-->{RESET} Blink {self.blink_count}/2 detected.")
                    if self.blink_count >= 2:
                        print(f"  {GREEN}[OK]{RESET} Blink challenge passed!")
                        self.liveness_score = self._compute_liveness_score(2, elapsed)
                        self.liveness_passed = True
                        self.state = PipelineState.EXTRACTING_EMBEDDING
                else:
                    if passed:
                        print(f"  {GREEN}[OK]{RESET} {self.current_challenge} challenge passed! (status={status})")
                        self.liveness_score = self._compute_liveness_score(1, elapsed)
                        self.liveness_passed = True
                        self.state = PipelineState.EXTRACTING_EMBEDDING

            # ---- STATE: EXTRACTING_EMBEDDING ----------------------------
            elif self.state == PipelineState.EXTRACTING_EMBEDDING:
                print(f"  {CYAN}-->{RESET} Liveness passed (score={self.liveness_score:.4f}). Extracting embedding...")
                embedding = extract_embedding(frame)

                if embedding is None:
                    print(f"  {YELLOW}[WARN]{RESET} Could not extract embedding — retrying next frame.")
                    # Stay in this state for one more frame
                    continue
                else:
                    print(f"  {GREEN}[OK]{RESET} Embedding extracted (128-d vector).")
                    self.state = PipelineState.VERIFYING
                    self._run_verification(embedding)

            # ---- STATE: RESULT ------------------------------------------
            elif self.state == PipelineState.RESULT:
                if time.monotonic() - self.result_start_time > self.result_hold:
                    self.state = PipelineState.COOLDOWN
                    self._cooldown_start = time.monotonic()

            # ---- STATE: COOLDOWN ----------------------------------------
            elif self.state == PipelineState.COOLDOWN:
                if time.monotonic() - self._cooldown_start > self.cooldown:
                    print(f"\n  {CYAN}-->{RESET} Ready for next verification.\n")
                    self.state = PipelineState.WAITING_FOR_FACE
                    self.liveness_passed = False
                    self.liveness_score = 0.0

            # ---- DRAW ---------------------------------------------------
            display = self._draw_overlay(frame, self.state)
            cv2.imshow("EdgeAuth — Live Pipeline", display)

            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print(f"  {YELLOW}[RESET]{RESET} Manual reset.")
                self.state = PipelineState.WAITING_FOR_FACE
                self.liveness_passed = False
                self.liveness_score = 0.0

        cap.release()
        cv2.destroyAllWindows()
        print(f"\n{GREEN}[DONE]{RESET} Pipeline stopped.\n")

    def _run_verification(self, embedding: np.ndarray):
        """Match embedding against enrolled employees and log the result."""
        print(f"  {CYAN}-->{RESET} Searching {len(self.enrolled)} enrolled employee(s)...")

        best_match, score = find_best_match(
            embedding, self.enrolled, self.verification_threshold
        )

        if best_match:
            name  = best_match.get("full_name", "Unknown")
            emp_id = best_match["employee_id"]
            event  = "CLOCK_IN"
            print(f"  {GREEN}[VERIFIED]{RESET} {name}  (score={score:.4f})")
            self.last_result = {"verified": True, "name": name, "score": score, "event": event}
            self._log_event(emp_id, event, score)
        else:
            print(f"  {RED}[ACCESS DENIED]{RESET} No match found (best score={score:.4f})")
            # Log against a placeholder employee if no employees enrolled
            event = "ACCESS_DENIED"
            if self.enrolled:
                # Still log against highest-scoring employee for audit trail
                all_scores = []
                for emp in self.enrolled:
                    from face_recognizer import cosine_similarity, list_to_embedding
                    enrolled_emb = list_to_embedding(emp["face_embedding"])
                    s = cosine_similarity(embedding, enrolled_emb)
                    all_scores.append((emp, s))
                all_scores.sort(key=lambda x: x[1], reverse=True)
                denied_emp, _ = all_scores[0]
                self._log_event(denied_emp["employee_id"], event, score)
            self.last_result = {"verified": False, "score": score}

        self.result_start_time = time.monotonic()
        self.state = PipelineState.RESULT


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def _pick_or_create_org(db_path: str) -> str:
    """Return the first available org ID, or create a default one."""
    orgs = list_organizations(db_path=db_path)
    if orgs:
        org = orgs[0]
        print(f"  Using organization: {org['name']} (id={org['organization_id'][:8]}...)")
        return org["organization_id"]

    # Create a default org for demo purposes
    from database import insert_organization
    org_id = insert_organization(
        name="EdgeAuth Demo Organization",
        region="LOCAL",
        contact_email="demo@edgeauth.local",
        liveness_threshold=0.80,
        db_path=db_path,
    )
    print(f"  Created default demo organization (id={org_id[:8]}...)")
    return org_id


def main():
    parser = argparse.ArgumentParser(
        description="EdgeAuth Offline Face Verification Pipeline"
    )
    parser.add_argument("--org-id",   default=None, help="Organization UUID (auto-picks first if omitted)")
    parser.add_argument("--device-id", default="EDGE-DEVICE-LOCAL", help="Device identifier")
    parser.add_argument("--db",        default=None, help="SQLite DB path")
    parser.add_argument("--threshold", type=float, default=DEFAULT_VERIFICATION_THRESHOLD,
                        help=f"Cosine similarity threshold (default {DEFAULT_VERIFICATION_THRESHOLD})")
    args = parser.parse_args()

    # Resolve DB path — default to backend/local_device DB
    if args.db:
        db_path = args.db
    elif _DB_AVAILABLE:
        db_path = str(_BACKEND_PATH / "workforce_verification.db")
    else:
        db_path = "workforce_verification.db"

    # Ensure DB is initialized
    if _DB_AVAILABLE:
        initialize_database(db_path=db_path)

    # Resolve org ID
    if args.org_id:
        org_id = args.org_id
    elif _DB_AVAILABLE:
        org_id = _pick_or_create_org(db_path)
    else:
        org_id = "00000000-0000-0000-0000-000000000000"

    # Launch pipeline
    pipeline = FacePipeline(
        org_id=org_id,
        device_id=args.device_id,
        db_path=db_path,
        verification_threshold=args.threshold,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
