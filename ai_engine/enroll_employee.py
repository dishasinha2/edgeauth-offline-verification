"""
enroll_employee.py
------------------
CLI tool to enroll a new employee into the EdgeAuth offline database.

Flow:
  1. Opens webcam and shows live preview.
  2. User presses SPACE to capture a face snapshot.
  3. Extracts a 128-d face embedding using face_recognition (offline ResNet).
  4. Inserts employee record + embedding into the SQLite database.
  5. Saves enrollment photo to ai_engine/enrollments/<employee_id>.jpg.

Usage:
  python enroll_employee.py --name "Arjun Sharma" --dept "Engineering" --role "Site Supervisor"
  python enroll_employee.py --name "Priya Nair"   --org-id <uuid>       --db ../my.db

Options:
  --name     Full name of the employee (required)
  --org-id   Organization UUID (auto-picks first org if omitted)
  --dept     Department (optional)
  --role     Job role (optional)
  --db       SQLite DB path (defaults to backend/local_device DB)

Author: EdgeAuth Offline Verification Platform
"""

import argparse
import logging
import os
import sys
import uuid
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — make backend module importable
# ---------------------------------------------------------------------------

_AI_ENGINE_DIR = Path(__file__).resolve().parent
_REPO_ROOT     = _AI_ENGINE_DIR.parent
_BACKEND_PATH  = _REPO_ROOT / "backend" / "local_device"

if str(_BACKEND_PATH) not in sys.path:
    sys.path.insert(0, str(_BACKEND_PATH))

from face_recognizer import (
    extract_embedding,
    embedding_to_list,
    get_face_bounding_box,
)

try:
    from database import (
        initialize_database,
        insert_employee,
        insert_organization,
        list_organizations,
    )
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False
    print("[WARN] backend/local_device/database.py not found — DB writes disabled.")

# ---------------------------------------------------------------------------
# Logging & colours
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("enroll_employee")

GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# ---------------------------------------------------------------------------
# Enrollments folder
# ---------------------------------------------------------------------------

ENROLLMENTS_DIR = _AI_ENGINE_DIR / "enrollments"
ENROLLMENTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pick_or_create_org(db_path: str) -> str:
    """Return the first org ID, or create a demo org."""
    orgs = list_organizations(db_path=db_path)
    if orgs:
        org = orgs[0]
        print(f"  {CYAN}-->{RESET} Using org: {org['name']} (id={org['organization_id'][:8]}...)")
        return org["organization_id"]

    org_id = insert_organization(
        name="EdgeAuth Demo Organization",
        region="LOCAL",
        contact_email="demo@edgeauth.local",
        liveness_threshold=0.80,
        db_path=db_path,
    )
    print(f"  {GREEN}[OK]{RESET} Created default org (id={org_id[:8]}...)")
    return org_id


def _draw_preview(frame: np.ndarray, name: str, ready: bool) -> np.ndarray:
    display = cv2.flip(frame, 1)
    h, w = display.shape[:2]

    # Top bar
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (10, 15, 25), -1)
    cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

    cv2.putText(display, "EdgeAuth — Employee Enrollment",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 118), 2)
    cv2.putText(display, f"Enrolling: {name}",
                (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    # Face bounding box
    box = get_face_bounding_box(frame)
    if box:
        top, right, bottom, left = box
        flipped_left  = w - right
        flipped_right = w - left
        color = (0, 255, 128) if ready else (100, 100, 255)
        cv2.rectangle(display, (flipped_left, top), (flipped_right, bottom), color, 2)
        if ready:
            cv2.putText(display, "FACE DETECTED — Press SPACE to capture",
                        (15, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 128), 2)
        else:
            cv2.putText(display, "Searching for face...",
                        (15, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
    else:
        cv2.putText(display, "No face detected — please look at the camera",
                    (15, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 255), 2)

    cv2.putText(display, "SPACE = Capture  |  Q = Quit",
                (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    return display


# ---------------------------------------------------------------------------
# Main enrollment logic
# ---------------------------------------------------------------------------


def enroll(
    full_name: str,
    org_id: str,
    department: str,
    role: str,
    db_path: str,
) -> None:
    """Run the interactive enrollment webcam session."""
    print(f"\n{BOLD}{CYAN}{'=' * 50}{RESET}")
    print(f"{BOLD}{CYAN}  EdgeAuth — Enrolling: {full_name}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 50}{RESET}\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(f"  {RED}[ERROR]{RESET} Cannot open webcam.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print(f"  {CYAN}-->{RESET} Look at the camera. Press SPACE when your face is clearly visible.")
    print(f"  {CYAN}-->{RESET} Press 'q' to cancel.\n")

    captured_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        box = get_face_bounding_box(frame)
        face_ready = box is not None

        display = _draw_preview(frame, full_name, face_ready)
        cv2.imshow("EdgeAuth — Enrollment", display)

        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            print(f"\n  {YELLOW}[CANCELLED]{RESET} Enrollment cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            return
        elif key == ord(' '):  # SPACE
            if not face_ready:
                print(f"  {YELLOW}[WARN]{RESET} No face detected — please look at the camera.")
                continue

            print(f"\n  {CYAN}-->{RESET} Capturing frame...")
            captured_frame = frame.copy()

            # Quick visual feedback — green flash
            flash = np.full_like(display, (50, 255, 50), dtype=np.uint8)
            cv2.addWeighted(display, 0.5, flash, 0.5, 0, display)
            cv2.imshow("EdgeAuth — Enrollment", display)
            cv2.waitKey(150)
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured_frame is None:
        print(f"  {RED}[ERROR]{RESET} No frame captured.")
        return

    # Extract embedding
    print(f"  {CYAN}-->{RESET} Extracting face embedding...")
    embedding = extract_embedding(captured_frame)
    if embedding is None:
        print(f"  {RED}[ERROR]{RESET} Could not extract face embedding from captured frame.")
        print(f"         Please retry in better lighting with a clear face view.")
        return

    print(f"  {GREEN}[OK]{RESET} 128-d embedding extracted.")

    # Save enrollment photo
    emp_id = str(uuid.uuid4())
    photo_path = ENROLLMENTS_DIR / f"{emp_id}.jpg"
    cv2.imwrite(str(photo_path), captured_frame)
    print(f"  {GREEN}[OK]{RESET} Photo saved: {photo_path}")

    # Write to database
    if _DB_AVAILABLE:
        try:
            embedding_list = embedding_to_list(embedding)
            emp_id = insert_employee(
                organization_id=org_id,
                full_name=full_name,
                face_embedding=embedding_list,
                department=department,
                role=role,
                embedding_model="face_recognition_dlib_resnet_v1",
                employee_id=emp_id,
                db_path=db_path,
            )
            print(f"  {GREEN}[OK]{RESET} Employee inserted into DB: id={emp_id[:8]}...")
        except Exception as exc:
            print(f"  {RED}[ERROR]{RESET} DB insert failed: {exc}")
            logger.exception("DB insert failed")
            return
    else:
        print(f"  {YELLOW}[WARN]{RESET} DB not available — embedding NOT persisted.")

    print(f"\n  {BOLD}{GREEN}Enrollment complete!{RESET}")
    print(f"  Name:       {full_name}")
    print(f"  ID:         {emp_id}")
    print(f"  Dept:       {department or 'N/A'}")
    print(f"  Role:       {role or 'N/A'}")
    print(f"  Org:        {org_id[:8]}...")
    print(f"  Photo:      {photo_path}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Enroll a new employee into the EdgeAuth offline database"
    )
    parser.add_argument("--name",   required=True, help="Employee full name")
    parser.add_argument("--org-id", default=None,  help="Organization UUID")
    parser.add_argument("--dept",   default="",    help="Department")
    parser.add_argument("--role",   default="",    help="Job role")
    parser.add_argument("--db",     default=None,  help="SQLite DB path")
    args = parser.parse_args()

    # Resolve DB path
    if args.db:
        db_path = args.db
    elif _DB_AVAILABLE:
        db_path = str(_BACKEND_PATH / "workforce_verification.db")
    else:
        db_path = "workforce_verification.db"

    if _DB_AVAILABLE:
        initialize_database(db_path=db_path)

    # Resolve org ID
    org_id = args.org_id
    if not org_id and _DB_AVAILABLE:
        org_id = _pick_or_create_org(db_path)
    elif not org_id:
        org_id = "00000000-0000-0000-0000-000000000000"

    enroll(
        full_name=args.name,
        org_id=org_id,
        department=args.dept,
        role=args.role,
        db_path=db_path,
    )


if __name__ == "__main__":
    main()
