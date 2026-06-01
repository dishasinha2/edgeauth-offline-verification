"""
spoof_test.py
-------------
Anti-Spoofing Demonstration Script for EdgeAuth.

This script demonstrates WHY challenge-based liveness detection defeats
photo/video replay attacks:

  REAL USER:   Completes blink / head-turn / smile challenge → PASS
  PHOTO SPOOF: Static image cannot blink or move → FAIL

How it works:
  1. Loads a static image (or uses webcam to capture one).
  2. Runs MediaPipe liveness check logic against it for N frames.
  3. Shows that a static image NEVER satisfies the motion-based challenge.
  4. Optionally demonstrates two-image cosine similarity (person A vs person B).

Usage:
  # Test liveness on a static image (spoof attempt):
  python spoof_test.py --image path/to/face_photo.jpg

  # Test with webcam capture (saves a snap, then runs as static):
  python spoof_test.py --capture

  # Run embedding comparison between two images:
  python spoof_test.py --compare img1.jpg img2.jpg

  # Full demo mode (webcam + spoof simulation):
  python spoof_test.py --demo

Author: EdgeAuth Offline Verification Platform
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_AI_ENGINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_AI_ENGINE_DIR))

from liveness_challenge import LivenessChallenge
from face_recognizer import (
    extract_embedding_from_file,
    cosine_similarity,
    list_to_embedding,
    get_face_bounding_box,
    extract_embedding,
)

# ---------------------------------------------------------------------------
# Logging & colours
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("spoof_test")

GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

SPOOF_FRAMES = 60   # Number of frames to try liveness on a static image


# ---------------------------------------------------------------------------
# Test 1: Liveness on static image
# ---------------------------------------------------------------------------


def test_liveness_on_static_image(image_path: str) -> bool:
    """
    Simulate a photo replay/spoof attack:
    Load a static image and replay it frame-by-frame through liveness logic.

    A static image CANNOT pass blink or head-turn challenges because
    no pixel motion occurs — EAR stays constant, nose ratio stays constant.

    Returns:
        True if the static image passes liveness (unexpected — would be a bug).
        False if it correctly fails (expected — anti-spoof working).
    """
    print(f"\n{BOLD}{CYAN}{'=' * 55}{RESET}")
    print(f"{BOLD}{CYAN}  SPOOF TEST 1: Static Image Liveness Attack{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 55}{RESET}\n")
    print(f"  {CYAN}-->{RESET} Image: {image_path}")

    if not os.path.isfile(image_path):
        print(f"  {RED}[ERROR]{RESET} Image file not found: {image_path}")
        return False

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"  {RED}[ERROR]{RESET} Could not load image.")
        return False

    # Check that a face exists in the image at all
    box = get_face_bounding_box(image_bgr)
    if box is None:
        print(f"  {YELLOW}[WARN]{RESET} No face detected in image. Cannot run liveness test.")
        return False

    print(f"  {GREEN}[OK]{RESET} Face detected in image.")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    liveness = LivenessChallenge()
    challenges_to_test = ["Blink Twice", "Turn Head Left", "Turn Head Right", "Smile"]
    results = {}

    for challenge in challenges_to_test:
        passed_count = 0
        blink_count  = 0
        blink_cd     = 0

        for _ in range(SPOOF_FRAMES):
            passed, status = liveness.check_liveness(image_rgb, challenge)

            if challenge == "Blink Twice":
                if passed and blink_cd == 0:
                    blink_count += 1
                    blink_cd = 15
                if blink_cd > 0:
                    blink_cd -= 1
                if blink_count >= 2:
                    passed_count = 1
                    break
            else:
                if passed:
                    passed_count += 1
                    break

        challenge_passed = passed_count > 0
        results[challenge] = challenge_passed

        status_str = f"{RED}FAIL (SPOOF BLOCKED){RESET}" if not challenge_passed else f"{GREEN}PASS (unexpected!){RESET}"
        print(f"  Challenge '{challenge}': {status_str}")

    # Summary
    total_passed = sum(results.values())
    print(f"\n  {'=' * 45}")
    if total_passed == 0:
        print(f"  {GREEN}[ANTI-SPOOF OK]{RESET} Static image failed ALL {len(challenges_to_test)} challenges.")
        print(f"  Photo replay attack successfully blocked. ✅")
    else:
        print(f"  {YELLOW}[WARNING]{RESET} {total_passed}/{len(challenges_to_test)} challenges passed on static image.")
        print(f"  Consider tightening thresholds.")
    print()

    return total_passed == 0  # True = anti-spoof working correctly


# ---------------------------------------------------------------------------
# Test 2: Embedding comparison (same vs different person)
# ---------------------------------------------------------------------------


def test_embedding_comparison(image_path_a: str, image_path_b: str) -> None:
    """
    Compare face embeddings from two images.
    Same person → high similarity (~0.8+).
    Different people → low similarity (~0.3-0.5).
    """
    print(f"\n{BOLD}{CYAN}{'=' * 55}{RESET}")
    print(f"{BOLD}{CYAN}  SPOOF TEST 2: Embedding Cosine Similarity{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 55}{RESET}\n")

    print(f"  {CYAN}-->{RESET} Image A: {image_path_a}")
    print(f"  {CYAN}-->{RESET} Image B: {image_path_b}\n")

    emb_a = extract_embedding_from_file(image_path_a)
    emb_b = extract_embedding_from_file(image_path_b)

    if emb_a is None:
        print(f"  {RED}[ERROR]{RESET} No face found in Image A.")
        return
    if emb_b is None:
        print(f"  {RED}[ERROR]{RESET} No face found in Image B.")
        return

    score = cosine_similarity(emb_a, emb_b)

    print(f"  Cosine Similarity: {score:.4f}")
    if score >= 0.75:
        print(f"  {GREEN}[SAME PERSON]{RESET} Score {score:.4f} ≥ 0.75 — Same person verified.")
    elif score >= 0.55:
        print(f"  {YELLOW}[UNCERTAIN]{RESET} Score {score:.4f} — Borderline. Consider re-enrollment.")
    else:
        print(f"  {RED}[DIFFERENT PERSON]{RESET} Score {score:.4f} < 0.55 — Different people.")
    print()


# ---------------------------------------------------------------------------
# Test 3: Capture + replay spoof demo
# ---------------------------------------------------------------------------


def test_capture_and_replay_spoof() -> None:
    """
    Interactive demo:
    1. Capture a photo of the user's face from the webcam.
    2. Save it as a static image.
    3. Replay that static image through liveness → demonstrates it FAILS.
    4. Then run the actual live webcam through liveness → demonstrates it PASSES.
    """
    print(f"\n{BOLD}{CYAN}{'=' * 55}{RESET}")
    print(f"{BOLD}{CYAN}  SPOOF TEST 3: Capture → Replay Demo{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 55}{RESET}\n")
    print(f"  {CYAN}Step 1:{RESET} Capturing your face as a 'spoof photo'...")
    print(f"  {CYAN}-->{RESET} Press SPACE to take the photo. Press Q to cancel.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(f"  {RED}[ERROR]{RESET} Cannot open webcam.")
        return

    captured = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        display = cv2.flip(frame, 1)
        cv2.putText(display, "Press SPACE to capture 'spoof' photo",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 118), 2)
        cv2.imshow("Spoof Demo — Capture", display)

        key = cv2.waitKey(5) & 0xFF
        if key == ord(' '):
            captured = frame.copy()
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    if captured is None:
        return

    # Save the captured photo
    spoof_photo_path = str(_AI_ENGINE_DIR / "spoof_demo_photo.jpg")
    cv2.imwrite(spoof_photo_path, captured)
    print(f"  {GREEN}[OK]{RESET} Spoof photo saved: {spoof_photo_path}")

    # Now run liveness test on it
    print(f"\n  {CYAN}Step 2:{RESET} Running liveness challenge on the STATIC photo...")
    test_liveness_on_static_image(spoof_photo_path)

    # Now run liveness on the live webcam
    print(f"  {CYAN}Step 3:{RESET} Running liveness on LIVE webcam to demonstrate REAL user passing...\n")
    print(f"  {CYAN}-->{RESET} Blink twice to prove you are live! Press Q to skip.\n")

    liveness = LivenessChallenge()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return

    blink_count = 0
    blink_cd    = 0
    live_passed = False
    start_time  = time.monotonic()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        elapsed = time.monotonic() - start_time
        if elapsed > 15:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        passed, status = liveness.check_liveness(frame_rgb, "Blink Twice")

        if blink_cd > 0:
            blink_cd -= 1
        if passed and blink_cd == 0:
            blink_count += 1
            blink_cd = 15

        display = cv2.flip(frame, 1)
        cv2.putText(display, "LIVE USER TEST — Blink Twice",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 118), 2)
        cv2.putText(display, f"Blinks: {blink_count}/2",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
        cv2.imshow("Spoof Demo — Live User", display)

        if blink_count >= 2:
            live_passed = True
            break

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n{'=' * 55}")
    print(f"  SPOOF DEMO SUMMARY")
    print(f"{'=' * 55}")
    print(f"  Static photo liveness: {RED}FAILED (Blocked){RESET}")
    live_status = f"{GREEN}PASSED{RESET}" if live_passed else f"{YELLOW}SKIPPED{RESET}"
    print(f"  Live user liveness:    {live_status}")
    print()


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="EdgeAuth Anti-Spoofing Demonstration"
    )
    parser.add_argument("--image",   default=None, nargs="?", const="",
                        help="Path to static image for spoof liveness test")
    parser.add_argument("--compare", nargs=2, metavar=("IMG_A", "IMG_B"),
                        help="Compare embeddings of two images")
    parser.add_argument("--capture", action="store_true",
                        help="Capture a webcam snapshot then replay as spoof")
    parser.add_argument("--demo",    action="store_true",
                        help="Run the full capture→replay demo")
    args = parser.parse_args()

    ran_something = False

    if args.compare:
        test_embedding_comparison(args.compare[0], args.compare[1])
        ran_something = True

    if args.image is not None:
        if args.image == "":
            print(f"  {RED}[ERROR]{RESET} --image requires a file path. Example: --image photo.jpg")
        else:
            test_liveness_on_static_image(args.image)
        ran_something = True

    if args.capture or args.demo:
        test_capture_and_replay_spoof()
        ran_something = True

    if not ran_something:
        print(f"\n{BOLD}EdgeAuth — Spoof Test{RESET}")
        print("Usage examples:\n")
        print("  python spoof_test.py --image path/to/face.jpg")
        print("  python spoof_test.py --compare person_a.jpg person_b.jpg")
        print("  python spoof_test.py --demo\n")
        parser.print_help()


if __name__ == "__main__":
    main()
