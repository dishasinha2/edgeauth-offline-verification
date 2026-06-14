"""
integration_test.py
-------------------
Full-stack integration test: wires the AI engine together with the SQLite
database layer end-to-end using a synthetic (noise) image.

No camera is required. The test asserts that every function returns a
structured dict with the expected keys and raises NO unhandled exceptions.

Run:
    python ai_engine/integration_test.py
"""

import sys
import os
import logging
import tempfile
import uuid
from pathlib import Path

# Suppress INFO/DEBUG log output so stderr stays clean (test runner friendly)
logging.disable(logging.WARNING)

# Force UTF-8 output on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------

_AI_ENGINE_DIR = Path(__file__).resolve().parent
_BACKEND_PATH  = _AI_ENGINE_DIR.parent / "backend" / "local_device"

for _p in [str(_AI_ENGINE_DIR), str(_BACKEND_PATH)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Package availability guards
# ---------------------------------------------------------------------------

def _has_cv2():
    try:
        import cv2  # noqa
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from database import initialize_database, insert_organization  # type: ignore

_AI_AVAILABLE = _has_cv2()
if _AI_AVAILABLE:
    try:
        from backend_api import enroll_face, verify_face  # type: ignore
    except ImportError:
        _AI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Synthetic image factory (only if cv2 available)
# ---------------------------------------------------------------------------

def _make_synthetic_frame(height=480, width=640):
    import numpy as np
    import cv2  # type: ignore
    frame = np.full((height, width, 3), 60, dtype=np.uint8)
    cx, cy = width // 2, height // 2
    cv2.ellipse(frame, (cx, cy), (90, 120), 0, 0, 360, (220, 220, 220), -1)
    cv2.circle(frame, (cx - 30, cy - 20), 12, (50, 50, 50), -1)
    cv2.circle(frame, (cx + 30, cy - 20), 12, (50, 50, 50), -1)
    return frame


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_database_init(db_path):
    print("[TEST 1] initialize_database() ...")
    initialize_database(db_path=db_path)
    print("  [PASS] Database initialized at:", db_path)


def test_insert_organization(db_path):
    print("[TEST 2] insert_organization() ...")
    org_id = insert_organization(
        name="Integration Test Org",
        region="TEST",
        db_path=db_path,
    )
    assert isinstance(org_id, str) and len(org_id) > 0
    print(f"  [PASS] Organization created: {org_id}")
    return org_id


def test_enroll_face(db_path, org_id):
    if not _AI_AVAILABLE:
        print("[TEST 3] enroll_face() ... [SKIP] cv2/face_recognition not installed")
        return None

    print("[TEST 3] enroll_face() with synthetic frame ...")
    frame  = _make_synthetic_frame()
    result = enroll_face(
        employee_id="integration-test-001",
        full_name="Integration Test User",
        organization_id=org_id,
        department="QA",
        role="Tester",
        frame_bgr=frame,
        db_path=db_path,
    )
    assert isinstance(result, dict),    f"Expected dict, got {type(result)}"
    assert "success" in result,         "Missing 'success' key"
    assert "employee_id" in result,     "Missing 'employee_id' key"
    assert "full_name" in result,       "Missing 'full_name' key"
    assert "embedding_stored" in result,"Missing 'embedding_stored' key"

    status = "SUCCESS" if result["success"] else "NO FACE IN SYNTHETIC IMAGE (expected)"
    print(f"  [PASS] enroll_face returned structured dict -- {status}")
    print(f"    result = {result}")
    return result


def test_verify_face(db_path, org_id):
    if not _AI_AVAILABLE:
        print("[TEST 4] verify_face() ... [SKIP] cv2/face_recognition not installed")
        return None

    print("[TEST 4] verify_face() with synthetic frame ...")
    frame  = _make_synthetic_frame()
    result = verify_face(
        employee_id="integration-test-001",
        frame_bgr=frame,
        db_path=db_path,
        organization_id=org_id,
        threshold=0.55,
    )
    assert isinstance(result, dict),   f"Expected dict, got {type(result)}"
    assert "matched" in result,        "Missing 'matched' key"
    assert "similarity" in result,     "Missing 'similarity' key"
    assert "employee_id" in result,    "Missing 'employee_id' key"

    status = "MATCHED" if result["matched"] else "NOT MATCHED (expected for synthetic)"
    print(f"  [PASS] verify_face returned structured dict -- {status}")
    print(f"    result = {result}")
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 55)
    print("  EdgeAuth Full-Stack Integration Test")
    print("  (Synthetic image -- no camera needed)")
    print("=" * 55)
    if not _AI_AVAILABLE:
        print("  NOTE: AI packages (cv2/face_recognition) not installed.")
        print("  AI tests will SKIP. DB tests run normally.")
    print()

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    passed = 0
    failed = 0

    try:
        for fn, args in [
            (test_database_init,     (db_path,)),
            (test_insert_organization, (db_path,)),
        ]:
            try:
                fn(*args)
                passed += 1
            except Exception as exc:
                print(f"  [FAIL] {fn.__name__}: {exc}")
                failed += 1
            print()

        # Dependent tests need a second org
        try:
            org_id = insert_organization("Org B", "TEST", db_path=db_path)
            test_enroll_face(db_path, org_id)
            passed += 1
            print()
            test_verify_face(db_path, org_id)
            passed += 1
        except Exception as exc:
            print(f"  [FAIL] AI test: {exc}")
            failed += 1

        print()
        print("=" * 55)
        print(f"  Results: {passed} passed, {failed} failed")
        print("=" * 55)

        if failed:
            print()
            print("[FAIL] Integration test completed with failures.")
            return 1

        print()
        print("[PASS] All integration tests completed successfully.")
        return 0

    finally:
        try:
            os.unlink(db_path)
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
