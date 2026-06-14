# -*- coding: utf-8 -*-
"""
test_new_modules.py
-------------------
Pure-logic validation suite for all NEW EdgeAuth AI modules.

Strategy: all third-party imports (cv2, mediapipe, face_recognition) are
stubbed via unittest.mock BEFORE any project module is imported, so the test
runs with ONLY the standard library + numpy — no camera, no model files.

Tests:
  1.  face_recognizer    -- compare_embeddings() match / no-match
  2.  face_recognizer    -- match_face() identical embeddings
  3.  face_recognizer    -- match_face() orthogonal embeddings (no match)
  4.  face_recognizer    -- cosine_similarity() range
  5.  face_recognizer    -- embedding_to_list / list_to_embedding round-trip
  6.  face_recognizer    -- extract_embedding() blank frame -> None
  7.  face_recognizer    -- extract_embedding() face found -> 128-d array
  8.  face_verification_engine -- imports & _make_result() schema
  9.  face_verification_engine -- _make_result() error field
  10. face_verification_engine -- check_face_detected() blank -> False
  11. face_verification_engine -- check_face_detected() face -> True
  12. face_verification_engine -- run_face_recognition() blank frame
  13. LivenessSequenceState -- initialization
  14. LivenessSequenceState -- Blink Twice advances after 2 blinks
  15. LivenessSequenceState -- all challenges complete
  16. LivenessSequenceState -- current_challenge_passed() flag
  17. LivenessSequenceState -- get_display_status()
  18. LivenessSequenceState -- DEFAULT_CHALLENGE_SEQUENCE exported
  19. backend_api          -- imports (enroll_face, verify_face, verify_liveness)
  20. backend_api          -- enroll_face() blank frame -> graceful error
  21. backend_api          -- enroll_face() with mocked embedding
  22. backend_api          -- verify_face() in-process store lookup
  23. backend_api          -- verify_face() unknown employee -> graceful error
  24. backend_api          -- to_json() serialisation round-trip
  25. Final JSON schema    -- all required keys, correct types

Run:
  python test_new_modules.py
  (requires only: numpy, standard library)

Author: EdgeAuth Offline Verification Platform
"""

from __future__ import annotations

import json
import logging
import os
import sys
import types
import traceback
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np

# ---------------------------------------------------------------------------
# stdout UTF-8 safety (Windows CP1252)
# ---------------------------------------------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_AI_DIR = Path(__file__).resolve().parent
if str(_AI_DIR) not in sys.path:
    sys.path.insert(0, str(_AI_DIR))

logging.basicConfig(level=logging.CRITICAL)

# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------
GREEN  = "\033[92m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

passed_count = 0
failed_count = 0


def _ok(name: str, detail: str = "") -> None:
    global passed_count
    passed_count += 1
    tag = f"{GREEN}[PASS]{RESET}"
    suffix = f"  ->  {detail}" if detail else ""
    print(f"  {tag}  {name}{suffix}")


def _fail(name: str, error: str) -> None:
    global failed_count
    failed_count += 1
    tag = f"{RED}[FAIL]{RESET}"
    print(f"  {tag}  {name}  ->  {error}")


def _section(title: str) -> None:
    dashes = "-" * max(2, 55 - len(title))
    print(f"\n{BOLD}{CYAN}-- {title} {dashes}{RESET}")


# ---------------------------------------------------------------------------
# Helpers: synthetic embeddings
# ---------------------------------------------------------------------------

def _rand_emb() -> np.ndarray:
    """Random L2-normalised 128-d embedding (simulates a face)."""
    v = np.random.randn(128).astype(np.float64)
    return v / np.linalg.norm(v)


def _near_emb(base: np.ndarray, noise: float = 0.02) -> np.ndarray:
    """Very similar embedding (same person, slight variation)."""
    v = base + np.random.randn(128).astype(np.float64) * noise
    return v / np.linalg.norm(v)


def _ortho_emb(base: np.ndarray) -> np.ndarray:
    """Embedding orthogonal to base (different person)."""
    v = np.random.randn(128).astype(np.float64)
    v = v - np.dot(v, base) * base
    return v / np.linalg.norm(v)


def _blank_bgr() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


# ===========================================================================
# STUB FABRIC
# ===========================================================================
# Build minimal fake modules for cv2, mediapipe, face_recognition so that
# project modules can be imported without the actual packages installed.

def _make_cv2_stub() -> types.ModuleType:
    mod = types.ModuleType("cv2")
    mod.COLOR_BGR2RGB = 4
    mod.cvtColor       = lambda img, _code: img          # identity
    mod.VideoCapture   = MagicMock()
    mod.imread         = MagicMock(return_value=None)
    mod.imwrite        = MagicMock(return_value=True)
    mod.imshow         = MagicMock()
    mod.waitKey        = MagicMock(return_value=0xFF & ord('q'))
    mod.destroyAllWindows = MagicMock()
    mod.rectangle      = MagicMock()
    mod.putText        = MagicMock()
    mod.addWeighted    = MagicMock()
    mod.flip           = lambda img, _: img
    mod.CAP_PROP_FRAME_WIDTH  = 3
    mod.CAP_PROP_FRAME_HEIGHT = 4
    mod.FONT_HERSHEY_SIMPLEX  = 0
    return mod


def _make_mediapipe_stub(face_found: bool = False) -> types.ModuleType:
    """
    Build a mediapipe stub.
    If face_found=True the FaceLandmarker returns a fake landmark list.
    """
    mp = types.ModuleType("mediapipe")

    # --- mp.Image ---
    class FakeImage:
        def __init__(self, image_format, data):
            pass
    mp.Image = FakeImage

    class FakeImageFormat:
        SRGB = 1
    mp.ImageFormat = FakeImageFormat

    # --- landmark ---
    class FakeLM:
        def __init__(self, x=0.5, y=0.5, z=0.0):
            self.x, self.y, self.z = x, y, z

    # 478 landmarks (MediaPipe FaceLandmarker uses 478)
    fake_landmarks = [FakeLM(0.5, 0.5, 0.0) for _ in range(478)]

    class FakeDetectionResult:
        face_landmarks = [fake_landmarks] if face_found else []

    class FakeLandmarker:
        @staticmethod
        def create_from_options(options):
            inst = FakeLandmarker()
            return inst
        def detect(self, img):
            return FakeDetectionResult()

    # --- stubs for tasks.python.vision & python ---
    tasks_mod   = types.ModuleType("mediapipe.tasks")
    python_mod  = types.ModuleType("mediapipe.tasks.python")
    vision_mod  = types.ModuleType("mediapipe.tasks.python.vision")

    class FakeBaseOptions:
        def __init__(self, model_asset_path=""):
            pass

    class FaceLandmarkerOptions:
        def __init__(self, **kwargs):
            pass

    vision_mod.FaceLandmarker        = FakeLandmarker
    vision_mod.FaceLandmarkerOptions = FaceLandmarkerOptions
    python_mod.BaseOptions           = FakeBaseOptions

    mp.tasks  = tasks_mod
    tasks_mod.python = python_mod
    python_mod.vision = vision_mod

    # Register sub-modules
    sys.modules["mediapipe.tasks"]              = tasks_mod
    sys.modules["mediapipe.tasks.python"]       = python_mod
    sys.modules["mediapipe.tasks.python.vision"]= vision_mod

    return mp


def _make_face_recognition_stub(
    locations: list | None = None,
    encodings: list | None = None,
) -> types.ModuleType:
    mod = types.ModuleType("face_recognition")
    mod.load_image_file = MagicMock(return_value=np.zeros((480, 640, 3), dtype=np.uint8))
    mod.face_locations  = MagicMock(return_value=locations or [])
    mod.face_encodings  = MagicMock(return_value=encodings or [])
    return mod


# Persistent stubs — created ONCE; we never replace these objects.
# We update their .return_value attributes to change behaviour per-test.
_CV2_STUB = _make_cv2_stub()
_FR_STUB  = _make_face_recognition_stub(locations=[], encodings=[])
_MP_STUB  = _make_mediapipe_stub(face_found=False)


def _set_fr_results(locations=None, encodings=None) -> None:
    """Update face_recognition stub return values (without replacing the object)."""
    _FR_STUB.face_locations.return_value = locations or []
    _FR_STUB.face_encodings.return_value = encodings or []
    # Also update on whatever reference the already-imported module holds
    # (works because fr.face_recognition IS _FR_STUB if stubs were installed first)


def _install_stubs(face_found_in_mediapipe: bool = False) -> None:
    """Install the persistent stub objects into sys.modules."""
    sys.modules["cv2"]              = _CV2_STUB
    mp = _make_mediapipe_stub(face_found=face_found_in_mediapipe)
    sys.modules["mediapipe"]        = mp
    sys.modules["face_recognition"] = _FR_STUB


# ===========================================================================
# PHASE 1 — Install stubs & import face_recognizer
# ===========================================================================

_install_stubs(face_found_in_mediapipe=False)
_set_fr_results(locations=[], encodings=[])

# Force fresh imports
for _mod in list(sys.modules.keys()):
    if _mod in ("face_recognizer", "liveness_challenge",
                "face_verification_engine", "backend_api"):
        del sys.modules[_mod]

try:
    import face_recognizer as fr
    _FR_OK = True
except Exception as e:
    _FR_OK = False
    print(f"{RED}FATAL{RESET}: Could not import face_recognizer: {e}")
    traceback.print_exc()
    sys.exit(1)


# ===========================================================================
# SECTION 1 — face_recognizer
# ===========================================================================

_section("SECTION 1 — face_recognizer: core functions")

# T01 — compare_embeddings: same person (high similarity)
try:
    a = _rand_emb()
    b = _near_emb(a, noise=0.02)
    matched, score = fr.compare_embeddings(a, b, threshold=0.75)
    assert isinstance(matched, bool)
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0
    _ok("T01  compare_embeddings() -- near embeddings", f"score={score:.4f} matched={matched}")
except Exception as e:
    _fail("T01  compare_embeddings() -- near embeddings", str(e))

# T02 — compare_embeddings: different person (low / orthogonal)
try:
    a = _rand_emb()
    c = _ortho_emb(a)
    matched, score = fr.compare_embeddings(a, c, threshold=0.75)
    assert isinstance(matched, bool)
    assert isinstance(score, float)
    _ok("T02  compare_embeddings() -- orthogonal embeddings", f"score={score:.4f} matched={matched}")
except Exception as e:
    _fail("T02  compare_embeddings() -- orthogonal embeddings", str(e))

# T03 — match_face: identical -> always match
try:
    x = _rand_emb()
    matched, score = fr.match_face(x, x, threshold=0.75)
    assert matched is True, f"Expected True, got {matched}"
    assert score >= 0.99, f"Expected >=0.99, got {score}"
    _ok("T03  match_face() -- identical embeddings -> matched=True", f"score={score:.6f}")
except Exception as e:
    _fail("T03  match_face() -- identical embeddings", str(e))

# T04 — match_face: orthogonal -> no match
try:
    p = _rand_emb()
    q = _ortho_emb(p)
    matched, score = fr.match_face(p, q, threshold=0.75)
    assert matched is False, f"Expected False, got {matched}"
    _ok("T04  match_face() -- orthogonal -> matched=False", f"score={score:.4f}")
except Exception as e:
    _fail("T04  match_face() -- orthogonal", str(e))

# T05 — cosine_similarity in [-1, 1]
try:
    s = fr.cosine_similarity(_rand_emb(), _rand_emb())
    assert -1.0 <= s <= 1.0
    _ok("T05  cosine_similarity() -- range [-1, 1]", f"score={s:.4f}")
except Exception as e:
    _fail("T05  cosine_similarity()", str(e))

# T06 — embedding round-trip
try:
    emb  = _rand_emb()
    lst  = fr.embedding_to_list(emb)
    back = fr.list_to_embedding(lst)
    assert len(lst) == 128
    diff = float(np.linalg.norm(emb - back))
    assert diff < 1e-5, f"Round-trip error {diff}"
    _ok("T06  embedding_to_list/list_to_embedding round-trip", f"dim=128 err={diff:.2e}")
except Exception as e:
    _fail("T06  embedding round-trip", str(e))

# T07 — extract_embedding blank frame -> None  (stub returns no locations)
try:
    result = fr.extract_embedding(_blank_bgr())
    assert result is None, f"Expected None, got {type(result)}"
    _ok("T07  extract_embedding() -- no face -> None")
except Exception as e:
    _fail("T07  extract_embedding() -- no face -> None", str(e))

# T08 — extract_embedding with a face found (patch stub to return encoding)
try:
    fake_enc = _rand_emb().tolist()
    _set_fr_results(locations=[(0, 100, 100, 0)], encodings=[fake_enc])
    result = fr.extract_embedding(_blank_bgr())
    assert result is not None, "Expected embedding, got None"
    assert result.shape == (128,)
    _ok("T08  extract_embedding() -- face found -> 128-d array", f"shape={result.shape}")
except Exception as e:
    _fail("T08  extract_embedding() -- face found", str(e))
finally:
    _set_fr_results()


# ===========================================================================
# SECTION 2 — face_verification_engine
# ===========================================================================

_section("SECTION 2 -- face_verification_engine: imports & schema")

# Reload with stubs in place (liveness_challenge depends on mediapipe)
for _mod in list(sys.modules.keys()):
    if _mod in ("liveness_challenge", "face_verification_engine"):
        del sys.modules[_mod]

_install_stubs(face_found_in_mediapipe=True)
_set_fr_results()

try:
    import liveness_challenge as lc_mod
    from face_verification_engine import (
        LivenessSequenceState,
        DEFAULT_CHALLENGE_SEQUENCE,
        run_face_recognition,
        check_face_detected,
        _make_result,
    )
    _ok("T09  face_verification_engine imports OK",
        "LivenessSequenceState, run_face_recognition, _make_result, ...")
except ImportError as e:
    _fail("T09  face_verification_engine imports", str(e))
    traceback.print_exc()
    sys.exit(1)

# T10 — _make_result schema
try:
    r = _make_result(
        face_detected=True, blink_verified=True, head_pose_verified=True,
        liveness_verified=True, face_matched=True, similarity_score=0.91,
        verification_passed=True,
    )
    REQUIRED = {"face_detected","blink_verified","head_pose_verified",
                "liveness_verified","face_matched","similarity_score","verification_passed"}
    missing = REQUIRED - set(r.keys())
    assert not missing, f"Missing keys: {missing}"
    assert r["verification_passed"] is True
    assert isinstance(r["similarity_score"], float)
    _ok("T10  _make_result() -- canonical schema", str(sorted(REQUIRED)))
except Exception as e:
    _fail("T10  _make_result() -- schema", str(e))

# T11 — _make_result with error
try:
    r = _make_result(error="No face detected")
    assert r["verification_passed"] is False
    assert r["error"] == "No face detected"
    _ok("T11  _make_result() -- error field propagates correctly")
except Exception as e:
    _fail("T11  _make_result() -- error field", str(e))

# T12 — check_face_detected blank frame -> False
try:
    _set_fr_results()  # ensure no locations
    detected = check_face_detected(_blank_bgr())
    assert detected is False
    _ok("T12  check_face_detected() -- blank frame -> False")
except Exception as e:
    _fail("T12  check_face_detected() -- blank frame", str(e))

# T13 — check_face_detected when face present -> True
try:
    _set_fr_results(locations=[(0, 100, 100, 0)])
    detected = check_face_detected(_blank_bgr())
    assert detected is True
    _ok("T13  check_face_detected() -- face present -> True")
except Exception as e:
    _fail("T13  check_face_detected() -- face present", str(e))
finally:
    _set_fr_results()

# T14 — run_face_recognition blank frame -> graceful error
try:
    _set_fr_results()  # no face, no encodings
    r = run_face_recognition(_blank_bgr(), _rand_emb())
    assert r["face_detected"] is False
    assert r["face_matched"] is False
    assert r["error"] is not None
    _ok("T14  run_face_recognition() -- no face -> graceful error", r["error"])
except Exception as e:
    _fail("T14  run_face_recognition() -- no face", str(e))


# ===========================================================================
# SECTION 3 — LivenessSequenceState
# ===========================================================================

_section("SECTION 3 -- LivenessSequenceState: sequential flow")


class _AlwaysPassEngine:
    """Mock engine: any challenge always passes on the first call."""
    def check_liveness(self, frame_rgb, challenge):
        return True, "MockPass"


class _NeverPassEngine:
    """Mock engine: always returns False (challenge never passed)."""
    def check_liveness(self, frame_rgb, challenge):
        return False, "Pending"


DUMMY_RGB = np.zeros((480, 640, 3), dtype=np.uint8)

# T15 — initialization
try:
    s = LivenessSequenceState(["Blink Twice", "Turn Head Left", "Turn Head Right"])
    assert s.total_count == 3
    assert s.completed_count == 0
    assert s.current_challenge == "Blink Twice"
    assert not s.is_complete()
    _ok("T15  LivenessSequenceState -- init", "3 challenges, index=0")
except Exception as e:
    _fail("T15  LivenessSequenceState -- init", str(e))

# T16 — Blink Twice advances after 2 debounced blinks
try:
    s   = LivenessSequenceState(["Blink Twice", "Turn Head Left"])
    eng = _AlwaysPassEngine()
    initial_challenge = s.current_challenge

    blink_advanced = False
    for _ in range(200):  # enough frames to handle debounce
        if s.current_challenge != "Blink Twice":
            blink_advanced = True
            break
        s.process_frame(DUMMY_RGB, eng)

    assert blink_advanced, "Should have advanced past Blink Twice"
    assert "Blink Twice" in s.get_passed_challenges()
    _ok("T16  LivenessSequenceState -- Blink Twice advances after 2 blinks")
except Exception as e:
    _fail("T16  LivenessSequenceState -- Blink Twice advance", str(e))
    traceback.print_exc()

# T17 — full sequence completes
try:
    s   = LivenessSequenceState(["Blink Twice", "Turn Head Left"])
    eng = _AlwaysPassEngine()
    complete = False
    for _ in range(300):
        done, _ = s.process_frame(DUMMY_RGB, eng)
        if done:
            complete = True
            break
    assert complete, "Sequence should complete"
    assert s.is_complete()
    assert "Blink Twice"     in s.get_passed_challenges()
    assert "Turn Head Left"  in s.get_passed_challenges()
    _ok("T17  LivenessSequenceState -- full sequence completes", f"passed={s.get_passed_challenges()}")
except Exception as e:
    _fail("T17  LivenessSequenceState -- full sequence", str(e))
    traceback.print_exc()

# T18 — current_challenge_passed() flag set after advance
try:
    s   = LivenessSequenceState(["Turn Head Left", "Smile"])
    eng = _AlwaysPassEngine()
    # First frame advances past Turn Head Left
    s.process_frame(DUMMY_RGB, eng)
    assert s.current_challenge_passed() is True
    _ok("T18  LivenessSequenceState -- current_challenge_passed() flag set")
except Exception as e:
    _fail("T18  LivenessSequenceState -- current_challenge_passed()", str(e))

# T19 — get_display_status()
try:
    s = LivenessSequenceState(["Blink Twice"])
    status = s.get_display_status()
    assert "Blink" in status, f"Unexpected: {status}"
    _ok("T19  LivenessSequenceState -- get_display_status()", f"'{status}'")
except Exception as e:
    _fail("T19  LivenessSequenceState -- get_display_status()", str(e))

# T20 — DEFAULT_CHALLENGE_SEQUENCE exported
try:
    assert len(DEFAULT_CHALLENGE_SEQUENCE) >= 2
    _ok("T20  DEFAULT_CHALLENGE_SEQUENCE exported", str(list(DEFAULT_CHALLENGE_SEQUENCE)))
except Exception as e:
    _fail("T20  DEFAULT_CHALLENGE_SEQUENCE", str(e))

# T21 — timeout resets the current challenge (no skip)
try:
    import face_verification_engine as fve_mod
    orig = fve_mod.CHALLENGE_TIMEOUT_SECONDS
    fve_mod.CHALLENGE_TIMEOUT_SECONDS = 0.0001  # instant timeout

    s   = LivenessSequenceState(["Blink Twice"])
    eng = _NeverPassEngine()

    import time; time.sleep(0.01)  # ensure elapsed > timeout
    complete, msg = s.process_frame(DUMMY_RGB, eng)

    # Should have reset (timeout message), not completed
    assert not complete
    assert s.current_challenge == "Blink Twice"     # still same challenge
    assert s.completed_count == 0                   # nothing passed
    _ok("T21  LivenessSequenceState -- timeout resets, no skip, no completion")
    fve_mod.CHALLENGE_TIMEOUT_SECONDS = orig
except Exception as e:
    _fail("T21  LivenessSequenceState -- timeout reset", str(e))


# ===========================================================================
# SECTION 4 — backend_api
# ===========================================================================

_section("SECTION 4 -- backend_api: enroll_face(), verify_face(), to_json()")

# Clean import
for _mod in list(sys.modules.keys()):
    if _mod == "backend_api":
        del sys.modules[_mod]

try:
    import backend_api as ba
    _ok("T22  backend_api imports OK", "enroll_face, verify_face, verify_liveness, to_json")
except Exception as e:
    _fail("T22  backend_api imports", str(e))
    traceback.print_exc()
    sys.exit(1)

# T23 — enroll_face() with blank frame (no face) -> success=False, graceful error
try:
    _set_fr_results()   # no face
    r = ba.enroll_face(employee_id="TEST-BLANK", full_name="Ghost", frame_bgr=_blank_bgr())
    assert r["success"] is False
    assert r["employee_id"] == "TEST-BLANK"
    assert r["error"] is not None
    _ok("T23  enroll_face() -- no face -> success=False, error returned", r["error"])
except Exception as e:
    _fail("T23  enroll_face() -- no face graceful error", str(e))
    traceback.print_exc()

# T24 — enroll_face() with mocked face detection -> stores embedding
_ENROLL_EMB = _rand_emb()
_ENROLL_ID  = "TEST-EMP-MOCK-001"

try:
    _set_fr_results(
        locations=[(0, 100, 100, 0)],
        encodings=[_ENROLL_EMB.tolist()],
    )
    r = ba.enroll_face(
        employee_id=_ENROLL_ID,
        full_name="Priya Nair",
        frame_bgr=_blank_bgr(),
    )
    assert r["success"] is True, f"Expected success=True, got error: {r.get('error')}"
    assert r["employee_id"] == _ENROLL_ID
    assert r["embedding_stored"] is True
    assert _ENROLL_ID in ba._EMBEDDING_STORE
    _ok("T24  enroll_face() -- face found -> success=True, embedding stored",
        f"id={_ENROLL_ID}")
except Exception as e:
    _fail("T24  enroll_face() -- success path", str(e))
    traceback.print_exc()
finally:
    _set_fr_results()

# T25 — verify_face() reads from in-process store
try:
    live_emb = _near_emb(_ENROLL_EMB, noise=0.01)
    _set_fr_results(
        locations=[(0, 100, 100, 0)],
        encodings=[live_emb.tolist()],
    )
    r = ba.verify_face(employee_id=_ENROLL_ID, frame_bgr=_blank_bgr(), threshold=0.75)
    assert r["employee_id"] == _ENROLL_ID
    assert "matched"    in r
    assert "similarity" in r
    assert isinstance(r["similarity"], float)
    assert r["error"] is None
    _ok("T25  verify_face() -- in-process store -> structured result",
        f"matched={r['matched']} sim={r['similarity']:.4f}")
except Exception as e:
    _fail("T25  verify_face() -- in-process store", str(e))
    traceback.print_exc()
finally:
    _set_fr_results()

# T26 — verify_face() unknown employee_id -> graceful error
try:
    r = ba.verify_face(employee_id="NOBODY-999", frame_bgr=_blank_bgr())
    assert r["matched"] is False
    assert r["error"] is not None
    _ok("T26  verify_face() -- unknown employee -> graceful error", r["error"])
except Exception as e:
    _fail("T26  verify_face() -- unknown employee", str(e))

# T27 — to_json() round-trip
try:
    payload = {
        "face_detected": True, "blink_verified": True,
        "head_pose_verified": True, "liveness_verified": True,
        "face_matched": True, "similarity_score": 0.912345,
        "verification_passed": True,
    }
    js  = ba.to_json(payload)
    obj = json.loads(js)
    assert obj["verification_passed"] is True
    assert obj["similarity_score"] == 0.912345
    _ok("T27  to_json() -- serialisation + parse round-trip OK")
except Exception as e:
    _fail("T27  to_json()", str(e))


# ===========================================================================
# SECTION 5 — Final JSON schema
# ===========================================================================

_section("SECTION 5 -- Final JSON output schema")

REQUIRED_KEYS = {
    "face_detected", "blink_verified", "head_pose_verified",
    "liveness_verified", "face_matched", "similarity_score", "verification_passed",
}

try:
    r = _make_result(
        face_detected=True, blink_verified=True, head_pose_verified=True,
        liveness_verified=True, face_matched=True,
        similarity_score=0.91, verification_passed=True,
    )
    missing = REQUIRED_KEYS - set(r.keys())
    assert not missing, f"Missing: {missing}"
    assert r["face_detected"]       is True
    assert r["blink_verified"]      is True
    assert r["head_pose_verified"]  is True
    assert r["liveness_verified"]   is True
    assert r["face_matched"]        is True
    assert r["verification_passed"] is True
    assert 0.0 <= r["similarity_score"] <= 1.0
    _ok("T28  Final JSON schema -- all keys present, correct types",
        str(sorted(r.keys())))

    print(f"\n  {CYAN}Sample final JSON output:{RESET}")
    for line in json.dumps(r, indent=2).splitlines():
        print(f"    {line}")
except Exception as e:
    _fail("T28  Final JSON schema", str(e))


# ===========================================================================
# Summary
# ===========================================================================

total = passed_count + failed_count
print(f"\n{BOLD}{'=' * 62}{RESET}")
print(f"{BOLD}  VALIDATION SUMMARY{RESET}")
print(f"{'=' * 62}")
print(f"  Total tests : {total}")
print(f"  {GREEN}Passed      : {passed_count}{RESET}")
fail_colour = RED if failed_count else GREEN
print(f"  {fail_colour}Failed      : {failed_count}{RESET}")
print(f"{'=' * 62}")

if failed_count == 0:
    print(f"\n{BOLD}{GREEN}  ALL {total} TESTS PASSED")
    print(f"  New modules verified and working.{RESET}\n")
else:
    print(f"\n{BOLD}{RED}  {failed_count} TEST(S) FAILED -- review errors above.{RESET}\n")
    sys.exit(1)
