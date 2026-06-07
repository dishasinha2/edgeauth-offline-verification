"""
server.py
---------
EdgeAuth Local HTTP Server — Flask wrapper for the AI engine.

Exposes six REST endpoints that the React Native frontend calls on-device:
  POST /enroll          — Enroll a new employee face embedding
  POST /verify          — 1-to-1 face verification with automatic log insert
  POST /verify/full     — Combined face verify + employee lookup (frontend camera screen)
  POST /liveness        — Run the sequential liveness challenge sequence
  GET  /sync/status     — Report sync queue statistics
  GET  /health          — Health probe for the device management layer

The sync daemon is started as a background thread on server startup so that
verification logs are uploaded to AWS whenever connectivity is available.

Usage:
    python server.py

Author: EdgeAuth Offline Verification Platform
"""

import sys
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap — identical pattern to backend_api.py
# ---------------------------------------------------------------------------

_AI_ENGINE_DIR = Path(__file__).resolve().parent
_REPO_ROOT     = _AI_ENGINE_DIR.parent
_BACKEND_PATH  = _REPO_ROOT / "backend" / "local_device"

for _p in [str(_AI_ENGINE_DIR), str(_BACKEND_PATH)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------

import base64
import datetime
import logging
import socket
import threading

# ---------------------------------------------------------------------------
# Third-party — loaded lazily to allow structural import without AI packages
# ---------------------------------------------------------------------------

from flask import Flask, jsonify, request   # type: ignore[import]
from flask_cors import CORS             # type: ignore[import]

# ---------------------------------------------------------------------------
# Internal — AI engine (optional at import time; required at request time)
# ---------------------------------------------------------------------------

try:
    from backend_api import enroll_face, verify_face, verify_liveness  # type: ignore[import]
    _AI_ENGINE_AVAILABLE = True
except ImportError as _ai_err:
    _AI_ENGINE_AVAILABLE = False
    _AI_ENGINE_ERROR = str(_ai_err)

# ---------------------------------------------------------------------------
# Internal — database layer
# ---------------------------------------------------------------------------

from database import (                  # type: ignore[import]
    initialize_database,
    insert_log,
    get_employee,
    get_sync_queue_stats,
    get_connection,
    DB_PATH,
)

# ---------------------------------------------------------------------------
# Internal — sync engine
# ---------------------------------------------------------------------------

from sync_engine import run_sync_daemon  # type: ignore[import]

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("edgeauth.server")

app = Flask(__name__)
CORS(app)  # Allow React Native / Expo dev client on any origin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_frame(frame_b64: str):
    """
    Decode a base64-encoded JPEG string to an OpenCV BGR numpy array.
    Raises ValueError on bad input.
    """
    import cv2        # lazy: only imported when an actual request arrives
    import numpy as np

    try:
        img_bytes = base64.b64decode(frame_b64)
    except Exception as exc:
        raise ValueError(f"base64 decode failed: {exc}") from exc

    buf = np.frombuffer(img_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("cv2.imdecode returned None -- not a valid JPEG/PNG image")
    return bgr


def _utc_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _error_500(exc: Exception):
    """Generic 500 response."""
    logger.exception("Unhandled server error: %s", exc)
    return jsonify({"error": str(exc), "success": False}), 500


def _error_400(message: str):
    """Generic 400 response."""
    return jsonify({"error": message, "success": False}), 400


def _ai_unavailable():
    """503 response when the AI engine packages are not installed."""
    reason = _AI_ENGINE_ERROR if not _AI_ENGINE_AVAILABLE else "unknown"
    return jsonify({
        "error": f"AI engine unavailable: {reason}",
        "hint": "Run: pip install -r ai_engine/requirements.txt",
        "success": False,
    }), 503


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """Health probe — confirms server is alive and reports DB path."""
    return jsonify({
        "status":  "ok",
        "service": "EdgeAuth Local Server",
        "db_path": DB_PATH,
    }), 200


# ---------------------------------------------------------------------------
# POST /enroll
# ---------------------------------------------------------------------------

@app.route("/enroll", methods=["POST"])
def enroll():
    """
    Enroll a new employee face.

    Required JSON fields:
        full_name       (str)
        organization_id (str)
        frame_b64       (str) — base64-encoded JPEG

    Optional JSON fields:
        employee_id     (str)
        department      (str)
        role            (str)
    """
    try:
        if not _AI_ENGINE_AVAILABLE:
            return _ai_unavailable()
        body = request.get_json(force=True, silent=True) or {}

        # Validate frame_b64
        frame_b64 = body.get("frame_b64", "")
        if not frame_b64:
            return _error_400("Invalid or missing frame_b64")

        try:
            bgr_array = _decode_frame(frame_b64)
        except ValueError as exc:
            return _error_400(f"Invalid or missing frame_b64: {exc}")

        # Call enroll_face
        result = enroll_face(
            employee_id=body.get("employee_id") or None,
            full_name=body.get("full_name", "Unknown"),
            organization_id=body.get("organization_id", "00000000-0000-0000-0000-000000000000"),
            department=body.get("department", ""),
            role=body.get("role", ""),
            frame_bgr=bgr_array,
            db_path=DB_PATH,
        )

        status_code = 200 if result.get("success") else 400
        return jsonify(result), status_code

    except Exception as exc:
        return _error_500(exc)


# ---------------------------------------------------------------------------
# POST /verify
# ---------------------------------------------------------------------------

@app.route("/verify", methods=["POST"])
def verify():
    """
    1-to-1 face verification.

    Required JSON fields:
        employee_id     (str)
        organization_id (str)
        frame_b64       (str) — base64-encoded JPEG

    Optional JSON fields:
        threshold        (float, default 0.55)
        liveness_score   (float, default 0.0)  — from a preceding /liveness call
        liveness_passed  (bool,  default False) — from a preceding /liveness call
    """
    try:
        if not _AI_ENGINE_AVAILABLE:
            return _ai_unavailable()
        body = request.get_json(force=True, silent=True) or {}

        frame_b64 = body.get("frame_b64", "")
        if not frame_b64:
            return _error_400("Invalid or missing frame_b64")

        try:
            bgr_array = _decode_frame(frame_b64)
        except ValueError as exc:
            return _error_400(f"Invalid or missing frame_b64: {exc}")

        employee_id     = body.get("employee_id", "")
        organization_id = body.get("organization_id", "")
        threshold       = float(body.get("threshold", 0.55))

        # Real liveness values forwarded from the frontend after /liveness completes
        liveness_score  = float(body.get("liveness_score", 0.0))
        liveness_passed = bool(body.get("liveness_passed", False))

        result = verify_face(
            employee_id=employee_id,
            frame_bgr=bgr_array,
            db_path=DB_PATH,
            organization_id=organization_id,
            threshold=threshold,
        )

        # Determine event type and insert audit log with real liveness values
        event_type = "ACCESS_GRANTED" if result.get("matched") else "ACCESS_DENIED"
        log_id = insert_log(
            organization_id=organization_id,
            employee_id=employee_id,
            event_type=event_type,
            verification_score=result.get("similarity", 0.0),
            liveness_score=liveness_score,
            liveness_passed=liveness_passed,
            device_id=socket.gethostname(),
            db_path=DB_PATH,
        )

        result["log_id"] = log_id
        return jsonify(result), 200

    except Exception as exc:
        return _error_500(exc)


# ---------------------------------------------------------------------------
# POST /verify/full
# ---------------------------------------------------------------------------

@app.route("/verify/full", methods=["POST"])
def verify_full():
    """
    Combined face-verify + employee metadata lookup.
    This is the primary endpoint called from the frontend camera screen.

    Required JSON fields:
        employee_id     (str)
        organization_id (str)
        frame_b64       (str) — base64-encoded JPEG

    Optional JSON fields:
        threshold        (float, default 0.55)
        liveness_score   (float, default 0.0)  — forwarded from a preceding /liveness call
        liveness_passed  (bool,  default False) — forwarded from a preceding /liveness call
    """
    try:
        if not _AI_ENGINE_AVAILABLE:
            return _ai_unavailable()
        body = request.get_json(force=True, silent=True) or {}

        frame_b64 = body.get("frame_b64", "")
        if not frame_b64:
            return _error_400("Invalid or missing frame_b64")

        try:
            bgr_array = _decode_frame(frame_b64)
        except ValueError as exc:
            return _error_400(f"Invalid or missing frame_b64: {exc}")

        employee_id     = body.get("employee_id", "")
        organization_id = body.get("organization_id", "")
        threshold       = float(body.get("threshold", 0.55))

        # Real liveness values forwarded from the frontend after /liveness completes
        liveness_score  = float(body.get("liveness_score", 0.0))
        liveness_passed = bool(body.get("liveness_passed", False))

        # Run face verification
        face_result = verify_face(
            employee_id=employee_id,
            frame_bgr=bgr_array,
            db_path=DB_PATH,
            organization_id=organization_id,
            threshold=threshold,
        )

        face_matched     = face_result.get("matched", False)
        similarity_score = face_result.get("similarity", 0.0)
        error_msg        = face_result.get("error")

        # Look up employee metadata
        employee_name = ""
        department    = ""
        role          = ""
        try:
            emp = get_employee(employee_id, db_path=DB_PATH)
            if emp:
                employee_name = emp.get("full_name", "")
                department    = emp.get("department", "") or ""
                role          = emp.get("role", "") or ""
        except Exception as lookup_exc:
            logger.warning("Employee lookup failed: %s", lookup_exc)

        # Determine event classification
        event_type          = "ACCESS_GRANTED" if face_matched else "ACCESS_DENIED"
        verification_passed = face_matched

        # Insert audit log with real liveness values from the preceding /liveness call
        log_id = insert_log(
            organization_id=organization_id,
            employee_id=employee_id,
            event_type=event_type,
            verification_score=similarity_score,
            liveness_score=liveness_score,
            liveness_passed=liveness_passed,
            device_id=socket.gethostname(),
            db_path=DB_PATH,
        )

        return jsonify({
            "face_matched":          face_matched,
            "similarity_score":      similarity_score,
            "verification_passed":   verification_passed,
            "liveness_score":        liveness_score,
            "liveness_passed":       liveness_passed,
            "employee_id":           employee_id,
            "employee_name":         employee_name,
            "department":            department,
            "role":                  role,
            "event_type":            event_type,
            "log_id":                log_id,
            "timestamp":             _utc_iso(),
            "error":                 error_msg,
        }), 200

    except Exception as exc:
        return _error_500(exc)


# ---------------------------------------------------------------------------
# GET /logs
# ---------------------------------------------------------------------------

@app.route("/logs", methods=["GET"])
def get_logs():
    """
    Query verification logs for a given organization.

    Query parameters:
        organization_id  (str, required)
        limit            (int, optional, default 50)
        offset           (int, optional, default 0)

    Response:
        {
            "logs":  [ { ...log fields... } ],
            "count": int
        }
    """
    try:
        organization_id = request.args.get("organization_id", "")
        if not organization_id:
            return _error_400("Query parameter 'organization_id' is required")

        limit  = int(request.args.get("limit",  50))
        offset = int(request.args.get("offset",  0))

        # Clamp to sane maximums
        limit  = max(1, min(limit, 500))
        offset = max(0, offset)

        rows_out = []
        conn = get_connection(DB_PATH)
        try:
            rows = conn.execute(
                """
                SELECT * FROM logs
                WHERE organization_id = ?
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
                """,
                (organization_id, limit, offset),
            ).fetchall()
            import json as _json
            for row in rows:
                entry = dict(row)
                # Deserialize metadata JSON blob if present
                if entry.get("metadata"):
                    try:
                        entry["metadata"] = _json.loads(entry["metadata"])
                    except (ValueError, TypeError):
                        entry["metadata"] = {}
                # Convert SQLite INTEGER to Python bool for readability
                entry["liveness_passed"] = bool(entry.get("liveness_passed", 0))
                entry["synced"]          = bool(entry.get("synced", 0))
                rows_out.append(entry)
        finally:
            conn.close()

        return jsonify({
            "logs":  rows_out,
            "count": len(rows_out),
        }), 200

    except Exception as exc:
        return _error_500(exc)


# ---------------------------------------------------------------------------
# POST /liveness
# ---------------------------------------------------------------------------

@app.route("/liveness", methods=["POST"])
def liveness():
    """
    Run the sequential liveness challenge sequence.

    Optional JSON fields:
        challenges              (list[str], default ["Blink Twice", "Turn Head Left", "Turn Head Right"])
        timeout_per_challenge   (float, default 15.0)
        headless                (bool, default True)
    """
    try:
        if not _AI_ENGINE_AVAILABLE:
            return _ai_unavailable()
        body = request.get_json(force=True, silent=True) or {}

        challenges            = body.get("challenges", ["Blink Twice", "Turn Head Left", "Turn Head Right"])
        timeout_per_challenge = float(body.get("timeout_per_challenge", 15.0))
        headless              = bool(body.get("headless", True))

        result = verify_liveness(
            challenges=challenges,
            timeout_per_challenge=timeout_per_challenge,
            headless=headless,
        )

        return jsonify(result), 200

    except Exception as exc:
        return _error_500(exc)


# ---------------------------------------------------------------------------
# GET /sync/status
# ---------------------------------------------------------------------------

@app.route("/sync/status", methods=["GET"])
def sync_status():
    """
    Return a snapshot of the sync queue statistics.

    Response:
        {
            "pending":           int,
            "in_flight":         int,
            "failed_retryable":  int
        }
    """
    try:
        raw = get_sync_queue_stats(db_path=DB_PATH)
        # get_sync_queue_stats returns keys that match the status column values
        # e.g. {"PENDING": 3, "IN_FLIGHT": 1, "FAILED_RETRYABLE": 0}
        return jsonify({
            "pending":          raw.get("PENDING", 0),
            "in_flight":        raw.get("IN_FLIGHT", 0),
            "failed_retryable": raw.get("FAILED_RETRYABLE", 0),
        }), 200

    except Exception as exc:
        return _error_500(exc)


# ---------------------------------------------------------------------------
# Server startup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # Startup check: TFLite model must be present or the server exits early
    # with a clear, actionable message rather than crashing mid-request.
    # -----------------------------------------------------------------------
    REQUIRED_MODEL = _AI_ENGINE_DIR / "models" / "mobilefacenet.tflite"
    if not REQUIRED_MODEL.exists():
        print("=" * 60)
        print("ERROR: TFLite model file not found.")
        print(f"  Expected: {REQUIRED_MODEL}")
        print("  Run: python ai_engine/models/download_models.py")
        print("       (or: python ai_engine/convert_to_tflite.py)")
        print("=" * 60)
        sys.exit(1)

    # Initialize the SQLite schema (idempotent)
    initialize_database(db_path=DB_PATH)

    # Start the sync daemon as a background daemon thread
    stop_event = threading.Event()
    sync_thread = threading.Thread(
        target=run_sync_daemon,
        kwargs={"db_path": DB_PATH, "stop_event": stop_event},
        daemon=True,
        name="SyncDaemon",
    )
    sync_thread.start()
    logger.info("SyncDaemon background thread started.")

    # Start Flask (blocking)
    app.run(host="0.0.0.0", port=5000, debug=False)
