"""
database.py
-----------
SQLite Storage Layer for the Offline-First Workforce Verification Platform.

Handles:
  - Schema initialization (organizations, employees, logs, sync_queue)
  - CRUD operations for all tables
  - Face embedding storage as JSON-serialized strings
  - Sync queue management with idempotency guarantees
  - TRUE STRUCTURAL PURGE: DELETE FROM sync_queue (not just status flags)

Phase 5 Update:
  The purge logic has been corrected from a soft-flag approach (synced=1) to a
  hard structural DELETE from sync_queue. Data safety is preserved because the
  immutable audit record is kept in the `logs` table (synced=1 flag only there).
  The sync obligation (the queue row) is structurally removed on HTTP 200 only.

Author: Workforce Verification Platform
Architecture: Offline-First, Edge AI
"""

import sqlite3
import json
import uuid
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_PATH = os.environ.get("WVP_DB_PATH", "workforce_verification.db")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("database")

# ---------------------------------------------------------------------------
# Connection Factory
# ---------------------------------------------------------------------------


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """
    Return a SQLite connection with WAL journal mode for concurrent read safety,
    foreign key enforcement, and row_factory set to sqlite3.Row for dict-like access.
    """
    conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA synchronous=FULL;")  # Fsync on every write — data safety first
    return conn


# ---------------------------------------------------------------------------
# Schema Initialization
# ---------------------------------------------------------------------------


SCHEMA_SQL = """
-- ============================================================
-- ORGANIZATIONS TABLE
-- Stores the tenant / organization context for multi-org support.
-- ============================================================
CREATE TABLE IF NOT EXISTS organizations (
    organization_id   TEXT PRIMARY KEY,          -- UUID, immutable
    name              TEXT NOT NULL,
    region            TEXT NOT NULL,             -- e.g., "APAC", "US-EAST"
    contact_email     TEXT,
    is_active         INTEGER NOT NULL DEFAULT 1, -- 1 = active, 0 = deactivated
    liveness_threshold REAL NOT NULL DEFAULT 0.80, -- Adaptive per-org threshold
    created_at        TEXT NOT NULL,             -- ISO-8601 UTC
    updated_at        TEXT NOT NULL
);

-- ============================================================
-- EMPLOYEES TABLE
-- Stores workforce identity + face embedding for local AI inference.
-- Face embeddings are stored as JSON-serialized float arrays
-- (e.g., 128-dim or 512-dim vectors from FaceNet/ArcFace).
-- ============================================================
CREATE TABLE IF NOT EXISTS employees (
    employee_id       TEXT PRIMARY KEY,          -- UUID, immutable
    organization_id   TEXT NOT NULL REFERENCES organizations(organization_id),
    full_name         TEXT NOT NULL,
    department        TEXT,
    role              TEXT,
    face_embedding    TEXT NOT NULL,             -- JSON array: "[0.123, -0.456, ...]"
    embedding_model   TEXT NOT NULL DEFAULT 'facenet_512d',
    face_photo_s3_key TEXT,                      -- S3 object key for raw enrollment photo
    is_active         INTEGER NOT NULL DEFAULT 1,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_employees_org ON employees(organization_id);

-- ============================================================
-- LOGS TABLE
-- Immutable event log of each verification attempt.
-- Entries persist locally until AWS explicitly confirms receipt (HTTP 200).
-- ============================================================
CREATE TABLE IF NOT EXISTS logs (
    log_id            TEXT PRIMARY KEY,          -- UUID v4, idempotency key
    organization_id   TEXT NOT NULL REFERENCES organizations(organization_id),
    employee_id       TEXT NOT NULL REFERENCES employees(employee_id),
    event_type        TEXT NOT NULL,             -- "CLOCK_IN" | "CLOCK_OUT" | "ACCESS_DENIED"
    verification_score REAL NOT NULL,            -- Cosine similarity, 0.0 - 1.0
    liveness_score    REAL NOT NULL,             -- Liveness detection confidence, 0.0 - 1.0
    liveness_passed   INTEGER NOT NULL,          -- 1 = passed, 0 = failed
    device_id         TEXT NOT NULL,             -- Hardware identifier of the edge device
    latitude          REAL,                      -- GPS coords if available
    longitude         REAL,
    timestamp         TEXT NOT NULL,             -- ISO-8601 UTC of the event
    metadata          TEXT,                      -- JSON blob for future extensibility
    synced            INTEGER NOT NULL DEFAULT 0 -- 0 = pending, 1 = confirmed by AWS
);

CREATE INDEX IF NOT EXISTS idx_logs_org_ts ON logs(organization_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_logs_synced  ON logs(synced);

-- ============================================================
-- SYNC_QUEUE TABLE
-- Tracks which log entries need to be sent to AWS.
-- A row is HARD DELETED (not just flagged) when AWS returns HTTP 200.
-- This is the Phase 5 correct structural purge behavior.
-- ============================================================
CREATE TABLE IF NOT EXISTS sync_queue (
    queue_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    log_id            TEXT NOT NULL UNIQUE REFERENCES logs(log_id),
    organization_id   TEXT NOT NULL,
    enqueued_at       TEXT NOT NULL,             -- ISO-8601 UTC
    attempt_count     INTEGER NOT NULL DEFAULT 0,
    last_attempt_at   TEXT,                      -- ISO-8601 UTC of most recent attempt
    last_error        TEXT,                      -- Last HTTP error or exception message
    status            TEXT NOT NULL DEFAULT 'PENDING'
                          CHECK(status IN ('PENDING', 'IN_FLIGHT', 'FAILED_RETRYABLE'))
);

CREATE INDEX IF NOT EXISTS idx_sync_queue_status ON sync_queue(status);
CREATE INDEX IF NOT EXISTS idx_sync_queue_org    ON sync_queue(organization_id);
"""


def initialize_database(db_path: str = DB_PATH) -> None:
    """
    Create all tables and indexes if they do not already exist.
    Safe to call on every application startup (idempotent).
    """
    logger.info("Initializing database at: %s", db_path)
    conn = get_connection(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        logger.info("Database schema ready.")
    except sqlite3.Error as exc:
        logger.error("Schema initialization failed: %s", exc)
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Helper: UTC timestamp
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _new_uuid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# ORGANIZATIONS — CRUD
# ---------------------------------------------------------------------------


def insert_organization(
    name: str,
    region: str,
    contact_email: Optional[str] = None,
    organization_id: Optional[str] = None,
    liveness_threshold: float = 0.80,
    db_path: str = DB_PATH,
) -> str:
    """
    Insert a new organization record. Returns the generated organization_id.
    liveness_threshold is the adaptive per-org face liveness confidence floor.
    """
    org_id = organization_id or _new_uuid()
    now = _utc_now()
    sql = """
        INSERT INTO organizations
            (organization_id, name, region, contact_email, liveness_threshold, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    conn = get_connection(db_path)
    try:
        conn.execute(sql, (org_id, name, region, contact_email, liveness_threshold, now, now))
        conn.commit()
        logger.info("Inserted organization '%s' (id=%s).", name, org_id)
        return org_id
    except sqlite3.IntegrityError as exc:
        logger.warning("Organization insert conflict: %s", exc)
        raise
    finally:
        conn.close()


def get_organization(
    organization_id: str, db_path: str = DB_PATH
) -> Optional[Dict[str, Any]]:
    """Fetch a single organization by ID. Returns None if not found."""
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM organizations WHERE organization_id = ?", (organization_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_organizations(db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Return all active organizations."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM organizations WHERE is_active = 1 ORDER BY name"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# EMPLOYEES — CRUD
# ---------------------------------------------------------------------------


def insert_employee(
    organization_id: str,
    full_name: str,
    face_embedding: List[float],
    department: Optional[str] = None,
    role: Optional[str] = None,
    embedding_model: str = "facenet_512d",
    face_photo_s3_key: Optional[str] = None,
    employee_id: Optional[str] = None,
    db_path: str = DB_PATH,
) -> str:
    """
    Insert a new employee with their face embedding.
    The embedding is serialized to a JSON string for SQLite storage.
    face_photo_s3_key links to the raw enrollment photo stored in S3.
    Returns the generated employee_id.
    """
    emp_id = employee_id or _new_uuid()
    now = _utc_now()
    embedding_json = json.dumps(face_embedding)

    sql = """
        INSERT INTO employees
            (employee_id, organization_id, full_name, department, role,
             face_embedding, embedding_model, face_photo_s3_key, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    conn = get_connection(db_path)
    try:
        conn.execute(
            sql,
            (emp_id, organization_id, full_name, department, role,
             embedding_json, embedding_model, face_photo_s3_key, now, now),
        )
        conn.commit()
        logger.info(
            "Inserted employee '%s' (id=%s) for org '%s'. S3 key: %s",
            full_name, emp_id, organization_id, face_photo_s3_key or "none",
        )
        return emp_id
    except sqlite3.IntegrityError as exc:
        logger.warning("Employee insert conflict: %s", exc)
        raise
    finally:
        conn.close()


def get_employee(employee_id: str, db_path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    """
    Fetch a single employee and deserialize their face_embedding back to a list.
    Returns None if not found.
    """
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM employees WHERE employee_id = ?", (employee_id,)
        ).fetchone()
        if not row:
            return None
        emp = dict(row)
        emp["face_embedding"] = json.loads(emp["face_embedding"])
        return emp
    finally:
        conn.close()


def get_employees_by_organization(
    organization_id: str, db_path: str = DB_PATH
) -> List[Dict[str, Any]]:
    """
    Return all active employees for a given organization with deserialized embeddings.
    Used by the edge AI inference engine to load identity vectors into memory.
    """
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM employees WHERE organization_id = ? AND is_active = 1",
            (organization_id,),
        ).fetchall()
        result = []
        for row in rows:
            emp = dict(row)
            emp["face_embedding"] = json.loads(emp["face_embedding"])
            result.append(emp)
        logger.debug(
            "Loaded %d employee embeddings for org '%s'.", len(result), organization_id
        )
        return result
    finally:
        conn.close()


def update_employee_embedding(
    employee_id: str,
    new_embedding: List[float],
    embedding_model: str,
    db_path: str = DB_PATH,
) -> bool:
    """Update an employee's face embedding (e.g., after re-enrollment)."""
    now = _utc_now()
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            """UPDATE employees SET face_embedding = ?, embedding_model = ?, updated_at = ?
               WHERE employee_id = ?""",
            (json.dumps(new_embedding), embedding_model, now, employee_id),
        )
        conn.commit()
        if cursor.rowcount == 0:
            logger.warning(
                "update_employee_embedding: no employee found with id '%s'.", employee_id
            )
            return False
        logger.info("Updated embedding for employee '%s'.", employee_id)
        return True
    finally:
        conn.close()


def deactivate_employee(employee_id: str, db_path: str = DB_PATH) -> bool:
    """Soft-delete an employee by setting is_active = 0."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            "UPDATE employees SET is_active = 0, updated_at = ? WHERE employee_id = ?",
            (_utc_now(), employee_id),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# LOGS — CRUD
# ---------------------------------------------------------------------------


def insert_log(
    organization_id: str,
    employee_id: str,
    event_type: str,
    verification_score: float,
    liveness_score: float,
    liveness_passed: bool,
    device_id: str,
    timestamp: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
    log_id: Optional[str] = None,
    db_path: str = DB_PATH,
) -> str:
    """
    Insert a new verification event log.
    Also automatically enqueues the log in the sync_queue for cloud upload.
    Returns the generated log_id (UUID v4).

    This is the PRIMARY write path called by the AI inference engine after
    every face verification attempt.
    """
    if event_type not in ("CLOCK_IN", "CLOCK_OUT", "ACCESS_DENIED", "ACCESS_GRANTED"):
        raise ValueError(
            f"Invalid event_type: '{event_type}'. "
            "Must be one of CLOCK_IN, CLOCK_OUT, ACCESS_DENIED, ACCESS_GRANTED."
        )

    log_entry_id = log_id or _new_uuid()
    ts = timestamp or _utc_now()
    metadata_json = json.dumps(metadata) if metadata else None

    sql = """
        INSERT INTO logs
            (log_id, organization_id, employee_id, event_type, verification_score,
             liveness_score, liveness_passed, device_id, latitude, longitude,
             timestamp, metadata, synced)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
    """
    conn = get_connection(db_path)
    try:
        with conn:  # Atomic transaction: insert log + enqueue sync together
            conn.execute(
                sql,
                (
                    log_entry_id, organization_id, employee_id, event_type,
                    verification_score, liveness_score,
                    1 if liveness_passed else 0,
                    device_id, latitude, longitude, ts, metadata_json,
                ),
            )
            # Immediately enqueue for sync — idempotent via UNIQUE constraint on log_id
            _enqueue_sync_locked(conn, log_entry_id, organization_id)

        logger.info(
            "Log inserted: log_id=%s, employee=%s, event=%s, score=%.4f",
            log_entry_id, employee_id, event_type, verification_score,
        )
        return log_entry_id
    except sqlite3.IntegrityError as exc:
        logger.warning(
            "Log insert failed (possible duplicate log_id '%s'): %s", log_entry_id, exc
        )
        raise
    finally:
        conn.close()


def get_log(log_id: str, db_path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    """Fetch a single log entry by ID."""
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM logs WHERE log_id = ?", (log_id,)).fetchone()
        if not row:
            return None
        entry = dict(row)
        if entry.get("metadata"):
            entry["metadata"] = json.loads(entry["metadata"])
        return entry
    finally:
        conn.close()


def get_unsynced_logs(
    organization_id: Optional[str] = None, db_path: str = DB_PATH
) -> List[Dict[str, Any]]:
    """Return all log entries that have not yet been confirmed by AWS."""
    conn = get_connection(db_path)
    try:
        if organization_id:
            rows = conn.execute(
                "SELECT * FROM logs WHERE synced = 0 AND organization_id = ? ORDER BY timestamp",
                (organization_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM logs WHERE synced = 0 ORDER BY timestamp"
            ).fetchall()
        result = []
        for row in rows:
            entry = dict(row)
            if entry.get("metadata"):
                entry["metadata"] = json.loads(entry["metadata"])
            result.append(entry)
        return result
    finally:
        conn.close()


def mark_log_as_synced(log_id: str, db_path: str = DB_PATH) -> bool:
    """
    Mark a log entry as successfully synced AND atomically hard-delete the
    corresponding sync_queue row.

    Called ONLY after AWS returns HTTP 200 for this specific log_id.

    BUG FIX: The previous implementation only set logs.synced = 1 but did NOT
    remove the sync_queue row.  This caused the entry to reappear as PENDING
    on every subsequent sync cycle (because get_pending_sync_entries JOINs on
    sync_queue, which was never cleaned up).  Both operations now execute in a
    single atomic transaction so a crash between them cannot leave an orphaned
    queue row.
    """
    conn = get_connection(db_path)
    try:
        with conn:  # Atomic: both succeed or both roll back
            cursor = conn.execute(
                "UPDATE logs SET synced = 1 WHERE log_id = ?", (log_id,)
            )
            conn.execute(
                "DELETE FROM sync_queue WHERE log_id = ?", (log_id,)
            )
        updated = cursor.rowcount > 0
        if updated:
            logger.info(
                "mark_log_as_synced: log_id='%s' marked synced=1 and "
                "sync_queue row hard-deleted.", log_id
            )
        return updated
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# SYNC_QUEUE — CRUD
# ---------------------------------------------------------------------------


def _enqueue_sync_locked(
    conn: sqlite3.Connection,
    log_id: str,
    organization_id: str,
) -> None:
    """
    Internal helper — enqueue a sync entry using an ALREADY OPEN connection.
    Must be called within an active transaction. Uses INSERT OR IGNORE to be
    idempotent: calling this twice for the same log_id is harmless.
    """
    conn.execute(
        """
        INSERT OR IGNORE INTO sync_queue (log_id, organization_id, enqueued_at, status)
        VALUES (?, ?, ?, 'PENDING')
        """,
        (log_id, organization_id, _utc_now()),
    )


def enqueue_sync(log_id: str, organization_id: str, db_path: str = DB_PATH) -> None:
    """
    Public-facing enqueue — opens its own connection.
    Safe to call if a log was inserted without going through insert_log().
    """
    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            INSERT OR IGNORE INTO sync_queue (log_id, organization_id, enqueued_at, status)
            VALUES (?, ?, ?, 'PENDING')
            """,
            (log_id, organization_id, _utc_now()),
        )
        conn.commit()
        logger.debug("Enqueued log_id '%s' for sync.", log_id)
    finally:
        conn.close()


def get_pending_sync_entries(
    limit: int = 100, db_path: str = DB_PATH
) -> List[Dict[str, Any]]:
    """
    Return up to `limit` sync queue entries that are ready to be sent.
    Entries with status PENDING or FAILED_RETRYABLE are eligible.
    Ordered by enqueued_at ascending (oldest first) for fairness.
    """
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            """
            SELECT sq.*, l.employee_id, l.event_type, l.verification_score,
                   l.liveness_score, l.liveness_passed, l.device_id,
                   l.latitude, l.longitude, l.timestamp as event_timestamp,
                   l.metadata
            FROM sync_queue sq
            JOIN logs l ON sq.log_id = l.log_id
            WHERE sq.status IN ('PENDING', 'FAILED_RETRYABLE')
            ORDER BY sq.enqueued_at ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        result = []
        for row in rows:
            entry = dict(row)
            if entry.get("metadata"):
                try:
                    entry["metadata"] = json.loads(entry["metadata"])
                except (json.JSONDecodeError, TypeError):
                    entry["metadata"] = {}
            result.append(entry)
        return result
    finally:
        conn.close()


def mark_sync_in_flight(queue_id: int, db_path: str = DB_PATH) -> None:
    """Mark a queue entry as currently being attempted (prevents duplicate sends)."""
    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            UPDATE sync_queue
            SET status = 'IN_FLIGHT', last_attempt_at = ?, attempt_count = attempt_count + 1
            WHERE queue_id = ?
            """,
            (_utc_now(), queue_id),
        )
        conn.commit()
    finally:
        conn.close()


def mark_sync_success(queue_id: int, log_id: str, db_path: str = DB_PATH) -> None:
    """
    CRITICAL SAFETY FUNCTION — Phase 5 Corrected Purge Logic.

    Called exclusively when AWS API Gateway returns HTTP 200 with a valid
    verification token confirming receipt of this log_id.

    Atomically performs a TRUE STRUCTURAL DELETE of the sync_queue row AND
    marks the log as synced=1 in the logs table.

    Design rationale:
      - The sync_queue row is HARD DELETED (not just status-flagged) because
        keeping resolved entries wastes storage and complicates queue queries.
      - The logs table entry is NEVER deleted — it is the immutable audit trail.
        Only the 'synced' flag is set to 1 to mark cloud confirmation.
      - Both operations execute in a single atomic transaction so a crash
        between them cannot leave the system in an inconsistent state.
    """
    conn = get_connection(db_path)
    try:
        with conn:  # Atomic — both operations succeed or both roll back
            # Phase 5: TRUE STRUCTURAL DELETE of the sync obligation
            conn.execute(
                "DELETE FROM sync_queue WHERE queue_id = ?", (queue_id,)
            )
            # Mark the immutable log record as cloud-confirmed
            conn.execute(
                "UPDATE logs SET synced = 1 WHERE log_id = ?", (log_id,)
            )
        logger.info(
            "Sync confirmed for log_id='%s' (queue_id=%d). "
            "Row HARD DELETED from sync_queue; audit log preserved.",
            log_id, queue_id,
        )
    finally:
        conn.close()


def purge_synced_log(log_id: str, db_path: str = DB_PATH) -> bool:
    """
    Phase 5 — True Structural Purge by log_id.

    Executes a direct DELETE FROM sync_queue WHERE log_id = ? command.
    This is the canonical purge entry point when only the log_id is known
    (e.g., called from an HTTP 200 handler that receives a log_id list rather
    than the internal queue_id).

    Also marks logs.synced = 1 atomically.

    Returns True if a row was actually deleted, False if log_id was not present.
    """
    conn = get_connection(db_path)
    try:
        with conn:
            cursor = conn.execute(
                "DELETE FROM sync_queue WHERE log_id = ?", (log_id,)
            )
            conn.execute(
                "UPDATE logs SET synced = 1 WHERE log_id = ?", (log_id,)
            )
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info(
                "purge_synced_log: HARD DELETED sync_queue row for log_id='%s'.", log_id
            )
        else:
            logger.debug(
                "purge_synced_log: log_id='%s' was not in queue (already purged?).", log_id
            )
        return deleted
    finally:
        conn.close()


def mark_sync_failed(
    queue_id: int,
    error_message: str,
    db_path: str = DB_PATH,
) -> None:
    """
    Mark a sync entry as failed but retryable.
    The sync engine will pick it up again on the next cycle with exponential backoff.
    """
    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            UPDATE sync_queue
            SET status = 'FAILED_RETRYABLE', last_attempt_at = ?, last_error = ?
            WHERE queue_id = ?
            """,
            (_utc_now(), error_message[:512], queue_id),
        )
        conn.commit()
        logger.warning("Sync failed for queue_id=%d: %s", queue_id, error_message)
    finally:
        conn.close()


def delete_synced_queue_entries(db_path: str = DB_PATH) -> int:
    """
    Crash Recovery: reset stale IN_FLIGHT entries that were never resolved
    (e.g., after a process crash mid-flight). Resets them to PENDING so they retry.

    Note: this does NOT hard-delete — re-queued entries will be hard-deleted
    only when AWS returns HTTP 200 via mark_sync_success() or purge_synced_log().

    Returns the number of entries reset.
    """
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            "UPDATE sync_queue SET status = 'PENDING' WHERE status = 'IN_FLIGHT'"
        )
        conn.commit()
        count = cursor.rowcount
        if count > 0:
            logger.info("Reset %d stale IN_FLIGHT sync entries to PENDING.", count)
        return count
    finally:
        conn.close()


def get_sync_queue_stats(db_path: str = DB_PATH) -> Dict[str, int]:
    """Return a summary count of sync queue entries grouped by status."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT status, COUNT(*) as count FROM sync_queue GROUP BY status"
        ).fetchall()
        stats = {row["status"]: row["count"] for row in rows}
        logger.info("Sync queue stats: %s", stats)
        return stats
    finally:
        conn.close()
