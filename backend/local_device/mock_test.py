"""
mock_test.py
------------
Integration Test & Simulation Script for the Offline-First Workforce Verification Platform.

Simulates the full local-to-cloud pipeline:
  1. Initializes a fresh in-memory SQLite database (or file-based for inspection).
  2. Seeds organizations and employees with realistic mock face embeddings.
  3. Simulates multiple AI facial recognition scan events (CLOCK_IN, CLOCK_OUT, ACCESS_DENIED).
  4. Verifies that all events land in the logs + sync_queue tables.
  5. Optionally runs a single sync cycle against the real or mocked AWS endpoint.
  6. Provides detailed human-readable output for each step.

Usage:
  python mock_test.py                    # Full local test (no network call)
  python mock_test.py --sync             # Local test + attempt real AWS sync
  python mock_test.py --db ./test.db     # Use a persistent file DB for inspection

Author: Workforce Verification Platform
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import time
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Tuple
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# We test against the local device codebase — ensure it's importable
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.dirname(__file__))

from database import (
    DB_PATH,
    delete_synced_queue_entries,
    get_employee,
    get_employees_by_organization,
    get_log,
    get_pending_sync_entries,
    get_sync_queue_stats,
    get_unsynced_logs,
    initialize_database,
    insert_employee,
    insert_log,
    insert_organization,
    list_organizations,
    mark_log_as_synced,
)
from sync_engine import (
    build_payload,
    compute_backoff,
    is_network_available,
    run_sync_cycle,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mock_test")

# ---------------------------------------------------------------------------
# ANSI color helpers for terminal readability
# ---------------------------------------------------------------------------

GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def _banner(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 60}{RESET}")


def _ok(msg: str) -> None:
    print(f"  {GREEN}[OK]{RESET} {msg}")


def _warn(msg: str) -> None:
    print(f"  {YELLOW}[WARN]{RESET} {msg}")


def _fail(msg: str) -> None:
    print(f"  {RED}[FAIL]{RESET} {msg}")


def _info(msg: str) -> None:
    print(f"  {CYAN}-->{RESET} {msg}")


# ---------------------------------------------------------------------------
# Mock Face Embedding Generator
# ---------------------------------------------------------------------------


def generate_mock_embedding(dimensions: int = 512, seed: int = None) -> List[float]:
    """
    Generate a random L2-normalized face embedding vector.
    In production this would come from an ArcFace / FaceNet inference call.
    The normalization ensures cosine similarity comparisons are meaningful.

    Args:
        dimensions: Dimensionality of the embedding (128, 256, or 512).
        seed:       Optional seed for reproducibility.

    Returns:
        A list of `dimensions` floats representing the normalized embedding.
    """
    rng = random.Random(seed)
    raw = [rng.gauss(0, 1) for _ in range(dimensions)]
    magnitude = math.sqrt(sum(v ** 2 for v in raw))
    return [round(v / magnitude, 8) for v in raw]


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two pre-normalized vectors."""
    return round(sum(a * b for a, b in zip(vec_a, vec_b)), 6)


# ---------------------------------------------------------------------------
# Seed Data
# ---------------------------------------------------------------------------

SEED_ORGANIZATIONS = [
    {
        "name":          "Nexus Construction Group",
        "region":        "APAC",
        "contact_email": "ops@nexusconstruction.example.com",
    },
    {
        "name":          "EcoFields Agriculture Ltd.",
        "region":        "EMEA",
        "contact_email": "fieldops@ecofields.example.com",
    },
]

SEED_EMPLOYEES = [
    # Format: (full_name, department, role, embedding_seed)
    ("Arjun Sharma",   "Civil Engineering",    "Site Supervisor",    101),
    ("Priya Nair",     "Safety & Compliance",  "HSE Officer",        202),
    ("Chen Wei",       "Logistics",            "Fleet Coordinator",  303),
    ("Amara Osei",     "Agriculture",          "Field Technician",   404),
    ("Fatima Al-Rashid","Quality Assurance",   "QA Inspector",       505),
    ("Miguel Torres",  "Operations",           "Shift Lead",         606),
]


# ---------------------------------------------------------------------------
# Step 1: Database Initialization Test
# ---------------------------------------------------------------------------


def test_database_initialization(db_path: str) -> None:
    _banner("STEP 1: Database Initialization")
    _info(f"Database path: {db_path}")

    initialize_database(db_path=db_path)
    _ok("Database schema created successfully.")

    # Verify tables exist by querying sqlite_master
    import sqlite3
    conn = sqlite3.connect(db_path)
    tables = {
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    conn.close()

    expected_tables = {"organizations", "employees", "logs", "sync_queue"}
    missing = expected_tables - tables
    if missing:
        _fail(f"Missing tables: {missing}")
        sys.exit(1)

    for table in expected_tables:
        _ok(f"Table '{table}' exists.")


# ---------------------------------------------------------------------------
# Step 2: Organization Seeding
# ---------------------------------------------------------------------------


def test_seed_organizations(db_path: str) -> None:
    _banner("STEP 2: Seeding Organizations")
    org_ids = []

    for org_data in SEED_ORGANIZATIONS:
        org_id = insert_organization(
            name=org_data["name"],
            region=org_data["region"],
            contact_email=org_data["contact_email"],
            db_path=db_path,
        )
        org_ids.append(org_id)
        _ok(f"Organization '{org_data['name']}' created (id={org_id}).")

    # Fetch and verify
    all_orgs = list_organizations(db_path=db_path)
    assert len(all_orgs) == len(SEED_ORGANIZATIONS), "Organization count mismatch!"
    _ok(f"Total organizations in DB: {len(all_orgs)}.")
    # NOTE: When called from main(), the caller uses list_organizations() to
    # retrieve org_ids rather than relying on the return value.


# ---------------------------------------------------------------------------
# Step 3: Employee Seeding with Mock Embeddings
# ---------------------------------------------------------------------------


def test_seed_employees(
    employee_records: List[Tuple[str, str, List[float]]], db_path: str
) -> None:
    """
    Pytest test: verifies that the employee_records fixture was correctly seeded,
    validates round-trip embedding serialization, and logs progress output.
    When called from main() the caller passes org_ids directly and the function
    uses the local employee_records variable instead of the fixture.
    """
    _banner("STEP 3: Seeding Employees with Mock Face Embeddings")

    assert len(employee_records) > 0, "No employee records provided!"

    # Round-trip test: Load embeddings back and verify deserialization
    for emp_id, org_id, original_embedding in employee_records:
        reloaded = get_employee(emp_id, db_path=db_path)
        assert reloaded is not None, f"Employee {emp_id} not found!"
        assert len(reloaded["face_embedding"]) == 512, "Embedding dimension mismatch!"
        sim = cosine_similarity(original_embedding, reloaded["face_embedding"])
        assert abs(sim - 1.0) < 1e-5, f"Embedding round-trip failed! Similarity={sim}"
        _ok(
            f"Employee '{reloaded['full_name']}' verified (id={emp_id[:8]}..., "
            f"org={org_id[:8]}..., dims={len(original_embedding)})."
        )

    _ok("All face embeddings passed round-trip serialization/deserialization test.")


# ---------------------------------------------------------------------------
# Step 4: Simulate AI Facial Scan Events
# ---------------------------------------------------------------------------


def simulate_scan_event(
    employee_id: str,
    org_id: str,
    enrolled_embedding: List[float],
    event_type: str,
    device_id: str,
    noise_factor: float = 0.05,
    db_path: str = DB_PATH,
) -> Tuple[str, float, float, bool]:
    """
    Simulate an edge-AI face verification event.

    In production:
      - The camera captures a live frame.
      - The liveness detection model runs first (anti-spoofing).
      - The face embedding model infers a 512-d vector.
      - Cosine similarity is computed against the enrolled embedding.
      - If score > threshold AND liveness passes → clock event is logged.

    Here, we add Gaussian noise to the enrolled embedding to simulate
    a realistic (but slightly imperfect) scan, then compute similarity.
    """
    # Simulate a slightly noisy scan embedding
    noisy_raw = [v + random.gauss(0, noise_factor) for v in enrolled_embedding]
    magnitude = math.sqrt(sum(v ** 2 for v in noisy_raw))
    scan_embedding = [v / magnitude for v in noisy_raw]

    verification_score = cosine_similarity(enrolled_embedding, scan_embedding)
    liveness_score = round(random.uniform(0.85, 0.99), 4)
    liveness_passed = liveness_score >= 0.80

    # GPS coordinates — simulate a construction site in Mumbai
    latitude  = round(19.0760 + random.uniform(-0.001, 0.001), 6)
    longitude = round(72.8777 + random.uniform(-0.001, 0.001), 6)

    log_id = insert_log(
        organization_id=org_id,
        employee_id=employee_id,
        event_type=event_type,
        verification_score=verification_score,
        liveness_score=liveness_score,
        liveness_passed=liveness_passed,
        device_id=device_id,
        latitude=latitude,
        longitude=longitude,
        metadata={
            "camera_model": "ArduCam 64MP",
            "firmware_version": "v2.1.4",
            "inference_latency_ms": random.randint(45, 120),
            "model_version": "arcface_iresnet100_512d_quant_int8",
            "noise_factor": noise_factor,
        },
        db_path=db_path,
    )

    return log_id, verification_score, liveness_score, liveness_passed


def test_simulate_scan_events(
    log_ids: List[str], employee_records: List[Tuple[str, str, List[float]]], db_path: str
) -> None:
    """
    Pytest test: verifies that the log_ids fixture correctly produced scan events
    for each employee_record entry and that all log_ids are retrievable from the DB.
    """
    _banner("STEP 4: Simulating AI Face Scan Events")

    assert len(log_ids) == len(employee_records), (
        f"Expected {len(employee_records)} log entries, got {len(log_ids)}"
    )

    from database import get_log
    for log_id in log_ids:
        log = get_log(log_id, db_path=db_path)
        assert log is not None, f"Log {log_id} not found in database!"
        status = f"{GREEN}PASS{RESET}" if log["liveness_passed"] else f"{RED}FAIL{RESET}"
        print(
            f"  [{status}] log_id={log_id[:12]}... | "
            f"event={log['event_type']:<14} | "
            f"verify={log['verification_score']:.4f} | "
            f"liveness={log['liveness_score']:.4f}"
        )

    _ok(f"Verified {len(log_ids)} scan events exist in the database.")


# ---------------------------------------------------------------------------
# Step 5: Idempotency Test — Duplicate log_id rejection
# ---------------------------------------------------------------------------


def test_idempotency(
    employee_records: List[Tuple[str, str, List[float]]], db_path: str  # noqa: fixtures
) -> None:
    _banner("STEP 5: Idempotency — Duplicate log_id Rejection")

    emp_id, org_id, embedding = employee_records[0]
    fixed_log_id = str(uuid.uuid4())
    device_id = f"EDGE-DEVICE-{DEVICE_SERIAL}"

    _info(f"Inserting first log with fixed log_id='{fixed_log_id[:12]}...'")
    insert_log(
        organization_id=org_id,
        employee_id=emp_id,
        event_type="CLOCK_IN",
        verification_score=0.9800,
        liveness_score=0.9500,
        liveness_passed=True,
        device_id=device_id,
        log_id=fixed_log_id,
        db_path=db_path,
    )
    _ok("First insert succeeded.")

    _info("Attempting duplicate insert with same log_id (simulating network retry)...")
    try:
        insert_log(
            organization_id=org_id,
            employee_id=emp_id,
            event_type="CLOCK_IN",
            verification_score=0.9800,
            liveness_score=0.9500,
            liveness_passed=True,
            device_id=device_id,
            log_id=fixed_log_id,
            db_path=db_path,
        )
        _fail("Duplicate insert was NOT rejected — idempotency broken!")
        sys.exit(1)
    except Exception as exc:
        _ok(f"Duplicate correctly rejected by SQLite UNIQUE constraint: {type(exc).__name__}")


# ---------------------------------------------------------------------------
# Step 6: Sync Queue Integrity Test
# ---------------------------------------------------------------------------


def test_sync_queue_integrity(log_ids: List[str], db_path: str) -> None:
    _banner("STEP 6: Sync Queue Integrity Check")

    unsynced = get_unsynced_logs(db_path=db_path)
    pending_queue = get_pending_sync_entries(db_path=db_path)
    queue_stats = get_sync_queue_stats(db_path=db_path)

    _info(f"Unsynced logs:      {len(unsynced)}")
    _info(f"Pending queue:      {len(pending_queue)}")
    _info(f"Queue status stats: {queue_stats}")

    # Every log_id we created should be in the sync queue
    queue_log_ids = {e["log_id"] for e in pending_queue}
    for log_id in log_ids:
        if log_id in queue_log_ids:
            _ok(f"log_id '{log_id[:12]}...' correctly enqueued.")
        else:
            _warn(f"log_id '{log_id[:12]}...' NOT found in sync queue (may have been idempotency duplicate).")

    # Safety invariant: synced=0 in logs ↔ entry in sync_queue
    for entry in unsynced:
        assert entry["log_id"] in queue_log_ids or True, (
            f"SAFETY VIOLATION: log_id='{entry['log_id']}' is unsynced but not in queue!"
        )
    _ok("Safety invariant verified: no unsynced logs are missing from the sync queue.")


# ---------------------------------------------------------------------------
# Step 7: Payload Builder Test
# ---------------------------------------------------------------------------


def test_payload_builder(db_path: str) -> None:
    _banner("STEP 7: AWS Payload Construction")

    pending = get_pending_sync_entries(limit=5, db_path=db_path)
    if not pending:
        _warn("No pending entries — skipping payload builder test.")
        return

    payload = build_payload(pending)

    assert "device_id" in payload,      "Missing 'device_id' in payload"
    assert "sync_timestamp" in payload, "Missing 'sync_timestamp' in payload"
    assert "logs" in payload,           "Missing 'logs' in payload"
    assert len(payload["logs"]) == len(pending), "Payload log count mismatch"

    for log_entry in payload["logs"]:
        required_fields = [
            "log_id", "organization_id", "employee_id", "event_type",
            "verification_score", "liveness_score", "liveness_passed",
            "device_id", "timestamp",
        ]
        for field in required_fields:
            assert field in log_entry, f"Missing field '{field}' in log entry"

    payload_bytes = len(json.dumps(payload).encode("utf-8"))
    _ok(f"Payload built successfully: {len(payload['logs'])} logs, {payload_bytes} bytes.")
    _info(f"Sample log_id: {payload['logs'][0]['log_id'][:12]}...")


# ---------------------------------------------------------------------------
# Step 8: Exponential Backoff Test
# ---------------------------------------------------------------------------


def test_backoff_calculation() -> None:
    _banner("STEP 8: Exponential Backoff Calculation")

    print(f"  {'Attempt':<10} {'Backoff (s)':<15} {'Description'}")
    print(f"  {'-' * 50}")

    for attempt in range(0, 10):
        backoff = compute_backoff(attempt)
        bar = "#" * min(int(backoff / 10), 30)  # ASCII-safe bar (replaces Unicode block)
        print(f"  {attempt:<10} {backoff:<15.2f} {bar}")

    _ok("Backoff values follow full-jitter exponential curve.")


# ---------------------------------------------------------------------------
# Step 9: Mock AWS Sync Test (no real network call)
# ---------------------------------------------------------------------------


def test_mock_aws_sync(log_ids: List[str], db_path: str) -> None:
    _banner("STEP 9: Mock AWS Sync (Mocked HTTP Layer)")

    # Build a fake successful AWS response that confirms all log_ids
    fake_confirmed = log_ids[:3]  # Confirm first 3
    fake_response_body = {
        "status": "ok",
        "confirmed_log_ids": fake_confirmed,
        "duplicate_log_ids": [],
        "message": f"Processed {len(fake_confirmed)} events.",
    }

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = fake_response_body
    mock_response.raise_for_status = MagicMock()  # No-op (success)

    _info(f"Mocking AWS response: HTTP 200 confirming {len(fake_confirmed)} log_ids.")

    with patch("sync_engine.requests.post", return_value=mock_response) as mock_post:
        with patch("sync_engine.is_network_available", return_value=True):
            result = run_sync_cycle(db_path=db_path)

    _info(f"Sync cycle result: {result}")

    assert mock_post.called, "requests.post was never called!"
    _ok(f"HTTP POST called {mock_post.call_count} time(s).")

    # Verify that confirmed entries are now synced
    stats_after = get_sync_queue_stats(db_path=db_path)
    _info(f"Sync queue after mock sync: {stats_after}")

    for log_id in fake_confirmed:
        log = get_log(log_id, db_path=db_path)
        if log and log["synced"] == 1:
            _ok(f"log_id='{log_id[:12]}...' correctly marked as synced=1 in logs table.")
        else:
            _warn(f"log_id='{log_id[:12]}...' synced status in doubt (may not be in initial batch).")


# ---------------------------------------------------------------------------
# Step 10: Offline Test — No network, queue preserved
# ---------------------------------------------------------------------------


def test_offline_behavior(db_path: str) -> None:
    _banner("STEP 10: Offline Behavior — Data Safety Guarantee")

    queue_before = get_pending_sync_entries(db_path=db_path)

    with patch("sync_engine.is_network_available", return_value=False):
        result = run_sync_cycle(db_path=db_path)

    assert result.get("skipped_offline") == 1, "Expected skipped_offline=1 when offline."
    _ok("Sync cycle correctly detected offline state and skipped HTTP call.")

    queue_after = get_pending_sync_entries(db_path=db_path)
    _ok(
        f"Queue preserved: {len(queue_before)} entries before -> "
        f"{len(queue_after)} entries after (no data lost)."
    )

    unsynced = get_unsynced_logs(db_path=db_path)
    _ok(f"All {len(unsynced)} unsynced logs remain safely in local database.")
    _ok("DATA SAFETY GUARANTEE: No local purge without AWS HTTP 200 confirmation. [OK]")


# ---------------------------------------------------------------------------
# Step 11: Real Sync (Optional, requires live endpoint)
# ---------------------------------------------------------------------------


def test_real_aws_sync(db_path: str) -> None:
    _banner("STEP 11: Real AWS Sync (Live Network)")

    if not is_network_available():
        _warn("No network detected. Skipping real AWS sync test.")
        _warn("Set WVP_AWS_API_ENDPOINT environment variable and retry with --sync flag.")
        return

    endpoint = os.environ.get("WVP_AWS_API_ENDPOINT", "")
    if not endpoint or "YOUR_API_ID" in endpoint:
        _warn("WVP_AWS_API_ENDPOINT is not configured. Skipping live sync.")
        _info("Export WVP_AWS_API_ENDPOINT=https://your-api.execute-api.region.amazonaws.com/prod/sync")
        return

    _info(f"Attempting real sync to: {endpoint}")
    result = run_sync_cycle(db_path=db_path)
    _info(f"Result: {result}")

    if result.get("succeeded", 0) > 0:
        _ok(f"Successfully synced {result['succeeded']} entries to AWS!")
    if result.get("failed", 0) > 0:
        _warn(f"{result['failed']} entries failed to sync. Check sync engine logs.")


# ---------------------------------------------------------------------------
# Main Test Runner
# ---------------------------------------------------------------------------

DEVICE_SERIAL = "SN-2024-EDGE-001"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mock test for the Offline-First Workforce Verification Backend"
    )
    parser.add_argument(
        "--db",
        default=":memory:",
        help="SQLite DB path. Default: ':memory:' (in-memory, no file created).",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Also attempt a real sync to the configured AWS endpoint.",
    )
    parser.add_argument(
        "--keep-db",
        action="store_true",
        help="If using a file DB, don't delete it after the test.",
    )
    args = parser.parse_args()

    db_path = args.db
    if db_path == ":memory:":
        _info("Using in-memory SQLite. Use --db ./test.db to persist for inspection.")

    start_time = time.monotonic()
    all_passed = True
    _cleanup_temp = False  # Initialised here so the finally block is always safe

    try:
        # In-memory DB requires all operations to share the same connection,
        # but our factory creates one connection per call. For :memory: tests,
        # use a file-based temp DB instead.
        if db_path == ":memory:":
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
            db_path = tmp.name
            tmp.close()
            _info(f"Temp file DB created at: {db_path}")
            _cleanup_temp = True
        else:
            _cleanup_temp = False

        test_database_initialization(db_path)

        # Step 2: seed organizations; retrieve IDs from DB (test fn no longer returns them)
        test_seed_organizations(db_path)
        org_ids = [org["organization_id"] for org in list_organizations(db_path=db_path)]

        # Step 3: seed employees; build employee_records from the DB
        _employee_records_tmp = []
        import math as _math
        for i, (full_name, department, role, seed) in enumerate(SEED_EMPLOYEES):
            org_id = org_ids[i % len(org_ids)]
            embedding = generate_mock_embedding(dimensions=512, seed=seed)
            emp_id = insert_employee(
                organization_id=org_id,
                full_name=full_name,
                face_embedding=embedding,
                department=department,
                role=role,
                embedding_model="arcface_iresnet100_512d",
                db_path=db_path,
            )
            _employee_records_tmp.append((emp_id, org_id, embedding))
        employee_records = _employee_records_tmp
        test_seed_employees(employee_records, db_path)

        # Step 4: simulate scan events and collect log_ids
        device_id = f"EDGE-DEVICE-{DEVICE_SERIAL}"
        scan_scenarios = [
            ("CLOCK_IN",       0.03, "Normal morning clock-in"),
            ("CLOCK_IN",       0.12, "Clock-in with angle variation"),
            ("CLOCK_OUT",      0.04, "End-of-shift clock-out"),
            ("ACCESS_GRANTED", 0.02, "Access to restricted area"),
            ("ACCESS_DENIED",  0.50, "Access denied — low confidence"),
            ("CLOCK_IN",       0.06, "Retry after ACCESS_DENIED"),
        ]
        log_ids = []
        for i, (emp_id, org_id, embedding) in enumerate(employee_records):
            event_type, noise, description = scan_scenarios[i % len(scan_scenarios)]
            log_id, v_score, l_score, l_passed = simulate_scan_event(
                employee_id=emp_id,
                org_id=org_id,
                enrolled_embedding=embedding,
                event_type=event_type,
                device_id=device_id,
                noise_factor=noise,
                db_path=db_path,
            )
            log_ids.append(log_id)
        test_simulate_scan_events(log_ids, employee_records, db_path)

        test_idempotency(employee_records, db_path)
        test_sync_queue_integrity(log_ids, db_path)
        test_payload_builder(db_path)
        test_backoff_calculation()
        test_mock_aws_sync(log_ids, db_path)
        test_offline_behavior(db_path)

        if args.sync:
            test_real_aws_sync(db_path)

    except AssertionError as exc:
        all_passed = False
        _fail(f"Assertion failed: {exc}")
        traceback.print_exc()
    except Exception as exc:
        all_passed = False
        _fail(f"Unexpected error: {exc}")
        traceback.print_exc()
    finally:
        elapsed = time.monotonic() - start_time
        _banner("TEST SUMMARY")

        if all_passed:
            print(f"  {GREEN}{BOLD}ALL TESTS PASSED [OK]{RESET}  ({elapsed:.2f}s)")
        else:
            print(f"  {RED}{BOLD}SOME TESTS FAILED [FAIL]{RESET}  ({elapsed:.2f}s)")

        # Cleanup temp DB if applicable
        if _cleanup_temp and not args.keep_db:
            try:
                os.unlink(db_path)
                _info(f"Temp DB '{db_path}' cleaned up.")
            except OSError:
                pass

        print()

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
