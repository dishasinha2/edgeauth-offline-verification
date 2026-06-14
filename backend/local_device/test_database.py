"""
test_database.py
----------------
Unit tests for database.py — schema initialization, CRUD operations,
idempotency guarantees, and sync queue lifecycle.

Run with:
    pytest test_database.py -v
"""

import json
import math
import random
import uuid
import pytest
import sqlite3

from database import (
    initialize_database,
    insert_organization,
    get_organization,
    list_organizations,
    insert_employee,
    get_employee,
    get_employees_by_organization,
    update_employee_embedding,
    deactivate_employee,
    insert_log,
    get_log,
    get_unsynced_logs,
    mark_log_as_synced,
    enqueue_sync,
    get_pending_sync_entries,
    mark_sync_in_flight,
    mark_sync_success,
    mark_sync_failed,
    delete_synced_queue_entries,
    get_sync_queue_stats,
    purge_synced_log,
)


# =============================================================================
# Schema Tests
# =============================================================================

class TestSchemaInitialization:

    def test_all_tables_created(self, db_path):
        conn = sqlite3.connect(db_path)
        tables = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()
        assert {"organizations", "employees", "logs", "sync_queue"}.issubset(tables)

    def test_idempotent_initialization(self, db_path):
        """Calling initialize_database twice must not raise or corrupt data."""
        initialize_database(db_path=db_path)
        initialize_database(db_path=db_path)

    def test_foreign_keys_enforced(self, db_path):
        """Inserting an employee with a nonexistent org_id must fail."""
        with pytest.raises(Exception):
            insert_employee(
                organization_id="nonexistent-org-id",
                full_name="Ghost Employee",
                face_embedding=[0.1] * 512,
                db_path=db_path,
            )

    def test_wal_journal_mode(self, db_path):
        conn = sqlite3.connect(db_path)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        assert mode == "wal", f"Expected WAL, got '{mode}'"


# =============================================================================
# Organization Tests
# =============================================================================

class TestOrganizations:

    def test_insert_and_retrieve(self, db_path):
        org_id = insert_organization("Acme Corp", "US-EAST", db_path=db_path)
        org = get_organization(org_id, db_path=db_path)
        assert org is not None
        assert org["name"] == "Acme Corp"
        assert org["region"] == "US-EAST"
        assert org["is_active"] == 1

    def test_insert_returns_valid_uuid(self, db_path):
        org_id = insert_organization("Test Org", "APAC", db_path=db_path)
        uuid.UUID(org_id)  # Must not raise ValueError

    def test_custom_organization_id(self, db_path):
        custom_id = str(uuid.uuid4())
        returned_id = insert_organization(
            "Custom ID Org", "EMEA",
            organization_id=custom_id,
            db_path=db_path,
        )
        assert returned_id == custom_id
        org = get_organization(custom_id, db_path=db_path)
        assert org is not None

    def test_duplicate_organization_id_rejected(self, db_path):
        org_id = insert_organization("First Org", "US-WEST", db_path=db_path)
        with pytest.raises(Exception):
            insert_organization(
                "Second Org", "US-WEST",
                organization_id=org_id,
                db_path=db_path,
            )

    def test_get_nonexistent_organization(self, db_path):
        result = get_organization("does-not-exist", db_path=db_path)
        assert result is None

    def test_list_organizations(self, db_path):
        insert_organization("Org Alpha", "APAC", db_path=db_path)
        insert_organization("Org Beta",  "EMEA", db_path=db_path)
        orgs = list_organizations(db_path=db_path)
        assert len(orgs) == 2
        names = {o["name"] for o in orgs}
        assert "Org Alpha" in names
        assert "Org Beta"  in names

    def test_list_organizations_alphabetical(self, db_path):
        insert_organization("Zebra Corp", "US",   db_path=db_path)
        insert_organization("Apple Inc",  "APAC", db_path=db_path)
        orgs = list_organizations(db_path=db_path)
        assert orgs[0]["name"] == "Apple Inc"
        assert orgs[1]["name"] == "Zebra Corp"


# =============================================================================
# Employee Tests
# =============================================================================

class TestEmployees:

    def _make_embedding(self, seed: int, dims: int = 512) -> list:
        rng = random.Random(seed)
        raw = [rng.gauss(0, 1) for _ in range(dims)]
        mag = math.sqrt(sum(v ** 2 for v in raw))
        return [round(v / mag, 8) for v in raw]

    def test_insert_and_retrieve(self, db_path, sample_embedding_512d):
        org_id = insert_organization("Org", "APAC", db_path=db_path)
        emp_id = insert_employee(
            organization_id=org_id,
            full_name="Jane Doe",
            face_embedding=sample_embedding_512d,
            department="HR",
            role="Manager",
            db_path=db_path,
        )
        emp = get_employee(emp_id, db_path=db_path)
        assert emp is not None
        assert emp["full_name"] == "Jane Doe"
        assert emp["department"] == "HR"
        assert emp["is_active"] == 1

    def test_embedding_round_trip(self, db_path, sample_embedding_512d):
        """Stored embedding must be bit-for-bit identical after JSON round-trip."""
        org_id = insert_employee_org(db_path)
        emp_id = insert_employee(
            organization_id=org_id,
            full_name="Embed Test",
            face_embedding=sample_embedding_512d,
            db_path=db_path,
        )
        emp = get_employee(emp_id, db_path=db_path)
        assert emp["face_embedding"] == sample_embedding_512d

    def test_embedding_dimensions(self, db_path, sample_embedding_512d):
        org_id = insert_employee_org(db_path)
        emp_id = insert_employee(
            organization_id=org_id,
            full_name="Dims Test",
            face_embedding=sample_embedding_512d,
            db_path=db_path,
        )
        emp = get_employee(emp_id, db_path=db_path)
        assert len(emp["face_embedding"]) == 512

    def test_get_employees_by_organization(self, db_path, sample_embedding_512d):
        org_id = insert_organization("Multi Emp Org", "TEST", db_path=db_path)
        for i in range(3):
            insert_employee(
                organization_id=org_id,
                full_name=f"Employee {i}",
                face_embedding=self._make_embedding(i),
                db_path=db_path,
            )
        employees = get_employees_by_organization(org_id, db_path=db_path)
        assert len(employees) == 3

    def test_update_embedding(self, db_path, sample_embedding_512d):
        org_id = insert_employee_org(db_path)
        emp_id = insert_employee(
            organization_id=org_id,
            full_name="Update Test",
            face_embedding=sample_embedding_512d,
            db_path=db_path,
        )
        new_embedding = self._make_embedding(seed=999)
        success = update_employee_embedding(
            emp_id, new_embedding, "arcface_v2", db_path=db_path
        )
        assert success is True
        emp = get_employee(emp_id, db_path=db_path)
        assert emp["face_embedding"] == new_embedding
        assert emp["embedding_model"] == "arcface_v2"

    def test_deactivate_employee(self, db_path, sample_embedding_512d):
        org_id = insert_employee_org(db_path)
        emp_id = insert_employee(
            organization_id=org_id,
            full_name="Deactivate Test",
            face_embedding=sample_embedding_512d,
            db_path=db_path,
        )
        result = deactivate_employee(emp_id, db_path=db_path)
        assert result is True
        emp = get_employee(emp_id, db_path=db_path)
        assert emp["is_active"] == 0

    def test_deactivated_employees_hidden_from_list(self, db_path, sample_embedding_512d):
        org_id = insert_organization("Hidden Org", "TEST", db_path=db_path)
        emp_id = insert_employee(
            organization_id=org_id,
            full_name="Hidden",
            face_embedding=sample_embedding_512d,
            db_path=db_path,
        )
        deactivate_employee(emp_id, db_path=db_path)
        employees = get_employees_by_organization(org_id, db_path=db_path)
        assert len(employees) == 0


# =============================================================================
# Log Tests
# =============================================================================

class TestLogs:

    def test_insert_log_success(self, seeded_db):
        data = seeded_db
        assert len(data["log_ids"]) == 3
        for log_id in data["log_ids"]:
            log = get_log(log_id, db_path=data["db_path"])
            assert log is not None
            assert log["synced"] == 0  # All start as unsynced

    def test_insert_log_auto_enqueues(self, seeded_db):
        data = seeded_db
        pending = get_pending_sync_entries(db_path=data["db_path"])
        pending_ids = {e["log_id"] for e in pending}
        for log_id in data["log_ids"]:
            assert log_id in pending_ids, f"log_id '{log_id}' not found in sync queue"

    def test_duplicate_log_id_rejected(self, seeded_db):
        data = seeded_db
        fixed_id = str(uuid.uuid4())
        insert_log(
            organization_id=data["org_id"],
            employee_id=data["employee_ids"][0],
            event_type="CLOCK_IN",
            verification_score=0.95,
            liveness_score=0.92,
            liveness_passed=True,
            device_id="DEV-001",
            log_id=fixed_id,
            db_path=data["db_path"],
        )
        with pytest.raises(Exception):
            insert_log(
                organization_id=data["org_id"],
                employee_id=data["employee_ids"][0],
                event_type="CLOCK_IN",
                verification_score=0.95,
                liveness_score=0.92,
                liveness_passed=True,
                device_id="DEV-001",
                log_id=fixed_id,       # Same log_id — must be rejected
                db_path=data["db_path"],
            )

    def test_invalid_event_type_rejected(self, seeded_db):
        data = seeded_db
        with pytest.raises(ValueError, match="Invalid event_type"):
            insert_log(
                organization_id=data["org_id"],
                employee_id=data["employee_ids"][0],
                event_type="INVALID_EVENT",
                verification_score=0.9,
                liveness_score=0.9,
                liveness_passed=True,
                device_id="DEV-001",
                db_path=data["db_path"],
            )

    def test_get_unsynced_logs(self, seeded_db):
        data = seeded_db
        unsynced = get_unsynced_logs(db_path=data["db_path"])
        assert len(unsynced) == 3

    def test_get_unsynced_logs_by_org(self, seeded_db):
        data = seeded_db
        unsynced = get_unsynced_logs(
            organization_id=data["org_id"], db_path=data["db_path"]
        )
        assert len(unsynced) == 3

    def test_mark_log_synced(self, seeded_db):
        data = seeded_db
        log_id = data["log_ids"][0]
        success = mark_log_as_synced(log_id, db_path=data["db_path"])
        assert success is True
        log = get_log(log_id, db_path=data["db_path"])
        assert log["synced"] == 1

    def test_metadata_round_trip(self, seeded_db):
        data = seeded_db
        metadata = {"camera": "ArduCam", "firmware": "v2.1.4", "latency_ms": 87}
        log_id = insert_log(
            organization_id=data["org_id"],
            employee_id=data["employee_ids"][0],
            event_type="CLOCK_IN",
            verification_score=0.98,
            liveness_score=0.96,
            liveness_passed=True,
            device_id="DEV-META-TEST",
            metadata=metadata,
            db_path=data["db_path"],
        )
        log = get_log(log_id, db_path=data["db_path"])
        assert log["metadata"] == metadata


# =============================================================================
# Sync Queue Tests
# =============================================================================

class TestSyncQueue:

    def test_pending_entries_present_after_insert(self, seeded_db):
        data = seeded_db
        pending = get_pending_sync_entries(db_path=data["db_path"])
        assert len(pending) >= 3

    def test_mark_in_flight(self, seeded_db):
        data = seeded_db
        pending = get_pending_sync_entries(db_path=data["db_path"])
        queue_id = pending[0]["queue_id"]
        mark_sync_in_flight(queue_id, db_path=data["db_path"])
        stats = get_sync_queue_stats(db_path=data["db_path"])
        assert stats.get("IN_FLIGHT", 0) >= 1

    def test_mark_sync_success_removes_from_queue(self, seeded_db):
        data = seeded_db
        pending = get_pending_sync_entries(db_path=data["db_path"])
        entry   = pending[0]
        mark_sync_in_flight(entry["queue_id"], db_path=data["db_path"])
        mark_sync_success(entry["queue_id"], entry["log_id"], db_path=data["db_path"])

        remaining = get_pending_sync_entries(db_path=data["db_path"])
        remaining_ids = {e["log_id"] for e in remaining}
        assert entry["log_id"] not in remaining_ids

    def test_mark_sync_success_sets_synced_flag(self, seeded_db):
        data = seeded_db
        pending = get_pending_sync_entries(db_path=data["db_path"])
        entry   = pending[0]
        mark_sync_in_flight(entry["queue_id"], db_path=data["db_path"])
        mark_sync_success(entry["queue_id"], entry["log_id"], db_path=data["db_path"])

        log = get_log(entry["log_id"], db_path=data["db_path"])
        assert log["synced"] == 1, "Log must be marked synced=1 after cloud confirmation"

    def test_mark_sync_failed_keeps_in_queue(self, seeded_db):
        data = seeded_db
        pending = get_pending_sync_entries(db_path=data["db_path"])
        entry   = pending[0]
        mark_sync_in_flight(entry["queue_id"], db_path=data["db_path"])
        mark_sync_failed(entry["queue_id"], "Network timeout", db_path=data["db_path"])

        log = get_log(entry["log_id"], db_path=data["db_path"])
        assert log["synced"] == 0, "Log must remain synced=0 after a failed sync attempt"

        # Entry should still be present in queue as FAILED_RETRYABLE
        stats = get_sync_queue_stats(db_path=data["db_path"])
        assert stats.get("FAILED_RETRYABLE", 0) >= 1

    def test_crash_recovery_resets_in_flight(self, seeded_db):
        data = seeded_db
        pending = get_pending_sync_entries(db_path=data["db_path"])
        # Simulate a crash: mark entries as IN_FLIGHT but never resolve them
        for entry in pending:
            mark_sync_in_flight(entry["queue_id"], db_path=data["db_path"])

        stats_before = get_sync_queue_stats(db_path=data["db_path"])
        assert stats_before.get("IN_FLIGHT", 0) == len(pending)

        # Crash recovery resets IN_FLIGHT → PENDING
        reset_count = delete_synced_queue_entries(db_path=data["db_path"])
        assert reset_count == len(pending)

        stats_after = get_sync_queue_stats(db_path=data["db_path"])
        assert stats_after.get("IN_FLIGHT", 0) == 0
        assert stats_after.get("PENDING", 0) == len(pending)

    def test_enqueue_sync_is_idempotent(self, seeded_db):
        data = seeded_db
        log_id = data["log_ids"][0]
        # Enqueue the same log_id multiple times — should not raise or duplicate
        enqueue_sync(log_id, data["org_id"], db_path=data["db_path"])
        enqueue_sync(log_id, data["org_id"], db_path=data["db_path"])
        enqueue_sync(log_id, data["org_id"], db_path=data["db_path"])

        pending = get_pending_sync_entries(db_path=data["db_path"])
        matching = [e for e in pending if e["log_id"] == log_id]
        assert len(matching) == 1, "Idempotent enqueue must produce exactly one queue entry"

    def test_data_never_purged_without_aws_200(self, seeded_db):
        """
        CORE SAFETY INVARIANT TEST:
        Even after max retry failures, the raw log entry must persist in the
        logs table. Only the sync queue status changes.
        """
        data   = seeded_db
        log_id = data["log_ids"][0]

        # Simulate 10 consecutive failed attempts
        pending = get_pending_sync_entries(db_path=data["db_path"])
        entry   = next(e for e in pending if e["log_id"] == log_id)
        for _ in range(10):
            mark_sync_in_flight(entry["queue_id"], db_path=data["db_path"])
            mark_sync_failed(entry["queue_id"], "Simulated failure", db_path=data["db_path"])

        # The log entry itself must still be intact
        log = get_log(log_id, db_path=data["db_path"])
        assert log is not None, "Log MUST NOT be deleted even after repeated sync failures"
        assert log["synced"] == 0, "Log must remain synced=0 without HTTP 200 from AWS"


    def test_purge_synced_log_hard_deletes_queue_row(self, seeded_db):
        """
        purge_synced_log() must hard-delete the sync_queue row and mark
        the log as synced=1 atomically, called by log_id (not queue_id).
        """
        data   = seeded_db
        log_id = data["log_ids"][0]

        deleted = purge_synced_log(log_id, db_path=data["db_path"])
        assert deleted is True

        # Queue row must be gone
        remaining = get_pending_sync_entries(db_path=data["db_path"])
        remaining_ids = {e["log_id"] for e in remaining}
        assert log_id not in remaining_ids

        # Audit log must still exist and be marked synced
        log = get_log(log_id, db_path=data["db_path"])
        assert log is not None
        assert log["synced"] == 1

    def test_purge_synced_log_returns_false_for_unknown_log_id(self, seeded_db):
        """Purging a log_id not in the queue must return False (idempotent)."""
        data = seeded_db
        result = purge_synced_log("nonexistent-log-id", db_path=data["db_path"])
        assert result is False


# =============================================================================
# Multi-Organization Isolation Tests  (Task 1E)
# =============================================================================

class TestMultiOrgIsolation:
    """
    Verify that get_employees_by_organization() returns ONLY the employees
    belonging to the requested organization, never bleeding across org boundaries.
    This is the critical multi-tenancy safety invariant.
    """

    def _make_embedding(self, seed: int, dims: int = 512) -> list:
        import math, random as _rnd
        rng = _rnd.Random(seed)
        raw = [rng.gauss(0, 1) for _ in range(dims)]
        mag = math.sqrt(sum(v ** 2 for v in raw))
        return [round(v / mag, 8) for v in raw]

    def test_org_a_does_not_see_org_b_employees(self, db_path):
        """
        Employees enrolled under Org B must NEVER appear in Org A's query
        and vice versa.
        """
        # Create two completely separate organizations
        org_a_id = insert_organization("Org Alpha", "APAC", db_path=db_path)
        org_b_id = insert_organization("Org Beta",  "EMEA", db_path=db_path)

        # Enroll one employee in each org
        emp_a_id = insert_employee(
            organization_id=org_a_id,
            full_name="Alice (Org Alpha)",
            face_embedding=self._make_embedding(seed=1),
            db_path=db_path,
        )
        emp_b_id = insert_employee(
            organization_id=org_b_id,
            full_name="Bob (Org Beta)",
            face_embedding=self._make_embedding(seed=2),
            db_path=db_path,
        )

        # Query each org's employee list
        alpha_employees = get_employees_by_organization(org_a_id, db_path=db_path)
        beta_employees  = get_employees_by_organization(org_b_id, db_path=db_path)

        alpha_ids = {emp["employee_id"] for emp in alpha_employees}
        beta_ids  = {emp["employee_id"] for emp in beta_employees}

        # Org A must contain Alice, NOT Bob
        assert emp_a_id in alpha_ids, "Alice must appear in Org Alpha's employee list"
        assert emp_b_id not in alpha_ids, (
            "Bob (Org Beta) must NOT appear in Org Alpha's employee list — "
            "multi-org isolation violated!"
        )

        # Org B must contain Bob, NOT Alice
        assert emp_b_id in beta_ids, "Bob must appear in Org Beta's employee list"
        assert emp_a_id not in beta_ids, (
            "Alice (Org Alpha) must NOT appear in Org Beta's employee list — "
            "multi-org isolation violated!"
        )

    def test_org_isolation_with_multiple_employees(self, db_path):
        """
        Multiple employees in each org — cross-org leakage must be zero.
        """
        org_x_id = insert_organization("Org X", "US-EAST", db_path=db_path)
        org_y_id = insert_organization("Org Y", "US-WEST", db_path=db_path)

        # Insert 3 employees in Org X
        x_ids = set()
        for i in range(3):
            x_ids.add(insert_employee(
                organization_id=org_x_id,
                full_name=f"X-Employee-{i}",
                face_embedding=self._make_embedding(seed=10 + i),
                db_path=db_path,
            ))

        # Insert 2 employees in Org Y
        y_ids = set()
        for i in range(2):
            y_ids.add(insert_employee(
                organization_id=org_y_id,
                full_name=f"Y-Employee-{i}",
                face_embedding=self._make_embedding(seed=20 + i),
                db_path=db_path,
            ))

        x_results = {e["employee_id"] for e in get_employees_by_organization(org_x_id, db_path=db_path)}
        y_results = {e["employee_id"] for e in get_employees_by_organization(org_y_id, db_path=db_path)}

        # Correct counts
        assert len(x_results) == 3, f"Org X should have 3 employees, got {len(x_results)}"
        assert len(y_results) == 2, f"Org Y should have 2 employees, got {len(y_results)}"

        # No overlap between the two sets
        overlap = x_results & y_results
        assert not overlap, (
            f"Cross-org isolation failed — {len(overlap)} employee(s) appear in both orgs: {overlap}"
        )

    def test_empty_org_returns_empty_list(self, db_path):
        """An org with no employees must return an empty list, not None or an error."""
        org_id = insert_organization("Empty Org", "TEST", db_path=db_path)
        employees = get_employees_by_organization(org_id, db_path=db_path)
        assert employees == [], (
            f"Empty org should return [], got {employees!r}"
        )


# =============================================================================
# Helpers
# =============================================================================

def insert_employee_org(db_path: str) -> str:
    """Helper to quickly seed an organization for employee tests."""
    return insert_organization(
        f"Test Org {uuid.uuid4().hex[:6]}",
        "TEST",
        db_path=db_path,
    )
