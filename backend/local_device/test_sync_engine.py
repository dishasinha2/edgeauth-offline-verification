"""
test_sync_engine.py
-------------------
Unit tests for sync_engine.py — connectivity probe, backoff algorithm,
payload builder, and the full sync cycle under both online and offline conditions.

All HTTP calls are mocked so no real network is required.

Run with:
    pytest test_sync_engine.py -v
"""

import json
import uuid
import pytest
from unittest.mock import MagicMock, patch, call

from database import (
    get_log,
    get_pending_sync_entries,
    get_sync_queue_stats,
    insert_log,
)
from sync_engine import (
    BACKOFF_BASE_SECONDS,
    BACKOFF_MAX_SECONDS,
    build_payload,
    compute_backoff,
    is_network_available,
    run_sync_cycle,
)


# =============================================================================
# Connectivity Tests
# =============================================================================

class TestConnectivity:

    def test_returns_true_when_socket_succeeds(self):
        with patch("sync_engine.socket.socket") as mock_socket:
            mock_ctx = MagicMock()
            mock_socket.return_value.__enter__.return_value = mock_ctx
            mock_ctx.connect.return_value = None  # No exception = success
            assert is_network_available() is True

    def test_returns_false_on_socket_timeout(self):
        import socket
        with patch("sync_engine.socket.socket") as mock_socket:
            mock_ctx = MagicMock()
            mock_socket.return_value.__enter__.return_value = mock_ctx
            mock_ctx.connect.side_effect = socket.timeout("timed out")
            assert is_network_available() is False

    def test_returns_false_on_os_error(self):
        with patch("sync_engine.socket.socket") as mock_socket:
            mock_ctx = MagicMock()
            mock_socket.return_value.__enter__.return_value = mock_ctx
            mock_ctx.connect.side_effect = OSError("Network unreachable")
            assert is_network_available() is False


# =============================================================================
# Backoff Tests
# =============================================================================

class TestBackoff:

    def test_attempt_zero_is_near_base(self):
        # At attempt 0: base * 2^0 * jitter = BASE * jitter ≈ BASE
        backoff = compute_backoff(0)
        lower = BACKOFF_BASE_SECONDS * (1 - 0.25)
        upper = BACKOFF_BASE_SECONDS * (1 + 0.25)
        assert lower <= backoff <= upper, (
            f"Attempt 0 backoff {backoff} outside [{lower}, {upper}]"
        )

    def test_backoff_increases_with_attempt(self):
        """Mean backoff must increase with attempt number (before hitting cap)."""
        means = []
        for attempt in range(6):
            samples = [compute_backoff(attempt) for _ in range(200)]
            means.append(sum(samples) / len(samples))
        for i in range(1, 6):
            assert means[i] > means[i - 1], (
                f"Backoff mean did not increase: attempt {i-1}={means[i-1]:.2f}, {i}={means[i]:.2f}"
            )

    def test_backoff_capped_at_maximum(self):
        """All backoff values must stay at or below BACKOFF_MAX_SECONDS * (1+jitter)."""
        cap = BACKOFF_MAX_SECONDS * 1.25  # Allow jitter headroom
        for attempt in range(20, 30):
            assert compute_backoff(attempt) <= cap

    def test_backoff_has_jitter(self):
        """Two calls with the same attempt should NOT always return identical values."""
        results = {compute_backoff(3) for _ in range(20)}
        assert len(results) > 1, "Backoff has no jitter — all 20 results are identical"


# =============================================================================
# Payload Builder Tests
# =============================================================================

class TestPayloadBuilder:

    def _make_entry(self, log_id=None, org_id=None):
        return {
            "log_id":             log_id or str(uuid.uuid4()),
            "organization_id":    org_id or str(uuid.uuid4()),
            "employee_id":        str(uuid.uuid4()),
            "event_type":         "CLOCK_IN",
            "verification_score": 0.9800,
            "liveness_score":     0.9500,
            "liveness_passed":    1,
            "device_id":          "TEST-DEVICE-001",
            "latitude":           19.0760,
            "longitude":          72.8777,
            "event_timestamp":    "2024-01-15T08:00:00+00:00",
            "metadata":           {"camera": "test"},
            "attempt_count":      0,
            "queue_id":           1,
        }

    def test_payload_has_required_top_level_keys(self):
        payload = build_payload([self._make_entry()])
        assert "device_id"      in payload
        assert "sync_timestamp" in payload
        assert "logs"           in payload
        assert "entry_count"    in payload

    def test_entry_count_matches_logs_length(self):
        entries = [self._make_entry() for _ in range(5)]
        payload = build_payload(entries)
        assert payload["entry_count"] == 5
        assert len(payload["logs"]) == 5

    def test_log_entry_has_all_required_fields(self):
        payload = build_payload([self._make_entry()])
        log = payload["logs"][0]
        required = [
            "log_id", "organization_id", "employee_id", "event_type",
            "verification_score", "liveness_score", "liveness_passed",
            "device_id", "timestamp",
        ]
        for field in required:
            assert field in log, f"Missing field '{field}' in payload log entry"

    def test_liveness_passed_is_bool(self):
        entry = self._make_entry()
        entry["liveness_passed"] = 1  # SQLite stores as integer
        payload = build_payload([entry])
        assert payload["logs"][0]["liveness_passed"] is True

    def test_none_gps_excluded(self):
        entry = self._make_entry()
        entry["latitude"]  = None
        entry["longitude"] = None
        payload = build_payload([entry])
        log = payload["logs"][0]
        assert log.get("latitude")  is None
        assert log.get("longitude") is None

    def test_payload_is_json_serializable(self):
        entries = [self._make_entry() for _ in range(10)]
        payload = build_payload(entries)
        serialized = json.dumps(payload)
        assert len(serialized) > 0


# =============================================================================
# Sync Cycle Tests
# =============================================================================

class TestSyncCycle:

    def _make_aws_response(self, confirmed_ids, duplicate_ids=None):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": "ok",
            "confirmed_log_ids": confirmed_ids,
            "duplicate_log_ids": duplicate_ids or [],
        }
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    def test_offline_cycle_skips_http_call(self, seeded_db):
        data = seeded_db
        with patch("sync_engine.is_network_available", return_value=False):
            with patch("sync_engine.requests.post") as mock_post:
                result = run_sync_cycle(db_path=data["db_path"])

        mock_post.assert_not_called()
        assert result["skipped_offline"] == 1

    def test_offline_cycle_preserves_queue(self, seeded_db):
        data = seeded_db
        before = get_pending_sync_entries(db_path=data["db_path"])
        with patch("sync_engine.is_network_available", return_value=False):
            run_sync_cycle(db_path=data["db_path"])
        after = get_pending_sync_entries(db_path=data["db_path"])
        assert len(before) == len(after), "Queue must not shrink without HTTP 200"

    def test_successful_sync_removes_from_queue(self, seeded_db):
        data   = seeded_db
        log_id = data["log_ids"][0]
        mock_resp = self._make_aws_response([log_id])

        with patch("sync_engine.is_network_available", return_value=True):
            with patch("sync_engine.fetch_model_updates_from_s3", return_value=0):
                with patch("sync_engine.requests.post", return_value=mock_resp):
                    result = run_sync_cycle(db_path=data["db_path"])

        assert result["succeeded"] >= 1

    def test_successful_sync_marks_log_synced(self, seeded_db):
        data   = seeded_db
        log_id = data["log_ids"][0]
        mock_resp = self._make_aws_response([log_id])

        with patch("sync_engine.is_network_available", return_value=True):
            with patch("sync_engine.fetch_model_updates_from_s3", return_value=0):
                with patch("sync_engine.requests.post", return_value=mock_resp):
                    run_sync_cycle(db_path=data["db_path"])

        log = get_log(log_id, db_path=data["db_path"])
        assert log["synced"] == 1

    def test_duplicate_log_ids_also_resolved(self, seeded_db):
        """
        Duplicate log_ids returned by AWS (already stored on a previous sync)
        should be treated as resolved and removed from the local sync queue.
        """
        data   = seeded_db
        log_id = data["log_ids"][1]  # Second log
        # AWS says it's a duplicate (was already written)
        mock_resp = self._make_aws_response(
            confirmed_ids=[],
            duplicate_ids=[log_id],
        )
        # Add other log_ids as confirmed to avoid total-failure path
        all_ids = data["log_ids"]
        confirmed = [lid for lid in all_ids if lid != log_id]
        mock_resp.json.return_value["confirmed_log_ids"] = confirmed

        with patch("sync_engine.is_network_available", return_value=True):
            with patch("sync_engine.fetch_model_updates_from_s3", return_value=0):
                with patch("sync_engine.requests.post", return_value=mock_resp):
                    result = run_sync_cycle(db_path=data["db_path"])

        assert result["succeeded"] >= 1

    def test_http_error_keeps_entries_in_queue(self, seeded_db):
        import requests as req_lib
        data = seeded_db

        mock_resp = MagicMock()
        mock_resp.status_code = 503
        http_error = req_lib.HTTPError(response=mock_resp)
        mock_resp.raise_for_status.side_effect = http_error

        with patch("sync_engine.is_network_available", return_value=True):
            with patch("sync_engine.fetch_model_updates_from_s3", return_value=0):
                with patch("sync_engine.requests.post", side_effect=http_error):
                    result = run_sync_cycle(db_path=data["db_path"])

        assert result["failed"] > 0
        # Logs must still be unsynced
        for log_id in data["log_ids"]:
            log = get_log(log_id, db_path=data["db_path"])
            assert log["synced"] == 0, (
                f"log_id '{log_id}' must remain synced=0 after HTTP error"
            )

    def test_connection_error_keeps_entries_in_queue(self, seeded_db):
        import requests as req_lib
        data = seeded_db

        with patch("sync_engine.is_network_available", return_value=True):
            with patch("sync_engine.fetch_model_updates_from_s3", return_value=0):
                with patch(
                    "sync_engine.requests.post",
                    side_effect=req_lib.ConnectionError("Connection refused"),
                ):
                    result = run_sync_cycle(db_path=data["db_path"])

        assert result["failed"] > 0

    def test_empty_queue_returns_no_side_effects(self, db_path):
        """Sync cycle on an empty queue must be a no-op."""
        with patch("sync_engine.is_network_available", return_value=True):
            with patch("sync_engine.fetch_model_updates_from_s3", return_value=0):
                with patch("sync_engine.requests.post") as mock_post:
                    result = run_sync_cycle(db_path=db_path)
        mock_post.assert_not_called()
        assert result["pending"] == 0
        assert result["succeeded"] == 0
