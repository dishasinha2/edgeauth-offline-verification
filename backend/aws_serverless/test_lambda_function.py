"""
test_lambda_function.py
-----------------------
Unit tests for lambda_function.py — payload validation, idempotency logic,
DynamoDB write behaviour, and the full Lambda handler response contract.

All DynamoDB calls are mocked. No AWS credentials required.

Run with:
    pip install pytest boto3
    pytest test_lambda_function.py -v
"""

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, call

import pytest
from botocore.exceptions import ClientError

# Lambda modules are in the aws_serverless directory — adjust sys.path if needed
import sys, os
# Ensure the aws_serverless directory is on the path for direct import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lambda_function import (
    build_dynamodb_item,
    check_existing_log_ids,
    lambda_handler,
    validate_log_entry,
    validate_payload,
    write_entries_to_dynamodb,
)


# =============================================================================
# Fixtures
# =============================================================================

def make_log_entry(**overrides):
    """Return a valid log entry dict with optional field overrides."""
    base = {
        "log_id":             str(uuid.uuid4()),
        "organization_id":    str(uuid.uuid4()),
        "employee_id":        str(uuid.uuid4()),
        "event_type":         "CLOCK_IN",
        "verification_score": 0.9821,
        "liveness_score":     0.9700,
        "liveness_passed":    True,
        "device_id":          "EDGE-DEVICE-001",
        "latitude":           19.0760,
        "longitude":          72.8777,
        "timestamp":          "2024-01-15T08:00:00+00:00",
        "metadata":           {"fw": "v2.1.4"},
    }
    base.update(overrides)
    return base


def make_api_event(body, http_method="POST", path="/sync"):
    """Wrap a body dict in an API Gateway Lambda Proxy event structure."""
    return {
        "httpMethod":     http_method,
        "path":           path,
        "resource":       path,
        "body":           json.dumps(body),
        "headers":        {"Content-Type": "application/json"},
        "requestContext": {"requestId": str(uuid.uuid4())},
    }


def make_context():
    """Return a minimal Lambda context mock."""
    ctx = MagicMock()
    ctx.aws_request_id = str(uuid.uuid4())
    ctx.function_name  = "WorkforceVerificationSync-test"
    return ctx


def mock_table(existing_log_ids=None):
    """Return a MagicMock DynamoDB table that simulates GSI and put_item."""
    table = MagicMock()
    _existing = set(existing_log_ids or [])

    def query_side_effect(**kwargs):
        key_cond = kwargs.get("KeyConditionExpression")
        # Extract the actual value from the boto3 condition object's _values tuple
        # _values is (Key object, compared_value), so index [1] is the log_id string
        queried_id = None
        if key_cond is not None and hasattr(key_cond, "_values"):
            queried_id = key_cond._values[1]
        count = 1 if queried_id in _existing else 0
        return {"Count": count, "Items": [{"log_id": queried_id}] if count else []}

    table.query.side_effect = query_side_effect
    table.put_item.return_value = {"ResponseMetadata": {"HTTPStatusCode": 200}}
    return table


# =============================================================================
# Validation Tests
# =============================================================================

class TestValidateLogEntry:

    def test_valid_entry_passes(self):
        entry = make_log_entry()
        is_valid, reason = validate_log_entry(entry, 0)
        assert is_valid is True
        assert reason is None

    def test_missing_required_field_fails(self):
        for field in ["log_id", "organization_id", "employee_id",
                       "event_type", "verification_score", "liveness_score",
                       "liveness_passed", "device_id", "timestamp"]:
            entry = make_log_entry()
            del entry[field]
            is_valid, reason = validate_log_entry(entry, 0)
            assert is_valid is False
            assert field in reason

    def test_invalid_event_type_fails(self):
        entry = make_log_entry(event_type="INVALID")
        is_valid, reason = validate_log_entry(entry, 0)
        assert is_valid is False

    def test_score_out_of_range_fails(self):
        entry = make_log_entry(verification_score=1.5)
        is_valid, reason = validate_log_entry(entry, 0)
        assert is_valid is False
        assert "verification_score" in reason

    def test_invalid_timestamp_fails(self):
        entry = make_log_entry(timestamp="not-a-timestamp")
        is_valid, reason = validate_log_entry(entry, 0)
        assert is_valid is False
        assert "timestamp" in reason

    def test_invalid_uuid_fails(self):
        entry = make_log_entry(log_id="not-a-uuid")
        is_valid, reason = validate_log_entry(entry, 0)
        assert is_valid is False
        assert "log_id" in reason

    def test_all_valid_event_types_pass(self):
        for event_type in ("CLOCK_IN", "CLOCK_OUT", "ACCESS_GRANTED", "ACCESS_DENIED"):
            entry = make_log_entry(event_type=event_type)
            is_valid, _ = validate_log_entry(entry, 0)
            assert is_valid is True, f"event_type '{event_type}' should be valid"

    def test_non_dict_entry_fails(self):
        is_valid, reason = validate_log_entry("not a dict", 0)
        assert is_valid is False


class TestValidatePayload:

    def test_valid_payload_passes(self):
        payload = {"logs": [make_log_entry()]}
        is_valid, error, entries = validate_payload(payload)
        assert is_valid is True
        assert len(entries) == 1

    def test_missing_logs_field_fails(self):
        is_valid, error, entries = validate_payload({"data": []})
        assert is_valid is False
        assert "logs" in error

    def test_empty_logs_array_fails(self):
        is_valid, error, entries = validate_payload({"logs": []})
        assert is_valid is False

    def test_oversized_batch_fails(self):
        logs = [make_log_entry() for _ in range(501)]
        is_valid, error, _ = validate_payload({"logs": logs})
        assert is_valid is False
        assert "500" in error

    def test_non_list_logs_fails(self):
        is_valid, error, _ = validate_payload({"logs": "not a list"})
        assert is_valid is False

    def test_partial_valid_returns_only_valid_entries(self):
        good = make_log_entry()
        bad  = {"log_id": "bad"}  # Missing most fields
        payload = {"logs": [good, bad]}
        is_valid, error, entries = validate_payload(payload)
        assert is_valid is True
        assert len(entries) == 1
        assert entries[0]["log_id"] == good["log_id"]


# =============================================================================
# DynamoDB Item Builder Tests
# =============================================================================

class TestBuildDynamoDBItem:

    def test_required_keys_present(self):
        entry = make_log_entry()
        item  = build_dynamodb_item(entry, "2024-01-15T08:00:00+00:00")
        for key in ["organization_id", "timestamp", "log_id", "employee_id",
                    "event_type", "device_id", "ingestion_time", "ttl"]:
            assert key in item, f"Key '{key}' missing from DynamoDB item"

    def test_none_values_excluded(self):
        entry = make_log_entry(latitude=None, longitude=None)
        item  = build_dynamodb_item(entry, "2024-01-15T08:00:00+00:00")
        assert "latitude"  not in item
        assert "longitude" not in item

    def test_ttl_is_integer_in_future(self):
        import time
        entry = make_log_entry()
        item  = build_dynamodb_item(entry, "now")
        assert isinstance(item["ttl"], int)
        assert item["ttl"] > int(time.time())

    def test_metadata_serialized_as_json_string(self):
        metadata = {"camera": "test", "latency_ms": 87}
        entry = make_log_entry(metadata=metadata)
        item  = build_dynamodb_item(entry, "now")
        assert isinstance(item["metadata"], str)
        assert json.loads(item["metadata"]) == metadata

    def test_scores_stored_as_strings(self):
        entry = make_log_entry(verification_score=0.9821, liveness_score=0.9500)
        item  = build_dynamodb_item(entry, "now")
        # Stored as string to avoid float precision issues with DynamoDB Decimal
        assert isinstance(item["verification_score"], str)
        assert isinstance(item["liveness_score"], str)


# =============================================================================
# DynamoDB Write Tests
# =============================================================================

class TestWriteEntriesToDynamoDB:

    def test_new_entry_written_and_confirmed(self):
        table = mock_table(existing_log_ids=[])
        entry = make_log_entry()

        confirmed, duplicates, errors = write_entries_to_dynamodb(
            table=table,
            entries=[entry],
            existing_log_ids={entry["log_id"]: False},
            ingestion_time=datetime.now(timezone.utc).isoformat(),
        )

        assert entry["log_id"] in confirmed
        assert len(duplicates) == 0
        assert len(errors) == 0
        table.put_item.assert_called_once()

    def test_pre_identified_duplicate_skips_write(self):
        table = mock_table()
        entry = make_log_entry()

        confirmed, duplicates, errors = write_entries_to_dynamodb(
            table=table,
            entries=[entry],
            existing_log_ids={entry["log_id"]: True},  # Pre-check says it exists
            ingestion_time=datetime.now(timezone.utc).isoformat(),
        )

        table.put_item.assert_not_called()
        assert entry["log_id"] in duplicates
        assert len(confirmed) == 0

    def test_conditional_write_failure_classifies_as_duplicate(self):
        table = mock_table()

        error_response = {"Error": {"Code": "ConditionalCheckFailedException", "Message": "Exists"}}
        table.put_item.side_effect = ClientError(error_response, "PutItem")

        entry = make_log_entry()
        confirmed, duplicates, errors = write_entries_to_dynamodb(
            table=table,
            entries=[entry],
            existing_log_ids={entry["log_id"]: False},
            ingestion_time=datetime.now(timezone.utc).isoformat(),
        )

        assert entry["log_id"] in duplicates
        assert len(confirmed) == 0
        assert len(errors) == 0

    def test_dynamodb_throughput_error_classifies_as_error(self):
        table = mock_table()

        error_response = {"Error": {"Code": "ProvisionedThroughputExceededException", "Message": "Throttled"}}
        table.put_item.side_effect = ClientError(error_response, "PutItem")

        entry = make_log_entry()
        confirmed, duplicates, errors = write_entries_to_dynamodb(
            table=table,
            entries=[entry],
            existing_log_ids={entry["log_id"]: False},
            ingestion_time=datetime.now(timezone.utc).isoformat(),
        )

        assert entry["log_id"] in errors
        assert len(confirmed) == 0
        assert len(duplicates) == 0

    def test_batch_with_mixed_outcomes(self):
        entries = [make_log_entry() for _ in range(4)]

        table = MagicMock()
        # entry 0: success
        # entry 1: pre-identified duplicate
        # entry 2: conditional write failure (race condition)
        # entry 3: throughput error
        cond_error  = ClientError({"Error": {"Code": "ConditionalCheckFailedException", "Message": ""}}, "PutItem")
        throt_error = ClientError({"Error": {"Code": "ProvisionedThroughputExceededException", "Message": ""}}, "PutItem")

        table.put_item.side_effect = [None, cond_error, throt_error]
        table.query.return_value = {"Count": 0}

        existing_map = {
            entries[0]["log_id"]: False,
            entries[1]["log_id"]: True,   # Pre-identified duplicate
            entries[2]["log_id"]: False,
            entries[3]["log_id"]: False,
        }

        confirmed, duplicates, errors = write_entries_to_dynamodb(
            table=table,
            entries=entries,
            existing_log_ids=existing_map,
            ingestion_time=datetime.now(timezone.utc).isoformat(),
        )

        assert entries[0]["log_id"] in confirmed
        assert entries[1]["log_id"] in duplicates
        assert entries[2]["log_id"] in duplicates  # ConditionalCheckFailed
        assert entries[3]["log_id"] in errors


# =============================================================================
# Full Lambda Handler Tests
# =============================================================================

class TestLambdaHandler:

    def _invoke(self, body, table_mock=None, http_method="POST"):
        event   = make_api_event(body, http_method)
        context = make_context()

        if table_mock is None:
            table_mock = mock_table()

        with patch("lambda_function._get_dynamodb_table", return_value=table_mock):
            response = lambda_handler(event, context)

        return response, json.loads(response["body"])

    def test_happy_path_returns_200(self):
        entry     = make_log_entry()
        table     = mock_table()
        response, body = self._invoke({"logs": [entry]}, table_mock=table)
        assert response["statusCode"] == 200
        assert body["status"] == "ok"

    def test_confirmed_log_ids_in_response(self):
        entry = make_log_entry()
        table = mock_table()
        response, body = self._invoke({"logs": [entry]}, table_mock=table)
        assert entry["log_id"] in body["confirmed_log_ids"]

    def test_empty_body_returns_400(self):
        context  = make_context()
        event    = {"httpMethod": "POST", "path": "/sync", "resource": "/sync", "body": ""}
        response = lambda_handler(event, context)
        assert response["statusCode"] == 400

    def test_invalid_json_returns_400(self):
        context  = make_context()
        event    = {"httpMethod": "POST", "path": "/sync", "resource": "/sync", "body": "{not valid json"}
        response = lambda_handler(event, context)
        assert response["statusCode"] == 400

    def test_missing_logs_field_returns_422(self):
        response, body = self._invoke({"data": []})
        assert response["statusCode"] == 422

    def test_empty_logs_array_returns_422(self):
        response, body = self._invoke({"logs": []})
        assert response["statusCode"] == 422

    def test_batch_over_500_returns_422(self):
        logs = [make_log_entry() for _ in range(501)]
        response, body = self._invoke({"logs": logs})
        assert response["statusCode"] == 422

    def test_options_preflight_returns_200(self):
        context  = make_context()
        event    = {"httpMethod": "OPTIONS", "body": ""}
        response = lambda_handler(event, context)
        assert response["statusCode"] == 200

    def test_duplicate_log_ids_in_response(self):
        entry = make_log_entry()
        # Table pre-check: this log_id already exists
        table = mock_table(existing_log_ids=[entry["log_id"]])
        response, body = self._invoke({"logs": [entry]}, table_mock=table)
        assert response["statusCode"] == 200
        assert entry["log_id"] in body["duplicate_log_ids"]
        assert entry["log_id"] not in body["confirmed_log_ids"]

    def test_request_id_in_response(self):
        response, body = self._invoke({"logs": [make_log_entry()]})
        assert "request_id" in body
        # Must be a valid UUID
        uuid.UUID(body["request_id"])

    def test_dynamodb_connection_failure_returns_500(self):
        context = make_context()
        event   = make_api_event({"logs": [make_log_entry()]})
        with patch("lambda_function._get_dynamodb_table", side_effect=Exception("No credentials")):
            response = lambda_handler(event, context)
        assert response["statusCode"] == 500

    def test_response_has_cors_headers(self):
        response, _ = self._invoke({"logs": [make_log_entry()]})
        headers = response.get("headers", {})
        assert "Access-Control-Allow-Origin" in headers

    def test_content_type_header_present(self):
        response, _ = self._invoke({"logs": [make_log_entry()]})
        assert response["headers"]["Content-Type"] == "application/json"

    def test_body_is_already_a_dict_not_string(self):
        """API Gateway can pass body as a parsed dict in some test configurations."""
        entry    = make_log_entry()
        context  = make_context()
        table    = mock_table()
        # Pass body as dict (not JSON string)
        event = {
            "httpMethod": "POST",
            "path":       "/sync",
            "resource":   "/sync",
            "body":       {"logs": [entry]},  # Already parsed
            "headers":    {},
        }
        with patch("lambda_function._get_dynamodb_table", return_value=table):
            response = lambda_handler(event, context)
        assert response["statusCode"] == 200

    def test_multiple_organizations_in_single_batch(self):
        org1 = str(uuid.uuid4())
        org2 = str(uuid.uuid4())
        entries = [
            make_log_entry(organization_id=org1),
            make_log_entry(organization_id=org1),
            make_log_entry(organization_id=org2),
        ]
        table = mock_table()
        response, body = self._invoke({"logs": entries}, table_mock=table)
        assert response["statusCode"] == 200
        assert len(body["confirmed_log_ids"]) == 3
