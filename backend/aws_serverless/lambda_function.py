"""
lambda_function.py
------------------
AWS Lambda Handler for the Offline-First Workforce Verification Platform.

Responsibilities:
  - Routes incoming API Gateway requests to the correct handler based on
    HTTP method + resource path (mono-Lambda pattern).
  - POST   /sync                    — Batch-sync edge-device verification logs
  - POST   /organizations           — Register a new company workspace
  - GET    /organizations/{id}      — Retrieve workspace metadata + liveness thresholds
  - POST   /employees               — Register a new employee + upload face photo to S3
  - GET    /employees/{id}          — Fetch worker metadata
  - PUT    /employees/{id}/embedding — Update / re-calibrate facial feature weights

DynamoDB Tables:
  - WorkforceVerificationLogs — Immutable event log (PK: organization_id, SK: timestamp)
  - WorkforceOrganizations    — Organization registry (PK: organization_id)
  - WorkforceEmployees        — Employee biometric registry (PK: employee_id)

S3 Integration:
  When POST /employees is called with a base64-encoded face photo in the request body,
  the raw image is decoded and uploaded to the 'wvp-employee-media' S3 bucket under:
    employees/{organization_id}/{employee_id}/enrollment.jpg

Idempotency (Log Sync):
  A conditional PutItem (ConditionExpression: attribute_not_exists(log_id)) is used
  for the /sync endpoint to atomically detect and report duplicates.

Author: Workforce Verification Platform
"""

import base64
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import boto3
from boto3.dynamodb.conditions import Key as DynamoKey
from botocore.exceptions import ClientError

# ---------------------------------------------------------------------------
# Configuration (injected via Lambda environment variables)
# ---------------------------------------------------------------------------

DYNAMODB_TABLE_NAME          = os.environ.get("DYNAMODB_TABLE_NAME",       "WorkforceVerificationLogs")
ORGANIZATIONS_TABLE_NAME     = os.environ.get("ORGANIZATIONS_TABLE_NAME",  "WorkforceOrganizations")
EMPLOYEES_TABLE_NAME         = os.environ.get("EMPLOYEES_TABLE_NAME",      "WorkforceEmployees")
S3_MEDIA_BUCKET              = os.environ.get("S3_MEDIA_BUCKET",           "wvp-employee-media")
AWS_REGION                   = os.environ.get("AWS_REGION",                "ap-south-1")
LOG_LEVEL                    = os.environ.get("LOG_LEVEL",                 "INFO")

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

logger = logging.getLogger("lambda_function")
logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))


def _log(level: str, message: str, **kwargs: Any) -> None:
    """Emit a structured JSON log line parseable by CloudWatch Logs Insights."""
    record = {
        "level":     level,
        "message":   message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **kwargs,
    }
    getattr(logger, level.lower(), logger.info)(json.dumps(record))


# ---------------------------------------------------------------------------
# AWS Client Cache (reused across warm Lambda invocations)
# ---------------------------------------------------------------------------

_dynamodb_resource = None
_s3_client         = None


def _get_dynamodb_table(table_name: str = DYNAMODB_TABLE_NAME):
    """Lazily initialize the DynamoDB resource and return the named table."""
    global _dynamodb_resource
    if _dynamodb_resource is None:
        _dynamodb_resource = boto3.resource("dynamodb", region_name=AWS_REGION)
    return _dynamodb_resource.Table(table_name)


def _get_s3_client():
    """Lazily initialize and return the S3 client."""
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3", region_name=AWS_REGION)
    return _s3_client


# ---------------------------------------------------------------------------
# Common Response Builder
# ---------------------------------------------------------------------------


def _build_response(status_code: int, body: Dict[str, Any]) -> Dict[str, Any]:
    """Build a properly formatted API Gateway Lambda proxy response."""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type":                 "application/json",
            "X-Powered-By":                 "WorkforceVerificationPlatform",
            "Access-Control-Allow-Origin":  "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, x-api-key, X-Device-ID, X-Sync-Attempt",
        },
        "body": json.dumps(body),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_uuid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Request Body Parser
# ---------------------------------------------------------------------------


def _parse_body(event: Dict[str, Any]) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Parse the API Gateway event body (JSON string or already-parsed dict).
    Returns (parsed_body, error_response) — if error_response is not None,
    return it immediately from the handler.
    """
    raw_body = event.get("body", "")
    if not raw_body:
        return None, _build_response(400, {"status": "error", "error": "Request body is empty."})
    try:
        if isinstance(raw_body, str):
            return json.loads(raw_body), None
        return raw_body, None
    except json.JSONDecodeError as exc:
        return None, _build_response(400, {"status": "error", "error": f"Invalid JSON: {exc}"})


# ---------------------------------------------------------------------------
# ============================================================
# HANDLER: POST /sync  — Batch log sync from edge devices
# ============================================================
# ---------------------------------------------------------------------------

REQUIRED_LOG_FIELDS = {
    "log_id":             str,
    "organization_id":    str,
    "employee_id":        str,
    "event_type":         str,
    "verification_score": (int, float),
    "liveness_score":     (int, float),
    "liveness_passed":    bool,
    "device_id":          str,
    "timestamp":          str,
}
VALID_EVENT_TYPES = {"CLOCK_IN", "CLOCK_OUT", "ACCESS_GRANTED", "ACCESS_DENIED"}


def validate_log_entry(entry: Any, index: int) -> Tuple[bool, Optional[str]]:
    """Validate a single log entry from the incoming payload."""
    if not isinstance(entry, dict):
        return False, f"Entry[{index}] is not a JSON object."

    for field, expected_type in REQUIRED_LOG_FIELDS.items():
        if field not in entry:
            return False, f"Entry[{index}] missing required field '{field}'."
        if not isinstance(entry[field], expected_type):
            return False, (
                f"Entry[{index}] field '{field}' has wrong type "
                f"(expected {expected_type}, got {type(entry[field]).__name__})."
            )

    if entry["event_type"] not in VALID_EVENT_TYPES:
        return False, (
            f"Entry[{index}] has invalid event_type '{entry['event_type']}'. "
            f"Must be one of {VALID_EVENT_TYPES}."
        )

    for score_field in ("verification_score", "liveness_score"):
        val = entry[score_field]
        if not (0.0 <= val <= 1.0):
            return False, f"Entry[{index}] '{score_field}'={val} is outside [0.0, 1.0]."

    try:
        datetime.fromisoformat(entry["timestamp"].replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return False, f"Entry[{index}] 'timestamp' is not a valid ISO-8601 string."

    try:
        uuid.UUID(entry["log_id"])
    except ValueError:
        return False, f"Entry[{index}] 'log_id' is not a valid UUID: '{entry['log_id']}'."

    return True, None


def validate_payload(body: Dict[str, Any]) -> Tuple[bool, Optional[str], List[Dict]]:
    """Validate the top-level sync payload envelope."""
    if not isinstance(body, dict):
        return False, "Request body must be a JSON object.", []
    if "logs" not in body:
        return False, "Missing required top-level field 'logs'.", []
    if not isinstance(body["logs"], list):
        return False, "'logs' must be a JSON array.", []
    if len(body["logs"]) == 0:
        return False, "'logs' array is empty — nothing to process.", []
    if len(body["logs"]) > 500:
        return False, f"Batch too large: {len(body['logs'])} entries (max 500).", []

    valid_entries    = []
    validation_errors = []
    for i, entry in enumerate(body["logs"]):
        is_valid, reason = validate_log_entry(entry, i)
        if is_valid:
            valid_entries.append(entry)
        else:
            validation_errors.append(reason)
            _log("warning", "Validation failed for log entry", index=i, reason=reason)

    if not valid_entries:
        return False, f"All entries failed validation: {validation_errors}", []
    return True, None, valid_entries


def check_existing_log_ids(table, log_ids: List[str], organization_id: str) -> Dict[str, bool]:
    """Batch-check which log_ids already exist in DynamoDB to detect duplicates."""
    existing: Dict[str, bool] = {lid: False for lid in log_ids}
    if not log_ids:
        return existing

    CHUNK_SIZE = 25
    for chunk_start in range(0, len(log_ids), CHUNK_SIZE):
        chunk = log_ids[chunk_start: chunk_start + CHUNK_SIZE]
        for log_id in chunk:
            try:
                response = table.query(
                    IndexName="log_id-index",
                    KeyConditionExpression=DynamoKey("log_id").eq(log_id),
                    ProjectionExpression="log_id",
                    Limit=1,
                )
                if response.get("Count", 0) > 0:
                    existing[log_id] = True
            except ClientError as exc:
                error_code = exc.response["Error"]["Code"]
                _log("warning", "GSI query failed for log_id", log_id=log_id, error=error_code)
                existing[log_id] = False
    return existing


def build_dynamodb_item(entry: Dict[str, Any], ingestion_time: str) -> Dict[str, Any]:
    """Transform a validated log entry into a DynamoDB item."""
    item = {
        "organization_id":    entry["organization_id"],
        "timestamp":          entry["timestamp"],
        "log_id":             entry["log_id"],
        "employee_id":        entry["employee_id"],
        "event_type":         entry["event_type"],
        "verification_score": str(entry["verification_score"]),
        "liveness_score":     str(entry["liveness_score"]),
        "liveness_passed":    entry["liveness_passed"],
        "device_id":          entry["device_id"],
        "latitude":           str(entry["latitude"])  if entry.get("latitude")  is not None else None,
        "longitude":          str(entry["longitude"]) if entry.get("longitude") is not None else None,
        "metadata":           json.dumps(entry.get("metadata") or {}),
        "ingestion_time":     ingestion_time,
        "data_source":        "edge_device_sync",
        "ttl":                int(time.time()) + (365 * 24 * 3600 * 2),
    }
    return {k: v for k, v in item.items() if v is not None}


def write_entries_to_dynamodb(
    table,
    entries: List[Dict[str, Any]],
    existing_log_ids: Dict[str, bool],
    ingestion_time: str,
) -> Tuple[List[str], List[str], List[str]]:
    """Write new entries to DynamoDB using conditional PutItem (idempotency)."""
    confirmed_ids: List[str] = []
    duplicate_ids: List[str] = []
    error_ids:     List[str] = []

    for entry in entries:
        log_id = entry["log_id"]

        if existing_log_ids.get(log_id):
            _log("info", "Duplicate detected via pre-check", log_id=log_id)
            duplicate_ids.append(log_id)
            continue

        item = build_dynamodb_item(entry, ingestion_time)
        try:
            table.put_item(Item=item, ConditionExpression="attribute_not_exists(log_id)")
            confirmed_ids.append(log_id)
            _log("info", "Log written to DynamoDB", log_id=log_id,
                 organization_id=entry["organization_id"], event_type=entry["event_type"])

        except ClientError as exc:
            error_code = exc.response["Error"]["Code"]
            if error_code == "ConditionalCheckFailedException":
                _log("info", "Duplicate detected via conditional write", log_id=log_id)
                duplicate_ids.append(log_id)
            elif error_code in ("ProvisionedThroughputExceededException", "RequestLimitExceeded"):
                _log("error", "DynamoDB throughput exceeded", log_id=log_id, error=error_code)
                error_ids.append(log_id)
            elif error_code == "ValidationException":
                _log("error", "DynamoDB validation error", log_id=log_id,
                     error=str(exc.response["Error"]["Message"]))
                error_ids.append(log_id)
            else:
                _log("error", "Unexpected DynamoDB error", log_id=log_id, error=error_code)
                error_ids.append(log_id)
        except Exception as exc:  # pylint: disable=broad-except
            _log("error", "Unknown error writing to DynamoDB", log_id=log_id, error=str(exc))
            error_ids.append(log_id)

    return confirmed_ids, duplicate_ids, error_ids


def handle_sync(event: Dict[str, Any], context: Any, request_id: str) -> Dict[str, Any]:
    """POST /sync — Batch edge-device log sync handler."""
    ingestion_time = _utc_now()

    body, err = _parse_body(event)
    if err:
        return err

    is_valid, validation_error, valid_entries = validate_payload(body)
    if not is_valid:
        _log("warning", "Payload validation failed", request_id=request_id, reason=validation_error)
        return _build_response(422, {"status": "error", "error": validation_error,
                                     "request_id": request_id})

    _log("info", "Payload validated", request_id=request_id,
         total_entries=len(body["logs"]), valid_entries=len(valid_entries),
         device_id=body.get("device_id", "unknown"))

    try:
        table = _get_dynamodb_table(DYNAMODB_TABLE_NAME)
    except Exception as exc:
        _log("error", "DynamoDB table connection failed", request_id=request_id, error=str(exc))
        return _build_response(500, {"status": "error",
                                     "error": "Database connection failed. Please retry.",
                                     "request_id": request_id})

    all_log_ids   = [e["log_id"] for e in valid_entries]
    existing_map  = check_existing_log_ids(table, all_log_ids, organization_id="*")

    pre_check_duplicates = sum(1 for v in existing_map.values() if v)
    _log("info", "Idempotency pre-check complete", request_id=request_id,
         total=len(all_log_ids), pre_existing=pre_check_duplicates)

    confirmed_ids, duplicate_ids, error_ids = write_entries_to_dynamodb(
        table=table, entries=valid_entries,
        existing_log_ids=existing_map, ingestion_time=ingestion_time,
    )

    _log("info", "Sync batch processed", request_id=request_id,
         confirmed_count=len(confirmed_ids), duplicate_count=len(duplicate_ids),
         error_count=len(error_ids), device_id=body.get("device_id", "unknown"),
         ingestion_time=ingestion_time,
         _aws={
             "Namespace": "WorkforceVerificationPlatform",
             "Metrics": [{"Name": "ConfirmedLogs", "Unit": "Count"},
                         {"Name": "DuplicateLogs", "Unit": "Count"},
                         {"Name": "ErrorLogs",     "Unit": "Count"}],
             "Dimensions": [["device_id"]],
         },
         ConfirmedLogs=len(confirmed_ids),
         DuplicateLogs=len(duplicate_ids),
         ErrorLogs=len(error_ids))

    if error_ids and not confirmed_ids and not duplicate_ids:
        return _build_response(500, {
            "status": "error",
            "error": f"All {len(error_ids)} entries failed to write. Retry is safe.",
            "error_log_ids": error_ids,
            "request_id": request_id,
        })

    message = (
        f"Processed {len(valid_entries)} events: "
        f"{len(confirmed_ids)} new, "
        f"{len(duplicate_ids)} duplicate"
        + (f", {len(error_ids)} errors" if error_ids else "")
        + "."
    )
    _log("info", message, request_id=request_id)
    return _build_response(200, {
        "status":            "ok",
        "confirmed_log_ids": confirmed_ids,
        "duplicate_log_ids": duplicate_ids,
        "error_log_ids":     error_ids,
        "message":           message,
        "request_id":        request_id,
        "ingestion_time":    ingestion_time,
    })


# ---------------------------------------------------------------------------
# ============================================================
# HANDLER: POST /organizations — Register a company workspace
# ============================================================
# ---------------------------------------------------------------------------


def handle_create_organization(
    event: Dict[str, Any], context: Any, request_id: str
) -> Dict[str, Any]:
    """
    POST /organizations
    Create a new organization (company workspace) in DynamoDB.

    Required body fields:
      - name            (str)  : Human-readable organization name
      - region          (str)  : Deployment region e.g. "APAC", "US-EAST"
    Optional body fields:
      - contact_email   (str)  : Primary contact
      - liveness_threshold (float, 0.0-1.0): Per-org adaptive liveness floor (default 0.80)
      - organization_id (str)  : Pre-assigned UUID (idempotent registration)
    """
    body, err = _parse_body(event)
    if err:
        return err

    name   = body.get("name", "").strip()
    region = body.get("region", "").strip()
    if not name:
        return _build_response(422, {"status": "error", "error": "'name' is required.",
                                     "request_id": request_id})
    if not region:
        return _build_response(422, {"status": "error", "error": "'region' is required.",
                                     "request_id": request_id})

    liveness_threshold = body.get("liveness_threshold", 0.80)
    if not isinstance(liveness_threshold, (int, float)) or not (0.0 <= liveness_threshold <= 1.0):
        return _build_response(422, {
            "status": "error",
            "error":  "'liveness_threshold' must be a float in [0.0, 1.0].",
            "request_id": request_id,
        })

    contact_email   = body.get("contact_email")
    organization_id = body.get("organization_id") or _new_uuid()
    now             = _utc_now()

    item = {
        "organization_id":    organization_id,
        "name":               name,
        "region":             region,
        "contact_email":      contact_email,
        "is_active":          True,
        "liveness_threshold": str(liveness_threshold),
        "created_at":         now,
        "updated_at":         now,
    }
    item = {k: v for k, v in item.items() if v is not None}

    try:
        table = _get_dynamodb_table(ORGANIZATIONS_TABLE_NAME)
        table.put_item(
            Item=item,
            ConditionExpression="attribute_not_exists(organization_id)",
        )
        _log("info", "Organization created", organization_id=organization_id,
             name=name, region=region, request_id=request_id)
        return _build_response(201, {
            "status":          "created",
            "organization_id": organization_id,
            "name":            name,
            "region":          region,
            "liveness_threshold": liveness_threshold,
            "created_at":      now,
            "request_id":      request_id,
        })

    except ClientError as exc:
        error_code = exc.response["Error"]["Code"]
        if error_code == "ConditionalCheckFailedException":
            return _build_response(409, {
                "status":  "conflict",
                "error":   f"Organization '{organization_id}' already exists.",
                "request_id": request_id,
            })
        _log("error", "DynamoDB error creating organization", error=error_code,
             request_id=request_id)
        return _build_response(500, {"status": "error", "error": f"DynamoDB error: {error_code}",
                                     "request_id": request_id})


# ---------------------------------------------------------------------------
# ============================================================
# HANDLER: GET /organizations/{id}
# ============================================================
# ---------------------------------------------------------------------------


def handle_get_organization(
    event: Dict[str, Any], context: Any, request_id: str
) -> Dict[str, Any]:
    """
    GET /organizations/{id}
    Retrieve workspace metadata including adaptive liveness threshold.
    """
    path_params     = event.get("pathParameters") or {}
    organization_id = path_params.get("id", "").strip()

    if not organization_id:
        return _build_response(400, {"status": "error", "error": "Organization ID is required.",
                                     "request_id": request_id})

    try:
        table    = _get_dynamodb_table(ORGANIZATIONS_TABLE_NAME)
        response = table.get_item(Key={"organization_id": organization_id})
        item     = response.get("Item")
        if not item:
            return _build_response(404, {
                "status":  "not_found",
                "error":   f"Organization '{organization_id}' does not exist.",
                "request_id": request_id,
            })

        # Convert Decimal liveness_threshold back to float for JSON response
        if "liveness_threshold" in item:
            item["liveness_threshold"] = float(item["liveness_threshold"])

        _log("info", "Organization fetched", organization_id=organization_id,
             request_id=request_id)
        return _build_response(200, {"status": "ok", "organization": item,
                                     "request_id": request_id})

    except ClientError as exc:
        error_code = exc.response["Error"]["Code"]
        _log("error", "DynamoDB error fetching organization", error=error_code,
             request_id=request_id)
        return _build_response(500, {"status": "error", "error": f"DynamoDB error: {error_code}",
                                     "request_id": request_id})


# ---------------------------------------------------------------------------
# ============================================================
# HANDLER: POST /employees — Register employee + S3 photo upload
# ============================================================
# ---------------------------------------------------------------------------


def _upload_face_photo_to_s3(
    organization_id: str,
    employee_id: str,
    photo_b64: str,
    request_id: str,
) -> Optional[str]:
    """
    Decode a base64-encoded face photo and upload it to S3.

    S3 key format: employees/{organization_id}/{employee_id}/enrollment.jpg

    Returns the S3 object key on success, or None if the upload fails (non-fatal;
    the employee registration still succeeds without the photo).
    """
    try:
        image_bytes = base64.b64decode(photo_b64)
    except Exception as exc:
        _log("warning", "Failed to decode base64 face photo — skipping S3 upload",
             request_id=request_id, error=str(exc))
        return None

    s3_key = f"employees/{organization_id}/{employee_id}/enrollment.jpg"
    try:
        s3 = _get_s3_client()
        s3.put_object(
            Bucket=S3_MEDIA_BUCKET,
            Key=s3_key,
            Body=image_bytes,
            ContentType="image/jpeg",
            ServerSideEncryption="AES256",
            Metadata={
                "organization_id": organization_id,
                "employee_id":     employee_id,
                "upload_time":     _utc_now(),
                "source":          "enrollment_api",
            },
        )
        _log("info", "Face photo uploaded to S3",
             s3_bucket=S3_MEDIA_BUCKET, s3_key=s3_key,
             size_bytes=len(image_bytes), request_id=request_id)
        return s3_key
    except ClientError as exc:
        error_code = exc.response["Error"]["Code"]
        _log("error", "S3 upload failed — employee registration will continue without photo",
             s3_key=s3_key, error=error_code, request_id=request_id)
        return None


def handle_create_employee(
    event: Dict[str, Any], context: Any, request_id: str
) -> Dict[str, Any]:
    """
    POST /employees
    Register a new employee with biometric metadata.
    Optionally uploads a base64-encoded face photo to S3.

    Required body fields:
      - organization_id (str)        : Parent organization UUID
      - full_name       (str)        : Employee full name
      - face_embedding  (list[float]): Pre-computed embedding vector (128d or 512d)
    Optional body fields:
      - department      (str)
      - role            (str)
      - embedding_model (str)        : e.g., "arcface_iresnet100_512d" (default: "facenet_512d")
      - face_photo_b64  (str)        : Base64-encoded JPEG of enrollment photo
      - employee_id     (str)        : Pre-assigned UUID (idempotent registration)
    """
    body, err = _parse_body(event)
    if err:
        return err

    # Validate required fields
    organization_id = body.get("organization_id", "").strip()
    full_name       = body.get("full_name", "").strip()
    face_embedding  = body.get("face_embedding")

    if not organization_id:
        return _build_response(422, {"status": "error",
                                     "error": "'organization_id' is required.",
                                     "request_id": request_id})
    if not full_name:
        return _build_response(422, {"status": "error",
                                     "error": "'full_name' is required.",
                                     "request_id": request_id})
    if not face_embedding or not isinstance(face_embedding, list):
        return _build_response(422, {"status": "error",
                                     "error": "'face_embedding' must be a non-empty JSON array.",
                                     "request_id": request_id})
    if not all(isinstance(v, (int, float)) for v in face_embedding):
        return _build_response(422, {"status": "error",
                                     "error": "'face_embedding' must contain only numbers.",
                                     "request_id": request_id})
    if len(face_embedding) not in (128, 256, 512):
        return _build_response(422, {"status": "error",
                                     "error": (f"'face_embedding' has {len(face_embedding)} "
                                               "dimensions; expected 128, 256, or 512."),
                                     "request_id": request_id})

    employee_id     = body.get("employee_id") or _new_uuid()
    department      = body.get("department")
    role            = body.get("role")
    embedding_model = body.get("embedding_model", "facenet_512d")
    face_photo_b64  = body.get("face_photo_b64")
    now             = _utc_now()

    # Upload face photo to S3 if provided
    face_photo_s3_key = None
    if face_photo_b64:
        face_photo_s3_key = _upload_face_photo_to_s3(
            organization_id, employee_id, face_photo_b64, request_id
        )

    item = {
        "employee_id":       employee_id,
        "organization_id":   organization_id,
        "full_name":         full_name,
        "department":        department,
        "role":              role,
        "face_embedding":    json.dumps(face_embedding),   # Serialized for DynamoDB storage
        "embedding_model":   embedding_model,
        "face_photo_s3_key": face_photo_s3_key,
        "face_photo_url":    (
            f"s3://{S3_MEDIA_BUCKET}/{face_photo_s3_key}" if face_photo_s3_key else None
        ),
        "is_active":         True,
        "created_at":        now,
        "updated_at":        now,
    }
    item = {k: v for k, v in item.items() if v is not None}

    try:
        table = _get_dynamodb_table(EMPLOYEES_TABLE_NAME)
        table.put_item(
            Item=item,
            ConditionExpression="attribute_not_exists(employee_id)",
        )
        _log("info", "Employee registered",
             employee_id=employee_id, organization_id=organization_id,
             full_name=full_name, face_photo_s3_key=face_photo_s3_key or "none",
             request_id=request_id)
        return _build_response(201, {
            "status":            "created",
            "employee_id":       employee_id,
            "organization_id":   organization_id,
            "full_name":         full_name,
            "embedding_model":   embedding_model,
            "embedding_dims":    len(face_embedding),
            "face_photo_s3_key": face_photo_s3_key,
            "face_photo_url":    item.get("face_photo_url"),
            "created_at":        now,
            "request_id":        request_id,
        })

    except ClientError as exc:
        error_code = exc.response["Error"]["Code"]
        if error_code == "ConditionalCheckFailedException":
            return _build_response(409, {
                "status":  "conflict",
                "error":   f"Employee '{employee_id}' already exists.",
                "request_id": request_id,
            })
        _log("error", "DynamoDB error creating employee", error=error_code,
             request_id=request_id)
        return _build_response(500, {"status": "error", "error": f"DynamoDB error: {error_code}",
                                     "request_id": request_id})


# ---------------------------------------------------------------------------
# ============================================================
# HANDLER: GET /employees/{id}
# ============================================================
# ---------------------------------------------------------------------------


def handle_get_employee(
    event: Dict[str, Any], context: Any, request_id: str
) -> Dict[str, Any]:
    """
    GET /employees/{id}
    Fetch worker metadata. The face_embedding vector is deserialized from
    its stored JSON string back to a list for the response.
    """
    path_params = event.get("pathParameters") or {}
    employee_id = path_params.get("id", "").strip()

    if not employee_id:
        return _build_response(400, {"status": "error", "error": "Employee ID is required.",
                                     "request_id": request_id})

    try:
        table    = _get_dynamodb_table(EMPLOYEES_TABLE_NAME)
        response = table.get_item(Key={"employee_id": employee_id})
        item     = response.get("Item")
        if not item:
            return _build_response(404, {
                "status":  "not_found",
                "error":   f"Employee '{employee_id}' does not exist.",
                "request_id": request_id,
            })

        # Deserialize the embedding back to a list for callers
        if "face_embedding" in item:
            try:
                item["face_embedding"] = json.loads(item["face_embedding"])
            except (json.JSONDecodeError, TypeError):
                item["face_embedding"] = []

        _log("info", "Employee fetched", employee_id=employee_id, request_id=request_id)
        return _build_response(200, {"status": "ok", "employee": item,
                                     "request_id": request_id})

    except ClientError as exc:
        error_code = exc.response["Error"]["Code"]
        _log("error", "DynamoDB error fetching employee", error=error_code,
             request_id=request_id)
        return _build_response(500, {"status": "error", "error": f"DynamoDB error: {error_code}",
                                     "request_id": request_id})


# ---------------------------------------------------------------------------
# ============================================================
# HANDLER: PUT /employees/{id}/embedding — Update facial weights
# ============================================================
# ---------------------------------------------------------------------------


def handle_update_employee_embedding(
    event: Dict[str, Any], context: Any, request_id: str
) -> Dict[str, Any]:
    """
    PUT /employees/{id}/embedding
    Update or re-calibrate an employee's facial feature weights (face embedding).
    Also optionally uploads a new enrollment photo to S3.

    Required body fields:
      - face_embedding  (list[float]): New embedding vector (128d or 512d)
    Optional body fields:
      - embedding_model (str)        : Model that produced the new embedding
      - face_photo_b64  (str)        : New base64-encoded enrollment photo
    """
    path_params = event.get("pathParameters") or {}
    employee_id = path_params.get("id", "").strip()

    if not employee_id:
        return _build_response(400, {"status": "error", "error": "Employee ID is required.",
                                     "request_id": request_id})

    body, err = _parse_body(event)
    if err:
        return err

    face_embedding = body.get("face_embedding")
    if not face_embedding or not isinstance(face_embedding, list):
        return _build_response(422, {"status": "error",
                                     "error": "'face_embedding' must be a non-empty JSON array.",
                                     "request_id": request_id})
    if not all(isinstance(v, (int, float)) for v in face_embedding):
        return _build_response(422, {"status": "error",
                                     "error": "'face_embedding' must contain only numbers.",
                                     "request_id": request_id})
    if len(face_embedding) not in (128, 256, 512):
        return _build_response(422, {
            "status": "error",
            "error":  (f"'face_embedding' has {len(face_embedding)} dimensions; "
                       "expected 128, 256, or 512."),
            "request_id": request_id,
        })

    embedding_model = body.get("embedding_model", "facenet_512d")
    face_photo_b64  = body.get("face_photo_b64")
    now             = _utc_now()

    # Verify the employee exists before updating
    try:
        emp_table = _get_dynamodb_table(EMPLOYEES_TABLE_NAME)
        check     = emp_table.get_item(Key={"employee_id": employee_id})
        if not check.get("Item"):
            return _build_response(404, {
                "status":  "not_found",
                "error":   f"Employee '{employee_id}' does not exist.",
                "request_id": request_id,
            })
        existing_org_id = check["Item"].get("organization_id", "unknown")
    except ClientError as exc:
        error_code = exc.response["Error"]["Code"]
        _log("error", "DynamoDB get failed during embedding update",
             employee_id=employee_id, error=error_code, request_id=request_id)
        return _build_response(500, {"status": "error", "error": f"DynamoDB error: {error_code}",
                                     "request_id": request_id})

    # Upload new enrollment photo to S3 if provided
    new_s3_key = None
    if face_photo_b64:
        new_s3_key = _upload_face_photo_to_s3(
            existing_org_id, employee_id, face_photo_b64, request_id
        )

    # Build update expression
    update_expr      = "SET face_embedding = :emb, embedding_model = :model, updated_at = :ts"
    expr_attr_values = {
        ":emb":   json.dumps(face_embedding),
        ":model": embedding_model,
        ":ts":    now,
    }
    if new_s3_key:
        update_expr += ", face_photo_s3_key = :s3k, face_photo_url = :url"
        expr_attr_values[":s3k"] = new_s3_key
        expr_attr_values[":url"] = f"s3://{S3_MEDIA_BUCKET}/{new_s3_key}"

    try:
        emp_table.update_item(
            Key={"employee_id": employee_id},
            UpdateExpression=update_expr,
            ExpressionAttributeValues=expr_attr_values,
            ConditionExpression="attribute_exists(employee_id)",
        )
        _log("info", "Employee embedding updated",
             employee_id=employee_id, embedding_model=embedding_model,
             embedding_dims=len(face_embedding), new_photo_uploaded=new_s3_key is not None,
             request_id=request_id)
        return _build_response(200, {
            "status":          "updated",
            "employee_id":     employee_id,
            "embedding_model": embedding_model,
            "embedding_dims":  len(face_embedding),
            "face_photo_s3_key": new_s3_key,
            "updated_at":      now,
            "request_id":      request_id,
        })

    except ClientError as exc:
        error_code = exc.response["Error"]["Code"]
        if error_code == "ConditionalCheckFailedException":
            return _build_response(404, {
                "status":  "not_found",
                "error":   f"Employee '{employee_id}' not found during update.",
                "request_id": request_id,
            })
        _log("error", "DynamoDB error updating embedding", error=error_code,
             request_id=request_id)
        return _build_response(500, {"status": "error", "error": f"DynamoDB error: {error_code}",
                                     "request_id": request_id})


# ---------------------------------------------------------------------------
# ============================================================
# MAIN ROUTER — Lambda Entry Point
# ============================================================
# ---------------------------------------------------------------------------


def handle_verify_liveness(
    event: Dict[str, Any], context: Any, request_id: str
) -> Dict[str, Any]:
    """
    POST /verify-liveness
    Decode a base64 camera frame and run the requested liveness challenge.

    Required body fields:
      - image     (str): Base64-encoded image
      - challenge (str): "Blink Twice", "Turn Head Left", or "Turn Head Right"
    """
    body, err = _parse_body(event)
    if err:
        return err

    try:
        from backend.local_device.ai_service import verify_liveness_request
    except Exception as exc:  # pylint: disable=broad-except
        _log("error", "AI service import failed", request_id=request_id, error=str(exc))
        return _build_response(500, {
            "success": False,
            "challenge": body.get("challenge") if isinstance(body, dict) else None,
            "status": "AI service unavailable",
            "request_id": request_id,
        })

    result = verify_liveness_request(body)
    result["request_id"] = request_id
    return _build_response(200, result)


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Main AWS Lambda entry point — routes all API Gateway requests.

    Route Table:
      OPTIONS  *                        -> CORS preflight (200)
      POST     /sync                    -> handle_sync
      POST     /organizations           -> handle_create_organization
      GET      /organizations/{id}      -> handle_get_organization
      POST     /employees               -> handle_create_employee
      GET      /employees/{id}          -> handle_get_employee
      PUT      /employees/{id}/embedding -> handle_update_employee_embedding
      POST     /verify-liveness         -> handle_verify_liveness
      *        other                    -> 404
    """
    request_id = (
        context.aws_request_id if context and hasattr(context, "aws_request_id")
        else _new_uuid()
    )

    http_method   = event.get("httpMethod", "")
    resource_path = event.get("resource",  event.get("path", ""))

    _log("info", "Lambda invoked",
         request_id=request_id,
         function_name=getattr(context, "function_name", "local"),
         http_method=http_method,
         path=resource_path)

    # ------------------------------------------------------------------
    # CORS Preflight
    # ------------------------------------------------------------------
    if http_method == "OPTIONS":
        return _build_response(200, {"status": "ok"})

    # ------------------------------------------------------------------
    # Route dispatch
    # ------------------------------------------------------------------

    # POST /sync
    if http_method == "POST" and resource_path in ("/sync", "/prod/sync"):
        return handle_sync(event, context, request_id)

    # POST /organizations
    if http_method == "POST" and resource_path in ("/organizations", "/prod/organizations"):
        return handle_create_organization(event, context, request_id)

    # GET /organizations/{id}
    if http_method == "GET" and (
        resource_path.startswith("/organizations/") or
        resource_path.startswith("/prod/organizations/") or
        resource_path in ("/organizations/{id}", "/prod/organizations/{id}")
    ):
        return handle_get_organization(event, context, request_id)

    # POST /employees
    if http_method == "POST" and resource_path in ("/employees", "/prod/employees"):
        return handle_create_employee(event, context, request_id)

    # POST /verify-liveness
    if http_method == "POST" and resource_path in ("/verify-liveness", "/prod/verify-liveness"):
        return handle_verify_liveness(event, context, request_id)

    # GET /employees/{id}
    if http_method == "GET" and (
        resource_path.startswith("/employees/") or
        resource_path.startswith("/prod/employees/") or
        resource_path in ("/employees/{id}", "/prod/employees/{id}")
    ) and not resource_path.endswith("/embedding"):
        return handle_get_employee(event, context, request_id)

    # PUT /employees/{id}/embedding
    if http_method == "PUT" and (
        "/embedding" in resource_path or
        resource_path in ("/employees/{id}/embedding", "/prod/employees/{id}/embedding")
    ):
        return handle_update_employee_embedding(event, context, request_id)

    # ------------------------------------------------------------------
    # 404 — No matching route
    # ------------------------------------------------------------------
    _log("warning", "No route matched", http_method=http_method,
         path=resource_path, request_id=request_id)
    return _build_response(404, {
        "status":  "error",
        "error":   f"No route found for {http_method} {resource_path}",
        "request_id": request_id,
    })
