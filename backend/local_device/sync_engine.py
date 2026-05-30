"""
sync_engine.py
--------------
Background Synchronization Engine for the Offline-First Workforce Verification Platform.

Responsibilities:
  - Polls the local sync_queue on a configurable interval
  - Checks network connectivity before each cycle (offline-first)
  - POSTs batched payloads to the AWS API Gateway endpoint
  - Implements exponential backoff with jitter on transient failures
  - Atomically HARD DELETES entries from sync_queue ONLY after HTTP 200 (Phase 5)
  - Resets stale IN_FLIGHT entries on startup (crash recovery)
  - Fetches updated face-embedding models (.onnx/.bin) from S3 when connectivity returns
  - Designed to run as a long-lived background thread or daemon process

Phase 5 Update:
  On confirmed HTTP 200, the engine now calls mark_sync_success() — which issues
  a TRUE STRUCTURAL DELETE from sync_queue — instead of any soft-flag mutation.

S3 Model Sync:
  When connectivity is (re-)established, fetch_model_updates_from_s3() downloads
  any updated .onnx or .bin model files from the wvp-models S3 bucket prefix
  to the local model directory. The edge device AI engine hot-reloads them.

Author: Workforce Verification Platform
"""

import json
import logging
import os
import random
import socket
import time
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from database import (
    delete_synced_queue_entries,
    get_pending_sync_entries,
    get_sync_queue_stats,
    mark_sync_failed,
    mark_sync_in_flight,
    mark_sync_success,
    DB_PATH,
)

# ---------------------------------------------------------------------------
# Configuration (override via environment variables in production)
# ---------------------------------------------------------------------------

AWS_API_ENDPOINT = os.environ.get(
    "WVP_AWS_API_ENDPOINT",
    "https://YOUR_API_ID.execute-api.YOUR_REGION.amazonaws.com/prod/sync",
)
API_KEY    = os.environ.get("WVP_API_KEY", "")          # AWS API Gateway API key
DEVICE_ID  = os.environ.get("WVP_DEVICE_ID", socket.gethostname())

SYNC_INTERVAL_SECONDS = int(os.environ.get("WVP_SYNC_INTERVAL", "30"))
BATCH_SIZE    = int(os.environ.get("WVP_BATCH_SIZE",    "50"))
MAX_ATTEMPTS  = int(os.environ.get("WVP_MAX_ATTEMPTS",  "10"))

# Exponential backoff parameters
BACKOFF_BASE_SECONDS  = float(os.environ.get("WVP_BACKOFF_BASE", "2.0"))
BACKOFF_MAX_SECONDS   = float(os.environ.get("WVP_BACKOFF_MAX",  "300.0"))
BACKOFF_JITTER_FACTOR = 0.25  # +/-25% jitter to avoid thundering herd

# Connectivity probe
CONNECTIVITY_HOST    = os.environ.get("WVP_CONNECTIVITY_HOST",    "8.8.8.8")
CONNECTIVITY_PORT    = int(os.environ.get("WVP_CONNECTIVITY_PORT", "53"))
CONNECTIVITY_TIMEOUT = float(os.environ.get("WVP_CONNECTIVITY_TIMEOUT", "3.0"))

HTTP_TIMEOUT = float(os.environ.get("WVP_HTTP_TIMEOUT", "30.0"))

# S3 model sync configuration
S3_MODELS_BUCKET      = os.environ.get("WVP_S3_MODELS_BUCKET", "wvp-models")
S3_MODELS_PREFIX      = os.environ.get("WVP_S3_MODELS_PREFIX", "face-embedding-models/")
LOCAL_MODELS_DIR      = os.environ.get("WVP_LOCAL_MODELS_DIR",  "./models")
MODEL_EXTENSIONS      = (".onnx", ".bin", ".pt", ".tflite")
MODEL_SYNC_ENABLED          = os.environ.get("WVP_MODEL_SYNC_ENABLED", "true").lower() == "true"
# Minimum seconds between S3 model-sync checks (default: 1 hour).
# Previously, model sync ran on every single sync cycle (every 30 s by default),
# causing an S3 LIST call every 30 s — wasteful and costly.  This constant
# enforces a cooldown so checks only happen at most once per hour.
MODEL_SYNC_INTERVAL_SECONDS = int(os.environ.get("WVP_MODEL_SYNC_INTERVAL", "3600"))

# Module-level tracker — updated after each model sync attempt.
# Initialized to negative infinity so the first connectivity check always
# triggers a model sync regardless of how long the process has been running.
_last_model_sync_time: float = float("-inf")

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sync_engine")

# ---------------------------------------------------------------------------
# Connectivity Check
# ---------------------------------------------------------------------------


def is_network_available(
    host: str = CONNECTIVITY_HOST,
    port: int = CONNECTIVITY_PORT,
    timeout: float = CONNECTIVITY_TIMEOUT,
) -> bool:
    """
    Attempt a lightweight TCP socket connection to probe internet availability.
    Uses Google's public DNS (8.8.8.8:53) as a stable, low-latency canary.
    Returns True if the connection succeeds, False otherwise.

    Why TCP socket over ICMP ping?
      - ICMP requires root privileges on many Linux distributions.
      - TCP is universally available and sufficient for our needs.
    """
    try:
        socket.setdefaulttimeout(timeout)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((host, port))
        return True
    except (socket.timeout, socket.error, OSError):
        return False


# ---------------------------------------------------------------------------
# Backoff Calculator
# ---------------------------------------------------------------------------


def compute_backoff(attempt: int) -> float:
    """
    Full-jitter exponential backoff.
    Formula: min(BASE * 2^attempt, MAX) * uniform(1 - JITTER, 1 + JITTER)
    """
    base_delay = min(BACKOFF_BASE_SECONDS * (2 ** attempt), BACKOFF_MAX_SECONDS)
    jitter     = random.uniform(1.0 - BACKOFF_JITTER_FACTOR, 1.0 + BACKOFF_JITTER_FACTOR)
    return base_delay * jitter


# ---------------------------------------------------------------------------
# S3 Model Update Fetch
# ---------------------------------------------------------------------------


def fetch_model_updates_from_s3(
    local_model_dir: str = LOCAL_MODELS_DIR,
    s3_bucket: str = S3_MODELS_BUCKET,
    s3_prefix: str = S3_MODELS_PREFIX,
) -> int:
    """
    Download updated face-embedding model files (.onnx, .bin, .pt, .tflite)
    from the S3 bucket to the local model directory when connectivity is restored.

    Strategy:
      - Lists objects under the configured S3 prefix.
      - For each model file: checks the local file's ETag (MD5) against the S3 ETag.
        If they differ (or the file is missing locally), downloads the new version.
      - Atomically replaces the local file so a running inference engine sees a
        consistent model file (write to .tmp, then os.replace).

    On edge devices where boto3 is NOT installed (Raspberry Pi minimal builds),
    this function gracefully logs a warning and returns 0 instead of crashing.

    Returns:
        Number of model files successfully downloaded/updated.
    """
    if not MODEL_SYNC_ENABLED:
        logger.debug("S3 model sync is disabled (WVP_MODEL_SYNC_ENABLED=false).")
        return 0

    try:
        import boto3                        # Optional — only available on devices with AWS SDK
        from botocore.exceptions import ClientError
    except ImportError:
        logger.warning(
            "boto3 not installed on this device. S3 model sync skipped. "
            "Install with: pip install boto3"
        )
        return 0

    os.makedirs(local_model_dir, exist_ok=True)
    updated_count = 0

    try:
        s3_client = boto3.client("s3")
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix)

        for page in pages:
            for obj in page.get("Contents", []):
                s3_key  = obj["Key"]
                s3_etag = obj["ETag"].strip('"')   # ETag is quoted by S3

                # Only process model files by extension
                if not any(s3_key.endswith(ext) for ext in MODEL_EXTENSIONS):
                    continue

                # Derive local path: strip the prefix, keep relative structure
                relative_name = s3_key[len(s3_prefix):]
                if not relative_name:
                    continue
                local_path = os.path.join(local_model_dir, relative_name)
                local_dir  = os.path.dirname(local_path)
                os.makedirs(local_dir, exist_ok=True)

                # Check if local file is already up to date (ETag comparison)
                needs_update = True
                if os.path.exists(local_path):
                    import hashlib
                    with open(local_path, "rb") as fh:
                        local_md5 = hashlib.md5(fh.read()).hexdigest()
                    if local_md5 == s3_etag:
                        logger.debug(
                            "Model '%s' is up to date (ETag match). Skipping.", s3_key
                        )
                        needs_update = False

                if needs_update:
                    tmp_path = local_path + ".tmp"
                    logger.info(
                        "Downloading updated model: s3://%s/%s -> %s",
                        s3_bucket, s3_key, local_path,
                    )
                    try:
                        s3_client.download_file(s3_bucket, s3_key, tmp_path)
                        os.replace(tmp_path, local_path)   # Atomic replace
                        updated_count += 1
                        logger.info("Model updated successfully: %s", local_path)
                    except (ClientError, OSError) as exc:
                        logger.error(
                            "Failed to download model '%s': %s", s3_key, exc
                        )
                        if os.path.exists(tmp_path):
                            try:
                                os.unlink(tmp_path)
                            except OSError:
                                pass

    except Exception as exc:  # pylint: disable=broad-except
        logger.error("S3 model sync failed: %s", exc)

    if updated_count > 0:
        logger.info(
            "S3 model sync complete: %d model file(s) updated in '%s'.",
            updated_count, local_model_dir,
        )
    else:
        logger.debug("S3 model sync: no updates found in s3://%s/%s", s3_bucket, s3_prefix)

    return updated_count


# ---------------------------------------------------------------------------
# Payload Builder
# ---------------------------------------------------------------------------


def build_payload(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Construct the JSON payload to POST to AWS API Gateway.
    The top-level structure carries device metadata + a list of log events.
    """
    return {
        "device_id":      DEVICE_ID,
        "sync_timestamp": datetime.now(timezone.utc).isoformat(),
        "entry_count":    len(entries),
        "logs": [
            {
                "log_id":             entry["log_id"],
                "organization_id":    entry["organization_id"],
                "employee_id":        entry["employee_id"],
                "event_type":         entry["event_type"],
                "verification_score": entry["verification_score"],
                "liveness_score":     entry["liveness_score"],
                "liveness_passed":    bool(entry["liveness_passed"]),
                "device_id":          entry["device_id"],
                "latitude":           entry.get("latitude"),
                "longitude":          entry.get("longitude"),
                "timestamp":          entry["event_timestamp"],
                "metadata":           entry.get("metadata") or {},
            }
            for entry in entries
        ],
    }


# ---------------------------------------------------------------------------
# HTTP POST with retry semantics
# ---------------------------------------------------------------------------


def post_batch_to_aws(
    payload: Dict[str, Any],
    attempt: int = 0,
) -> requests.Response:
    """
    POST a sync payload to AWS API Gateway.
    Does NOT implement retry logic here — that is handled by the sync loop
    which re-reads the queue and calculates backoff between full cycles.

    Raises:
        requests.RequestException — on any network-level failure.
        requests.HTTPError        — for 4xx/5xx responses.
    """
    headers = {
        "Content-Type":   "application/json",
        "X-Device-ID":    DEVICE_ID,
        "X-Sync-Attempt": str(attempt),
    }
    if API_KEY:
        headers["x-api-key"] = API_KEY

    response = requests.post(
        AWS_API_ENDPOINT,
        data=json.dumps(payload),
        headers=headers,
        timeout=HTTP_TIMEOUT,
    )
    response.raise_for_status()  # Raise HTTPError for 4xx / 5xx
    return response


# ---------------------------------------------------------------------------
# Core Sync Logic
# ---------------------------------------------------------------------------


def run_sync_cycle(db_path: str = DB_PATH) -> Dict[str, int]:
    """
    Execute a single synchronization cycle:
      1. Check network availability.
      2. Optionally fetch updated AI models from S3 (first network-available cycle).
      3. Fetch pending sync queue entries.
      4. Split them into batches by organization.
      5. For each batch, POST to AWS and handle the response.
      6. On HTTP 200: HARD DELETE the sync_queue row (Phase 5 purge).
      7. Return cycle statistics.

    Returns a dict with keys: pending, succeeded, failed, skipped_offline
    """
    stats = {"pending": 0, "succeeded": 0, "failed": 0, "skipped_offline": 0}

    if not is_network_available():
        logger.info("Network unavailable. Skipping sync cycle.")
        stats["skipped_offline"] = 1
        return stats

    # Opportunistically fetch model updates now that connectivity is confirmed,
    # but only if the cooldown interval has elapsed to avoid an S3 LIST call
    # on every 30-second sync cycle (BUG FIX: was previously unconstrained).
    global _last_model_sync_time
    now_ts = time.monotonic()
    if now_ts - _last_model_sync_time >= MODEL_SYNC_INTERVAL_SECONDS:
        try:
            fetch_model_updates_from_s3()
            _last_model_sync_time = now_ts
        except Exception as exc:  # pylint: disable=broad-except
            # Model fetch failure must never block the data sync
            logger.warning("Model fetch encountered an error (non-fatal): %s", exc)
    else:
        logger.debug(
            "Model sync skipped (cooldown active, next check in %.0fs).",
            MODEL_SYNC_INTERVAL_SECONDS - (now_ts - _last_model_sync_time),
        )

    pending_entries = get_pending_sync_entries(limit=BATCH_SIZE, db_path=db_path)
    stats["pending"] = len(pending_entries)

    if not pending_entries:
        logger.debug("Sync queue is empty. Nothing to sync.")
        return stats

    logger.info("Starting sync cycle. %d entries pending.", len(pending_entries))

    # Group entries into batches by organization for structured payloads
    org_batches: Dict[str, List[Dict[str, Any]]] = {}
    for entry in pending_entries:
        org_id = entry["organization_id"]
        org_batches.setdefault(org_id, []).append(entry)

    for org_id, batch in org_batches.items():
        logger.info("Syncing %d entries for organization '%s'.", len(batch), org_id)

        # Mark each entry in-flight to prevent duplicate sends from concurrent threads
        for entry in batch:
            mark_sync_in_flight(entry["queue_id"], db_path=db_path)

        payload = build_payload(batch)
        attempt = max(e["attempt_count"] for e in batch)

        try:
            response      = post_batch_to_aws(payload, attempt=attempt)
            response_data = response.json()

            # AWS Lambda returns per-log_id results for idempotency reporting
            confirmed_ids: List[str] = response_data.get("confirmed_log_ids", [])
            duplicate_ids: List[str] = response_data.get("duplicate_log_ids", [])

            # All confirmed + duplicates are considered successfully resolved
            resolved_ids = set(confirmed_ids) | set(duplicate_ids)

            for entry in batch:
                log_id = entry["log_id"]
                if log_id in resolved_ids:
                    # Phase 5: TRUE STRUCTURAL DELETE from sync_queue on HTTP 200
                    mark_sync_success(entry["queue_id"], log_id, db_path=db_path)
                    stats["succeeded"] += 1
                    logger.info(
                        "Synced log_id='%s' (duplicate=%s). Queue row HARD DELETED.",
                        log_id, log_id in duplicate_ids,
                    )
                else:
                    # Lambda acknowledged the batch but didn't confirm this specific log
                    mark_sync_failed(
                        entry["queue_id"],
                        f"log_id not in AWS confirmed set. Response: {response.status_code}",
                        db_path=db_path,
                    )
                    stats["failed"] += 1

        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else 0
            error_msg   = f"HTTP {status_code}: {str(exc)}"

            if status_code in (400, 422):
                logger.error(
                    "Permanent HTTP error for org '%s' batch: %s. Will still retry.",
                    org_id, error_msg,
                )
            else:
                logger.warning("Transient HTTP error for org '%s': %s.", org_id, error_msg)

            for entry in batch:
                if entry["attempt_count"] >= MAX_ATTEMPTS:
                    logger.error(
                        "Max retries exceeded for log_id='%s'. Keeping as FAILED_RETRYABLE "
                        "for operator inspection. Local log is preserved.",
                        entry["log_id"],
                    )
                mark_sync_failed(entry["queue_id"], error_msg, db_path=db_path)
                stats["failed"] += 1

        except requests.ConnectionError as exc:
            error_msg = f"Connection error: {str(exc)}"
            logger.warning("Connection error for org '%s': %s.", org_id, error_msg)
            for entry in batch:
                mark_sync_failed(entry["queue_id"], error_msg, db_path=db_path)
                stats["failed"] += 1

        except requests.Timeout as exc:
            error_msg = f"Request timed out after {HTTP_TIMEOUT}s: {str(exc)}"
            logger.warning("Timeout for org '%s': %s.", org_id, error_msg)
            for entry in batch:
                mark_sync_failed(entry["queue_id"], error_msg, db_path=db_path)
                stats["failed"] += 1

        except (ValueError, KeyError) as exc:
            # JSON decode failure or unexpected response shape
            error_msg = f"Response parsing error: {str(exc)}"
            logger.error("Unexpected AWS response for org '%s': %s.", org_id, error_msg)
            for entry in batch:
                mark_sync_failed(entry["queue_id"], error_msg, db_path=db_path)
                stats["failed"] += 1

    # Update pending count to reflect what remains after this cycle
    # (previously reported the pre-cycle count, which was misleading).
    stats["pending"] = stats["pending"] - stats["succeeded"]

    logger.info(
        "Sync cycle complete. Succeeded: %d, Failed: %d, Remaining pending: %d.",
        stats["succeeded"], stats["failed"], stats["pending"],
    )
    return stats


# ---------------------------------------------------------------------------
# Daemon Loop
# ---------------------------------------------------------------------------


def run_sync_daemon(
    db_path: str = DB_PATH,
    interval: int = SYNC_INTERVAL_SECONDS,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """
    Long-running sync daemon. Intended to run in a background thread.

    Startup tasks:
      - Resets any stale IN_FLIGHT entries from a previous crash.
      - Logs current queue stats.

    Loop behavior:
      - Runs a sync cycle every `interval` seconds.
      - If a cycle fails entirely (e.g., network down), waits with exponential
        backoff before the next attempt (but never exceeds BACKOFF_MAX_SECONDS).
      - Respects a threading.Event for clean shutdown.

    Args:
        db_path:    Path to the SQLite database file.
        interval:   Seconds between successful sync cycles.
        stop_event: If provided, loop exits when this event is set.
    """
    logger.info("=" * 60)
    logger.info(
        "Sync Daemon starting (device_id='%s', endpoint='%s').",
        DEVICE_ID, AWS_API_ENDPOINT,
    )
    logger.info(
        "Sync interval: %ds | Batch size: %d | Max retries: %d",
        interval, BATCH_SIZE, MAX_ATTEMPTS,
    )
    logger.info(
        "S3 model sync: enabled=%s | bucket=%s | prefix=%s",
        MODEL_SYNC_ENABLED, S3_MODELS_BUCKET, S3_MODELS_PREFIX,
    )
    logger.info("=" * 60)

    # Crash recovery: reset any entries that were left IN_FLIGHT
    stale = delete_synced_queue_entries(db_path=db_path)
    if stale:
        logger.info("Crash recovery: reset %d stale IN_FLIGHT entries to PENDING.", stale)

    # Log initial queue health
    stats = get_sync_queue_stats(db_path=db_path)
    logger.info("Initial queue state: %s", stats)

    consecutive_failures = 0

    while True:
        if stop_event and stop_event.is_set():
            logger.info("Stop event received. Shutting down sync daemon.")
            break

        cycle_start = time.monotonic()

        try:
            cycle_result = run_sync_cycle(db_path=db_path)

            if cycle_result.get("skipped_offline"):
                consecutive_failures += 1
                backoff = compute_backoff(consecutive_failures)
                logger.info(
                    "Offline. Will retry in %.1fs (attempt #%d).",
                    backoff, consecutive_failures,
                )
                _interruptible_sleep(backoff, stop_event)
                continue

            if cycle_result.get("failed", 0) > 0 and cycle_result.get("succeeded", 0) == 0:
                consecutive_failures += 1
                backoff = compute_backoff(consecutive_failures)
                logger.warning(
                    "All %d entries failed this cycle. Backing off %.1fs (attempt #%d).",
                    cycle_result["failed"], backoff, consecutive_failures,
                )
                _interruptible_sleep(backoff, stop_event)
                continue

            # Partial or full success — reset failure counter, sleep normal interval
            consecutive_failures = 0
            elapsed    = time.monotonic() - cycle_start
            sleep_time = max(0.0, interval - elapsed)
            logger.debug("Cycle complete in %.2fs. Sleeping %.1fs.", elapsed, sleep_time)
            _interruptible_sleep(sleep_time, stop_event)

        except Exception as exc:  # pylint: disable=broad-except
            consecutive_failures += 1
            backoff = compute_backoff(consecutive_failures)
            logger.exception(
                "Unexpected error in sync daemon (attempt #%d). Backing off %.1fs: %s",
                consecutive_failures, backoff, exc,
            )
            _interruptible_sleep(backoff, stop_event)


def _interruptible_sleep(seconds: float, stop_event: Optional[threading.Event]) -> None:
    """Sleep for `seconds` but wake up early if stop_event is set."""
    if stop_event:
        stop_event.wait(timeout=seconds)
    else:
        time.sleep(seconds)


# ---------------------------------------------------------------------------
# Entry Point — Run as a standalone script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from database import initialize_database

    initialize_database()

    stop = threading.Event()

    def _on_sigint(signum, frame):  # noqa: ANN001
        logger.info("Signal received. Signaling daemon to stop...")
        stop.set()

    try:
        import signal
        signal.signal(signal.SIGINT,  _on_sigint)
        signal.signal(signal.SIGTERM, _on_sigint)
    except (ImportError, OSError):
        pass  # Windows may not support all signals; graceful degradation

    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        logger.info("Running a single sync cycle (--once mode).")
        result = run_sync_cycle()
        logger.info("Result: %s", result)
        sys.exit(0 if result.get("failed", 0) == 0 else 1)

    if len(sys.argv) > 1 and sys.argv[1] == "--fetch-models":
        logger.info("Running model fetch from S3 only.")
        n = fetch_model_updates_from_s3()
        logger.info("Downloaded %d model file(s).", n)
        sys.exit(0)

    logger.info("Starting sync daemon in foreground. Press Ctrl+C to stop.")
    run_sync_daemon(stop_event=stop)
    logger.info("Sync daemon stopped cleanly.")
