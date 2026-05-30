"""
conftest.py
-----------
pytest shared fixtures for the Workforce Verification Platform test suite.
Provides isolated, temporary SQLite databases for every test function,
guaranteeing complete test isolation with zero shared state.

Run all tests with:
    pytest -v

Run with coverage:
    pytest -v --cov=. --cov-report=term-missing
"""

import math
import os
import random
import tempfile
import pytest

from database import initialize_database


@pytest.fixture(scope="function")
def db_path(tmp_path):
    """
    Pytest fixture that creates a fresh, isolated SQLite database file
    for each individual test function. The database is fully initialized
    with the production schema and automatically deleted after the test.

    Usage in tests:
        def test_something(db_path):
            insert_organization(..., db_path=db_path)
    """
    db_file = tmp_path / "test_workforce_verification.db"
    initialize_database(db_path=str(db_file))
    yield str(db_file)
    # tmp_path is cleaned up automatically by pytest


@pytest.fixture(scope="session")
def sample_embedding_512d():
    """
    Returns a deterministic, L2-normalized 512-dimensional face embedding.
    Used across multiple tests to avoid repeated computation.
    """
    rng = random.Random(42)
    raw = [rng.gauss(0, 1) for _ in range(512)]
    magnitude = math.sqrt(sum(v ** 2 for v in raw))
    return [round(v / magnitude, 8) for v in raw]


@pytest.fixture(scope="function")
def seeded_db(db_path, sample_embedding_512d):
    """
    Pytest fixture that returns a fully seeded database with:
      - 1 organization
      - 2 employees with mock embeddings
      - 3 verification log entries already inserted (all pending sync)

    Returns a dict with org_id, employee_ids, log_ids, and db_path
    for convenient use in test assertions.
    """
    from database import insert_organization, insert_employee, insert_log

    org_id = insert_organization(
        name="Test Organization",
        region="TEST",
        contact_email="test@example.com",
        db_path=db_path,
    )

    emp1_id = insert_employee(
        organization_id=org_id,
        full_name="Alice Test",
        face_embedding=sample_embedding_512d,
        department="Engineering",
        role="Engineer",
        db_path=db_path,
    )

    # Second employee with a different embedding (different seed)
    rng = random.Random(99)
    raw = [rng.gauss(0, 1) for _ in range(512)]
    magnitude = math.sqrt(sum(v ** 2 for v in raw))
    emp2_embedding = [round(v / magnitude, 8) for v in raw]

    emp2_id = insert_employee(
        organization_id=org_id,
        full_name="Bob Test",
        face_embedding=emp2_embedding,
        department="Operations",
        role="Operator",
        db_path=db_path,
    )

    log_ids = []
    events = [
        (emp1_id, "CLOCK_IN",     0.9821, 0.9700, True),
        (emp1_id, "CLOCK_OUT",    0.9654, 0.9500, True),
        (emp2_id, "ACCESS_DENIED",0.4200, 0.8800, False),
    ]
    for emp_id, event_type, v_score, l_score, l_passed in events:
        log_id = insert_log(
            organization_id=org_id,
            employee_id=emp_id,
            event_type=event_type,
            verification_score=v_score,
            liveness_score=l_score,
            liveness_passed=l_passed,
            device_id="TEST-DEVICE-001",
            db_path=db_path,
        )
        log_ids.append(log_id)

    return {
        "db_path":      db_path,
        "org_id":       org_id,
        "employee_ids": [emp1_id, emp2_id],
        "log_ids":      log_ids,
    }


# ---------------------------------------------------------------------------
# Fixtures for mock_test.py integration tests
# These provide the sequential data pipeline that mock_test.py's step
# functions depend on. Each fixture builds on the previous, mirroring
# the main() function's call chain.
# ---------------------------------------------------------------------------

def _generate_mock_embedding(dimensions: int = 512, seed: int = None):
    """Generate a random L2-normalized face embedding vector."""
    rng = random.Random(seed)
    raw = [rng.gauss(0, 1) for _ in range(dimensions)]
    magnitude = math.sqrt(sum(v ** 2 for v in raw))
    return [round(v / magnitude, 8) for v in raw]


_SEED_ORGANIZATIONS = [
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

_SEED_EMPLOYEES = [
    # (full_name, department, role, embedding_seed)
    ("Arjun Sharma",    "Civil Engineering",   "Site Supervisor",   101),
    ("Priya Nair",      "Safety & Compliance", "HSE Officer",       202),
    ("Chen Wei",        "Logistics",           "Fleet Coordinator", 303),
    ("Amara Osei",      "Agriculture",         "Field Technician",  404),
    ("Fatima Al-Rashid","Quality Assurance",   "QA Inspector",      505),
    ("Miguel Torres",   "Operations",          "Shift Lead",        606),
]


@pytest.fixture(scope="function")
def org_ids(db_path):
    """
    Fixture: seeds the two mock organizations into db_path and returns
    their UUIDs as a list. Mirrors Step 2 of mock_test.main().
    """
    from database import insert_organization

    ids = []
    for org_data in _SEED_ORGANIZATIONS:
        org_id = insert_organization(
            name=org_data["name"],
            region=org_data["region"],
            contact_email=org_data["contact_email"],
            db_path=db_path,
        )
        ids.append(org_id)
    return ids


@pytest.fixture(scope="function")
def employee_records(org_ids, db_path):
    """
    Fixture: seeds mock employees (with L2-normalized 512-d embeddings) into
    db_path and returns a list of (employee_id, org_id, embedding) tuples.
    Mirrors Step 3 of mock_test.main().
    """
    from database import insert_employee

    records = []
    for i, (full_name, department, role, seed) in enumerate(_SEED_EMPLOYEES):
        org_id = org_ids[i % len(org_ids)]
        embedding = _generate_mock_embedding(dimensions=512, seed=seed)
        emp_id = insert_employee(
            organization_id=org_id,
            full_name=full_name,
            face_embedding=embedding,
            department=department,
            role=role,
            embedding_model="arcface_iresnet100_512d",
            db_path=db_path,
        )
        records.append((emp_id, org_id, embedding))
    return records


@pytest.fixture(scope="function")
def log_ids(employee_records, db_path):
    """
    Fixture: simulates facial scan events for each employee and returns the
    list of log_ids written to the database. Mirrors Step 4 of mock_test.main().
    """
    from database import insert_log

    scan_scenarios = [
        ("CLOCK_IN",       0.03),
        ("CLOCK_IN",       0.12),
        ("CLOCK_OUT",      0.04),
        ("ACCESS_GRANTED", 0.02),
        ("ACCESS_DENIED",  0.50),
        ("CLOCK_IN",       0.06),
    ]

    ids = []
    for i, (emp_id, org_id, embedding) in enumerate(employee_records):
        event_type, noise_factor = scan_scenarios[i % len(scan_scenarios)]

        # Simulate noisy scan embedding
        noisy_raw = [v + random.gauss(0, noise_factor) for v in embedding]
        magnitude = math.sqrt(sum(v ** 2 for v in noisy_raw))
        scan_embedding = [v / magnitude for v in noisy_raw]
        verification_score = round(
            sum(a * b for a, b in zip(embedding, scan_embedding)), 6
        )
        liveness_score = round(random.uniform(0.85, 0.99), 4)
        liveness_passed = liveness_score >= 0.80

        log_id = insert_log(
            organization_id=org_id,
            employee_id=emp_id,
            event_type=event_type,
            verification_score=verification_score,
            liveness_score=liveness_score,
            liveness_passed=liveness_passed,
            device_id="EDGE-DEVICE-SN-2024-EDGE-001",
            latitude=round(19.0760 + random.uniform(-0.001, 0.001), 6),
            longitude=round(72.8777 + random.uniform(-0.001, 0.001), 6),
            metadata={
                "camera_model": "ArduCam 64MP",
                "firmware_version": "v2.1.4",
                "noise_factor": noise_factor,
            },
            db_path=db_path,
        )
        ids.append(log_id)
    return ids
