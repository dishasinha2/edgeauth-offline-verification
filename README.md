# EdgeAuth — Offline Workforce Verification Platform

**Offline Multi-Organization Workforce Verification Platform using Edge AI Facial Recognition and Adaptive Liveness Detection for Remote and Low-Connectivity Environments.**

---

## Architecture Overview

```
Camera Frame
      ↓
MediaPipe FaceLandmarker (468 landmarks)
      ↓
Challenge-Based Liveness Detection
  • Blink Twice
  • Turn Head Left / Right
  • Smile
      ↓
Liveness Passed?
      ↓
face_recognition (dlib ResNet — offline, pretrained)
      ↓
128-d Embedding → Cosine Similarity
      ↓
Verified / ACCESS DENIED
      ↓
SQLite (offline log + sync queue)
      ↓
AWS Lambda Sync (when connectivity available)
```

---

## Project Structure

```
edgeauth-offline-verification/
│
├── ai_engine/                        # Edge AI inference engine
│   ├── liveness_challenge.py         # Challenge-based liveness (Blink/Turn/Smile)
│   ├── face_recognizer.py            # ★ NEW: 128-d embedding extraction + cosine similarity
│   ├── face_pipeline.py              # ★ NEW: Unified live verification pipeline
│   ├── enroll_employee.py            # ★ NEW: Webcam enrollment CLI tool
│   ├── spoof_test.py                 # ★ NEW: Anti-spoofing demonstration
│   ├── blink_detection.py            # Standalone blink EAR detector
│   ├── headpose_detection.py         # Standalone head pose detector
│   ├── facemesh_test.py              # MediaPipe face mesh visualizer
│   ├── mock_test.py                  # AI engine integration test
│   ├── face_landmarker.task          # MediaPipe model file
│   ├── enrollments/                  # Saved enrollment photos (gitignored)
│   └── requirements.txt
│
├── backend/
│   ├── local_device/                 # Edge device SQLite backend
│   │   ├── database.py               # Schema + CRUD (organizations, employees, logs)
│   │   ├── sync_engine.py            # AWS sync with exponential backoff
│   │   ├── mock_test.py              # Full integration simulation
│   │   ├── test_database.py          # pytest suite
│   │   └── test_sync_engine.py       # pytest suite
│   └── aws_serverless/               # AWS Lambda backend
│       ├── lambda_function.py        # REST API handler
│       ├── template.yaml             # SAM deployment template
│       └── test_lambda_function.py   # pytest suite
│
├── frontend/                         # React Native mobile app
│   ├── App.tsx                       # Splash + Login screens
│   └── ...
│
├── run.py                            # Quick launcher for liveness challenge
├── run_liveness.bat                  # Windows launcher
└── .gitignore
```

---

## Quick Start

### 1. Install dependencies

```bash
cd ai_engine
pip install -r requirements.txt
```

> **Note:** `face_recognition` downloads a ~100MB pretrained dlib ResNet model on first install. This is a one-time download — all subsequent inference runs fully offline.

---

### 2. Enroll an employee

```bash
cd ai_engine
python enroll_employee.py --name "Arjun Sharma" --dept "Engineering" --role "Site Supervisor"
```

- Opens webcam preview
- Press **SPACE** when face is clearly visible
- Embedding stored in SQLite, photo saved to `ai_engine/enrollments/`

---

### 3. Run the live verification pipeline

```bash
cd ai_engine
python face_pipeline.py
```

Full flow:
1. Face detected → random liveness challenge assigned
2. Complete challenge (blink / turn head / smile)
3. Embedding extracted and matched against enrolled employees
4. Result logged to SQLite and queued for AWS sync

Options:
```bash
python face_pipeline.py --threshold 0.80       # stricter matching
python face_pipeline.py --device-id EDGE-001   # set device ID
python face_pipeline.py --db ./custom.db        # custom DB path
```

---

### 4. Run the liveness challenge standalone

```bash
python run.py
# or
run_liveness.bat
```

---

### 5. Anti-spoofing test

```bash
cd ai_engine

# Test a static photo — proves it FAILS all challenges
python spoof_test.py --image path/to/face_photo.jpg

# Compare two face photos for similarity
python spoof_test.py --compare person_a.jpg person_b.jpg

# Interactive capture → replay demo
python spoof_test.py --demo
```

---

### 6. Backend tests

```bash
cd backend/local_device
pip install -r requirements.txt
python mock_test.py          # full integration simulation
pytest test_database.py      # unit tests
pytest test_sync_engine.py   # sync engine tests
```

---

## Liveness Detection

Challenges are **randomly selected** per session from:

| Challenge       | Detection Method                        | Anti-Spoof Strength |
|----------------|-----------------------------------------|---------------------|
| Blink Twice    | Eye Aspect Ratio (EAR) < 0.20 × 2      | ✅ High             |
| Turn Head Left | Nose-to-cheek ratio < 0.35             | ✅ High             |
| Turn Head Right| Nose-to-cheek ratio > 0.65             | ✅ High             |
| Smile          | Mouth width/height ratio > 5.0         | ✅ Medium           |

A static photo or video replay **cannot** pass these challenges — the landmark geometry stays frozen.

---

## Face Recognition

- **Model:** dlib ResNet (via `face_recognition` library) — pretrained, offline
- **Embedding:** 128-dimensional L2-normalized float vector
- **Similarity:** Cosine similarity (dot product of normalized vectors)
- **Default threshold:** 0.75 (configurable per-org via `liveness_threshold` DB field)
- **Storage:** JSON-serialized in SQLite `employees.face_embedding`

---

## Offline-First Design

- All verification happens **on-device** — no internet required
- Logs stored locally in SQLite with full audit trail
- Sync queue uploads to AWS when connectivity is available
- **Data safety guarantee:** local data is never purged until AWS returns HTTP 200

---

## Environment Variables

| Variable              | Default                        | Description                    |
|----------------------|--------------------------------|--------------------------------|
| `WVP_DB_PATH`        | `workforce_verification.db`    | SQLite database path           |
| `WVP_AWS_API_ENDPOINT` | (not set)                   | AWS API Gateway sync endpoint  |
| `EDGEAUTH_FACE_MODEL`| `hog`                          | `hog` (fast) or `cnn` (GPU)   |
