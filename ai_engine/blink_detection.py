import cv2
import mediapipe as mp
import numpy as np
import time

LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
EAR_THRESHOLD = 0.20

_face_mesh = None


def _get_face_mesh():
    global _face_mesh
    if _face_mesh is None:
        mp_face_mesh = mp.solutions.face_mesh
        _face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    return _face_mesh


def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def eye_aspect_ratio(eye_points):
    # eye_points should be a list of 6 points: [p1, p2, p3, p4, p5, p6]
    # Vertical distances
    v1 = euclidean_distance(eye_points[1], eye_points[5])
    v2 = euclidean_distance(eye_points[2], eye_points[4])
    # Horizontal distance
    h = euclidean_distance(eye_points[0], eye_points[3])
    
    if h == 0:
        return 0
    ear = (v1 + v2) / (2.0 * h)
    return ear


def verify_blink(image_bgr):
    """
    Backend-callable blink verification.
    Accepts an OpenCV BGR image and returns a JSON-compatible dictionary.
    """
    if image_bgr is None:
        return {"success": False, "status": "Invalid image"}

    h, w, _ = image_bgr.shape
    rgb_frame = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = _get_face_mesh().process(rgb_frame)

    if not results.multi_face_landmarks:
        return {"success": False, "status": "No face detected"}

    landmarks = results.multi_face_landmarks[0].landmark

    def get_coords(indices):
        return [(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in indices]

    left_eye_points = get_coords(LEFT_EYE_INDICES)
    right_eye_points = get_coords(RIGHT_EYE_INDICES)

    left_ear = eye_aspect_ratio(left_eye_points)
    right_ear = eye_aspect_ratio(right_eye_points)
    avg_ear = (left_ear + right_ear) / 2.0

    if avg_ear < EAR_THRESHOLD:
        return {"success": True, "status": "Blink Detected!", "ear": float(avg_ear)}

    return {"success": False, "status": "Eyes Open", "ear": float(avg_ear)}


def main():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)

    print("Starting Blink Detection Test. Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                
                def get_coords(indices):
                    return [(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in indices]
                
                left_eye_points = get_coords(LEFT_EYE_INDICES)
                right_eye_points = get_coords(RIGHT_EYE_INDICES)
                
                left_ear = eye_aspect_ratio(left_eye_points)
                right_ear = eye_aspect_ratio(right_eye_points)
                
                avg_ear = (left_ear + right_ear) / 2.0
                
                # Draw eye points
                for p in left_eye_points + right_eye_points:
                    cv2.circle(frame, p, 2, (0, 255, 0), -1)

                status_text = "Eyes Open"
                if avg_ear < EAR_THRESHOLD:
                    status_text = "Blink Detected!"
                    cv2.putText(frame, status_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                cv2.putText(frame, f"EAR: {avg_ear:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Blink Detection", cv2.flip(frame, 1))
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
