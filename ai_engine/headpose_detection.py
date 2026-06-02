import cv2
import mediapipe as mp

NOSE_TIP = 1
LEFT_CHEEK = 234
RIGHT_CHEEK = 454
HEAD_TURN_LEFT_THRESHOLD = 0.35
HEAD_TURN_RIGHT_THRESHOLD = 0.65

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


def verify_headpose(image_bgr):
    """
    Backend-callable head-pose verification.
    Accepts an OpenCV BGR image and returns a JSON-compatible dictionary.
    """
    if image_bgr is None:
        return {"success": False, "status": "Invalid image"}

    rgb_frame = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = _get_face_mesh().process(rgb_frame)

    if not results.multi_face_landmarks:
        return {"success": False, "status": "No face detected"}

    landmarks = results.multi_face_landmarks[0].landmark

    nose_x = landmarks[NOSE_TIP].x
    left_cheek_x = landmarks[LEFT_CHEEK].x
    right_cheek_x = landmarks[RIGHT_CHEEK].x

    dist_left = abs(nose_x - left_cheek_x)
    dist_right = abs(nose_x - right_cheek_x)

    if dist_right == 0 or dist_left == 0:
        return {"success": False, "status": "Center"}

    ratio = dist_left / (dist_left + dist_right)

    status_text = "Center"
    success = False
    if ratio > HEAD_TURN_RIGHT_THRESHOLD:
        status_text = "Turned Right"
        success = True
    elif ratio < HEAD_TURN_LEFT_THRESHOLD:
        status_text = "Turned Left"
        success = True

    return {"success": success, "status": status_text, "ratio": float(ratio)}


def main():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    print("Starting Head Pose Detection Test. Press 'q' to quit.")

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
                
                nose_x = landmarks[NOSE_TIP].x
                left_cheek_x = landmarks[LEFT_CHEEK].x
                right_cheek_x = landmarks[RIGHT_CHEEK].x
                
                # Calculate the ratio of distances from nose to cheeks
                # (nose_x - right_cheek_x) / (left_cheek_x - right_cheek_x)
                # left_cheek is on the right side of the image (due to mirror), right cheek is on the left side
                
                dist_left = abs(nose_x - left_cheek_x)
                dist_right = abs(nose_x - right_cheek_x)
                
                if dist_right == 0 or dist_left == 0:
                    continue
                    
                ratio = dist_left / (dist_left + dist_right)
                
                status_text = "Center"
                # If ratio is close to 0.5, head is centered. 
                # If ratio is > 0.65, head is turned right (from user's perspective, nose is closer to right cheek)
                # If ratio is < 0.35, head is turned left
                if ratio > 0.65:
                    status_text = "Turned Right"
                elif ratio < 0.35:
                    status_text = "Turned Left"

                # Draw markers
                nose_pos = (int(nose_x * w), int(landmarks[NOSE_TIP].y * h))
                left_pos = (int(left_cheek_x * w), int(landmarks[LEFT_CHEEK].y * h))
                right_pos = (int(right_cheek_x * w), int(landmarks[RIGHT_CHEEK].y * h))
                
                cv2.circle(frame, nose_pos, 4, (0, 0, 255), -1)
                cv2.circle(frame, left_pos, 4, (255, 0, 0), -1)
                cv2.circle(frame, right_pos, 4, (255, 0, 0), -1)
                
                cv2.putText(frame, f"Pose: {status_text} (Ratio: {ratio:.2f})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Head Pose Detection", cv2.flip(frame, 1))
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
