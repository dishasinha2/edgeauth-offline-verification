import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import random
import os

class LivenessChallenge:
    def __init__(self):
        # Initialize FaceLandmarker using the Tasks API
        base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                               output_face_blendshapes=False,
                                               output_facial_transformation_matrixes=False,
                                               num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        self.challenges = ["Blink Twice", "Turn Head Left", "Turn Head Right"]
        
        # Landmarks (same indices apply for the new API)
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        self.NOSE_TIP = 1
        self.LEFT_CHEEK = 234
        self.RIGHT_CHEEK = 454
        
        self.EAR_THRESHOLD = 0.20
        self.HEAD_TURN_LEFT_THRESHOLD = 0.35
        self.HEAD_TURN_RIGHT_THRESHOLD = 0.65

    def euclidean_distance(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def eye_aspect_ratio(self, eye_points):
        v1 = self.euclidean_distance(eye_points[1], eye_points[5])
        v2 = self.euclidean_distance(eye_points[2], eye_points[4])
        h = self.euclidean_distance(eye_points[0], eye_points[3])
        if h == 0: return 0
        return (v1 + v2) / (2.0 * h)
        
    def check_liveness(self, frame_rgb, current_challenge):
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Detect landmarks
        detection_result = self.detector.detect(mp_image)
        
        if not detection_result.face_landmarks:
            return False, "No face detected"
            
        landmarks = detection_result.face_landmarks[0]
        
        if current_challenge == "Blink Twice":
            left_points = [(landmarks[i].x, landmarks[i].y) for i in self.LEFT_EYE]
            right_points = [(landmarks[i].x, landmarks[i].y) for i in self.RIGHT_EYE]
            ear = (self.eye_aspect_ratio(left_points) + self.eye_aspect_ratio(right_points)) / 2.0
            if ear < self.EAR_THRESHOLD:
                return True, "Blinked"
                
        elif current_challenge == "Turn Head Left":
            ratio = self.get_head_pose_ratio(landmarks)
            if ratio < self.HEAD_TURN_LEFT_THRESHOLD:
                return True, "Turned Left"
                
        elif current_challenge == "Turn Head Right":
            ratio = self.get_head_pose_ratio(landmarks)
            if ratio > self.HEAD_TURN_RIGHT_THRESHOLD:
                return True, "Turned Right"
                
        return False, "Pending"

    def get_head_pose_ratio(self, landmarks):
        nose_x = landmarks[self.NOSE_TIP].x
        left_cheek_x = landmarks[self.LEFT_CHEEK].x
        right_cheek_x = landmarks[self.RIGHT_CHEEK].x
        
        dist_left = abs(nose_x - left_cheek_x)
        dist_right = abs(nose_x - right_cheek_x)
        
        if (dist_left + dist_right) == 0: return 0.5
        return dist_left / (dist_left + dist_right)

if __name__ == "__main__":
    if not os.path.exists("face_landmarker.task"):
        print("Model file missing. Run setup_models.py first.")
        exit(1)
        
    liveness = LivenessChallenge()
    cap = cv2.VideoCapture(2)
    
    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Please ensure your camera is connected and not being used by another application.")
        exit(1)
    
    current_challenge = random.choice(liveness.challenges)
    print(f"Initial Challenge: {current_challenge}")
    
    blink_count = 0
    blink_cooldown = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: continue
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        passed, status = liveness.check_liveness(rgb_frame, current_challenge)
        
        if blink_cooldown > 0:
            blink_cooldown -= 1
            
        if current_challenge == "Blink Twice":
            if passed and blink_cooldown == 0:
                blink_count += 1
                blink_cooldown = 15
            # We flip the frame first so the text doesn't appear backwards
            display_frame = cv2.flip(frame, 1)
            cv2.putText(display_frame, f"Blinks: {blink_count}/2", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            if blink_count >= 2:
                current_challenge = random.choice(liveness.challenges)
                blink_count = 0
        else:
            display_frame = cv2.flip(frame, 1)
            if passed:
                status = "Challenge Passed!"
                cv2.putText(
                    display_frame,
                    status,
                    (50, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2
                )
                cv2.imshow("Liveness Challenge", display_frame)
                cv2.waitKey(1000)
        
                current_challenge = random.choice(liveness.challenges)
                
        cv2.putText(display_frame, f"Challenge: {current_challenge}", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(display_frame, f"Status: {status}", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow("Liveness Challenge", display_frame)
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
