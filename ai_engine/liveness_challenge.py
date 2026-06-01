import cv2
import mediapipe as mp
import numpy as np
import random
import time

class LivenessChallenge:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.challenges = ["Blink Twice", "Turn Head Left", "Turn Head Right"]
        
        # Landmarks
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
        results = self.face_mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            return False, "No face detected"
            
        landmarks = results.multi_face_landmarks[0].landmark
        
        if current_challenge == "Blink Twice":
            # Just detecting one blink for simplicity in this frame, 
            # the caller should track state across frames for "Twice"
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
    # Test orchestrator
    liveness = LivenessChallenge()
    cap = cv2.VideoCapture(0)
    
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
                blink_cooldown = 15 # wait 15 frames before next blink counts
            cv2.putText(frame, f"Blinks: {blink_count}/2", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            if blink_count >= 2:
                current_challenge = random.choice(liveness.challenges)
                blink_count = 0
        else:
            if passed:
                current_challenge = random.choice(liveness.challenges)
                
        cv2.putText(frame, f"Challenge: {current_challenge}", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Status: {status}", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow("Liveness Challenge", cv2.flip(frame, 1))
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
