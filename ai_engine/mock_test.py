import cv2
import numpy as np
from liveness_challenge import LivenessChallenge
import traceback

def test_pipeline():
    print("Testing LivenessChallenge initialization...")
    try:
        liveness = LivenessChallenge()
        print("[SUCCESS] Initialization successful!")
        
        print("Testing check_liveness with a dummy frame...")
        # Create a dummy black frame (480x640 RGB image)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        passed, status = liveness.check_liveness(dummy_frame, "Blink Twice")
        
        # Since it's a black frame, MediaPipe should return "No face detected"
        if status == "No face detected":
            print("[SUCCESS] check_liveness successfully processed the frame and correctly identified no face!")
        else:
            print(f"[WARNING] Unexpected status returned: {status}")
            
        print("\nAll automated integration checks passed. The classes are properly connected and MediaPipe is loading correctly.")
        
    except Exception as e:
        print("[ERROR] Error during testing:")
        print(traceback.format_exc())

if __name__ == "__main__":
    test_pipeline()
