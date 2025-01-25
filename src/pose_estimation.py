# pose_estimation.py

import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

def extract_landmarks(image_bgr, visualize=False):
    """
    Runs MediaPipe Pose, returns:
      - all_landmarks: dict {id: (x, y)} for all recognized points
      - annotated_img: same image (unmodified) if visualize=False
      - feedback: text summary
    We skip the default drawing of all 33 points to keep it minimal.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.9,
        min_tracking_confidence=0.9
    ) as pose_detector:
        results = pose_detector.process(image_rgb)

    if not results.pose_landmarks:
        return None, image_bgr, "No landmarks detected."

    # Convert raw 33 landmarks to a dictionary of normalized coords
    all_landmarks = {}
    for idx, lm in enumerate(results.pose_landmarks.landmark):
        all_landmarks[idx] = (lm.x, lm.y)

    # For demonstration, let's say the side is "left" by default:
    feedback = "Chosen side: LEFT\n(You can add coverage logic here.)"

    return all_landmarks, image_bgr, feedback