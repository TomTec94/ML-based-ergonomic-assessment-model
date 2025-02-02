# pose_estimation.py
import cv2
import mediapipe as mp
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

mp_pose = mp.solutions.pose

def extract_landmarks(image_bgr, visualize=False):
    """
    Runs MediaPipe Pose on image_bgr.
    Returns:
      - all_landmarks: dict mapping landmark id to (x, y)
      - annotated_img: the original image (unchanged if visualize=False)
      - feedback: a summary message (e.g., chosen side)
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as pose_detector:
        results = pose_detector.process(image_rgb)

    if not results.pose_landmarks:
        logging.error("No landmarks detected.")
        return None, image_bgr, "No landmarks detected."

    all_landmarks = {}
    for idx, lm in enumerate(results.pose_landmarks.landmark):
        all_landmarks[idx] = (lm.x, lm.y)

    feedback = "Chosen side: LEFT"  # This may be updated later with a proper side detection
    return all_landmarks, image_bgr, feedback