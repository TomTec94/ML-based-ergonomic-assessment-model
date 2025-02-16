# BachelorThesis/src/pose_estimation.py
import cv2
import mediapipe as mp
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

mp_pose = mp.solutions.pose

def extract_landmarks(image_bgr, visualize=False):
    """
    Runs MediaPipe Pose on image_bgr.
    Returns:
      - all_landmarks: dict mapping landmark id to (x, y) (all detected landmarks are preserved)
      - annotated_img: the original image (unchanged if visualize=False)
      - feedback: a summary message indicating the chosen side (based on a symmetric heuristic)
                  and any warnings about missing landmarks.
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

    # Build dictionary of all landmarks.
    all_landmarks = {}
    for idx, lm in enumerate(results.pose_landmarks.landmark):
        all_landmarks[idx] = (lm.x, lm.y)

    feedback_msgs = []

    # Define symmetric pairs for the following landmarks:
    # ear: (7,8), shoulder: (11,12), elbow: (13,14), wrist: (15,16),
    # hip: (23,24), knee: (25,26), ankle: (27,28)
    symmetric_pairs = {
        'ear': (7, 8),
        'shoulder': (11, 12),
        'elbow': (13, 14),
        'wrist': (15, 16),
        'hip': (23, 24),
        'knee': (25, 26),
        'ankle': (27, 28)
    }
    for name, (l_id, r_id) in symmetric_pairs.items():
        if l_id not in all_landmarks and r_id not in all_landmarks:
            feedback_msgs.append(f"Neither left nor right {name} landmark could be detected.")
        elif l_id not in all_landmarks:
            feedback_msgs.append(f"Left {name} landmark (ID {l_id}) could not be detected; defaulting to RIGHT for {name}.")
        elif r_id not in all_landmarks:
            feedback_msgs.append(f"Right {name} landmark (ID {r_id}) could not be detected; defaulting to LEFT for {name}.")

    left_ids = [7, 11, 13, 15, 23, 25, 27]
    right_ids = [8, 12, 14, 16, 24, 26, 28]

    left_xs = [all_landmarks[l_id][0] for l_id in left_ids if l_id in all_landmarks]
    right_xs = [all_landmarks[r_id][0] for r_id in right_ids if r_id in all_landmarks]

    if not left_xs and not right_xs:
        feedback_msgs.append("No symmetric landmarks available to decide side. Defaulting to LEFT.")
        chosen_side = "LEFT"
    elif not left_xs:
        feedback_msgs.append("No left-side landmarks available. Defaulting to RIGHT.")
        chosen_side = "RIGHT"
    elif not right_xs:
        feedback_msgs.append("No right-side landmarks available. Defaulting to LEFT.")
        chosen_side = "LEFT"
    else:
        avg_left = sum(left_xs) / len(left_xs)
        avg_right = sum(right_xs) / len(right_xs)
        if avg_left < avg_right:
            chosen_side = "LEFT"
        else:
            chosen_side = "RIGHT"
        feedback_msgs.append(f"Chosen side based on average x-coordinates: {chosen_side} (Left avg: {avg_left:.3f}, Right avg: {avg_right:.3f})")

    feedback = "; ".join(feedback_msgs)
    return all_landmarks, image_bgr, feedback