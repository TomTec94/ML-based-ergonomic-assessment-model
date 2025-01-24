import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose


def extract_landmarks(image_bgr):
    """
    Runs MediaPipe Pose on a single BGR image and returns
    a dictionary of landmark coordinates if successful.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    with mp_pose.Pose(static_image_mode=True) as pose_detector:
        results = pose_detector.process(image_rgb)
        if not results.pose_landmarks:
            return None

        # Convert to a dictionary: {id: (x, y)}
        landmarks_dict = {}
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            landmarks_dict[idx] = (lm.x, lm.y)
        return landmarks_dict