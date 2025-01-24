import os
import cv2
from pose_estimation import extract_landmarks
from rule_based_model import classify_posture

def internal_evaluation(image_dir):
    """
    Runs the pipeline on each image in 'image_dir'
    and prints classification results for sanity-checking.
    """
    valid_ext = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_ext)]

    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            print(f"[ERROR] Could not read {img_path}. Skipping.")
            continue

        landmarks = extract_landmarks(image_bgr)
        posture_label = classify_posture(landmarks)

        print(f"File: {img_file} => {posture_label}")