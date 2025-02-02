# internal_evaluation.py
import os
import cv2
import logging
from pose_estimation import extract_landmarks
from angle_calculation import compute_posture_angles
from rule_based_model import rule_based_posture_analysis

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def internal_evaluation(image_dir):
    valid_ext = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_ext)]

    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            logging.error(f"Could not read {img_path}. Skipping.")
            continue

        landmarks_dict, _, feedback_msg = extract_landmarks(image_bgr)
        if landmarks_dict is None:
            logging.error(f"No landmarks detected in {img_file}.")
            continue

        side_used = "LEFT"
        if "RIGHT" in feedback_msg.upper():
            side_used = "RIGHT"

        angles_dict = compute_posture_angles(landmarks_dict, side=side_used.lower())
        results = rule_based_posture_analysis(image_bgr, angles_dict, side=side_used.lower(), landmarks_dict=landmarks_dict)
        logging.info(f"File: {img_file} => Score: {results['score']}%, Rating: {results['rating']}")

if __name__ == "__main__":
    internal_evaluation("../data/raw_images")