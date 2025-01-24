# main.py
import argparse
import os
import cv2
from pose_estimation import extract_landmarks
from rule_based_model import classify_posture

def analyze_images(image_dir):
    valid_ext = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_ext)]

    for img_file in image_files:
        path = os.path.join(image_dir, img_file)
        image_bgr = cv2.imread(path)
        if image_bgr is None:
            print(f"[WARN] Cannot read {img_file}, skipping.")
            continue

        landmarks = extract_landmarks(image_bgr)
        result = classify_posture(landmarks)
        print(f"{img_file}: {result}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Posture Detection CLI")
    parser.add_argument("--dir", type=str, default="../data/test_images",
                        help="Directory of images to analyze.")
    args = parser.parse_args()

    print(f"Analyzing folder: {args.dir}")
    analyze_images(args.dir)


#python main.py --dir ../data/test_images