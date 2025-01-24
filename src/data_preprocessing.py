# data_preprocessing.py

import os
import cv2


def preprocess_images(input_dir, output_dir, new_size=(640, 480)):
    """
    Reads, resizes, and saves images to maintain a uniform dimension.
    Useful for consistent input to the pipeline.
    """
    valid_ext = ('.jpg', '.png', '.jpeg')
    os.makedirs(output_dir, exist_ok=True)

    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith(valid_ext):
            in_path = os.path.join(input_dir, file_name)
            image = cv2.imread(in_path)
            if image is None:
                print(f"[WARN] Failed to read {file_name}. Skipping.")
                continue

            resized = cv2.resize(image, new_size)
            out_path = os.path.join(output_dir, file_name)
            cv2.imwrite(out_path, resized)
            print(f"Processed: {out_path}")



#python -c "from data_preprocessing import preprocess_images; preprocess_images('../data/raw', '../data/processed')"