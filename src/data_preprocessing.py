# data_preprocessing.py

import os
import cv2
import imghdr
import shutil


def preprocess_images(input_dir, output_dir):
    """
    Examines each file in 'input_dir' to determine if it's an actual image (extension + imghdr).
    If valid, copies it unchanged to 'output_dir'.
    No resizing or bounding-box detection is performed.

    Steps:
      1) Check extension (.jpg, .jpeg, .png).
      2) Double-check the type with imghdr.
      3) Attempt to read with OpenCV to confirm readability (optional but safer).
      4) If valid, copy file as-is to 'output_dir'.
      5) Otherwise, skip and log a warning.

    Parameters:
      - input_dir: folder containing files to check
      - output_dir: folder where valid images are copied
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Starting image-check (no resizing) from '{input_dir}' to '{output_dir}' ...")

    valid_ext = ('.jpg', '.jpeg', '.png')
    num_copied = 0
    num_invalid = 0

    for file_name in os.listdir(input_dir):
        in_path = os.path.join(input_dir, file_name)

        # 1) Quick extension check
        if not file_name.lower().endswith(valid_ext):
            print(f" - [X] '{file_name}' -> Not a valid image extension.")
            num_invalid += 1
            continue

        # 2) Confirm via imghdr
        file_type = imghdr.what(in_path)
        if file_type not in ['jpeg', 'png']:
            print(f" - [X] '{file_name}' -> Not a valid image (imghdr says '{file_type}').")
            num_invalid += 1
            continue

        # 3) Optional: attempt to load with OpenCV for a final check
        image = cv2.imread(in_path)
        if image is None:
            print(f" - [X] '{file_name}' -> Could not be read by OpenCV. Skipping.")
            num_invalid += 1
            continue

        # 4) Copy the file unmodified
        out_path = os.path.join(output_dir, file_name)
        shutil.copy2(in_path, out_path)

        print(f" - [✔] '{file_name}' -> Verified as an image, copied to '{out_path}'.")
        num_copied += 1

    print("Processing completed.")
    print(f"   Copied images: {num_copied}")
    print(f"   Invalid/skipped files: {num_invalid}")

# Example usage from a Terminal:
# python -c "from data_preprocessing import preprocess_images; \
#            preprocess_images('../data/raw_images', '../data/checked_images')"