# data_preprocessing.py

import os
import cv2
import imghdr
import shutil


def preprocess_images(input_dir, output_dir, min_width=400, min_height=400):
    """
    Validates each file in 'input_dir' to ensure it is a loadable image
    and meets minimum dimension requirements (min_width x min_height).
    If valid, copies it unchanged to 'output_dir'.

    Steps:
      1) Check basic extension (.jpg, .jpeg, .png).
      2) Confirm via imghdr that the file is 'jpeg' or 'png'.
      3) Attempt to read with OpenCV to ensure it is loadable.
      4) Verify image width >= min_width and height >= min_height.
      5) If all checks pass, copy the file to 'output_dir'. Otherwise, skip.

    :param input_dir:   The folder containing images to verify.
    :param output_dir:  The folder where valid images are copied.
    :param min_width:   Required minimum width (default=400).
    :param min_height:  Required minimum height (default=400).
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Starting image-check from '{input_dir}' to '{output_dir}' ...\n"
          f"Required minimum dimensions: {min_width}x{min_height}\n")

    valid_exts = ('.jpg', '.jpeg', '.png')
    num_copied = 0
    num_invalid = 0

    for file_name in os.listdir(input_dir):
        in_path = os.path.join(input_dir, file_name)

        # 1) Quick extension check
        if not file_name.lower().endswith(valid_exts):
            print(f"[X] '{file_name}' - invalid extension.")
            num_invalid += 1
            continue

        # 2) Confirm via imghdr
        file_type = imghdr.what(in_path)
        if file_type not in ['jpeg', 'png']:
            print(f"[X] '{file_name}' - imghdr says '{file_type}', not accepted.")
            num_invalid += 1
            continue

        # 3) Try loading with OpenCV
        image = cv2.imread(in_path)
        if image is None:
            print(f"[X] '{file_name}' - cannot read via OpenCV.")
            num_invalid += 1
            continue

        # 4) Check image dimensions
        h, w, _ = image.shape
        if w < min_width or h < min_height:
            print(f"[X] '{file_name}' - too small ({w}x{h}). "
                  f"Needs at least {min_width}x{min_height}.")
            num_invalid += 1
            continue

        # 5) Copy the file unchanged
        out_path = os.path.join(output_dir, file_name)
        shutil.copy2(in_path, out_path)
        print(f"[OK] '{file_name}' -> copied to '{out_path}'.")
        num_copied += 1

    print("\n=== Processing Completed ===")
    print(f"Copied images: {num_copied}")
    print(f"Invalid/skipped files: {num_invalid}")

# Example command-line usage:
#   python -c "from data_preprocessing import preprocess_images; \
#   preprocess_images('../data/raw_images', '../data/checked_images', 400, 400)"