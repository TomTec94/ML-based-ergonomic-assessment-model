# data_preprocessing.py
import os
import cv2
import imghdr
import shutil
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def preprocess_images(input_dir, output_dir, min_width=400, min_height=400):
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Starting image-check from '{input_dir}' to '{output_dir}' ...")
    logging.info(f"Required minimum dimensions: {min_width}x{min_height}")

    valid_exts = ('.jpg', '.jpeg', '.png')
    num_copied = 0
    num_invalid = 0

    for file_name in os.listdir(input_dir):
        in_path = os.path.join(input_dir, file_name)

        if not file_name.lower().endswith(valid_exts):
            logging.warning(f"'{file_name}' - invalid extension.")
            num_invalid += 1
            continue

        file_type = imghdr.what(in_path)
        if file_type not in ['jpeg', 'png']:
            logging.warning(f"'{file_name}' - imghdr says '{file_type}', not accepted.")
            num_invalid += 1
            continue

        image = cv2.imread(in_path)
        if image is None:
            logging.warning(f"'{file_name}' - cannot be read via OpenCV.")
            num_invalid += 1
            continue

        h, w, _ = image.shape
        if w < min_width or h < min_height:
            logging.warning(f"'{file_name}' - too small ({w}x{h}). Needs at least {min_width}x{min_height}.")
            num_invalid += 1
            continue

        out_path = os.path.join(output_dir, file_name)
        shutil.copy2(in_path, out_path)
        logging.info(f"'{file_name}' -> copied to '{out_path}'.")
        num_copied += 1

    logging.info("=== Processing Completed ===")
    logging.info(f"Copied images: {num_copied}")
    logging.info(f"Invalid/skipped files: {num_invalid}")