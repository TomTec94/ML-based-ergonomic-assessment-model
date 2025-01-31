# main.py

import os
import sys
import argparse
import cv2

from data_preprocessing import preprocess_images
from pose_estimation import extract_landmarks
from angle_calculation import compute_posture_angles
from rule_based_model import rule_based_posture_analysis


###########################
# 2) LAUNCH UI
###########################
def launch_gui():
    print("=== Starting the Tkinter GUI ===")
    from ui_tool import ErgoApp
    app = ErgoApp()
    app.mainloop()

###########################
# 3) MAIN
###########################
def main():
    parser = argparse.ArgumentParser(
        description="Either run the CLI pipeline or the Tkinter GUI."
    )
    parser.add_argument("--gui", action="store_true",
                        help="Run the UI instead of CLI batch.")
    parser.add_argument("--input_dir", type=str, default="../data/raw_images")
    parser.add_argument("--output_dir", type=str, default="./data/processed")
    parser.add_argument("--no_preprocess", action="store_true")
    parser.add_argument("--single_image", type=str, default="")
    args = parser.parse_args()

    if args.gui:
        launch_gui()
    else:
        launch_gui()

if __name__ == "__main__":
    main()