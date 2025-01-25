# main.py

import argparse
import os
import math
import cv2

from pose_estimation import extract_landmarks
from angle_calculation import compute_posture_angles

LEFT_HIP, LEFT_KNEE, LEFT_ANKLE = 23, 25, 27
LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST = 11, 13, 15
LEFT_EAR = 7

RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE = 24, 26, 28
RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST = 12, 14, 16
RIGHT_EAR = 8


def draw_relevant_landmarks_and_lines(image_bgr, landmarks_dict, side='left'):
    """
    Draws only the circles (key landmarks) and lines (connections)
    for knee, hip, elbow, and head_to_shoulder angles. No arcs.
    """
    h, w, _ = image_bgr.shape

    if side == 'left':
        hip_id, knee_id, ankle_id = LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
        shoulder_id, elbow_id, wrist_id = LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST
        ear_id = LEFT_EAR
    else:
        hip_id, knee_id, ankle_id = RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
        shoulder_id, elbow_id, wrist_id = RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST
        ear_id = RIGHT_EAR

    relevant_ids = [hip_id, knee_id, ankle_id, shoulder_id, elbow_id, wrist_id, ear_id]
    lines = [
        (hip_id, knee_id), (knee_id, ankle_id),
        (shoulder_id, hip_id), (hip_id, knee_id),
        (shoulder_id, elbow_id), (elbow_id, wrist_id),
        (ear_id, shoulder_id), (shoulder_id, hip_id)
    ]

    # Draw circles (green) at the relevant landmarks
    for lid in relevant_ids:
        if lid in landmarks_dict:
            x_norm, y_norm = landmarks_dict[lid]
            x_px, y_px = int(x_norm * w), int(y_norm * h)
            cv2.circle(image_bgr, (x_px, y_px), 5, (0, 255, 0), -1)

    # Draw connecting lines (cyan)
    for (id1, id2) in lines:
        if id1 in landmarks_dict and id2 in landmarks_dict:
            x1n, y1n = landmarks_dict[id1]
            x2n, y2n = landmarks_dict[id2]
            x1, y1 = int(x1n * w), int(y1n * h)
            x2, y2 = int(x2n * w), int(y2n * h)
            cv2.line(image_bgr, (x1, y1), (x2, y2), (255, 255, 0), 2)


def overlay_angle_labels(image_bgr, landmarks_dict, angles, side='left'):
    """
    Places numeric angle text near 'point B' but does NOT draw arcs anymore.
    """
    h, w, _ = image_bgr.shape

    if side == 'left':
        hip_id, knee_id, ankle_id = LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
        shoulder_id, elbow_id, wrist_id = LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST
        ear_id = LEFT_EAR
    else:
        hip_id, knee_id, ankle_id = RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
        shoulder_id, elbow_id, wrist_id = RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST
        ear_id = RIGHT_EAR

    # A-B-C sets for each angle
    angle_map = {
        'knee_angle': (hip_id, knee_id, ankle_id),
        'hip_angle': (shoulder_id, hip_id, knee_id),
        'elbow_angle': (shoulder_id, elbow_id, wrist_id),
        'head_to_shoulder_angle': (ear_id, shoulder_id, hip_id)
    }

    for angle_name, angle_val in angles.items():
        if angle_val is None:
            continue

        if angle_name not in angle_map:
            continue

        a_id, b_id, c_id = angle_map[angle_name]
        if a_id not in landmarks_dict or b_id not in landmarks_dict or c_id not in landmarks_dict:
            continue

        # If you once swapped A<->C for "inside arcs," you can remove that swap now.
        # We'll just keep the numeric angle. Example:
        #
        # if angle_name in ('hip_angle','elbow_angle','head_to_shoulder_angle'):
        #    ax, ay, cx, cy = cx, cy, ax, ay
        #
        # But we remove arcs entirely.

        bx, by = landmarks_dict[b_id]
        bx_px, by_px = int(bx * w), int(by * h)

        # Only place text
        text_str = f"{angle_name}: {angle_val:.1f}°"
        cv2.putText(
            image_bgr,
            text_str,
            (bx_px + 10, by_px - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2,
            cv2.LINE_AA
        )


def run_pipeline(image_dir, visualize=True, default_side='left'):
    valid_ext = ('.jpg', '.jpeg', '.png')
    files = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_ext)]

    for file_name in files:
        path = os.path.join(image_dir, file_name)
        image_bgr = cv2.imread(path)
        if image_bgr is None:
            print(f"[X] Could not open '{file_name}'. Skipping.")
            continue

        # Get landmarks from pose_estimation
        landmarks_dict, annotated_img, feedback_msg = extract_landmarks(image_bgr, visualize=False)
        if landmarks_dict is None:
            print(f"\n=== {file_name} ===\nNo landmarks found.\n-----------\n")
            continue

        # Determine side
        side_used = default_side
        if "CHOSEN SIDE: RIGHT" in feedback_msg.upper():
            side_used = "right"

        # Draw relevant circles/lines (no arcs)
        draw_relevant_landmarks_and_lines(annotated_img, landmarks_dict, side=side_used)

        # Compute angles
        angles = compute_posture_angles(landmarks_dict, side=side_used)

        # Place numeric labels only, no arcs
        overlay_angle_labels(annotated_img, landmarks_dict, angles, side=side_used)

        print(f"\n=== {file_name} ===")
        print("Feedback:", feedback_msg)
        print("Angles:")
        for angle_name, val in angles.items():
            if val is not None:
                print(f"  {angle_name}: {val:.1f}°")
            else:
                print(f"  {angle_name}: N/A")
        print("--------------------------------\n")

        if visualize:
            cv2.imshow("Angles (no arcs)", annotated_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Remove yellow arcs entirely; only numeric angle labels remain."
    )
    parser.add_argument("--dir", type=str, default="../data/processed",
                        help="Folder containing images.")
    parser.add_argument("--side", type=str, default="left",
                        help="Fallback side if coverage not determined automatically.")
    parser.add_argument("--no-visualize", action="store_true",
                        help="If set, skip pop-up windows.")
    args = parser.parse_args()

    run_pipeline(
        image_dir=args.dir,
        visualize=not args.no_visualize,
        default_side=args.side
    )

if __name__ == "__main__":
    main()