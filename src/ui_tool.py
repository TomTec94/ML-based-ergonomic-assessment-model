# ui_tool.py
import os
import cv2
import numpy as np
import math
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import scrolledtext
import logging

from tkinterdnd2 import TkinterDnD, DND_FILES

from pose_estimation import extract_landmarks
from angle_calculation import compute_posture_angles
from rule_based_model import rule_based_posture_analysis, evaluate_angle

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def draw_relevant_landmarks_and_lines(image_bgr, landmarks_dict, side='left'):
    """
    Draw green circles and cyan lines for key landmarks.
    """
    h, w, _ = image_bgr.shape

    if side == 'left':
        hip_id, knee_id, ankle_id = 23, 25, 27
        shoulder_id, elbow_id, wrist_id = 11, 13, 15
        ear_id = 7
    else:
        hip_id, knee_id, ankle_id = 24, 26, 28
        shoulder_id, elbow_id, wrist_id = 12, 14, 16
        ear_id = 8

    relevant_ids = [hip_id, knee_id, ankle_id, shoulder_id, elbow_id, wrist_id, ear_id]
    lines = [
        (hip_id, knee_id),
        (knee_id, ankle_id),
        (shoulder_id, hip_id),
        (shoulder_id, elbow_id),
        (elbow_id, wrist_id),
        (ear_id, shoulder_id)
    ]

    for lid in relevant_ids:
        if lid in landmarks_dict:
            x_norm, y_norm = landmarks_dict[lid]
            x_px, y_px = int(x_norm * w), int(y_norm * h)
            cv2.circle(image_bgr, (x_px, y_px), 5, (0, 255, 0), -1)

    for (id1, id2) in lines:
        if id1 in landmarks_dict and id2 in landmarks_dict:
            x1n, y1n = landmarks_dict[id1]
            x2n, y2n = landmarks_dict[id2]
            x1, y1 = int(x1n * w), int(y1n * h)
            x2, y2 = int(x2n * w), int(y2n * h)
            cv2.line(image_bgr, (x1, y1), (x2, y2), (255, 255, 0), 2)


def overlay_color_coded_angles(image_bgr, landmarks_dict, angles_dict, side='left'):
    """
    Overlays text labels for the measured angles (knee, hip, elbow, head-to-shoulder)
    on the image using a single overlay so that all labels remain visible.

    Additionally, for each angle an arc (drawn in yellow) is placed between the two landmark segments
    (the "Schenkel") to indicate the angle. The text now also uses the Unicode degree symbol (\u00B0).
    """
    h, w, _ = image_bgr.shape

    if side == 'left':
        hip_id, knee_id, ankle_id = 23, 25, 27
        shoulder_id, elbow_id, wrist_id = 11, 13, 15
        ear_id = 7
    else:
        hip_id, knee_id, ankle_id = 24, 26, 28
        shoulder_id, elbow_id, wrist_id = 12, 14, 16
        ear_id = 8

    label_map = {
        'knee_angle': 'Knee angle',
        'hip_angle': 'Hip angle',
        'elbow_angle': 'Elbow angle',
        'head_to_shoulder_angle': 'Head-to-Shoulder angle'
    }
    # The triple for each angle: (pointA, vertex, pointC)
    angle_map = {
        'knee_angle': (hip_id, knee_id, ankle_id),  # Vertex = knee
        'hip_angle': (shoulder_id, hip_id, knee_id),  # Vertex = hip
        'elbow_angle': (shoulder_id, elbow_id, wrist_id),  # Vertex = elbow
        'head_to_shoulder_angle': (ear_id, shoulder_id, hip_id)  # Vertex = shoulder
    }
    # Custom offsets for text placement near the vertex
    offset_map = {
        'knee_angle': (10, 15),
        'hip_angle': (10, -15),
        'elbow_angle': (10, 15),
        'head_to_shoulder_angle': (10, -15)
    }

    # Create one overlay image for all annotations.
    overlay = image_bgr.copy()

    for angle_name, angle_val in angles_dict.items():
        if angle_val is None:
            continue

        eval_info = evaluate_angle(angle_name, angle_val)
        color_text = (0, 255, 0) if eval_info['ok'] else (0, 0, 255)

        label_str = label_map.get(angle_name, angle_name)
        # Use Unicode degree symbol \u00B0 so it displays correctly
        text_str = f"{label_str} = {angle_val:.1f}\u00B0"

        triple = angle_map.get(angle_name)
        if not triple:
            continue
        # Use the vertex from the triple for text placement.
        _, vertex_id, _ = triple
        if vertex_id not in landmarks_dict:
            continue

        bx, by = landmarks_dict[vertex_id]
        bx_px, by_px = int(bx * w), int(by * h)
        offset = offset_map.get(angle_name, (10, 10))
        offset_x, offset_y = offset

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2
        (tw, th), base = cv2.getTextSize(text_str, font, scale, thickness)
        base += 4

        rect_left = bx_px + offset_x
        rect_top = by_px + offset_y - th
        rect_right = rect_left + tw
        rect_bottom = rect_top + th + base

        rect_left = max(rect_left, 0)
        rect_top = max(rect_top, 0)
        rect_right = min(rect_right, w)
        rect_bottom = min(rect_bottom, h)

        cv2.rectangle(overlay, (rect_left, rect_top), (rect_right, rect_bottom), (255, 255, 255), -1)
        text_x = rect_left
        text_y = rect_bottom - base // 2
        cv2.putText(overlay, text_str, (text_x, text_y), font, scale, color_text, thickness, cv2.LINE_AA)

        # --- Draw the arc indicating the angle ---
        # Get the two endpoints from the triple:
        pointA = landmarks_dict.get(triple[0])
        pointC = landmarks_dict.get(triple[2])
        if pointA is None or pointC is None:
            continue
        ax_px, ay_px = int(pointA[0] * w), int(pointA[1] * h)
        cx_px, cy_px = int(pointC[0] * w), int(pointC[1] * h)

        # Define a helper function to compute arc start and end angles.
        def get_arc_angles(vertex, pointA, pointC):
            dxA = pointA[0] - vertex[0]
            dyA = pointA[1] - vertex[1]
            dxC = pointC[0] - vertex[0]
            dyC = pointC[1] - vertex[1]
            angleA = math.degrees(math.atan2(dyA, dxA))
            angleC = math.degrees(math.atan2(dyC, dxC))
            if angleA < 0:
                angleA += 360
            if angleC < 0:
                angleC += 360
            diff = abs(angleA - angleC)
            if diff > 180:
                diff = 360 - diff
            start = min(angleA, angleC)
            return start, start + diff

        start_angle, end_angle = get_arc_angles((bx, by), pointA, pointC)
        r = 20  # fixed radius for the arc
        cv2.ellipse(overlay, (bx_px, by_px), (r, r), 0, start_angle, end_angle, (0, 255, 255), 2)

    alpha = 0.5
    cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0, image_bgr)


def format_results_text(base_name, side_used, angles_dict, results, landmarks_dict):
    """
    Builds a textual summary of the analysis.
    """
    landmarks_count = len(landmarks_dict) if landmarks_dict else 0

    lines = []
    lines.append("-----------------------------------")
    lines.append(f"File Name: {base_name}")
    lines.append(f"Detected Landmarks: {landmarks_count}")
    lines.append(f"Chosen Side: {side_used}\n")
    lines.append("Detected Angles:")
    for angle_name, angle_val in angles_dict.items():
        if angle_val is not None:
            lines.append(f"  - {angle_name}: {angle_val:.1f}\u00B0")
    score = results.get('score', 0)
    rating = results.get('rating', "N/A")
    lines.append("")
    lines.append(f"SCORE: {score}% ({rating})\n")
    lines.append("Result Details:")
    evaluations = results.get('evaluations', [])
    for ev in evaluations:
        lines.append(f"  - {ev['message']}")
        if ev['ok'] is False and ev['solutions']:
            for sol in ev['solutions']:
                lines.append(f"      * {sol}")
    lines.append("-----------------------------------\n")
    return "\n".join(lines)


class ErgoApp(TkinterDnD.Tk):
    """
    A TkinterDnD-based GUI for posture detection.
    """

    def __init__(self):
        super().__init__()
        self.title("ErgoApp: Posture Detection")
        self.geometry("1200x800")

        # Left side: annotated image
        self.frame_left = tk.Frame(self, width=800, height=800, bg="gray")
        self.frame_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right side: text results and drop area
        self.frame_right = tk.Frame(self, width=400, height=800)
        self.frame_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.image_label = tk.Label(self.frame_left, bg="black")
        self.image_label.pack(fill=tk.BOTH, expand=True)

        self.info_label = tk.Label(self.frame_right, text="Drag an Image File onto the Drop Area:", font=("Arial", 14))
        self.info_label.pack(pady=10)

        self.drop_label = tk.Label(self.frame_right, text="DROP HERE", bg="#ccc", width=40, height=6,
                                   font=("Arial", 16))
        self.drop_label.pack(pady=20, fill=tk.X)

        self.drop_label.drop_target_register(DND_FILES)
        self.drop_label.dnd_bind('<<Drop>>', self.drop_event)

        self.scrollbar = tk.Scrollbar(self.frame_right, orient=tk.VERTICAL)
        self.result_text = tk.Text(self.frame_right, wrap="word", font=("Arial", 14), yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.result_text.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.annotated_cv2_image = None

    def drop_event(self, event):
        paths = self.parse_drop_files(event.data)
        if not paths:
            self.log_message("No valid file dropped.")
            return
        file_path = paths[0]
        self.log_message(f"Dropped file: {file_path}")
        if not os.path.isfile(file_path):
            self.log_message(f"Not a valid file: {file_path}")
            return
        self.load_and_process_image(file_path)

    def parse_drop_files(self, drop_data):
        raw_files = drop_data.split('}')
        file_list = []
        for raw_f in raw_files:
            f = raw_f.strip().strip('{').strip()
            if f:
                file_list.append(f)
        return file_list

    def load_and_process_image(self, file_path):
        if not os.path.exists(file_path):
            self.log_message(f"File not found: {file_path}")
            return

        # Try loading the image with OpenCV
        image_bgr = cv2.imread(file_path)
        if image_bgr is None:
            self.log_message(f"OpenCV could not open the image: {file_path}. Attempting to open with PIL...")
            try:
                pil_image = Image.open(file_path)
                image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            except Exception as e:
                self.log_message(f"Failed to open image via PIL: {e}")
                return

        # Run pose detection
        landmarks_dict, _, feedback_msg = extract_landmarks(image_bgr, visualize=False)
        if landmarks_dict is None:
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert(tk.END, f"No landmarks found in {file_path}\n")
            return

        side_used = "LEFT"
        if "RIGHT" in feedback_msg.upper():
            side_used = "RIGHT"

        angles_dict = compute_posture_angles(landmarks_dict, side=side_used.lower())

        # Create an annotated copy of the image
        final_annot = image_bgr.copy()
        draw_relevant_landmarks_and_lines(final_annot, landmarks_dict, side=side_used.lower())
        overlay_color_coded_angles(final_annot, landmarks_dict, angles_dict, side=side_used.lower())

        results = rule_based_posture_analysis(final_annot, angles_dict, side=side_used.lower(),
                                              landmarks_dict=landmarks_dict)

        base_name = os.path.basename(file_path)
        summary = format_results_text(base_name, side_used, angles_dict, results, landmarks_dict)
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, summary)

        self.annotated_cv2_image = final_annot
        self.display_annotated_image(self.annotated_cv2_image)

    def display_annotated_image(self, cv2_img):
        rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        MAX_SIZE = (800, 800)
        try:
            pil_img.thumbnail(MAX_SIZE, Image.Resampling.LANCZOS)
        except AttributeError:
            pil_img.thumbnail(MAX_SIZE, Image.ANTIALIAS)
        tk_img = ImageTk.PhotoImage(pil_img)
        self.image_label.configure(image=tk_img)
        self.image_label.image = tk_img

    def log_message(self, msg):
        self.result_text.insert(tk.END, msg + "\n")
        self.result_text.see(tk.END)


if __name__ == "__main__":
    app = ErgoApp()
    app.mainloop()