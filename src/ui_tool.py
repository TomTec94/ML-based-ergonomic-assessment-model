# ui_tool.py
"""
Example Tkinter-based UI for posture detection, now supporting drag-and-drop loading of images.
Requires 'tkinterdnd2' to be installed (pip install tkinterdnd2).
"""

import os
import cv2
from PIL import Image, ImageTk
# The main difference is that we inherit from 'TkinterDnD.Tk' instead of tk.Tk:
from tkinterdnd2 import TkinterDnD, DND_FILES
import tkinter as tk
from tkinter import scrolledtext

# Import your pipeline steps
from pose_estimation import extract_landmarks
from angle_calculation import compute_posture_angles
from rule_based_model import rule_based_posture_analysis, evaluate_angle


def draw_relevant_landmarks_and_lines(image_bgr, landmarks_dict, side='left'):
    """
    Draw green circles + cyan lines for knee/hip/elbow/head angles.
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
        (hip_id, knee_id),
        (shoulder_id, elbow_id),
        (elbow_id, wrist_id),
        (ear_id, shoulder_id),
        (shoulder_id, hip_id)
    ]

    # Green circles
    for lid in relevant_ids:
        if lid in landmarks_dict:
            x_norm, y_norm = landmarks_dict[lid]
            x_px, y_px = int(x_norm * w), int(y_norm * h)
            cv2.circle(image_bgr, (x_px, y_px), 5, (0, 255, 0), -1)

    # Cyan lines
    for (id1, id2) in lines:
        if id1 in landmarks_dict and id2 in landmarks_dict:
            x1n, y1n = landmarks_dict[id1]
            x2n, y2n = landmarks_dict[id2]
            x1, y1 = int(x1n * w), int(y1n * h)
            x2, y2 = int(x2n * w), int(y2n * h)
            cv2.line(image_bgr, (x1, y1), (x2, y2), (255, 255, 0), 2)


def overlay_color_coded_angles(image_bgr, landmarks_dict, angles_dict, side='left'):
    """
    Color-code angles: green if within ±5°, red if out-of-range.
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
    angle_map = {
        'knee_angle': (hip_id, knee_id, ankle_id),
        'hip_angle': (shoulder_id, hip_id, knee_id),
        'elbow_angle': (shoulder_id, elbow_id, wrist_id),
        'head_to_shoulder_angle': (ear_id, shoulder_id, hip_id)
    }

    for angle_name, angle_val in angles_dict.items():
        if angle_val is None:
            continue

        eval_info = evaluate_angle(angle_name, angle_val)
        color_text = (0, 255, 0) if eval_info['ok'] else (0, 0, 255)

        label_str = label_map.get(angle_name, angle_name)
        text_str = f"{label_str} = {angle_val:.1f}°"

        triple = angle_map.get(angle_name)
        if not triple:
            continue
        _, b_id, _ = triple
        if b_id not in landmarks_dict:
            continue

        bx, by = landmarks_dict[b_id]
        bx_px, by_px = int(bx * w), int(by * h)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thick = 2
        (tw, th), base = cv2.getTextSize(text_str, font, scale, thick)
        base += 4

        rect_left = bx_px + 10
        rect_top = by_px - 10 - th
        rect_right = rect_left + tw
        rect_bottom = rect_top + th + base

        # clamp
        if rect_left < 0:
            rect_left = 0
        if rect_top < 0:
            rect_top = 0
        if rect_right > w:
            rect_right = w
        if rect_bottom > h:
            rect_bottom = h

        overlay = image_bgr.copy()
        cv2.rectangle(overlay, (rect_left, rect_top), (rect_right, rect_bottom), (255, 255, 255), -1)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0, image_bgr)

        text_x = rect_left
        text_y = rect_bottom - base // 2
        cv2.putText(image_bgr, text_str, (text_x, text_y), font, scale, color_text, thick, cv2.LINE_AA)


def format_results_text(base_name, side_used, angles_dict, results, landmarks_dict):
    """
    Build a textual summary of the analysis for the right panel in English.
    """
    landmarks_count = len(landmarks_dict) if landmarks_dict else 0

    lines = []
    lines.append("-----------------------------------")
    lines.append(f"File Name: {base_name}")
    lines.append(f"Number of detected Landmarks: {landmarks_count}")
    lines.append(f"Chosen side: {side_used}\n")

    lines.append("Detected angles:")
    for angle_name, angle_val in angles_dict.items():
        if angle_val is not None:
            lines.append(f"  • {angle_name}: {angle_val:.1f}°")

    score = results.get('score', 0)
    rating = results.get('rating', "N/A")
    lines.append("")
    lines.append(f"SCORE: {score}% ({rating})\n")

    lines.append("Result:")
    evaluations = results.get('evaluations', [])
    for ev in evaluations:
        if ev['ok'] is False:
            lines.append(f"  • {ev['message']}")
            if ev['solutions']:
                for sol in ev['solutions']:
                    lines.append(f"    - {sol}")
        else:
            lines.append(f"  • {ev['message']}")

    lines.append("-----------------------------------\n")
    return "\n".join(lines)


###########################################################
# Drag-and-Drop Integration with 'tkinterdnd2'
###########################################################
from tkinterdnd2 import TkinterDnD, DND_FILES


class ErgoApp(TkinterDnD.Tk):
    """
    A TkinterDnD-based GUI for analyzing posture from side-view images.
    Allows drag-and-drop of an image file onto a drop label.
    """

    def __init__(self):
        super().__init__()
        self.title("ErgoApp: Simple Posture Detection (Drag & Drop)")
        self.geometry("1200x800")

        # Left side: annotated image
        self.frame_left = tk.Frame(self, width=800, height=800, bg="gray")
        self.frame_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right side: text results + drag area
        self.frame_right = tk.Frame(self, width=400, height=800)
        self.frame_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.image_label = tk.Label(self.frame_left, bg="black")
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # Info label at top of the right side
        self.info_label = tk.Label(
            self.frame_right,
            text="Drag an Image File onto the Drop Area below:",
            font=("Arial", 14)
        )
        self.info_label.pack(pady=10)

        # Drop area
        self.drop_label = tk.Label(self.frame_right, text="DROP HERE", bg="#ccc", width=40, height=6, font=("Arial", 16))
        self.drop_label.pack(pady=20, fill=tk.X)

        # Register for file drops
        self.drop_label.drop_target_register(DND_FILES)
        self.drop_label.dnd_bind('<<Drop>>', self.drop_event)

        # Scrolled text for results
        self.scrollbar = tk.Scrollbar(self.frame_right, orient=tk.VERTICAL)
        self.result_text = tk.Text(
            self.frame_right, wrap="word", font=("Arial", 14), yscrollcommand=self.scrollbar.set
        )
        self.scrollbar.config(command=self.result_text.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.annotated_cv2_image = None

    def drop_event(self, event):
        """
        Called when a file is dropped onto self.drop_label.
        'event.data' may contain braces and multiple paths. We'll parse them.
        """
        paths = self.parse_drop_files(event.data)
        if not paths:
            self.log_message("No valid file dropped.")
            return

        # For simplicity, just process the first file if multiple are dropped
        file_path = paths[0]
        self.log_message(f"Dropped file: {file_path}")
        if not os.path.isfile(file_path):
            self.log_message(f"[X] Not a valid file: {file_path}")
            return

        self.load_and_process_image(file_path)

    def parse_drop_files(self, drop_data):
        """
        Convert the raw drop string to a list of file paths.
        For example, on Windows/macOS it might look like:
        '{C:/some path/img1.jpg} {C:/some path/img2.png}'
        """
        raw_files = drop_data.split('}')
        file_list = []
        for raw_f in raw_files:
            f = raw_f.strip().strip('{').strip()
            if f:
                file_list.append(f)
        return file_list

    def load_and_process_image(self, file_path):
        """
        Actually process the image, run posture pipeline, display results & image.
        """
        if not os.path.exists(file_path):
            self.log_message(f"File not found: {file_path}")
            return

        image_bgr = cv2.imread(file_path)
        if image_bgr is None:
            self.log_message(f"Could not open {file_path}")
            return

        # Pose detection
        landmarks_dict, _, feedback_msg = extract_landmarks(image_bgr, visualize=False)
        if landmarks_dict is None:
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert(tk.END, f"No landmarks found in {file_path}\n\n")
            return

        side_used = "LEFT"
        if "RIGHT" in feedback_msg.upper():
            side_used = "RIGHT"

        # Angles
        angles_dict = compute_posture_angles(landmarks_dict, side=side_used)

        # Create annotated copy
        final_annot = image_bgr.copy()
        draw_relevant_landmarks_and_lines(final_annot, landmarks_dict, side=side_used)
        overlay_color_coded_angles(final_annot, landmarks_dict, angles_dict, side=side_used)

        # Rule-based
        results = rule_based_posture_analysis(
            final_annot, angles_dict, side=side_used, landmarks_dict=landmarks_dict
        )

        # Show textual
        base_name = os.path.basename(file_path)
        summary = format_results_text(base_name, side_used, angles_dict, results, landmarks_dict)
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, summary)

        # Display annotated in left panel
        self.annotated_cv2_image = final_annot
        self.display_annotated_image(self.annotated_cv2_image)

    def display_annotated_image(self, cv2_img):
        """
        Convert BGR->RGB->Pillow->resize->display on the left label.
        """
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
        """
        Quick helper to append lines to self.result_text.
        """
        self.result_text.insert(tk.END, msg + "\n")
        self.result_text.see(tk.END)


# If you just want to test this file by itself:
if __name__ == "__main__":
    app = ErgoApp()
    app.mainloop()