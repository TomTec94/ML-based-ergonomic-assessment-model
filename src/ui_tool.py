# BachelorThesis/src/ui_tool.py
import os
import cv2
import numpy as np
import math
from PIL import Image, ImageTk, ImageDraw, ImageFont
import tkinter as tk
import logging
import traceback

from tkinterdnd2 import TkinterDnD, DND_FILES

from pose_estimation import extract_landmarks
from angle_calculation import compute_posture_angles
from rule_based_model import rule_based_posture_analysis, evaluate_angle, ANGLE_CONFIG, update_angle_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def get_scale_factor(image_width, image_height, base_width=800.0, base_height=800.0):
    """
    Computes a uniform scaling factor based on the image dimensions.
    Uses the minimum of width and height ratios.
    """
    return min(image_width / base_width, image_height / base_height)

def draw_relevant_landmarks_and_lines(image_bgr, landmarks_dict, side='left'):
    """
    Draws green circles and cyan lines for the detected landmarks.
    The sizes are scaled based on the image dimensions.
    """
    h, w, _ = image_bgr.shape
    scale = get_scale_factor(w, h)

    if side == 'left':
        hip_id, knee_id, ankle_id = 23, 25, 27
        shoulder_id, elbow_id, wrist_id = 11, 13, 15
        ear_id = 7
    else:
        hip_id, knee_id, ankle_id = 24, 26, 28
        shoulder_id, elbow_id, wrist_id = 12, 14, 16
        ear_id = 8

    circle_radius = max(1, int(5 * scale))
    line_thickness = max(1, int(2 * scale))

    for lid in [hip_id, knee_id, ankle_id, shoulder_id, elbow_id, wrist_id, ear_id]:
        if lid in landmarks_dict:
            x_px = int(landmarks_dict[lid][0] * w)
            y_px = int(landmarks_dict[lid][1] * h)
            cv2.circle(image_bgr, (x_px, y_px), circle_radius, (0, 255, 0), -1)

    for (p1, p2) in [(hip_id, knee_id), (knee_id, ankle_id),
                     (shoulder_id, hip_id), (shoulder_id, elbow_id),
                     (elbow_id, wrist_id), (ear_id, shoulder_id)]:
        if p1 in landmarks_dict and p2 in landmarks_dict:
            x1 = int(landmarks_dict[p1][0] * w)
            y1 = int(landmarks_dict[p1][1] * h)
            x2 = int(landmarks_dict[p2][0] * w)
            y2 = int(landmarks_dict[p2][1] * h)
            cv2.line(image_bgr, (x1, y1), (x2, y2), (255, 255, 0), line_thickness)

def get_arc_angles(vertex, pointA, pointC):
    """
    Computes the start and end angles (in degrees) for an arc drawn at 'vertex'
    that spans from pointA to pointC.
    """
    dxA = pointA[0] - vertex[0]
    dyA = pointA[1] - vertex[1]
    dxC = pointC[0] - vertex[0]
    dyC = pointC[1] - vertex[1]
    angleA = math.degrees(math.atan2(dyA, dxA)) % 360
    angleC = math.degrees(math.atan2(dyC, dxC)) % 360
    diff = (angleC - angleA) % 360
    if diff > 180:
        diff = 360 - diff
        start_angle = angleC
        end_angle = angleA
    else:
        start_angle = angleA
        end_angle = angleC
    return start_angle, end_angle

def overlay_color_coded_angles(image_bgr, landmarks_dict, angles_dict, side='left'):
    """
    Overlays angle labels (with a Unicode degree symbol) and draws a yellow arc
    that spans from one landmark segment to the other.
    All drawing parameters are scaled based on the image dimensions.
    """
    h, w, _ = image_bgr.shape
    scale = get_scale_factor(w, h)

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
        'head_to_shoulder_angle': 'Neck angle'
    }
    angle_map = {
        'knee_angle': (hip_id, knee_id, ankle_id),      # vertex = knee
        'hip_angle': (shoulder_id, hip_id, knee_id),       # vertex = hip
        'elbow_angle': (shoulder_id, elbow_id, wrist_id),  # vertex = elbow
        'head_to_shoulder_angle': (ear_id, shoulder_id, hip_id)  # vertex = shoulder (used as neck)
    }
    offset_map = {
        'knee_angle': (10 * scale, 15 * scale),
        'hip_angle': (10 * scale, -15 * scale),
        'elbow_angle': (10 * scale, 15 * scale),
        'head_to_shoulder_angle': (10 * scale, -15 * scale)
    }

    circle_radius = max(1, int(5 * scale))
    line_thickness = max(1, int(2 * scale))
    arc_radius = max(1, int(20 * scale))
    font_size = max(20, int(20 * scale))

    overlay = image_bgr.copy()
    overlay_pil = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(overlay_pil)
    try:
        font = ImageFont.truetype("arialbd.ttf", font_size)
    except IOError:
        font = ImageFont.truetype("arial.ttf", font_size) if os.path.exists("arial.ttf") else ImageFont.load_default()

    for angle_name, angle_val in angles_dict.items():
        if angle_val is None:
            continue
        eval_info = evaluate_angle(angle_name, angle_val)
        color_text = (0, 255, 0) if eval_info['ok'] else (255, 0, 0)
        text_str = f"{label_map.get(angle_name, angle_name)} = {angle_val:.1f}{chr(176)}"
        triple = angle_map.get(angle_name)
        if not triple:
            continue
        _, vertex_id, _ = triple
        if vertex_id not in landmarks_dict:
            continue
        bx, by = landmarks_dict[vertex_id]
        bx_px, by_px = int(bx * w), int(by * h)
        offset = offset_map.get(angle_name, (10 * scale, 10 * scale))
        pos = (bx_px + offset[0], by_px + offset[1])
        bbox = draw.textbbox((0, 0), text_str, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.rectangle([pos, (pos[0] + text_width + 4, pos[1] + text_height + 4)], fill="white")
        draw.text((pos[0] + 2, pos[1] + 2), text_str, font=font, fill=color_text)

        pointA = landmarks_dict.get(triple[0])
        pointC = landmarks_dict.get(triple[2])
        if pointA is None or pointC is None:
            continue
        start_angle, end_angle = get_arc_angles((bx, by), pointA, pointC)
        cv2.ellipse(overlay, (bx_px, by_px), (arc_radius, arc_radius), 0, start_angle, end_angle, (0, 255, 255),
                    line_thickness)

    overlay_cv2 = cv2.cvtColor(np.array(overlay_pil), cv2.COLOR_RGB2BGR)
    cv2.addWeighted(overlay_cv2, 0.8, overlay, 0.2, 0, image_bgr)

def format_results_text(base_name, side_used, angles_dict, results, landmarks_dict):
    """
    Builds a textual summary of the analysis.
    Appends the overall assessment at the bottom.
    """
    landmarks_count = len(landmarks_dict) if landmarks_dict else 0
    lines = []
    lines.append("-----------------------------------")
    lines.append(f"File Name: {base_name}")
    lines.append(f"Detected Landmarks: {landmarks_count}")
    lines.append(f"Chosen Side: {side_used}\n")
    lines.append("Assessment Results:")
    for ev in results.get('evaluations', []):
        lines.append(f"  - {ev['message']}")
        if not ev['ok'] and ev.get('solutions'):
            for sol in ev['solutions']:
                lines.append(f"      * {sol}")
    if 'overall' in results:
        lines.append("")
        lines.append(results['overall'])
    lines.append("-----------------------------------\n")
    return "\n".join(lines)

class ErgoApp(TkinterDnD.Tk):
    """
    A TkinterDnD-based GUI for posture detection.
    """

    def __init__(self):
        super().__init__()
        self.title("ErgoApp: Posture Detection")
        self.geometry("1200x900")

        # Create right frame with two subframes: one for threshold adjustments and one for drop/results.
        self.frame_right = tk.Frame(self, width=400)
        self.frame_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Adjustment frame for threshold inputs and side selection.
        self.adjustment_frame = tk.Frame(self.frame_right, bd=2, relief=tk.GROOVE)
        self.adjustment_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self._create_adjustment_form()

        # Drop area frame.
        self.drop_frame = tk.Frame(self.frame_right)
        self.drop_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.drop_label = tk.Label(self.drop_frame, text="DROP HERE", bg="#ccc", width=40, height=3, font=("Arial", 16))
        self.drop_label.pack(pady=5, fill=tk.X)
        self.drop_label.drop_target_register(DND_FILES)
        self.drop_label.dnd_bind('<<Drop>>', self.drop_event)

        # Text result frame.
        self.result_frame = tk.Frame(self.frame_right)
        self.result_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.scrollbar = tk.Scrollbar(self.result_frame, orient=tk.VERTICAL)
        self.result_text = tk.Text(self.result_frame, wrap=tk.WORD, font=("Arial", 14),
                                   yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.result_text.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Left frame for annotated image.
        self.frame_left = tk.Frame(self, width=800, height=900, bg="gray")
        self.frame_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.image_label = tk.Label(self.frame_left, bg="black")
        self.image_label.pack(fill=tk.BOTH, expand=True)

        self.annotated_cv2_image = None
        self.current_file_path = None
        self.selected_side = None  # Will store the side after clicking "Apply"

    def _create_adjustment_form(self):
        """
        Creates the input fields for side selection and threshold adjustments.
        """
        # Dropdown for selecting the side from which the image was taken.
        side_label = tk.Label(self.adjustment_frame, text="Select Image Side:", font=("Arial", 12))
        side_label.grid(row=0, column=0, columnspan=2, pady=(2, 10), sticky="w")
        self.side_var = tk.StringVar()
        self.side_dropdown = tk.OptionMenu(self.adjustment_frame, self.side_var, "Left", "Right")
        self.side_dropdown.config(font=("Arial", 12))
        self.side_dropdown.grid(row=0, column=2, columnspan=3, padx=5, pady=2, sticky="w")

        # Title label for threshold adjustments.
        title_label = tk.Label(self.adjustment_frame, text="Threshold Adjustments", font=("Arial", 14, "bold"))
        title_label.grid(row=1, column=0, columnspan=5, pady=(2, 10))

        self.angle_keys = ['knee_angle', 'hip_angle', 'elbow_angle', 'head_to_shoulder_angle']
        self.threshold_entries = {}

        # Threshold entries start at row 2.
        for i, angle_key in enumerate(self.angle_keys, start=2):
            display_name = "Neck angle" if angle_key == 'head_to_shoulder_angle' else angle_key.replace("_", " ").title()
            label = tk.Label(self.adjustment_frame, text=display_name + ":", font=("Arial", 12))
            label.grid(row=i, column=0, padx=5, pady=2, sticky=tk.W)

            target_val = ANGLE_CONFIG[angle_key]['target']
            target_entry = tk.Entry(self.adjustment_frame, width=5, font=("Arial", 12))
            target_entry.insert(0, str(target_val))
            target_entry.grid(row=i, column=1, padx=5, pady=2)
            target_label = tk.Label(self.adjustment_frame, text="Target", font=("Arial", 10))
            target_label.grid(row=i, column=2, padx=2, pady=2, sticky=tk.W)

            tol_val = ANGLE_CONFIG[angle_key]['tolerance']
            tol_entry = tk.Entry(self.adjustment_frame, width=5, font=("Arial", 12))
            tol_entry.insert(0, str(tol_val))
            tol_entry.grid(row=i, column=3, padx=5, pady=2)
            tol_label = tk.Label(self.adjustment_frame, text="Tolerance", font=("Arial", 10))
            tol_label.grid(row=i, column=4, padx=2, pady=2, sticky=tk.W)

            self.threshold_entries[angle_key] = {"target": target_entry, "tolerance": tol_entry}

        apply_btn = tk.Button(self.adjustment_frame, text="Apply", font=("Arial", 12, "bold"),
                              command=self.apply_thresholds)
        apply_btn.grid(row=len(self.angle_keys) + 2, column=0, columnspan=5, pady=10)

    def apply_thresholds(self):
        """
        Reads values from the adjustment form, updates the ANGLE_CONFIG for the current session,
        and updates the side selection based on the dropdown. Then, reprocesses the current image (if any).
        """
        for angle_key in self.angle_keys:
            try:
                new_target = float(self.threshold_entries[angle_key]["target"].get())
                new_tolerance = float(self.threshold_entries[angle_key]["tolerance"].get())
                update_angle_config(angle_key, new_target=new_target, new_tolerance=new_tolerance)
                self.log_message(f"Updated {angle_key}: Target={new_target}, Tolerance={new_tolerance}")
            except ValueError:
                self.log_message(f"Invalid input for {angle_key}. Please enter numeric values.")

        selected_side = self.side_var.get()
        if not selected_side:
            self.log_message("Error: No side selected. Please choose 'Left' or 'Right' from the dropdown and click Apply.")
            return
        self.selected_side = selected_side.upper()
        self.log_message("Side selection updated: " + self.selected_side)

        if self.current_file_path:
            self.log_message("Reprocessing current image with new thresholds and side selection...")
            self.load_and_process_image(self.current_file_path)

    def drop_event(self, event):
        paths = self.parse_drop_files(event.data)
        if not paths:
            self.log_message("No valid file dropped.")
            return
        file_path = paths[0]
        self.current_file_path = file_path
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
        self.log_message(f"Processing file: {file_path}")
        if not os.path.exists(file_path):
            self.log_message(f"File not found: {file_path}")
            return

        image_bgr = cv2.imread(file_path)
        if image_bgr is None:
            self.log_message(f"OpenCV could not open the image: {file_path}. Attempting to open with PIL...")
            try:
                pil_image = Image.open(file_path)
                image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                self.log_message("Image loaded via PIL.")
            except Exception as e:
                self.log_message(f"Failed to open image via PIL: {e}")
                return
        else:
            self.log_message("Image loaded successfully via OpenCV.")

        MAX_SIZE = (800, 800)
        h, w, _ = image_bgr.shape
        if w > MAX_SIZE[0] or h > MAX_SIZE[1]:
            scale_factor = min(MAX_SIZE[0] / w, MAX_SIZE[1] / h)
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            image_bgr = cv2.resize(image_bgr, (new_w, new_h))
            self.log_message(f"Image resized to {new_w}x{new_h} for processing.")

        try:
            self.log_message("Running pose detection...")
            landmarks_dict, _, feedback_msg = extract_landmarks(image_bgr, visualize=False)
            if landmarks_dict is None:
                self.log_message("No landmarks found.")
                self.result_text.delete("1.0", tk.END)
                self.result_text.insert("1.0", "No landmarks found.\n")
                return
            self.log_message("Landmarks extracted successfully.")

            if not self.selected_side:
                self.log_message("Error: No side has been set. Please select 'Left' or 'Right' and click Apply before processing an image.")
                return
            side_used = self.selected_side
            self.log_message("Chosen side: " + side_used)
            angles_dict = compute_posture_angles(landmarks_dict, side=side_used.lower())
            self.log_message("Angles computed: " + str(angles_dict))
            final_annot = image_bgr.copy()
            draw_relevant_landmarks_and_lines(final_annot, landmarks_dict, side=side_used.lower())
            overlay_color_coded_angles(final_annot, landmarks_dict, angles_dict, side=side_used.lower())
            self.log_message("Overlay drawn.")
            results = rule_based_posture_analysis(final_annot, angles_dict, side=side_used.lower(), landmarks_dict=landmarks_dict)
            self.log_message("Results computed: " + str(results))
            base_name = os.path.basename(file_path)
            summary = format_results_text(base_name, side_used, angles_dict, results, landmarks_dict)
            self.log_message("Summary formatted.")
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert("1.0", summary)
            self.annotated_cv2_image = final_annot
            self.display_annotated_image(self.annotated_cv2_image)
        except Exception as e:
            self.log_message("Error during processing: " + str(e))
            self.log_message(traceback.format_exc())

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