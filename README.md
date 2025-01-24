
# Posture Detection Project

This repository provides a rule-based approach to ergonomic posture detection using [MediaPipe](https://github.com/google/mediapipe) for landmark extraction. The system computes joint angles (e.g., hip-knee-ankle) in side-view images and classifies them as either *ergonomic* or *non-ergonomic* based on predefined thresholds.

---

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Setup](#setup)
4. [Usage](#usage)
5. [Project Structure](#project-structure)
6. [How It Works](#how-it-works)

---

## Overview
Many organizations rely on manual assessments (e.g., protractors, checklists) to verify ergonomics at computer workstations. This project automates part of the process:
- **MediaPipe** is used to locate body landmarks in side-view images (hips, knees, ankles, etc.).
- **Angles** (such as knee angle) are computed from the extracted coordinates.
- **Predefined thresholds** determine whether each angle is acceptable or non-ergonomic.

The current version serves as a proof of concept that can be extended or integrated into larger workflows.

---

## Prerequisites
1. **Python 3.8+** installed on your system (64-bit recommended).
2. A virtual environment or conda environment (optional but recommended).
3. Installed dependencies listed in `requirements.txt` (e.g., `mediapipe`, `opencv-python`, `numpy`).

---

## Setup
1. **Clone or Download** this repository:
   ```bash
   git clone https://github.com/your-username/my_posture_project.git
   cd my_posture_project
   ```

## Usage
```bash
	1.	Prepare Side-View Images
	•	Place your side-view images in a directory such as data/test_images/.
	•	Ensure each image clearly shows the user’s hips, knees, and ankles.
	2.	Run the Pipeline
python src/main.py --dir data/test_images

	•	By default, each image in data/test_images/ is processed.
	•	Change the --dir argument to scan a different folder.

	3.	Observe Output
	•	The console prints each image’s classification (ergonomic vs. non-ergonomic).
	•	If consistent misclassifications occur, you can adjust thresholds in src/rule_based_model.py.
 
```
 
## Project Structure
```bash

my_posture_project/
├── data/
│   ├── raw/                   # Optional: original images
│   ├── processed/             # Optional: resized or preprocessed images
│   └── test_images/           # Images to be analyzed by main.py
├── notebooks/
│   └── EDA.ipynb              # Exploratory notebook for landmark detection checks
├── src/
│   ├── data_preprocessing.py  # Resizing or reformatting images
│   ├── pose_estimation.py     # MediaPipe-based landmark extraction
│   ├── angle_calculation.py   # Functions for computing angles
│   ├── rule_based_model.py    # Threshold-based classification logic
│   ├── evaluation.py          # Internal evaluation or consistency checks
│   └── main.py                # CLI entry point to run the pipeline
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```
## How It Works
	1.	Landmark Extraction (MediaPipe)
	•	pose_estimation.py loads each image, converts to RGB, and applies MediaPipe Pose in static mode.
	•	Returns a dictionary of (x, y) coordinates for each recognized landmark (e.g., left knee = ID 25).
	2.	Angle Calculation
	•	angle_calculation.py uses vector math to compute angles between three points.
	•	For example, hip–knee–ankle might be ~90° for a typical seated position.
	3.	Threshold-Based Classification
	•	rule_based_model.py checks computed angles against recommended ergonomic ranges (e.g., 80–100° for the knee).
	•	If any angle is out of range, it labels the posture non-ergonomic; otherwise, ergonomic.
	4.	Results & Adjustments
	•	The console displays classification results for each image.
	•	If needed, refine angle thresholds to align with different ergonomic guidelines or user preferences.