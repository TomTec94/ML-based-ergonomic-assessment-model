# ErgoApp: Posture Detection

ErgoApp is a GUI-based tool for assessing workstation posture. It uses MediaPipe for landmark detection, OpenCV for image processing, and a rule-based model to compute key joint angles (knee, hip, elbow, and neck). The overall posture is classified as "Ergonomic", "Mostly ergonomic", or "Non ergonomic" based on user-adjustable thresholds.

## Features

- **Landmark Detection:** Uses MediaPipe to detect essential body landmarks.
- **Angle Calculation:** Computes joint angles via vector math.
- **Threshold-Based Assessment:** Compares measured angles against default thresholds (Knee: 90°±10°, Hip: 98°±8°, Elbow: 95°±5°, Neck: 160°±10°) to provide an overall ergonomic evaluation.
- **GUI Interface:** Simple drag-and-drop functionality with real-time visual and textual feedback.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone git@github.com:TomTec94/ML-based-ergonomic-assessment-model.git
   
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   
3. **Run the Application:**
   ```bash
   python main.py
   
## Usage
	•	Load Image: Drag and drop a side-view image (JPG/JPEG/PNG) into the GUI.
	•	Adjust Thresholds: Use the provided form to modify target angles and tolerances.
	•	Choose the side of the shown person via the drop down menu
	•	View Results: The application displays annotated images and a text summary with an overall assessment.

## Just run the preprocessing piplenine
    python src/data_preprocessing.py data/raw_images data/processed

## Project Structure
   ```bash
      BachelorThesis/
   ├── data/            # Raw and processed images
   ├── src/             # Source code (main.py, ui_tool.py, pose_estimation.py, etc.)
   ├── requirements.txt
   └── README.md
```
## Dependencies	
    •	MediaPipe
	•	OpenCV
	•	Pillow
	•	TkinterDnD2
