# angle_calculation.py
import math
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def compute_2d_angle(a, b, c):
    """
    Returns the angle at point b (in degrees) given points a, b, c.
    """
    if not a or not b or not c:
        logging.warning("Invalid input to compute_2d_angle.")
        return None

    ax, ay = a
    bx, by = b
    cx, cy = c

    # Vectors BA and BC
    ba = (ax - bx, ay - by)
    bc = (cx - bx, cy - by)

    dot_prod = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

    if mag_ba == 0 or mag_bc == 0:
        logging.warning("Zero-length vector encountered in compute_2d_angle.")
        return None

    cos_angle = dot_prod / (mag_ba * mag_bc)
    cos_angle = max(min(cos_angle, 1.0), -1.0)
    angle_rad = math.acos(cos_angle)
    return math.degrees(angle_rad)

# Landmark IDs for left side
LEFT_HIP, LEFT_KNEE, LEFT_ANKLE = 23, 25, 27
LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST = 11, 13, 15
LEFT_EAR = 7

# Landmark IDs for right side
RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE = 24, 26, 28
RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST = 12, 14, 16
RIGHT_EAR = 8

def compute_posture_angles(landmarks_dict, side='left'):
    """
    Computes key posture angles:
      - Knee angle (hip–knee–ankle)
      - Hip angle (shoulder–hip–knee)
      - Elbow angle (shoulder–elbow–wrist)
      - Head-to-shoulder angle (ear–shoulder–hip)
    Returns a dictionary with the measured angles.
    """
    angles = {
        'knee_angle': None,
        'hip_angle': None,
        'elbow_angle': None,
        'head_to_shoulder_angle': None
    }

    if side == 'left':
        hip_id, knee_id, ankle_id = LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
        shoulder_id, elbow_id, wrist_id = LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST
        ear_id = LEFT_EAR
    else:
        hip_id, knee_id, ankle_id = RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
        shoulder_id, elbow_id, wrist_id = RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST
        ear_id = RIGHT_EAR

    if hip_id in landmarks_dict and knee_id in landmarks_dict and ankle_id in landmarks_dict:
        a = landmarks_dict[hip_id]
        b = landmarks_dict[knee_id]
        c = landmarks_dict[ankle_id]
        angles['knee_angle'] = compute_2d_angle(a, b, c)

    if shoulder_id in landmarks_dict and hip_id in landmarks_dict and knee_id in landmarks_dict:
        a = landmarks_dict[shoulder_id]
        b = landmarks_dict[hip_id]
        c = landmarks_dict[knee_id]
        angles['hip_angle'] = compute_2d_angle(a, b, c)

    if shoulder_id in landmarks_dict and elbow_id in landmarks_dict and wrist_id in landmarks_dict:
        a = landmarks_dict[shoulder_id]
        b = landmarks_dict[elbow_id]
        c = landmarks_dict[wrist_id]
        angles['elbow_angle'] = compute_2d_angle(a, b, c)

    if ear_id in landmarks_dict and shoulder_id in landmarks_dict and hip_id in landmarks_dict:
        a = landmarks_dict[ear_id]
        b = landmarks_dict[shoulder_id]
        c = landmarks_dict[hip_id]
        angles['head_to_shoulder_angle'] = compute_2d_angle(a, b, c)

    return angles