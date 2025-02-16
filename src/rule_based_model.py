# BachelorThesis/src/rule_based_model.py
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Updated configuration dictionary for angle targets and tolerances.
ANGLE_CONFIG = {
    'knee_angle': {
        'description': "Knee angle",
        'target': 90,       # Target angle in degrees
        'tolerance': 10,    # Allowed deviation in degrees
        'solutions': [
            "Lower or raise your seat so that the knee is around 90°."
        ]
    },
    'hip_angle': {
        'description': "Hip angle",
        'target': 98,
        'tolerance': 8,
        'solutions': [
            "Adjust seat depth or desk height to achieve approximately 98° at the hips."
        ]
    },
    'elbow_angle': {
        'description': "Elbow angle",
        'target': 95,
        'tolerance': 5,
        'solutions': [
            "Adjust the armrest height so that your elbows are about 95°."
        ]
    },
    'head_to_shoulder_angle': {
        'description': "Head to Shoulder angle",
        'target': 160,
        'tolerance': 10,
        'solutions': [
            "Raise or lower your monitor to reduce neck bending.",
            "Keep your head upright; adjust the screen distance."
        ]
    }
}

def update_angle_config(angle_name, new_target=None, new_tolerance=None):
    """
    Dynamically updates the target and/or tolerance for a given angle.
    """
    if angle_name not in ANGLE_CONFIG:
        logging.warning(f"'{angle_name}' not found in ANGLE_CONFIG.")
        return
    if new_target is not None:
        ANGLE_CONFIG[angle_name]['target'] = new_target
    if new_tolerance is not None:
        ANGLE_CONFIG[angle_name]['tolerance'] = new_tolerance

def evaluate_angle(angle_name, angle_value):
    """
    Checks whether angle_value is within the acceptable range for angle_name.
    Returns a dictionary with evaluation details.
    """
    rec = ANGLE_CONFIG.get(angle_name)
    if not rec:
        return {
            'angle_name': angle_name,
            'ok': None,
            'diff': None,
            'message': f"No recommended config found for {angle_name}.",
            'solutions': []
        }

    target = rec['target']
    tolerance = rec['tolerance']
    diff = abs(angle_value - target)
    in_tolerance = diff <= tolerance

    if in_tolerance:
        return {
            'angle_name': angle_name,
            'ok': True,
            'diff': diff,
            'message': f"{rec['description']} is good ({angle_value:.1f}°).",
            'solutions': []
        }
    else:
        msg = (f"{rec['description']} is out of range by {diff:.1f}° "
               f"(measured {angle_value:.1f}°, target ~{target}° ±{tolerance}).")
        return {
            'angle_name': angle_name,
            'ok': False,
            'diff': diff,
            'message': msg,
            'solutions': rec['solutions']
        }

def rule_based_posture_analysis(image_bgr, angles_dict, side='left', landmarks_dict=None):
    """
    Evaluates the measured angles and returns evaluation details.
    Computes an overall assessment as follows:
      - "Ergonomic" if no angle is out of range.
      - "Mostly ergonomic" if exactly one angle is out of range.
      - "Non ergonomic" if two or more angles are out of range.

    Returns a dictionary with:
      - evaluations: a list of evaluation dictionaries (one per angle)
      - messages: a list of messages (each evaluation message, plus any solution suggestions)
      - overall: the overall assessment message.
    """
    evaluations = []
    for angle_name, angle_val in angles_dict.items():
        if angle_val is not None:
            result = evaluate_angle(angle_name, angle_val)
            evaluations.append(result)

    messages = []
    out_of_range_count = 0
    for ev in evaluations:
        messages.append(ev['message'])
        if not ev['ok']:
            out_of_range_count += 1
            for sol in ev['solutions']:
                messages.append(f"Try: {sol}")

    if out_of_range_count == 0:
        overall = "Overall assessment: Ergonomic"
    elif out_of_range_count == 1:
        overall = "Overall assessment: Mostly ergonomic"
    else:
        overall = "Overall assessment: Non ergonomic"

    messages.append(overall)

    return {
        'evaluations': evaluations,
        'messages': messages,
        'overall': overall
    }