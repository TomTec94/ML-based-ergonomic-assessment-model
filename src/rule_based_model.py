# rule_based_model.py
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Configuration dictionary for angle targets and tolerances.
ANGLE_CONFIG = {
    'knee_angle': {
        'description': "Knee angle",
        'target': 100,       # Target angle in degrees
        'tolerance': 5,      # Allowed deviation in degrees
        'solutions': [
            "Lower or raise your seat so that the knee is around 100°."
        ]
    },
    'hip_angle': {
        'description': "Hip angle",
        'target': 100,
        'tolerance': 5,
        'solutions': [
            "Adjust seat depth or desk height to achieve approximately 100° at the hips."
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
        'tolerance': 5,
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

def compute_overall_score(evaluations):
    """
    Computes an overall score based on the number of angles out-of-range.
    Returns a tuple (score, rating_str).
    """
    total = len(evaluations)
    if total == 0:
        return 0, "No angles measured"

    num_out_of_range = sum(1 for ev in evaluations if ev['ok'] is False)

    if num_out_of_range == 0:
        score = 100
    elif num_out_of_range == 1:
        score = 80
    elif num_out_of_range == 2:
        score = 60
    elif num_out_of_range == 3:
        score = 40
    else:
        score = 20

    if score >= 80:
        rating = "Great"
    elif score >= 60:
        rating = "OK"
    else:
        rating = "Poor"

    return score, rating

def rule_based_posture_analysis(image_bgr, angles_dict, side='left', landmarks_dict=None):
    """
    Evaluates the measured angles and computes an overall posture score.
    Returns a dictionary containing detailed evaluations and the overall score.
    """
    evaluations = []
    for angle_name, angle_val in angles_dict.items():
        if angle_val is not None:
            result = evaluate_angle(angle_name, angle_val)
            evaluations.append(result)

    score, rating = compute_overall_score(evaluations)
    messages = []
    for ev in evaluations:
        messages.append(ev['message'])
        if ev['ok'] is False:
            for sol in ev['solutions']:
                messages.append(f"Try: {sol}")

    summary = f"Overall Score: {score}%, rated {rating}"
    messages.append(summary)

    return {
        'evaluations': evaluations,
        'score': score,
        'rating': rating,
        'messages': messages
    }