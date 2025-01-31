# rule_based_model.py
"""
Refactored to allow adjustable targets and tolerances for each angle,
so users can tailor the system for different ergonomic requirements.

Example usage:
    from rule_based_model import update_angle_config, rule_based_posture_analysis

    # Update the knee angle to 90° ± 10° tolerance
    update_angle_config('knee_angle', new_target=90, new_tolerance=10)

    # Then call rule_based_posture_analysis(...) as usual
"""

###############################################################################
# 1) CONFIG DICTIONARY WITH PER-ANGLE TARGET AND TOLERANCE
###############################################################################
ANGLE_CONFIG = {
    'knee_angle': {
        'description': "Knee angle",
        'target': 100,       # Default target angle in degrees
        'tolerance': 5,      # ± tolerance around the target
        'solutions': [
            "Lower or raise your seat so the knee is at ~100"
        ]
    },
    'hip_angle': {
        'description': "Hip angle",
        'target': 100,
        'tolerance': 5,
        'solutions': [
            "Adjust seat depth or desk height to achieve ~100° at hips"
        ]
    },
    'elbow_angle': {
        'description': "Elbow angle",
        'target': 95,
        'tolerance': 5,
        'solutions': [
            "Raise or lower armrests so elbows are about 95°"
        ]
    },
    'head_to_shoulder_angle': {
        'description': "Head to Shoulder angle",
        'target': 160,
        'tolerance': 5,
        'solutions': [
            "Raise or lower your monitor to reduce neck bending",
            "Keep your head upright; adjust screen distance"
        ]
    }
}


def update_angle_config(angle_name, new_target=None, new_tolerance=None):
    """
    Dynamically updates the target and/or tolerance for a given angle.
    If 'new_target' or 'new_tolerance' is None, that value is left unchanged.

    :param angle_name: string key (e.g. 'knee_angle', 'hip_angle')
    :param new_target: new desired target angle (int or float) or None
    :param new_tolerance: new desired tolerance (int or float) or None
    """
    if angle_name not in ANGLE_CONFIG:
        print(f"[Warning] '{angle_name}' not found in ANGLE_CONFIG.")
        return

    if new_target is not None:
        ANGLE_CONFIG[angle_name]['target'] = new_target

    if new_tolerance is not None:
        ANGLE_CONFIG[angle_name]['tolerance'] = new_tolerance


###############################################################################
# 2) CORE EVALUATION LOGIC
###############################################################################
def evaluate_angle(angle_name, angle_value):
    """
    Checks if 'angle_value' lies within ±tolerance of the recommended target
    for 'angle_name'.

    Returns a dict:
      {
        'angle_name': angle_name,
        'ok': bool,
        'diff': float,
        'message': str,
        'solutions': [...]
      }

    'ok' indicates whether the angle is considered 'good'.
    'diff' is how many degrees difference from the target value.
    'message' is a user-friendly summary.
    'solutions' suggests possible actions if 'ok' is False.
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
        msg = (
            f"{rec['description']} is out of range by {diff:.1f}° "
            f"(measured {angle_value:.1f}°, target ~{target}° ±{tolerance})."
        )
        return {
            'angle_name': angle_name,
            'ok': False,
            'diff': diff,
            'message': msg,
            'solutions': rec['solutions']
        }


def compute_overall_score(evaluations):
    """
    Simple scoring:
      - 0 angles out of range => 100%
      - 1 => 80%
      - 2 => 60%
      - 3 => 40%
      - 4 or more => 20%

    Returns (score, rating_str)
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
    Evaluate the measured angles in 'angles_dict' using the threshold-based approach,
    then compute an overall score for the posture.

    :param image_bgr: The original image (not directly used for classification, but kept if needed)
    :param angles_dict: Dict of { 'knee_angle': val, 'hip_angle': val, 'elbow_angle': val, ... }
    :param side: e.g., 'left' or 'right', not strictly used here but could influence certain logic
    :param landmarks_dict: optional, if needed for additional context

    :return: A dict containing:
        {
            'evaluations': [ ... ],
            'score': int,
            'rating': str,
            'messages': [ ... ]
        }
    """
    # 1) Evaluate each angle
    evaluations = []
    for angle_name, angle_val in angles_dict.items():
        if angle_val is not None:
            result = evaluate_angle(angle_name, angle_val)
            evaluations.append(result)

    # 2) Compute overall score
    score, rating = compute_overall_score(evaluations)

    # 3) Summarize messages
    messages = []
    for ev in evaluations:
        messages.append(ev['message'])
        if ev['ok'] is False:
            # if an angle is out of range, we also mention possible solutions
            for sol in ev['solutions']:
                messages.append(f"Try:\n  {sol}")

    summary = f"Overall Score: {score}%, rated {rating}"
    messages.append(summary)

    return {
        'evaluations': evaluations,
        'score': score,
        'rating': rating,
        'messages': messages
    }