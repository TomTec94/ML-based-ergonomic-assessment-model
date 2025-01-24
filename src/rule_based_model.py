from angle_calculation import compute_2d_angle

def classify_posture(landmarks):
    """
    Returns 'ergonomic' or 'non-ergonomic'
    based on a set of angle thresholds.
    """
    if landmarks is None:
        return "non-ergonomic"

    # Example thresholds
    knee_min, knee_max = 80.0, 100.0
    head_min_angle = 160.0  # e.g., user should not tilt head too far forward

    # Grab relevant points for left side (hip=23, knee=25, ankle=27)
    hip = (landmarks[23][0], landmarks[23][1])
    knee = (landmarks[25][0], landmarks[25][1])
    ankle = (landmarks[27][0], landmarks[27][1])

    knee_angle = compute_2d_angle(hip, knee, ankle)
    if knee_angle is None:
        return "non-ergonomic"

    # If knee angle outside [80, 100], posture flagged
    if not (knee_min <= knee_angle <= knee_max):
        return "non-ergonomic"

    # Check head angle if needed (shoulder=11, ear=7, hip=23)
    shoulder = (landmarks[11][0], landmarks[11][1])
    ear      = (landmarks[7][0],  landmarks[7][1])
    head_angle = compute_2d_angle(shoulder, ear, hip)

    # if head angle is too small (means forward-lean)
    if head_angle and head_angle < head_min_angle:
        return "non-ergonomic"

    return "ergonomic"