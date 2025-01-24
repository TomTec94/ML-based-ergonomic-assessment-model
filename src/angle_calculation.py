import math

def compute_2d_angle(a, b, c):
    """
    Computes the angle at point b, given coordinates a, b, c in 2D.
    Returns angle in degrees.
    """
    ax, ay = a
    bx, by = b
    cx, cy = c

    # Vector BA and BC
    ba = (ax - bx, ay - by)
    bc = (cx - bx, cy - by)

    dot_prod = ba[0]*bc[0] + ba[1]*bc[1]
    mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)

    if mag_ba * mag_bc == 0:
        return None

    cos_angle = dot_prod / (mag_ba * mag_bc)
    # clamp to avoid floating errors
    cos_angle = max(min(cos_angle, 1.0), -1.0)

    angle_rad = math.acos(cos_angle)
    angle_deg = math.degrees(angle_rad)
    return angle_deg