"""Finite tool cylinder versus conservative robot link capsules, offline only.

The tool starts at the flange plane: no arbitrary length is excluded. Only the
shared mounting face of the last coaxial wrist link is allowed to touch it.
"""
import math
import numpy as np

ROBOT_RADIUS_M = .050
TOOL_RADIUS_M = 1.5 * ROBOT_RADIUS_M
TOOL_LENGTH_M = .246
REQUIRED_CLEARANCE_M = .010
LINK_NAMES = ('shoulder', 'upper_arm', 'forearm', 'wrist_1', 'wrist_2')


def point_cylinder_distance(point, radius=TOOL_RADIUS_M, length=TOOL_LENGTH_M):
    """Distance to solid flat-capped cylinder in flange coordinates (+Z)."""
    x, y, z = point
    radial = max(math.hypot(x, y)-radius, 0.)
    axial = max(-z, z-length, 0.)
    return math.hypot(radial, axial)


def segment_cylinder_distance(start, end, radius=TOOL_RADIUS_M, length=TOOL_LENGTH_M):
    """Convex 1-D minimization of exact point-to-cylinder distance."""
    start, end = np.asarray(start), np.asarray(end)
    delta = end-start
    def value(t):
        return point_cylinder_distance(start+t*delta, radius, length)
    low, high = 0., 1.
    ratio = (math.sqrt(5)-1)/2
    left, right = high-ratio*(high-low), low+ratio*(high-low)
    fl, fr = value(left), value(right)
    for _ in range(48):
        if fl < fr:
            high, right, fr = right, left, fl
            left = high-ratio*(high-low)
            fl = value(left)
        else:
            low, left, fl = left, right, fr
            right = low+ratio*(high-low)
            fr = value(right)
    return min(value(0), value(1), fl, fr)


def tool_body_clearances(flange, points):
    """Signed exterior clearance; negative means proxy solid intersection."""
    local = (np.asarray(points)-flange[:3, 3])@flange[:3, :3]
    gaps = {name: segment_cylinder_distance(local[i], local[i+1])-ROBOT_RADIUS_M
            for i, name in enumerate(LINK_NAMES)}
    # Base and its mounting ring, using a conservative capsule envelope.
    base = (np.array([[0., 0., 0.], [0., 0., .035]])-flange[:3, 3])@flange[:3, :3]
    gaps['base'] = segment_cylinder_distance(*base)-.100
    # The final link is coaxial and ends exactly at the mounting plane. Check
    # that its whole axis remains behind it, rather than deleting tool length.
    if np.linalg.norm(local[5, :2]) > 1e-8 or local[5, 2] >= 0 or np.linalg.norm(local[6]) > 1e-8:
        raise ValueError('Last wrist link is not confined behind the mounting face')
    return gaps


def geometry_description():
    return dict(robot_link_radius_m=ROBOT_RADIUS_M, robot_link_diameter_m=2*ROBOT_RADIUS_M,
                tool_radius_m=TOOL_RADIUS_M, tool_diameter_m=2*TOOL_RADIUS_M,
                tool_length_m=TOOL_LENGTH_M, thickness_ratio=1.5,
                tool_shape='flat-capped solid cylinder, flange Z=0 through TCP Z=246 mm',
                body_shape='50 mm radius link capsules and 100 mm radius base capsule',
                required_clearance_m=REQUIRED_CLEARANCE_M,
                mounting_contact='Only the common flange face with the last coaxial wrist link is allowed; no tool length is omitted',
                dimensional_source='Conservative 100 mm link diameter from existing animation; not an on-site measurement')
