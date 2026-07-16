"""
utils.py
--------
Shared constants and small helper functions used across the Air Writing
application: landmark indices, coordinate conversion, and geometry helpers.
"""

import math

# ---------------------------------------------------------------------------
# MediaPipe Hand Landmark indices (21 points per hand)
# ---------------------------------------------------------------------------
WRIST = 0

THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# Convenient groupings used by the gesture detector
FINGER_TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_PIPS = [THUMB_IP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
FINGER_MCPS = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]


def normalized_to_pixel(landmark, frame_width, frame_height):
    """Convert a MediaPipe normalized landmark (0-1) into pixel coordinates."""
    x = int(landmark.x * frame_width)
    y = int(landmark.y * frame_height)
    return x, y


def landmarks_to_pixel_list(hand_landmarks, frame_width, frame_height):
    """Convert an entire list of 21 normalized landmarks into pixel tuples."""
    return [normalized_to_pixel(lm, frame_width, frame_height) for lm in hand_landmarks]


def distance(p1, p2):
    """Euclidean distance between two (x, y) pixel points."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def moving_average(points, window=5):
    """
    Smooth a list of (x, y) points using a simple moving average.
    Used to reduce fingertip jitter before drawing.
    """
    if len(points) < window:
        window = len(points)
    if window == 0:
        return points[-1] if points else None

    recent = points[-window:]
    avg_x = sum(p[0] for p in recent) / window
    avg_y = sum(p[1] for p in recent) / window
    return int(avg_x), int(avg_y)


def clamp(value, low, high):
    return max(low, min(high, value))


def midpoint(p1, p2):
    """Midpoint between two (x, y) pixel points."""
    return (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))


def simplify_points(points, min_dist=3):
    """
    Drop points that are too close to the previous kept point. Fingertip
    tracking reports near-duplicate points when the hand is briefly still,
    which otherwise causes bumps/artifacts in the spline fit below.
    """
    if not points:
        return []
    result = [points[0]]
    for p in points[1:]:
        if distance(result[-1], p) >= min_dist:
            result.append(p)
    return result


def catmull_rom_spline(points, samples_per_segment=12):
    """
    Fit a smooth curve through a list of (x, y) points using Catmull-Rom
    spline interpolation. This is what turns a raw, jittery hand-tracked
    stroke into a clean "finished handwriting" curve once a word is done.
    Needs at least 3 points; shorter input is returned unchanged.
    """
    if len(points) < 3:
        return list(points)

    # Duplicate the first/last point so every real segment has 4 control
    # points to interpolate between (standard Catmull-Rom padding trick).
    pts = [points[0]] + list(points) + [points[-1]]
    result = []

    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
        for j in range(samples_per_segment):
            t = j / samples_per_segment
            t2, t3 = t * t, t * t * t
            x = 0.5 * (
                (2 * p1[0])
                + (-p0[0] + p2[0]) * t
                + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
            )
            y = 0.5 * (
                (2 * p1[1])
                + (-p0[1] + p2[1]) * t
                + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
            )
            result.append((int(x), int(y)))

    result.append(pts[-2])
    return result
