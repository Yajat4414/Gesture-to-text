"""
gesture_detector.py
--------------------
Determines which fingers are extended and classifies that into one of
the app's gestures:

    DRAW    -> index finger only              (write on the canvas)
    PAUSE   -> index + middle fingers         (move without drawing)
    CLEAR   -> open hand (all 5 fingers up)   (wipe the canvas)
    COLOR   -> index + middle + ring fingers  (cycle brush color)
    ERASE   -> closed fist (all fingers curled in)
    MOVE    -> pinch (thumb tip touching index tip, hand not a full fist)
               -> pick up and drag the finished word under the pinch point
    IDLE    -> anything else / no hand

Finger-extended detection uses DISTANCE FROM THE WRIST rather than a raw
y-coordinate comparison. Distance-based checks are invariant to in-plane
hand tilt/rotation, which is what caused the old y-comparison approach to
misread a tilted "pinch" as a fist (or vice versa) -- rotating the hand
changes each landmark's y-coordinate a lot, but barely changes how far a
fingertip is from the wrist.
"""

from utils import (
    FINGER_TIPS, FINGER_PIPS,
    WRIST, THUMB_TIP, THUMB_MCP, INDEX_TIP, MIDDLE_MCP,
    distance, midpoint,
)

DRAW = "DRAW"
PAUSE = "PAUSE"
CLEAR = "CLEAR"
ERASE = "ERASE"
MOVE = "MOVE"
COLOR = "COLOR"
IDLE = "IDLE"

# Pinch distance threshold, expressed as a fraction of the hand's own size
# (wrist-to-middle-knuckle distance) rather than frame width. This makes it
# scale-invariant -- the same physical pinch triggers it whether your hand
# is close to or far from the camera.
PINCH_THRESHOLD_RATIO = 0.4

# Margin added to the "extended" comparison so borderline/noisy frames
# don't flicker between extended/curled.
EXTEND_MARGIN = 1.05


def fingers_extended(landmarks_px, handedness="Right"):
    """
    Given 21 pixel-space landmarks, return [thumb, index, middle, ring, pinky]
    booleans for whether each finger is extended, using wrist-distance
    comparisons (robust to hand tilt/rotation in the image plane).
    """
    wrist = landmarks_px[WRIST]

    thumb_tip_dist = distance(landmarks_px[THUMB_TIP], wrist)
    thumb_mcp_dist = distance(landmarks_px[THUMB_MCP], wrist)
    fingers = [thumb_tip_dist > thumb_mcp_dist * EXTEND_MARGIN]

    for tip_idx, pip_idx in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        tip_dist = distance(landmarks_px[tip_idx], wrist)
        pip_dist = distance(landmarks_px[pip_idx], wrist)
        fingers.append(tip_dist > pip_dist * EXTEND_MARGIN)

    return fingers  # [thumb, index, middle, ring, pinky]


def classify_gesture(landmarks_px, frame_width, handedness="Right"):
    """
    Classify the current hand pose into one of the gesture constants.
    Returns (gesture, point) where point is the (x, y) pixel position to
    use for drawing/pointing/dragging -- the index tip normally, or the
    thumb-index midpoint while pinching.
    """
    thumb, index, middle, ring, pinky = fingers_extended(landmarks_px, handedness)
    index_point = landmarks_px[INDEX_TIP]

    # Closed fist: every finger curled in, regardless of tilt/rotation.
    # Checked first and independently of the pinch test below so a fist
    # can never be misread as a pinch.
    if not (thumb or index or middle or ring or pinky):
        return ERASE, index_point

    # Pinch: thumb and index tip touching. Threshold is scaled by the
    # hand's own size (wrist -> middle-knuckle) so it holds up regardless
    # of distance from the camera or hand tilt.
    hand_scale = distance(landmarks_px[WRIST], landmarks_px[MIDDLE_MCP]) or 1
    pinch_dist = distance(landmarks_px[THUMB_TIP], landmarks_px[INDEX_TIP])
    if pinch_dist < PINCH_THRESHOLD_RATIO * hand_scale:
        pinch_point = midpoint(landmarks_px[THUMB_TIP], landmarks_px[INDEX_TIP])
        return MOVE, pinch_point

    if index and middle and ring and not pinky:
        return COLOR, index_point

    if index and middle and not ring and not pinky:
        return PAUSE, index_point

    if index and not middle and not ring and not pinky:
        return DRAW, index_point

    if thumb and index and middle and ring and pinky:
        return CLEAR, index_point

    return IDLE, index_point
