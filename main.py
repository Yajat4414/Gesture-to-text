"""
main.py
-------
Air Writing — write in mid-air using your index finger as a virtual pen,
tracked live through your webcam.

Gestures:
    ☝  Index finger only        -> Draw
    ✌  Index + middle           -> Pause (move without drawing)
    🤟 Index + middle + ring    -> Cycle brush color
    🖐  Open hand (all fingers)  -> Clear canvas
    ✊ Closed fist               -> Erase near fingertip
    🤏 Pinch (thumb + index)    -> Pick up & drag a finished word

A word is finalized automatically ~1 second after you stop drawing: the
raw stroke is smoothed into a clean curve and becomes its own object that
can be grabbed and moved with the pinch gesture.

Keyboard:
    C -> Clear canvas
    S -> Save drawing to disk
    Q -> Quit

Setup:
    pip install -r requirements.txt
    python download_model.py     # one-time model download
    python main.py
"""

import time
import cv2

from hand_tracker import HandTracker
from gesture_detector import classify_gesture, DRAW, PAUSE, CLEAR, ERASE, MOVE, COLOR, IDLE
from drawing_engine import DrawingEngine
from canvas import Canvas
from utils import landmarks_to_pixel_list

WINDOW_NAME = "Air Writing"
SAVE_PATH = "air_writing_output.png"

GESTURE_LABELS = {
    DRAW: "Drawing",
    PAUSE: "Paused",
    CLEAR: "Clear",
    ERASE: "Erasing (fist)",
    MOVE: "Moving word (pinch)",
    COLOR: "Color change",
    IDLE: "Idle",
}


def open_webcam(preferred_index=0):
    cap = cv2.VideoCapture(preferred_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam at index {preferred_index}. "
            "Try a different index (1, 2, ...) or check camera permissions."
        )
    return cap


def draw_hud(frame, gesture, color, fps):
    label = GESTURE_LABELS.get(gesture, gesture)
    cv2.rectangle(frame, (0, 0), (260, 70), (0, 0, 0), thickness=-1)
    cv2.putText(frame, f"Gesture: {label}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.0f}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(frame, (240, 35), 15, color, thickness=-1)
    cv2.circle(frame, (240, 35), 15, (200, 200, 200), thickness=1)


def main():
    cap = open_webcam(preferred_index=1)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    tracker = HandTracker(num_hands=1)
    canvas = Canvas(frame_width, frame_height)
    engine = DrawingEngine(canvas)

    prev_time = time.time()
    fps = 0.0

    print("Air Writing started. Press Q to quit, C to clear, S to save.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from webcam. Exiting.")
                break

            frame = cv2.flip(frame, 1)  # mirror view

            hands = tracker.process(frame)
            gesture = IDLE

            if hands:
                hand_landmarks = hands[0]
                landmarks_px = landmarks_to_pixel_list(
                    hand_landmarks, frame_width, frame_height
                )
                gesture, point = classify_gesture(landmarks_px, frame_width)
                engine.update(gesture, point)

                # Visual feedback: mark the fingertip/pinch point being tracked
                cv2.circle(frame, point, 8, engine.current_color, -1)
            else:
                engine.update(IDLE, None)

            # Highlight the word currently being dragged, if any
            if engine.grabbed_word_id is not None:
                word = canvas.get_word(engine.grabbed_word_id)
                if word is not None:
                    x, y, w, h = word["x"], word["y"], word["w"], word["h"]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 200, 0), 2)

            output = canvas.merge_with_frame(frame)

            now = time.time()
            fps = 1.0 / (now - prev_time) if now > prev_time else fps
            prev_time = now
            draw_hud(output, gesture, engine.current_color, fps)

            cv2.imshow(WINDOW_NAME, output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                canvas.clear()
            elif key == ord('s'):
                canvas.save(SAVE_PATH)
                print(f"Saved drawing to {SAVE_PATH}")

    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
