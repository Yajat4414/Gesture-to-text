"""
hand_tracker.py
----------------
Wraps MediaPipe's HandLandmarker (Tasks API) to detect a hand in a video
frame and return its 21 landmarks.

Note: Modern MediaPipe (0.10.x) removed the old `mp.solutions.hands` API.
Hand detection now goes through the unified Tasks API, which needs a
`.task` model file. Run `download_model.py` once before using this module
(see README.md).
"""

import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")


class HandTracker:
    def __init__(self, model_path=MODEL_PATH, num_hands=1,
                 min_detection_confidence=0.6, min_presence_confidence=0.6,
                 min_tracking_confidence=0.6):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Hand landmark model not found at '{model_path}'.\n"
                "Run 'python download_model.py' first to download it."
            )

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def process(self, frame_bgr):
        """
        Run hand detection on a single BGR frame (as read by OpenCV).
        Returns a list of hands, each a list of 21 landmark objects
        (with .x, .y, .z in normalized 0-1 coordinates), or an empty
        list if no hand was detected.
        """
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # VIDEO mode requires monotonically increasing timestamps in ms
        self._timestamp_ms += 1
        result = self.landmarker.detect_for_video(mp_image, self._timestamp_ms)

        if not result.hand_landmarks:
            return []
        return result.hand_landmarks  # list of list[NormalizedLandmark]

    def close(self):
        self.landmarker.close()
