"""
download_model.py
------------------
Downloads the MediaPipe HandLandmarker model file used by hand_tracker.py.
Run this once before starting the app:

    python download_model.py
"""

import os
import urllib.request

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")


def download():
    if os.path.exists(OUTPUT_PATH):
        print(f"Model already exists at {OUTPUT_PATH}, skipping download.")
        return

    print(f"Downloading hand landmark model from:\n  {MODEL_URL}")
    urllib.request.urlretrieve(MODEL_URL, OUTPUT_PATH)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    download()
