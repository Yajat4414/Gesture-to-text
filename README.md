# Air Writing ✍️

Write in mid-air using your index finger as a virtual pen. A webcam tracks
your hand in real time (via MediaPipe), and your fingertip's movement is
drawn as strokes on screen, overlaid on the live camera feed.

## How it works

```
Camera → OpenCV frame → MediaPipe HandLandmarker → 21 hand landmarks
       → extract index fingertip → classify gesture → drawing engine
       → canvas layer → merge with camera frame → display
```

The camera image and the drawing are kept as **separate layers** and merged
each frame — this keeps the drawing crisp, makes clearing trivial, and
means the camera feed is never permanently altered.

## Gestures

| Gesture | Hand pose | Action |
|---|---|---|
| ☝️ | Index finger only | **Draw** |
| ✌️ | Index + middle | **Pause** (move without drawing) |
| 🤟 | Index + middle + ring | **Cycle brush color** |
| 🖐️ | All 5 fingers open | **Clear canvas** |
| ✊ | Closed fist (all fingers curled in) | **Erase** near fingertip |
| 🤏 | Thumb + index pinched together | **Pick up & drag** a finished word |

Finger-extended detection is based on each fingertip's **distance from the
wrist** rather than its raw screen position, so it holds up when you tilt
or rotate your hand — this is what fixed the fist-vs-tilt misfire.

### Auto-finalize & smoothing

You don't have to do anything to "finish" a word — about **1 second**
after you stop drawing (any gesture other than ☝️ Draw), the stroke you
just wrote is automatically:

1. cleaned up (near-duplicate points from jitter removed), then
2. fit with a smooth curve (Catmull-Rom spline), then
3. turned into its own standalone "word" object on the canvas.

That finished word can then be picked up and moved anywhere with a pinch
(🤏) — grab it by pinching anywhere inside its bounding box, drag, and
release by opening your pinch.

## Keyboard shortcuts

- `C` — clear canvas
- `S` — save current drawing to `air_writing_output.png`
- `Q` — quit

## Setup

Requires Python 3.9+ and a working webcam.

```bash
# 1. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the hand-tracking model (one-time, ~10 MB)
python download_model.py

# 4. Run it
python main.py
```

> **Note on MediaPipe versions:** Recent MediaPipe releases (0.10.x) removed
> the older `mp.solutions.hands` API that many older tutorials reference.
> This project uses the current, supported **Tasks API**
> (`mediapipe.tasks.vision.HandLandmarker`), which requires the downloaded
> `.task` model file above rather than a model bundled with the package.

## Project structure

```
air-writing/
├── main.py              # App entry point / video loop
├── hand_tracker.py       # Wraps MediaPipe HandLandmarker
├── gesture_detector.py   # Finger-state → gesture classification
├── drawing_engine.py     # Stroke state, brush color/size, smoothing
├── canvas.py              # Drawing layer + merge-with-camera logic
├── utils.py               # Landmark indices, coordinate/geometry helpers
├── download_model.py      # Fetches the HandLandmarker .task model
├── requirements.txt
└── README.md
```

## Troubleshooting

- **"Could not open webcam"** — try `cv2.VideoCapture(1)` (or `2`) in
  `main.py`'s `open_webcam()` if you have multiple cameras, and check that
  no other app is using the camera.
- **Low FPS on integrated graphics** (e.g. Intel Iris Xe) — this runs fine
  on CPU; if it feels sluggish, lower your webcam capture resolution by
  setting `cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)` /
  `cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)` right after `open_webcam()`.
- **Jittery lines while actively drawing** — increase `SMOOTHING_WINDOW` in
  `drawing_engine.py` (trades off responsiveness for smoothness). Note the
  *finished* word is always smoothed regardless, via the spline fit.
- **Word finalizes too soon / too late** — adjust `WORD_FINALIZE_DELAY`
  (seconds) in `drawing_engine.py`. Lower = words finish faster but you
  have less time to lift your pen mid-letter; higher = more forgiving but
  slower to see the "finished" version appear.
- **Erase (fist) or move (pinch) misfire** — tune `EXTEND_MARGIN` and
  `PINCH_THRESHOLD_RATIO` in `gesture_detector.py`. Raising `EXTEND_MARGIN`
  makes the fist check stricter (less likely to false-trigger); lowering
  `PINCH_THRESHOLD_RATIO` requires your thumb and index to be closer
  together before it counts as a pinch.
- **Gestures misfire in general** — lighting matters a lot for hand
  detection; try a well-lit, plain background. You can also tighten
  thresholds in `hand_tracker.py` (`min_detection_confidence`, etc.).

## Possible next steps

Ideas from the original design doc worth building next: shape recognition,
handwriting-to-text (OCR), undo/redo, multi-hand support, exporting to
SVG/PDF, or a proper on-screen toolbar for color/brush selection.
