"""
drawing_engine.py
-------------------
Turns fingertip positions + gestures into strokes on the Canvas.

While the user is actively drawing (DRAW gesture), raw fingertip points
are collected and shown immediately as a live, lightly-smoothed preview
line -- this keeps the drawing feel responsive.

Once the user stops drawing for WORD_FINALIZE_DELAY seconds (i.e. they've
"finished writing a word"), the raw points collected during that stroke
are automatically:
  1. simplified (duplicate/near-duplicate points removed), then
  2. fit with a Catmull-Rom spline for a smooth, "finished handwriting"
     curve, then
  3. rasterized into its own movable "word" object on the canvas.

Finished words can then be picked up and dragged with a thumb-index
pinch (MOVE gesture).
"""

import time

from utils import moving_average, simplify_points, catmull_rom_spline
from gesture_detector import DRAW, PAUSE, CLEAR, ERASE, MOVE, COLOR

# A small palette to cycle through with the COLOR gesture (BGR order)
PALETTE = [
    (0, 0, 255),      # red
    (0, 255, 0),      # green
    (255, 0, 0),      # blue
    (0, 255, 255),    # yellow
    (255, 255, 255),  # white
]

ERASE_RADIUS = 30
BRUSH_THICKNESS = 6
SMOOTHING_WINDOW = 5
COLOR_DEBOUNCE_FRAMES = 15

# How long (seconds) the hand must stop actively drawing before the
# in-progress stroke is treated as a "finished word" and smoothed.
WORD_FINALIZE_DELAY = 1.0
# A DRAW frame more recent than this counts as "still actively drawing".
STILL_DRAWING_WINDOW = 0.05


class DrawingEngine:
    def __init__(self, canvas):
        self.canvas = canvas
        self.prev_point = None
        self.recent_points = []   # live smoothing buffer (moving average)
        self.active_stroke = []   # raw points collected for the current word
        self.color_index = 0
        self.brush_thickness = BRUSH_THICKNESS
        self._color_cooldown = 0
        self._last_draw_time = None

        self.grabbed_word_id = None
        self._grab_offset = (0, 0)

    @property
    def current_color(self):
        return PALETTE[self.color_index]

    def _next_color(self):
        self.color_index = (self.color_index + 1) % len(PALETTE)

    def update(self, gesture, point):
        """Apply one frame's worth of gesture + fingertip/pinch point."""
        now = time.time()

        if self._color_cooldown > 0:
            self._color_cooldown -= 1

        if gesture == DRAW:
            self.recent_points.append(point)
            smoothed = moving_average(self.recent_points, SMOOTHING_WINDOW)
            self.canvas.draw_preview_line(self.prev_point, smoothed, self.current_color,
                                           self.brush_thickness)
            self.prev_point = smoothed
            self.active_stroke.append(smoothed)
            self._last_draw_time = now
            self.grabbed_word_id = None

        elif gesture == MOVE:
            self._handle_move(point)
            self._reset_stroke()

        elif gesture == ERASE:
            self.canvas.erase_near(point, ERASE_RADIUS)
            self._reset_stroke()
            self.grabbed_word_id = None

        elif gesture == CLEAR:
            self.canvas.clear()
            self._abandon_active_stroke()
            self.grabbed_word_id = None

        elif gesture == COLOR:
            if self._color_cooldown == 0:
                self._next_color()
                self._color_cooldown = COLOR_DEBOUNCE_FRAMES
            self._reset_stroke()
            self.grabbed_word_id = None

        else:  # PAUSE / IDLE
            self._reset_stroke()
            self.grabbed_word_id = None

        self._maybe_finalize_word(now)

    # ----------------------------------------------------------------
    # Word finalization: raw stroke -> smoothed, movable word
    # ----------------------------------------------------------------
    def _maybe_finalize_word(self, now):
        if not self.active_stroke or self._last_draw_time is None:
            return
        still_drawing = (now - self._last_draw_time) < STILL_DRAWING_WINDOW
        idle_long_enough = (now - self._last_draw_time) >= WORD_FINALIZE_DELAY
        if not still_drawing and idle_long_enough:
            self._finalize_word()

    def _finalize_word(self):
        pts = simplify_points(self.active_stroke, min_dist=3)
        if pts:
            smooth_pts = catmull_rom_spline(pts, samples_per_segment=12) if len(pts) >= 3 else pts
            self.canvas.add_word(smooth_pts, self.current_color, self.brush_thickness)
        self.canvas.clear_preview()
        self.active_stroke = []
        self._last_draw_time = None

    def _abandon_active_stroke(self):
        """Discard the in-progress stroke without turning it into a word
        (used when the canvas is cleared mid-stroke)."""
        self.active_stroke = []
        self._last_draw_time = None
        self._reset_stroke()

    # ----------------------------------------------------------------
    # Drag a finished word around via pinch
    # ----------------------------------------------------------------
    def _handle_move(self, point):
        if point is None:
            return
        if self.grabbed_word_id is None:
            word_id = self.canvas.word_at(point)
            if word_id is not None:
                word = self.canvas.get_word(word_id)
                self._grab_offset = (point[0] - word["x"], point[1] - word["y"])
                self.grabbed_word_id = word_id
        else:
            new_x = point[0] - self._grab_offset[0]
            new_y = point[1] - self._grab_offset[1]
            self.canvas.move_word(self.grabbed_word_id, new_x, new_y)

    def _reset_stroke(self):
        self.prev_point = None
        self.recent_points = []
