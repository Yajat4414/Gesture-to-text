"""
canvas.py
---------
Manages the drawing surface, which is made of two kinds of content:

  - `preview`: the raw, in-progress stroke for whatever the user is
    currently writing. Drawn immediately for responsive feedback, but
    replaced once the word is finished (see drawing_engine.py).

  - `words`: a list of finished, smoothed strokes ("words"), each stored
    as its own small sub-image with a position on the canvas. Because
    each word is a separate object, it can be picked up and dragged
    independently via the pinch/MOVE gesture.
"""

import numpy as np
import cv2

WORD_PADDING = 15


class Canvas:
    def __init__(self, width, height, channels=3):
        self.width = width
        self.height = height
        self.channels = channels
        self.blank = np.zeros((height, width, channels), dtype=np.uint8)
        self.preview = self.blank.copy()
        self.words = []  # list of {id, image, x, y, w, h}
        self._next_id = 1

    # ------------------------------------------------------------------
    # Live preview: raw stroke while the user is actively drawing
    # ------------------------------------------------------------------
    def draw_preview_line(self, pt1, pt2, color, thickness):
        if pt1 is None or pt2 is None:
            return
        cv2.line(self.preview, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)

    def clear_preview(self):
        self.preview = self.blank.copy()

    # ------------------------------------------------------------------
    # Finished, movable "words"
    # ------------------------------------------------------------------
    def add_word(self, points, color, thickness):
        """Rasterize a finished (already-smoothed) stroke into its own
        movable sub-image and store it. Returns the new word's id."""
        if not points:
            return None

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x0 = max(min(xs) - WORD_PADDING, 0)
        y0 = max(min(ys) - WORD_PADDING, 0)
        x1 = min(max(xs) + WORD_PADDING, self.width)
        y1 = min(max(ys) + WORD_PADDING, self.height)
        w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0:
            return None

        sub_img = np.zeros((h, w, self.channels), dtype=np.uint8)
        local_pts = [(px - x0, py - y0) for px, py in points]

        if len(local_pts) == 1:
            cv2.circle(sub_img, local_pts[0], max(thickness // 2, 2), color, -1)
        else:
            for p1, p2 in zip(local_pts, local_pts[1:]):
                cv2.line(sub_img, p1, p2, color, thickness, lineType=cv2.LINE_AA)

        word_id = self._next_id
        self._next_id += 1
        self.words.append({"id": word_id, "image": sub_img, "x": x0, "y": y0, "w": w, "h": h})
        return word_id

    def word_at(self, point):
        """Return the id of the topmost (most recently written) word whose
        bounding box contains `point`, or None."""
        if point is None:
            return None
        px, py = point
        for word in reversed(self.words):
            if word["x"] <= px <= word["x"] + word["w"] and word["y"] <= py <= word["y"] + word["h"]:
                return word["id"]
        return None

    def get_word(self, word_id):
        for word in self.words:
            if word["id"] == word_id:
                return word
        return None

    def move_word(self, word_id, new_x, new_y):
        word = self.get_word(word_id)
        if word is None:
            return
        word["x"] = int(np.clip(new_x, 0, max(self.width - word["w"], 0)))
        word["y"] = int(np.clip(new_y, 0, max(self.height - word["h"], 0)))

    # ------------------------------------------------------------------
    # Erase / clear
    # ------------------------------------------------------------------
    def erase_near(self, center, radius):
        """Erase both the live preview and any finished word pixels near
        `center` (used by the closed-fist ERASE gesture)."""
        if center is None:
            return
        cv2.circle(self.preview, center, radius, (0, 0, 0), thickness=-1)

        cx, cy = center
        for word in self.words:
            lx, ly = cx - word["x"], cy - word["y"]
            if -radius <= lx <= word["w"] + radius and -radius <= ly <= word["h"] + radius:
                cv2.circle(word["image"], (lx, ly), radius, (0, 0, 0), thickness=-1)

        # Drop words that have been fully erased away
        self.words = [w for w in self.words if np.any(w["image"])]

    def clear(self):
        self.preview = self.blank.copy()
        self.words = []

    # ------------------------------------------------------------------
    # Compositing / output
    # ------------------------------------------------------------------
    def _composite(self):
        layer = self.preview.copy()
        for word in self.words:
            x, y, w, h = word["x"], word["y"], word["w"], word["h"]
            sub = word["image"]
            gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            roi = layer[y:y + h, x:x + w]
            bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            fg = cv2.bitwise_and(sub, sub, mask=mask)
            layer[y:y + h, x:x + w] = cv2.add(bg, fg)
        return layer

    def merge_with_frame(self, frame_bgr):
        layer = self._composite()
        gray_layer = cv2.cvtColor(layer, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_layer, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        background = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask_inv)
        foreground = cv2.bitwise_and(layer, layer, mask=mask)
        return cv2.add(background, foreground)

    def save(self, path):
        cv2.imwrite(path, self._composite())
