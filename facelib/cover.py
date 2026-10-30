"""
Is each eye really visible, or is something (a hand) in front of it?

Why this exists: the face model always draws an eye outline, even when a hand hides the
eye, so the eyelid measure in analysis.py still reads "open". This check looks at the
PICTURE inside each eye outline instead. A visible eye has a dark iris, white of the eye
and lashes, so the pixels vary a lot. A hand or other cover is a smooth patch, so they
barely vary.

The number is "image detail" per eye: (bright pixels - dark pixels) / average brightness
inside the eye outline. Dividing by the average makes it work for light and dark skin and
for bright and dim rooms. A covered eye is one whose detail has dropped to a fraction of
what this person's open eyes show (COVER_RATIO), judged against:
  * the other eye in the same frame, and
  * the person's own "normal" detail, learned from the frames where both eyes looked alike.
So the first seconds of the exam should show a clear view of the face.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import config as cfg

# Outline of each eye (MediaPipe's own FACE_LANDMARKS_RIGHT_EYE / LEFT_EYE lists).
# A goes with the iris landmark 468, B with 473 (see EYE_A / EYE_B in analysis.py).
CONTOUR_A = [7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246]
CONTOUR_B = [249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466]

MIN_PIXELS = 30        # an outline smaller than this has too little to judge
SMOOTHING = 0.6        # weight of the previous value: stops one noisy frame from flipping the result
BALANCED = 0.75        # two eyes within this ratio of each other are treated as "both uncovered"
WARMUP_UPDATES = 15    # balanced frames needed before the learned normal is trusted
BASELINE_RATE = 0.02   # how fast the learned normal follows slow changes (light, distance)


def eye_detail(gray, pts, ids):
    """Image detail inside one eye outline, or None if the outline is too small to judge."""
    outline = cv2.convexHull(np.round(pts[ids]).astype(np.int32))
    x, y, w, h = cv2.boundingRect(outline)
    x0, y0 = max(x - 1, 0), max(y - 1, 0)
    crop = gray[y0:y + h + 2, x0:x + w + 2]
    if crop.size == 0:
        return None
    crop = cv2.GaussianBlur(crop, (3, 3), 0)  # camera noise would look like detail
    mask = np.zeros(crop.shape, np.uint8)
    cv2.fillConvexPoly(mask, outline - (x0, y0), 255)
    values = crop[mask > 0].astype(np.float32)
    if values.size < MIN_PIXELS:
        return None
    low, high = np.percentile(values, [10, 90])
    return float((high - low) / (values.mean() + 10.0))


@dataclass
class CoverResult:
    detail_l: float = None        # image detail of the eye on the left of the picture
    detail_r: float = None
    normal: float = None          # this person's learned normal (None while still learning)
    covered: set = field(default_factory=set)   # {"LEFT"} / {"RIGHT"} / both
    reference: float = None       # what the covered eye was compared with


class CoverDetector:
    """Keeps the smoothed per-eye detail and the learned normal between frames."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.smooth = {"LEFT": None, "RIGHT": None}
        self.baseline = None
        self._updates = 0
        self._both_since = None

    def update(self, gray, pts, ids_left, ids_right, ear_left, ear_right, frontal, now=None):
        """Measure both eyes. `frontal` is False while the head is turned far (the far eye is
        then hidden by the nose, which is not a cover). Shut eyes are skipped: the eyelid
        measure already handles those."""
        for side, ids, ear in (("LEFT", ids_left, ear_left), ("RIGHT", ids_right, ear_right)):
            detail = eye_detail(gray, pts, ids) if ear >= cfg.EAR_CLOSED else None
            previous = self.smooth[side]
            if detail is None:
                self.smooth[side] = None
            elif previous is None:
                self.smooth[side] = detail
            else:
                self.smooth[side] = SMOOTHING * previous + (1 - SMOOTHING) * detail

        result = CoverResult(self.smooth["LEFT"], self.smooth["RIGHT"])
        trusted = self.baseline if self._updates >= WARMUP_UPDATES else None
        result.normal = trusted
        if not frontal:
            self._both_since = None
            return result

        for side, mine, other in (("LEFT", result.detail_l, result.detail_r),
                                  ("RIGHT", result.detail_r, result.detail_l)):
            if mine is None:
                continue
            reference = max(trusted or 0.0, other or 0.0)
            if mine < cfg.COVER_FLOOR or (reference > 0 and mine < cfg.COVER_RATIO * reference):
                result.covered.add(side)
                result.reference = reference or None

        # Learn what "normal" looks like from frames where both eyes are visible and alike.
        left, right = result.detail_l, result.detail_r
        if left is not None and right is not None and not result.covered:
            low, high = sorted((left, right))
            if high > 0 and low / high >= BALANCED:
                mean = (left + right) / 2
                self.baseline = mean if self.baseline is None else (
                    (1 - BASELINE_RATE) * self.baseline + BASELINE_RATE * mean)
                self._updates += 1

        # Both "covered" for a long time: more likely the light or camera changed than a hand
        # held there. The flag has long since fired, so start learning normal again.
        if len(result.covered) == 2 and now is not None:
            if self._both_since is None:
                self._both_since = now
            elif now - self._both_since > cfg.COVER_RELEARN_SECONDS:
                self.baseline, self._updates, self._both_since = None, 0, None
        else:
            self._both_since = None
        return result
