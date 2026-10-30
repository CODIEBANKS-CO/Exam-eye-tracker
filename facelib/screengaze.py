"""
Gaze mapped onto the computer screen (run:  python gaze_demo.py --screen).

How it works
  1. FEATURES. From each frame we take a few numbers that change when the eyes look at a
     different part of the screen: where the iris sits inside the eye (sideways and up/down,
     measured in eye-widths and along the eye's own axis so a tilted head does not matter),
     the head turn and tilt, and where the head is in the picture.
  2. CALIBRATION. The person looks at 9 dots shown at known places on the screen. For each dot
     we record the features. A small curve fit (ridge regression, numpy only) then learns
     "these feature numbers  ->  this point on the screen" for THIS person, camera and seat.
     A cross-check predicts each dot from the other eight and reports the typical error.
  3. TRACKING. Every frame the features go through the fitted curve to give a screen point,
     which is smoothed (One Euro filter) so the dot does not shake. The point is drawn on a
     full-screen "gaze pad" and, if it stays outside the screen, GAZE_OFF_SCREEN is raised.

Limits (a webcam is not an eye tracker): expect an error of several centimetres on screen,
more if the head moves or the seat changes after calibration. Use it for "which part of the
screen", not for exact pixels. Recalibrate (press c) if the person shifts position.
"""

import ctypes
import json
import os
import sys
from collections import deque
from datetime import datetime

import cv2
import numpy as np

from . import config as cfg

DOT_COLOR = (0, 255, 255)   # BGR yellow: the calibration dot
EYE_LINEAR_RIDGE = 1e-4     # tiny penalty on the two main eye terms
WHITE = (255, 255, 255)


# ------------------------------- screen size -----------------------------
def screen_size():
    """(width, height) of the main screen in pixels."""
    if cfg.SCREEN_WIDTH and cfg.SCREEN_HEIGHT:
        return int(cfg.SCREEN_WIDTH), int(cfg.SCREEN_HEIGHT)
    try:
        user32 = ctypes.windll.user32        # Windows only
        try:
            user32.SetProcessDPIAware()      # report real pixels on scaled (125%, 150%) displays
        except Exception:
            pass
        width, height = int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
        if width > 0 and height > 0:
            return width, height
    except Exception:
        pass
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        width, height = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return int(width), int(height)
    except Exception:
        return 1920, 1080


# --------------------------------- features ------------------------------
def gaze_features(state):
    """The six numbers [hx, hy, yaw, pitch, nose_x, nose_y] for this frame, or None when the
    eyes cannot be used (no face, eyes shut or covered, head turned too far)."""
    if state.pts is None or state.frame is None or state.yaw is None or state.left_eye is None:
        return None
    if state.eyes_closed or state.eyes_covered or abs(state.yaw) > cfg.GAZE_MAX_YAW:
        return None
    pts = state.pts
    height, width = state.frame.shape[:2]
    rows = []
    for side, eye in (("LEFT", state.left_eye), ("RIGHT", state.right_eye)):
        if state.skipped_eye == side or side in state.covered_eyes:
            continue  # an eye that is shut, hidden or covered says nothing about the gaze
        iris, a, b = eye[0], eye[1], eye[2]
        corner_a, corner_b = pts[a], pts[b]
        if corner_a[0] > corner_b[0]:
            corner_a, corner_b = corner_b, corner_a
        axis = corner_b - corner_a
        eye_width = float(np.hypot(axis[0], axis[1]))
        if eye_width < 4:
            continue
        along = axis / eye_width                      # unit vector along the eye
        across = np.array([-along[1], along[0]])      # unit vector pointing down the face
        offset = pts[iris] - (corner_a + corner_b) / 2
        rows.append((float(offset @ along) / eye_width, float(offset @ across) / eye_width))
    if not rows:
        return None
    hx, hy = np.mean(rows, axis=0)
    nose = pts[1]
    return np.array([hx, hy, state.yaw, state.pitch, nose[0] / width, nose[1] / height])


def head_signature(state):
    """Where the head sits in the camera picture: [nose x, nose y (both as a fraction of the
    picture width) and the distance between the two irises (the same unit, so it changes when
    the person leans closer or further)]. None when there is no face to measure."""
    if state.pts is None or state.frame is None or state.left_eye is None or state.right_eye is None:
        return None
    pts = state.pts
    width = state.frame.shape[1]
    gap = float(np.hypot(*(pts[state.left_eye[0]] - pts[state.right_eye[0]])))
    return np.array([pts[1][0] / width, pts[1][1] / width, gap / width])


def light_balance(state):
    """How much brighter one side of the face is than the other, 0 (even) to 1, or None."""
    if state.pts is None or state.frame is None:
        return None
    height, width = state.frame.shape[:2]
    x0, y0 = state.pts.min(axis=0)
    x1, y1 = state.pts.max(axis=0)
    x0, x1 = int(max(x0, 0)), int(min(x1, width))
    y0, y1 = int(max(y0, 0)), int(min(y1, height))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None
    gray = cv2.cvtColor(state.frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    mid = gray.shape[1] // 2
    left, right = float(gray[:, :mid].mean()), float(gray[:, mid:].mean())
    return abs(left - right) / max(left + right, 1.0)


# ---------------------------------- mapper -------------------------------
class GazeMapper:
    """Learns features -> screen point (each 0..1 across the screen) and applies it."""

    def __init__(self, ridge=None):
        self.ridge = cfg.GAZE_RIDGE if ridge is None else ridge
        self.mean = None
        self.coef = None
        self.error_px = None     # typical error from the cross-check, in screen pixels
        self.check_px = None     # error on dots it was NOT trained on (accuracy check), in pixels
        self.screen = None       # (width, height) the mapping was learned for
        self.head_ref = None     # head_signature() while calibrating: for the "head moved" note

    @staticmethod
    def expand(features):
        """Fixed scaling so every column is roughly -1..1; the eye numbers also get the
        curved terms, the head numbers only a straight line."""
        hx, hy, yaw, pitch, nose_x, nose_y = np.atleast_2d(features).T
        sx, sy = hx * 10.0, hy * 10.0
        return np.column_stack([sx, sy, sx * sy, sx ** 2, sy ** 2,
                                yaw / 10.0, pitch / 10.0, (nose_x - 0.5) * 4.0, (nose_y - 0.5) * 4.0])

    def fit(self, features, targets):
        x = self.expand(features)
        self.mean = x.mean(axis=0)
        z = np.column_stack([np.ones(len(x)), x - self.mean])
        # Ridge penalty per column (it keeps noisy columns from being over-trusted): none on the
        # constant, almost none on the two straight eye terms that carry the signal, and
        # `ridge` on the curved eye terms and the head numbers.
        weights = np.array([0.0, EYE_LINEAR_RIDGE, EYE_LINEAR_RIDGE] + [self.ridge] * (z.shape[1] - 3))
        self.coef = np.linalg.solve(z.T @ z + np.diag(weights * len(x)), z.T @ np.asarray(targets, dtype=float))
        return self

    def predict(self, features):
        """(N, 2) array of screen points in 0..1 units (can fall outside 0..1)."""
        x = self.expand(features) - self.mean
        return np.column_stack([np.ones(len(x)), x]) @ self.coef

    def to_dict(self):
        return {"version": 1, "created": datetime.now().isoformat(timespec="seconds"),
                "screen": list(self.screen) if self.screen else None, "error_px": self.error_px,
                "check_px": self.check_px,
                "head_ref": None if self.head_ref is None else [float(v) for v in self.head_ref],
                "ridge": self.ridge, "mean": self.mean.tolist(), "coef": self.coef.tolist()}

    def summary(self):
        """One line about how good the calibration was, for the gaze pad."""
        parts = []
        if self.check_px is not None:
            parts.append(f"accuracy check on unseen dots about {self.check_px:.0f} px")
        if self.error_px is not None:
            parts.append(f"cross-check {self.error_px:.0f} px")
        return ", ".join(parts)

    @classmethod
    def from_dict(cls, data):
        mapper = cls(data.get("ridge"))
        mapper.mean = np.array(data["mean"], dtype=float)
        mapper.coef = np.array(data["coef"], dtype=float)
        mapper.error_px = data.get("error_px")
        mapper.check_px = data.get("check_px")
        mapper.screen = tuple(data["screen"]) if data.get("screen") else None
        if data.get("head_ref"):
            mapper.head_ref = np.array(data["head_ref"], dtype=float)
        return mapper


def cross_error(features, targets, groups, width, height, ridge=None):
    """Predict each calibration dot from all the others. Returns the median error in pixels
    (and the list). Corner dots have to be extrapolated, so this is a cautious estimate."""
    features, targets, groups = np.asarray(features), np.asarray(targets), np.asarray(groups)
    errors = []
    for group in np.unique(groups):
        train = groups != group
        mapper = GazeMapper(ridge).fit(features[train], targets[train])
        guess = mapper.predict(features[~train]).mean(axis=0)
        goal = targets[~train][0]
        errors.append(float(np.hypot((guess[0] - goal[0]) * width, (guess[1] - goal[1]) * height)))
    return float(np.median(errors)), errors


def save_calibration(mapper, width, height, path=None):
    path = path or cfg.GAZE_CAL_FILE
    mapper.screen = (width, height)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapper.to_dict(), f, indent=1)


def load_calibration(width, height, path=None):
    """The saved mapping, or None (with a printed reason) if there is none for this screen."""
    path = path or cfg.GAZE_CAL_FILE
    if not os.path.exists(path):
        print("No saved calibration yet: calibrating now.")
        return None
    try:
        with open(path, encoding="utf-8") as f:
            mapper = GazeMapper.from_dict(json.load(f))
    except (OSError, ValueError, KeyError):
        print("The saved calibration could not be read: calibrating now.")
        return None
    if mapper.screen != (width, height):
        print(f"The saved calibration was made for a {mapper.screen} screen, this one is "
              f"{(width, height)}: calibrating now.")
        return None
    return mapper


# --------------------------------- smoothing -----------------------------
class OneEuro:
    """One Euro filter: steady when the eyes rest, quick when they jump."""

    def __init__(self, min_cutoff, beta, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, value, t):
        if self.x is None or t <= self.t:
            self.x, self.dx, self.t = value, 0.0, t
            return value
        dt = t - self.t
        speed = (value - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * speed + (1 - a_d) * self.dx
        a = self._alpha(self.min_cutoff + self.beta * abs(self.dx), dt)
        self.x = a * value + (1 - a) * self.x
        self.t = t
        return self.x


class GazeTracker:
    """Turns each FrameState into a smoothed screen point and an off-screen verdict."""

    def __init__(self, mapper, width, height):
        self.mapper, self.width, self.height = mapper, width, height
        self.fx = OneEuro(cfg.GAZE_SMOOTH_MIN_CUTOFF, cfg.GAZE_SMOOTH_BETA)
        self.fy = OneEuro(cfg.GAZE_SMOOTH_MIN_CUTOFF, cfg.GAZE_SMOOTH_BETA)
        self._last_t = None
        self._recent = deque(maxlen=max(1, cfg.GAZE_FEATURE_FRAMES))
        self.offset = (0.0, 0.0)           # pixels added by the z re-centre
        self.head_ref = None if mapper.head_ref is None else np.array(mapper.head_ref)
        self.head_shift = None             # smoothed: how far the head moved since calibration
        self.head_scale = None             # smoothed: iris distance now / at calibration
        self.balance = None                # smoothed: left/right light difference of the face
        self.features = None               # the (median) features used on the last frame, for --debug

    def set_offset(self, dx, dy, head=None):
        """Shift every gaze point by (dx, dy) pixels (the z re-centre). `head` is the head
        signature at the moment of re-centring, which becomes the new "where I sit" reference."""
        self.offset = (float(dx), float(dy))
        if head is not None:
            self.head_ref = np.array(head)
        self.fx.reset()
        self.fy.reset()
        self._recent.clear()
        self.head_shift = self.head_scale = None

    def _watch_head_and_light(self, state):
        """Slowly updated numbers behind the notes on the pad."""
        def blend(old, new):
            return new if old is None else 0.9 * old + 0.1 * new
        head = head_signature(state)
        if head is not None and self.head_ref is not None:
            shift = float(np.hypot(head[0] - self.head_ref[0], head[1] - self.head_ref[1]))
            self.head_shift = blend(self.head_shift, shift)
            self.head_scale = blend(self.head_scale, float(head[2] / max(self.head_ref[2], 1e-6)))
        balance = light_balance(state)
        if balance is not None:
            self.balance = blend(self.balance, balance)

    @property
    def notes(self):
        """Short plain-language reasons the dot may be off right now (shown on the pad)."""
        notes = []
        if self.head_shift is not None and self.head_shift > cfg.GAZE_HEAD_MOVE_WARN:
            notes.append("Your head has moved since calibration: sit as you did, or press z / c")
        elif self.head_scale is not None and abs(self.head_scale - 1.0) > cfg.GAZE_HEAD_SCALE_WARN:
            notes.append("You are " + ("closer to" if self.head_scale > 1 else "further from")
                         + " the camera than at calibration: sit as you did, or press z / c")
        if self.balance is not None and self.balance > cfg.GAZE_LIGHT_BALANCE_WARN:
            notes.append("Light on your face is uneven (one side brighter): face the light")
        return notes

    def update(self, state, now):
        """Fill state.gaze_x / gaze_y / gaze_off. Returns (x, y) or None when there is no gaze."""
        self._watch_head_and_light(state)
        features = gaze_features(state)
        self.features = None
        if features is None:
            state.gaze_off = False    # unknown, not "off screen": other flags cover shut eyes etc.
            return None
        if self._last_t is not None and now - self._last_t > 1.0:
            self.fx.reset()           # a gap: do not glide in from the old position
            self.fy.reset()
            self._recent.clear()
        self._last_t = now
        # The middle value of the last few frames removes one-frame glitches (a bad iris
        # landmark in poor light) without the lag of heavy smoothing.
        self._recent.append(features)
        features = np.median(np.array(self._recent), axis=0)
        self.features = features
        nx, ny = self.mapper.predict(features)[0]
        x = self.fx(nx * self.width + self.offset[0], now)
        y = self.fy(ny * self.height + self.offset[1], now)
        state.gaze_x, state.gaze_y = float(x), float(y)
        margin_x, margin_y = cfg.GAZE_OFF_MARGIN * self.width, cfg.GAZE_OFF_MARGIN * self.height
        state.gaze_off = bool(x < -margin_x or x > self.width + margin_x
                              or y < -margin_y or y > self.height + margin_y)
        if state.gaze_off:
            state.details["GAZE_OFF_SCREEN"] = (
                f"gaze at ({x:.0f}, {y:.0f}) px on a {self.width}x{self.height} screen")
        return x, y


# ---------------------------- optional: real mouse -----------------------
class MouseFollower:
    """Moves the real mouse pointer to the gaze point (Windows only). Toggle with m."""

    def __init__(self):
        self.on = False
        self._warned = False

    def move(self, x, y):
        if not self.on:
            return
        try:
            ctypes.windll.user32.SetCursorPos(int(x), int(y))
        except Exception:
            if not self._warned:
                print("Moving the real mouse pointer is only supported on Windows.")
                self._warned = True
            self.on = False


# --------------------------------- drawing -------------------------------
def _put(canvas, text, x, y, scale=0.8, color=WHITE):
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def _centered(canvas, text, y, scale=0.9, color=WHITE):
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0]
    _put(canvas, text, max((canvas.shape[1] - size[0]) // 2, 0), y, scale, color)


def thumbnail(frame, state, width=320):
    """Small camera picture with the iris dots, for a corner of the full-screen window."""
    height = int(frame.shape[0] * width / frame.shape[1])
    small = cv2.resize(frame, (width, height))
    if state is not None and state.pts is not None:
        scale = width / frame.shape[1]
        for eye in (state.left_eye, state.right_eye):
            cv2.circle(small, tuple((state.pts[eye[0]] * scale).astype(int)), 3, (0, 255, 0), -1)
    return small


def _paste_corner(canvas, small):
    h, w = small.shape[:2]
    H, W = canvas.shape[:2]
    canvas[H - h - 8:H - 8, W - w - 8:W - 8] = small
    cv2.rectangle(canvas, (W - w - 8, H - h - 8), (W - 8, H - 8), (90, 90, 90), 1)


def draw_pad(width, height, state, small, active_flags, status_lines, trail, mouse_on, error_text,
             notes=()):
    """The full-screen gaze pad: faint grid, the gaze dot with a tail, readouts, flag banner."""
    canvas = np.full((height, width, 3), 18, np.uint8)
    for i in range(1, 8):
        cv2.line(canvas, (i * width // 8, 0), (i * width // 8, height), (32, 32, 32), 1)
    for i in range(1, 6):
        cv2.line(canvas, (0, i * height // 6), (width, i * height // 6), (32, 32, 32), 1)
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (70, 70, 70), 2)
    _paste_corner(canvas, small)   # first, so the gaze dot and tail are drawn on top of it

    have_gaze = state.gaze_x is not None and state.zone[0] not in ("CLOSED", "COVERED")
    if have_gaze:
        x = int(np.clip(state.gaze_x, 0, width - 1))
        y = int(np.clip(state.gaze_y, 0, height - 1))
        trail.append((x, y))
    points = list(trail)
    for i in range(1, len(points)):
        shade = int(40 + 150 * i / len(points))
        cv2.line(canvas, points[i - 1], points[i], (shade, shade, 0), 2)
    if have_gaze:
        color = (0, 0, 255) if state.gaze_off else (0, 220, 0)
        cv2.circle(canvas, (x, y), 26, color, 3)
        cv2.circle(canvas, (x, y), 8, color, -1)

    lines = [
        "Gaze pad   q = quit   c = recalibrate   z = re-centre   m = real mouse "
        + ("(ON)" if mouse_on else "(off)"),
        (f"Gaze: ({state.gaze_x:.0f}, {state.gaze_y:.0f}) px" + ("   OFF SCREEN" if state.gaze_off else "")
         if state.gaze_x is not None else "Gaze: -   (eyes shut, covered or head turned far)"),
        "Looking: " + (", ".join(sorted(state.looking)) if state.looking else "CENTER"),
    ]
    if error_text:
        lines.append("Calibration: " + error_text)
    lines.extend(status_lines)
    for i, text in enumerate(lines):
        _put(canvas, text, 20, 36 + 32 * i, 0.7)
    for j, text in enumerate(notes):   # why the dot may be off right now (yellow)
        _put(canvas, text, 20, 36 + 32 * len(lines) + 32 * j, 0.7, DOT_COLOR)

    if active_flags:
        cv2.rectangle(canvas, (0, height - 52), (width, height), (0, 0, 200), -1)
        _put(canvas, "FLAG: " + ", ".join(active_flags), 20, height - 16, 0.9)
    return canvas


# -------------------------------- calibration ----------------------------
class CalibrationAborted(Exception):
    """Escape pressed, or the camera stopped, during calibration."""


def cal_points():
    """The nine dot positions as (x, y) fractions of the screen, centre first."""
    m = cfg.GAZE_CAL_EDGE
    lo, mid, hi = m, 0.5, 1.0 - m
    return [(mid, mid), (lo, lo), (hi, lo), (hi, hi), (lo, hi), (mid, lo), (hi, mid), (mid, hi), (lo, mid)]


DISPLAY = None   # set by the exam app: a function(canvas) -> key (0 = none) that shows the screen itself


def _show(win, canvas):
    if DISPLAY is not None:
        key = DISPLAY(canvas) or 0
    else:
        cv2.imshow(win, canvas)
        key = cv2.waitKey(1) & 0xFF
    if key in (27, ord("q")):
        raise CalibrationAborted()
    return key


def _next(step):
    out = step()
    if out is None:
        raise CalibrationAborted()
    return out


def validation_points():
    """Five dots in places the calibration did not use, to test how good it really is."""
    return [(0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7), (0.5, 0.78)]


def _collect_dot(step, win, width, height, point, label, lumas=None, heads=None):
    """Show one dot; return the good feature rows recorded while the person looks at it."""
    settle, collect = cfg.GAZE_CAL_SETTLE_SECONDS, cfg.GAZE_CAL_COLLECT_SECONDS
    cx, cy = int(point[0] * width), int(point[1] * height)
    start, rows = None, []
    while True:
        frame, _, state, now = _next(step)
        start = now if start is None else start
        elapsed = now - start
        if elapsed >= settle + collect:
            break
        if elapsed >= settle:
            features = gaze_features(state)
            if features is not None:
                rows.append(features)
                if lumas is not None and state.luma is not None:
                    lumas.append(state.luma)
                if heads is not None and head_signature(state) is not None:
                    heads.append(head_signature(state))
        progress = min(elapsed / (settle + collect), 1.0)
        canvas = np.full((height, width, 3), 18, np.uint8)
        cv2.circle(canvas, (cx, cy), int(34 - 22 * progress), (90, 90, 90), 2)      # shrinking ring
        cv2.circle(canvas, (cx, cy), 9, DOT_COLOR, -1)                              # the dot
        # Text on the half of the screen away from the dot, and no camera picture here: nothing
        # may ever cover a dot, including the ones in the corners.
        text_y = height - 30 if cy < height // 2 else 40
        note = "" if gaze_features(state) is not None else "   (cannot see your eyes clearly)"
        _put(canvas, f"{label}.  Keep your head still.{note}", 20, text_y, 0.8)
        _show(win, canvas)
    return _drop_blinks(rows)


def _drop_blinks(rows):
    """Remove frames whose eye numbers are far from the typical ones at this dot (blinks, jumps)."""
    if len(rows) < 5:
        return rows
    arr = np.array(rows)
    centre = np.median(arr[:, :2], axis=0)
    dist = np.hypot(*(arr[:, :2] - centre).T)
    limit = 3.5 * np.median(dist) + 0.01
    return [row for row, d in zip(rows, dist) if d <= limit]


def _gather(step, win, width, height, point, label, lumas, heads=None):
    """Collect one dot, showing it again if too few good frames were seen."""
    rows = []
    for _ in range(1 + cfg.GAZE_CAL_RETRIES):
        rows = _collect_dot(step, win, width, height, point, label, lumas, heads)
        if len(rows) >= cfg.GAZE_CAL_MIN_SAMPLES:
            break
    return rows if len(rows) >= cfg.GAZE_CAL_MIN_SAMPLES else []


def _fit_all(data, width, height):
    """Fit on every collected dot. `data` maps dot key -> (point, rows).
    Returns (mapper, report rows, features, targets, groups)."""
    features, targets, groups = [], [], []
    for key in sorted(data):
        point, rows = data[key]
        features += rows
        targets += [point] * len(rows)
        groups += [key] * len(rows)
    mapper = GazeMapper().fit(features, targets)
    mapper.screen = (width, height)
    mapper.error_px, cross = cross_error(features, targets, groups, width, height)
    report = dot_report(mapper, features, targets, groups, width, height, cross)
    return mapper, report, features, targets, groups


def dot_label(key):
    return str(key + 1) if key < 100 else f"check {key - 99}"


def dot_report(mapper, features, targets, groups, width, height, cross):
    """One row per dot: where it was, frames kept, scatter (how much the eye reading shakes at that
    dot, in screen pixels: the lighting/noise measure), fit error and cross-check error."""
    features, targets, groups = np.asarray(features), np.asarray(targets), np.asarray(groups)
    rows = []
    for k, key in enumerate(np.unique(groups)):
        sel = groups == key
        predicted = mapper.predict(features[sel]) * (width, height)
        goal = targets[sel][0] * (width, height)
        rows.append({
            "dot": dot_label(int(key)), "key": int(key),
            "target_px": [int(round(goal[0])), int(round(goal[1]))],
            "frames": int(sel.sum()),
            "scatter_px": int(round(float(np.hypot(*predicted.std(axis=0))))),
            "fit_error_px": int(round(float(np.hypot(*(predicted.mean(axis=0) - goal))))),
            "cross_error_px": int(round(cross[k])),
        })
    return rows


def unsteady_dots(report, width, height):
    """Keys of the dots whose eye reading shook much more than the others (usually bad light)."""
    scatters = [r["scatter_px"] for r in report if r["key"] < 100]
    if not scatters:
        return []
    limit = max(1.8 * float(np.median(scatters)), 0.04 * float(np.hypot(width, height)))
    return [r["key"] for r in report if r["key"] < 100 and r["scatter_px"] > limit]


def print_report(report, unsteady):
    print("Dot     target (px)   frames  scatter  fit-err  cross-err")
    for r in report:
        flag = "   <- unsteady" if r["key"] in unsteady else ""
        print(f"{r['dot']:<7} ({r['target_px'][0]:>5},{r['target_px'][1]:>5})  {r['frames']:>5}   "
              f"{r['scatter_px']:>5}px  {r['fit_error_px']:>5}px  {r['cross_error_px']:>6}px{flag}")


def save_debug(mapper, report, unsteady, redone, luma, width, height):
    """Write everything about this calibration to a small file that can be shared for debugging."""
    try:
        os.makedirs(os.path.dirname(cfg.GAZE_CAL_DEBUG_FILE), exist_ok=True)
        with open(cfg.GAZE_CAL_DEBUG_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "created": datetime.now().isoformat(timespec="seconds"),
                "screen": [width, height],
                "face_light_mean": None if luma is None else round(luma, 1),
                "cross_check_px": mapper.error_px, "accuracy_check_px": mapper.check_px,
                "dots_redone": redone, "dots_unsteady_at_end": unsteady,
                "settings": {k: getattr(cfg, k) for k in (
                    "GAZE_RIDGE", "GAZE_CAL_EDGE", "GAZE_CAL_SETTLE_SECONDS", "GAZE_CAL_COLLECT_SECONDS",
                    "GAZE_FEATURE_FRAMES", "GAZE_SMOOTH_MIN_CUTOFF", "GAZE_SMOOTH_BETA",
                    "ENHANCE_LOW_LIGHT", "ENHANCE_CONTRAST", "TARGET_LUMA")},
                "dots": report}, f, indent=1)
        print(f"Calibration details saved to {cfg.GAZE_CAL_DEBUG_FILE}")
    except OSError:
        pass


def recentre(step, win, width, height, mapper, seconds=None):
    """The z key. The person looks at the middle dot; the mapping is shifted so that the middle
    of the screen is where it says they look. Fixes a dot that sits a constant distance off
    (the person shifted a little, or the light changed). Returns (dx, dy, head signature) or
    None if it could not be done (face not seen, Esc)."""
    saved = cfg.GAZE_CAL_COLLECT_SECONDS
    cfg.GAZE_CAL_COLLECT_SECONDS = cfg.GAZE_RECENTRE_SECONDS if seconds is None else seconds
    heads = []
    try:
        rows = _gather(step, win, width, height, (0.5, 0.5), "Re-centre: look at the middle dot, head still",
                       None, heads)
    except CalibrationAborted:
        return None
    finally:
        cfg.GAZE_CAL_COLLECT_SECONDS = saved
    if not rows or not heads:
        print("Re-centre failed: could not see your eyes clearly. Try again.")
        return None
    guess = mapper.predict(rows).mean(axis=0)
    dx, dy = (0.5 - guess[0]) * width, (0.5 - guess[1]) * height
    print(f"Re-centred: moved the dot by ({dx:+.0f}, {dy:+.0f}) px")
    return dx, dy, np.median(np.array(heads), axis=0)


def run_calibration(step, win, width, height):
    """Run the whole calibration. `step()` returns (frame, raw, state, now) for the next camera
    frame, or None when the camera stops. Returns a fitted GazeMapper, or None if abandoned."""
    try:
        return _run_calibration(step, win, width, height)
    except CalibrationAborted:
        print("Calibration cancelled.")
        return None


def _message_screen(step, win, width, height, lines, seconds, color=WHITE):
    """Show a few lines of text for `seconds` (camera keeps running)."""
    since = None
    while True:
        _, _, _, now = _next(step)
        since = now if since is None else since
        if now - since > seconds:
            return
        canvas = np.full((height, width, 3), 18, np.uint8)
        for i, text in enumerate(lines):
            _centered(canvas, text, height // 2 - 20 + 44 * i, 1.0 if i == 0 else 0.8,
                      color if i == 0 else WHITE)
        _show(win, canvas)


def _intro(step, win, width, height):
    """Introduction: time to sit comfortably, with a light check. SPACE starts at once."""
    start = None
    while True:
        frame, _, state, now = _next(step)
        start = now if start is None else start
        left = cfg.GAZE_CAL_START_SECONDS - (now - start)
        canvas = np.full((height, width, 3), 18, np.uint8)
        _centered(canvas, "Gaze calibration", height // 2 - 110, 1.3)
        _centered(canvas, "Sit the way you will sit during the exam, with your face in the camera picture.",
                  height // 2 - 50, 0.8)
        _centered(canvas, "Dots will appear one by one. Look straight at each dot until it moves.",
                  height // 2 - 10, 0.8)
        _centered(canvas, "Move only your eyes as much as you can. Keep your head still.",
                  height // 2 + 30, 0.8)
        if state.luma is None:
            light, light_color = "Light on your face: no face found yet", (0, 0, 255)
        elif state.luma < cfg.LOW_LIGHT_WARN:
            light, light_color = (f"Light on your face: {state.luma:.0f}  TOO DARK. Put a lamp or the screen "
                                  "light in FRONT of you, not behind", (0, 0, 255))
        else:
            light, light_color = f"Light on your face: {state.luma:.0f}  OK (even light, nothing bright behind you)", (0, 220, 0)
        _centered(canvas, light, height // 2 + 80, 0.8, light_color)
        _centered(canvas, f"Starting in {max(left, 0):.0f} s   (SPACE to start now, Esc to cancel)",
                  height // 2 + 130, 0.8, DOT_COLOR)
        _paste_corner(canvas, thumbnail(frame, state))
        key = _show(win, canvas)
        if key == 32 or left <= 0:
            return


def _run_calibration(step, win, width, height):
    _intro(step, win, width, height)
    points = cal_points()
    lumas, heads = [], []
    data = {}
    for index, point in enumerate(points):
        rows = _gather(step, win, width, height, point, f"Look at the dot  ({index + 1} of {len(points)})",
                       lumas, heads)
        if rows:
            data[index] = (point, rows)
        else:
            print(f"Calibration dot {index + 1}: too few good frames (face lost or eyes shut), skipped.")
    if len(data) < 6:
        print("Calibration failed: too many dots had no clear view of the eyes. "
              "Check the light and your position, then try again.")
        return None

    mapper, report, *_ = _fit_all(data, width, height)

    # Dots where the eye reading shook much more than at the others: do those again once.
    redone = []
    bad = unsteady_dots(report, width, height) if cfg.GAZE_CAL_REDO_BAD else []
    if bad:
        print("Unsteady eye reading at dot(s) " + ", ".join(str(k + 1) for k in bad) + ": doing them again.")
        _message_screen(step, win, width, height,
                        ["Some dots were unsteady (often poor light)",
                         "Redoing dot(s) " + ", ".join(str(k + 1) for k in bad) + ". Look straight at each dot."],
                        2.5, (0, 200, 255))
        for key in bad:
            point = points[key]
            rows = _gather(step, win, width, height, point, f"Again: dot {key + 1}", lumas, heads)
            if not rows:
                continue
            old_scatter = next(r["scatter_px"] for r in report if r["key"] == key)
            new_pred = mapper.predict(rows) * (width, height)
            if float(np.hypot(*new_pred.std(axis=0))) < old_scatter:   # keep whichever shook less
                data[key] = (point, rows)
                redone.append(key + 1)
        mapper, report, *_ = _fit_all(data, width, height)

    # Accuracy check on dots the fit has never seen; their frames are then added to the fit.
    check_errors = []
    if cfg.GAZE_VALIDATE:
        vpoints = validation_points()
        for k, point in enumerate(vpoints):
            rows = _gather(step, win, width, height, point,
                           f"Accuracy check ({k + 1} of {len(vpoints)}): look at the dot", lumas, heads)
            if not rows:
                continue
            guess = mapper.predict(rows).mean(axis=0)
            check_errors.append(float(np.hypot((guess[0] - point[0]) * width, (guess[1] - point[1]) * height)))
            data[100 + k] = (point, rows)
    check_px = float(np.mean(check_errors)) if check_errors else None
    if check_errors:
        mapper, report, *_ = _fit_all(data, width, height)
    mapper.check_px = check_px
    if heads:
        mapper.head_ref = np.median(np.array(heads), axis=0)   # where the head sat while calibrating

    unsteady = unsteady_dots(report, width, height)
    print_report(report, unsteady)
    diagonal = float(np.hypot(width, height))
    headline = check_px if check_px is not None else mapper.error_px
    warn = headline > cfg.GAZE_CAL_WARN_FRACTION * diagonal
    print(f"Calibration done: {mapper.summary()} ({100 * headline / diagonal:.0f}% of the screen diagonal).")
    if warn:
        print("That is large. Recalibrate (press c) with a steadier head and better light on your face.")
    save_debug(mapper, report, unsteady, redone, float(np.mean(lumas)) if lumas else None, width, height)

    kind = "Accuracy check" if check_px is not None else "Typical error"
    lines = [f"{kind}: about {headline:.0f} px" + ("   (large: press c to redo)" if warn else ""),
             f"Dots redone: {', '.join(map(str, redone))}" if redone else "All dots were steady."]
    if unsteady:
        lines.append("Still unsteady: dot(s) " + ", ".join(str(k + 1) for k in unsteady if k < 100)
                     + ".  Try more even light on your face.")
    _message_screen(step, win, width, height, ["Calibration finished"] + lines, 3.0,
                    (0, 0, 255) if warn else (0, 220, 0))
    return mapper


# ------------------------------- debug log -------------------------------
class GazeLog:
    """--debug: one row per camera frame to cfg.GAZE_LOG_FILE (numbers only, no pictures), so a
    drifting or jumping dot can be studied afterwards, e.g. in a spreadsheet."""

    COLUMNS = ["time", "hx", "hy", "yaw", "pitch", "nose_x", "nose_y", "gaze_x", "gaze_y", "off_screen",
               "light", "ear_left", "ear_right", "head_shift", "head_scale", "light_balance", "offset_x", "offset_y"]

    def __init__(self, path=None):
        import csv
        self.path = path or cfg.GAZE_LOG_FILE
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._csv = csv.writer(self._file)
        self._csv.writerow(self.COLUMNS)
        self._t0 = None
        self._rows = 0

    @staticmethod
    def _n(value, digits=4):
        return "" if value is None or value != value else round(float(value), digits)

    def add(self, state, tracker, now):
        self._t0 = now if self._t0 is None else self._t0
        f = tracker.features if tracker.features is not None else [None] * 6
        self._csv.writerow([
            self._n(now - self._t0, 3), self._n(f[0]), self._n(f[1]), self._n(f[2], 2), self._n(f[3], 2),
            self._n(f[4]), self._n(f[5]), self._n(state.gaze_x, 1), self._n(state.gaze_y, 1),
            int(bool(state.gaze_off)), self._n(state.luma, 1), self._n(state.ear_l, 3), self._n(state.ear_r, 3),
            self._n(tracker.head_shift), self._n(tracker.head_scale), self._n(tracker.balance),
            self._n(tracker.offset[0], 1), self._n(tracker.offset[1], 1)])
        self._rows += 1
        if self._rows % 30 == 0:
            self._file.flush()

    def close(self):
        try:
            self._file.close()
        except OSError:
            pass
