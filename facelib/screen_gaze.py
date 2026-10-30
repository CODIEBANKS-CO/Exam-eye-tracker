"""
Gaze mapped to the screen: where on the computer screen is the person looking?

How it works (the same idea as an eye-tracker's calibration, kept simple):

  1. CALIBRATION. The person looks at a few dots shown on the screen (a 3x3 grid). For every
     dot we record, over about a second, how the eyes look to the camera: where each iris
     sits inside its eye, plus the head angles and position (gaze_features below).
  2. LEARNING. A small ridge regression learns the link "eye measurements -> screen x, y"
     from those samples (GazeMapper). It is tested by leaving each dot out in turn and
     predicting it, which gives an honest error figure.
  3. TRACKING. Every frame the same measurements go through the learned link, the result is
     smoothed (a One Euro filter) and drawn as a dot, like a touchpad cursor.

Nothing here touches the camera: GazeSession gets the FrameState of each frame (from
analysis.py), changes it by adding gaze_xy / gaze_cell / gaze_off_screen, and draws the
full-screen picture. gaze_screen.py runs the camera loop around it.

Honest limits: a webcam sees an iris only about 10-15 pixels wide, so the dot is approximate
(a few centimetres at best) and drifts if the person moves their head or the seat. Use it for
"which part of the screen / is the person looking at the screen", not for pinpointing words.
"""

import json
import os
from collections import deque
from datetime import datetime

import cv2
import numpy as np

from . import config as cfg
from .checks import Check

ROLL_IDS = (33, 263)   # outer eye corners: their tilt is the head roll
NOSE_ID = 1
FEATURE_NAMES = ["dx", "dy", "yaw", "pitch", "nose_x", "nose_y"]
RIDGE_CHOICES = (1e-3, 1e-2, 1e-1, 1.0, 10.0)


# ------------------------------ screen size ------------------------------
def screen_size():
    """(width, height) of the screen in real pixels."""
    if cfg.SCREEN_WIDTH and cfg.SCREEN_HEIGHT:
        return int(cfg.SCREEN_WIDTH), int(cfg.SCREEN_HEIGHT)
    try:  # Windows: ask for real pixels, not scaled ones
        import ctypes
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
        w, h = ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)
        if w > 0 and h > 0:
            return int(w), int(h)
    except Exception:
        pass
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        if w > 0 and h > 0:
            return int(w), int(h)
    except Exception:
        pass
    print("Could not detect the screen size; assuming 1920x1080. "
          "Set SCREEN_WIDTH and SCREEN_HEIGHT in facelib/config.py if that is wrong.")
    return 1920, 1080


def cell_name(x, y, width, height):
    """Name of the cell of a 3x3 grid over the screen that contains the point."""
    col = min(2, max(0, int(3 * x / width)))
    row = min(2, max(0, int(3 * y / height)))
    if row == 1 and col == 1:
        return "CENTER"
    return f'{("TOP", "MIDDLE", "BOTTOM")[row]}-{("LEFT", "CENTER", "RIGHT")[col]}'.replace("MIDDLE-", "").replace("-CENTER", "") \
        if (row == 1 or col == 1) else f'{("TOP", "MIDDLE", "BOTTOM")[row]}-{("LEFT", "CENTER", "RIGHT")[col]}'


# ------------------------------- the features ----------------------------
def gaze_features(state):
    """The numbers the mapping is learned from, or None when this frame cannot be used.

    [dx, dy]      where the irises sit inside the eyes, in units of eye width, measured from
                  the middle of each eye's corners and turned upright (so head roll cancels);
                  averaged over both eyes
    [yaw, pitch]  head turn and tilt in units of 30 degrees
    [nose_x, nose_y]  where the nose is in the picture (the head moving sideways or up/down)
    """
    if state.n_faces != 1 or state.pts is None or state.left_eye is None or state.yaw is None:
        return None
    if state.eyes_closed or state.eyes_covered or state.hidden_eye or state.far_eye:
        return None
    if min(state.ear_l, state.ear_r) < cfg.EAR_CLOSED * 1.4:  # a half-blink bends the iris reading
        return None
    pts = state.pts
    height, width = state.frame.shape[:2]

    a, b = sorted(ROLL_IDS, key=lambda i: pts[i][0])
    roll = np.arctan2(pts[b][1] - pts[a][1], pts[b][0] - pts[a][0])
    cos, sin = np.cos(-roll), np.sin(-roll)

    offsets = []
    for eye in (state.left_eye, state.right_eye):
        iris, corner_a, corner_b = eye[0], eye[1], eye[2]
        centre = (pts[corner_a] + pts[corner_b]) / 2
        span = np.linalg.norm(pts[corner_a] - pts[corner_b])
        if span < 1e-6:
            return None
        d = (pts[iris] - centre) / span
        offsets.append((cos * d[0] - sin * d[1], sin * d[0] + cos * d[1]))
    dx, dy = np.mean(offsets, axis=0)
    return np.array([dx, dy, state.yaw / 30.0, state.pitch / 30.0,
                     pts[NOSE_ID][0] / width, pts[NOSE_ID][1] / height], dtype=np.float64)


def _design(features):
    """Columns the regression uses: 1, dx, dy, dx^2, dx*dy, dy^2, yaw, pitch, nose_x, nose_y."""
    f = np.atleast_2d(features)
    dx, dy = f[:, 0], f[:, 1]
    return np.column_stack([np.ones(len(f)), dx, dy, dx * dx, dx * dy, dy * dy,
                            f[:, 2], f[:, 3], f[:, 4], f[:, 5]])


# ------------------------------- the mapping -----------------------------
def _ridge(x, y, lam):
    penalty = lam * np.eye(x.shape[1])
    penalty[0, 0] = 0.0  # never shrink the constant term
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


class GazeMapper:
    """Learns and applies the link from eye measurements to screen pixels."""

    def __init__(self, width, height):
        self.width, self.height = int(width), int(height)
        self.weights = None
        self.mean = None
        self.std = None
        self.lam = None
        self.cv_error_px = None     # error on a dot that was left out of the fit (honest figure)
        self.offset = np.zeros(2)   # re-centering correction (press z while looking at the middle)

    # fitting
    def fit(self, features, targets_px, groups):
        """features (n, 6), targets_px (n, 2) screen pixels, groups (n,) which dot each row is.
        Returns the cross-validated error in pixels."""
        features = np.asarray(features, dtype=np.float64)
        scale = np.array([self.width, self.height], dtype=np.float64)
        y = np.asarray(targets_px, dtype=np.float64) / scale
        groups = np.asarray(groups)
        design = _design(features)
        self.mean = design.mean(axis=0)
        self.std = design.std(axis=0)
        self.mean[0], self.std[0] = 0.0, 1.0
        self.std[self.std < 1e-9] = 1.0
        x = (design - self.mean) / self.std

        ids = np.unique(groups)
        if len(ids) < 4:
            raise ValueError("need at least 4 different calibration dots")

        best_lam, best_err = None, None
        for lam in RIDGE_CHOICES:
            errors = []
            for g in ids:
                train, test = groups != g, groups == g
                w = _ridge(x[train], y[train], lam)
                predicted = (x[test] @ w).mean(axis=0) * scale
                errors.append(np.linalg.norm(predicted - y[test].mean(axis=0) * scale))
            err = float(np.mean(errors))
            if best_err is None or err < best_err:
                best_lam, best_err = lam, err
        self.lam, self.cv_error_px = best_lam, best_err
        self.weights = _ridge(x, y, best_lam)
        self.offset = np.zeros(2)
        return best_err

    # using it
    def predict_raw(self, features):
        """Screen pixels (x, y) for one feature row, before the re-centering offset."""
        x = (_design(features) - self.mean) / self.std
        return (x @ self.weights)[0] * np.array([self.width, self.height], dtype=np.float64)

    def predict(self, features):
        return self.predict_raw(features) + self.offset

    # saving
    def save(self, path=None):
        path = path or cfg.GAZE_FILE
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {"version": 1, "created": datetime.now().isoformat(timespec="seconds"),
                "screen": [self.width, self.height], "features": FEATURE_NAMES,
                "weights": self.weights.tolist(), "mean": self.mean.tolist(),
                "std": self.std.tolist(), "lam": self.lam, "cv_error_px": self.cv_error_px}
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, path)

    @classmethod
    def load(cls, width, height, path=None):
        """The saved mapper, or None if there is none or it was made for another screen size."""
        path = path or cfg.GAZE_FILE
        if not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if [int(width), int(height)] != list(data["screen"]):
                return None
            mapper = cls(width, height)
            mapper.weights = np.array(data["weights"])
            mapper.mean, mapper.std = np.array(data["mean"]), np.array(data["std"])
            mapper.lam, mapper.cv_error_px = data.get("lam"), data.get("cv_error_px")
            return mapper
        except (OSError, ValueError, KeyError):
            return None


class OneEuroFilter:
    """Smooths the dot: steady when the eyes rest, quick when they really move."""

    def __init__(self, min_cutoff, beta, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.t = self.x = self.dx = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, value, t):
        value = np.asarray(value, dtype=np.float64)
        if self.t is None:
            self.t, self.x, self.dx = t, value, np.zeros_like(value)
            return value
        dt = max(t - self.t, 1e-3)
        raw_speed = (value - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * raw_speed + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self.dx))
        a = self._alpha(cutoff, dt)
        self.x = a * value + (1 - a) * self.x
        self.t = t
        return self.x


def _trim(rows):
    """Drop frames at one calibration dot that are far from the typical reading
    (a blink, or the eyes still moving to the dot)."""
    if len(rows) < 8:
        return np.ones(len(rows), dtype=bool)
    mid = np.median(rows[:, :2], axis=0)
    mad = np.maximum(np.median(np.abs(rows[:, :2] - mid), axis=0), 1e-3)
    keep = np.all(np.abs(rows[:, :2] - mid) <= 3.5 * 1.4826 * mad, axis=1)
    return keep if keep.sum() >= len(rows) // 2 else np.ones(len(rows), dtype=bool)


# -------------------------------- the session -----------------------------
class GazeSession:
    """Calibration and live tracking for one person, plus the full-screen picture.

    phases:  INTRO (waiting for SPACE) -> CAL (dots) -> LIVE (tracking)
    """

    def __init__(self, width, height, mapper=None):
        self.width, self.height = int(width), int(height)
        self.mapper = mapper
        self.phase = "LIVE" if mapper else "INTRO"
        self.targets = self._make_targets()
        self.idx = 0
        self.attempt = 0
        self.point_start = None
        self.samples = []            # (feature row, target index)
        self.message = ""
        self.show_preview = True
        self.trail = deque(maxlen=18)
        self.recent_raw = deque(maxlen=20)
        self.filter = None
        self.latest = None           # (x, y) of the dot on the last frame, or None

    # ---- calibration targets
    def _make_targets(self):
        cols, rows = cfg.CAL_GRID
        m = cfg.CAL_MARGIN

        def spread(n, size):
            return [size / 2] if n == 1 else [size * (m + (1 - 2 * m) * i / (n - 1)) for i in range(n)]
        xs, ys = spread(cols, self.width), spread(rows, self.height)
        order = []
        for r, y in enumerate(ys):
            row = [(x, y) for x in xs]
            order.extend(row if r % 2 == 0 else row[::-1])  # snake: short jumps between dots
        return order

    @property
    def current_target(self):
        """(x, y) of the dot being shown during calibration, else None."""
        return self.targets[self.idx] if self.phase == "CAL" else None

    @property
    def live(self):
        return self.phase == "LIVE"

    def start_calibration(self, now):
        self.phase, self.idx, self.attempt = "CAL", 0, 0
        self.point_start, self.samples = now, []
        self.message = ""
        self.filter, self.latest = None, None
        self.trail.clear()

    # ---- keys
    def handle_key(self, key, now):
        if key == ord(" ") and self.phase == "INTRO":
            self.start_calibration(now)
        elif key == ord("c"):
            self.start_calibration(now)
        elif key == ord("x") and self.phase == "CAL":
            self.phase = "LIVE" if self.mapper else "INTRO"
            self.message = "Calibration cancelled."
        elif key == ord("z") and self.phase == "LIVE" and self.recent_raw:
            centre = np.array([self.width / 2, self.height / 2])
            self.mapper.offset = centre - np.mean(self.recent_raw, axis=0)
            self.filter, self.trail = None, deque(maxlen=self.trail.maxlen)
            self.message = "Re-centred: the dot now sits in the middle while you look at the middle."
        elif key == ord("d"):
            self.show_preview = not self.show_preview

    # ---- every frame
    def update(self, state, now):
        """Look at this frame; in LIVE fill in state.gaze_*; in CAL collect samples."""
        state.gaze_xy, state.gaze_cell, state.gaze_off_screen = None, None, False
        if self.phase == "CAL":
            self._update_calibration(state, now)
        elif self.phase == "LIVE":
            self._update_live(state, now)

    def _update_calibration(self, state, now):
        elapsed = now - self.point_start
        if cfg.CAL_SETTLE_SECONDS <= elapsed < cfg.CAL_SETTLE_SECONDS + cfg.CAL_CAPTURE_SECONDS:
            features = gaze_features(state)
            if features is not None:
                self.samples.append((features, self.idx))
        if elapsed < cfg.CAL_SETTLE_SECONDS + cfg.CAL_CAPTURE_SECONDS:
            return
        good = sum(1 for _, i in self.samples if i == self.idx)
        if good >= cfg.CAL_MIN_SAMPLES:
            self.idx, self.attempt, self.message = self.idx + 1, 0, ""
        elif self.attempt < cfg.CAL_MAX_RETRIES:
            self.attempt += 1
            self.samples = [s for s in self.samples if s[1] != self.idx]
            self.message = "Could not see your eyes clearly on that dot, trying it again."
        else:
            self.phase = "LIVE" if self.mapper else "INTRO"
            self.message = ("Calibration failed: the camera could not see your eyes clearly. "
                            "Sit facing the camera in good light and press C to try again.")
            return
        self.point_start = now
        if self.idx >= len(self.targets):
            self._finish_calibration()

    def _finish_calibration(self):
        rows = np.array([f for f, _ in self.samples])
        groups = np.array([g for _, g in self.samples])
        keep = np.zeros(len(rows), dtype=bool)
        for g in np.unique(groups):
            mask = groups == g
            keep[mask] = _trim(rows[mask])
        rows, groups = rows[keep], groups[keep]
        targets = np.array([self.targets[g] for g in groups], dtype=np.float64)
        mapper = GazeMapper(self.width, self.height)
        try:
            error = mapper.fit(rows, targets, groups)
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.phase = "LIVE" if self.mapper else "INTRO"
            self.message = f"Calibration failed ({exc}). Press C to try again."
            return
        self.mapper = mapper
        try:
            mapper.save()
        except OSError:
            pass
        self.phase = "LIVE"
        self.filter = None
        share = 100.0 * error / self.width
        self.message = f"Calibrated. Test error about {error:.0f} px ({share:.0f}% of the screen width)."
        if error > cfg.CAL_WARN_ERROR * self.width:
            self.message += " That is rough: press C and repeat with your head still and good light."

    def _update_live(self, state, now):
        features = gaze_features(state)
        if features is None:
            return
        raw = self.mapper.predict_raw(features)
        self.recent_raw.append(raw)
        if self.filter is None:
            self.filter = OneEuroFilter(cfg.GAZE_MIN_CUTOFF, cfg.GAZE_BETA)
        point = self.filter(raw + self.mapper.offset, now)
        x, y = float(point[0]), float(point[1])
        self.latest = (x, y)
        self.trail.append((x, y))
        state.gaze_xy = (x, y)
        state.gaze_cell = cell_name(x, y, self.width, self.height)
        mx, my = cfg.OFFSCREEN_MARGIN * self.width, cfg.OFFSCREEN_MARGIN * self.height
        state.gaze_off_screen = not (-mx <= x <= self.width + mx and -my <= y <= self.height + my)

    # ---- the picture
    def render(self, frame, state, active_flags, now):
        canvas = np.full((self.height, self.width, 3), 20, np.uint8)
        if self.phase == "CAL":
            self._draw_calibration(canvas, now)
        elif self.phase == "LIVE":
            self._draw_live(canvas, state)
        else:
            self._draw_intro(canvas, state)
        self._draw_preview(canvas, frame)
        if self.message:
            self._text(canvas, self.message, 30, self.height - 30, 0.8, (80, 220, 255))
        if active_flags:
            cv2.rectangle(canvas, (0, 0), (self.width, 50), (0, 0, 200), -1)
            self._text(canvas, "FLAG: " + ", ".join(active_flags), 20, 34, 0.9, (255, 255, 255))
        return canvas

    def _scale(self):
        return max(0.6, self.width / 1920.0)

    def _text(self, canvas, text, x, y, size, color=(255, 255, 255)):
        s = size * self._scale() / 0.8 * 0.8
        cv2.putText(canvas, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, s, color,
                    max(1, int(round(2 * self._scale()))), cv2.LINE_AA)

    def _draw_intro(self, canvas, state):
        lines = [
            ("Gaze to screen: calibration", 1.2, (255, 255, 255)),
            ("1. Sit where you will sit for the exam, face the camera, light on your face.", 0.8, (220, 220, 220)),
            ("2. Look at each dot until it disappears. Keep your head fairly still, move only your eyes.", 0.8, (220, 220, 220)),
            ("3. About 12 seconds for 9 dots. Press SPACE to start.", 0.8, (220, 220, 220)),
            ("Face found: " + ("yes" if state.n_faces == 1 else "NO" if state.n_faces == 0 else "more than one face"),
             0.8, (120, 255, 120) if state.n_faces == 1 else (80, 80, 255)),
            ("Keys: SPACE start   C recalibrate   Z re-centre   D camera preview   Q quit", 0.7, (160, 160, 160)),
        ]
        y = self.height * 0.28
        for text, size, color in lines:
            self._text(canvas, text, self.width * 0.1, y, size, color)
            y += 62 * self._scale() * (1.3 if size > 1 else 1.0)

    def _draw_calibration(self, canvas, now):
        x, y = (int(v) for v in self.targets[self.idx])
        elapsed = now - self.point_start
        settle, capture = cfg.CAL_SETTLE_SECONDS, cfg.CAL_CAPTURE_SECONDS
        unit = self._scale()
        if elapsed < settle:       # a big dot that shrinks: the eyes follow it in
            radius = int((34 - 22 * min(1.0, elapsed / settle)) * unit)
            cv2.circle(canvas, (x, y), max(radius, 4), (255, 255, 255), -1, cv2.LINE_AA)
        else:                      # measuring: small dot with a progress ring
            cv2.circle(canvas, (x, y), int(10 * unit), (80, 255, 80), -1, cv2.LINE_AA)
            share = min(1.0, (elapsed - settle) / capture)
            cv2.ellipse(canvas, (x, y), (int(26 * unit),) * 2, -90, 0, 360 * share,
                        (80, 255, 80), max(2, int(3 * unit)), cv2.LINE_AA)
        self._text(canvas, f"Dot {min(self.idx + 1, len(self.targets))} of {len(self.targets)}"
                           "   (X cancels)", self.width * 0.35, self.height * 0.5, 0.8, (200, 200, 200))

    def _draw_live(self, canvas, state):
        w, h = self.width, self.height
        for i in (1, 2):  # faint 3x3 grid
            cv2.line(canvas, (w * i // 3, 0), (w * i // 3, h), (45, 45, 45), 1)
            cv2.line(canvas, (0, h * i // 3), (w, h * i // 3), (45, 45, 45), 1)
        if self.latest is not None:
            col, row = min(2, max(0, int(3 * self.latest[0] / w))), min(2, max(0, int(3 * self.latest[1] / h)))
            overlay = canvas.copy()
            cv2.rectangle(overlay, (col * w // 3, row * h // 3), ((col + 1) * w // 3, (row + 1) * h // 3),
                          (70, 70, 30), -1)
            canvas[:] = cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0)
        trail = list(self.trail)
        for i, (px, py) in enumerate(trail):
            fade = (i + 1) / max(len(trail), 1)
            cv2.circle(canvas, (int(px), int(py)), max(2, int(8 * fade * self._scale())),
                       (int(60 * fade), int(180 * fade), int(60 * fade)), -1, cv2.LINE_AA)
        if self.latest is not None:
            px, py = int(self.latest[0]), int(self.latest[1])
            colour = (60, 60, 255) if state.gaze_off_screen else (80, 255, 80)
            cv2.circle(canvas, (px, py), int(20 * self._scale()), colour, 3, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), int(5 * self._scale()), colour, -1, cv2.LINE_AA)
        info = "Gaze: -"
        if state.gaze_xy is not None:
            info = f"Gaze: x {state.gaze_xy[0]:.0f}  y {state.gaze_xy[1]:.0f}   {state.gaze_cell}"
            if state.gaze_off_screen:
                info += "   OFF SCREEN"
        else:
            info += "   (eyes not clearly visible)"
        self._text(canvas, info, 30, 90, 0.8)
        if self.mapper is not None and self.mapper.cv_error_px is not None:
            self._text(canvas, f"Calibration test error: {self.mapper.cv_error_px:.0f} px", 30, 125, 0.7, (170, 170, 170))
        self._text(canvas, "Keys: C recalibrate   Z re-centre (look at the middle first)   D preview   Q quit",
                   30, 160, 0.7, (130, 130, 130))

    def _draw_preview(self, canvas, frame):
        if not self.show_preview or frame is None:
            return
        pw = max(240, self.width // 6)
        ph = int(frame.shape[0] * pw / frame.shape[1])
        small = cv2.resize(frame, (pw, ph))
        x0, y0 = self.width - pw - 20, self.height - ph - 20
        canvas[y0:y0 + ph, x0:x0 + pw] = small
        cv2.rectangle(canvas, (x0, y0), (x0 + pw, y0 + ph), (200, 200, 200), 1)


def gaze_checks():
    """Extra checks for the Monitor when the screen mapping is running."""
    return [Check("GAZE_OFF_SCREEN", cfg.OFFSCREEN_SECONDS, lambda s: s.gaze_off_screen)]
