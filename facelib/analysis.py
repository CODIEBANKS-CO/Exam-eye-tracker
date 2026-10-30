"""
Face geometry and per-frame analysis.

Nothing here touches the camera, the window or the log. Each frame goes through
Analyzer.analyse(), which returns a FrameState: one plain object holding every
measurement (head angles, eye openness, iris position, ...). The checks in
checks.py only read that object.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import config as cfg
from .cover import CONTOUR_A, CONTOUR_B, CoverDetector

# ------------------------------ landmark indices -------------------------
# Each eye: (iris centre, corner A, corner B, upper lid, lower lid)
EYE_A = (468, 33, 133, 159, 145)
EYE_B = (473, 362, 263, 386, 374)

# Landmarks used for head pose: nose tip, chin, outer eye corners, mouth corners
POSE_IDS = [1, 152, 33, 263, 61, 291]
# Generic 3D face model (millimetre-ish units) matching POSE_IDS order
MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),           # nose tip
        (0.0, -330.0, -65.0),      # chin
        (-225.0, 170.0, -135.0),   # eye outer corner, image-left
        (225.0, 170.0, -135.0),    # eye outer corner, image-right
        (-150.0, -150.0, -125.0),  # mouth corner, image-left
        (150.0, -150.0, -125.0),   # mouth corner, image-right
    ],
    dtype=np.float64,
)


# --------------------------------- geometry ------------------------------
def landmarks_to_pixels(landmarks, width, height):
    """Convert a face's normalised landmarks to an (N, 2) pixel array."""
    return np.array([(lm.x * width, lm.y * height) for lm in landmarks])


def eye_ratios(pts, eye):
    """Return (horizontal, vertical) iris position inside the eye, each 0..1."""
    iris, a, b, top, bottom = eye
    left, right = sorted((pts[a][0], pts[b][0]))
    h = (pts[iris][0] - left) / max(right - left, 1e-6)
    v = (pts[iris][1] - pts[top][1]) / max(pts[bottom][1] - pts[top][1], 1e-6)
    return h, v


def gaze_zone(h, v):
    """Name the coarse gaze direction from the averaged iris ratios."""
    horiz = "LEFT" if h < cfg.GAZE_LOW else "RIGHT" if h > cfg.GAZE_HIGH else "CENTER"
    vert = "UP" if v < cfg.VERT_LOW else "DOWN" if v > cfg.VERT_HIGH else "CENTER"
    return horiz, vert


def head_pose(pts, width, height):
    """Return (yaw, pitch) in degrees, or None if solvePnP fails."""
    # Assign each left/right pair (eye corners, mouth corners) by where it really is in
    # the picture, so the pose does not depend on how MediaPipe names the sides.
    ids = list(POSE_IDS)
    for i, j in ((2, 3), (4, 5)):
        if pts[ids[i]][0] > pts[ids[j]][0]:
            ids[i], ids[j] = ids[j], ids[i]
    image_points = np.array([pts[k] for k in ids], dtype=np.float64)
    focal = float(width)  # rough guess; fine for a demo
    camera = np.array(
        [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64
    )
    ok, rvec, _ = cv2.solvePnP(
        MODEL_POINTS, image_points, camera, np.zeros((4, 1)),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    rmat, _ = cv2.Rodrigues(rvec)
    # Direction the nose points, in camera coordinates (x right, y down, z away from
    # the camera). Reading yaw and pitch from this vector avoids the angle ambiguities
    # of Euler decomposition. Positive yaw = nose toward the left of the picture,
    # positive pitch = nose pointing down.
    fx, fy, fz = rmat @ np.array([0.0, 0.0, 1.0])
    yaw = np.degrees(np.arctan2(-fx, -fz))
    pitch = np.degrees(np.arcsin(np.clip(fy, -1.0, 1.0)))
    return float(yaw), float(pitch)


def eye_openness(pts, eye):
    """Eye aspect ratio: lid gap / eye width. About 0.25+ open, near 0 closed."""
    _, a, b, top, bottom = eye
    gap = abs(pts[bottom][1] - pts[top][1])
    return gap / max(abs(pts[a][0] - pts[b][0]), 1e-6)


def one_eye_hidden(ear_l, ear_r, yaw):
    """Return "LEFT" or "RIGHT" (as seen in the preview) when exactly one eye
    looks shut or covered, otherwise None."""
    if yaw is not None and abs(yaw) > cfg.YAW_LIMIT:
        return None
    closed_l, closed_r = ear_l < cfg.EAR_CLOSED, ear_r < cfg.EAR_CLOSED
    if closed_l and closed_r:
        return None  # both shut is the EYES_CLOSED case
    if closed_l != closed_r or abs(ear_l - ear_r) > cfg.EAR_ASYMMETRY:
        return "LEFT" if ear_l < ear_r else "RIGHT"
    return None


def eye_look(blendshapes, direction):
    """Average of the model's eyeLook<direction>Left / Right scores (0..1), or None if absent.
    `direction` is "Up" or "Down"."""
    scores = [blendshapes[name] for name in (f"eyeLook{direction}Left", f"eyeLook{direction}Right")
              if name in blendshapes]
    return sum(scores) / len(scores) if scores else None


def face_luma(gray, pts, width, height):
    """Mean brightness (0-255) of the middle of the face, or None if too small.

    Only the central half of the face box is used so hair and background at the
    edges do not skew the number.
    """
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    mx, my = (x1 - x0) * 0.25, (y1 - y0) * 0.25
    x0, x1 = int(max(x0 + mx, 0)), int(min(x1 - mx, width))
    y0, y1 = int(max(y0 + my, 0)), int(min(y1 - my, height))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return float(gray[y0:y1, x0:x1].mean())


def brighten(frame, luma):
    """Gamma-lift (plus optional local contrast) so dark faces and irises are
    easier for the model to see. Used for the model input only."""
    luma = min(max(luma, 5.0), float(cfg.TARGET_LUMA))
    gamma = float(np.clip(np.log(cfg.TARGET_LUMA / 255) / np.log(luma / 255), 0.4, 1.0))
    table = (255 * (np.arange(256) / 255) ** gamma).astype(np.uint8)
    lifted = cv2.LUT(frame, table)
    if not cfg.ENHANCE_CONTRAST:
        return lifted
    lab = cv2.cvtColor(lifted, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    l_chan = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l_chan)
    return cv2.cvtColor(cv2.merge((l_chan, a_chan, b_chan)), cv2.COLOR_LAB2BGR)


def directions(yaw, pitch, horiz_zone):
    """Return the set of directions the person is looking away in.

    UP / DOWN come from head pitch only (the vertical iris ratio is too noisy).
    LEFT / RIGHT come from head yaw OR the iris zone. LEFT and RIGHT are as seen
    in the mirrored preview, so they match what the person sees on screen.
    """
    found = set()
    if pitch is not None:
        down_amount = pitch if cfg.POSITIVE_PITCH_IS_DOWN else -pitch
        if down_amount > cfg.PITCH_DOWN_LIMIT:
            found.add("DOWN")
        elif -down_amount > cfg.PITCH_UP_LIMIT:
            found.add("UP")
    if yaw is not None:
        if yaw > cfg.YAW_LIMIT:
            found.add("LEFT")
        elif yaw < -cfg.YAW_LIMIT:
            found.add("RIGHT")
    if horiz_zone in ("LEFT", "RIGHT"):
        found.add(horiz_zone)
    return found


# ------------------------------ per-frame state --------------------------
@dataclass
class FrameState:
    """Everything measured on one frame. Checks read this and nothing else."""

    n_faces: int = 0
    now: float = 0.0                     # time of this frame in seconds (set by the Monitor)
    frame: object = None                 # the raw BGR picture (for checks that need pixels)
    pts: object = None                   # (478, 2) pixel landmarks of the first face
    yaw: float = None                    # + = nose toward the left of the picture
    pitch: float = None                  # + = nose pointing down
    ear_l: float = None                  # openness of the eye on the left of the picture
    ear_r: float = None
    eyes_closed: bool = False            # both eyes shut
    hidden_eye: str = None               # "LEFT" / "RIGHT": exactly one eye shut or covered
    covered_eyes: set = field(default_factory=set)   # eyes whose picture shows no eye (a hand over it)
    eyes_covered: bool = False           # both eyes unseen and at least one of them covered
    detail_l: float = None               # image detail inside each eye outline (see cover.py)
    detail_r: float = None
    detail_normal: float = None          # this person's learned normal detail
    far_eye: str = None                  # eye hidden behind the nose when the head is turned far
    skipped_eye: str = None              # the eye whose iris is ignored (hidden_eye or far_eye)
    left_eye: tuple = None               # landmark tuple of the eye on the left of the picture
    right_eye: tuple = None
    zone: tuple = ("-", "-")             # coarse gaze zone, e.g. ("CENTER", "CENTER")
    iris_h: float = None                 # iris position 0 (left corner) .. 1 (right corner)
    extreme: str = None                  # "LEFT" / "RIGHT" when the iris is at the very edge
    turned_one_eye: bool = False         # head turned far, one eye visible, its iris centred
    looking: set = field(default_factory=set)       # {"UP", "DOWN", "LEFT", "RIGHT"}
    blendshapes: dict = field(default_factory=dict)  # expression scores, if enabled
    gaze_xy: tuple = None                # (x, y) screen pixels the person is looking at (gaze_screen.py)
    gaze_cell: str = None                # e.g. "TOP-LEFT", "CENTER" (3x3 grid over the screen)
    gaze_off_screen: bool = False        # gaze point is well outside the screen edges
    eye_up: float = None                 # 0..1 how far the eyes look up (needs USE_BLENDSHAPES)
    eye_down: float = None               # 0..1 how far the eyes look down
    gaze_x: float = None                 # where the eyes look on the screen, in screen pixels
    gaze_y: float = None                 #   (set by screengaze.GazeTracker in --screen mode)
    gaze_off: bool = False               # that point is outside the screen (plus a margin)
    luma: float = None                   # smoothed face brightness 0-255
    details: dict = field(default_factory=dict)      # extra log text keyed by check name


class Analyzer:
    """Turns the landmarker's result for one frame into a FrameState."""

    def __init__(self):
        self.luma = None  # smoothed face brightness; None until a face has been seen
        self.cover = CoverDetector()

    def model_input(self, frame, gray):
        """The picture to send to the model: brightened when the face is dark."""
        ref = self.luma if self.luma is not None else float(gray.mean())
        if cfg.ENHANCE_LOW_LIGHT and ref < cfg.TARGET_LUMA:
            return brighten(frame, ref)
        return frame

    def analyse(self, result, frame, gray, now=None):
        height, width = frame.shape[:2]
        faces = result.face_landmarks or []
        state = FrameState(n_faces=len(faces), frame=frame, luma=self.luma)
        if not faces:
            self.cover.reset()  # the next face starts with a fresh idea of "normal"
            return state

        pts = landmarks_to_pixels(faces[0], width, height)
        state.pts = pts

        shapes = getattr(result, "face_blendshapes", None) or []
        if shapes:
            state.blendshapes = {c.category_name: c.score for c in shapes[0]}
            state.eye_up = eye_look(state.blendshapes, "Up")
            state.eye_down = eye_look(state.blendshapes, "Down")

        measured = face_luma(gray, pts, width, height)
        if measured is not None:
            self.luma = measured if self.luma is None else 0.8 * self.luma + 0.2 * measured
        state.luma = self.luma

        pose = head_pose(pts, width, height)
        if pose:
            state.yaw, state.pitch = pose
        yaw = state.yaw

        # Name the eyes by where they really are in the picture, so LEFT/RIGHT
        # always match the preview whatever MediaPipe calls each side.
        if pts[EYE_A[0]][0] <= pts[EYE_B[0]][0]:
            left_eye, right_eye = EYE_A, EYE_B
        else:
            left_eye, right_eye = EYE_B, EYE_A
        state.left_eye, state.right_eye = left_eye, right_eye

        state.ear_l = eye_openness(pts, left_eye)
        state.ear_r = eye_openness(pts, right_eye)
        state.eyes_closed = state.ear_l < cfg.EAR_CLOSED and state.ear_r < cfg.EAR_CLOSED
        state.hidden_eye = one_eye_hidden(state.ear_l, state.ear_r, yaw)

        # A hand over an eye: the outline still reads "open", so look at the picture itself.
        if cfg.COVER_CHECK:
            ids_l, ids_r = (CONTOUR_A, CONTOUR_B) if left_eye == EYE_A else (CONTOUR_B, CONTOUR_A)
            frontal = yaw is not None and abs(yaw) <= cfg.YAW_LIMIT
            seen = self.cover.update(gray, pts, ids_l, ids_r, state.ear_l, state.ear_r, frontal, now)
            state.detail_l, state.detail_r = seen.detail_l, seen.detail_r
            state.detail_normal = seen.normal
            state.covered_eyes = seen.covered
            shut = {side for side, ear in (("LEFT", state.ear_l), ("RIGHT", state.ear_r))
                    if ear < cfg.EAR_CLOSED}
            if seen.covered and len(seen.covered | shut) == 2:
                state.eyes_covered = True       # both eyes unseen
                state.hidden_eye = None         # EYES_COVERED says it all
                def shown(value):
                    return "shut" if value is None else f"{value:.2f}"
                state.details["EYES_COVERED"] = (
                    f"image detail left {shown(seen.detail_l)}, right {shown(seen.detail_r)}"
                    + (f", normal about {seen.normal:.2f}" if seen.normal else ""))
            elif seen.covered and state.hidden_eye is None:
                side = next(iter(seen.covered))
                state.hidden_eye = side
                detail = seen.detail_l if side == "LEFT" else seen.detail_r
                state.details[f"{side}_EYE_HIDDEN"] = (
                    f"{side.lower()} eye looks covered: image detail {detail:.2f} "
                    f"against {seen.reference or 0:.2f} for an uncovered eye")

        # Head turned far: the eye on the side the nose points to is hidden
        # behind the nose bridge, so only the other eye is trusted.
        if yaw is not None and abs(yaw) > cfg.YAW_ONE_EYE:
            state.far_eye = "LEFT" if yaw > 0 else "RIGHT"
        skipped = state.hidden_eye or state.far_eye  # the eye whose iris we ignore
        state.skipped_eye = skipped

        (hl, vl), (hr, vr) = eye_ratios(pts, left_eye), eye_ratios(pts, right_eye)
        if state.eyes_closed or state.eyes_covered:
            # iris hidden, so its position means nothing
            state.zone = ("CLOSED" if state.eyes_closed else "COVERED", "-")
        else:
            if skipped == "LEFT":      # only trust the eye that is visible
                h, v = hr, vr
            elif skipped == "RIGHT":
                h, v = hl, vl
            else:
                h, v = (hl + hr) / 2, (vl + vr) / 2
            state.iris_h = h
            state.zone = gaze_zone(h, v)

            # Iris pushed to the very edge of the eye.
            state.extreme = ("LEFT" if h < cfg.EXTREME_LOW
                             else "RIGHT" if h > cfg.EXTREME_HIGH else None)
            if state.extreme:
                state.details[f"EYES_EXTREME_{state.extreme}"] = (
                    f"iris position {h:.2f} (0 = left corner, 1 = right corner)")

            # Head turned, only one eye visible, and that iris is centred.
            if state.far_eye and cfg.GAZE_LOW <= h <= cfg.GAZE_HIGH:
                state.turned_one_eye = True
                visible = "RIGHT" if state.far_eye == "LEFT" else "LEFT"
                state.details["HEAD_TURNED_ONE_EYE"] = (
                    f"head turned {yaw:+.0f} deg, only the {visible} eye visible, "
                    f"its iris centred ({h:.2f})")

        # An extreme iris gets its own flag, so it is not also counted as LOOKING_LEFT/RIGHT.
        state.looking = directions(
            state.yaw, state.pitch, None if state.extreme else state.zone[0])
        return state
