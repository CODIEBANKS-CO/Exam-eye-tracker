"""
gaze_demo.py - webcam exam-monitor demo (OpenCV + MediaPipe FaceLandmarker).

This file only runs the camera loop and draws the preview. The measuring lives in
facelib/analysis.py, the list of flags in facelib/checks.py, and every setting in
facelib/config.py.

Install:  pip install opencv-python mediapipe numpy
Run:      python gaze_demo.py                 (press q or Esc to quit)
          python gaze_demo.py --screen        map the gaze onto the screen (calibrates first)
          python gaze_demo.py --screen --load reuse the saved calibration
          python gaze_demo.py --screen --debug also write calibration/gaze_log.csv (one row per frame)

The first run downloads MediaPipe's face_landmarker.task model into the models/
folder. If you are offline, download it manually (the address is LANDMARKER_URL in
facelib/config.py) and save it there.

Note: this gives coarse "looking away" detection, not exact screen coordinates.
Flags are evidence for a human reviewer, not proof of anything.
"""

import argparse
import os
import shutil
import time
from collections import deque

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from facelib import config as cfg
from facelib.analysis import Analyzer
from facelib.camera import open_camera
from facelib.checks import Monitor, build_checks
from facelib.events import log_event
from facelib.models import ensure_file
from facelib.screengaze import (GazeLog, GazeTracker, MouseFollower, draw_pad, load_calibration,
                                recentre, run_calibration, save_calibration, screen_size, thumbnail)


def prepare_landmarker_model():
    """Find the landmarker model, reusing one from an older version of this project."""
    legacy = os.path.join(cfg.BASE_DIR, "face_landmarker.task")
    if not os.path.exists(cfg.LANDMARKER_PATH) and os.path.exists(legacy):
        os.makedirs(cfg.MODEL_DIR, exist_ok=True)
        shutil.copy(legacy, cfg.LANDMARKER_PATH)
    ensure_file(cfg.LANDMARKER_PATH, cfg.LANDMARKER_URL, "face landmark model")


def draw_overlay(frame, state, monitor, status_rules):
    """Return a copy of `frame` sized for the screen, with the readouts drawn on it."""
    height, width = frame.shape[:2]
    if width > cfg.DISPLAY_WIDTH:
        shown = cv2.resize(frame, (cfg.DISPLAY_WIDTH, int(height * cfg.DISPLAY_WIDTH / width)))
    else:
        shown = frame
    sh, sw = shown.shape[:2]

    if state.ear_l is None:
        eyes_line = "Eyes: -"
    else:
        eyes_line = f"Eyes L {state.ear_l:.2f}  R {state.ear_r:.2f}"
        if state.eyes_closed:
            eyes_line += "  BOTH CLOSED"
        elif state.eyes_covered:
            eyes_line += "  BOTH COVERED"
        elif state.hidden_eye:
            eyes_line += f"  {state.hidden_eye} HIDDEN"
    if state.detail_l is None and state.detail_r is None:
        detail_line = "Eye detail: -"
    else:
        shown_l = "-" if state.detail_l is None else f"{state.detail_l:.2f}"
        shown_r = "-" if state.detail_r is None else f"{state.detail_r:.2f}"
        detail_line = f"Eye detail L {shown_l}  R {shown_r}"
        if state.detail_normal:
            detail_line += f"  (normal {state.detail_normal:.2f})"
    if state.iris_h is None:
        iris_line = "Iris: -"
    else:
        iris_line = f"Iris: {state.iris_h:.2f}" + (f"  EXTREME {state.extreme}" if state.extreme else "")

    lines = [
        f"Faces: {state.n_faces}",
        (f"Yaw: {state.yaw:+.0f}  Pitch: {state.pitch:+.0f}"
         if state.yaw is not None else "Yaw/Pitch: -"),
        f"Gaze: {state.zone[0]} / {state.zone[1]}",
        "Looking: " + (", ".join(sorted(state.looking)) if state.looking else "CENTER"),
        eyes_line,
        detail_line,
        iris_line,
        f"Light: {state.luma:.0f}" if state.luma is not None else "Light: -",
    ]
    if state.eye_up is not None and state.eye_down is not None:
        lines.append(f"Eye look: up {state.eye_up:.2f}  down {state.eye_down:.2f}")
    for rule in status_rules:  # running readouts from rules that keep state (glances, identity)
        lines.append(rule.status_text)

    for i, text in enumerate(lines):
        cv2.putText(shown, text, (10, 28 + 26 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if state.luma is not None and state.luma < cfg.LOW_LIGHT_WARN:
        cv2.putText(shown, "Too dark: add light in front of your face",
                    (10, 28 + 26 * len(lines)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    active = monitor.active_names
    if active:
        cv2.rectangle(shown, (0, sh - 44), (sw, sh), (0, 0, 200), -1)
        cv2.putText(shown, "FLAG: " + ", ".join(active), (10, sh - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return shown


class Pipeline:
    """One camera frame in, one analysed FrameState out (used by every mode)."""

    def __init__(self, cap, landmarker, analyzer):
        self.cap, self.landmarker, self.analyzer = cap, landmarker, analyzer
        self.last_ts = -1

    def step(self):
        """Return (display_frame, raw_frame, state, now), or None when the camera stops."""
        ok, frame = self.cap.read()
        if not ok:
            return None
        frame = cv2.flip(frame, 1)  # mirror view feels natural
        raw = frame.copy()          # untouched copy for snapshots and pixel-based checks
        now = time.time()

        # VIDEO mode needs strictly increasing timestamps in milliseconds
        ts = max(int(now * 1000), self.last_ts + 1)
        self.last_ts = ts
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(self.analyzer.model_input(frame, gray), cv2.COLOR_BGR2RGB)
        result = self.landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        return frame, raw, self.analyzer.analyse(result, raw, gray, now), now


def draw_iris_dots(frame, state):
    """Iris dots on the preview: green = open and visible, red = shut or hidden."""
    if state.pts is None:
        return
    width = frame.shape[1]
    for eye, ear_eye, side in ((state.left_eye, state.ear_l, "LEFT"),
                               (state.right_eye, state.ear_r, "RIGHT")):
        shut = (ear_eye < cfg.EAR_CLOSED or state.skipped_eye == side
                or side in state.covered_eyes)
        cv2.circle(frame, tuple(state.pts[eye[0]].astype(int)),
                   max(3, width // 320), (0, 0, 255) if shut else (0, 255, 0), -1)


class ExpressionPrinter:
    """PRINT_BLENDSHAPES: print the strongest expression scores once a second."""

    def __init__(self):
        self.last = 0.0

    def __call__(self, state, now):
        if cfg.PRINT_BLENDSHAPES and state.blendshapes and now - self.last >= 1.0:
            self.last = now
            top = sorted(state.blendshapes.items(), key=lambda kv: -kv[1])[:5]
            print("expression scores:", ", ".join(f"{k} {v:.2f}" for k, v in top))


def run_flags(monitor, state, raw, now):
    for name, seconds, detail in monitor.update(state, now):
        log_event(name, seconds, raw, detail)


def run_preview_mode(pipe, monitor, status_rules):
    """The normal mode: camera preview with the readouts."""
    show_scores = ExpressionPrinter()
    while True:
        out = pipe.step()
        if out is None:
            break
        frame, raw, state, now = out
        run_flags(monitor, state, raw, now)
        show_scores(state, now)
        draw_iris_dots(frame, state)
        cv2.imshow("Gaze demo (q to quit)", draw_overlay(frame, state, monitor, status_rules))
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break


def run_screen_mode(pipe, monitor, status_rules, use_saved, mouse_on, debug=False):
    """Calibrate, then show the full-screen gaze pad and raise GAZE_OFF_SCREEN."""
    width, height = screen_size()
    print(f"Screen size: {width}x{height}")
    win = "Gaze pad"
    cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    mapper = load_calibration(width, height) if use_saved else None
    if mapper is None:
        mapper = run_calibration(pipe.step, win, width, height)
        if mapper is None:
            return
        save_calibration(mapper, width, height)
    tracker = GazeTracker(mapper, width, height)
    mouse = MouseFollower()
    mouse.on = mouse_on
    trail = deque(maxlen=cfg.GAZE_TRAIL)
    show_scores = ExpressionPrinter()
    log = GazeLog() if debug else None
    if log:
        print(f"Debug log: {log.path}")

    try:
        while True:
            out = pipe.step()
            if out is None:
                break
            frame, raw, state, now = out
            tracker.update(state, now)
            if log:
                log.add(state, tracker, now)
            run_flags(monitor, state, raw, now)
            show_scores(state, now)
            draw_iris_dots(frame, state)
            if state.gaze_x is not None and state.gaze_x == state.gaze_x:
                mouse.move(min(max(state.gaze_x, 0), width - 1), min(max(state.gaze_y, 0), height - 1))
            canvas = draw_pad(width, height, state, thumbnail(frame, state), monitor.active_names,
                              [r.status_text for r in status_rules], trail, mouse.on, mapper.summary(),
                              tracker.notes)
            cv2.imshow(win, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                mouse.on = not mouse.on
            if key == ord("z"):
                shift = recentre(pipe.step, win, width, height, mapper)
                if shift is not None:
                    tracker.set_offset(*shift[:2], head=shift[2])
                    trail.clear()
            if key == ord("c"):
                fresh = run_calibration(pipe.step, win, width, height)
                if fresh is not None:
                    mapper = fresh
                    save_calibration(mapper, width, height)
                    tracker = GazeTracker(mapper, width, height)   # also clears any z shift
                    trail.clear()
    finally:
        if log:
            log.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Webcam exam monitor demo.")
    parser.add_argument("--screen", action="store_true",
                        help="map the gaze onto the screen (a calibration comes first)")
    parser.add_argument("--load", action="store_true",
                        help="with --screen: reuse the saved calibration instead of calibrating")
    parser.add_argument("--mouse", action="store_true",
                        help="with --screen: also move the real mouse pointer (Windows only)")
    parser.add_argument("--debug", action="store_true",
                        help="with --screen: write calibration/gaze_log.csv, one row per camera frame")
    args, _ = parser.parse_known_args(argv)
    return args


def main(argv=None):
    args = parse_args(argv)
    prepare_landmarker_model()
    monitor = Monitor(build_checks())
    status_rules = [c.rule for c in monitor.checks if hasattr(c.rule, "status_text")]
    cap = open_camera()

    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=cfg.LANDMARKER_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=2,  # 2 is enough to spot "more than one person"
        output_face_blendshapes=cfg.USE_BLENDSHAPES,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        pipe = Pipeline(cap, landmarker, Analyzer())
        if args.screen:
            run_screen_mode(pipe, monitor, status_rules, args.load, args.mouse, args.debug)
        else:
            run_preview_mode(pipe, monitor, status_rules)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
