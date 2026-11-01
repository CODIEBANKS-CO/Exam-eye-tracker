"""
The webcam monitor, running in a background thread next to the exam server.

It uses exactly the same pieces as `gaze_demo.py` (the camera, the Analyzer, the checks array,
and the gaze calibration and tracker from facelib/screengaze.py) but draws no window of its own.
It runs from the moment the server starts, so the start page can show a camera check, and it
only records flags while an exam is in progress.

Face match: before the calibration the page asks for a "face check". This thread then takes a few pictures
of the candidate's face (SessionIdentity in facelib/identity.py), keeps their signatures in memory only, and
from the moment the exam starts compares the live face with them every second.

Calibration: the exam page shows the calibration screen. This thread runs the same
`run_calibration` the `--screen` mode uses; the pictures it would draw in a window are handed to
the page as JPEG frames instead, and the page passes key presses back.
"""

import os
import threading
from collections import deque

import cv2
import mediapipe as mp  # noqa: F401  (imported here so a missing install fails early)
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from facelib import config as cfg
from facelib import screengaze
from facelib.analysis import Analyzer
from facelib.camera import open_camera
from facelib.checks import Monitor, build_checks
from facelib.events import log_event
from facelib.identity import SessionIdentity

# What the candidate is told while a flag is active (calm, no accusations).
FRIENDLY = {
    "IDENTITY_MISMATCH": "We could not confirm that you are the candidate who started this exam. Please sit in front of the camera, facing the screen.",
    "IDENTITY_UNVERIFIED": "We cannot confirm who is at the screen. Please face the camera.",
    "GAZE_OFF_SCREEN": "You appear to be looking away from the screen. Please look back at the exam.",
    "REPEATED_GLANCES": "You have looked away several times. Please keep your eyes on the exam.",
    "NO_FACE": "We cannot see your face. Please sit in front of the camera.",
    "MULTIPLE_FACES": "More than one face is in view. Only the candidate should be visible.",
    "EYES_COVERED": "Please keep your eyes uncovered.",
    "LEFT_EYE_HIDDEN": "Please keep your face and eyes uncovered.",
    "RIGHT_EYE_HIDDEN": "Please keep your face and eyes uncovered.",
    "EYES_CLOSED": "Please keep your eyes open.",
}
DEFAULT_WARNING = "Please face the screen and keep your eyes on the exam."
KEYS = {"space": 32, "esc": 27}


def warning_text(active_names):
    """One short message for the flags currently raised, or ''."""
    if not active_names:
        return ""
    for name in FRIENDLY:           # most specific message first
        if name in active_names:
            return FRIENDLY[name]
    return DEFAULT_WARNING


class NullMonitor:
    """Stand-in used with --no-camera: the exam runs, nothing is watched."""

    def __init__(self):
        self.status = {"state": "off", "message": "Camera monitoring is switched off (--no-camera)",
                       "faces": None, "light": None, "active": [], "gaze": None, "identity": None}

    def start(self): pass
    def stop(self): pass
    def begin_exam(self, on_flag, context): pass
    def end_exam(self): pass
    def begin_calibration(self, w, h): return False
    def calibration(self): return {"state": "idle"}
    def calibration_frame(self): return None
    def press_key(self, name): pass
    def skip_gaze(self): pass
    def set_gaze_view(self, fullscreen, w, h): pass
    def gaze(self): return None
    def face_available(self): return False
    def begin_face_enrol(self): return False
    def face(self): return {"state": "idle"}
    def skip_face(self, reason=""): pass
    def face_summary(self): return None
    def want_preview(self, on=True): pass
    def preview(self): return None


class CameraMonitor(threading.Thread):
    """Reads the camera, keeps `status` up to date and reports flags to the exam."""

    def __init__(self):
        super().__init__(daemon=True)
        self._lock = threading.Lock()
        self._stop_flag = threading.Event()
        self._reset = threading.Event()
        self._on_flag = None       # callable(name, seconds, detail, snapshot=) while an exam runs
        self._context = lambda: ""
        self._status = {"state": "starting", "message": "Starting the camera...", "faces": None,
                        "light": None, "active": [], "gaze": None, "identity": None}
        # gaze calibration and tracking
        self._cal_request = None   # (w, h) waiting to be run
        self._cal_state = "idle"   # idle | running | done | failed | cancelled | skipped
        self._cal_info = {}
        self._cancelled = False
        self._canvas, self._canvas_n, self._jpeg = None, 0, (0, b"")
        self._keys = deque()
        self._tracker = None
        self._gaze_size = None
        self._gaze_paused = False
        # face match
        self._identity = None      # SessionIdentity: made on the first face check (it loads two models)
        self._face_request = False
        self._face_state = "idle"  # idle | running | done | failed | unavailable | skipped
        self._face_info = {}
        self._face_cancel = False
        # small camera picture for the invigilator's room view (only made while somebody asks for it)
        self._want_preview = False
        self._preview = (0, b"")
        self._preview_t = -1e9

    # --------------------------- used by the server ---------------------------
    @property
    def status(self):
        with self._lock:
            return dict(self._status)

    def begin_exam(self, on_flag, context):
        """Start recording flags (timers start from zero). `context()` names the current question."""
        with self._lock:
            self._on_flag, self._context = on_flag, context
        self._reset.set()

    def end_exam(self):
        with self._lock:
            self._on_flag = None
            self._tracker = None
            self._status["gaze"] = None
            self._status["identity"] = None
            if self._identity is not None:        # the reference is thrown away with the exam
                self._identity.discard()
            self._identity, self._face_state, self._face_info = None, "idle", {}

    def stop(self):
        self._stop_flag.set()

    def begin_calibration(self, w, h):
        """Ask the thread to run the gaze calibration for a w x h window. False if one is running."""
        with self._lock:
            if self._cal_state == "running":
                return False
            self._cal_request, self._cal_state, self._cal_info = (int(w), int(h)), "running", {}
            self._cancelled, self._tracker, self._canvas = False, None, None
            self._keys.clear()
            self._status["gaze"] = None
        return True

    def calibration(self):
        with self._lock:
            return {"state": self._cal_state, **self._cal_info}

    def calibration_frame(self):
        """The calibration screen as JPEG bytes (None if there is nothing to show)."""
        with self._lock:
            canvas, n = self._canvas, self._canvas_n
            if canvas is None:
                return None
            if self._jpeg[0] == n:
                return self._jpeg[1]
        ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])
        data = buf.tobytes() if ok else None
        with self._lock:
            if data is not None:
                self._jpeg = (n, data)
        return data

    def press_key(self, name):
        if name in KEYS:
            with self._lock:
                self._keys.append(KEYS[name])

    def skip_gaze(self):
        """The candidate carries on without gaze tracking."""
        with self._lock:
            self._tracker, self._cal_state, self._status["gaze"] = None, "skipped", None

    def set_gaze_view(self, fullscreen, w, h):
        """The page reports its size: gaze checks pause if it is not the size it was calibrated at."""
        with self._lock:
            if self._gaze_size is None:
                return
            gw, gh = self._gaze_size
            tol = cfg.EXAM_GAZE_SIZE_TOLERANCE
            mismatch = abs(w - gw) > tol * gw or abs(h - gh) > tol * gh
            self._gaze_paused = (not fullscreen) or mismatch

    def gaze(self):
        with self._lock:
            return self._status.get("gaze")

    # ---- face match ----
    def face_available(self):
        """The face check can be offered: switched on and the camera is running."""
        return bool(cfg.EXAM_FACE_MATCH) and self.status["state"] == "running"

    def begin_face_enrol(self):
        """Ask the thread to make the reference from the candidate's face. False if one is running."""
        with self._lock:
            if self._face_state == "running":
                return False
            self._face_request, self._face_cancel = True, False
            self._face_state, self._face_info = "running", {"progress": 0.0, "hint": "Look at the screen and hold still."}
            if self._identity is not None:
                self._identity.discard()
            self._status["identity"] = None
        return True

    def face(self):
        with self._lock:
            return {**self._face_info, "state": self._face_state}

    def skip_face(self, reason=""):
        """The candidate carries on without face matching."""
        with self._lock:
            self._face_cancel = True
            if self._identity is not None:
                self._identity.discard()
            self._identity, self._face_state = None, "skipped"
            self._face_info = {"reason": str(reason)[:200]}
            self._status["identity"] = None

    def face_summary(self):
        """What goes into the result: the reference quality and the match statistics (no signatures)."""
        with self._lock:
            if self._face_state == "skipped":
                return {"skipped": True, **self._face_info}
            if self._identity is None or self._face_state != "done":
                return {"enrolled": False, "state": self._face_state}
            out = self._identity.summary()
            if self._face_info.get("snapshot"):
                out["reference_snapshot"] = self._face_info["snapshot"]
            return out

    def want_preview(self, on=True):
        with self._lock:
            self._want_preview = bool(on)

    def preview(self):
        """(number, jpeg bytes) of the latest small camera picture, or None before the first one."""
        with self._lock:
            return self._preview if self._preview[0] else None

    def _make_preview(self, raw, now):
        with self._lock:
            wanted, last = self._want_preview, self._preview_t
        if not wanted or now - last < cfg.ROOM_PREVIEW_MIN_SECONDS:
            return
        h, w = raw.shape[:2]
        width = min(cfg.ROOM_PREVIEW_WIDTH, w)
        small = cv2.resize(raw, (width, max(1, int(h * width / w))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, cfg.ROOM_PREVIEW_QUALITY])
        if ok:
            with self._lock:
                self._preview, self._preview_t = (self._preview[0] + 1, buf.tobytes()), now

    # ------------------------------- the thread -------------------------------
    def _set(self, **kw):
        with self._lock:
            self._status.update(kw)

    def _display(self, canvas):
        """Called by the calibration for every picture it would draw; hands it to the page."""
        with self._lock:
            self._canvas, self._canvas_n = canvas, self._canvas_n + 1
            key = self._keys.popleft() if self._keys else 0
            if key == 27:
                self._cancelled = True
        return key

    def _calibrate(self, pipe, w, h):
        def step():
            if self._stop_flag.is_set():
                return None
            out = pipe.step()
            if out is not None:
                state = out[2]
                self._set(faces=state.n_faces, light=None if state.luma is None else round(state.luma))
            return out

        screengaze.DISPLAY = self._display
        try:
            mapper = screengaze.run_calibration(step, "exam", w, h)
        except Exception as exc:     # a calibration problem must not stop the monitor
            print(f"[CALIBRATION ERROR] {exc}")
            mapper = None
        finally:
            screengaze.DISPLAY = None
        with self._lock:
            if mapper is None:
                self._cal_state = "cancelled" if self._cancelled else "failed"
                self._cal_info = {}
            else:
                self._tracker, self._gaze_size, self._gaze_paused = screengaze.GazeTracker(mapper, w, h), (w, h), False
                headline = mapper.check_px if mapper.check_px is not None else mapper.error_px
                diag = (w * w + h * h) ** 0.5
                self._cal_info = {"screen": [w, h], "accuracy_px": round(float(headline), 1),
                                  "accuracy_fraction": round(float(headline) / diag, 4),
                                  "cross_check_px": None if mapper.error_px is None else round(float(mapper.error_px), 1),
                                  "good": float(headline) / diag <= cfg.EXAM_MAX_CAL_ERROR_FRACTION}
                self._cal_state = "done"

    def _enrol(self, pipe):
        """Make the reference from the candidate's face (the face check step of the exam)."""
        try:
            with self._lock:
                ident = self._identity
            if ident is None:
                ident = SessionIdentity()       # downloads / loads the two models: may fail (offline, blocked)
                with self._lock:
                    self._identity = ident
        except (Exception, SystemExit) as exc:  # ensure_file stops with SystemExit when a download is blocked
            print(f"[FACE CHECK UNAVAILABLE] {exc}")
            with self._lock:
                self._face_state, self._face_info = "unavailable", {"message": str(exc).splitlines()[0][:200]}
            return
        begun = False
        while not self._stop_flag.is_set():
            with self._lock:
                if self._face_cancel:
                    return
            out = pipe.step()
            if out is None:
                with self._lock:
                    self._face_state, self._face_info = "failed", {"message": "The camera stopped sending pictures."}
                return
            _, raw, state, now = out
            state.now = now
            self._set(faces=state.n_faces, light=None if state.luma is None else round(state.luma))
            if not begun:
                ident.begin_enrol(now)
                begun = True
            result = ident.enrol_step(state)
            info = ident.info()
            if result == "done":
                snap = None
                if ident.reference_frame is not None:
                    snap = log_event("FACE_ENROLLED", 0.0, ident.reference_frame, "picture of the candidate at the start")
                    ident.reference_frame = None
                    info["snapshot"] = os.path.basename(snap) if snap else None
            with self._lock:
                if self._face_cancel:
                    return
                self._face_info = {k: v for k, v in info.items() if k != "state"}
                self._face_state = result
            if result != "running":      # done or failed: back to the main loop
                return

    def _make_monitor(self):
        """A fresh Monitor: the usual checks plus the face-match checks once a reference exists."""
        checks = build_checks(gallery_identity=False)
        with self._lock:
            ident = self._identity if self._face_state == "done" else None
        if ident is not None:
            checks += ident.checks()
        return Monitor(checks)

    def run(self):
        try:
            from gaze_demo import Pipeline, prepare_landmarker_model
            prepare_landmarker_model()
            cap = open_camera()
            options = vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=cfg.LANDMARKER_PATH),
                running_mode=vision.RunningMode.VIDEO, num_faces=2,
                output_face_blendshapes=cfg.USE_BLENDSHAPES,
                min_face_detection_confidence=0.5, min_tracking_confidence=0.5)
            with vision.FaceLandmarker.create_from_options(options) as landmarker:
                pipe = Pipeline(cap, landmarker, Analyzer())
                monitor = self._make_monitor()
                self._set(state="running", message="Camera on")
                while not self._stop_flag.is_set():
                    with self._lock:
                        request, self._cal_request = self._cal_request, None
                        face_request, self._face_request = self._face_request, False
                    if face_request:
                        self._enrol(pipe)
                        monitor = self._make_monitor()
                        continue
                    if request is not None:
                        self._calibrate(pipe, *request)
                        monitor = self._make_monitor()
                        continue
                    out = pipe.step()
                    if out is None:
                        self._set(state="error", message="The camera stopped sending pictures.")
                        break
                    _, raw, state, now = out
                    self._make_preview(raw, now)
                    if self._reset.is_set():
                        self._reset.clear()
                        monitor = self._make_monitor()
                    with self._lock:
                        tracker, paused, size = self._tracker, self._gaze_paused, self._gaze_size
                    gaze = None
                    if tracker is not None:
                        tracker.update(state, now)          # fills state.gaze_x / gaze_y / gaze_off
                        if paused:
                            state.gaze_off = False
                        if state.gaze_x is not None and state.gaze_x == state.gaze_x:
                            gaze = {"x": round(state.gaze_x / size[0], 4), "y": round(state.gaze_y / size[1], 4),
                                    "off": bool(state.gaze_off), "paused": bool(paused)}
                    fired = monitor.update(state, now)
                    with self._lock:
                        ident = self._identity if self._face_state == "done" else None
                    self._set(faces=state.n_faces, light=None if state.luma is None else round(state.luma),
                              active=list(monitor.active_names), gaze=gaze,
                              identity=None if ident is None else ident.live(now))
                    with self._lock:
                        on_flag, context = self._on_flag, self._context
                    if on_flag is None:
                        continue
                    for name, seconds, detail in fired:
                        tag = context()
                        snap = log_event(name, seconds, raw, (f"{tag}: {detail}" if detail else tag) if tag else detail)
                        on_flag(name, seconds, detail, snapshot=os.path.basename(snap) if snap else None)
            cap.release()
        except Exception as exc:   # the exam must keep running even if the camera fails
            self._set(state="error", message=f"Camera monitor stopped: {exc}", active=[], gaze=None)
            print(f"[MONITOR ERROR] {exc}")
