"""
Optional identity check (part B): is the person in front of the camera the candidate
who was enrolled?

How it works
  * enroll_face.py captures a few pictures of the candidate and stores, for each, a
    short list of numbers (a "face signature", 128 values). No photos are stored.
  * During the exam, every few seconds the live face is turned into a signature the
    same way and compared with the candidate's stored signatures (cosine similarity).
  * If the similarity stays below ID_MATCH_THRESHOLD for ID_MISMATCH_SECONDS, the
    IDENTITY_MISMATCH flag is raised for a human to review.

Models (small ONNX files from the OpenCV model zoo, downloaded on first use):
  * YuNet  finds the face
  * SFace  turns an aligned face into the 128-number signature

In the exam (examapp) there is no enrolment step done beforehand: SessionIdentity makes the reference
live when the exam starts, keeps it in memory only, and then keeps matching the live face every second
until the exam ends (see the bottom of this file).

Important: face recognition is less accurate for some groups of people than others,
and lighting, glasses and camera quality all change the score. Treat a mismatch as a
reason for a person to look, never as proof, and test it on your own candidates
before relying on it. The signatures are biometric data: get consent, keep the
gallery file private, and delete it after the exam window.
"""

import json
import os
from datetime import datetime

import cv2
import numpy as np

from . import config as cfg
from .checks import Check
from .models import ensure_file


def cosine(a, b):
    """Cosine similarity of two signatures: 1.0 = identical direction, 0 = unrelated."""
    a, b = np.asarray(a, dtype=np.float32).ravel(), np.asarray(b, dtype=np.float32).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# --------------------------------- embedder ------------------------------
class FaceEmbedder:
    """Finds the largest face in a picture and returns its signature."""

    def __init__(self):
        ensure_file(cfg.YUNET_PATH, cfg.YUNET_URL, "face detector model (YuNet)")
        ensure_file(cfg.SFACE_PATH, cfg.SFACE_URL, "face recognition model (SFace)")
        self.detector = cv2.FaceDetectorYN.create(
            cfg.YUNET_PATH, "", (320, 320), cfg.ID_MIN_FACE_SCORE, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(cfg.SFACE_PATH, "")

    def embed(self, frame):
        """Return (signature or None, number_of_faces, box or None).

        `box` is (x, y, w, h) of the chosen face in the coordinates of `frame`.
        """
        height, width = frame.shape[:2]
        scale = min(1.0, cfg.ID_MAX_WIDTH / width)  # detect on a smaller picture: faster
        small = cv2.resize(frame, (int(width * scale), int(height * scale))) if scale < 1.0 else frame
        small_h, small_w = small.shape[:2]

        self.detector.setInputSize((small_w, small_h))
        _, faces = self.detector.detect(small)
        if faces is None or len(faces) == 0:
            return None, 0, None

        best = max(faces, key=lambda f: f[2] * f[3])  # the largest face
        aligned = self.recognizer.alignCrop(small, best)
        signature = np.asarray(self.recognizer.feature(aligned), dtype=np.float32).ravel()
        x, y, w, h = (float(v) / scale for v in best[:4])
        return signature, len(faces), (int(x), int(y), int(w), int(h))


# --------------------------------- gallery -------------------------------
class FaceGallery:
    """The enrolled people, kept as an array:

        [{"name": "Ada", "created": "...", "embeddings": [[128 numbers], ...]}, ...]
    """

    def __init__(self, path=None, persist=True):
        self.path = path or cfg.GALLERY_FILE
        self.persist = persist          # False: kept in memory only, never read from or written to disk
        self.people = []
        self.load()

    def load(self):
        if self.persist and os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                self.people = json.load(f).get("people", [])

    def save(self):
        if not self.persist:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "people": self.people}, f, indent=1)
        os.replace(tmp, self.path)

    def names(self):
        return [p["name"] for p in self.people]

    def enroll(self, name, signatures):
        """Store (or replace) a person's signatures."""
        entry = {
            "name": name,
            "created": datetime.now().isoformat(timespec="seconds"),
            "embeddings": [[round(float(v), 6) for v in s] for s in signatures],
        }
        self.people = [p for p in self.people if p["name"] != name] + [entry]
        self.save()

    def remove(self, name):
        before = len(self.people)
        self.people = [p for p in self.people if p["name"] != name]
        if len(self.people) != before:
            self.save()
            return True
        return False

    def best_match(self, signature, name=None):
        """Return (name, score): the best cosine similarity against everyone in the
        gallery, or only against `name` when given. (None, -1.0) if nobody matches."""
        best_name, best_score = None, -1.0
        for person in self.people:
            if name is not None and person["name"] != name:
                continue
            for stored in person["embeddings"]:
                score = cosine(signature, stored)
                if score > best_score:
                    best_name, best_score = person["name"], score
        return best_name, best_score


# -------------------------------- the check ------------------------------
class IdentityVerifier:
    """The rule behind the IDENTITY_MISMATCH check.

    Called once per frame, but only compares faces every `interval` seconds, and only
    when one face is clearly visible and roughly facing the camera (a turned head or
    closed eyes would give a poor signature and a false mismatch).

    It also keeps what a reviewer and the exam need:
      * a face that went out of view (or a second face that left) is checked at the very next
        good view instead of waiting for the interval, so swapping people while the candidate
        is out of the picture is caught as fast as a swap in front of the camera;
      * `last_score`, `last_verified` and the counts, for the invigilator's screen and the result;
      * `unverified_seconds(now)`: how long it is since a view matched (see IdentityUnverified).
    """

    def __init__(self, gallery, embedder, expected_name=None, interval=None):
        self.gallery = gallery
        self.embedder = embedder
        self.expected_name = expected_name
        self.interval = cfg.ID_CHECK_SECONDS if interval is None else interval
        self.reset()

    def reset(self):
        """Forget the running state (not the reference): used when the exam clock starts."""
        self.mismatch = False
        self.detail = ""
        self.status_text = "ID: checking..."
        self.last_score = None
        self.lowest_score = None
        self.last_verified = None      # time of the last check that matched
        self.started = None            # time of the first frame seen
        self.checks_done = 0
        self.failed_checks = 0
        self._last_check = None
        self._gap = False              # a face was missing / doubled since the last comparison

    def _good_view(self, state):
        return (state.n_faces == 1 and not state.eyes_closed
                and state.yaw is not None and abs(state.yaw) <= cfg.YAW_LIMIT
                and abs(state.pitch) <= max(cfg.PITCH_UP_LIMIT, cfg.PITCH_DOWN_LIMIT))

    def unverified_seconds(self, now):
        """Seconds since a comparison last matched (or since the first frame, if none has yet)."""
        ref = self.last_verified if self.last_verified is not None else self.started
        return 0.0 if ref is None else max(0.0, now - ref)

    def summary(self):
        return {"checks": self.checks_done, "failed_checks": self.failed_checks,
                "last_score": None if self.last_score is None else round(self.last_score, 3),
                "lowest_score": None if self.lowest_score is None else round(self.lowest_score, 3),
                "threshold": cfg.ID_MATCH_THRESHOLD}

    def __call__(self, state):
        if self.started is None:
            self.started = state.now
        if state.n_faces != 1:
            self._gap = True
        due = self._last_check is None or self._gap or state.now - self._last_check >= self.interval
        if due and self._good_view(state):
            self._last_check, self._gap = state.now, False
            signature, _, _ = self.embedder.embed(state.frame)
            if signature is not None:  # no face found by this detector: leave the last result
                name, score = self.gallery.best_match(signature, self.expected_name)
                self.checks_done += 1
                self.last_score = score
                self.lowest_score = score if self.lowest_score is None else min(self.lowest_score, score)
                self.mismatch = score < cfg.ID_MATCH_THRESHOLD
                if self.mismatch:
                    self.failed_checks += 1
                else:
                    self.last_verified = state.now
                self.status_text = f"ID: {name} {score:.2f} " + ("MISMATCH" if self.mismatch else "OK")
                self.detail = (f"best match {name} with score {score:.2f}, "
                               f"needs {cfg.ID_MATCH_THRESHOLD:.2f}") if self.mismatch else ""
        return self.mismatch, self.detail


class IdentityUnverified:
    """The rule behind IDENTITY_UNVERIFIED: somebody is in front of the camera but no clear,
    front-facing view has matched the reference for `seconds`. This closes the gap an impostor
    could use by never facing the camera, without calling it a mismatch."""

    def __init__(self, verifier, seconds=None):
        self.verifier = verifier
        self.seconds = cfg.ID_UNVERIFIED_SECONDS if seconds is None else seconds

    def __call__(self, state):
        v = self.verifier
        if v.started is None or state.n_faces < 1 or v.mismatch:
            return False
        gap = v.unverified_seconds(state.now)
        return (True, f"no clear front-facing view matched for {gap:.0f} s") if gap > self.seconds else False


def build_identity_check():
    """Create the IDENTITY_MISMATCH Check (used by checks.build_checks)."""
    gallery = FaceGallery()
    if not gallery.people:
        raise SystemExit('USE_IDENTITY is on but nobody is enrolled yet.\n'
                         'Run:  python enroll_face.py "Candidate Name"')
    if cfg.ID_EXPECTED_NAME and cfg.ID_EXPECTED_NAME not in gallery.names():
        raise SystemExit(f'"{cfg.ID_EXPECTED_NAME}" is not in the gallery. '
                         f'Enrolled: {", ".join(gallery.names())}')
    verifier = IdentityVerifier(gallery, FaceEmbedder(), cfg.ID_EXPECTED_NAME)
    return Check("IDENTITY_MISMATCH", cfg.ID_MISMATCH_SECONDS, verifier)


# ------------------------------ the exam: live reference ----------------------------
class SessionIdentity:
    """Who is sitting this exam.

    1. `enrol_step(state)` is fed every camera frame while the candidate looks at the screen. It takes
       `EXAM_FACE_SAMPLES` pictures of a clear, single, front-facing, well-lit face (a little apart), checks
       that they are of one person, and keeps their signatures IN MEMORY ONLY (never written to a file).
    2. `checks()` then returns the two checks the camera monitor runs for the whole exam:
         IDENTITY_MISMATCH   a face in a clear view does not match the reference (compared every second,
                             and at once when a face comes back into view after being missing)
         IDENTITY_UNVERIFIED somebody is there but no clear view has matched for a long time
    3. `discard()` throws the reference away when the exam ends.

    Face recognition makes mistakes (some faces, poor light, look-alikes), and a printed photo can fool
    it. A flag is a reason for a person to look at the picture, never proof.
    """

    def __init__(self, embedder=None):
        self.embedder = embedder or FaceEmbedder()
        self.discard()

    def discard(self):
        self.gallery = FaceGallery(persist=False)
        self.verifier = IdentityVerifier(self.gallery, self.embedder, None, cfg.EXAM_FACE_CHECK_SECONDS)
        self.samples = []
        self.state = "idle"            # idle | running | done | failed
        self.hint = ""
        self.message = ""
        self.consistency = None
        self.reference_frame = None    # one picture of the face at the start (for the reviewer), if enabled
        self._begin = None
        self._last_sample = None

    @property
    def progress(self):
        return min(1.0, len(self.samples) / float(cfg.EXAM_FACE_SAMPLES))

    def begin_enrol(self, now):
        self.discard()
        self.state, self._begin = "running", now
        self.hint = "Look at the screen and hold still."

    def _view_problem(self, state):
        """Why this frame cannot be used for the reference, in words for the candidate (None = fine)."""
        if state.n_faces == 0:
            return "We cannot see your face. Sit in front of the camera."
        if state.n_faces > 1:
            return "More than one face is in view. Only you should be in view."
        if state.luma is not None and state.luma < 60:
            return "It is too dark. Put a light in front of you."
        if state.eyes_closed:
            return "Keep your eyes open."
        if (state.yaw is None or abs(state.yaw) > cfg.YAW_LIMIT
                or abs(state.pitch) > max(cfg.PITCH_UP_LIMIT, cfg.PITCH_DOWN_LIMIT)):
            return "Look straight at the screen."
        return None

    def enrol_step(self, state):
        """Feed one frame (with `state.now` set to the frame's time). Returns the state: 'running', 'done' or 'failed'."""
        if self.state != "running":
            return self.state
        now = state.now
        if now - self._begin > cfg.EXAM_FACE_ENROL_SECONDS:
            return self._fail("The face check took too long. " + (self.hint or ""))
        problem = self._view_problem(state)
        if problem:
            self.hint = problem
            return self.state
        if self._last_sample is not None and now - self._last_sample < cfg.EXAM_FACE_SAMPLE_GAP:
            return self.state
        signature, _, box = self.embedder.embed(state.frame)
        if signature is None:
            self.hint = "We cannot see your face clearly. Look straight at the screen."
            return self.state
        width = state.frame.shape[1]
        if box is not None and box[2] < cfg.ID_ENROL_MIN_FACE_FRACTION * width:
            self.hint = "Move a little closer to the camera."
            return self.state
        self._last_sample = now
        self.samples.append(signature)
        if self.reference_frame is None and cfg.EXAM_FACE_SAVE_REFERENCE:
            self.reference_frame = state.frame.copy()
        self.hint = "Hold still..."
        if len(self.samples) < cfg.EXAM_FACE_SAMPLES:
            return self.state
        # enough pictures: are they one person?
        lowest = min(cosine(a, b) for i, a in enumerate(self.samples) for b in self.samples[i + 1:])
        self.consistency = lowest
        if lowest < cfg.ID_ENROL_CONSISTENCY:
            self.samples, self.reference_frame, self._last_sample = [], None, None
            self.hint = "The pictures did not look like the same person. Only you should be in view. Trying again..."
            return self.state
        self.gallery.enroll("candidate", self.samples)
        self.verifier.reset()
        self.state, self.hint = "done", "Face check complete."
        return self.state

    def _fail(self, message):
        self.state, self.message = "failed", message
        self.samples = []
        return self.state

    def checks(self):
        """The two Check objects for the exam. Safe to call again: it restarts the running state."""
        from .checks import Check
        self.verifier.reset()
        return [Check("IDENTITY_MISMATCH", cfg.EXAM_FACE_MISMATCH_SECONDS, self.verifier),
                Check("IDENTITY_UNVERIFIED", 5.0, IdentityUnverified(self.verifier))]

    def info(self):
        """What the page and the invigilator are shown."""
        return {"state": self.state, "progress": round(self.progress, 2), "hint": self.hint, "message": self.message,
                "consistency": None if self.consistency is None else round(self.consistency, 3)}

    def live(self, now=None):
        """Running readout for the invigilator while the exam runs."""
        v = self.verifier
        if self.state != "done":
            return None
        gap = 0.0 if now is None else v.unverified_seconds(now)
        if v.mismatch:
            status = "mismatch"
        elif v.checks_done and gap > cfg.ID_UNVERIFIED_SECONDS:
            status = "unverified"
        elif v.checks_done:
            status = "ok"
        else:
            status = "waiting"
        return {"status": status, "score": None if v.last_score is None else round(v.last_score, 2),
                "threshold": cfg.ID_MATCH_THRESHOLD, "since_verified": round(gap)}

    def summary(self):
        out = {"enrolled": self.state == "done", "reference_pictures": cfg.EXAM_FACE_SAMPLES if self.state == "done" else 0,
               "reference_consistency": None if self.consistency is None else round(self.consistency, 3)}
        out.update(self.verifier.summary())
        return out
