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

    def __init__(self, path=None):
        self.path = path or cfg.GALLERY_FILE
        self.people = []
        self.load()

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                self.people = json.load(f).get("people", [])

    def save(self):
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

    Called once per frame, but only compares faces every ID_CHECK_SECONDS, and only
    when one face is clearly visible and roughly facing the camera (a turned head or
    closed eyes would give a poor signature and a false mismatch).
    """

    def __init__(self, gallery, embedder, expected_name=None):
        self.gallery = gallery
        self.embedder = embedder
        self.expected_name = expected_name
        self.mismatch = False
        self.detail = ""
        self.status_text = "ID: checking..."
        self._last_check = None

    def _good_view(self, state):
        return (state.n_faces == 1 and not state.eyes_closed
                and state.yaw is not None and abs(state.yaw) <= cfg.YAW_LIMIT
                and abs(state.pitch) <= max(cfg.PITCH_UP_LIMIT, cfg.PITCH_DOWN_LIMIT))

    def __call__(self, state):
        due = self._last_check is None or state.now - self._last_check >= cfg.ID_CHECK_SECONDS
        if due and self._good_view(state):
            self._last_check = state.now
            signature, _, _ = self.embedder.embed(state.frame)
            if signature is not None:  # no face found by this detector: leave the last result
                name, score = self.gallery.best_match(signature, self.expected_name)
                self.mismatch = score < cfg.ID_MATCH_THRESHOLD
                self.status_text = f"ID: {name} {score:.2f} " + ("MISMATCH" if self.mismatch else "OK")
                self.detail = (f"best match {name} with score {score:.2f}, "
                               f"needs {cfg.ID_MATCH_THRESHOLD:.2f}") if self.mismatch else ""
        return self.mismatch, self.detail


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
