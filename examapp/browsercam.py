"""
A candidate's camera when it is a web browser on another computer.

The page captures small JPEG pictures from the candidate's camera and posts them to the exam computer
(`POST /api/frame?token=...`). Here they are decoded and handed to a monitor that runs exactly the checks of the
local camera monitor (flags, face check, gaze calibration, live picture for the invigilator): `StreamMonitor` is
`CameraMonitor` with this feed in place of the webcam.

Only the newest picture is kept. If the analysis is slower than the browser sends, older pictures are dropped
(and counted), so a busy computer shows as fewer analysed pictures, never as a growing delay.
"""

import threading
import time

import cv2
import numpy as np

from facelib import config as cfg
from .monitor import CameraMonitor

MIN_SIDE, MAX_SIDE = 120, 2000          # a picture outside these sizes (in pixels) is refused


class FrameFeed:
    """The newest decoded picture, passed from the HTTP thread to the monitor thread."""

    def __init__(self, stale_seconds=None, clock=time.monotonic):
        self.stale = cfg.BROWSER_STALE_SECONDS if stale_seconds is None else stale_seconds
        self.clock = clock
        self._cond = threading.Condition()
        self._frame, self._seq, self._read_seq = None, 0, 0
        self._last = None                  # clock time of the newest picture
        self.received = self.dropped = 0
        self.closed = False
        self.size = None                   # (width, height) of the newest picture

    def put(self, jpeg):
        """Decode and keep one picture. Returns 'ok', 'bad' (not a usable JPEG) or 'closed'."""
        if self.closed:
            return "closed"
        if not isinstance(jpeg, (bytes, bytearray)) or len(jpeg) > cfg.BROWSER_MAX_FRAME or bytes(jpeg[:3]) != b"\xff\xd8\xff":
            return "bad"
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return "bad"
        h, w = frame.shape[:2]
        if not (MIN_SIDE <= w <= MAX_SIDE and MIN_SIDE <= h <= MAX_SIDE):
            return "bad"
        with self._cond:
            if self._frame is not None and self._seq != self._read_seq:
                self.dropped += 1          # the monitor never took the previous one
            self._frame, self._seq, self._last, self.size = frame, self._seq + 1, self.clock(), (w, h)
            self._jpeg = bytes(jpeg)
            self.received += 1
            self._cond.notify_all()
        return "ok"

    def read(self, timeout=0.5):
        """The next picture the monitor has not seen, or None after `timeout` seconds (or when closed)."""
        with self._cond:
            if not self.closed and self._seq == self._read_seq:
                self._cond.wait(timeout)
            if self.closed or self._seq == self._read_seq:
                return None
            self._read_seq = self._seq
            return self._frame

    def latest(self):
        """(number of pictures so far, the newest picture as JPEG bytes or None): for the invigilator's live view."""
        with self._cond:
            return self.received, getattr(self, "_jpeg", None)

    def age(self):
        """Seconds since the newest picture, or None if none came yet."""
        with self._cond:
            return None if self._last is None else self.clock() - self._last

    def close(self):
        with self._cond:
            self.closed = True
            self._frame = None
            self._cond.notify_all()


class BrowserCamera:
    """Looks like the cv2 camera the monitor reads from: read() waits for the next picture."""

    def __init__(self, feed):
        self.feed = feed

    def read(self):
        while not self.feed.closed:
            frame = self.feed.read(0.5)
            if frame is not None:
                return True, frame
        return False, None

    def release(self):
        self.feed.close()


class StreamMonitor(CameraMonitor):
    """The camera monitor of one browser candidate. Its camera is the pictures the browser posts."""

    def __init__(self, feed=None):
        super().__init__()
        self.feed = feed or FrameFeed()

    def _open_camera(self):
        return BrowserCamera(self.feed)

    def stop(self):
        super().stop()
        self.feed.close()                  # lets a read that is waiting for a picture return

    def put_frame(self, jpeg):
        return self.feed.put(jpeg)

    @property
    def status(self):
        st = super().status
        if st["state"] == "running":
            age = self.feed.age()
            if age is None:
                st.update(state="starting", message="Waiting for the camera pictures from the browser...")
            elif age > self.feed.stale:
                st.update(state="error", faces=None,
                          message=f"No camera pictures from the browser for {int(age)} seconds.")
        return st

    def feed_stats(self):
        return {"received": self.feed.received, "dropped": self.feed.dropped, "size": self.feed.size,
                "age": None if self.feed.age() is None else round(self.feed.age(), 1)}
