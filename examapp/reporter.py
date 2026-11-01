"""
The candidate computer's link to the invigilator's computer (the hub).

It runs next to the exam: it joins the room with the room code, then about once a second sends the same
state the local invigilator console would show (name, question, time left, camera, flags, events), a small
camera picture every few seconds, and the picture taken at each flag when the hub asks for it.

* The camera picture is sent only while an exam is in progress, after the candidate has agreed, and never
  before or after. The exam page tells the candidate this in the notice they tick.
* If the hub cannot be reached the exam carries on as normal; the reporter keeps trying. The invigilator
  sees "connection lost" for that computer.
* Only the hub's address and the room code are needed. The reporter never receives the answer key.
"""

import json
import os
import threading
import time
import urllib.error
import urllib.request

from facelib import config as cfg


def normalise_url(url):
    """'192.168.1.10:8765' and 'http://192.168.1.10:8765/' both become 'http://192.168.1.10:8765'."""
    url = str(url or "").strip().rstrip("/")
    if not url:
        raise ValueError("The hub address is empty.")
    if "://" not in url:
        url = "http://" + url
    if not url.startswith(("http://", "https://")):
        raise ValueError("The hub address must start with http:// or https://")
    return url


class Reporter(threading.Thread):
    def __init__(self, app, hub_url, code, label=None, seconds=None, timeout=4.0, log=print):
        super().__init__(daemon=True)
        self.app, self.url, self.code = app, normalise_url(hub_url), code
        self.label = label or _computer_name()
        self.seconds = cfg.ROOM_REPORT_SECONDS if seconds is None else seconds
        self.timeout, self.log = timeout, log
        self.token = None
        self.state = "connecting"            # connecting | connected | offline | wrong_code | stopped
        self.message = ""
        self._halt = threading.Event()
        self._frame_every, self._last_frame, self._sent_frame_n = 2.5, 0.0, 0
        self._sent_pictures = set()
        self._need = []
        self._send_frames = False
        app.room_link = True

    # ----------------------------------------------------------------- the network
    def _call(self, path, data, ctype="application/json", token=True):
        headers = {"Content-Type": ctype}
        if token and self.token:
            headers["X-Hub-Token"] = self.token
        req = urllib.request.Request(self.url + path, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            try:
                body = json.loads(e.read() or b"{}")
            except ValueError:
                body = {}
            return e.code, body

    def _set(self, state, message=""):
        if state != self.state:
            self.log({"connected": "Connected to the invigilator's computer.",
                      "offline": f"Cannot reach the invigilator's computer ({message}). The exam carries on; trying again.",
                      "wrong_code": "The room code is not right. Ask the invigilator for the code and restart this program.",
                      }.get(state, ""))
        self.state, self.message = state, message

    def connect(self):
        status, body = self._call("/api/hub/join", json.dumps({"code": self.code, "label": self.label}).encode(), token=False)
        if status == 200:
            self.token = body["token"]
            self.seconds = max(0.5, float(body.get("report_seconds", self.seconds)))
            self._set("connected")
            return True
        if status == 403:
            self._set("wrong_code", body.get("error", ""))
            self._halt.set()
            return False
        if status == 429:
            self._set("offline", body.get("error", "locked"))
            self._halt.wait(float(body.get("retry_seconds", 30)))
            return False
        self._set("offline", body.get("error", f"HTTP {status}"))
        return False

    # ----------------------------------------------------------------- one round
    def step(self):
        """Report once (and send a picture or two if due). Returns False when the hub did not accept it."""
        if self.token is None and not self.connect():
            return False
        report = self.app.live_view()
        status, reply = self._call("/api/hub/report", json.dumps(report).encode())
        if status == 403:                      # the hub was restarted: join again next round
            self.token = None
            self._set("offline", "the hub does not know this computer; joining again")
            return False
        if status != 200:
            self._set("offline", f"HTTP {status}")
            return False
        self._set("connected")
        self._frame_every = float(reply.get("frame_every", self._frame_every))
        self._send_frames = bool(reply.get("send_frames"))
        self._need = list(reply.get("need", []))
        for name in self._need:
            self._send_picture(name)
        self._send_preview()
        return True

    def _send_picture(self, name):
        if name != os.path.basename(name):
            return
        try:
            with open(os.path.join(cfg.SNAPSHOT_DIR, name), "rb") as f:
                data = f.read()
        except OSError:
            return
        self._call("/api/hub/picture?name=" + urllib.request.quote(name), data, "image/jpeg")

    def _send_preview(self):
        """A small camera picture, only while this computer's exam is in progress."""
        monitor = self.app.monitor
        s = self.app.session
        live = s is not None and s.finished is None
        monitor.want_preview(bool(live and self._send_frames))
        if not (live and self._send_frames):
            self._sent_frame_n = 0
            return
        now = time.monotonic()
        if now - self._last_frame < self._frame_every:
            return
        prev = monitor.preview()
        if prev is None or prev[0] == self._sent_frame_n:
            return
        self._last_frame, self._sent_frame_n = now, prev[0]
        self._call("/api/hub/frame", prev[1], "image/jpeg")

    # ----------------------------------------------------------------- the thread
    def run(self):
        while not self._halt.is_set():
            try:
                self.step()
            except (urllib.error.URLError, OSError, ValueError) as exc:
                self._set("offline", getattr(exc, "reason", None) or str(exc))
            except Exception as exc:                     # the exam must go on whatever happens here
                self._set("offline", f"unexpected problem: {exc}")
            self._halt.wait(self.seconds)
        self.app.monitor.want_preview(False)
        self.state = "stopped" if self.state != "wrong_code" else self.state

    def stop(self):
        self._halt.set()


def _computer_name():
    import socket
    try:
        return socket.gethostname()[:60]
    except OSError:
        return "Candidate computer"
