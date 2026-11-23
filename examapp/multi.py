"""
One exam computer, many candidates, each in a web browser.

The candidate opens the exam's address, types the personal access code from the registered list, allows the camera
and sits the exam. Nothing is installed on the candidate's computer. This computer (the host) keeps the questions,
the answer key, the clock and the marking, analyses every candidate's camera pictures, and shows all the candidates to
the invigilator in the same grid and list as a room.

How it is put together (nothing new is invented for the candidate's exam itself):

* A Sitting is one candidate's attempt: an ExamApp (the same exam logic as the one-computer exam) with a
  StreamMonitor (the same camera checks, fed with the pictures the browser posts) and a LocalReporter that gives the
  invigilator's room the same reports a candidate computer in a room would send.
* MultiHost is the ExamApp the invigilator console talks to. It adds `/api/join` (access code in, sitting token out)
  and sends every other request to the sitting the token belongs to.
* The access code is the credential. A second browser using the same code takes over the sitting (the first one is
  cut off, both are recorded), so a crashed browser can come back, and so that someone else using a candidate's code
  is visible to the invigilator.
"""

import os
import secrets
import threading
import time
from collections import OrderedDict

from facelib import config as cfg
from facelib.events import log_note
from .browsercam import StreamMonitor
from .hub import Room
from .monitor import NullMonitor
from .reporter import Reporter
from .server import ExamApp, _json_body
from .store import ExamStateError, StoreError


class FailureGate:
    """Slows down guessing of access codes: too many wrong codes from one address locks that address for a while."""

    def __init__(self, clock=time.monotonic, max_failures=8, window=60.0, lock_seconds=30.0):
        self.clock, self.max_failures, self.window, self.lock_seconds = clock, max_failures, window, lock_seconds
        self._fails = {}                       # ip -> [times of recent failures]
        self._locked = {}                      # ip -> time until which it is locked
        self._lock = threading.Lock()

    def wait(self, ip):
        """Seconds this address must still wait, or 0."""
        with self._lock:
            until = self._locked.get(ip, 0.0)
            now = self.clock()
            if until > now:
                return until - now
            self._locked.pop(ip, None)
            return 0.0

    def failed(self, ip):
        with self._lock:
            now = self.clock()
            times = [t for t in self._fails.get(ip, []) if now - t < self.window] + [now]
            self._fails[ip] = times
            if len(times) >= self.max_failures:
                self._locked[ip] = now + self.lock_seconds
                self._fails[ip] = []
            if len(self._fails) > 5000:        # never grow without bound
                self._fails = {k: v for k, v in self._fails.items() if v and now - v[-1] < self.window}

    def passed(self, ip):
        with self._lock:
            self._fails.pop(ip, None)


class LocalReporter(Reporter):
    """The Reporter of a candidate computer in a room, but talking to the room in this same program."""

    def __init__(self, app, host, label, ip=""):
        super().__init__(app, "127.0.0.1:1", host.room.code, label=label)
        self.host, self.ip = host, ip

    def _send_preview(self):
        """Pictures for the invigilator come straight from the browser's pictures (MultiHost._push_loop), not from here."""
        return

    def _call(self, path, data, ctype="application/json", token=True):
        path, _, query = path.partition("?")
        name = ""
        if query.startswith("name="):
            from urllib.parse import unquote
            name = unquote(query[5:])
        status, payload = self.host.hub_post(path, self.token if token else "", name, data, self.ip)
        return status, payload or {}


class Sitting:
    """One candidate's attempt in a browser."""

    def __init__(self, host, candidate, ip):
        self.host, self.candidate, self.ip = host, candidate, ip
        self.joined = time.time()
        self.monitor = StreamMonitor()
        app = self.app = ExamApp(self.monitor, exam=host.exam, clock=host.clock, minutes=host.minutes_override)
        app.browser_camera = True
        app.store = host.store
        app.context_prefix = candidate["candidate_id"]
        app.on_finish = self._finished
        session = app.register(candidate["name"], candidate["candidate_id"])
        self.token = session.token
        self.reporter = LocalReporter(app, host, candidate["name"], ip)
        self.done = False

    def start(self):
        self.monitor.start()
        self.reporter.start()

    def retoken(self):
        """A new browser took this sitting over: the old browser's token stops working."""
        self.token = secrets.token_urlsafe(16)
        self.app.session.token = self.token
        return self.token

    def _finished(self, result):
        self.done = True
        self.monitor.stop()                        # frees the analysis thread; the reporter goes on showing the result
        self.host._sitting_done(self, result)

    def stop(self):
        self.reporter.stop()
        self.monitor.stop()


class MultiHost(ExamApp):
    """The exam computer for browser candidates. Everything the invigilator console needs comes from ExamApp
    (sign-in, the room, pictures); this class adds the candidates' side."""

    def __init__(self, exam, store, clock=None, minutes=None, max_sittings=None, gate=None, open_join=False, names_file=None):
        super().__init__(NullMonitor(), exam=exam, clock=clock, room=Room(), minutes=minutes)
        self.exam, self.store = exam, store
        self.minutes_override = minutes
        self.max_sittings = cfg.BROWSER_MAX_SITTINGS if max_sittings is None else max_sittings
        self.gate = gate or FailureGate()
        self.open_join = bool(open_join)           # demo: no codes, the candidate types name and ID and is registered on the spot
        self.names_file = names_file
        self.sittings = OrderedDict()              # candidate_id -> Sitting
        self._by_token = {}
        self._sit_lock = threading.RLock()
        self._halt = threading.Event()
        threading.Thread(target=self._push_loop, name="live-pictures", daemon=True).start()

    # ------------------------------------------------------------------ public side
    def info(self):
        """What the start page of the candidates needs (no camera state: every candidate has their own)."""
        return {"title": self.title, "questions": len(self.questions), "minutes": self.minutes,
                "exam_id": self.exam_id, "browser": True, "open_join": self.open_join, "room": True, "face": bool(cfg.EXAM_FACE_MATCH),
                "face_required": bool(self.rule("face_required")), "gaze": bool(self.rule("gaze")),
                "show_score": bool(self.rule("show_score")), "show_gaze": bool(cfg.EXAM_SHOW_GAZE_DOT),
                "frame_fps": cfg.BROWSER_FRAME_FPS, "frame_width": cfg.BROWSER_FRAME_WIDTH,
                "frame_quality": cfg.BROWSER_FRAME_QUALITY, "in_progress": False, "camera": {"state": "off", "message": ""}}

    def handle(self, method, path, body, ip=""):
        if method == "GET" and path == "/api/info":
            return 200, self.info()
        if method == "POST" and path == "/api/join":
            return self._join(body, ip)
        if path in ("/api/admin/login", "/api/admin/logout"):
            return super().handle(method, path, body, ip)
        sitting = self._find(body.get("token"))
        if sitting is None:
            return 401, {"error": "No exam session for this token. Reload the page."}
        return sitting.app.handle(method, path, body, ip)

    def frame(self, token, raw, ip=""):
        sitting = self._find(token)
        if sitting is None:
            return 401, {"error": "No exam session for this token."}
        s = sitting.app.session
        if s is None or s.finished is not None or sitting.done:
            return 410, {"error": "The exam is over."}
        result = sitting.monitor.put_frame(raw)
        if result != "ok":
            return 400, {"error": "Not a usable camera picture."}
        return 200, {"ok": True, "fps": sitting.app.frame_fps()}

    def calibration_frame(self, token):
        sitting = self._find(token)
        return None if sitting is None else sitting.app.calibration_frame(token)

    def _find(self, token):
        token = str(token or "")
        with self._sit_lock:
            return self._by_token.get(token)

    # ------------------------------------------------------------------ joining
    def _join(self, body, ip):
        if self.open_join:
            return self._join_open(body, ip)
        wait = self.gate.wait(ip)
        if wait > 0:
            return 429, {"error": f"Too many wrong codes. Wait {int(wait) + 1} seconds and try again.",
                         "retry_seconds": int(wait) + 1}
        cand = self.store.find_candidate(body.get("code"))
        if cand is None or cand["exam_id"] != self.exam_id:
            self.gate.failed(ip)
            log_note("JOIN_WRONG_CODE", ip)
            return 403, {"error": "That access code is not right. Check it against your slip."}
        self.gate.passed(ip)
        return self._join_candidate(cand, body, ip)

    def _join_open(self, body, ip):
        """Demo mode: no access code. The name and ID the candidate types register them, and the same ID with the same
        name later is the same person coming back (a takeover, as with a code)."""
        name, cid = str(body.get("name") or "").strip(), str(body.get("candidate_id") or "").strip()
        if not name or not cid:
            return 400, {"error": "Type your full name and your candidate ID."}
        with self._sit_lock:
            cand = None
            for c in self.store.list_candidates(self.exam_id):
                if c["candidate_id"].lower() == cid.lower():
                    if c["name"].strip().lower() != name.lower():
                        return 409, {"error": "That candidate ID is already used by someone with another name. Check your ID."}
                    cand = c
                    break
            if cand is None:
                if not body.get("consent"):
                    return 400, {"error": "You must accept the monitoring notice to join."}
                try:
                    cand = self.store.add_candidates(self.exam_id, [{"name": name, "candidate_id": cid}], "open-join")[0]
                except (StoreError, ExamStateError) as e:
                    return 400, {"error": str(e)}
                self._write_names()
        return self._join_candidate(cand, body, ip)

    def _write_names(self):
        """The list of who joined, as a CSV next to the program (name, id, code, status), rewritten whenever it changes."""
        if not self.names_file:
            return
        try:
            tmp = self.names_file + ".part"
            with open(tmp, "w", encoding="utf-8", newline="") as f:
                f.write(self.store.export_candidates_csv(self.exam_id))
            os.replace(tmp, self.names_file)
        except OSError as e:
            log_note("NAMES_FILE_FAILED", str(e))

    def _join_candidate(self, cand, body, ip):
        if not body.get("consent"):
            return 400, {"error": "You must accept the monitoring notice to join."}
        exam = self.store.get(self.exam_id)
        with self._sit_lock:
            existing = self.sittings.get(cand["candidate_id"])
            if cand["status"] == "submitted" or (existing is not None and existing.done):
                return 409, {"error": "You have already submitted this exam."}
            if exam.status != "running":
                return 409, {"error": f"This exam is {exam.status}, so it cannot be joined now."}
            if existing is not None and cand["status"] == "registered":
                self._discard(existing)                # the invigilator reset this candidate: start again from nothing
                existing = None
            if existing is not None:
                return self._resume(existing, ip)
            if cand["status"] == "in_progress":
                return 409, {"error": "Your earlier sitting was interrupted and cannot be resumed. "
                                      "Ask the invigilator to let you sit again."}
            if len(self.sittings) >= self.max_sittings:
                return 503, {"error": "This exam computer is full. Tell the invigilator."}
            try:
                self.store.mark_candidate(self.exam_id, cand["candidate_id"], "in_progress", "exam-app", "joined from " + ip)
            except (StoreError, ExamStateError) as e:
                return 409, {"error": str(e)}
            sitting = Sitting(self, cand, ip)
            self.sittings[cand["candidate_id"]] = sitting
            self._by_token[sitting.token] = sitting
            sitting.start()
            self._audit("sitting_joined", f"candidate {cand['candidate_id']} from {ip}")
            self._write_names()
            return 200, self._joined(sitting, False)

    def _discard(self, sitting):
        self._by_token.pop(sitting.token, None)
        self.sittings.pop(sitting.candidate["candidate_id"], None)
        sitting.stop()
        if sitting.reporter.token:
            self.room.drop(sitting.reporter.token)
        self._audit("sitting_discarded", f"candidate {sitting.candidate['candidate_id']} (reset by the invigilator)")

    def _resume(self, sitting, ip):
        """The same code again: the old browser is cut off, the new one takes the sitting."""
        old = sitting.token
        self._by_token.pop(old, None)
        token = sitting.retoken()
        self._by_token[token] = sitting
        cid = sitting.candidate["candidate_id"]
        sitting.app.session.add_event("system", "REJOINED", f"joined again from {ip} (was {sitting.ip}); the earlier browser was cut off")
        log_note("SITTING_REJOINED", f"{cid} from {ip} (was {sitting.ip})")
        self._audit("sitting_rejoined", f"candidate {cid} from {ip}, was {sitting.ip}")
        sitting.ip = ip
        sitting.reporter.ip = ip
        return 200, self._joined(sitting, True)

    def _joined(self, sitting, resumed):
        return {"token": sitting.token, "resumed": resumed, "name": sitting.candidate["name"],
                "candidate_id": sitting.candidate["candidate_id"], "begun": sitting.app.session.begun}

    # ------------------------------------------------------------------ finishing and shutting down
    def _sitting_done(self, sitting, result):
        try:
            self.store.mark_candidate(self.exam_id, sitting.candidate["candidate_id"], "submitted", "exam-app", result["finish_reason"])
        except (StoreError, ExamStateError) as e:
            log_note("CANDIDATE_STATUS_FAILED", str(e))
        self._write_names()

    def counts(self):
        with self._sit_lock:
            total = len(self.sittings)
            done = sum(1 for s in self.sittings.values() if s.done)
        return {"sittings": total, "finished": done, "in_progress": total - done}

    def _push_loop(self):
        """Gives the invigilator's room the newest picture of every candidate: about twice a second each for a small room, and
        several times a second for the candidate whose detail is open. No network is involved: the pictures are on this computer."""
        last = {}
        while not self._halt.wait(0.1):
            try:
                with self._sit_lock:
                    sittings = [s for s in self.sittings.values() if not s.done]
                now = time.monotonic()
                for s in sittings:
                    tok = s.reporter.token
                    if not tok:
                        continue
                    interval = 0.3 if self.room.watched(tok) else (0.5 if len(sittings) <= 12 else 1.0)
                    n, jpeg = s.monitor.feed.latest()
                    seen = last.get(tok, (0, -1e9))
                    if jpeg is None or n == seen[0] or now - seen[1] < interval:
                        continue
                    if self.room.put_frame(tok, jpeg) == "ok":
                        last[tok] = (n, now)
            except Exception as e:      # noqa: BLE001  the exam must go on whatever happens here
                log_note("LIVE_PICTURES_FAILED", str(e))

    def stop(self):
        self._halt.set()
        with self._sit_lock:
            for sitting in self.sittings.values():
                sitting.stop()
