"""
The exam server: a small local web server (Python standard library only) that serves the exam
page and keeps the exam state. The page never receives the answer key, and the clock is the
server's, so changing the page's timer does nothing.

Order of events for a candidate:  start (name, ID, consent)  ->  gaze calibration (the page shows the
dots, the camera monitor collects them)  ->  begin (the clock starts)  ->  questions  ->  submit.

Only one candidate can sit the exam on this computer at a time. A page refresh picks the same
exam up again (the page keeps its token for the browser tab).
"""

import csv
import hmac
import json
import mimetypes
import os
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from facelib import config as cfg
from facelib.events import log_note

from .auth import AdminAuth
from .engine import ExamSession, QuestionError, load_questions  # noqa: F401
from .monitor import warning_text

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
BROWSER_EVENTS = {   # what the page may report -> is it a flag worth keeping in events.csv?
    "LEFT_EXAM_WINDOW": True, "RETURNED_TO_EXAM": False, "FULLSCREEN_EXITED": True,
    "COPY_ATTEMPT": True, "PASTE_ATTEMPT": True, "RIGHT_CLICK": False, "PAGE_RELOADED": False,
}
MAX_BODY = 20_000


class ExamApp:
    """Everything the handler needs; separate from HTTP so tests can drive it directly."""

    def __init__(self, monitor, questions_path=None, clock=None):
        self.title, self.questions = load_questions(questions_path or cfg.EXAM_QUESTIONS_FILE)
        self.monitor = monitor
        self.clock = clock
        self.session = None
        self.result_path = None
        self.lock = threading.RLock()
        self._camera_loss_logged = False
        self.auth = AdminAuth(cfg.EXAM_ADMIN_KEY)     # the invigilator key is made here, once per run
        self.session_number = 0                       # counts candidates registered since the server started

    # ------------------------------ helpers -------------------------------
    def _new_session(self, name, candidate_id):
        kw = {} if self.clock is None else {"clock": self.clock}
        return ExamSession(self.questions, name, candidate_id, defer_start=True, **kw)

    def gaze_wanted(self):
        """Gaze calibration is used when switched on and the camera is really running."""
        return bool(cfg.EXAM_GAZE) and self.monitor.status["state"] == "running"

    def info(self):
        status = self.monitor.status
        return {"title": self.title, "questions": len(self.questions), "minutes": cfg.EXAM_MINUTES,
                "camera": status, "in_progress": self.session is not None and self.session.finished is None,
                "show_score": cfg.EXAM_SHOW_SCORE, "gaze": self.gaze_wanted(),
                "show_gaze": bool(cfg.EXAM_SHOW_GAZE_DOT)}

    def _on_flag(self, name, seconds, detail, snapshot=None):
        with self.lock:
            if self.session is not None and self.session.finished is None:
                self.session.add_event("camera", name, detail, seconds, snapshot)

    def _context(self):
        s = self.session
        n = None if s is None else s.current_number
        return f"Q{n}" if n else ""

    def _finish(self, reason):
        s = self.session
        result = s.submit(reason)
        self.monitor.end_exam()
        if self.result_path is None:
            self.result_path = s.save_result()
            self._append_summary(result)
        return result

    def _append_summary(self, result):
        path = os.path.join(cfg.EXAM_RESULTS_DIR, "summary.csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["finished", "candidate", "candidate_id", "score", "out_of", "answered",
                            "finish_reason", "flag_count", "flag_names", "result_file"])
            w.writerow([result["finished"], result["candidate"], result["candidate_id"], result["score"],
                        result["out_of"], result["answered"], result["finish_reason"], len(result["flags"]),
                        " ".join(sorted(result["flag_counts"])), os.path.basename(self.result_path)])

    def _check_time(self):
        """Finish the exam by itself when time is up (whoever asks first triggers it)."""
        s = self.session
        if s is not None and s.finished is None and s.expired:
            self._finish("time up")

    def _check_camera(self):
        s, status = self.session, self.monitor.status
        if s is not None and s.begun and s.finished is None and status["state"] == "error" \
                and not self._camera_loss_logged:
            self._camera_loss_logged = True
            s.add_event("system", "CAMERA_MONITOR_STOPPED", status.get("message", ""))
            log_note("CAMERA_MONITOR_STOPPED", status.get("message", ""))

    def _auth(self, body):
        s = self.session
        if s is None or not hmac.compare_digest(str(body.get("token", "")), s.token):
            return None
        return s

    def _final_view(self):
        r = self.session.finished
        out = {"finished": True, "reason": r["finish_reason"], "answered": r["answered"],
               "out_of": r["out_of"], "show_score": cfg.EXAM_SHOW_SCORE}
        if cfg.EXAM_SHOW_SCORE:
            out["score"] = r["score"]
        return out

    # ------------------------------- routes -------------------------------
    def handle(self, method, path, body):
        """Return (status, json-able dict)."""
        with self.lock:
            self._check_time()
            self._check_camera()
            if method == "GET" and path == "/api/info":
                return 200, self.info()
            if method == "POST" and path == "/api/start":
                return self._start(body)
            if method == "POST" and path == "/api/admin/login":
                return self._admin_login(body)
            if method == "POST" and path == "/api/admin/logout":
                self.auth.logout(body.get("token"))
                return 200, {"ok": True}
            s = self._auth(body)
            if s is None:
                return 401, {"error": "No exam session for this token. Reload the page."}
            if method == "POST" and path == "/api/state":
                if s.finished is not None:
                    return 200, self._final_view()
                if body.get("w") and body.get("h"):
                    try:
                        self.monitor.set_gaze_view(bool(body.get("fs")), float(body["w"]), float(body["h"]))
                    except (TypeError, ValueError):
                        pass
                warn = ""
                if cfg.EXAM_SHOW_WARNINGS and s.begun:
                    warn = warning_text(self.monitor.status.get("active", []))
                return 200, {"finished": False, "remaining": round(s.remaining(), 1), "warning": warn,
                             "camera": self.monitor.status["state"]}
            if method == "POST" and path == "/api/exam":
                if s.finished is not None:
                    return 200, self._final_view()
                return 200, s.payload()
            if s.finished is not None:
                return 409, self._final_view()
            if method == "POST" and path.startswith("/api/cal/"):
                return self._calibration(s, path, body)
            if method == "POST" and path == "/api/begin":
                return self._begin(s)
            if method == "POST" and path == "/api/gaze":
                g = self.monitor.gaze() if cfg.EXAM_SHOW_GAZE_DOT and s.begun else None
                return 200, {"gaze": g}
            if method == "POST" and path == "/api/view":
                s.view(str(body.get("qid", "")))
                return 200, {"ok": True}
            if method == "POST" and path == "/api/answer":
                ok = s.answer(str(body.get("qid", "")), body.get("choice"))
                return (200, {"ok": True}) if ok else (409, {"error": "Answer not accepted (time is up?)"})
            if method == "POST" and path == "/api/mark":
                return 200, {"ok": s.mark(str(body.get("qid", "")), bool(body.get("on")))}
            if method == "POST" and path == "/api/event":
                return self._browser_event(s, body)
            if method == "POST" and path == "/api/submit":
                if not s.begun:
                    return 409, {"error": "The exam has not started."}
                self._finish("submitted by candidate")
                return 200, self._final_view()
            return 404, {"error": "Unknown request"}

    def _start(self, body):
        name, cid = str(body.get("name", "")).strip(), str(body.get("candidate_id", "")).strip()
        if not name or not cid or len(name) > 80 or len(cid) > 40:
            return 400, {"error": "Please enter your name and candidate ID."}
        if not body.get("consent"):
            return 400, {"error": "You must accept the monitoring notice to start."}
        if self.session is not None and self.session.finished is None:
            return 409, {"error": "An exam is already in progress on this computer."}
        state = self.monitor.status["state"]
        if state == "error":
            return 503, {"error": "The camera is not working: " + self.monitor.status.get("message", "")
                         + " Fix it, or restart with --no-camera."}
        if state == "starting":
            return 503, {"error": "The camera is still starting. Wait a moment and try again."}
        self.session, self.result_path, self._camera_loss_logged = self._new_session(name, cid), None, False
        self.session_number += 1
        log_note("EXAM_REGISTERED", f"{name} ({cid})")
        return 200, {"token": self.session.token, "gaze": self.gaze_wanted()}

    # --------------------------- calibration and begin ---------------------------
    def _calibration(self, s, path, body):
        if s.begun:
            return 409, {"error": "The exam has already started."}
        if path == "/api/cal/start":
            try:
                w, h = int(body.get("w")), int(body.get("h"))
            except (TypeError, ValueError):
                return 400, {"error": "Window size missing."}
            if not (300 <= w <= 10000 and 300 <= h <= 10000):
                return 400, {"error": "Window size out of range."}
            if not self.gaze_wanted():
                return 409, {"error": "Gaze tracking is not available."}
            if not self.monitor.begin_calibration(w, h):
                return 409, {"error": "A calibration is already running."}
            return 200, {"ok": True}
        if path == "/api/cal/status":
            c = self.monitor.calibration()
            return 200, c
        if path == "/api/cal/key":
            self.monitor.press_key(str(body.get("key", "")))
            return 200, {"ok": True}
        if path == "/api/cal/skip":
            self.monitor.skip_gaze()
            s.calibration = {"skipped": True, "reason": str(body.get("reason", ""))[:200]}
            s.add_event("system", "GAZE_TRACKING_SKIPPED", s.calibration["reason"])
            log_note("GAZE_TRACKING_SKIPPED", f"{s.name} ({s.candidate_id}) {s.calibration['reason']}")
            return 200, {"ok": True}
        return 404, {"error": "Unknown request"}

    def calibration_frame(self, token):
        """JPEG of the calibration screen for the page (None when the token is wrong or no picture)."""
        with self.lock:
            if self._auth({"token": token}) is None:
                return None
        return self.monitor.calibration_frame()

    def _begin(self, s):
        if s.begun:
            return 200, {"ok": True}
        cal = self.monitor.calibration()
        if self.gaze_wanted() and cal.get("state") not in ("done", "skipped"):
            return 409, {"error": "Calibrate your gaze first (or choose to continue without it)."}
        if cal.get("state") == "done":
            s.calibration = {k: v for k, v in cal.items() if k != "state"}
            log_note("GAZE_CALIBRATED", f"{s.name} ({s.candidate_id}) accuracy about "
                     f"{cal.get('accuracy_px')} px on {cal.get('screen')}")
        s.begin()
        self.monitor.begin_exam(self._on_flag, self._context)
        log_note("EXAM_STARTED", f"{s.name} ({s.candidate_id})")
        return 200, {"ok": True}

    def _browser_event(self, s, body):
        name = str(body.get("name", ""))
        if name not in BROWSER_EVENTS:
            return 400, {"error": "Unknown event"}
        if not s.begun:
            return 200, {"ok": True}      # before the exam starts nothing is recorded
        detail = str(body.get("detail", ""))[:200]
        event = s.add_event("browser", name, detail)
        if BROWSER_EVENTS[name]:
            log_note(name, f"Q{event['question']}: {detail}" if event["question"] else detail)
        return 200, {"ok": True}

    # ----------------------------- invigilator view -----------------------------
    def _admin_login(self, body):
        outcome, value = self.auth.login(body.get("key"))
        if outcome == "ok":
            log_note("INVIGILATOR_SIGNED_IN")
            return 200, {"token": value, "title": self.title}
        if outcome == "locked":
            wait = max(1, int(round(value)))
            log_note("INVIGILATOR_SIGN_IN_LOCKED", f"too many wrong keys, locked for {wait} s")
            return 429, {"error": f"Too many wrong keys. Wait {wait} seconds and try again.", "retry_seconds": wait}
        log_note("INVIGILATOR_WRONG_KEY")
        return 403, {"error": "That key is not right."}

    def admin_live(self, token):
        """What the invigilator page shows. None if the sign-in token is not valid."""
        if not self.auth.valid(token):
            return None
        with self.lock:
            self._check_time()
            s = self.session
            status = self.monitor.status
            out = {"title": self.title, "minutes": cfg.EXAM_MINUTES, "off_margin": cfg.GAZE_OFF_MARGIN,
                   "camera": status["state"], "camera_message": status.get("message", ""),
                   "faces": status.get("faces"), "light": status.get("light"),
                   "active": status.get("active", []), "gaze": self.monitor.gaze(),
                   "cal": self.monitor.calibration(), "session_number": self.session_number, "session": None}
            if s is not None:
                out["session"] = {
                    "candidate": s.name, "candidate_id": s.candidate_id, "begun": s.begun,
                    "finished": s.finished is not None, "reason": s.finish_reason,
                    "remaining": round(s.remaining(), 1), "question": s.current_number,
                    "answered": len(s.answers), "total": len(s.order),
                    "events": s.events[-60:], "event_count": len(s.events),
                    "score": s.finished["score"] if s.finished else None,
                    "calibration": s.calibration}
            return out

    def admin_history(self, token, limit=12):
        """The last finished candidates (from results/summary.csv), newest first. None if not signed in."""
        if not self.auth.valid(token):
            return None
        path = os.path.join(cfg.EXAM_RESULTS_DIR, "summary.csv")
        try:
            with open(path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        except OSError:
            rows = []
        keep = ("finished", "candidate", "candidate_id", "score", "out_of", "finish_reason", "flag_count")
        return [{k: r.get(k, "") for k in keep} for r in reversed(rows[-limit:])]

    def admin_snapshot(self, token, name):
        """The picture taken when a camera flag fired, as JPEG bytes. Only pictures that belong to the
        current candidate's flags can be asked for, so no other file on the computer is reachable."""
        if not self.auth.valid(token):
            return None
        with self.lock:
            s = self.session
            allowed = set() if s is None else {e.get("snapshot") for e in s.events}
        if not name or name != os.path.basename(name) or name not in allowed:
            return None
        try:
            with open(os.path.join(cfg.SNAPSHOT_DIR, name), "rb") as f:
                return f.read()
        except OSError:
            return None

    def admin_result(self, token):
        """The finished candidate's full result record, or None."""
        if not self.auth.valid(token):
            return None
        with self.lock:
            s = self.session
            return None if s is None or s.finished is None else s.finished


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ExamServer"

        def log_message(self, *args):   # keep the console for flags
            pass

        def _send(self, status, payload, ctype="application/json", raw=None):
            data = raw if raw is not None else json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def _allowed_host(self):
            """On a loopback-only server only 127.0.0.1 / localhost names are served, which stops a
            web page on another site from reaching the exam through a rebinding trick."""
            if cfg.EXAM_HOST not in ("127.0.0.1", "localhost", "::1"):
                return True
            host = self.headers.get("Host", "")
            name = host.rsplit(":", 1)[0] if not host.startswith("[") else host.split("]")[0] + "]"
            return name in ("127.0.0.1", "localhost", "[::1]")

        def _allowed_origin(self):
            """Block pages from other sites poking the local server (a browser sends Origin)."""
            if not self._allowed_host():
                return False
            origin = self.headers.get("Origin")
            if not origin:
                return True
            host = self.headers.get("Host", "")
            return origin in (f"http://{host}",)

        def do_GET(self):
            if not self._allowed_host():
                return self._send(403, {"error": "Forbidden"})
            parts = urlsplit(self.path)
            path, query = parts.path, parse_qs(parts.query)
            token = self.headers.get("X-Admin-Token", "")
            if path == "/api/cal/frame":
                data = app.calibration_frame(query.get("token", [""])[0])
                if data is None:
                    return self._send(404, {"error": "No picture"})
                return self._send(200, None, "image/jpeg", data)
            if path in ("/api/admin/live", "/api/admin/history", "/api/admin/result"):
                data = {"/api/admin/live": app.admin_live, "/api/admin/history": app.admin_history,
                        "/api/admin/result": app.admin_result}[path](token)
                if data is None and not app.auth.valid(token):
                    return self._send(403, {"error": "Not signed in"})
                if data is None:
                    return self._send(404, {"error": "Nothing to show yet"})
                return self._send(200, data)
            if path == "/api/admin/snapshot":
                if not app.auth.valid(token):
                    return self._send(403, {"error": "Not signed in"})
                data = app.admin_snapshot(token, query.get("name", [""])[0])
                if data is None:
                    return self._send(404, {"error": "No such picture"})
                return self._send(200, None, "image/jpeg", data)
            if path.startswith("/api/"):
                status, payload = app.handle("GET", path, {})
                return self._send(status, payload)
            name = {"/": "index.html", "/index.html": "index.html", "/admin": "admin.html",
                    "/invigilator": "admin.html"}.get(path, path.lstrip("/"))
            full = os.path.normpath(os.path.join(STATIC_DIR, name))
            if not full.startswith(STATIC_DIR + os.sep) or not os.path.isfile(full):
                return self._send(404, {"error": "Not found"})
            with open(full, "rb") as f:
                self._send(200, None, mimetypes.guess_type(full)[0] or "application/octet-stream", f.read())

        def do_POST(self):
            if not self._allowed_origin():
                return self._send(403, {"error": "Forbidden"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > MAX_BODY:
                    return self._send(413, {"error": "Too large"})
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError
            except ValueError:
                return self._send(400, {"error": "Bad request"})
            status, payload = app.handle("POST", self.path.split("?")[0], body)
            self._send(status, payload)
    return Handler


def serve(app, host=None, port=None):
    """Create the HTTP server (not yet running)."""
    return ThreadingHTTPServer((host or cfg.EXAM_HOST, cfg.EXAM_PORT if port is None else port), make_handler(app))
