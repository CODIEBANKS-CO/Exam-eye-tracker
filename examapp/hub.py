"""
The room: many candidate computers, one invigilator computer.

Each candidate's computer runs the exam and the camera checks as it always has. It also reports to the
invigilator's computer (the "hub"): what the candidate is doing, the flags, a small camera picture every
few seconds, and the picture taken at each flag. The hub keeps all of that in memory and shows it in the
invigilator console as a grid or a list.

Trust: a candidate computer is not trusted. Everything it sends is checked and cut down to known fields and
sizes before it is kept, pictures must really be JPEG files, and a computer can only change its own entry.
Joining needs the room code the hub prints (five wrong codes lock joining for 30 seconds).

Privacy: the hub only keeps a camera picture while that candidate's exam is in progress (after consent), and
it throws the picture away when the exam ends. Nothing is written to disk here except short log lines.
"""

import hmac
import math
import re
import secrets
import threading
import time
from collections import OrderedDict

from facelib import config as cfg
from facelib.events import log_note

from .auth import AdminAuth, make_key

MAX_CLIENTS = 300
MAX_FRAME = 150_000        # bytes of one preview picture
MAX_PICTURE = 400_000      # bytes of one flag picture
MAX_PICTURES = 60          # flag pictures kept per candidate computer
MAX_EVENTS = 80            # events kept per report
FRAME_EVERY = 2.5          # seconds between preview pictures (slower with a big room, see frame_every())
MINOR = {"RETURNED_TO_EXAM", "RIGHT_CLICK", "PAGE_RELOADED", "FACE_ENROLLED"}
NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,120}$")
CAMERA_STATES = {"starting", "running", "error", "off"}
SOURCES = {"camera", "browser", "system", "hub"}


# ------------------------------- cleaning what a candidate computer sends -------------------------------
def _text(v, limit):
    if not isinstance(v, str):
        return ""
    return "".join(c for c in v if c == " " or (c.isprintable() and c not in "  "))[:limit]


def _num(v, lo=-1e9, hi=1e9, default=None):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return default
    if not math.isfinite(v):
        return default
    return max(lo, min(hi, float(v)))


def _int(v, lo=0, hi=1_000_000, default=None):
    n = _num(v, lo, hi, None)
    return default if n is None else int(n)


def _flag(v):
    return bool(v) if isinstance(v, (bool, int)) else False


def _snap(v):
    return v if isinstance(v, str) and NAME_RE.match(v) else None


def _event(e):
    if not isinstance(e, dict):
        return None
    name = _text(e.get("name"), 60)
    if not name:
        return None
    source = e.get("source") if e.get("source") in SOURCES else "system"
    return {"time_s": _num(e.get("time_s"), 0, 1e6, 0.0), "question": _int(e.get("question"), 0, 1000),
            "source": source, "name": name, "detail": _text(e.get("detail"), 240),
            "seconds": _num(e.get("seconds"), 0, 1e5), "snapshot": _snap(e.get("snapshot"))}


def _small(d, floats=(), ints=(), flags=(), texts=()):
    """Keep only the named fields of a dict, each cut to its type."""
    out = {}
    if not isinstance(d, dict):
        return out
    for k in floats:
        out[k] = _num(d.get(k))
    for k in ints:
        out[k] = _int(d.get(k))
    for k in flags:
        out[k] = _flag(d.get(k))
    for k, n in texts:
        out[k] = _text(d.get(k), n)
    return out


def clean_report(raw):
    """What the console is allowed to show about one candidate computer. Never raises."""
    raw = raw if isinstance(raw, dict) else {}
    cal = raw.get("cal") if isinstance(raw.get("cal"), dict) else {}
    out = {
        "title": _text(raw.get("title"), 120) or "Exam",
        "minutes": _num(raw.get("minutes"), 0, 10_000, 0.0),
        "off_margin": _num(raw.get("off_margin"), 0, 5, 0.1),
        "camera": raw.get("camera") if raw.get("camera") in CAMERA_STATES else "starting",
        "camera_message": _text(raw.get("camera_message"), 200),
        "faces": _int(raw.get("faces"), 0, 50),
        "light": _num(raw.get("light"), 0, 255),
        "active": [_text(a, 60) for a in (raw.get("active") or [])[:20] if isinstance(a, str) and a],
        "session_number": _int(raw.get("session_number"), 0, 1_000_000, 0),
        "elapsed": _num(raw.get("elapsed"), 0, 1e6),
    }
    g = raw.get("gaze")
    out["gaze"] = None if not isinstance(g, dict) or _num(g.get("x")) is None or _num(g.get("y")) is None else {
        "x": _num(g.get("x"), -50, 50), "y": _num(g.get("y"), -50, 50), "off": _flag(g.get("off")), "paused": _flag(g.get("paused"))}
    scr = cal.get("screen")
    out["cal"] = _small(cal, floats=("accuracy_px", "accuracy_fraction"), flags=("good",), texts=(("state", 20),))
    out["cal"]["state"] = out["cal"]["state"] or "idle"
    out["cal"]["screen"] = [_num(scr[0], 1, 100_000, 1), _num(scr[1], 1, 100_000, 1)] if (
        isinstance(scr, (list, tuple)) and len(scr) == 2 and _num(scr[0]) and _num(scr[1])) else None
    i = raw.get("identity")
    out["identity"] = None if not isinstance(i, dict) else {
        **_small(i, floats=("score", "threshold"), ints=("since_verified",), texts=(("status", 20),))}
    f = raw.get("face")
    out["face"] = None if not isinstance(f, dict) else _small(f, floats=("progress",), texts=(("state", 20), ("hint", 120), ("message", 200)))
    s = raw.get("session")
    if not isinstance(s, dict):
        out["session"] = None
    else:
        events = [x for x in (_event(e) for e in (s.get("events") or [])[-MAX_EVENTS:]) if x]
        fm = s.get("face_match")
        cal_s = s.get("calibration")
        out["session"] = {
            "candidate": _text(s.get("candidate"), 80), "candidate_id": _text(s.get("candidate_id"), 40),
            "begun": _flag(s.get("begun")), "finished": _flag(s.get("finished")), "reason": _text(s.get("reason"), 60),
            "remaining": _num(s.get("remaining"), 0, 1e6, 0.0), "question": _int(s.get("question"), 0, 1000),
            "answered": _int(s.get("answered"), 0, 1000, 0), "total": _int(s.get("total"), 0, 1000, 0),
            "events": events, "event_count": _int(s.get("event_count"), 0, 1_000_000, len(events)),
            "score": _int(s.get("score"), 0, 1000),
            "calibration": {"skipped": _flag(cal_s.get("skipped"))} if isinstance(cal_s, dict) else None,
            "face_match": None if not isinstance(fm, dict) else {
                **_small(fm, ints=("checks", "failed_checks", "reference_pictures"), floats=("lowest_score", "last_score", "threshold"),
                         flags=("skipped", "enrolled")), "reference_snapshot": _snap(fm.get("reference_snapshot"))},
        }
    return out


# ------------------------------------------- the room --------------------------------------------
class Client:
    def __init__(self, cid, secret, label, ip, now, wall):
        self.cid, self.secret, self.label, self.ip = cid, secret, label, ip
        self.joined = wall
        self.last_seen = now
        self.report = None
        self.hub_events = []                 # connection events the hub saw itself
        self.frame = None                    # (n, bytes, wall time)
        self.frame_n = 0
        self.pictures = OrderedDict()        # flag pictures: name -> bytes
        self.lost = False
        self.lost_at = None                  # monotonic time of the loss
        self.lost_elapsed = None
        self.reports = 0


class Room:
    def __init__(self, code=None, clock=time.monotonic, wall=time.time, lost_seconds=None,
                 max_clients=MAX_CLIENTS, max_failures=5, lock_seconds=30.0):
        self.gate = AdminAuth(code, clock=clock, max_failures=max_failures, lock_seconds=lock_seconds)
        self.code = self.gate.key
        self.clock, self.wall = clock, wall
        self.lost_seconds = cfg.ROOM_LOST_SECONDS if lost_seconds is None else lost_seconds
        self.max_clients = max_clients
        self.clients = OrderedDict()
        self.lock = threading.RLock()

    # ---------------------------------- joining and reporting ----------------------------------
    def join(self, code, label="", ip=""):
        """('ok', token) | ('wrong', None) | ('locked', seconds) | ('full', None)"""
        with self.lock:
            if len(self.clients) >= self.max_clients:
                return "full", None
            outcome, value = self.gate.login(code)
            if outcome != "ok":
                return outcome, value
            cid, secret = secrets.token_hex(6), secrets.token_urlsafe(18)
            c = Client(cid, secret, _text(label, 60) or "Candidate computer", _text(ip, 45), self.clock(), self.wall())
            self.clients[cid] = c
            log_note("ROOM_COMPUTER_JOINED", f"{c.label} ({c.ip})")
            return "ok", f"{cid}.{secret}"

    def _find(self, token):
        cid, _, secret = str(token or "").partition(".")
        c = self.clients.get(cid)
        if c is None or not hmac.compare_digest(secret, c.secret):
            return None
        return c

    def report(self, token, raw):
        """Store one report. Returns None if the token is not valid, else {'frame_every', 'need'}."""
        with self.lock:
            c = self._find(token)
            if c is None:
                return None
            now = self.clock()
            self._refresh(now)
            clean = clean_report(raw)
            was_lost = c.lost
            c.last_seen, c.reports = now, c.reports + 1
            c.report = clean
            if was_lost:
                self._event(c, "CONNECTION_RESTORED", f"back after {int(now - c.lost_at)} s")
                c.lost, c.lost_at = False, None
            s = clean["session"]
            if s is None or s["finished"]:
                c.frame = None                  # a picture is kept only while the exam is in progress
            have = set(c.pictures)
            need = []
            if s:
                names = [e["snapshot"] for e in s["events"] if e["snapshot"]]
                if s["face_match"] and s["face_match"].get("reference_snapshot"):
                    names.append(s["face_match"]["reference_snapshot"])
                need = [n for n in dict.fromkeys(reversed(names)) if n not in have][:4]
            return {"frame_every": self.frame_every(), "need": need,
                    "send_frames": bool(s and not s["finished"])}

    def frame_every(self):
        """A bigger room sends its pictures a little less often, so the hub keeps up."""
        return round(min(6.0, FRAME_EVERY + 0.04 * max(0, len(self.clients) - 10)), 1)

    def put_frame(self, token, data):
        with self.lock:
            c = self._find(token)
            if c is None:
                return "denied"
            s = c.report and c.report["session"]
            if not s or s["finished"]:
                return "not_now"
            if not _is_jpeg(data, MAX_FRAME):
                return "bad"
            c.frame_n += 1
            c.frame = (c.frame_n, data, self.wall())
            c.last_seen = self.clock()
            return "ok"

    def put_picture(self, token, name, data):
        with self.lock:
            c = self._find(token)
            if c is None:
                return "denied"
            if not NAME_RE.match(str(name or "")) or not _is_jpeg(data, MAX_PICTURE):
                return "bad"
            c.pictures[name] = data
            c.pictures.move_to_end(name)
            while len(c.pictures) > MAX_PICTURES:
                c.pictures.popitem(last=False)
            c.last_seen = self.clock()
            return "ok"

    # ---------------------------------- connection tracking ----------------------------------
    def _in_exam(self, c):
        s = c.report and c.report["session"]
        return bool(s and s["begun"] and not s["finished"])

    def _elapsed(self, c):
        r = c.report
        if not r:
            return 0.0
        if r.get("elapsed") is not None:
            return r["elapsed"]
        s = r["session"]
        return max(0.0, r["minutes"] * 60 - s["remaining"]) if s and s["begun"] else 0.0

    def _event(self, c, name, detail):
        s = c.report and c.report["session"]
        c.hub_events.append({"time_s": round(self._elapsed(c), 1), "question": (s or {}).get("question"), "source": "hub",
                             "name": name, "detail": detail, "seconds": None, "snapshot": None})
        del c.hub_events[:-40]
        if self._in_exam(c):
            log_note(name, f"{c.label}: {(s or {}).get('candidate', '')} {detail}".strip())

    def _refresh(self, now):
        for c in self.clients.values():
            if not c.lost and c.report is not None and now - c.last_seen > self.lost_seconds:
                c.lost, c.lost_at = True, c.last_seen
                self._event(c, "CONNECTION_LOST", "no report from the candidate's computer")

    # ------------------------------------ what the console reads ------------------------------------
    def _state(self, c):
        r, s = c.report, c.report and c.report["session"]
        if c.lost:
            return "lost"
        if r is None:
            return "connecting"
        if s is None:
            return "waiting"
        if s["finished"]:
            return "finished"
        return "in_exam" if s["begun"] else "setting_up"

    def _severity(self, c):
        """0 quiet, 1 working, 2 a flag in the last minute, 3 needs a look now."""
        r, state = c.report, self._state(c)
        if state == "lost":
            return 3 if self._in_exam(c) else 1
        if r is None or state in ("waiting", "connecting", "finished"):
            return 0
        if r["camera"] == "error" or [a for a in r["active"]]:
            return 3
        s = r["session"]
        recent = [e for e in s["events"] if e["name"] not in MINOR and e["source"] in ("camera", "browser") and
                  s["begun"] and self._elapsed(c) - e["time_s"] <= 60]
        return 2 if recent else 1

    def summary(self):
        with self.lock:
            now = self.clock()
            self._refresh(now)
            tiles = []
            for c in self.clients.values():
                r = c.report
                s = r["session"] if r else None
                flags = [e for e in (s["events"] if s else []) if e["name"] not in MINOR and e["source"] in ("camera", "browser")]
                tiles.append({
                    "id": c.cid, "label": c.label, "state": self._state(c), "severity": self._severity(c),
                    "name": s["candidate"] if s else "", "candidate_id": s["candidate_id"] if s else "",
                    "question": s["question"] if s else None, "answered": s["answered"] if s else 0, "total": s["total"] if s else 0,
                    "remaining": s["remaining"] if s else None, "score": s["score"] if s else None,
                    "active": r["active"] if r else [], "flag_count": len(flags),
                    "last_flag": flags[-1]["name"] if flags else None,
                    "camera": r["camera"] if r else None, "faces": r["faces"] if r else None, "light": r["light"] if r else None,
                    "identity": r["identity"]["status"] if r and r["identity"] else None,
                    "frame": c.frame[0] if c.frame else 0, "frame_age": None if not c.frame else round(self.wall() - c.frame[2], 1),
                    "age": round(now - c.last_seen, 1), "lost_for": round(now - c.lost_at, 1) if c.lost else None,
                })
            return {"code": self.code, "frame_every": self.frame_every(), "clients": tiles,
                    "counts": {"all": len(tiles), "lost": sum(t["state"] == "lost" for t in tiles),
                               "in_exam": sum(t["state"] == "in_exam" for t in tiles),
                               "flagged": sum(t["severity"] >= 3 for t in tiles)}}

    def detail(self, cid):
        """The same shape as the single-computer console reads, plus the connection."""
        with self.lock:
            now = self.clock()
            self._refresh(now)
            c = self.clients.get(cid)
            if c is None or c.report is None:
                return None
            d = dict(c.report)
            s = d["session"]
            if s is not None:
                events = sorted(s["events"] + c.hub_events, key=lambda e: e["time_s"])
                d["session"] = {**s, "events": events[-60:], "event_count": s["event_count"] + len(c.hub_events)}
            d["connection"] = {"id": c.cid, "label": c.label, "ip": c.ip, "connected": not c.lost,
                               "lost_for": round(now - c.lost_at, 1) if c.lost else None, "age": round(now - c.last_seen, 1)}
            d["frame"] = c.frame[0] if c.frame else 0
            return d

    def frame(self, cid):
        with self.lock:
            c = self.clients.get(cid)
            return None if c is None or c.frame is None else c.frame[1]

    def picture(self, cid, name):
        """Only a picture the candidate's own events (or its face reference) name can be asked for."""
        with self.lock:
            c = self.clients.get(cid)
            r = c and c.report
            s = r and r["session"]
            if not s or not name:
                return None
            allowed = {e["snapshot"] for e in s["events"]}
            if s["face_match"]:
                allowed.add(s["face_match"].get("reference_snapshot"))
            return c.pictures.get(name) if name in allowed else None

    def forget(self, cid):
        """Remove a computer that is gone. Only allowed once it has lost its connection or finished."""
        with self.lock:
            c = self.clients.get(cid)
            if c is None:
                return "missing"
            self._refresh(self.clock())
            if not (c.lost or self._state(c) in ("finished", "waiting")):
                return "busy"
            del self.clients[cid]
            return "ok"


def _is_jpeg(data, limit):
    return isinstance(data, (bytes, bytearray)) and 4 < len(data) <= limit and data[:3] == b"\xff\xd8\xff" and data[-2:] == b"\xff\xd9"
