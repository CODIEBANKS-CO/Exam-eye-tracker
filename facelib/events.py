"""Event log: one CSV row plus a snapshot picture each time a flag fires."""

import csv
import os
import threading
from datetime import datetime

import cv2

from . import config as cfg

LOG_HEADER = ["timestamp", "event", "seconds_at_flag", "snapshot", "detail"]
# Several candidates are monitored by one program in a room, each in its own thread: one lock keeps their rows and
# snapshot names from mixing (two candidates can raise the same flag in the same second).
_LOCK = threading.RLock()


def log_event(name, seconds, frame, detail=""):
    """Append a row to the CSV log and save a snapshot of the raw frame. Returns the snapshot's path."""
    with _LOCK:
        return _log_event(name, seconds, frame, detail)


def _log_event(name, seconds, frame, detail):
    os.makedirs(cfg.SNAPSHOT_DIR, exist_ok=True)
    stamp = datetime.now()
    snap = os.path.join(cfg.SNAPSHOT_DIR, f"{stamp:%Y%m%d_%H%M%S}_{name}.jpg")
    n = 1
    while os.path.exists(snap):                  # never overwrite another candidate's picture
        n += 1
        snap = os.path.join(cfg.SNAPSHOT_DIR, f"{stamp:%Y%m%d_%H%M%S}_{name}_{n}.jpg")
    cv2.imwrite(snap, frame)
    new_file = not os.path.exists(cfg.LOG_FILE)
    if not new_file:
        with open(cfg.LOG_FILE, newline="") as f:
            header = next(csv.reader(f), [])
        if header != LOG_HEADER:  # log from an older version: keep it, start a fresh one
            os.replace(cfg.LOG_FILE,
                       cfg.LOG_FILE.replace(".csv", f"_old_{stamp:%Y%m%d_%H%M%S}.csv"))
            new_file = True
    with open(cfg.LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(LOG_HEADER)
        writer.writerow([stamp.isoformat(timespec="seconds"), name, f"{seconds:.1f}", snap, detail])
    print(f"[FLAG] {stamp:%H:%M:%S} {name} ({seconds:.1f}s)" + (f" - {detail}" if detail else ""))
    return snap


def log_note(name, detail=""):
    """Same CSV, no picture: for events that do not come from the camera (e.g. leaving the exam tab)."""
    with _LOCK:
        _log_note(name, detail)


def _log_note(name, detail):
    stamp = datetime.now()
    new_file = not os.path.exists(cfg.LOG_FILE)
    if not new_file:
        with open(cfg.LOG_FILE, newline="") as f:
            header = next(csv.reader(f), [])
        if header != LOG_HEADER:
            os.replace(cfg.LOG_FILE,
                       cfg.LOG_FILE.replace(".csv", f"_old_{stamp:%Y%m%d_%H%M%S}.csv"))
            new_file = True
    with open(cfg.LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(LOG_HEADER)
        writer.writerow([stamp.isoformat(timespec="seconds"), name, "", "", detail])
    print(f"[NOTE] {stamp:%H:%M:%S} {name}" + (f" - {detail}" if detail else ""))
