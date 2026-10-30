"""
The array of checks.

Every flag is one Check(name, seconds, rule) in the list returned by build_checks().
A rule is any function that takes the frame's FrameState and returns True while the
condition is happening (or a pair (True, "extra detail for the log")). The Monitor
fires a flag once a rule has stayed True for `seconds`, then logs it.

To add a new flag, write a rule and add one line to build_checks(). For example,
a flag for a wide-open mouth held for 3 seconds:

    Check("MOUTH_OPEN", 3.0, lambda s: s.blendshapes.get("jawOpen", 0) > 0.35)

(Expression scores need USE_BLENDSHAPES = True in config.py.)
"""

from collections import deque
from dataclasses import dataclass
from typing import Callable

from . import config as cfg


@dataclass
class Check:
    name: str                 # flag name shown on screen and written to the log
    seconds: float            # how long the rule must stay True before the flag fires
    rule: Callable            # rule(FrameState) -> bool  or  (bool, detail_text)
    enabled: bool = True


class SustainedFlag:
    """Fires once per episode after a condition has held for `threshold` seconds."""

    def __init__(self, name, threshold):
        self.name = name
        self.threshold = threshold
        self.since = None
        self.fired = False

    def update(self, active, now):
        """Return (seconds_active, just_fired)."""
        if not active:
            self.since, self.fired = None, False
            return 0.0, False
        if self.since is None:
            self.since = now
        elapsed = now - self.since
        just_fired = elapsed >= self.threshold and not self.fired
        if just_fired:
            self.fired = True
        return elapsed, just_fired


class Monitor:
    """Runs every enabled check on each frame and reports the ones that just fired."""

    def __init__(self, checks):
        self.checks = [c for c in checks if c.enabled]
        self.flags = {c.name: SustainedFlag(c.name, c.seconds) for c in self.checks}

    def update(self, state, now):
        """Return a list of (name, seconds, detail) for flags that fired on this frame."""
        fired = []
        state.now = now
        for check in self.checks:
            outcome = check.rule(state)
            active, detail = outcome if isinstance(outcome, tuple) else (outcome, "")
            seconds, just_fired = self.flags[check.name].update(bool(active), now)
            if just_fired:
                fired.append((check.name, seconds, detail or state.details.get(check.name, "")))
        return fired

    @property
    def active_names(self):
        """Names of flags that are currently raised (shown in the red banner)."""
        return [name for name, flag in self.flags.items() if flag.fired]


# ------------------------------- the rules -------------------------------
def _looking(direction):
    return lambda s: direction in s.looking


def _mouth_open(s):
    return s.blendshapes.get("jawOpen", 0.0) > cfg.MOUTH_OPEN_SCORE


def _eyes_vertical(direction):
    """Eyes (not the head) looking UP or DOWN, from the model's eyeLook scores."""
    limit = cfg.EYE_UP_SCORE if direction == "up" else cfg.EYE_DOWN_SCORE

    def rule(s):
        score = s.eye_up if direction == "up" else s.eye_down
        if score is None or s.eyes_closed or score <= limit:
            return False
        return True, f"eyes looking {direction}: score {score:.2f}, limit {limit:.2f}"
    return rule


class GlanceCounter:
    """Rule for REPEATED_GLANCES: counts separate look-aways and fires on a burst of them.

    A look-away is a stretch of frames where the person is looking away (head turned, iris
    off-centre or iris at the edge). It ends when they have looked back for GLANCE_GAP_SECONDS,
    and counts only if it lasted at least GLANCE_MIN_SECONDS. When GLANCE_COUNT of them have
    happened inside GLANCE_WINDOW_SECONDS the rule goes True for GLANCE_FLAG_SECONDS (so the
    banner stays visible) and the count starts again from zero.
    """

    def __init__(self):
        self.status_text = self._status(0, False)
        self._away_since = None   # when the look-away in progress began
        self._last_away = None    # last frame that still looked away
        self._events = deque()    # end times of the look-aways that counted
        self._active_until = None
        self._detail = ""

    @staticmethod
    def _status(count, active):
        return (f"Glances: {count}/{cfg.GLANCE_COUNT} in {cfg.GLANCE_WINDOW_SECONDS:.0f} s"
                + ("  FLAG" if active else ""))

    def __call__(self, state):
        now = state.now
        away = state.n_faces > 0 and (bool(state.looking) or state.extreme is not None
                                      or state.gaze_off)
        if away:
            if self._away_since is None:
                self._away_since = now
            self._last_away = now
        elif self._away_since is not None and now - self._last_away >= cfg.GLANCE_GAP_SECONDS:
            if self._last_away - self._away_since >= cfg.GLANCE_MIN_SECONDS:
                self._events.append(self._last_away)
            self._away_since = None

        while self._events and now - self._events[0] > cfg.GLANCE_WINDOW_SECONDS:
            self._events.popleft()

        if len(self._events) >= cfg.GLANCE_COUNT:
            self._detail = (f"{len(self._events)} look-aways in the last "
                            f"{cfg.GLANCE_WINDOW_SECONDS:.0f} s")
            self._events.clear()
            self._active_until = now + cfg.GLANCE_FLAG_SECONDS

        active = self._active_until is not None and now < self._active_until
        self.status_text = self._status(len(self._events), active)
        return active, (self._detail if active else "")


# --------------------------- the checks array ----------------------------
def build_checks():
    """The list of checks the Monitor runs. Edit this list to add or remove flags."""
    checks = [
        # who is in front of the camera
        Check("NO_FACE",        cfg.NO_FACE_SECONDS,    lambda s: s.n_faces == 0),
        Check("MULTIPLE_FACES", cfg.MULTI_FACE_SECONDS, lambda s: s.n_faces > 1),

        # head direction (up / down from head tilt, left / right from head turn or iris)
        Check("LOOKING_UP",    cfg.AWAY_SECONDS, _looking("UP")),
        Check("LOOKING_DOWN",  cfg.AWAY_SECONDS, _looking("DOWN")),
        Check("LOOKING_LEFT",  cfg.AWAY_SECONDS, _looking("LEFT")),
        Check("LOOKING_RIGHT", cfg.AWAY_SECONDS, _looking("RIGHT")),

        # eyes
        Check("EYES_CLOSED",       cfg.EYES_CLOSED_SECONDS, lambda s: s.eyes_closed),
        Check("EYES_COVERED",      cfg.EYES_CLOSED_SECONDS, lambda s: s.eyes_covered),
        Check("LEFT_EYE_HIDDEN",   cfg.ONE_EYE_SECONDS,     lambda s: s.hidden_eye == "LEFT"),
        Check("RIGHT_EYE_HIDDEN",  cfg.ONE_EYE_SECONDS,     lambda s: s.hidden_eye == "RIGHT"),
        Check("EYES_EXTREME_LEFT",  cfg.EXTREME_SECONDS,    lambda s: s.extreme == "LEFT"),
        Check("EYES_EXTREME_RIGHT", cfg.EXTREME_SECONDS,    lambda s: s.extreme == "RIGHT"),
        Check("HEAD_TURNED_ONE_EYE", cfg.AWAY_SECONDS,      lambda s: s.turned_one_eye),

        # gaze point off the screen (only ever True in `gaze_demo.py --screen` mode)
        Check("GAZE_OFF_SCREEN", cfg.GAZE_OFF_SECONDS, lambda s: s.gaze_off),

        # many short look-aways inside a minute (fires at once; the rule does the timing)
        Check("REPEATED_GLANCES", 0.0, GlanceCounter(), enabled=cfg.GLANCE_CHECK),

        # expressions and eye-only up/down (need USE_BLENDSHAPES = True); add your own the same way
        Check("MOUTH_OPEN", cfg.MOUTH_OPEN_SECONDS, _mouth_open, enabled=cfg.USE_BLENDSHAPES),
        Check("EYES_UP",    cfg.EYE_VERT_SECONDS, _eyes_vertical("up"),   enabled=cfg.USE_BLENDSHAPES),
        Check("EYES_DOWN",  cfg.EYE_VERT_SECONDS, _eyes_vertical("down"), enabled=cfg.USE_BLENDSHAPES),
    ]

    if cfg.USE_IDENTITY:
        from .identity import build_identity_check  # imported here so it costs nothing when off
        checks.append(build_identity_check())

    return checks
