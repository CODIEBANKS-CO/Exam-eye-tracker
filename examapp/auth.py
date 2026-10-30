"""
Invigilator sign-in.

The program makes a short key by itself every time it starts (for example K7QM-4XD2) and prints it
in the terminal. The invigilator page signs in with it and then holds a longer session token, so
the key is typed (or arrives in the link) once and is not sent with every request.

* The key uses letters and digits that cannot be mixed up (no 0/O, 1/I/L).
* Case, spaces and the hyphen do not matter when the key is typed.
* Five wrong tries lock sign-in for 30 seconds, so the key cannot be guessed by trying.
"""

import hmac
import secrets
import threading
import time

ALPHABET = "23456789ABCDEFGHJKMNPQRSTUVWXYZ"   # 31 characters: no 0, O, 1, I or L
MAX_TOKENS = 20


def make_key(groups=2, size=4, rng=secrets):
    """A new random key such as 'K7QM-4XD2'."""
    return "-".join("".join(rng.choice(ALPHABET) for _ in range(size)) for _ in range(groups))


def normalise(key):
    """Letters and digits only, upper case: 'k7qm 4xd2' and 'K7QM-4XD2' are the same key."""
    return "".join(c for c in str(key or "").upper() if c.isalnum())


class AdminAuth:
    def __init__(self, key=None, clock=time.monotonic, max_failures=5, lock_seconds=30.0):
        key = (key or "").strip()
        self.key = key if normalise(key) else make_key()
        self.clock = clock
        self.max_failures, self.lock_seconds = max_failures, float(lock_seconds)
        self._failures, self._locked_until = 0, 0.0
        self._tokens = []
        self._lock = threading.Lock()

    def login(self, key):
        """Returns ('ok', token), ('wrong', None) or ('locked', seconds_to_wait)."""
        with self._lock:
            now = self.clock()
            if now < self._locked_until:                       # even the right key waits
                return "locked", self._locked_until - now
            if hmac.compare_digest(normalise(key), normalise(self.key)):
                self._failures = 0
                token = secrets.token_urlsafe(24)
                self._tokens.append(token)
                del self._tokens[:-MAX_TOKENS]
                return "ok", token
            self._failures += 1
            if self._failures >= self.max_failures:
                self._failures, self._locked_until = 0, now + self.lock_seconds
                return "locked", self.lock_seconds
            return "wrong", None

    def valid(self, token):
        token, found = str(token or ""), False
        with self._lock:
            for t in self._tokens:                             # no early exit
                found |= hmac.compare_digest(token, t)
        return found

    def logout(self, token):
        token = str(token or "")
        with self._lock:
            self._tokens = [t for t in self._tokens if not hmac.compare_digest(token, t)]
