"""Small helper: download a model file once, with a clear message if that fails."""

import os
import threading
import urllib.request

_LOCK = threading.Lock()      # many candidates' monitors start together: only one downloads a file


def ensure_file(path, url, label):
    """Make sure `path` exists, downloading it from `url` on first use."""
    with _LOCK:
        return _ensure_file(path, url, label)


def _ensure_file(path, url, label):
    if os.path.exists(path) and os.path.getsize(path) > 1024:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading {label} to {path} ...")
    part = path + ".part"            # a half-downloaded file must never look like the model
    try:
        urllib.request.urlretrieve(url, part)
    except Exception as exc:  # offline, blocked network, etc.
        if os.path.exists(part):
            os.remove(part)
        raise SystemExit(
            f"Could not download the {label} ({exc}).\n"
            f"Download it manually from:\n  {url}\n"
            f"and save it as:\n  {path}"
        )
    if os.path.getsize(part) <= 1024:  # an error page or a Git LFS pointer, not the model
        os.remove(part)
        raise SystemExit(
            f"The downloaded {label} was too small to be the real file.\n"
            f"Download it manually from:\n  {url}\n"
            f"and save it as:\n  {path}"
        )
    os.replace(part, path)
    return path
