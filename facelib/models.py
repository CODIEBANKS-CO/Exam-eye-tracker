"""Small helper: download a model file once, with a clear message if that fails."""

import os
import urllib.request


def ensure_file(path, url, label):
    """Make sure `path` exists, downloading it from `url` on first use."""
    if os.path.exists(path) and os.path.getsize(path) > 1024:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading {label} to {path} ...")
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as exc:  # offline, blocked network, etc.
        if os.path.exists(path):
            os.remove(path)
        raise SystemExit(
            f"Could not download the {label} ({exc}).\n"
            f"Download it manually from:\n  {url}\n"
            f"and save it as:\n  {path}"
        )
    if os.path.getsize(path) <= 1024:  # an error page or a Git LFS pointer, not the model
        os.remove(path)
        raise SystemExit(
            f"The downloaded {label} was too small to be the real file.\n"
            f"Download it manually from:\n  {url}\n"
            f"and save it as:\n  {path}"
        )
    return path
