"""Webcam helper shared by gaze_demo.py and enroll_face.py."""

import sys

import cv2

from . import config as cfg


def open_camera():
    """Open the webcam, ask for CAMERA_WIDTH x CAMERA_HEIGHT, report what it gave."""
    if sys.platform == "win32":
        cap = cv2.VideoCapture(cfg.CAMERA_INDEX, cv2.CAP_DSHOW)  # DirectShow honours size requests better
    else:
        cap = cv2.VideoCapture(cfg.CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam. Try another CAMERA_INDEX in facelib/config.py.")

    # MJPG is usually needed for 1080p at a smooth frame rate over USB.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.CAMERA_HEIGHT)

    got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera: asked for {cfg.CAMERA_WIDTH}x{cfg.CAMERA_HEIGHT}, got {got_w}x{got_h} at {fps:.0f} fps")
    if (got_w, got_h) != (cfg.CAMERA_WIDTH, cfg.CAMERA_HEIGHT):
        print("The camera or its driver does not offer that size, so it picked the closest one.")
    return cap
