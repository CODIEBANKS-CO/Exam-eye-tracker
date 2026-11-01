"""
All the settings in one place. Edit the numbers here; nothing else needs to change.

Run gaze_demo.py and watch the live numbers on the preview (Yaw/Pitch, Eyes L/R,
Iris, Light) to tune the thresholds on your own camera, lighting and face.
"""

import os

# Project folder (the folder that contains gaze_demo.py and facelib/).
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ------------------------------- camera ----------------------------------
CAMERA_INDEX = 0
CAMERA_WIDTH, CAMERA_HEIGHT = 1920, 1080  # requested; the camera uses the closest size it supports
DISPLAY_WIDTH = 1280   # preview window width only; the tracking itself runs at full camera size

# ----------------------------- flag timing -------------------------------
AWAY_SECONDS = 3.0         # looking up/down/left/right this long -> flag
NO_FACE_SECONDS = 2.0      # no face this long -> flag
MULTI_FACE_SECONDS = 1.0   # 2+ faces this long -> flag

# --------------------------- head direction ------------------------------
YAW_LIMIT = 25.0           # degrees of head turn left/right before flagging
PITCH_UP_LIMIT = 15.0      # degrees of head tilt UP before flagging
PITCH_DOWN_LIMIT = 20.0    # degrees of head tilt DOWN before flagging
# Positive pitch means the head is tilted DOWN in testing. If your live "Yaw/Pitch"
# readout shows the opposite (looking up gives a positive number), set this to False.
POSITIVE_PITCH_IS_DOWN = True

# ------------------------------- iris / gaze -----------------------------
GAZE_LOW, GAZE_HIGH = 0.38, 0.62   # horizontal iris ratio treated as "center"
VERT_LOW, VERT_HIGH = 0.30, 0.70   # vertical iris ratio (display only, noisy)

# Iris at the very edge of the eye (0 = left corner, 1 = right corner of the eye).
# An iris is only about 40% of the eye's width, so its centre can reach roughly
# 0.2-0.8 at most. Watch the live "Iris" number and set these just inside your own range.
EXTREME_LOW, EXTREME_HIGH = 0.30, 0.70
EXTREME_SECONDS = 1.5

# Head turned past this many degrees: the eye on the side the nose points to is mostly
# hidden behind the nose, so only the other eye is trusted. HEAD_TURNED_ONE_EYE fires
# when the head is turned this far and the visible eye's iris is still centred.
YAW_ONE_EYE = 30.0

# ------------------------------ eyes closed ------------------------------
# Measured from the SHAPE of the eyelids (lid gap / eye width), not from colour,
# so it does not depend on skin tone or iris colour.
EAR_CLOSED = 0.15          # below this an eye counts as closed; tune with the live "Eyes L / R" numbers
EYES_CLOSED_SECONDS = 2.0  # both eyes shut: blinks are ignored, flag only after this long
# One eye shut or covered ("only one iris visible"): the model always outputs
# landmarks even for a hidden eye, so this is inferred from the eyelids. It fires
# when exactly one eye is below EAR_CLOSED, or when the two eyes differ by more than
# EAR_ASYMMETRY. It is skipped while the head is turned past YAW_LIMIT, because a
# turned head naturally hides the far eye (LOOKING_LEFT/RIGHT covers that case).
EAR_ASYMMETRY = 0.10
ONE_EYE_SECONDS = 1.5

# ------------------------- eye covered by a hand -------------------------
# The model keeps drawing an eye outline even when a hand covers the eye, so the
# eyelid numbers above still read "open". This check looks at the PICTURE inside each
# eye outline (see facelib/cover.py): a real eye has a dark iris, white and lashes, a
# hand is a smooth patch. Watch "Eye detail L / R" on the preview: an open eye reads
# roughly 0.7 to 1.5, a covered one far lower. One covered eye raises LEFT/RIGHT_EYE_HIDDEN
# (after ONE_EYE_SECONDS); both raise EYES_COVERED (after EYES_CLOSED_SECONDS).
COVER_CHECK = True
COVER_RATIO = 0.60         # an eye counts as covered when its detail is below this fraction of
                           # normal (the other eye, or this person's learned normal)
COVER_FLOOR = 0.15         # ...or below this value whatever the other eye shows (a flat patch)
COVER_RELEARN_SECONDS = 30.0  # "both covered" this long: assume the light changed, learn normal again

# ------------------------ gaze mapped to the screen ----------------------
# Run `python gaze_demo.py --screen`. The person first looks at a series of dots; the program
# learns how their iris position and head pose map to a point on the screen (see
# facelib/screengaze.py). A dot on a full-screen "gaze pad" then follows where they look, like a
# trackpad, and GAZE_OFF_SCREEN is raised when the gaze point stays outside the screen.
# A webcam can only do this roughly (expect the dot to be off by a few centimetres, more if the
# head moves after calibration), so treat it as "which part of the screen", never a pixel.
SCREEN_WIDTH = None        # None = read the screen size from the computer; set both to override
SCREEN_HEIGHT = None
GAZE_MAX_YAW = 35.0        # head turned more than this: no gaze point (the LOOKING_* flags cover it)
GAZE_RIDGE = 0.005         # steadiness of the fitted mapping (head terms, curved terms); raise it if the dot jumps wildly
GAZE_CAL_EDGE = 0.08       # outer calibration dots sit this fraction in from the screen edge
GAZE_CAL_START_SECONDS = 6.0     # time to get comfortable before the first dot (SPACE skips it)
GAZE_CAL_SETTLE_SECONDS = 0.8    # at each dot: time to move the eyes there ...
GAZE_CAL_COLLECT_SECONDS = 1.2   # ... then this long to record
GAZE_CAL_MIN_SAMPLES = 8         # fewer good frames than this at a dot: the dot is shown again
GAZE_CAL_RETRIES = 1
GAZE_CAL_REDO_BAD = True          # dots whose reading shook much more than the rest are shown again once
GAZE_VALIDATE = True             # 5 extra dots the fit never saw: measures the real accuracy (adds ~10 s)
GAZE_FEATURE_FRAMES = 5          # while tracking, use the middle value of this many frames (glitch filter)
GAZE_CAL_DEBUG_FILE = os.path.join(BASE_DIR, "calibration", "gaze_debug.json")  # per-dot details of the last calibration
GAZE_CAL_WARN_FRACTION = 0.20    # warn "recalibrate" when the cross-check error is above this
                                 # fraction of the screen diagonal
GAZE_SMOOTH_MIN_CUTOFF = 0.5     # dot smoothing (One Euro filter): lower = steadier but laggier
GAZE_SMOOTH_BETA = 0.001         # higher = the dot follows fast eye movements sooner
GAZE_TRAIL = 25                  # points of tail drawn behind the dot
GAZE_OFF_MARGIN = 0.10     # gaze this fraction beyond a screen edge counts as "off screen"
GAZE_OFF_SECONDS = 2.0     # ... for this long -> GAZE_OFF_SCREEN
GAZE_CAL_FILE = os.path.join(BASE_DIR, "calibration", "gaze_cal.json")
# Notes on the gaze pad (they explain a drifting dot; none of them raises a flag). These two
# numbers are guesses: watch the notes on your own setup and adjust.
GAZE_HEAD_MOVE_WARN = 0.05       # head moved this fraction of the camera picture width since calibration
GAZE_HEAD_SCALE_WARN = 0.10      # or got this much closer / further from the camera (0.10 = 10 %)
GAZE_LIGHT_BALANCE_WARN = 0.30   # one side of the face this much brighter than the other (0.30 = 30 %)
GAZE_RECENTRE_SECONDS = 2.0      # press z: look at the centre dot this long to shift the whole mapping
GAZE_LOG_FILE = os.path.join(BASE_DIR, "calibration", "gaze_log.csv")  # written by --debug

# ------------------------------- low light -------------------------------
# The image sent to the model is brightened when the FACE region measures dark
# (0-255 scale). This is based on how much light reaches the camera, never on a
# skin-tone label, and it never changes the preview or the saved snapshots.
ENHANCE_LOW_LIGHT = True
# Extra local contrast (CLAHE) on top of the brightness lift. It can help the model
# see dark irises, but costs roughly 15-25 ms per frame at 1080p, so it is off by
# default. Try True if iris tracking is poor and your computer keeps up.
ENHANCE_CONTRAST = False
TARGET_LUMA = 100          # brighten toward this face brightness
LOW_LIGHT_WARN = 40        # show an "add light" message below this

# ------------------------ repeated quick glances -------------------------
# Someone who keeps glancing away for a second at a time never holds a look for
# AWAY_SECONDS, so the flags above miss them. This counts each separate "look-away" (head
# turned, iris off-centre or iris at the edge) and raises REPEATED_GLANCES when there are
# GLANCE_COUNT of them within GLANCE_WINDOW_SECONDS. The preview shows the running count.
GLANCE_CHECK = True
GLANCE_MIN_SECONDS = 0.5     # a look-away shorter than this is ignored (noise, a blink of movement)
GLANCE_GAP_SECONDS = 0.4     # back to the screen for less than this = still the same look-away
GLANCE_COUNT = 4             # this many look-aways ...
GLANCE_WINDOW_SECONDS = 60.0 # ... inside this many seconds raise the flag
GLANCE_FLAG_SECONDS = 5.0    # the red banner stays up this long; counting then starts again from zero

# ----------------------- expressions (optional) --------------------------
# Turns on the model's expression scores (about 52 values from 0 to 1, e.g. "jawOpen").
# Off by default because it costs a little speed. Add your own rules in checks.py.
USE_BLENDSHAPES = False
MOUTH_OPEN_SCORE = 0.35    # "jawOpen" above this counts as mouth open
MOUTH_OPEN_SECONDS = 3.0   # mouth open this long (talking) -> flag
PRINT_BLENDSHAPES = False  # True: print the strongest scores once a second, to help you pick thresholds

# Eyes looking UP or DOWN with the head still. The iris position up/down is too noisy to
# use, but the model's own "eyeLookUp / eyeLookDown" scores (0 to 1) are steadier. They only
# exist when USE_BLENDSHAPES = True. The preview then shows "Eye look: up .. down ..":
# read your numbers while looking at the middle, top and bottom of the screen, and set the
# limits above what you see while just reading the screen. A camera above the screen makes
# looking at the bottom of the screen read as a little "down" on its own.
EYE_UP_SCORE = 0.50
EYE_DOWN_SCORE = 0.50
EYE_VERT_SECONDS = 2.0     # held this long -> EYES_UP / EYES_DOWN

# --------------------- identity check (optional, part B) -----------------
# Compares the live face with a candidate enrolled using enroll_face.py.
# Off by default: it needs two small model files and the candidate's consent.
USE_IDENTITY = False
ID_EXPECTED_NAME = None    # name used at enrolment; None = best match in the whole gallery
ID_MATCH_THRESHOLD = 0.363 # cosine similarity needed to count as the same person
                           # (OpenCV's SFace example value; tune on your own candidates)
ID_CHECK_SECONDS = 2.0     # compare the face this often (the comparison is the slow part)
ID_MISMATCH_SECONDS = 6.0  # keep failing this long before IDENTITY_MISMATCH is flagged
ID_MAX_WIDTH = 960         # the picture is shrunk to this width before face detection
ID_MIN_FACE_SCORE = 0.8    # ignore weak face detections

# --- face match inside the exam (examapp): the reference is made live at the start ---
# The candidate's face is turned into a few signatures when the exam starts (kept in memory only), then
# compared with the live face every second until the exam ends. ID_MATCH_THRESHOLD above is the pass mark.
EXAM_FACE_MATCH = True             # switch the whole face-match feature on (needs the YuNet and SFace models)
EXAM_FACE_REQUIRED = False         # True: no "continue without face matching" button if the face check fails
EXAM_FACE_SAVE_REFERENCE = True    # save one picture of the face taken at the start, so a reviewer can compare
EXAM_FACE_SAMPLES = 6              # pictures used to make the reference (taken about half a second apart)
EXAM_FACE_ENROL_SECONDS = 25.0     # give up on the face check after this long
EXAM_FACE_SAMPLE_GAP = 0.5         # seconds between reference pictures
EXAM_FACE_CHECK_SECONDS = 1.0      # compare the live face this often during the exam
EXAM_FACE_MISMATCH_SECONDS = 3.0   # keep failing this long before IDENTITY_MISMATCH is flagged
ID_ENROL_CONSISTENCY = 0.45        # the reference pictures must look this alike (lowest pair), else the check is repeated
ID_ENROL_MIN_FACE_FRACTION = 0.12  # the face must be at least this fraction of the picture width (else: move closer)
ID_UNVERIFIED_SECONDS = 45.0       # flag IDENTITY_UNVERIFIED when no clear front-facing view matched for this long

# --- the room: many candidate computers reporting to one invigilator computer (examapp/hub.py) ---
ROOM_PREVIEW_MIN_SECONDS = 1.0     # the camera program makes a small preview picture at most this often (when the room asks)
ROOM_PREVIEW_WIDTH = 320           # pixels wide: small on purpose, so a whole room fits on the network
ROOM_PREVIEW_QUALITY = 60          # JPEG quality of the preview
ROOM_REPORT_SECONDS = 1.0          # a candidate computer reports its state this often
ROOM_LOST_SECONDS = 8.0            # no report for this long and the invigilator sees "connection lost"

# ------------------------------ files / paths ----------------------------
MODEL_DIR = os.path.join(BASE_DIR, "models")
LANDMARKER_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")
LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
YUNET_PATH = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
SFACE_PATH = os.path.join(MODEL_DIR, "face_recognition_sface_2021dec.onnx")
SFACE_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_recognition_sface/face_recognition_sface_2021dec.onnx")

GALLERY_FILE = os.path.join(BASE_DIR, "gallery", "faces.json")
LOG_FILE = os.path.join(BASE_DIR, "events.csv")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "snapshots")

# ------------------------------- the exam app -----------------------------
# Run `python exam_app.py` (see examapp/). The questions live in questions.json.
EXAM_QUESTIONS_FILE = os.path.join(BASE_DIR, "questions.json")
EXAM_RESULTS_DIR = os.path.join(BASE_DIR, "results")
EXAM_MINUTES = 20                  # time allowed (the server keeps the clock, not the page)
EXAM_SHUFFLE_QUESTIONS = True      # each candidate gets the questions in a different order
EXAM_SHUFFLE_OPTIONS = True        # ... and the answer options in a different order
EXAM_SHOW_SCORE = True             # show "You scored x / 15" at the end (False: just "submitted")
EXAM_SHOW_WARNINGS = True          # show the candidate a calm banner while a camera flag is active
EXAM_SUBMIT_GRACE_SECONDS = 5.0    # answers still count this long after time is up (slow network)
EXAM_HOST = "127.0.0.1"            # only this computer can open the exam page
EXAM_PORT = 8765
EXAM_GAZE = True                   # calibrate each candidate's gaze first, then flag/alert when they look off the screen
EXAM_MAX_CAL_ERROR_FRACTION = 0.15 # a calibration worse than this (fraction of the screen diagonal) is offered again
EXAM_SHOW_GAZE_DOT = False         # draw the candidate's gaze point on their exam page (for the demo; --show-gaze)
EXAM_GAZE_SIZE_TOLERANCE = 0.03    # window size may differ this much from the calibration, else gaze checks pause
EXAM_ADMIN_KEY = None              # invigilator key; None = a new one like K7QM-4XD2 is made each time the program starts
