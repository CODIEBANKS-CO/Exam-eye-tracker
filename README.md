# Exam eye tracker

Webcam exam monitor: it watches for no face, extra faces, head and eye direction, closed or
hidden eyes, and (optionally) a different person from the enrolled candidate. Every flag is
saved to `events.csv` with a snapshot, for a human to review.

## Install and run

    pip install opencv-python mediapipe numpy
    python gaze_demo.py          (q or Esc to quit)
    python gaze_demo.py --screen   gaze mapped onto the screen (see below)
    python exam_app.py             the exam: 15 multiple-choice questions + webcam monitor + gaze (see below)

Quick start for the exam (run these in a terminal opened in this folder; on Windows use `py` if `python` is not found):

    python exam_app.py                      one computer: the exam, the camera checks and the invigilator console
    python exam_app.py --no-camera          the same without a webcam (to write questions, or to try the page)
    python exam_app.py --hub                a room: on the invigilator's computer; prints the room code and the command below
    python exam_app.py --report-to 192.168.1.10:8765 --code K7QM-4XD2         a room: on each candidate's computer
                                            (use the address and code the hub printed; add --label "Lab PC 3" to name the computer)

To try a room on one computer, use two terminals: `python exam_app.py --hub --port 8765`, then
`python exam_app.py --report-to 127.0.0.1:8765 --code <the code> --port 8766 --no-camera` (only one program can use the
webcam), and open `http://127.0.0.1:8766` as the candidate. For a real room the computers must be on the same
network and the hub's computer must allow the program through its firewall (port 8765).

First run downloads the MediaPipe face model into `models/`, and the exam's face match downloads two more
models (YuNet and SFace), so the first run needs internet. Offline? Download the face model from the
address `LANDMARKER_URL` in `facelib/config.py` and save it in `models/`; the two face match models are at
`YUNET_URL` and `SFACE_URL` in the same file (use the download button on the GitHub page, not "save page as").

## Folder layout

    gaze_demo.py        runs the camera loop and draws the preview
    exam_app.py         starts the exam (a local web page) with the webcam monitor beside it
    exams.py            looks after the exam records: list, import, set, publish, start, close, audit
    questions.json      the exam questions (sample set: replace with your own)
    examapp/
        engine.py       exam logic: load/check questions, shuffle, clock, scoring, result record
        store.py        exam records in exams.db (SQLite): questions, time, rules, status, audit log
        server.py       small local web server (standard library only); never sends the answer key
        multi.py        many candidates in browsers on one exam computer: join by access code, one sitting each
        browsercam.py   camera pictures posted by a browser, fed to the same camera checks
        tls.py          makes the HTTPS certificate (browsers only allow the camera over https)
        monitor.py      the webcam monitor in a background thread; tags flags with the question number
        static/index.html   the exam page (start screen, questions, timer, submit, result)
    enroll_face.py      registers a candidate (identity check, optional)
    facelib/
        config.py       ALL settings: thresholds, timings, switches
        analysis.py     face geometry and the per-frame FrameState
        checks.py       the array of checks, one entry per flag
        events.py       CSV log and snapshots
        identity.py     enrolled-face gallery and the identity check
        cover.py        spots an eye covered by a hand (looks at the picture, not the outline)
        screengaze.py   maps the iris to a point on the screen (calibration, gaze pad)
        camera.py       webcam helper
        models.py       model download helper

## Tuning

Watch the numbers on the preview (Yaw/Pitch, Eyes L/R, Iris, Light), then change the matching
setting in `facelib/config.py`. Start with `EAR_CLOSED`, `EXTREME_LOW/HIGH`, `YAW_LIMIT`,
`PITCH_UP_LIMIT` and `PITCH_DOWN_LIMIT`. If looking up shows a positive pitch on your camera,
set `POSITIVE_PITCH_IS_DOWN = False`.

## A hand over an eye

The face model keeps drawing an eye outline even when a hand covers the eye, so the eyelid
numbers alone still read "open". `facelib/cover.py` therefore also measures how much detail
is in the picture inside each eye outline (a real eye has an iris, white and lashes; a hand
is smooth). The preview shows it as `Eye detail L / R`, with this person's learned `normal`.
One covered eye raises `LEFT_EYE_HIDDEN` / `RIGHT_EYE_HIDDEN`; both raise `EYES_COVERED`.
Tune `COVER_RATIO` (raise it to catch more, lower it to false-flag less) and `COVER_FLOOR` in
`facelib/config.py`. Start the exam with a clear view of the face so it can learn "normal".
It is skipped while the head is turned past `YAW_LIMIT`. Fingers that leave part of the eye
visible, sunglasses or a strong shadow across one eye can also change the number.

## Repeated quick glances

`REPEATED_GLANCES` catches someone who keeps glancing away for a second at a time and never
holds a look long enough for `LOOKING_*`. Each separate look-away (head turned, iris off-centre
or at the edge) of at least `GLANCE_MIN_SECONDS` counts; `GLANCE_COUNT` of them inside
`GLANCE_WINDOW_SECONDS` raises the flag (default 4 within 60 s). The preview shows the running
count ("Glances: 2/4 in 60 s"). Turn it off with `GLANCE_CHECK = False`.

## Eyes looking up or down (head still)

`EYES_UP` / `EYES_DOWN` use the model's own `eyeLookUp` / `eyeLookDown` scores, so they need
`USE_BLENDSHAPES = True` (this also switches on `MOUTH_OPEN`). The preview then shows
`Eye look: up .. down ..`. Look at the middle, top and bottom of the screen and set
`EYE_UP_SCORE` / `EYE_DOWN_SCORE` above what you see while just reading the screen. These
thresholds are guesses until you tune them on your camera.

## Gaze mapped onto the screen (like a trackpad)

    python gaze_demo.py --screen            calibrate, then the full-screen gaze pad
    python gaze_demo.py --screen --load     reuse the saved calibration (same screen, same seat)
    python gaze_demo.py --screen --mouse    also move the real mouse pointer (Windows only)
    python gaze_demo.py --screen --debug    also write calibration/gaze_log.csv (one row per frame)

1. **Calibration.** Nine yellow dots appear one at a time. Look straight at each one with your
   head still. The program records your iris position and head pose at each dot and fits a small
   curve ("these numbers = this point on the screen"). Light on your face is checked first.
   A dot whose reading shakes much more than the others (usually bad light) is shown again once.
   Five extra "accuracy check" dots in new places then measure the real error, and are added
   to the fit. A per-dot table is printed and saved to `calibration/gaze_debug.json`:
   `scatter` = how much the eye reading shook at that dot (a lighting/noise measure),
   `fit-err` = how far the fitted dot lands from where it really was,
   `cross-err` = the same when that dot was left out of the fit.
2. **Gaze pad.** A full-screen window shows a dot that follows where you look, with a short tail
   (green inside the screen, red outside), plus a small camera picture in the corner.
   Keys: `q`/Esc quit, `c` recalibrate (do this if you change seat or posture), `z` re-centre,
   `m` real mouse on/off. `z` shows the middle dot for 2 s and shifts the whole mapping so the
   middle of the screen is right again: use it when the dot sits a constant distance off, without
   redoing all the dots. Yellow notes on the pad say why the dot may be off right now: your head
   moved or you leaned closer/further since calibration (`GAZE_HEAD_MOVE_WARN`,
   `GAZE_HEAD_SCALE_WARN`), or one side of your face is much brighter (`GAZE_LIGHT_BALANCE_WARN`).
   The notes do not raise flags, and their limits are guesses to tune on your camera.
3. **Flag.** `GAZE_OFF_SCREEN` is raised when the gaze point stays beyond the screen edge
   (plus `GAZE_OFF_MARGIN`) for `GAZE_OFF_SECONDS`. It also counts towards `REPEATED_GLANCES`.

`--debug` writes `calibration/gaze_log.csv`: per frame the eye numbers, head pose, gaze point,
light, eye openness and the notes' numbers (no pictures). Open it in a spreadsheet to see
whether the dot drifts when the light or your head changes. If the dot is off, send
`gaze_debug.json` and the printed table (and the log if you used `--debug`).

Lighting: put the light in front of your face (a lamp behind the screen, or the screen itself is
fine), nothing bright behind you, no window to one side. Dark eyes in dim or one-sided light is
the hardest case for the iris landmarks.

Settings are in the "gaze mapped to the screen" block of `facelib/config.py`. It is not a pixel
accurate tracker: a webcam gives a few centimetres of error, more if the head moves after
calibration, and it only works while both eyes are visible and the head is within
`GAZE_MAX_YAW` of straight. The calibration file stores only the fitted numbers, no pictures.

## The exam

    python exam_app.py                 opens http://127.0.0.1:8765 in your browser
    python exam_app.py --no-camera     exam only, no webcam (fallback for the demo, or to write questions)
    python exam_app.py --show-gaze     also draw the candidate's gaze point on their page (for the demo)
    python exam_app.py --no-gaze       camera monitor on, but skip the gaze calibration and look-away alert
    python exam_app.py --questions other.json --minutes 30 --port 9000
    python exam_app.py --exam EX-7K3QMD   run an exam from the exam records (see "Exam records" below)

The candidate enters a name and ID, ticks the monitoring notice, and starts: 15 multiple-choice
questions, one at a time, with a question grid, "mark for review", a countdown, and a confirm
step before submitting. Keys: A-D or 1-4 choose, left/right arrows move. Clicking the chosen
option again clears it. The exam submits itself when time is up. A page refresh carries on
where the candidate was.

* **Gaze calibration comes first.** After the name, ID and consent, the exam window shows the same
  9 dots (plus 5 accuracy-check dots) as `gaze_demo.py --screen`. The candidate looks at each dot with
  the head still; the camera program collects the readings and fits the mapping for that person,
  seat and screen. The exam clock starts only when the calibration is done. If the eyes cannot be seen,
  or the accuracy is worse than `EXAM_MAX_CAL_ERROR_FRACTION` of the screen diagonal, the page offers
  "Calibrate again" or "Continue without gaze tracking". Skipping is recorded as `GAZE_TRACKING_SKIPPED`
  for the invigilator and in the result. The calibration picture is drawn by `facelib/screengaze.py` (the
  same code as the `--screen` mode) and sent to the page as JPEG frames; the page does not know anything
  about eyes.
* **Look-away alert.** While the exam runs, the gaze point is tracked. If it stays beyond the screen edge
  (plus 10 %) for 2 seconds, `GAZE_OFF_SCREEN` is raised: the candidate sees a red banner "You appear to be
  looking away from the screen. Please look back at the exam.", which disappears when they look back, and the
  flag goes to `events.csv` with a snapshot and into the result with the question number. Looking away also
  counts towards `REPEATED_GLANCES`. The banner needs `EXAM_SHOW_WARNINGS = True`.
* **Full screen matters.** The mapping is only valid at the window size it was calibrated at. Leaving full
  screen or resizing (or reloading the page) pauses the gaze checks and shows a "Return to full screen"
  overlay; the leave is recorded as `FULLSCREEN_EXITED`.
* **Invigilator console.** Every time the program starts it makes a new key (for example `K7QM-4XD2`),
  prints it, and opens the invigilator console in a second browser window with the key already in the
  link (`http://127.0.0.1:8765/admin#key=K7QM-4XD2`), so there is nothing to type. The key leaves the
  address bar at once and is never sent in a web address again; the page keeps a longer sign-in token for
  its browser tab. If you open `/admin` (or `/invigilator`) without the link, a sign-in page asks for the
  key: case, spaces and the hyphen do not matter. Five wrong keys lock sign-in for 30 seconds. Sign-ins,
  wrong keys and locks are written to `events.csv` (never the key itself). If the program is restarted the
  key changes and the console asks for the new one, saying why. `--no-invigilator-window` stops the second
  window opening (the key and link are still printed); `EXAM_ADMIN_KEY` in `facelib/config.py` sets your
  own fixed key instead.

  The console shows, in plain words: a large status band (waiting, setting up, "All clear", or the active
  flags in red), the candidate, question, answers and time left, camera, faces, light and calibration
  accuracy, a map of the candidate's screen with the gaze point moving on it (red when it leaves the screen,
  with a dashed line where a look-away is flagged), every flag and event with its time and question (with
  All / Camera / Browser / System filters; each camera flag shows the small picture the camera took at that
  moment, click it for a large view, and the red band shows the picture of the flag that is active), the earlier candidates from `results/summary.csv`, and a button
  to download the finished candidate's result. "Sound alerts" beeps on each new flag (off until you press
  it), and the browser tab title shows "(!)" while a flag is active. The pictures are the same files as in `snapshots/` (the result file names the one for each flag); only
  a signed-in invigilator can fetch them, and only those belonging to the current candidate's flags. They are
  pictures of a person, so decide how long to keep them. It is not shown to the candidate.
  `--show-gaze` additionally draws the gaze point on the candidate's own page for the demo. The console
  only answers `127.0.0.1` / `localhost` addresses and refuses requests from other web sites.
* **Questions.** `questions.json`: `id`, `text`, `options` (2 to 8) and `answer` (the 0-based
  number of the correct option). A mistake in the file is reported with the question number
  when the program starts. The 15 questions shipped are samples. Questions and options are
  shuffled for each candidate (`EXAM_SHUFFLE_QUESTIONS`, `EXAM_SHUFFLE_OPTIONS`).
* **Answer key and clock.** The page never receives the answer key, and the server keeps the
  clock, so editing the page does nothing. Scoring happens on the server.
* **Monitoring.** The webcam monitor starts with the program, so the start page can show a
  camera check (face found, light OK). It records flags only while an exam is running. Camera
  flags are saved to `events.csv` with a snapshot, and into the result, with the question number
  the candidate was on ("Q6"). The page also records: leaving the exam window or tab, leaving
  full screen, copy and paste attempts, and reloads. If the camera stops during the exam it is
  recorded as `CAMERA_MONITOR_STOPPED` and the exam carries on.
* **Warnings.** While a camera flag is active the candidate sees a calm banner ("We cannot see
  your face..."). `EXAM_SHOW_WARNINGS = False` hides it.
* **Results.** `results/<candidate id>_<time>.json` (score, each answer with the right answer,
  time per question, every flag with its time and question) and one line in `results/summary.csv`.
  `EXAM_SHOW_SCORE = False` hides the score from the candidate.
* **One candidate at a time** on a computer. The page only answers to this computer
  (`EXAM_HOST`). Settings are in the "the exam app" block of `facelib/config.py`.

The answer key sits in `questions.json` on the exam computer. For a real exam, keep that file
on a server the candidate cannot open. Flags are for a human reviewer, never proof.

## Exam records (milestone 2, piece 1)

An exam is now a record, kept in `exams.db` next to the program (SQLite, nothing to install): a title, the
time allowed, its rules, its questions with the answer key, a status and a log of everything done to it.
`exam_app.py` runs one of these records. This is the base for the author editor, registered students,
and the browser-only candidate that follow.

**Nothing changes for you at first.** `python exam_app.py` imports `questions.json` once as an exam (it says
`Imported ... as exam EX-XXXXXX`), opens it and runs it as before. Starting again reuses the same exam; the
same file content is never imported twice. If you change the file, it becomes a new exam (a new record).
The exam page, the camera checks and the result files work as before; the result file also says which exam
(`exam_id`, `exam_title`).

**The status** says what can happen to an exam:

    draft  ->  published  ->  running  ->  closed  ->  results
                  ^  |                       |  ^         |
                  |  +-- back to draft       |  +---------+  (withdraw the results)
                  +--------- reopen ---------+

* draft: being written. Only a draft can be changed (title, time, rules, questions).
* published: finished and locked. Candidates cannot sit it yet. It can go back to draft (`unpublish`).
* running: open for candidates. `python exam_app.py` moves a published exam to running when it starts.
* closed: no more candidates. results: the results are released.
* An exam that was never published can be deleted. One that was is kept for the record.

**Commands** (`python exams.py ...`; `--db FILE` uses another records file):

    list                               all exams            show EX-XXXXXX [--key]   one exam (--key shows the answers)
    import questions.json              make a draft         set EX-XXXXXX --minutes 45 --rule show_score=false
    publish | unpublish | start | close | release | withdraw | reopen EX-XXXXXX
    duplicate EX-XXXXXX                a draft copy, for a re-sit or a new version
    delete EX-XXXXXX                   a draft that was never published
    audit [EX-XXXXXX] [--verify]       who did what and when; --verify checks the log was not altered

To run a particular exam: `python exam_app.py --exam EX-XXXXXX` (it must be published or running; a draft
needs `publish` first, a closed one needs `reopen`). `--minutes` and `--no-gaze` on the command line apply to
that run only and are not saved in the record.

**Rules** an exam can set (each is true, false or left open; open means the setting in `facelib/config.py`):
`shuffle_questions`, `shuffle_options`, `show_score`, `show_warnings`, `gaze`, `face_required`.

**The time now belongs to the exam.** It is stored when the exam is made (from `EXAM_MINUTES`, or `--minutes`
at import time in `exams.py`). Changing `EXAM_MINUTES` in `config.py` later does not change an exam that is already
in the records: use `python exams.py unpublish EX-...`, `set ... --minutes N`, `publish`, or `--minutes N` for one run.

**The audit log** records every change (created, edited, published, started, closed, ...) and, while an exam runs,
each sitting (registered, started, finished with the result file name). Each row holds a hash of the row
before it, so `audit --verify` finds a row that was edited or removed afterwards. That catches accidents and
casual changes; someone who can rewrite the whole `exams.db` file can rewrite the whole chain.

The answer key is kept in `exams.db` and is never sent to a candidate's browser. Keep that file (and `results/`)
somewhere only the invigilator and author can open.

Tested: 103 checks on the records (validation, every allowed and refused status move, the log and its tamper
detection, saving and reopening the file, several threads at once), 46 on running an exam from a record and the
launcher choices (including a real launcher run over HTTP), 6 in Chromium, and the earlier exam flow (22 checks) is unchanged.
Not done yet in this piece: a web page for authors (next), registered candidates with access codes, a separate
invigilator key for each exam, and results kept in the database instead of `results/` files.

## A room of candidates (one invigilator, many computers)

Each candidate sits at their own computer and runs the exam and the camera checks exactly as before. The
invigilator's computer (the "hub") shows every candidate at once, as a grid of camera pictures or as a list.

    python exam_app.py --hub                                   on the invigilator's computer (no camera needed)
    python exam_app.py --report-to 192.168.1.10:8765 --code K7QM-4XD2     on each candidate's computer
    python exam_app.py --report-to 192.168.1.10:8765 --code K7QM-4XD2 --label "Lab PC 3"    (a name for the grid)

The hub prints its invigilator key, a **room code** and the exact command to give each candidate (with its
network address filled in), then opens the console. The candidate's computer needs only the hub's address and
the room code. Both computers must be on the same network, and the hub's computer must allow the program
through its firewall (port 8765, or the one you give with `--port`).

* **The grid.** One tile per computer: the latest camera picture (refreshed every 2.5 s, a little slower for a
  big room), the candidate's name, question and time left, and the flags that are active right now in plain
  words. Tiles needing a look turn red and move to the front. "Needs a look" shows only those; "List" is the
  same facts as a table. The tab title shows `(!)` and, with sound alerts on, a beep sounds when a candidate
  newly needs a look.
* **One candidate.** Click a tile (or a row) for the same detailed console as before: the red band, the live
  picture, gaze map, face match, the full log with the picture taken at each flag, large view. "All
  candidates" (or the Esc key) goes back.
* **Connection lost.** A computer that has sent nothing for `ROOM_LOST_SECONDS` (8) shows "Connection lost".
  During an exam that counts as needing a look (the program may have been closed), and it is written in that
  candidate's log. It clears by itself when the computer reports again. A lost computer can be removed from
  the room. If the hub cannot be reached, the candidate's exam carries on as normal.
* **What a candidate's computer sends:** its status (name, question, time left, camera, faces, flags, gaze,
  face match), a small camera picture (`ROOM_PREVIEW_WIDTH` 320 px) every few seconds **only while that exam is in
  progress**, and the picture taken at each flag. Never the answers or the answer key. The notice the
  candidate ticks says this when a room is watching. The hub keeps all pictures in memory only, deletes a
  candidate's preview when they finish, and writes nothing to disk except short log lines.
* **Trust.** The hub treats every candidate computer as untrusted: it cuts everything it receives down to
  known fields and sizes, accepts only real small JPEG files, lets a computer change only its own entry, and
  serves a picture only if that computer's own flags name it. Joining needs the room code (five wrong codes
  lock joining for 30 s); the invigilator key is separate.

Limits: this is for a room on one local network (tested up to a handful of computers, built for about 10 to
50). The questions, the answer key, the marking and the result files still live on each candidate's own
computer; moving them to the hub is part of the CBT engine milestone. The hub is plain HTTP, so on a shared
or public network the room code and the pictures could be read by others on that network: use a network you
trust. If the hub program is restarted, its pictures and log of connections are gone and the computers join
again by themselves; each candidate's own computer still has its results, `events.csv` and snapshots.

## Many candidates, only a browser (milestone 2, piece 2)

Each candidate opens a web address on their own computer, types a personal access code and sits the exam. Nothing
is installed on their computer. This computer (the **exam computer**) keeps the questions, the answer key, the clock
and the marking, analyses every candidate's camera pictures, and shows all of them to the invigilator in the same grid
and list as the room.

**Gaze calibration is required.** When an exam uses gaze tracking there is no "calibrate again or continue without it" choice: a candidate whose calibration fails is told what to fix and the calibration starts again by itself after a few seconds (after three failures the page also says to call the invigilator). The exam server also refuses to start the exam without a good calibration, and refuses the skip request, so it cannot be bypassed from the page. To run an exam without gaze tracking, switch the rule off (`exams.py set EX-... --rule gaze=false`, drafts only) or use `--no-gaze` for one run.

**Live pictures for the invigilator:** in this mode the grid shows each candidate about twice a second, and the candidate whose detail is open about 3 to 4 times a second (the browser sends 4 pictures a second during the exam, `BROWSER_FRAME_FPS`, so that is the most you can see). The older room (`--hub`) shows a picture every few seconds, faster (about 2 a second) for the candidate whose detail is open.

**Test run with no preparation (demo mode).** `python exam_app.py --demo` uses the sample questions (`questions.json`), needs no
registered list and no access codes: students open the printed address, type their name and ID, allow the camera and start.
The invigilator console opens by itself on the exam computer. The list of who joined (name, id, status) is written as
`candidates_EX-....csv` in the project folder and kept up to date. The same ID with the same name again takes the sitting over;
the same ID with another name is refused. **Anyone on the network can join in this mode**, so use it only for testing.

**Before the exam (on the exam computer):**

    python exams.py import questions.json                       a draft exam (note its id, e.g. EX-7K3QMD)
    python exams.py publish EX-7K3QMD
    python exams.py candidates EX-7K3QMD import class.csv       a CSV with a header (name, id) or a JSON list
    python exams.py candidates EX-7K3QMD export slips.csv       every candidate's access code, for printing
    pip install cryptography                                    once (it makes the HTTPS certificate)
    python exam_app.py --exam EX-7K3QMD --serve                 starts the exam for all of them

`--serve` prints the addresses to give out (like `https://192.168.1.20:8443`), the invigilator key and the
invigilator link (it opens on the exam computer only). Allow the program through the firewall for port 8443 when
the computer asks. All computers must be on the same network.

**Candidate:** opens the address, accepts the browser's one-time warning (*Advanced*, then *Continue*: the exam computer
made its own certificate, and a browser only lets a page use the camera over `https://`), types the code from their slip,
ticks the notice, and allows the camera when the browser asks. They see themselves in a small window, then press
**Start exam**; the face check, the gaze calibration and the exam follow as before. A blocked, missing or busy camera
gets a plain message and a *Try the camera again* button. A wrong code gets a plain message; too many wrong codes from one
computer lock that computer for 30 seconds.

**Same code in a second browser** takes the sitting over: the first browser is cut off with an explanation, the
invigilator sees a REJOINED flag and the audit log records it. This lets a candidate whose browser crashed carry on, and makes
someone else using their code visible. A page reload carries on in the same exam.

**If the exam computer is restarted** the sittings in memory are gone. Candidates who were in progress are marked so; for
each, run `python exams.py candidates EX-7K3QMD reset ID`, and they can join again from the start. Submitted candidates cannot
sit again. Up to `BROWSER_MAX_SITTINGS` (60) can sit at once; the 61st sees "this exam computer is full".

**Settings** (`facelib/config.py`): `BROWSER_FRAME_FPS` (pictures per second during the exam, 4), `BROWSER_FRAME_FPS_SETUP` (10
during the face check and the gaze calibration, which need several readings per dot), `BROWSER_FRAME_WIDTH` (640),
`BROWSER_FRAME_QUALITY`, `BROWSER_STALE_SECONDS` (no pictures for this long counts as a lost camera, 5),
`EXAM_HTTPS_PORT` (8443). `--no-https` serves plain HTTP (the camera then works only on the exam computer itself).

**What was tested:** the host and sittings with a stand-in for the face model (105 checks: ten sittings together, no
mix-up of names, answers, pictures or flags, one result each, takeover, reset, locked-out guessing, a full computer, an
interrupted sitting, bad and oversized pictures, the certificate, HTTPS with keep-alive) and ten real Chromium windows with
fake cameras over HTTPS (28 checks: joining, camera check, exam, takeover, reload, blocked camera, wrong code, submitting, the
invigilator's room and pictures). All the older checks still pass.

**Not tested here, so test it before relying on it:**

* The **real face model** on pictures from a browser (this was built without it: a stand-in said how many faces were in a picture).
* The **face check and the gaze calibration with real faces** from a browser. Calibration needs about 10 pictures per second from
  every candidate that is calibrating; ten calibrating at once is close to the measured limit of a 12-core computer, so
  **start the candidates' calibrations a little apart**, and see how it behaves on your computer.
* **Gaze accuracy at 640 pixels** (the camera program uses a bigger picture).
* **Ten real computers on a real Wi-Fi**. Do one rehearsal with several real laptops and phones first: the printed address, the firewall,
  the Wi-Fi speed (about 16 Mbit/s for ten candidates) and the browsers you will really use (Chrome and Edge were the targets).

## Feasibility test for browser-only candidates (milestone 2)

The question: if students only open a web address (nothing installed), can one computer take their camera
pictures and analyse them all? Two small tools in `feasibility/` answer it on your own hardware.

    pip install cryptography                         once (it makes the HTTPS certificate)
    python feasibility/stream_test.py                the test server; prints the addresses to open
    python feasibility/bench_analysis.py             how many pictures per second the analysis alone can do

**stream_test.py** serves a page over HTTPS. Open the printed `https://...` address in a browser, accept the
certificate warning once (Advanced, then Continue: the certificate is made by the program itself), press **Start
camera** and choose **Allow** when the browser asks. The page then sends small camera pictures to the test
computer, which runs the same analysis as the exam (face landmarks and the flag checks) on every stream.
Open `/dash` on the test computer for the numbers: pictures in and analysed per second, how many were dropped,
how long a picture waited before analysis, the processor use, the network use, and a verdict (KEEPING UP or
AT CAPACITY). Keep the page that is sending in view: a browser tab that is hidden or minimised is slowed down by the browser, so open `/dash` in a separate window beside it (or on another computer), and check that the page's own line "pictures per second per stream" is close to what you asked for, otherwise the test is lighter than planned. To pretend to be many students from one webcam: `https://ADDRESS:8443/?streams=10&fps=6`. For a
cleaner measure, open that page on a second computer so the web page's own work does not use the test
computer's processor. A slow computer shows up as dropped pictures (the newest is always kept), never as a
growing delay.

**What to look for with 10 students** (`streams=10`, a minute or so): the verdict says KEEPING UP, dropped is
under 5%, "waited before analysis" is under about 300 ms and the processor is under about 75%. If not, try
`fps=4`: the flags that matter last seconds, so 4 pictures per second is usually enough and uses a third less.
The network use is about 200 KB/s per student at 640 pixels wide (about 16 Mbit/s for 10), which a normal
local network handles.

**bench_analysis.py** uses one picture of a face (`--image me.jpg`, or one picture from the webcam) and runs
the analysis on 1, 2 and 4 streams at once as fast as it can, then says roughly how many students that is.

Also shown by the test: over plain `http://` on a network address the browser refuses the camera and the page
says so (over `http://localhost` it is allowed, which is why today's one-computer exam works), and a student
who blocks the camera gets a message on how to allow it.

What was and was not checked: the HTTPS certificate, the camera permission, the picture streaming, the
analysis chain, the dropped-picture behaviour under overload, the refusal of bad requests and the numbers page
were tested in a real Chromium with a fake camera showing a real face photo (32 checks). **The real face model
was not available where this was built, so the real analysis speed was never measured**: the test used a
stand-in, so the figures it printed there say nothing about your computer. Run both tools on the computer that
would be the hub. Pictures sent by a browser are 640 pixels wide here, not the 1080p the camera program uses
today, so how well gaze tracking works at that size also still has to be checked. This is a test tool with no
sign-in: do not leave it running on a shared network.

## Face match during the exam

The camera program keeps checking that the person at the screen is the one who started the exam.
(Settings are in the "face match inside the exam" block of `facelib/config.py`; `EXAM_FACE_MATCH = False`
turns it all off.)

1. **At the start.** After the consent tick, the exam window shows a short "Face check" before the
   gaze calibration. The candidate looks at the screen and holds still while the camera program takes
   `EXAM_FACE_SAMPLES` (6) clear, front-facing pictures. If the view is no good it says why ("We cannot
   see your face", "More than one face is in view", "It is too dark", "Look straight at the screen",
   "Move a little closer", "Keep your eyes open"). The pictures must also look like the same person, so a
   second person stepping in during the check makes it fail. The exam clock starts only after this.
2. **All the way through.** About once a second (`EXAM_FACE_CHECK_SECONDS`) the live face is compared with
   those pictures. A different face for 3 s (`EXAM_FACE_MISMATCH_SECONDS`) raises `IDENTITY_MISMATCH`: the
   candidate sees a calm warning, the invigilator console shows it in red. When a face comes back after being
   missing or after a second face was in view, it is compared at once, so swapping places while out of view
   is caught as fast as a swap in view.
3. **When nobody can be matched.** If no clear front-facing view matched for `ID_UNVERIFIED_SECONDS` (45 s)
   while somebody is in view, `IDENTITY_UNVERIFIED` is raised ("could not confirm who is at the screen"). That
   is deliberately a different flag from a mismatch: looking away all the time is not a different person.
4. **What the invigilator sees.** A "Face match" row in the console (same person, different from the start,
   not confirmed for N s, waiting, skipped), the live score, the picture taken at the start next to every
   identity flag's picture, and a `FACE_ENROLLED` entry in the log. The result file has a `face_match`
   summary (pictures taken, how many comparisons, how many failed, lowest score, the start picture's name).
5. **What is kept.** The reference is a set of face signatures held in memory only, thrown away when the exam
   ends: no gallery file is written. One picture of the candidate at the start is saved with the other
   snapshots so a person can compare (`EXAM_FACE_SAVE_REFERENCE = False` stops that).
6. **Skipping.** If the check fails or the models are missing, the page offers "Try the face check again"
   and "Continue without face matching" (recorded as `FACE_MATCH_SKIPPED`, with the reason). Set
   `EXAM_FACE_REQUIRED = True` and there is no way around it.

What this does and does not tell you:

* The reference is made at the start of the exam, so it shows "the same person all the way through". It
  does **not** show "the person who was registered": there is no student list or registration photo yet.
  That needs a roster (milestone 2); `enroll_face.py` below is the old way to register a face beforehand.
* Two people who look alike can score above the pass mark (`ID_MATCH_THRESHOLD` 0.363) and are not caught.
* A printed photo or a picture on a phone held up to the camera can fool it: there is no liveness check.
* Face recognition is less accurate for some groups and in poor light. Use the live score in the console to
  tune the threshold on your own candidates. Every identity flag is a prompt for a person to look at the two
  pictures, never proof.

## Adding a flag (the checks array)

Open `facelib/checks.py` and add one line to `build_checks()`:

    Check("MOUTH_OPEN", 3.0, lambda s: s.blendshapes.get("jawOpen", 0) > 0.35)

`s` is the frame's `FrameState` (see `analysis.py` for every field). The rule returns True
while the condition is happening; the flag fires once it has stayed True for the given
seconds. Expression scores need `USE_BLENDSHAPES = True`; set `PRINT_BLENDSHAPES = True` to
print the strongest scores and pick thresholds.

## Identity check against a registered face (optional, older route)

This checks against faces you registered beforehand. The face check inside the exam (above) does not need it.

1. `python enroll_face.py "Candidate Name"` (asks for consent, then SPACE takes 5 samples)
2. In `facelib/config.py` set `USE_IDENTITY = True` (and `ID_EXPECTED_NAME` if several people
   are enrolled)
3. Run `python gaze_demo.py`. A different face for 6 seconds raises `IDENTITY_MISMATCH`.

`python enroll_face.py --list` shows who is enrolled; `--delete "Name"` removes someone.
Only face signatures (128 numbers per sample) are stored, in `gallery/faces.json`, never
photos. They are biometric data: get consent, keep the file private, delete it after the
exam window. Face recognition is less accurate for some groups and in poor light, so tune
`ID_MATCH_THRESHOLD` on your own candidates and treat a mismatch as a reason for a person to
look, never as proof.

## Not verified yet

The exam page was tested in a real Chromium browser against the real server, and the real camera
monitor (including the calibration and the gaze tracking) was tested with a simulated eye. The gaze
accuracy inside the exam is whatever your own calibration gives (about 2.5 cm on your laptop last time). It has not been run with your webcam together with
your browser yet. Fullscreen and window-leave detection behave a little differently between
browsers, so try the browser you will use for the demo.

The room (hub, reporters and grid) was tested over real HTTP on one computer with several simulated candidate computers, a real Chromium and the real launcher. It has not run across two real computers: check the firewall, the printed address and the picture speed on your own network first, and try it with several real cameras before relying on it. The invigilator console was tested in Chromium. Not checked: whether your browser opens the second window by itself (if it opens a tab instead, or nothing, use the printed link), and the sound alert on your computer.

Face match: the logic, the page, the console and the exam flow were tested (65 checks on the camera
thread with scripted people, 20 in Chromium), but with a **stand-in for the face recognition model**,
because the real YuNet and SFace files could not be downloaded where this was built. So the real
models have never run here: the first exam run downloads them (needs internet), and the pass mark
`ID_MATCH_THRESHOLD`, how many pictures are needed and how well the candidate's face matches under your
light are untested on real people. Try it with two real people in front of your own camera first.

Everything was tested with simulated faces (the covered-eye measure was also checked on one
real photo). Not yet run: the real MediaPipe model and the two identity models (YuNet and
SFace) on a real webcam, and the covered-eye, glance-count and eye up/down thresholds on real
hands and real people. The screen mapping was tested with a simulated eye (it calibrated and
tracked correctly) but never with a real face: check the cross-check error on your own camera.
Check the live numbers and thresholds on your own camera first.
