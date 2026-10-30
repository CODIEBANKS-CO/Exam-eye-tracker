"""
exam_app.py - the exam: 15 multiple-choice questions in the browser, with the webcam monitor
running next to it.

Install:  pip install opencv-python mediapipe numpy      (nothing else: the web part is standard Python)
Run:      python exam_app.py                 opens http://127.0.0.1:8765 in your browser
          python exam_app.py --no-camera     exam only, no webcam (fallback / for writing questions)
          python exam_app.py --show-gaze     also draw the candidate's gaze point on their page (demo)
          python exam_app.py --no-gaze       skip the gaze calibration / look-away alert
          python exam_app.py --no-invigilator-window   do not open the invigilator console by itself
          python exam_app.py --questions my_questions.json --minutes 30 --port 9000

The invigilator console opens in a second browser window, already signed in: the program makes a
new key every time it starts (for example K7QM-4XD2) and puts it in that window's link.

Questions are in questions.json. Results go to results/ (one JSON file per candidate plus
summary.csv). Camera flags also go to events.csv with a snapshot, tagged with the question number.
"""

import argparse
import sys
import threading
import webbrowser

from facelib import config as cfg
from examapp.engine import QuestionError


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Exam app with webcam monitoring.")
    p.add_argument("--no-camera", action="store_true", help="run the exam without the webcam monitor")
    p.add_argument("--questions", help="question file (default: questions.json)")
    p.add_argument("--minutes", type=float, help="time allowed (default: EXAM_MINUTES in facelib/config.py)")
    p.add_argument("--port", type=int, help="port (default: EXAM_PORT)")
    p.add_argument("--no-browser", action="store_true", help="do not open the browser by myself")
    p.add_argument("--no-invigilator-window", action="store_true",
                   help="do not open the invigilator console by myself (the link is still printed)")
    p.add_argument("--no-gaze", action="store_true", help="skip the gaze calibration and the look-away alert")
    p.add_argument("--show-gaze", action="store_true", help="draw the candidate's gaze point on their page (demo)")
    args, _ = p.parse_known_args(argv)
    return args


def build(args):
    """Create the monitor and the app (separate from main() so tests can use them)."""
    from examapp.server import ExamApp
    if args.minutes:
        cfg.EXAM_MINUTES = args.minutes
    if args.no_gaze:
        cfg.EXAM_GAZE = False
    if args.show_gaze:
        cfg.EXAM_SHOW_GAZE_DOT = True
    if args.no_camera:
        from examapp.monitor import NullMonitor
        monitor = NullMonitor()
    else:
        from examapp.monitor import CameraMonitor
        monitor = CameraMonitor()
    return ExamApp(monitor, args.questions), monitor


def main(argv=None):
    args = parse_args(argv)
    try:
        app, monitor = build(args)
    except QuestionError as e:
        print(f"Question file problem: {e}")
        return 1
    from examapp.server import serve
    port = cfg.EXAM_PORT if args.port is None else args.port
    try:
        server = serve(app, port=port)
    except OSError as e:
        print(f"Cannot start on port {port}: {e}. Try --port 8766.")
        return 1
    monitor.start()
    url = f"http://{cfg.EXAM_HOST}:{port}"
    invigilator = f"{url}/admin#key={app.auth.key}"       # the key rides in the link: nothing to type
    print(f"Exam ready: {len(app.questions)} questions, {cfg.EXAM_MINUTES:g} minutes.")
    print(f"  Candidates open:   {url}")
    print(f"  Invigilator key:   {app.auth.key}")
    print(f"  Invigilator link:  {invigilator}   (keep private)")
    print("Press Ctrl+C here to stop the server.")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
        if not args.no_invigilator_window:
            threading.Timer(2.0, lambda: webbrowser.open_new(invigilator)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        monitor.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
