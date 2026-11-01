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

A room of candidates (one invigilator, many candidate computers on the same network):
          python exam_app.py --hub                       on the invigilator's computer: prints the room code and the
                                                         command each candidate runs
          python exam_app.py --report-to 192.168.1.10:8765 --code K7QM-4XD2     on each candidate's computer

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
    p.add_argument("--hub", action="store_true",
                   help="be the invigilator's computer for a room: candidates' computers report here (no camera needed)")
    p.add_argument("--report-to", metavar="ADDRESS", help="report to the invigilator's computer, e.g. 192.168.1.10:8765")
    p.add_argument("--code", help="the room code the invigilator's computer printed (with --report-to)")
    p.add_argument("--label", help="a name for this computer in the invigilator's grid (default: the computer's name)")
    args, _ = p.parse_known_args(argv)
    return args


def lan_addresses():
    """The addresses other computers on the network can use to reach this one."""
    import socket
    found = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sk:
            sk.connect(("10.255.255.255", 1))          # no traffic is sent: this only picks the network card
            found.append(sk.getsockname()[0])
    except OSError:
        pass
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            if ip not in found and not ip.startswith("127."):
                found.append(ip)
    except OSError:
        pass
    return found or ["<this computer's address>"]


def build(args):
    """Create the monitor and the app (separate from main() so tests can use them)."""
    from examapp.server import ExamApp
    if args.minutes:
        cfg.EXAM_MINUTES = args.minutes
    if args.no_gaze:
        cfg.EXAM_GAZE = False
    if args.show_gaze:
        cfg.EXAM_SHOW_GAZE_DOT = True
    room = None
    if args.hub:
        from examapp.hub import Room
        room = Room()
        args.no_camera = True          # the invigilator's computer does not run a camera
    if args.no_camera:
        from examapp.monitor import NullMonitor
        monitor = NullMonitor()
    else:
        from examapp.monitor import CameraMonitor
        monitor = CameraMonitor()
    return ExamApp(monitor, args.questions, room=room), monitor


def main(argv=None):
    args = parse_args(argv)
    try:
        app, monitor = build(args)
    except QuestionError as e:
        print(f"Question file problem: {e}")
        return 1
    from examapp.server import serve
    if args.report_to and args.hub:
        print("Use --hub on the invigilator's computer and --report-to on a candidate's computer, not both.")
        return 1
    if args.report_to and not args.code:
        print("--report-to needs the room code too: --code XXXX-XXXX (the invigilator's computer prints it).")
        return 1
    port = cfg.EXAM_PORT if args.port is None else args.port
    host = "0.0.0.0" if args.hub else None          # the hub must be reachable from the candidates' computers
    if args.hub:
        cfg.EXAM_HOST = host
    try:
        server = serve(app, host=host, port=port)
    except OSError as e:
        print(f"Cannot start on port {port}: {e}. Try --port 8766.")
        return 1
    monitor.start()
    reporter = None
    if args.report_to:
        from examapp.reporter import Reporter
        try:
            reporter = Reporter(app, args.report_to, args.code, label=args.label)
        except ValueError as e:
            print(e)
            return 1
        reporter.start()
    shown = "127.0.0.1" if args.hub else cfg.EXAM_HOST
    url = f"http://{shown}:{port}"
    invigilator = f"{url}/admin#key={app.auth.key}"       # the key rides in the link: nothing to type
    if args.hub:
        print(f"Room ready: {len(app.questions)} questions, {cfg.EXAM_MINUTES:g} minutes ({app.title}).")
        print(f"  Invigilator key:   {app.auth.key}")
        print(f"  Invigilator link:  {invigilator}   (keep private)")
        print(f"  Room code:         {app.room.code}")
        print("  On each candidate's computer run:")
        for ip in lan_addresses():
            print(f"      python exam_app.py --report-to {ip}:{port} --code {app.room.code}")
        print("  The candidates' computers must be on the same network and allow this program through the firewall.")
    else:
        print(f"Exam ready: {len(app.questions)} questions, {cfg.EXAM_MINUTES:g} minutes.")
        print(f"  Candidates open:   {url}")
        print(f"  Invigilator key:   {app.auth.key}")
        print(f"  Invigilator link:  {invigilator}   (keep private)")
        if reporter:
            print(f"  Reporting to:      {reporter.url}  (room code {args.code})")
    print("Press Ctrl+C here to stop the server.")
    if not args.no_browser:
        if args.hub:                                      # the hub opens only the invigilator console
            threading.Timer(0.8, lambda: webbrowser.open(invigilator)).start()
        else:
            threading.Timer(0.8, lambda: webbrowser.open(url)).start()
            if not args.no_invigilator_window and not reporter:
                threading.Timer(2.0, lambda: webbrowser.open_new(invigilator)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        if reporter:
            reporter.stop()
        monitor.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
