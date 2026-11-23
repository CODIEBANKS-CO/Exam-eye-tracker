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
          python exam_app.py --exam EX-7K3QMD         run an exam kept in the exam records (see exams.py)

Many candidates in their browsers, nothing installed on their computers (this computer does all the work):
          python exam_app.py --exam EX-7K3QMD --serve   candidates open the printed https:// address, type their personal
                                                         access code, allow the camera, and sit the exam. Register the
                                                         candidates first:  python exams.py candidates EX-7K3QMD import list.csv

A room of candidates (one invigilator, many candidate computers on the same network):
          python exam_app.py --hub                       on the invigilator's computer: prints the room code and the
                                                         command each candidate runs
          python exam_app.py --report-to 192.168.1.10:8765 --code K7QM-4XD2     on each candidate's computer

The invigilator console opens in a second browser window, already signed in: the program makes a
new key every time it starts (for example K7QM-4XD2) and puts it in that window's link.

Exams are kept as records in exams.db (questions, time, rules, status and an audit log); manage them with
exams.py. Running with no --exam imports questions.json (or --questions FILE) as an exam, once, and opens it.

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
    p.add_argument("--exam", metavar="ID", help="run this exam from the exam records (e.g. EX-7K3QMD; list them with exams.py list)")
    p.add_argument("--db", metavar="FILE", help="exam records file (default: exams.db)")
    p.add_argument("--minutes", type=float, help="time allowed (default: EXAM_MINUTES in facelib/config.py)")
    p.add_argument("--port", type=int, help="port (default: EXAM_PORT)")
    p.add_argument("--no-browser", action="store_true", help="do not open the browser by myself")
    p.add_argument("--no-invigilator-window", action="store_true",
                   help="do not open the invigilator console by myself (the link is still printed)")
    p.add_argument("--no-gaze", action="store_true", help="skip the gaze calibration and the look-away alert")
    p.add_argument("--show-gaze", action="store_true", help="draw the candidate's gaze point on their page (demo)")
    p.add_argument("--serve", action="store_true",
                   help="serve the exam to many candidates in their browsers over HTTPS (needs registered candidates)")
    p.add_argument("--demo", action="store_true",
                   help="a test run for many browsers with no access codes: uses the sample questions, students type their "
                        "name and ID, and the list of who joined is written to a CSV (implies --serve)")
    p.add_argument("--no-https", action="store_true", help="with --serve: plain HTTP (the camera only works on this computer then; for testing)")
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


class ExamChoiceError(Exception):
    """The exam asked for cannot be opened (not found, wrong status, bad question file)."""


def choose_exam(args):
    """Find the exam to run in the exam records and open it for candidates. Returns (store, exam).
    With --exam ID that exam is used. Otherwise the question file (--questions, or questions.json) is
    imported once (the same file content is never imported twice) and opened."""
    from examapp.store import ExamNotFound, ExamStateError, ExamStore, StoreError
    store = ExamStore(args.db or cfg.EXAM_DB_FILE)
    try:
        if args.exam:
            try:
                exam = store.get(args.exam)
            except ExamNotFound:
                known = ", ".join(e["id"] for e in store.list_exams()[:8]) or "none yet"
                raise ExamChoiceError(f"There is no exam {args.exam!r} in {store.path}. Exams there: {known}.")
        else:
            path = args.questions or cfg.EXAM_QUESTIONS_FILE
            exam, created = store.import_file(path, actor="launcher")
            if created:
                print(f"Imported {path} as exam {exam.id}.")
            if exam.status == "draft":
                exam = store.publish(exam.id, "launcher", "published automatically from the question file")
        return store, store.use_for_exam(exam.id, "launcher")
    except (ExamStateError, StoreError) as e:
        raise ExamChoiceError(str(e)) from None


def prepare_models():
    """Fetch the camera models now, once, so the candidates' sittings do not all try to at the same moment."""
    from gaze_demo import prepare_landmarker_model
    from facelib.models import ensure_file
    prepare_landmarker_model()
    if cfg.EXAM_FACE_MATCH:
        try:
            ensure_file(cfg.YUNET_PATH, cfg.YUNET_URL, "face detector model (YuNet)")
            ensure_file(cfg.SFACE_PATH, cfg.SFACE_URL, "face recognition model (SFace)")
        except SystemExit as e:                # the face check is then offered as unavailable; the exam still runs
            print(f"Face matching will not be available: {str(e).splitlines()[0]}")
            cfg.EXAM_FACE_MATCH = False


def build(args):
    """Create the monitor and the app (separate from main() so tests can use them)."""
    from examapp.server import ExamApp
    store, exam = choose_exam(args)
    if args.no_gaze:
        exam.rules["gaze"] = False         # the command line wins over the exam's own rule (this run only)
    if args.demo:
        args.serve = True
    if args.serve:
        if args.hub or args.report_to:
            raise ExamChoiceError("--serve cannot be combined with --hub or --report-to.")
        if args.demo:
            import os
            names = os.path.join(cfg.BASE_DIR, f"candidates_{exam.id}.csv")
            from examapp.multi import MultiHost
            return MultiHost(exam, store, minutes=args.minutes or None, open_join=True, names_file=names), None
        if not store.list_candidates(exam.id):
            raise ExamChoiceError(f"No candidates are registered for {exam.id}. Register them first, for example:\n"
                                  f"    python exams.py candidates {exam.id} import list.csv      (a CSV with name and id)")
        from examapp.multi import MultiHost
        return MultiHost(exam, store, minutes=args.minutes or None), None
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
    app = ExamApp(monitor, room=room, exam=exam, minutes=args.minutes or None)
    app.store = store
    return app, monitor


def serve_many(app, args):
    """--serve: HTTPS for the candidates on every network address, plain HTTP on this computer for the invigilator."""
    import os
    from examapp.server import serve
    port = args.port or cfg.EXAM_HTTPS_PORT
    cfg.EXAM_HOST = "0.0.0.0"
    tls = None
    if not args.no_https:
        from examapp.tls import make_cert
        tls = make_cert(os.path.join(cfg.BASE_DIR, "cert"), lan_addresses())
    try:
        public = serve(app, "0.0.0.0", port, tls=tls)
        local = serve(app, "127.0.0.1", cfg.EXAM_PORT)
    except OSError as e:
        print(f"Cannot start on port {port} or {cfg.EXAM_PORT}: {e}. Try --port 9443.")
        return 1
    scheme = "http" if args.no_https else "https"
    invigilator = f"http://127.0.0.1:{cfg.EXAM_PORT}/admin#key={app.auth.key}"
    counts = {}
    for c in app.store.list_candidates(app.exam_id):
        counts[c["status"]] = counts.get(c["status"], 0) + 1
    print(f"Exam: {app.title} ({app.exam_id}), {len(app.questions)} questions, {app.minutes:g} minutes.")
    print(f"  Registered candidates: {sum(counts.values())}  ({counts.get('submitted', 0)} submitted, "
          f"{counts.get('in_progress', 0)} marked in progress)")
    if app.open_join:
        print("  DEMO MODE: no access codes. Anyone on the network who opens the address can join by typing a name and an ID.")
        print(f"  Who joined is written to: {app.names_file}")
        print("  Candidates open one of these addresses in their browser:")
    else:
        print("  Candidates open one of these addresses in their browser (they type their own access code):")
    for ip in lan_addresses():
        print(f"      {scheme}://{ip}:{port}")
    if not args.no_https:
        print("  Their browser shows a one-time warning (\"Your connection is not private\"): choose Advanced, then Continue.")
        print("  That is expected: this computer made its own certificate. It is what lets the browser use the camera.")
    print(f"  Invigilator key:   {app.auth.key}")
    print(f"  Invigilator link:  {invigilator}   (keep private; opens on this computer only)")
    print(f"  Room code (not needed here): {app.room.code}")
    if counts.get("in_progress"):
        print(f"  {counts['in_progress']} candidate(s) were marked in progress when the program last stopped. Their sittings are\n"
              f"  gone from memory; let them sit again with:  python exams.py candidates {app.exam_id} reset <id>")
    print(f"  Up to {app.max_sittings} candidates at once. Allow this program through the firewall for port {port}.")
    print("Press Ctrl+C here to stop the server.")
    threading.Thread(target=local.serve_forever, daemon=True).start()
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(invigilator)).start()
    try:
        public.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        app.stop()
        public.server_close()
        local.shutdown()
        local.server_close()
    return 0


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.serve or args.demo:
            prepare_models()
        app, monitor = build(args)
        if args.serve or args.demo:
            return serve_many(app, args)
    except QuestionError as e:
        print(f"Question file problem: {e}")
        return 1
    except ExamChoiceError as e:
        print(f"Cannot open the exam: {e}")
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
        print(f"Room ready: {len(app.questions)} questions, {app.minutes:g} minutes ({app.title}, exam {app.exam_id}).")
        print(f"  Invigilator key:   {app.auth.key}")
        print(f"  Invigilator link:  {invigilator}   (keep private)")
        print(f"  Room code:         {app.room.code}")
        print("  On each candidate's computer run:")
        for ip in lan_addresses():
            print(f"      python exam_app.py --report-to {ip}:{port} --code {app.room.code}")
        print("  The candidates' computers must be on the same network and allow this program through the firewall.")
    else:
        print(f"Exam ready: {app.title} ({app.exam_id}): {len(app.questions)} questions, {app.minutes:g} minutes.")
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
