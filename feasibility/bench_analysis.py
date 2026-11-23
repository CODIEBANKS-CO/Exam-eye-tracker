"""
How many pictures per second can THIS computer analyse? (no network, no browser: just the analysis)

It takes one picture of a face (a photo you give with --image, or one picture from the webcam), then runs the real
face and flag analysis on it as fast as it can with 1, 2, 4... streams at the same time, and tells you how many
students that is. Run it on the computer that would be the hub, with the face model downloaded (the first run of
the exam or of gaze_demo.py does that).

    python feasibility/bench_analysis.py                    (uses the webcam once; look at the camera)
    python feasibility/bench_analysis.py --image me.jpg --seconds 8 --workers 1,2,4,8 --width 640
"""
import argparse
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import cv2

from facelib import config as cfg


class ImageCap:
    """A 'camera' that shows the same picture every time, as fast as it is asked."""
    def __init__(self, frame):
        self.frame = frame

    def isOpened(self): return True
    def read(self): return True, self.frame.copy()
    def release(self): pass


def worker(factory, frame, stop, out):
    from gaze_demo import Pipeline
    from facelib.analysis import Analyzer
    from facelib.checks import Monitor, build_checks
    n, spent, faces = 0, 0.0, 0
    try:
        with factory() as lm:
            pipe, monitor = Pipeline(ImageCap(frame), lm, Analyzer()), Monitor(build_checks(gallery_identity=False))
            for _ in range(3):                                   # warm up (the first pictures are slower)
                pipe.step()
            while not stop.is_set():
                t0 = time.perf_counter()
                _, _, state, now = pipe.step()
                monitor.update(state, now)
                spent += time.perf_counter() - t0
                n += 1
                faces = state.n_faces
    except BaseException as exc:
        out.append({"error": f"{type(exc).__name__}: {exc}"[:160]})
        return
    out.append({"n": n, "spent": spent, "faces": faces})


def get_frame(path, width):
    if path:
        frame = cv2.imread(path)
        if frame is None:
            raise SystemExit(f"Cannot read the picture {path}")
    else:
        from facelib.camera import open_camera
        cap = open_camera()
        for _ in range(15):
            ok, frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit("The webcam gave no picture.")
    h, w = frame.shape[:2]
    if w != width:
        frame = cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    return frame


def run(factory, frame, workers, seconds, quiet=False):
    results = []
    for w in workers:
        stop, out = threading.Event(), []
        threads = [threading.Thread(target=worker, args=(factory, frame, stop, out), daemon=True) for _ in range(w)]
        for t in threads:
            t.start()
        time.sleep(seconds)
        stop.set()
        for t in threads:
            t.join(30)
        errs = [o["error"] for o in out if "error" in o]
        good = [o for o in out if "n" in o]
        total = sum(o["n"] for o in good) / seconds
        ms = (sum(o["spent"] for o in good) / max(1, sum(o["n"] for o in good))) * 1000 if good else None
        row = {"workers": w, "fps_total": round(total, 1), "fps_each": round(total / w, 1), "ms": round(ms, 1) if ms else None,
               "faces": good[0]["faces"] if good else None, "error": errs[0] if errs else ""}
        results.append(row)
        if not quiet:
            print(f"  {w:>2} at once: {row['fps_total']:>6} pictures/s in total, {row['fps_each']:>5} each, {row['ms']} ms per picture"
                  + (f"   [{row['error']}]" if row["error"] else ""))
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description="How many pictures per second can this computer analyse?")
    p.add_argument("--image", help="a photo of a face (default: one picture from the webcam)")
    p.add_argument("--seconds", type=float, default=6.0, help="how long to run each test")
    p.add_argument("--workers", default="1,2,4", help="how many streams at once, e.g. 1,2,4,8")
    p.add_argument("--width", type=int, default=640, help="picture width sent by the browsers")
    p.add_argument("--fps", type=float, default=6.0, help="pictures per second each student's browser would send")
    args = p.parse_args(argv)
    from feasibility.stream_test import make_landmarker_factory
    factory, mode = make_landmarker_factory()
    if mode != "full":
        raise SystemExit("The face model is not on this computer, so the real analysis cannot be measured. Run gaze_demo.py once to download it.")
    frame = get_frame(args.image, args.width)
    print(f"{os.cpu_count()} processor cores. Picture {frame.shape[1]}x{frame.shape[0]}. Running each test for {args.seconds:g} s.")
    rows = run(factory, frame, [int(x) for x in args.workers.split(",") if x.strip()], args.seconds)
    if rows and rows[0]["faces"] == 0:
        print("WARNING: no face was found in the picture, so this measured the cheap no-face case. Use a clear, front-facing photo.")
    best = max(r["fps_total"] for r in rows if not r["error"]) if any(not r["error"] for r in rows) else 0
    print(f"\nBest total: {best} pictures per second. At {args.fps:g} pictures per second per student, with a quarter kept spare, "
          f"that is about {int(best * 0.75 / args.fps)} students for the analysis alone (the network and the web pages need some of the same computer).")
    return rows


if __name__ == "__main__":
    main()
