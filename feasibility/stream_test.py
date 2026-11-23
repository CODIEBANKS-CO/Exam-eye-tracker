"""
Feasibility test for browser-only candidates (milestone 2).

Question it answers: if students only open a web address (no program installed), can ONE computer take their
camera pictures and run the real face and gaze analysis for all of them at once?

What it does
  * serves a page over HTTPS (a self-signed certificate, so the browser shows a one-time warning). Browsers only
    allow the camera on HTTPS or on localhost, so this is the part that makes "just open the address" possible;
  * the page asks for camera permission, then sends small camera pictures to this computer;
  * for every stream this computer runs the same analysis the exam uses today (facelib: face landmarks, the
    flag checks) and reports how many pictures per second it keeps up with, how many it had to drop, the delay,
    the network use and the processor use.

Run    python feasibility/stream_test.py
Then   open the printed https:// address in a browser on this or another computer, accept the certificate
       warning once, press Start camera.  Open  <address>/dash  to see the numbers.
       To pretend to be many students from one webcam:  <address>/?streams=8&fps=6
Stop   Ctrl+C.  This is a TEST tool: no sign-in, do not leave it running on a shared network.

Nothing is saved: pictures are analysed and thrown away.
"""

import argparse
import collections
import datetime
import ipaddress
import json
import os
import re
import socket
import ssl
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import cv2
import numpy as np

from facelib import config as cfg

MAX_FRAME = 400_000          # bytes of one picture
MAX_STREAMS = 64
IDLE_SECONDS = 10.0          # a stream that sends nothing for this long is dropped from the report
WINDOW = 5.0                 # rates are measured over the last few seconds
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


# ---------------------------------------------------------------------------------------- one stream
class QueueCap:
    """Looks like a webcam to gaze_demo.Pipeline, but hands over the newest picture the browser sent.
    Only ONE picture is kept: if the analysis is slower than the camera, older pictures are dropped (counted),
    so a slow computer shows up as dropped pictures instead of a growing delay."""

    def __init__(self):
        self.cv, self.frame, self.stopped, self.last_wait = threading.Condition(), None, False, 0.0
        self.t_put, self.last_lag = 0.0, 0.0            # when the newest picture arrived; how old it was when taken for analysis

    def isOpened(self):
        return True

    def put(self, frame):
        with self.cv:
            replaced = self.frame is not None
            self.frame, self.t_put = frame, time.perf_counter()
            self.cv.notify()
        return replaced

    def read(self):
        t0 = time.perf_counter()
        with self.cv:
            while self.frame is None and not self.stopped:
                self.cv.wait(0.5)
            frame, self.frame = self.frame, None
            self.last_lag = (time.perf_counter() - self.t_put) * 1000 if frame is not None else 0.0
        self.last_wait = time.perf_counter() - t0
        return (frame is not None), frame

    def stop(self):
        with self.cv:
            self.stopped = True
            self.cv.notify_all()

    def release(self):
        pass


class Stream(threading.Thread):
    def __init__(self, sid, make_landmarker):
        super().__init__(daemon=True)
        self.sid, self.make_landmarker = sid, make_landmarker
        self.cap, self.stop_flag = QueueCap(), threading.Event()
        self.events = collections.deque()                       # (time, "recv" | "drop" | "done", bytes)
        self.ms = collections.deque(maxlen=200)                 # analysis time per picture
        self.dec = collections.deque(maxlen=200)                # decode time per picture
        self.lag = collections.deque(maxlen=200)                # how long a picture waited before analysis started
        self.rtt = collections.deque(maxlen=50)                 # the browser's own round trip, reported back
        self.result = {"faces": None, "flags": [], "ms": None}
        self.received = self.dropped = self.analysed = self.bad = 0
        self.born = self.last_seen = time.time()
        self.last_mean = None
        self.error = ""

    def offer(self, jpeg, rtt=None):
        t0 = time.perf_counter()
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            self.bad += 1
            return False
        self.dec.append((time.perf_counter() - t0) * 1000)
        self.last_seen = time.time()
        self.last_mean = [round(float(v), 1) for v in frame.reshape(-1, 3).mean(axis=0)]
        self.received += 1
        self.events.append((self.last_seen, "recv", len(jpeg)))
        if self.cap.put(frame):
            self.dropped += 1
            self.events.append((self.last_seen, "drop", 0))
        if rtt is not None and 0 <= rtt < 60_000:
            self.rtt.append(rtt)
        return True

    def run(self):
        try:
            from gaze_demo import Pipeline
            from facelib.analysis import Analyzer
            from facelib.checks import Monitor, build_checks
            ctx = self.make_landmarker()
            with ctx as landmarker:
                pipe = Pipeline(self.cap, landmarker, Analyzer()) if landmarker is not None else None
                monitor = Monitor(build_checks(gallery_identity=False))
                while not self.stop_flag.is_set():
                    t0 = time.perf_counter()
                    if pipe is None:                                   # decode-only: no model on this computer
                        ok, frame = self.cap.read()
                        if not ok:
                            break
                        faces, flags = None, []
                    else:
                        out = pipe.step()
                        if out is None:
                            break
                        _, _, state, now = out
                        monitor.update(state, now)
                        faces, flags = state.n_faces, monitor.active_names
                    ms = (time.perf_counter() - t0 - self.cap.last_wait) * 1000
                    self.analysed += 1
                    if self.analysed > 2:                                  # the first pictures are slow while everything warms up
                        self.ms.append(ms)
                        self.lag.append(self.cap.last_lag)
                    self.events.append((time.time(), "done", 0))
                    self.result = {"faces": faces, "flags": flags, "ms": round(ms, 1)}
        except BaseException as exc:                                  # the test must say why, not just stop
            self.error = f"{type(exc).__name__}: {exc}"[:200]

    def stop(self):
        self.stop_flag.set()
        self.cap.stop()

    def rate(self, kind, now):
        while self.events and now - self.events[0][0] > WINDOW:
            self.events.popleft()
        span = max(1.0, min(WINDOW, now - self.born))
        items = [e for e in list(self.events) if e[1] == kind]
        return len(items) / span, sum(e[2] for e in items) / span


def _p95(values):
    values = sorted(values)
    return values[int(0.95 * (len(values) - 1))] if values else None


# ---------------------------------------------------------------------------------------- the server
class Hub:
    def __init__(self, make_landmarker, mode):
        self.make_landmarker, self.mode = make_landmarker, mode
        self.streams, self.lock = {}, threading.Lock()
        self.cores = os.cpu_count() or 1
        self.cpu = 0.0                                           # this program's processor use, 0-100 (all cores)
        self.cpu_last = (time.process_time(), time.time())
        self.started = time.time()

    def stream(self, sid):
        with self.lock:
            s = self.streams.get(sid)
            if s is None:
                if len(self.streams) >= MAX_STREAMS:
                    return None
                s = self.streams[sid] = Stream(sid, self.make_landmarker)
                s.start()
            return s

    def sample_cpu(self):
        p, w = time.process_time(), time.time()
        dp, dw = p - self.cpu_last[0], w - self.cpu_last[1]
        if dw >= 0.5:
            self.cpu = max(0.0, min(100.0, 100.0 * dp / dw / self.cores))
            self.cpu_last = (p, w)

    def reap(self):
        now = time.time()
        with self.lock:
            for sid in [k for k, s in self.streams.items() if now - s.last_seen > IDLE_SECONDS or (not s.is_alive() and s.error)]:
                if not self.streams[sid].error:
                    self.streams.pop(sid).stop()

    def report(self):
        self.sample_cpu()
        self.reap()
        now = time.time()
        rows, tin, tout, tdrop, tbytes = [], 0.0, 0.0, 0.0, 0.0
        with self.lock:
            streams = list(self.streams.values())
        for s in streams:
            fin, bps = s.rate("recv", now)
            fout, _ = s.rate("done", now)
            fdrop, _ = s.rate("drop", now)
            ms = list(s.ms)
            rows.append({"id": s.sid, "fps_in": round(fin, 1), "fps_done": round(fout, 1), "dropped_pct": round(100 * fdrop / fin, 1) if fin else 0.0,
                         "ms_med": round(sorted(ms)[len(ms) // 2], 1) if ms else None, "ms_p95": round(_p95(ms), 1) if ms else None,
                         "decode_ms": round(sum(s.dec) / len(s.dec), 1) if s.dec else None,
                         "lag_ms": round(sum(s.lag) / len(s.lag)) if s.lag else None,
                         "rtt_ms": round(sum(s.rtt) / len(s.rtt)) if s.rtt else None, "kb_s": round(bps / 1024, 1),
                         "faces": s.result["faces"], "flags": s.result["flags"], "age": round(now - s.born, 1), "error": s.error,
                         "mean_rgb": s.last_mean})
            tin += fin; tout += fout; tdrop += fdrop; tbytes += bps
        n = len(rows)
        steady = n > 0 and all(r["age"] > 8 for r in rows)
        loss = (100 * tdrop / tin) if tin else 0.0
        if self.mode != "full":
            verdict = "Measuring the pictures only: the face model is not on this computer, so the processor figures are NOT the real cost."
        elif not n:
            verdict = "No stream yet."
        elif not steady:
            verdict = "Measuring... (give it about 10 seconds)"
        elif loss > 10 or any(r["error"] for r in rows):
            verdict = f"AT CAPACITY: {loss:.0f}% of the pictures were dropped. This computer cannot analyse {n} streams at this rate."
        else:
            verdict = f"KEEPING UP with {n} stream{'s' if n != 1 else ''}: {loss:.0f}% dropped."
        # rough ceiling: assume the processor use grows in step with the number of streams and keep 25% spare
        est = int(n * 75.0 / self.cpu) if (self.mode == "full" and steady and self.cpu >= 5 and loss <= 10) else None
        return {"mode": self.mode, "cores": self.cores, "cpu_pct": round(self.cpu, 1), "streams": n,
                "fps_in": round(tin, 1), "fps_done": round(tout, 1), "dropped_pct": round(loss, 1), "mbit_s": round(tbytes * 8 / 1e6, 2),
                "estimate_streams": est, "verdict": verdict, "rows": sorted(rows, key=lambda r: r["id"])}


def make_cert(directory, extra_hosts=()):
    """A self-signed certificate for this computer (needs `pip install cryptography`, or the openssl program)."""
    os.makedirs(directory, exist_ok=True)
    cert, key = os.path.join(directory, "cert.pem"), os.path.join(directory, "key.pem")
    names = ["localhost", socket.gethostname()] + [h for h in extra_hosts if not _is_ip(h)]
    ips = ["127.0.0.1"] + [h for h in extra_hosts if _is_ip(h)]
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
        k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "exam feasibility test")])
        now = datetime.datetime.now(datetime.timezone.utc)
        san = [x509.DNSName(n) for n in dict.fromkeys(names)] + [x509.IPAddress(ipaddress.ip_address(i)) for i in dict.fromkeys(ips)]
        c = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(k.public_key()).serial_number(x509.random_serial_number())
             .not_valid_before(now - datetime.timedelta(minutes=5)).not_valid_after(now + datetime.timedelta(days=30))
             .add_extension(x509.SubjectAlternativeName(san), critical=False).sign(k, hashes.SHA256()))
        with open(cert, "wb") as f:
            f.write(c.public_bytes(serialization.Encoding.PEM))
        with open(key, "wb") as f:
            f.write(k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption()))
        return cert, key
    except ImportError:
        pass
    import shutil
    import subprocess
    if shutil.which("openssl"):
        alt = ",".join([f"DNS:{n}" for n in names] + [f"IP:{i}" for i in ips])
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", key, "-out", cert, "-days", "30",
                        "-subj", "/CN=exam feasibility test", "-addext", f"subjectAltName={alt}"], check=True, capture_output=True)
        return cert, key
    raise SystemExit("Cannot make a certificate. Run:  pip install cryptography")


def _is_ip(text):
    try:
        ipaddress.ip_address(text)
        return True
    except ValueError:
        return False


def lan_addresses():
    found = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sk:
            sk.connect(("10.255.255.255", 1))
            found.append(sk.getsockname()[0])
    except OSError:
        pass
    return found


def make_handler(hub):
    page = open(os.path.join(HERE, "stream.html"), "rb").read()
    dash = open(os.path.join(HERE, "dash.html"), "rb").read()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "feasibility"

        def setup(self):
            self.bad_tls = False
            if isinstance(self.request, ssl.SSLSocket):
                try:
                    self.request.settimeout(10)
                    self.request.do_handshake()          # here, in this connection's own thread, never in the accept loop
                    self.request.settimeout(None)
                except (ssl.SSLError, OSError):
                    self.bad_tls = True
            super().setup()

        def handle(self):
            if not self.bad_tls:
                super().handle()

        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            if not isinstance(body, bytes):
                body = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if self.close_connection:
                self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self._send(200, page, "text/html; charset=utf-8")
            elif path == "/dash":
                self._send(200, dash, "text/html; charset=utf-8")
            elif path == "/report":
                self._send(200, hub.report())
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            parts = urlparse(self.path)
            if parts.path != "/frame":
                self.close_connection = True              # the body was not read, so this connection cannot be reused
                return self._send(404, {"error": "not found"})
            sid = parse_qs(parts.query).get("id", [""])[0]
            try:
                size = int(self.headers.get("Content-Length", "-1"))
            except ValueError:
                size = -1
            if not ID_RE.match(sid) or size <= 0:
                self.close_connection = True
                return self._send(400, {"error": "bad request"})
            if size > MAX_FRAME:
                self.close_connection = True              # do not read what we refuse
                return self._send(413, {"error": "picture too big"})
            data = self.rfile.read(size)
            if data[:3] != b"\xff\xd8\xff":
                return self._send(400, {"error": "not a JPEG picture"})
            s = hub.stream(sid)
            if s is None:
                return self._send(503, {"error": "too many streams"})
            try:
                rtt = float(self.headers.get("X-Rtt", ""))
            except ValueError:
                rtt = None
            if not s.offer(data, rtt):
                return self._send(400, {"error": "cannot read that picture"})
            self._send(200, {**s.result, "error": s.error})

    return Handler


def make_landmarker_factory(require_model=False):
    """Returns (factory, mode). `mode` is "full" when the real face model is present, else "decode-only"."""
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    from gaze_demo import prepare_landmarker_model
    try:
        prepare_landmarker_model()
        ok = os.path.exists(cfg.LANDMARKER_PATH)
    except BaseException as exc:                                  # SystemExit when the download fails
        print(f"The face model is not available ({exc}).")
        ok = False
    if not ok:
        if require_model:
            raise SystemExit("The face model is needed (see the README, 'First run downloads...').")

        class Nothing:
            def __enter__(self): return None
            def __exit__(self, *a): return False
        return (lambda: Nothing()), "decode-only"

    def factory():
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=cfg.LANDMARKER_PATH), running_mode=vision.RunningMode.VIDEO,
            num_faces=2, output_face_blendshapes=cfg.USE_BLENDSHAPES, min_face_detection_confidence=0.5, min_tracking_confidence=0.5)
        return vision.FaceLandmarker.create_from_options(options)
    return factory, "full"


class QuietServer(ThreadingHTTPServer):
    """A browser closing its page (or a failed certificate check) is normal: say nothing. Anything else is still shown."""
    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], (ConnectionError, ssl.SSLError, TimeoutError)):
            return
        super().handle_error(request, client_address)


def serve(hub, host="0.0.0.0", port=8443, https=True, certdir=None):
    srv = QuietServer((host, port), make_handler(hub))
    srv.daemon_threads = True
    if https:
        cert, key = make_cert(certdir or os.path.join(HERE, "cert"), lan_addresses())
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(cert, key)
        srv.socket = ctx.wrap_socket(srv.socket, server_side=True, do_handshake_on_connect=False)
    return srv


def main(argv=None):
    p = argparse.ArgumentParser(description="Feasibility test: browser cameras streamed to one computer and analysed.")
    p.add_argument("--port", type=int, default=8443)
    p.add_argument("--http", action="store_true", help="plain HTTP (the browser will refuse the camera: for showing why HTTPS is needed)")
    p.add_argument("--require-model", action="store_true", help="stop if the face model is missing instead of measuring pictures only")
    args = p.parse_args(argv)
    factory, mode = make_landmarker_factory(args.require_model)
    hub = Hub(factory, mode)
    srv = serve(hub, port=args.port, https=not args.http)
    scheme = "http" if args.http else "https"
    addrs = lan_addresses() or ["<this computer's address>"]
    print(f"Feasibility test running ({'real face analysis' if mode == 'full' else 'pictures only: face model not available'}); {hub.cores} processor cores.")
    for a in addrs:
        print(f"  Students open:   {scheme}://{a}:{args.port}/            (many students from one webcam: ?streams=8&fps=6)")
        print(f"  The numbers:     {scheme}://{a}:{args.port}/dash")
    if not args.http:
        print("  The browser warns that the certificate is not trusted (it is made by this program): choose Advanced, then Continue.")
    print("Ctrl+C to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for s in list(hub.streams.values()):
            s.stop()
        srv.server_close()


if __name__ == "__main__":
    main()
