"""
HTTPS for the exam computer.

A browser only lets a page use the camera when the page comes over HTTPS (or from this very computer). So when
candidates open the exam by a network address, the exam computer must speak HTTPS. It makes its own certificate
("self-signed"): the browser shows a one-time warning ("Your connection is not private": choose Advanced, then
Continue), and after that the connection is encrypted and the camera is allowed.

The certificate is kept in the `cert/` folder and reused. It is made again when this computer's addresses change (a
different network) or when it is about to expire. It needs `pip install cryptography` (or the `openssl` program).
Keep `cert/key.pem` private: anyone who has it could pose as the exam computer.
"""

import datetime
import ipaddress
import os
import shutil
import socket
import ssl
import subprocess

CERT_DAYS = 365
RENEW_BEFORE_DAYS = 14


def lan_addresses():
    """The addresses other computers on the network can use to reach this one."""
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
    return found


def _is_ip(text):
    try:
        ipaddress.ip_address(text)
        return True
    except ValueError:
        return False


def wanted_names(extra_hosts=()):
    """(DNS names, IP addresses) the certificate must be valid for."""
    names = ["localhost", socket.gethostname()] + [h for h in extra_hosts if not _is_ip(h)]
    ips = ["127.0.0.1"] + [h for h in extra_hosts if _is_ip(h)]
    return list(dict.fromkeys(n for n in names if n)), list(dict.fromkeys(ips))


def _still_good(directory, names, ips):
    cert, key, listing = (os.path.join(directory, n) for n in ("cert.pem", "key.pem", "hosts.txt"))
    if not (os.path.isfile(cert) and os.path.isfile(key) and os.path.isfile(listing)):
        return False
    try:
        with open(listing, encoding="utf-8") as f:
            lines = f.read().split()
        have = set(lines[1:])
        expires = datetime.datetime.fromisoformat(lines[0])
    except (OSError, ValueError, IndexError):
        return False
    if expires - datetime.datetime.now(datetime.timezone.utc) < datetime.timedelta(days=RENEW_BEFORE_DAYS):
        return False
    return set(names) | set(ips) <= have


def make_cert(directory, extra_hosts=()):
    """Return (cert path, key path): the existing certificate when it still fits, else a new one."""
    os.makedirs(directory, exist_ok=True)
    names, ips = wanted_names(extra_hosts)
    cert, key = os.path.join(directory, "cert.pem"), os.path.join(directory, "key.pem")
    if _still_good(directory, names, ips):
        return cert, key
    expires = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=CERT_DAYS)
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
        k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "exam computer")])
        now = datetime.datetime.now(datetime.timezone.utc)
        san = [x509.DNSName(n) for n in names] + [x509.IPAddress(ipaddress.ip_address(i)) for i in ips]
        c = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(k.public_key())
             .serial_number(x509.random_serial_number())
             .not_valid_before(now - datetime.timedelta(minutes=5)).not_valid_after(expires)
             .add_extension(x509.SubjectAlternativeName(san), critical=False).sign(k, hashes.SHA256()))
        with open(cert, "wb") as f:
            f.write(c.public_bytes(serialization.Encoding.PEM))
        with open(key, "wb") as f:
            f.write(k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                                    serialization.NoEncryption()))
    except ImportError:
        if not shutil.which("openssl"):
            raise SystemExit("Cannot make the HTTPS certificate. Run:  pip install cryptography") from None
        alt = ",".join([f"DNS:{n}" for n in names] + [f"IP:{i}" for i in ips])
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", key, "-out", cert,
                        "-days", str(CERT_DAYS), "-subj", "/CN=exam computer", "-addext", f"subjectAltName={alt}"],
                       check=True, capture_output=True)
    try:
        os.chmod(key, 0o600)
    except OSError:
        pass
    with open(os.path.join(directory, "hosts.txt"), "w", encoding="utf-8") as f:
        f.write(expires.isoformat() + "\n" + "\n".join(names + ips) + "\n")
    return cert, key


def server_context(cert, key):
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(cert, key)
    return ctx
