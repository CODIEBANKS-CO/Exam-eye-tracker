"""
Exam records: the exam as something that is kept, not just a file that is read.

An exam has a title, a time limit, rules, its questions (with the answer key) and a status:

    draft  ->  published  ->  running  ->  closed  ->  results
                  ^  |                       |  ^         |
                  |  +-- back to draft       |  +---------+  (results withdrawn)
                  +--------- reopened -------+

  draft      being written: questions, time and rules can be changed
  published  finished and locked: nothing can be changed, candidates cannot sit it yet
  running    open: candidates can sit it
  closed     no more candidates; marking and review
  results    the results are released to the candidates

Every change is written to an audit log. The log rows are chained (each row holds a hash of the
one before), so a row that is edited or removed afterwards can be detected with verify_audit().
That shows accidental or casual changes; someone who can rewrite the whole database file can
rewrite the whole chain too.

The store is a single SQLite file (Python standard library). Nothing here knows about web pages
or cameras. The answer key lives here and is never sent to a candidate's browser.
"""

import hashlib
import csv
import io
import json
import re
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass, field

from facelib import config as cfg
from .engine import Question, QuestionError, parse_questions

SCHEMA_VERSION = 2
STATUSES = ("draft", "published", "running", "closed", "results")
TRANSITIONS = {
    "draft": ("published",),
    "published": ("draft", "running"),
    "running": ("closed",),
    "closed": ("results", "published"),
    "results": ("closed",),
}
# What each move is called in the log and on the command line.
MOVES = {("draft", "published"): "publish", ("published", "draft"): "unpublish",
         ("published", "running"): "start", ("running", "closed"): "close",
         ("closed", "results"): "release_results", ("results", "closed"): "withdraw_results",
         ("closed", "published"): "reopen"}
# Rules an exam may set. A rule left out (or None) means "use the program's setting" (facelib/config.py).
RULE_NAMES = {"shuffle_questions": "EXAM_SHUFFLE_QUESTIONS", "shuffle_options": "EXAM_SHUFFLE_OPTIONS",
              "show_score": "EXAM_SHOW_SCORE", "show_warnings": "EXAM_SHOW_WARNINGS",
              "gaze": "EXAM_GAZE", "face_required": "EXAM_FACE_REQUIRED"}

MAX_TITLE, MAX_DESCRIPTION = 120, 2000
MAX_QUESTIONS, MAX_TEXT, MAX_OPTION = 500, 2000, 500
MAX_MINUTES = 600
MAX_CANDIDATES = 2000
CANDIDATE_STATUSES = ("registered", "in_progress", "submitted")
CANDIDATE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/\- ]{0,39}$")
CODE_LENGTH = 8
ID_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"    # no O, 0, I, 1: easy to read out and to type


class StoreError(ValueError):
    """The request cannot be done (bad data, or not allowed in the exam's status)."""


class ExamNotFound(StoreError):
    pass


class ExamStateError(StoreError):
    """The exam's status does not allow this."""


@dataclass
class Exam:
    id: str
    title: str
    description: str
    minutes: float
    status: str
    rules: dict
    questions: list = field(default_factory=list)       # [Question]
    created: float = 0.0
    updated: float = 0.0
    published_at: float = None
    started_at: float = None
    closed_at: float = None
    source: str = None

    def rule(self, name):
        """The exam's own setting for `name`, or the program's setting when the exam leaves it open."""
        value = self.rules.get(name)
        return getattr(cfg, RULE_NAMES[name]) if value is None else value

    def summary(self):
        return {"id": self.id, "title": self.title, "minutes": self.minutes, "status": self.status,
                "questions": len(self.questions), "created": self.created, "updated": self.updated}


def format_time(epoch):
    return "" if epoch is None else time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch))


def clean_rules(rules):
    """Check a rules dict and return it without the open (None) entries."""
    if rules is None:
        return {}
    if not isinstance(rules, dict):
        raise StoreError("The rules must be a set of name: true/false pairs.")
    out = {}
    for name, value in rules.items():
        if name not in RULE_NAMES:
            raise StoreError(f"Unknown rule {name!r}. The rules are: {', '.join(sorted(RULE_NAMES))}.")
        if value is None:
            continue
        if not isinstance(value, bool):
            raise StoreError(f"The rule {name!r} must be true or false.")
        out[name] = value
    return out


def clean_minutes(minutes):
    if isinstance(minutes, bool) or not isinstance(minutes, (int, float)) or not 0 < minutes <= MAX_MINUTES:
        raise StoreError(f"The time allowed must be a number of minutes above 0 and at most {MAX_MINUTES}.")
    return float(minutes)


def clean_title(title):
    if not isinstance(title, str) or not title.strip():
        raise StoreError("The exam needs a title.")
    if len(title.strip()) > MAX_TITLE:
        raise StoreError(f"The title is longer than {MAX_TITLE} characters.")
    return title.strip()


def clean_questions(items):
    """Check a list of question dicts (id optional) and return [Question]. Raises StoreError."""
    if not isinstance(items, list):
        raise StoreError("The questions must be a list.")
    if len(items) > MAX_QUESTIONS:
        raise StoreError(f"An exam can have at most {MAX_QUESTIONS} questions.")
    used = {i.get("id") for i in items if isinstance(i, dict) and isinstance(i.get("id"), str)}
    prepared, counter = [], 0
    for item in items:
        if isinstance(item, dict) and not (isinstance(item.get("id"), str) and item["id"].strip()):
            counter += 1
            while f"q{counter}" in used:
                counter += 1
            used.add(f"q{counter}")
            item = {**item, "id": f"q{counter}"}
        prepared.append(item)
    if not prepared:
        return []
    try:
        _, questions = parse_questions({"questions": prepared})
    except QuestionError as e:
        raise StoreError(str(e).replace("question file", "questions")) from None
    for n, q in enumerate(questions, 1):
        if len(q.text) > MAX_TEXT or any(len(o) > MAX_OPTION for o in q.options):
            raise StoreError(f"question {n} ({q.id}): the text or an option is too long "
                             f"({MAX_TEXT} / {MAX_OPTION} characters at most)")
    return questions


class ExamStore:
    """The exam records and their audit log in one SQLite file (path ':memory:' for tests)."""

    def __init__(self, path=None, clock=time.time):
        self.path = path or cfg.EXAM_DB_FILE
        self.clock = clock
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._migrate()

    def disconnect(self):
        """Close the database file (close() is the lifecycle move: running -> closed)."""
        with self._lock:
            self._db.close()

    # ------------------------------ plumbing ------------------------------
    def _migrate(self):
        with self._lock:
            version = self._db.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise StoreError(f"{self.path} was made by a newer version of this program "
                                 f"(database version {version}, this program knows {SCHEMA_VERSION}).")
            if version < 1:
                self._db.executescript("""
                    BEGIN;
                    CREATE TABLE exams(
                        id TEXT PRIMARY KEY, title TEXT NOT NULL, description TEXT NOT NULL DEFAULT '',
                        minutes REAL NOT NULL, status TEXT NOT NULL, rules TEXT NOT NULL DEFAULT '{}',
                        source TEXT, source_hash TEXT, created REAL NOT NULL, updated REAL NOT NULL,
                        published_at REAL, started_at REAL, closed_at REAL);
                    CREATE TABLE questions(
                        exam_id TEXT NOT NULL REFERENCES exams(id) ON DELETE CASCADE,
                        position INTEGER NOT NULL, qid TEXT NOT NULL, text TEXT NOT NULL,
                        options TEXT NOT NULL, answer INTEGER NOT NULL,
                        PRIMARY KEY(exam_id, position), UNIQUE(exam_id, qid));
                    CREATE TABLE audit(
                        seq INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, actor TEXT NOT NULL,
                        action TEXT NOT NULL, exam_id TEXT, detail TEXT NOT NULL DEFAULT '',
                        prev TEXT NOT NULL, hash TEXT NOT NULL);
                    PRAGMA user_version = 1;
                    COMMIT;""")
            if version < 2:
                self._db.executescript("""
                    BEGIN;
                    CREATE TABLE candidates(
                        exam_id TEXT NOT NULL REFERENCES exams(id) ON DELETE CASCADE,
                        candidate_id TEXT NOT NULL, name TEXT NOT NULL, code TEXT NOT NULL UNIQUE,
                        status TEXT NOT NULL DEFAULT 'registered', created REAL NOT NULL,
                        started_at REAL, finished_at REAL, PRIMARY KEY(exam_id, candidate_id));
                    PRAGMA user_version = 2;
                    COMMIT;""")

    def _transaction(self):
        return _Transaction(self)

    def _log(self, actor, action, exam_id=None, detail=""):
        """Append a chained row to the audit log. Call inside a transaction."""
        row = self._db.execute("SELECT hash FROM audit ORDER BY seq DESC LIMIT 1").fetchone()
        prev = row["hash"] if row else "0" * 64
        ts, actor, detail = self.clock(), str(actor or "unknown")[:60], str(detail)[:1000]
        digest = _audit_hash(prev, ts, actor, action, exam_id, detail)
        self._db.execute("INSERT INTO audit(ts, actor, action, exam_id, detail, prev, hash) VALUES (?,?,?,?,?,?,?)",
                         (ts, actor, action, exam_id, detail, prev, digest))

    def _new_id(self):
        for _ in range(50):
            candidate = "EX-" + "".join(secrets.choice(ID_ALPHABET) for _ in range(6))
            if not self._db.execute("SELECT 1 FROM exams WHERE id = ?", (candidate,)).fetchone():
                return candidate
        raise StoreError("Could not make a free exam id.")        # cannot happen in practice

    def _row(self, exam_id):
        row = self._db.execute("SELECT * FROM exams WHERE id = ?", (str(exam_id).strip().upper(),)).fetchone()
        if row is None:
            raise ExamNotFound(f"There is no exam {exam_id!r}.")
        return row

    def _exam(self, row):
        qs = self._db.execute("SELECT * FROM questions WHERE exam_id = ? ORDER BY position", (row["id"],)).fetchall()
        return Exam(id=row["id"], title=row["title"], description=row["description"], minutes=row["minutes"],
                    status=row["status"], rules=json.loads(row["rules"]),
                    questions=[Question(q["qid"], q["text"], json.loads(q["options"]), q["answer"]) for q in qs],
                    created=row["created"], updated=row["updated"], published_at=row["published_at"],
                    started_at=row["started_at"], closed_at=row["closed_at"], source=row["source"])

    def _write_questions(self, exam_id, questions):
        self._db.execute("DELETE FROM questions WHERE exam_id = ?", (exam_id,))
        for position, q in enumerate(questions):
            self._db.execute("INSERT INTO questions(exam_id, position, qid, text, options, answer) VALUES (?,?,?,?,?,?)",
                             (exam_id, position, q.id, q.text, json.dumps(q.options, ensure_ascii=False), q.answer))

    def _need_draft(self, row, what):
        if row["status"] != "draft":
            raise ExamStateError(f"{row['id']} is {row['status']}: {what} can only be changed while it is a draft. "
                                 f"Move it back to draft first (unpublish) if it has not started.")

    # ------------------------------- reading -------------------------------
    def get(self, exam_id):
        with self._lock:
            return self._exam(self._row(exam_id))

    def exists(self, exam_id):
        with self._lock:
            return self._db.execute("SELECT 1 FROM exams WHERE id = ?", (str(exam_id).strip().upper(),)).fetchone() is not None

    def list_exams(self, status=None):
        """Summaries (no questions), newest first."""
        if status is not None and status not in STATUSES:
            raise StoreError(f"Unknown status {status!r}.")
        with self._lock:
            sql = ("SELECT e.*, (SELECT COUNT(*) FROM questions q WHERE q.exam_id = e.id) AS n FROM exams e"
                   + (" WHERE status = ?" if status else "") + " ORDER BY created DESC, id")
            rows = self._db.execute(sql, (status,) if status else ()).fetchall()
        return [{"id": r["id"], "title": r["title"], "minutes": r["minutes"], "status": r["status"],
                 "questions": r["n"], "created": r["created"], "updated": r["updated"]} for r in rows]

    # ------------------------------- writing -------------------------------
    def create_exam(self, title, minutes=None, questions=None, rules=None, description="", actor="local",
                    source=None, source_hash=None):
        """Make a new exam as a draft. `questions` is a list of dicts (id optional)."""
        title = clean_title(title)
        minutes = clean_minutes(cfg.EXAM_MINUTES if minutes is None else minutes)
        rules = clean_rules(rules)
        qs = clean_questions(questions or [])
        description = str(description or "").strip()
        if len(description) > MAX_DESCRIPTION:
            raise StoreError(f"The description is longer than {MAX_DESCRIPTION} characters.")
        with self._lock, self._transaction():
            exam_id, now = self._new_id(), self.clock()
            self._db.execute("INSERT INTO exams(id, title, description, minutes, status, rules, source, source_hash, "
                             "created, updated) VALUES (?,?,?,?, 'draft', ?,?,?,?,?)",
                             (exam_id, title, description, minutes, json.dumps(rules), source, source_hash, now, now))
            self._write_questions(exam_id, qs)
            self._log(actor, "create", exam_id, f"{title!r}, {len(qs)} questions, {minutes:g} minutes"
                      + (f", from {source}" if source else ""))
            return self._exam(self._row(exam_id))

    def update_exam(self, exam_id, title=None, minutes=None, rules=None, description=None, actor="local"):
        """Change the title, time, rules or description of a draft. Only the arguments given are changed;
        for `rules`, the rules named are set (None opens a rule again) and the others are kept."""
        with self._lock, self._transaction():
            row = self._row(exam_id)
            self._need_draft(row, "the exam's details")
            changes, notes = {}, []
            if title is not None:
                changes["title"] = clean_title(title)
                notes.append("title")
            if minutes is not None:
                changes["minutes"] = clean_minutes(minutes)
                notes.append(f"minutes={changes['minutes']:g}")
            if description is not None:
                text = str(description).strip()
                if len(text) > MAX_DESCRIPTION:
                    raise StoreError(f"The description is longer than {MAX_DESCRIPTION} characters.")
                changes["description"] = text
                notes.append("description")
            if rules is not None:
                if not isinstance(rules, dict):
                    raise StoreError("The rules must be a set of name: true/false pairs.")
                merged = {**json.loads(row["rules"]), **rules}
                changes["rules"] = json.dumps(clean_rules(merged))
                notes.append("rules " + ", ".join(f"{k}={v}" for k, v in sorted(rules.items())))
            if not changes:
                return self._exam(row)
            changes["updated"] = self.clock()
            self._db.execute("UPDATE exams SET " + ", ".join(f"{k} = ?" for k in changes) + " WHERE id = ?",
                             (*changes.values(), row["id"]))
            self._log(actor, "update", row["id"], "; ".join(notes))
            return self._exam(self._row(row["id"]))

    def set_questions(self, exam_id, items, actor="local"):
        """Replace all the questions of a draft with `items` (a list of dicts, id optional)."""
        qs = clean_questions(items)
        with self._lock, self._transaction():
            row = self._row(exam_id)
            self._need_draft(row, "the questions")
            self._write_questions(row["id"], qs)
            self._db.execute("UPDATE exams SET updated = ? WHERE id = ?", (self.clock(), row["id"]))
            self._log(actor, "set_questions", row["id"], f"{len(qs)} questions")
            return self._exam(self._row(row["id"]))

    def delete_exam(self, exam_id, actor="local"):
        """Delete a draft. An exam that was ever published is kept (close it instead)."""
        with self._lock, self._transaction():
            row = self._row(exam_id)
            if row["status"] != "draft":
                raise ExamStateError(f"{row['id']} is {row['status']}: only a draft can be deleted.")
            if row["published_at"] is not None:
                raise ExamStateError(f"{row['id']} was published once, so it is kept for the record.")
            self._db.execute("DELETE FROM exams WHERE id = ?", (row["id"],))
            self._log(actor, "delete", row["id"], row["title"])

    def duplicate(self, exam_id, actor="local", title=None):
        """A new draft copy of an exam (same questions and rules), for a re-sit or a new version."""
        with self._lock:
            source = self.get(exam_id)
            return self.create_exam(title or f"{source.title} (copy)", source.minutes,
                                    [{"id": q.id, "text": q.text, "options": list(q.options), "answer": q.answer}
                                     for q in source.questions],
                                    source.rules, source.description, actor, source=f"copy of {source.id}")

    def import_data(self, data, actor="local", title=None, minutes=None, source=None):
        """Make a draft from question data in the question file's shape. The same data (compared by
        content) is not imported twice: the existing exam is returned. Returns (exam, created)."""
        if not isinstance(data, dict):
            raise StoreError("The question data must be an object with a \"questions\" list.")
        items = data.get("questions")
        if not isinstance(items, list) or not items:
            raise StoreError("The question data needs a non-empty \"questions\" list.")
        digest = hashlib.sha256(json.dumps({"t": data.get("title"), "q": items, "m": minutes},
                                           sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        with self._lock:
            found = self._db.execute("SELECT id FROM exams WHERE source_hash = ? ORDER BY created LIMIT 1",
                                     (digest,)).fetchone()
            if found:
                return self.get(found["id"]), False
            return self.create_exam(title or str(data.get("title") or "Exam"), minutes, items,
                                    data.get("rules"), str(data.get("description") or ""), actor,
                                    source=source or "import", source_hash=digest), True

    def import_file(self, path, actor="local", title=None, minutes=None):
        """Import a JSON question file as a draft (see import_data). Returns (exam, created)."""
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except OSError as e:
            raise StoreError(f"Cannot open the question file {path}: {e}") from None
        except ValueError as e:
            raise StoreError(f"The question file {path} is not valid JSON: {e}") from None
        return self.import_data(data, actor, title, minutes, source=f"file {_base(path)}")


    # ------------------------------- registered candidates -------------------------------
    # Each candidate registered for an exam gets a personal access code (XXXX-XXXX). The code is all
    # the candidate types to join: it names the exam and the person. Codes are kept here in clear so the
    # invigilator can print them again; keep exams.db private like the answer key.
    def _new_code(self):
        for _ in range(100):
            raw = "".join(secrets.choice(ID_ALPHABET) for _ in range(CODE_LENGTH))
            if not self._db.execute("SELECT 1 FROM candidates WHERE code = ?", (raw,)).fetchone():
                return raw
        raise StoreError("Could not make a free access code.")        # cannot happen in practice

    @staticmethod
    def normalise_code(text):
        """'k7qm-4xd2', ' K7QM 4XD2 ' and 'K7QM4XD2' are the same code."""
        return re.sub(r"[^A-Za-z0-9]", "", str(text or "")).upper()

    @staticmethod
    def show_code(raw):
        return f"{raw[:4]}-{raw[4:]}"

    def _candidate_dict(self, r):
        return {"exam_id": r["exam_id"], "candidate_id": r["candidate_id"], "name": r["name"],
                "code": self.show_code(r["code"]), "status": r["status"], "created": r["created"],
                "started_at": r["started_at"], "finished_at": r["finished_at"]}

    def add_candidates(self, exam_id, rows, actor="local"):
        """Register candidates for an exam. `rows` is a list of {"name": ..., "candidate_id": ...} (the id is
        optional: C001, C002, ... are made). All or none. Returns the new candidates, codes included.
        Allowed while the exam is a draft, published or running."""
        if not isinstance(rows, list) or not rows:
            raise StoreError("Give at least one candidate.")
        with self._lock, self._transaction():
            row = self._row(exam_id)
            if row["status"] in ("closed", "results"):
                raise ExamStateError(f"{row['id']} is {row['status']}: candidates can no longer be added.")
            have = {r["candidate_id"].lower() for r in self._db.execute(
                "SELECT candidate_id FROM candidates WHERE exam_id = ?", (row["id"],))}
            if len(have) + len(rows) > MAX_CANDIDATES:
                raise StoreError(f"An exam can have at most {MAX_CANDIDATES} registered candidates.")
            made, counter = [], len(have)
            for number, item in enumerate(rows, 1):
                where = f"candidate {number}"
                if not isinstance(item, dict):
                    raise StoreError(f"{where}: must be a name and an id")
                name = str(item.get("name") or "").strip()
                if not name or len(name) > 80:
                    raise StoreError(f"{where}: the name is missing or longer than 80 characters")
                cid = str(item.get("candidate_id") or "").strip()
                if not cid:
                    counter += 1
                    while f"C{counter:03d}".lower() in have:
                        counter += 1
                    cid = f"C{counter:03d}"
                if not CANDIDATE_ID_RE.match(cid):
                    raise StoreError(f"{where} ({name}): the id {cid!r} may use letters, digits and . _ / - and spaces, "
                                     f"up to 40 characters")
                if cid.lower() in have:
                    raise StoreError(f"{where} ({name}): the id {cid!r} is already registered for this exam")
                have.add(cid.lower())
                code = self._new_code()
                self._db.execute("INSERT INTO candidates(exam_id, candidate_id, name, code, status, created) "
                                 "VALUES (?,?,?,?, 'registered', ?)", (row["id"], cid, name, code, self.clock()))
                made.append(self._candidate_dict(self._db.execute(
                    "SELECT * FROM candidates WHERE code = ?", (code,)).fetchone()))
            self._log(actor, "register_candidates", row["id"], f"{len(made)} candidates")
            return made

    def import_candidates_data(self, exam_id, text, actor="local"):
        """Register candidates from CSV text (a header with name and id, in any order) or a JSON list."""
        text = str(text or "").lstrip("\ufeff")
        if text.lstrip().startswith(("[", "{")):
            try:
                data = json.loads(text)
            except ValueError as e:
                raise StoreError(f"The candidate list is not valid JSON: {e}") from None
            if isinstance(data, dict):
                data = data.get("candidates")
            if not isinstance(data, list):
                raise StoreError("The JSON must be a list of {\"name\": ..., \"id\": ...} objects.")
            rows = [{"name": (i or {}).get("name") if isinstance(i, dict) else None,
                     "candidate_id": next((i.get(k) for k in ("id", "candidate_id", "matric", "reg_no") if isinstance(i, dict) and i.get(k)), None)}
                    for i in data]
        else:
            lines = [ln for ln in text.splitlines() if ln.strip()]
            if not lines:
                raise StoreError("The candidate list is empty.")
            reader = list(csv.reader(lines))
            header = [h.strip().lower() for h in reader[0]]
            name_cols = [i for i, h in enumerate(header) if h in ("name", "full name", "full_name", "fullname", "candidate")]
            id_cols = [i for i, h in enumerate(header) if h in ("id", "candidate_id", "candidate id", "matric", "reg_no", "regno", "reg no")]
            if name_cols:
                ncol, icol, body = name_cols[0], (id_cols[0] if id_cols else None), reader[1:]
            else:                                     # no header: name first, id second
                ncol, icol, body = 0, (1 if len(reader[0]) > 1 else None), reader
            rows = []
            for r in body:
                if not any(c.strip() for c in r):
                    continue
                rows.append({"name": r[ncol] if ncol < len(r) else "",
                             "candidate_id": r[icol] if icol is not None and icol < len(r) else ""})
        return self.add_candidates(exam_id, rows, actor)

    def import_candidates_file(self, exam_id, path, actor="local"):
        try:
            with open(path, encoding="utf-8-sig") as f:
                text = f.read()
        except OSError as e:
            raise StoreError(f"Cannot open the candidate list {path}: {e}") from None
        return self.import_candidates_data(exam_id, text, actor)

    def list_candidates(self, exam_id):
        with self._lock:
            row = self._row(exam_id)
            rows = self._db.execute("SELECT * FROM candidates WHERE exam_id = ? ORDER BY candidate_id", (row["id"],)).fetchall()
            return [self._candidate_dict(r) for r in rows]

    def find_candidate(self, code):
        """The registered candidate with this access code, or None. (The code names the exam too.)"""
        raw = self.normalise_code(code)
        if len(raw) != CODE_LENGTH:
            return None
        with self._lock:
            r = self._db.execute("SELECT * FROM candidates WHERE code = ?", (raw,)).fetchone()
            return None if r is None else self._candidate_dict(r)

    def remove_candidate(self, exam_id, candidate_id, actor="local"):
        """Remove a candidate who has not started."""
        with self._lock, self._transaction():
            row = self._row(exam_id)
            c = self._db.execute("SELECT * FROM candidates WHERE exam_id = ? AND lower(candidate_id) = lower(?)",
                                 (row["id"], str(candidate_id).strip())).fetchone()
            if c is None:
                raise StoreError(f"{candidate_id!r} is not registered for {row['id']}.")
            if c["status"] != "registered":
                raise ExamStateError(f"{c['candidate_id']} is {c['status'].replace('_', ' ')} and cannot be removed.")
            self._db.execute("DELETE FROM candidates WHERE exam_id = ? AND candidate_id = ?", (row["id"], c["candidate_id"]))
            self._log(actor, "remove_candidate", row["id"], c["candidate_id"])

    def new_code(self, exam_id, candidate_id, actor="local"):
        """A new access code for a candidate who has not finished (the old one stops working)."""
        with self._lock, self._transaction():
            row = self._row(exam_id)
            c = self._db.execute("SELECT * FROM candidates WHERE exam_id = ? AND lower(candidate_id) = lower(?)",
                                 (row["id"], str(candidate_id).strip())).fetchone()
            if c is None:
                raise StoreError(f"{candidate_id!r} is not registered for {row['id']}.")
            if c["status"] == "submitted":
                raise ExamStateError(f"{c['candidate_id']} has already submitted.")
            code = self._new_code()
            self._db.execute("UPDATE candidates SET code = ? WHERE exam_id = ? AND candidate_id = ?",
                             (code, row["id"], c["candidate_id"]))
            self._log(actor, "new_code", row["id"], c["candidate_id"])
            return self.show_code(code)

    def mark_candidate(self, exam_id, candidate_id, status, actor="local", note=""):
        """registered -> in_progress -> submitted. Going back to registered (a reset, so the candidate can
        sit again after a failure) is allowed from in_progress only."""
        if status not in CANDIDATE_STATUSES:
            raise StoreError(f"Unknown candidate status {status!r}.")
        allowed = {"registered": ("in_progress",), "in_progress": ("submitted", "registered"), "submitted": ()}
        with self._lock, self._transaction():
            row = self._row(exam_id)
            c = self._db.execute("SELECT * FROM candidates WHERE exam_id = ? AND lower(candidate_id) = lower(?)",
                                 (row["id"], str(candidate_id).strip())).fetchone()
            if c is None:
                raise StoreError(f"{candidate_id!r} is not registered for {row['id']}.")
            if status == c["status"] and status == "in_progress":
                return self._candidate_dict(c)             # already in progress (a rejoin)
            if status not in allowed[c["status"]]:
                raise ExamStateError(f"{c['candidate_id']} is {c['status'].replace('_', ' ')} and cannot become "
                                     f"{status.replace('_', ' ')}.")
            now = self.clock()
            if status == "in_progress":
                self._db.execute("UPDATE candidates SET status = ?, started_at = ?, finished_at = NULL WHERE exam_id = ? "
                                 "AND candidate_id = ?", (status, now, row["id"], c["candidate_id"]))
            elif status == "submitted":
                self._db.execute("UPDATE candidates SET status = ?, finished_at = ? WHERE exam_id = ? AND candidate_id = ?",
                                 (status, now, row["id"], c["candidate_id"]))
            else:
                self._db.execute("UPDATE candidates SET status = ?, started_at = NULL WHERE exam_id = ? AND candidate_id = ?",
                                 (status, row["id"], c["candidate_id"]))
            self._log(actor, {"in_progress": "candidate_started", "submitted": "candidate_submitted",
                              "registered": "candidate_reset"}[status], row["id"], f"{c['candidate_id']} {note}".strip())
            return self._candidate_dict(self._db.execute("SELECT * FROM candidates WHERE exam_id = ? AND candidate_id = ?",
                                                         (row["id"], c["candidate_id"])).fetchone())

    def export_candidates_csv(self, exam_id):
        """name, id, code, status as CSV text (for printing the slips)."""
        out = io.StringIO()
        w = csv.writer(out, lineterminator="\n")
        w.writerow(["name", "id", "code", "status"])
        for c in self.list_candidates(exam_id):
            w.writerow([c["name"], c["candidate_id"], c["code"], c["status"]])
        return out.getvalue()

    # ------------------------------- lifecycle -------------------------------
    def set_status(self, exam_id, new_status, actor="local", note=""):
        """Move an exam along its lifecycle. Raises ExamStateError when the move is not allowed
        and StoreError when the exam is not ready for it (publishing checks the exam over)."""
        if new_status not in STATUSES:
            raise StoreError(f"Unknown status {new_status!r}.")
        with self._lock, self._transaction():
            row = self._row(exam_id)
            old = row["status"]
            if new_status not in TRANSITIONS[old]:
                allowed = ", ".join(TRANSITIONS[old])
                raise ExamStateError(f"{row['id']} is {old} and can only move to: {allowed}.")
            if new_status == "published" and old == "draft":
                self._check_ready(row)
            now = self.clock()
            fields = {"status": new_status, "updated": now}
            if new_status == "published" and old == "draft":
                fields["published_at"] = now
            if new_status == "running" and row["started_at"] is None:
                fields["started_at"] = now
            if new_status == "closed" and old == "running":
                fields["closed_at"] = now
            if new_status == "published" and old == "closed":
                fields["closed_at"] = None
            self._db.execute("UPDATE exams SET " + ", ".join(f"{k} = ?" for k in fields) + " WHERE id = ?",
                             (*fields.values(), row["id"]))
            self._log(actor, MOVES[(old, new_status)], row["id"], note or f"{old} -> {new_status}")
            return self._exam(self._row(row["id"]))

    def publish(self, exam_id, actor="local", note=""):
        return self.set_status(exam_id, "published", actor, note)

    def unpublish(self, exam_id, actor="local", note=""):
        return self.set_status(exam_id, "draft", actor, note)

    def start(self, exam_id, actor="local", note=""):
        return self.set_status(exam_id, "running", actor, note)

    def close(self, exam_id, actor="local", note=""):
        return self.set_status(exam_id, "closed", actor, note)

    def release_results(self, exam_id, actor="local", note=""):
        return self.set_status(exam_id, "results", actor, note)

    def withdraw_results(self, exam_id, actor="local", note=""):
        return self.set_status(exam_id, "closed", actor, note)

    def reopen(self, exam_id, actor="local", note=""):
        return self.set_status(exam_id, "published", actor, note)

    def _check_ready(self, row):
        exam = self._exam(row)
        problems = []
        if not exam.questions:
            problems.append("it has no questions")
        if not 0 < exam.minutes <= MAX_MINUTES:
            problems.append("the time allowed is not valid")
        if problems:
            raise StoreError(f"{exam.id} cannot be published: " + " and ".join(problems) + ".")

    def use_for_exam(self, exam_id, actor="launcher"):
        """Open an exam for candidates (published -> running; already running stays running).
        Raises ExamStateError for a draft, closed or results exam."""
        exam = self.get(exam_id)
        if exam.status == "running":
            return exam
        if exam.status == "published":
            return self.start(exam.id, actor, "opened for candidates")
        raise ExamStateError(f"{exam.id} ({exam.title}) is {exam.status}, so candidates cannot sit it. "
                             + {"draft": "Publish it first.",
                                "closed": "Reopen it first.",
                                "results": "Its results are released; withdraw them and reopen it first."}[exam.status])

    # --------------------------------- audit ---------------------------------
    def audit(self, exam_id=None, limit=100):
        """The log, newest first: [{seq, time, actor, action, exam_id, detail}]."""
        with self._lock:
            if exam_id:
                rows = self._db.execute("SELECT * FROM audit WHERE exam_id = ? ORDER BY seq DESC LIMIT ?",
                                        (str(exam_id).strip().upper(), int(limit))).fetchall()
            else:
                rows = self._db.execute("SELECT * FROM audit ORDER BY seq DESC LIMIT ?", (int(limit),)).fetchall()
        return [{"seq": r["seq"], "ts": r["ts"], "time": format_time(r["ts"]), "actor": r["actor"],
                 "action": r["action"], "exam_id": r["exam_id"], "detail": r["detail"]} for r in rows]

    def log_event(self, action, exam_id=None, detail="", actor="system"):
        """Add a line to the audit log for something that happened outside the store (a sitting started, ...)."""
        with self._lock, self._transaction():
            self._log(actor, action, exam_id, detail)

    def verify_audit(self):
        """Walk the chain. Returns (True, rows_checked) or (False, 'what is wrong')."""
        with self._lock:
            rows = self._db.execute("SELECT * FROM audit ORDER BY seq").fetchall()
        prev = "0" * 64
        for r in rows:
            if r["prev"] != prev:
                return False, f"row {r['seq']} does not follow the row before it (a row was removed or changed)"
            if r["hash"] != _audit_hash(r["prev"], r["ts"], r["actor"], r["action"], r["exam_id"], r["detail"]):
                return False, f"row {r['seq']} was changed after it was written"
            prev = r["hash"]
        return True, len(rows)


class _Transaction:
    """BEGIN IMMEDIATE ... COMMIT (or ROLLBACK when the block raises)."""

    def __init__(self, store):
        self.store = store

    def __enter__(self):
        self.store._db.execute("BEGIN IMMEDIATE")

    def __exit__(self, exc_type, exc, tb):
        self.store._db.execute("ROLLBACK" if exc_type else "COMMIT")
        return False


def _audit_hash(prev, ts, actor, action, exam_id, detail):
    text = json.dumps([prev, repr(float(ts)), actor, action, exam_id, detail], ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _base(path):
    import os
    return os.path.basename(str(path))
