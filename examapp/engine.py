"""
The exam itself, with no web or camera code in it: loading questions, shuffling, keeping the
answers and the clock, scoring, and building the result record.

A question file is JSON:
    {"title": "...", "questions": [{"id": "q1", "text": "...", "options": ["a", "b", "c", "d"],
                                    "answer": 2}, ...]}
`answer` is the 0-based position of the correct option in the file. The candidate sees the
options in a shuffled order (if EXAM_SHUFFLE_OPTIONS), and the session translates back.
"""

import json
import os
import random
import secrets
import time
from dataclasses import dataclass

from facelib import config as cfg


class QuestionError(ValueError):
    """The question file is not usable."""


@dataclass
class Question:
    id: str
    text: str
    options: list
    answer: int


def load_questions(path):
    """Read and check a question file. Returns (title, [Question])."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except OSError as e:
        raise QuestionError(f"Cannot open the question file {path}: {e}")
    except ValueError as e:
        raise QuestionError(f"The question file {path} is not valid JSON: {e}")
    items = data.get("questions") if isinstance(data, dict) else None
    if not isinstance(items, list) or not items:
        raise QuestionError("The question file needs a non-empty \"questions\" list.")
    questions, seen = [], set()
    for number, item in enumerate(items, 1):
        where = f"question {number}"
        if not isinstance(item, dict):
            raise QuestionError(f"{where}: must be an object")
        qid, text, options, answer = item.get("id"), item.get("text"), item.get("options"), item.get("answer")
        if not isinstance(qid, str) or not qid.strip():
            raise QuestionError(f"{where}: needs a text \"id\"")
        if qid in seen:
            raise QuestionError(f"{where}: the id {qid!r} is used twice")
        seen.add(qid)
        if not isinstance(text, str) or not text.strip():
            raise QuestionError(f"{where} ({qid}): needs \"text\"")
        if not isinstance(options, list) or not 2 <= len(options) <= 8 or \
                not all(isinstance(o, str) and o.strip() for o in options):
            raise QuestionError(f"{where} ({qid}): \"options\" must be 2 to 8 non-empty texts")
        if len(set(options)) != len(options):
            raise QuestionError(f"{where} ({qid}): two options are identical")
        if not isinstance(answer, int) or isinstance(answer, bool) or not 0 <= answer < len(options):
            raise QuestionError(f"{where} ({qid}): \"answer\" must be the 0-based number of an option")
        questions.append(Question(qid, text.strip(), [o.strip() for o in options], answer))
    return str(data.get("title") or "Exam"), questions


class ExamSession:
    """One candidate's attempt. Thread safe enough for a single-candidate local server."""

    def __init__(self, questions, name, candidate_id, minutes=None, shuffle_questions=None,
                 shuffle_options=None, clock=time.time, defer_start=False):
        self.clock = clock
        self.questions = {q.id: q for q in questions}
        self.name, self.candidate_id = name.strip(), candidate_id.strip()
        self.minutes = cfg.EXAM_MINUTES if minutes is None else minutes
        self.token = secrets.token_urlsafe(16)
        rng = random.Random(secrets.randbits(64))
        ids = [q.id for q in questions]
        if cfg.EXAM_SHUFFLE_QUESTIONS if shuffle_questions is None else shuffle_questions:
            rng.shuffle(ids)
        self.order = ids
        shuffle_o = cfg.EXAM_SHUFFLE_OPTIONS if shuffle_options is None else shuffle_options
        self.option_order = {}   # qid -> list: shown position -> position in the file
        for q in questions:
            positions = list(range(len(q.options)))
            if shuffle_o:
                rng.shuffle(positions)
            self.option_order[q.id] = positions
        self.started = clock()
        self.deadline = self.started + self.minutes * 60.0
        self.begun = not defer_start   # with defer_start the clock starts at begin() (after calibration)
        self.calibration = None        # summary of the gaze calibration, for the result
        self.answers = {}          # qid -> position in the file
        self.marked = set()
        self.seconds_on = {q.id: 0.0 for q in questions}
        self._viewing = None       # (qid, since)
        self.events = []           # camera flags and browser events, with time and question
        self.finished = None       # the result dict once submitted
        self.finish_reason = None

    # ------------------------------- clock -------------------------------
    def begin(self):
        """Start the clock (once). Used when the exam was created with defer_start."""
        if not self.begun:
            self.begun = True
            self.started = self.clock()
            self.deadline = self.started + self.minutes * 60.0
        return self.begun

    def remaining(self):
        if not self.begun:
            return self.minutes * 60.0
        return max(0.0, self.deadline - self.clock())

    @property
    def expired(self):
        return self.begun and self.clock() >= self.deadline

    def _accepting(self):
        """Answers are taken until the deadline plus a few seconds for a slow network."""
        return (self.begun and self.finished is None
                and self.clock() < self.deadline + cfg.EXAM_SUBMIT_GRACE_SECONDS)

    # ------------------------------ actions ------------------------------
    def view(self, qid):
        """The candidate moved to question `qid`: charge the time on the previous one."""
        self._stop_viewing()
        if self.begun and self.finished is None and qid in self.questions:
            self._viewing = (qid, self.clock())

    def _stop_viewing(self):
        if self._viewing is not None:
            qid, since = self._viewing
            self.seconds_on[qid] += max(0.0, min(self.clock(), self.deadline) - since)
            self._viewing = None

    @property
    def current_number(self):
        """Question number (1-based, as the candidate sees it) being viewed, or None."""
        if self._viewing is None:
            return None
        return self.order.index(self._viewing[0]) + 1

    def answer(self, qid, shown_index):
        """Record the answer given as the position the candidate saw. None clears the answer."""
        if qid not in self.questions or not self._accepting():
            return False
        if shown_index is None:
            self.answers.pop(qid, None)
            return True
        order = self.option_order[qid]
        if not isinstance(shown_index, int) or isinstance(shown_index, bool) or not 0 <= shown_index < len(order):
            return False
        self.answers[qid] = order[shown_index]
        return True

    def mark(self, qid, on):
        if qid not in self.questions or self.finished is not None or not self.begun:
            return False
        (self.marked.add if on else self.marked.discard)(qid)
        return True

    def add_event(self, source, name, detail="", seconds=None, snapshot=None):
        """Remember a flag (source 'camera') or browser event (source 'browser') with its context.
        `snapshot` is the file name (not the path) of the picture taken when a camera flag fired."""
        event = {"time_s": round(self.clock() - self.started, 1), "question": self.current_number,
                 "source": source, "name": name, "detail": detail}
        if seconds is not None:
            event["seconds"] = round(float(seconds), 1)
        if snapshot:
            event["snapshot"] = snapshot
        self.events.append(event)
        return event

    def submit(self, reason="submitted"):
        """Finish the exam (once) and return the result. Later calls return the same result."""
        if self.finished is not None:
            return self.finished
        self._stop_viewing()
        self.finish_reason = reason
        rows, correct = [], 0
        for number, qid in enumerate(self.order, 1):
            q = self.questions[qid]
            chosen = self.answers.get(qid)
            right = chosen == q.answer
            correct += bool(right)
            rows.append({"number": number, "id": qid, "question": q.text,
                         "chosen": None if chosen is None else q.options[chosen],
                         "correct_answer": q.options[q.answer], "correct": bool(right),
                         "marked_for_review": qid in self.marked,
                         "seconds_on_question": round(self.seconds_on[qid], 1)})
        counts = {}
        for e in self.events:
            counts[e["name"]] = counts.get(e["name"], 0) + 1
        self.finished = {
            "candidate": self.name, "candidate_id": self.candidate_id,
            "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.started)),
            "finished": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.clock())),
            "minutes_allowed": self.minutes,
            "minutes_used": round(min(self.clock(), self.deadline) - self.started, 1) / 60.0,
            "finish_reason": reason,
            "score": correct, "out_of": len(self.order),
            "answered": len(self.answers),
            "questions": rows,
            "gaze_calibration": self.calibration,
            "flag_counts": counts,
            "flags": self.events,
            "note": "Flags are evidence for a human reviewer, not proof of anything.",
        }
        self.finished["minutes_used"] = round(self.finished["minutes_used"], 2)
        return self.finished

    # ----------------------------- what the page sees ---------------------
    def payload(self):
        """Everything the page needs to draw the exam. Never contains the answer key."""
        shown = []
        for number, qid in enumerate(self.order, 1):
            q = self.questions[qid]
            shown.append({"id": qid, "number": number, "text": q.text,
                          "options": [q.options[p] for p in self.option_order[qid]]})
        answers = {}
        for qid, position in self.answers.items():
            answers[qid] = self.option_order[qid].index(position)
        return {"token": self.token, "candidate": self.name, "minutes": self.minutes,
                "remaining": round(self.remaining(), 1), "questions": shown, "answers": answers,
                "marked": sorted(self.marked), "finished": self.finished is not None,
                "begun": self.begun}

    def save_result(self, directory=None):
        """Write the result to results/<id>_<time>.json and return the path."""
        directory = directory or cfg.EXAM_RESULTS_DIR
        os.makedirs(directory, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in self.candidate_id) or "candidate"
        path = os.path.join(directory, f"{safe}_{time.strftime('%Y%m%d_%H%M%S', time.localtime(self.started))}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.finished or self.submit("saved"), f, indent=1, ensure_ascii=False)
        return path
