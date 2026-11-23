"""
exams.py - look after the exam records (the list of exams, their questions, rules and status).

    python exams.py list                          all exams
    python exams.py show EX-7K3QMD                one exam with its questions (add --key to show the answers)
    python exams.py import questions.json         make a draft from a question file (--title, --minutes)
    python exams.py set EX-7K3QMD --minutes 45 --title "Midterm" --rule show_score=false     (drafts only)
    python exams.py publish EX-7K3QMD             finish the draft: it is locked and ready
    python exams.py unpublish EX-7K3QMD           back to draft (to change it)
    python exams.py start EX-7K3QMD               open it for candidates (python exam_app.py --exam EX-7K3QMD does this too)
    python exams.py close EX-7K3QMD               no more candidates
    python exams.py release EX-7K3QMD             release the results      (withdraw: take them back)
    python exams.py reopen EX-7K3QMD              closed -> published again
    python exams.py duplicate EX-7K3QMD           a new draft copy, for a re-sit or a new version
    python exams.py delete EX-7K3QMD              delete a draft that was never published
    python exams.py audit [EX-7K3QMD] [--verify]  who did what and when; --verify checks the log was not altered

  Registered candidates (each gets a personal access code; the code is all they type to join):
    python exams.py candidates EX-7K3QMD import class.csv   a CSV with a header (name, id) or a JSON list
    python exams.py candidates EX-7K3QMD add "Ada Lovelace" --id A123
    python exams.py candidates EX-7K3QMD list               name, id, code, status
    python exams.py candidates EX-7K3QMD export slips.csv   the codes as a CSV file for printing (or to the screen)
    python exams.py candidates EX-7K3QMD newcode A123       a new code (the old one stops working)
    python exams.py candidates EX-7K3QMD reset A123         let a candidate who is in progress sit again (e.g. a crash)
    python exams.py candidates EX-7K3QMD remove A123        take out a candidate who has not started

Add --db FILE to use another exam records file (default: exams.db next to the program).
Status: draft -> published -> running -> closed -> results. Only a draft can be changed.
"""

import argparse
import sys

from facelib import config as cfg
from examapp.store import (ExamNotFound, ExamStateError, ExamStore, RULE_NAMES, STATUSES, StoreError,
                           format_time)

ACTOR = "exams.py"


def parse_rule(text):
    if "=" not in text:
        raise StoreError(f"A rule looks like name=true or name=false (or name=open), not {text!r}.")
    name, value = (x.strip() for x in text.split("=", 1))
    value = value.lower()
    if value in ("true", "yes", "on", "1"):
        return name, True
    if value in ("false", "no", "off", "0"):
        return name, False
    if value in ("open", "none", "default"):
        return name, None
    raise StoreError(f"The rule {name!r} must be true, false or open, not {value!r}.")


def show_table(rows):
    if not rows:
        print("No exams yet. Import one with: python exams.py import questions.json")
        return
    print(f"{'ID':<10} {'Status':<10} {'Qs':>3} {'Min':>5}  {'Updated':<19}  Title")
    for r in rows:
        print(f"{r['id']:<10} {r['status']:<10} {r['questions']:>3} {r['minutes']:>5g}  {format_time(r['updated']):<19}  {r['title']}")


def show_exam(exam, key):
    print(f"{exam.id}  {exam.title}")
    print(f"  status: {exam.status}    time allowed: {exam.minutes:g} minutes    questions: {len(exam.questions)}")
    if exam.description:
        print(f"  {exam.description}")
    print("  rules (open = the program's setting):")
    for name in sorted(RULE_NAMES):
        value = exam.rules.get(name)
        print(f"    {name:<18} {'open (' + str(getattr(cfg, RULE_NAMES[name])).lower() + ')' if value is None else str(value).lower()}")
    for label, when in (("created", exam.created), ("published", exam.published_at),
                        ("started", exam.started_at), ("closed", exam.closed_at)):
        if when:
            print(f"  {label}: {format_time(when)}")
    if exam.source:
        print(f"  source: {exam.source}")
    for n, q in enumerate(exam.questions, 1):
        print(f"\n  {n}. [{q.id}] {q.text}")
        for i, option in enumerate(q.options):
            mark = "*" if key and i == q.answer else " "
            print(f"     {mark} {chr(65 + i)}. {option}")
    if not key and exam.questions:
        print("\n  (answers hidden: add --key to show them)")


def candidates_command(store, args, out):
    a, arg = args.action, args.arg
    if a in ("import", "add", "newcode", "reset", "remove") and not arg:
        raise StoreError({"import": "Give the candidate list file: candidates EX-... import class.csv",
                          "add": "Give the name: candidates EX-... add \"Ada Lovelace\" --id A123"}.get(
                              a, "Give the candidate id."))
    if a == "import":
        made = store.import_candidates_file(args.id, arg, ACTOR)
        out(f"Registered {len(made)} candidates. Print the codes with: python exams.py candidates {args.id.upper()} export slips.csv")
    elif a == "add":
        c = store.add_candidates(args.id, [{"name": arg, "candidate_id": args.candidate_id}], ACTOR)[0]
        out(f"Registered {c['name']} ({c['candidate_id']}): access code {c['code']}")
    elif a == "list":
        rows = store.list_candidates(args.id)
        if not rows:
            out("No candidates registered yet.")
        for c in rows:
            out(f"{c['candidate_id']:<12} {c['name']:<30} {c['code']}  {c['status'].replace('_', ' ')}")
    elif a == "export":
        text = store.export_candidates_csv(args.id)
        if arg:
            with open(arg, "w", encoding="utf-8", newline="") as f:
                f.write(text)
            out(f"Wrote {arg}. It holds the access codes: keep it private and hand each candidate only their own.")
        else:
            out(text.rstrip("\n"))
    elif a == "newcode":
        out(f"New access code for {arg}: {store.new_code(args.id, arg, ACTOR)} (the old one no longer works)")
    elif a == "reset":
        store.mark_candidate(args.id, arg, "registered", ACTOR, "reset by the invigilator")
        out(f"{arg} can join again. Their earlier answers on the server are discarded when they join.")
    elif a == "remove":
        store.remove_candidate(args.id, arg, ACTOR)
        out(f"{arg} removed.")


def build_parser():
    p = argparse.ArgumentParser(prog="exams.py", description="Look after the exam records.")
    p.add_argument("--db", metavar="FILE", help="exam records file (default: exams.db)")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list the exams").add_argument("--status", choices=STATUSES)
    show = sub.add_parser("show", help="show one exam")
    show.add_argument("id")
    show.add_argument("--key", action="store_true", help="also show the correct answers")
    imp = sub.add_parser("import", help="make a draft from a JSON question file")
    imp.add_argument("file")
    imp.add_argument("--title")
    imp.add_argument("--minutes", type=float)
    st = sub.add_parser("set", help="change a draft's title, time or rules")
    st.add_argument("id")
    st.add_argument("--title")
    st.add_argument("--minutes", type=float)
    st.add_argument("--description")
    st.add_argument("--rule", action="append", default=[], metavar="NAME=true|false|open")
    for name, text in (("publish", "lock a finished draft"), ("unpublish", "back to draft"),
                       ("start", "open for candidates"), ("close", "no more candidates"),
                       ("release", "release the results"), ("withdraw", "take the results back"),
                       ("reopen", "closed -> published"), ("duplicate", "make a draft copy"),
                       ("delete", "delete a draft that was never published")):
        sub.add_parser(name, help=text).add_argument("id")
    ca = sub.add_parser("candidates", help="registered candidates and their access codes")
    ca.add_argument("id")
    ca.add_argument("action", choices=("import", "add", "list", "export", "newcode", "reset", "remove"))
    ca.add_argument("arg", nargs="?", help="a file (import, export), a name (add) or a candidate id (newcode, reset, remove)")
    ca.add_argument("--id", dest="candidate_id", help="the candidate's id (with add)")
    au = sub.add_parser("audit", help="the audit log")
    au.add_argument("id", nargs="?")
    au.add_argument("--limit", type=int, default=40)
    au.add_argument("--verify", action="store_true", help="check the log chain was not altered")
    return p


def run(args, out=print):
    store = ExamStore(args.db or cfg.EXAM_DB_FILE)
    try:
        c = args.command
        if c == "list":
            show_table(store.list_exams(args.status))
        elif c == "show":
            show_exam(store.get(args.id), args.key)
        elif c == "import":
            exam, created = store.import_file(args.file, ACTOR, args.title, args.minutes)
            out(f"{'Imported' if created else 'Already imported'} as {exam.id} ({exam.status}): "
                f"{exam.title}, {len(exam.questions)} questions, {exam.minutes:g} minutes.")
        elif c == "set":
            rules = dict(parse_rule(r) for r in args.rule) if args.rule else None
            exam = store.update_exam(args.id, args.title, args.minutes, rules, args.description, ACTOR)
            out(f"{exam.id} updated.")
        elif c in ("publish", "unpublish", "start", "close", "release", "withdraw", "reopen"):
            method = {"release": "release_results", "withdraw": "withdraw_results"}.get(c, c)
            exam = getattr(store, method)(args.id, ACTOR)
            out(f"{exam.id} is now {exam.status}.")
        elif c == "duplicate":
            exam = store.duplicate(args.id, ACTOR)
            out(f"Copied to {exam.id} (draft): {exam.title}")
        elif c == "delete":
            store.delete_exam(args.id, ACTOR)
            out(f"{args.id.upper()} deleted.")
        elif c == "candidates":
            candidates_command(store, args, out)
        elif c == "audit":
            if args.verify:
                ok, info = store.verify_audit()
                out(f"Audit log OK: {info} rows, each follows the one before." if ok else f"AUDIT LOG PROBLEM: {info}")
                if not ok:
                    return 2
            for r in store.audit(args.id, args.limit):
                out(f"{r['time']}  {r['actor']:<10} {r['action']:<18} {r['exam_id'] or '-':<10} {r['detail']}")
        return 0
    except (ExamNotFound, ExamStateError, StoreError) as e:
        print(f"Cannot do that: {e}")
        return 1
    finally:
        store.disconnect()


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
