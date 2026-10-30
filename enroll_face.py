"""
enroll_face.py - register a candidate's face for the optional identity check.

    python enroll_face.py "Candidate Name"      register (or re-register) a candidate
    python enroll_face.py --list                show who is enrolled
    python enroll_face.py --delete "Name"       remove a candidate

Then set USE_IDENTITY = True in facelib/config.py (and ID_EXPECTED_NAME = "Candidate Name"
if more than one person is enrolled) and run gaze_demo.py as usual.

What is stored: a short list of numbers per sample (a face signature), in
gallery/faces.json. No photos are saved. Get the candidate's consent first, keep the
file private, and delete it when it is no longer needed.
"""

import argparse
import time

import cv2

from facelib import config as cfg
from facelib.camera import open_camera
from facelib.identity import FaceEmbedder, FaceGallery, cosine

CONSENT = """
You are about to register a face for identity checking during an exam.

  * The program will take a few pictures of the person in front of the camera.
  * It stores only a list of numbers made from each picture (a face signature),
    in {path}, not the pictures themselves.
  * The signatures are used only to compare the live face with this registration
    during the exam. A mismatch is shown to a human reviewer, never used on its own.
  * Delete the registration when it is no longer needed:
        python enroll_face.py --delete "<name>"

The person being registered must agree to this.
"""


def list_people(gallery):
    if not gallery.people:
        print("Nobody is enrolled.")
    for p in gallery.people:
        print(f'{p["name"]}: {len(p["embeddings"])} samples, enrolled {p["created"]}')


def capture_samples(name, wanted):
    """Open the camera and collect `wanted` signatures; SPACE takes one, q finishes."""
    embedder = FaceEmbedder()
    cap = open_camera()
    samples, last_note = [], ""
    print("Look at the camera. Press SPACE to take a sample, q to finish.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Flip exactly like gaze_demo.py does, so the live and stored faces match in orientation.
        frame = cv2.flip(frame, 1)
        signature, n_faces, box = embedder.embed(frame)

        shown = frame.copy()
        if box is not None:
            x, y, w, h = box
            cv2.rectangle(shown, (x, y), (x + w, y + h), (0, 255, 0) if n_faces == 1 else (0, 0, 255), 2)
        if n_faces == 0:
            note = "No face found"
        elif n_faces > 1:
            note = "More than one face: only the candidate should be in view"
        else:
            note = f"Ready. SPACE = sample {len(samples) + 1} of {wanted}"
        if last_note:
            note += f"   ({last_note})"
        if shown.shape[1] > cfg.DISPLAY_WIDTH:
            scale = cfg.DISPLAY_WIDTH / shown.shape[1]
            shown = cv2.resize(shown, (cfg.DISPLAY_WIDTH, int(shown.shape[0] * scale)))
        cv2.putText(shown, f"Enrolling: {name}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(shown, note, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Enroll face (SPACE = sample, q = finish)", shown)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == 32:  # SPACE
            if signature is None or n_faces != 1:
                last_note = "not saved: need exactly one face"
            else:
                samples.append(signature)
                last_note = f"saved {len(samples)}"
                print(f"Sample {len(samples)} saved.")
                time.sleep(0.4)  # a short pause so the next sample is a different moment
            if len(samples) >= wanted:
                break

    cap.release()
    cv2.destroyAllWindows()
    return samples


def main():
    parser = argparse.ArgumentParser(description="Register a candidate's face for the identity check.")
    parser.add_argument("name", nargs="?", help="candidate name to enrol")
    parser.add_argument("--list", action="store_true", help="show who is enrolled")
    parser.add_argument("--delete", metavar="NAME", help="remove a candidate")
    parser.add_argument("--samples", type=int, default=5, help="how many pictures to take (default 5)")
    args = parser.parse_args()

    gallery = FaceGallery()
    if args.list:
        return list_people(gallery)
    if args.delete:
        print("Removed." if gallery.remove(args.delete) else f'"{args.delete}" is not enrolled.')
        return
    if not args.name:
        parser.error('give a name, e.g.  python enroll_face.py "Candidate Name"')

    print(CONSENT.format(path=cfg.GALLERY_FILE))
    if input("Type YES to confirm the person agrees: ").strip().upper() != "YES":
        raise SystemExit("Cancelled. Nothing was stored.")

    samples = capture_samples(args.name, args.samples)
    if not samples:
        raise SystemExit("No samples taken. Nothing was stored.")
    if len(samples) >= 2:
        lowest = min(cosine(a, b) for i, a in enumerate(samples) for b in samples[i + 1:])
        if lowest < cfg.ID_MATCH_THRESHOLD:
            print(f"Warning: two samples differ a lot (similarity {lowest:.2f}). "
                  "Make sure only one person was in view, then enrol again.")
    gallery.enroll(args.name, samples)
    print(f'Enrolled "{args.name}" with {len(samples)} samples.')


if __name__ == "__main__":
    main()
