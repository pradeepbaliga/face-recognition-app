"""
Simple face recognition app: enroll known faces from a folder, then recognize
them in a webcam stream or a single image.

Install
-------
    pip install face_recognition opencv-python numpy

    face_recognition compiles dlib, so you need cmake + a C++ toolchain:
        macOS:  brew install cmake
        Ubuntu: sudo apt install cmake build-essential

Folder layout
-------------
    known_faces/
        pradeep.jpg          <- one photo per person, filename = label
        alice.png
        bob/                 <- or a folder per person with several photos
            bob_1.jpg
            bob_2.jpg

Run
---
    python face_app.py                      # webcam
    python face_app.py photo.jpg            # single image
    python face_app.py --db other_faces     # different enrollment folder
"""

import argparse
import pickle
import sys
from pathlib import Path

import cv2
import face_recognition
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Distance below which two faces are considered the same person.
# 0.6 is the dlib default; lower = stricter = fewer false matches.
MATCH_THRESHOLD = 0.5

# Downscale factor for detection. Smaller = faster, but misses small faces.
SCALE = 0.25


# --------------------------------------------------------------------------
# 1. Enrollment: turn a folder of photos into a list of 128-d face embeddings
# --------------------------------------------------------------------------

def _encode_image(path: Path) -> np.ndarray | None:
    """Return the embedding of the largest face in an image, or None."""
    image = face_recognition.load_image_file(path)
    boxes = face_recognition.face_locations(image, model="hog")
    if not boxes:
        print(f"  ! no face found in {path.name}, skipping")
        return None
    # If the photo has several faces, keep the biggest one.
    boxes.sort(key=lambda b: (b[2] - b[0]) * (b[1] - b[3]), reverse=True)
    return face_recognition.face_encodings(image, [boxes[0]])[0]


def build_database(db_dir: Path, cache: Path | None = None):
    """Load (or build and cache) the known-face embeddings."""
    if cache and cache.exists() and cache.stat().st_mtime > db_dir.stat().st_mtime:
        with cache.open("rb") as fh:
            encodings, names = pickle.load(fh)
        print(f"Loaded {len(names)} enrolled faces from cache.")
        return encodings, names

    encodings, names = [], []
    print(f"Enrolling faces from {db_dir}/ ...")

    for entry in sorted(db_dir.iterdir()):
        if entry.is_dir():
            label = entry.name
            photos = [p for p in sorted(entry.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
        elif entry.suffix.lower() in IMAGE_EXTS:
            label = entry.stem
            photos = [entry]
        else:
            continue

        for photo in photos:
            vec = _encode_image(photo)
            if vec is not None:
                encodings.append(vec)
                names.append(label)
                print(f"  + {label}  ({photo.name})")

    if not encodings:
        sys.exit(f"No usable faces found in {db_dir}/. Add some photos first.")

    if cache:
        with cache.open("wb") as fh:
            pickle.dump((encodings, names), fh)

    print(f"Enrolled {len(encodings)} images across {len(set(names))} people.\n")
    return encodings, names


# --------------------------------------------------------------------------
# 2. Matching: nearest neighbour in embedding space
# --------------------------------------------------------------------------

def identify(query: np.ndarray, encodings, names, threshold=MATCH_THRESHOLD):
    """Return (name, distance) for the closest enrolled face."""
    distances = face_recognition.face_distance(encodings, query)
    best = int(np.argmin(distances))
    if distances[best] <= threshold:
        return names[best], float(distances[best])
    return "Unknown", float(distances[best])


def annotate(frame, boxes, labels, scale=1.0):
    """Draw boxes and labels back onto the full-resolution frame."""
    for (top, right, bottom, left), (name, dist) in zip(boxes, labels):
        top, right = int(top / scale), int(right / scale)
        bottom, left = int(bottom / scale), int(left / scale)
        color = (0, 180, 0) if name != "Unknown" else (0, 0, 220)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 24), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, f"{name} {dist:.2f}", (left + 6, bottom - 7),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1)
    return frame


# --------------------------------------------------------------------------
# 3. On-screen quit button (OpenCV has no widgets, so we draw one and
#    listen for clicks inside its rectangle)
# --------------------------------------------------------------------------

WINDOW = "face recognition"
BUTTON = (12, 12, 108, 42)          # x, y, width, height


def _in_button(x, y, rect=BUTTON):
    bx, by, bw, bh = rect
    return bx <= x <= bx + bw and by <= y <= by + bh


def draw_quit_button(frame, hover=False):
    bx, by, bw, bh = BUTTON
    fill = (40, 40, 210) if hover else (55, 55, 55)
    overlay = frame.copy()
    cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), fill, cv2.FILLED)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (235, 235, 235), 1)
    cv2.putText(frame, "QUIT", (bx + 21, by + 28),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
    return frame


def _on_mouse(event, x, y, flags, state):
    """Track hover state and set the quit flag on a click inside the button."""
    state["hover"] = _in_button(x, y)
    if event == cv2.EVENT_LBUTTONDOWN and state["hover"]:
        state["quit"] = True


# --------------------------------------------------------------------------
# 4. Entry points
# --------------------------------------------------------------------------

def run_image(path: Path, encodings, names, threshold=MATCH_THRESHOLD):
    image = face_recognition.load_image_file(path)
    boxes = face_recognition.face_locations(image, model="hog")
    vectors = face_recognition.face_encodings(image, boxes)
    labels = [identify(v, encodings, names, threshold) for v in vectors]

    for (name, dist) in labels:
        print(f"{name}  (distance {dist:.3f})")

    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("result", annotate(frame, boxes, labels))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_webcam(encodings, names, camera=0, threshold=MATCH_THRESHOLD):
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        sys.exit("Could not open the camera.")

    state = {"quit": False, "hover": False}
    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, _on_mouse, state)

    print("Click QUIT in the window, press q, or close the window to stop.")
    boxes, labels = [], []
    frame_no = 0

    while not state["quit"]:
        ok, frame = cap.read()
        if not ok:
            break

        # Only run the expensive part every other frame.
        if frame_no % 2 == 0:
            small = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb, model="hog")
            vectors = face_recognition.face_encodings(rgb, boxes)
            labels = [identify(v, encodings, names, threshold) for v in vectors]

        frame = annotate(frame, boxes, labels, SCALE)
        cv2.imshow(WINDOW, draw_quit_button(frame, state["hover"]))
        frame_no += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Also stop if the user closed the window with its own close control.
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Simple face recognition demo")
    ap.add_argument("image", nargs="?", help="image file; omit to use the webcam")
    ap.add_argument("--db", default="known_faces", help="folder of enrolled faces")
    ap.add_argument("--camera", type=int, default=0, help="camera index")
    ap.add_argument("--threshold", type=float, default=MATCH_THRESHOLD)
    args = ap.parse_args()

    db_dir = Path(args.db)
    if not db_dir.is_dir():
        sys.exit(f"Enrollment folder {db_dir}/ does not exist.")

    encodings, names = build_database(db_dir, cache=db_dir / ".encodings.pkl")

    if args.image:
        run_image(Path(args.image), encodings, names, args.threshold)
    else:
        run_webcam(encodings, names, args.camera, args.threshold)


if __name__ == "__main__":
    main()
