# face-recognition-app

A small face recognition demo in ~200 lines of Python. Drop photos of people
into a folder, run the script, and it identifies them live from your webcam or
in a still image.

No training step. No GPU. Runs on a laptop CPU at usable frame rates.

---

## Contents

- [How it works](#how-it-works)
- [Step 1 — Prerequisites](#step-1--prerequisites)
- [Step 2 — Get the code](#step-2--get-the-code)
- [Step 3 — Create the virtual environment](#step-3--create-the-virtual-environment)
- [Step 4 — Install dependencies](#step-4--install-dependencies)
- [Step 5 — Verify the install](#step-5--verify-the-install)
- [Step 6 — Add faces to enroll](#step-6--add-faces-to-enroll)
- [Step 7 — Run it](#step-7--run-it)
- [Step 8 — Stop it](#step-8--stop-it)
- [Step 9 — Tune the threshold](#step-9--tune-the-threshold)
- [Troubleshooting](#troubleshooting)
- [Publishing your own copy](#publishing-your-own-copy)
- [Limitations](#limitations)

---

## How it works

Every face recognition system is the same three stages:

1. **Detection** — find the bounding boxes of faces in the frame. This stage
   knows nothing about identity.
2. **Embedding** — crop and align each face, then run it through a network
   trained so photos of the same person land close together in vector space and
   different people land far apart. dlib's model outputs 128 dimensions.
3. **Matching** — nearest-neighbour search against your enrolled embeddings,
   with a distance threshold deciding "this person" vs "Unknown".

Enrollment is one forward pass per photo, so adding a person means storing one
more vector. There is no retraining, which is why this feels instant.

---

## Step 1 — Prerequisites

**Python 3.12 specifically.** Python 3.13 and 3.14 have no prebuilt `dlib`
wheel, and building dlib from source is a slow, failure-prone detour.

Check what you have:

```bash
python3 --version
```

If it isn't 3.12, install it (macOS, Homebrew):

```bash
brew install python@3.12
```

Linux: `sudo apt install python3.12 python3.12-venv`
Windows: download 3.12 from python.org and tick "Add to PATH".

You also need **git**, and a **webcam** if you want the live mode.

---

## Step 2 — Get the code

```bash
git clone https://github.com/pradeepbaliga/face-recognition-app.git
cd face-recognition-app
```

---

## Step 3 — Create the virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python --version                   # must print 3.12.x
```

Your prompt should now show `(.venv)`. **This does not persist** — every new
terminal window needs the `source` line again before `python` will work.

Optional macOS shortcut, so you never think about it again:

```bash
echo 'faceapp() { cd ~/face-recognition-app && source .venv/bin/activate; }' >> ~/.zshrc
source ~/.zshrc
```

Then just type `faceapp` in any new terminal.

---

## Step 4 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt --no-deps
```

**`--no-deps` is not optional.** Without it, pip tries to satisfy
`face_recognition`'s `dlib>=19.7` requirement by compiling dlib from source,
which is exactly what `dlib-bin` exists to avoid.

Two pins in `requirements.txt` are worth understanding:

| Pin | Why |
|-----|-----|
| `dlib-bin` instead of `dlib` | Same library, shipped as a prebuilt wheel. Nothing compiles, no cmake or C++ toolchain needed. |
| `setuptools<81` | `face_recognition_models` hasn't been updated since 2017 and imports `pkg_resources` to locate its bundled `.dat` files. setuptools 81 deprecated that module; 84 removed it. |

Do **not** run `pip install --upgrade setuptools` in this environment — it will
pull setuptools 84+ and break the model loader.

---

## Step 5 — Verify the install

```bash
python -c "import dlib, face_recognition, cv2; print('ok')"
```

If this prints `ok`, you're done installing. If it prints anything else, go to
[Troubleshooting](#troubleshooting) — the error messages in this stack are
unusually misleading.

---

## Step 6 — Add faces to enroll

Put photos in `known_faces/`. The filename becomes the label shown on screen.

```
known_faces/
    alice.jpg          -> labelled "alice"
    bob.png            -> labelled "bob"
    carol/             -> or a folder per person, for multiple photos
        carol_1.jpg
        carol_2.jpg
```

What works: clear, front-facing photos where the face is at least a couple
hundred pixels wide. If a photo has several faces, only the largest is used.

Enroll **two or three photos per person** under different lighting and angles.
A single embedding from one flattering photo generalizes badly.

`known_faces/` is gitignored — see [Limitations](#limitations) for why you
shouldn't commit face photos.

---

## Step 7 — Run it

Live webcam:

```bash
python face_app.py
```

Single image:

```bash
python face_app.py test.jpg
```

Other options:

```bash
python face_app.py --threshold 0.45     # stricter matching
python face_app.py --db other_faces     # different enrollment folder
python face_app.py --camera 1           # external camera
```

**First run on macOS** triggers a camera permission prompt. If you never see it
and just get a black window, go to System Settings → Privacy & Security →
Camera, enable it for Terminal or iTerm, then fully quit and reopen the
terminal — toggling the setting doesn't affect an already-running process.

The OpenCV window often opens *behind* the terminal. Check your window list
before assuming it crashed.

Testing tip: don't point it at a file that's already in `known_faces/`. That
compares a photo against itself, reports distance 0.00, and proves nothing. Use
a different photo of the same person, or just use the webcam.

---

## Step 8 — Stop it

Three ways, all clean:

- Click the **QUIT** button in the top-left of the video window
- Press **`q`** — the video window must have focus, not the terminal
- Close the window with its normal close control

Last resort, from the terminal: **Ctrl+C**. This skips camera cleanup, so if the
camera light stays on afterwards, close the terminal window entirely. Or:

```bash
pkill -f face_app.py
```

---

## Step 9 — Tune the threshold

The distance threshold is the whole usability/security tradeoff. dlib's default
is 0.6; this repo defaults to 0.5.

| Threshold | Behaviour |
|-----------|-----------|
| 0.6 | Permissive. Will match people who merely look similar. |
| 0.5 | Balanced default. |
| 0.45 | Strict. Prefers "Unknown" over a wrong name. |

Watch the number printed next to each face on screen — that's the actual
distance. If you consistently read as "Unknown" at 0.52, loosen it. If a
stranger matches you at 0.48, tighten it.

Performance knobs at the top of `face_app.py`: `SCALE` (detection downscale —
lower is faster but misses small faces) and the every-other-frame check inside
`run_webcam`.

---

## Troubleshooting

**`Please install face_recognition_models with this command...`**
Almost always a lie — the package *is* installed. `face_recognition` catches
every exception and prints this same message regardless of the real cause. Get
the actual error:

```bash
python -c "import face_recognition_models"
```

If it says `No module named 'pkg_resources'`, you have setuptools 81+:

```bash
pip install "setuptools<81"
```

**`Failed building wheel for dlib`**
You're building from source. Install the prebuilt wheel instead:

```bash
pip install dlib-bin
pip install face_recognition --no-deps
```

**`No matching distribution found for dlib-bin`**
Wrong Python version — no wheel exists for 3.13/3.14. Check with
`python -V` and rebuild the venv on 3.12 (Step 3).

**`zsh: command not found: python`**
The venv isn't active. Run `source .venv/bin/activate`. Your prompt should show
`(.venv)`.

**`No matching distribution found for pkg_resources`**
Expected — it isn't a standalone package. It ships inside setuptools. Use the
`setuptools<81` pin instead.

**`can't open file 'face_app.py'`**
You're in the wrong directory, or the file was never moved there. `ls` to check.

**`No face found in <file>, skipping` during enrollment**
The detector couldn't find a face. Try a larger, better-lit, front-facing photo.

**Stale enrollment results**
Embeddings are cached in `known_faces/.encodings.pkl` and rebuilt when the
folder's timestamp changes. If you edit a photo *in place* rather than adding
one, delete the pickle to force a re-enroll.

---

## Publishing your own copy

```bash
git init
git add .
git commit -m "Simple face recognition app"
```

Create an empty repo on github.com — no README, no .gitignore, no license,
since this repo already has all three. Then:

```bash
git remote add origin https://github.com/<you>/face-recognition-app.git
git branch -M main
git push -u origin main
```

Authenticate with a **fine-grained personal access token** (Settings →
Developer settings → Personal access tokens), scoped to this repo with
Contents: Read and write. At the push prompt, enter your username, then paste
the token as the password — GitHub account passwords haven't worked for git
since 2021.

Store it in the keychain so you only type it once:

```bash
git config --global credential.helper osxkeychain
```

Don't embed the token in the remote URL (`https://TOKEN@github.com/...`) — that
writes it in plaintext into `.git/config` and leaks it through shell history and
`git remote -v`.

Run `git status` before your first push and confirm no face photos are staged.

---

## Limitations

**This is recognition, not authentication.** It will happily match a printed
photo held up to the camera. Anything gating real access needs a liveness check.

**Detector limits.** HOG struggles with profile views, occlusion, and small or
poorly lit faces. Setting `model="cnn"` in `face_app.py` is more accurate and
much slower on CPU. For meaningfully better accuracy, InsightFace's `buffalo_l`
on onnxruntime beats dlib at comparable speed.

**Face embeddings are biometric data** under Illinois BIPA and the GDPR.
Storing vectors rather than raw images helps, but consent obligations attach to
the vectors too. `known_faces/` is gitignored for this reason — don't commit
photos of people, including your own, to a public repo.

For test data that's actually licensed for it, use
[Labeled Faces in the Wild](https://vis-www.cs.umass.edu/lfw/), which has
multiple photos per person — necessary for testing real matching rather than
self-matching.

---

## License

MIT — see [LICENSE](LICENSE).
