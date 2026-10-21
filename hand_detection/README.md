# Hand detection — week 1

Detects hands in photos with º and draws the 21-point hand skeleton, a bounding box and Left/Right.
First building block for the hand-gesture part of the IPCV project.

---

## Setup

```bash
# 1. Python environment (outside the Obsidian vault, ~200 MB)
uv venv ~/.venvs/ipcv --python 3.14
uv pip install --python ~/.venvs/ipcv/bin/python -r requirements.txt

# 2. Model (7.8 MB) — float16 hand landmarker
curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
sha256sum models/hand_landmarker.task
# expected (downloaded 6 Oct 2026):
# fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1
```

Tested with: Ubuntu 26.04, Python 3.14.4, mediapipe 1.0.1, opencv-contrib-python 4.x (installed by mediapipe).

> [!warning] MediaPipe 1.x
> `mp.solutions.hands` **no longer exists**. Most online tutorials use it and will fail.
> Use the Tasks API: `from mediapipe.tasks.python import vision` → `vision.HandLandmarker`.

---

## Run

```bash
~/.venvs/ipcv/bin/python detect_hands.py samples/                # all test photos
~/.venvs/ipcv/bin/python detect_hands.py my_photo.jpg            # one photo
~/.venvs/ipcv/bin/python detect_hands.py samples/ --pad 0.5      # close-up photos (see results)
```

| Option | Default | Meaning |
|---|---|---|
| `--num-hands` | 4 | max hands per image (2 players × 2 hands) |
| `--min-conf` | 0.5 | detection and presence threshold |
| `--pad` | 0 | grey border, as a fraction of image size, around the photo before detection |
| `--out` | `output/` | where results go |

**Output** in `output/`: `<name>_hands.jpg` (annotated), `results.json` (21 landmarks per hand, handedness, score, bbox, time), `_overview.jpg` (all results on one sheet).

---

## Files

```
hand_detection/
├── detect_hands.py      detection script
├── requirements.txt
├── Hand detection - how it works.md   how the model works (for the individual questions)
├── docs/                make_figures.py + img/ (figures used in that note)
├── models/              hand_landmarker.task (download, see Setup)
├── samples/             test photos + SOURCES.md (licences, authors)
└── output/              results of the last run
```

---

## Results on the 14 test photos (6 Oct 2026)

Laptop CPU, default settings (`--min-conf 0.5`, no padding).

| Photo | Hands visible | Found | Note |
|---|---|---|---|
| 01 fist, plain background | 1 | **0** | close-up: found with `--pad 0.5` |
| 02 two open palms | 2 | 1 | adjacent palms merged into one detection |
| 03 waving (webcam-like) | 1 | 1 | |
| 04 arms raised, dark stage | 2 | 2 | |
| 05 high-five, two people | 3 (two touching) | 2 | the two touching hands give one detection |
| 06 handshake | 2 (interlocked) | **0** | occlusion |
| 07 gesture, low light | 1 | 1 | |
| 08 hand holding a sign | 1 | 1 | low handedness score (0.55) |
| 09 two small fists | 2 | 2 | |
| 10 V sign, black and white | 1 | **0** | low contrast, no colour |
| 11 hand on hand, close-up | 2 | **0** | found 1 with `--pad 0.5` |
| 12 crowd | many | 2 | only the larger hands |
| 13 squirrel (negative) | 0 | 0 | no false positive |
| 14 train (negative) | 0 | 0 | no false positive |

- **Recall** on photos 01–11: **10 of 18** visible hands found. **Every isolated, medium-size hand was found (8 / 8: photos 03, 04, 07, 08, 09 and the free hand in 05).** Misses are touching or overlapping hands (02, 05, 06, 11), close-ups (01, 11) and the low-contrast black-and-white photo (10).
- **Precision**: all 12 detections are real hands; no false positives on the two negatives.
- **Time**: about 10 ms with no hand, about 18 ms with one hand, about 27 ms with two. The palm detector costs about 10 ms and each hand found adds about 8 ms.
- **`--pad 0.5`** recovers close-ups (01, 11) but loses small hands (05: 2 → 0, 09 and 12: 2 → 1). There is no single scale that works for everything.
- **For the game**: keep players far enough from the camera that hands look medium-size, and avoid gestures where hands touch or overlap.

---

## Known issues

- **Handedness on photos** — on normal (non-mirrored) photos the label matched the person's real hand (photos 04 and 09). If you flip the webcam image for a mirror view **before** detection, Left/Right swap. Check this with the webcam.
- **Audio** — `import mediapipe` also loads its audio module. Without a sound server (SSH, Docker) that crashes; the script replaces it with a dummy module.
- **Colour order** — OpenCV loads BGR and MediaPipe expects RGB: convert with `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`.
