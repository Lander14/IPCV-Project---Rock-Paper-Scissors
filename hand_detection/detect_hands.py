"""Detect hands in photos with MediaPipe Hand Landmarker (MediaPipe >= 1.0).

For each image it finds every hand (up to --num-hands), and outputs:
  - an annotated copy in output/  (skeleton of the 21 landmarks, bounding box, Left/Right + score)
  - output/results.json           (all landmarks per hand, for later gesture work)
  - a summary table in the terminal (hands found, handedness, score, time per image)
  - output/_overview.jpg          (all annotated images on one sheet, when the input is a folder)

Usage:
    python detect_hands.py samples/                 # every image in a folder
    python detect_hands.py samples/01_fist_plain.jpg
    python detect_hands.py samples/ --num-hands 2 --min-conf 0.6
    python detect_hands.py samples/ --pad 0.5      # close-up photos: add a border so the hand looks smaller
"""
import argparse
import json
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np

# MediaPipe imports its audio module at start-up, which needs a running sound server.
# We only need vision, so if audio is unavailable (SSH, Docker, demo laptop) use a dummy module.
try:
    import sounddevice  # noqa: F401
except Exception:
    sys.modules["sounddevice"] = types.ModuleType("sounddevice")

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

HERE = Path(__file__).resolve().parent
CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS
COLORS = {"Left": (255, 140, 0), "Right": (0, 200, 255)}  # BGR: blue-ish / orange-ish
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def make_detector(model_path: Path, num_hands: int, min_conf: float) -> vision.HandLandmarker:
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,    # single photos; use VIDEO / LIVE_STREAM for webcam
        num_hands=num_hands,
        min_hand_detection_confidence=min_conf,   # palm detector threshold
        min_hand_presence_confidence=min_conf,    # landmark model: "is there really a hand?"
    )
    return vision.HandLandmarker.create_from_options(options)


def detect(detector: vision.HandLandmarker, bgr: np.ndarray, pad: float = 0.0):
    """Return (hands, ms). pad > 0 adds a grey border of pad*size around the image first ("zoom out"):
    the palm detector works on a downscaled 192x192 input and misses hands that fill the whole photo."""
    b = int(pad * max(bgr.shape[:2]))
    if b:
        bgr = cv2.copyMakeBorder(bgr, b, b, b, b, cv2.BORDER_CONSTANT, value=(128, 128, 128))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)    # OpenCV loads BGR, MediaPipe expects RGB
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    t0 = time.perf_counter()
    result = detector.detect(image)
    ms = (time.perf_counter() - t0) * 1000
    hands = []
    h, w = bgr.shape[:2]
    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        pts = np.array([[lm.x * w - b, lm.y * h - b] for lm in landmarks])  # normalized -> pixels (original image)
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        hands.append({
            "handedness": handedness[0].category_name,                      # "Left" / "Right"
            "score": round(handedness[0].score, 3),
            "bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
            "landmarks_px": pts.round(1).tolist(),                           # 21 x (x, y)
            "landmarks_z": [round(lm.z, 4) for lm in landmarks],             # relative depth (wrist = 0)
        })
    return hands, ms


def draw(bgr: np.ndarray, hands: list) -> np.ndarray:
    out = bgr.copy()
    thick = max(2, round(max(out.shape[:2]) / 400))
    for hand in hands:
        color = COLORS.get(hand["handedness"], (0, 255, 0))
        pts = [tuple(int(v) for v in p) for p in hand["landmarks_px"]]
        for c in CONNECTIONS:
            cv2.line(out, pts[c.start], pts[c.end], color, thick, cv2.LINE_AA)
        for i, p in enumerate(pts):
            cv2.circle(out, p, thick + 2, (255, 255, 255) if i else (0, 0, 255), -1, cv2.LINE_AA)  # wrist = red
        x0, y0, x1, y1 = hand["bbox_xyxy"]
        m = int(0.08 * max(x1 - x0, y1 - y0))                                # small margin around the hand
        cv2.rectangle(out, (x0 - m, y0 - m), (x1 + m, y1 + m), color, thick)
        label = f"{hand['handedness']} {hand['score']:.2f}"
        scale = 0.6 * thick / 2 + 0.3
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        tx = int(np.clip(x0 - m, 0, out.shape[1] - tw))                     # keep the label inside the image
        cv2.putText(out, label, (tx, max(20, y0 - m - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thick, cv2.LINE_AA)
    return out


def overview(paths: list, out_path: Path, cols: int = 4, tile: int = 360):
    tiles = []
    for p in paths:
        im = cv2.imread(str(p))
        s = tile / max(im.shape[:2])
        im = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)))
        t = np.full((tile + 28, tile, 3), 255, np.uint8)
        t[:im.shape[0], :im.shape[1]] = im
        cv2.putText(t, p.stem.removesuffix("_hands"), (4, tile + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        tiles.append(t)
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(out_path), np.vstack(rows))


def main():
    ap = argparse.ArgumentParser(description="Detect hands in photos with MediaPipe Hand Landmarker")
    ap.add_argument("input", type=Path, help="image file or folder of images")
    ap.add_argument("--out", type=Path, default=HERE / "output")
    ap.add_argument("--model", type=Path, default=HERE / "models" / "hand_landmarker.task")
    ap.add_argument("--num-hands", type=int, default=4, help="max hands per image (2 players x 2 hands = 4)")
    ap.add_argument("--min-conf", type=float, default=0.5, help="detection / presence threshold (0-1)")
    ap.add_argument("--pad", type=float, default=0.0,
                    help="grey border as fraction of image size, e.g. 0.5 for close-up photos (default 0)")
    args = ap.parse_args()

    if not args.model.exists():
        sys.exit(f"Model not found: {args.model}\nDownload it (see README.md).")
    files = sorted(p for p in args.input.iterdir() if p.suffix.lower() in IMG_EXT) \
        if args.input.is_dir() else [args.input]
    args.out.mkdir(parents=True, exist_ok=True)

    results, written = {}, []
    with make_detector(args.model, args.num_hands, args.min_conf) as detector:
        print(f"{'image':32s} {'hands':>5s}  {'ms':>6s}  details")
        for f in files:
            bgr = cv2.imread(str(f))
            if bgr is None:
                print(f"{f.name:32s}  could not read, skipped"); continue
            hands, ms = detect(detector, bgr, args.pad)
            out_path = args.out / f"{f.stem}_hands.jpg"
            cv2.imwrite(str(out_path), draw(bgr, hands))
            written.append(out_path)
            results[f.name] = {"size_wh": [bgr.shape[1], bgr.shape[0]], "time_ms": round(ms, 1), "hands": hands}
            details = ", ".join(f"{h['handedness']} {h['score']:.2f}" for h in hands) or "-"
            print(f"{f.name:32s} {len(hands):5d}  {ms:6.1f}  {details}")

    (args.out / "results.json").write_text(json.dumps(results, indent=1))
    if len(written) > 1:
        overview(written, args.out / "_overview.jpg")
    times = [r["time_ms"] for r in results.values()]
    print(f"\n{len(results)} images, {sum(len(r['hands']) for r in results.values())} hands, "
          f"median {np.median(times):.1f} ms/image (first call includes model warm-up)")
    print(f"Annotated images + results.json in: {args.out}")


if __name__ == "__main__":
    main()
