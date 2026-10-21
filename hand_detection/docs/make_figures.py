"""Figures for 'Hand detection - how it works.md'.

Run from the hand_detection folder:   python docs/make_figures.py
Writes docs/img/HD_*.jpg (the note embeds them).
"""
import sys
import types
from pathlib import Path

try:
    import sounddevice  # noqa: F401
except Exception:
    sys.modules["sounddevice"] = types.ModuleType("sounddevice")
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import detect_hands as dh  # noqa: E402

IMG = ROOT / "docs" / "img"
IMG.mkdir(parents=True, exist_ok=True)
S = ROOT / "samples"
FINGER_COL = {"thumb": (180, 105, 255), "index": (0, 165, 255), "middle": (0, 200, 0),
              "ring": (255, 160, 0), "pinky": (220, 0, 220)}              # BGR
FINGER_IDS = {"thumb": [1, 2, 3, 4], "index": [5, 6, 7, 8], "middle": [9, 10, 11, 12],
              "ring": [13, 14, 15, 16], "pinky": [17, 18, 19, 20]}
PALM = [0, 1, 5, 9, 13, 17]                                                  # wrist + finger bases
RPS_FINGERS = {"index": (6, 8), "middle": (10, 12), "ring": (14, 16), "pinky": (18, 20)}  # (PIP, tip)


def put(img, text, org, scale=0.8, thick=2, dark=False):
    """Text with an outline, readable on any background (dark=True: black text, white halo)."""
    fg, bg = ((0, 0, 0), (255, 255, 255)) if dark else ((255, 255, 255), (0, 0, 0))
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, bg, thick + 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thick, cv2.LINE_AA)


def titled(img, text, h=40, scale=0.7):
    bar = np.full((h, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack([bar, img])


def to_height(img, h, nearest=False):
    s = h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * s), h),
                      interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)


def hstack(panels, gap=14, arrows=False):
    h = max(p.shape[0] for p in panels)
    out = []
    for i, p in enumerate(panels):
        p = np.vstack([p, np.full((h - p.shape[0], p.shape[1], 3), 255, np.uint8)])
        out.append(p)
        if i < len(panels) - 1:
            sep = np.full((h, 50 if arrows else gap, 3), 255, np.uint8)
            if arrows:
                cv2.arrowedLine(sep, (8, h // 2), (42, h // 2), (0, 0, 220), 4, tipLength=0.4)
            out.append(sep)
    return np.hstack(out)


det = dh.make_detector(ROOT / "models" / "hand_landmarker.task", 4, 0.5)

# 1 · the 21 landmarks, numbered, on a big open palm --------------------------------------
im = cv2.imread(str(S / "02_two_open_palms.jpg"))
pts = np.array(dh.detect(det, im)[0][0]["landmarks_px"])
x0, y0 = (pts.min(0) - 60).astype(int).clip(0)
x1, y1 = (pts.max(0) + 60).astype(int)
crop = cv2.resize(im[y0:y1, x0:x1], None, fx=2, fy=2)
p2 = (pts - [x0, y0]) * 2
for f, ids in FINGER_IDS.items():
    chain = [0] + ids
    for a, b in zip(chain, chain[1:]):
        cv2.line(crop, tuple(p2[a].astype(int)), tuple(p2[b].astype(int)), FINGER_COL[f], 5, cv2.LINE_AA)
for i, p in enumerate(p2.astype(int)):
    cv2.circle(crop, tuple(p), 10, (255, 255, 255), -1)
    cv2.circle(crop, tuple(p), 10, (0, 0, 0), 2)
    put(crop, str(i), (p[0] + 13, p[1] + 9), 0.9)
cv2.imwrite(str(IMG / "HD_landmarks_numbered.jpg"), crop)

# 2 · what each network sees (photo 03) -----------------------------------------------------
im = cv2.imread(str(S / "03_waving_webcam_like.jpg"))
H, W = im.shape[:2]
hand = dh.detect(det, im)[0][0]
pts = np.array(hand["landmarks_px"])
#   net 1 input: whole photo, aspect ratio kept, padded ("letterbox") to 192 x 192
s = 192 / max(H, W)
small = cv2.resize(im, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
lb = np.zeros((192, 192, 3), np.uint8)
oy, ox = (192 - small.shape[0]) // 2, (192 - small.shape[1]) // 2
lb[oy:oy + small.shape[0], ox:ox + small.shape[1]] = small
palm = pts[PALM] * s + [ox, oy]
c, r = palm.mean(0), (palm.max(0) - palm.min(0)).max() / 2 * 1.15
stage1 = cv2.resize(lb, (384, 384), interpolation=cv2.INTER_NEAREST)
cv2.rectangle(stage1, tuple((2 * (c - r)).astype(int)), tuple((2 * (c + r)).astype(int)), (0, 0, 255), 2)
#   net 2 input: square crop around the hand, rotated so wrist -> middle-finger base points up
v = pts[9] - pts[0]
angle = np.degrees(np.arctan2(v[1], v[0])) + 90
center = pts[PALM].mean(0) + 0.35 * v
side = 3.5 * np.linalg.norm(v)
M = cv2.getRotationMatrix2D(tuple(map(float, center)), float(angle), 224 / side)
M[:, 2] += [112 - center[0], 112 - center[1]]
stage2 = cv2.resize(cv2.warpAffine(im, M, (224, 224)), (384, 384), interpolation=cv2.INTER_NEAREST)
final = dh.draw(im, [hand])
fig = hstack([titled(to_height(im, 384), "1. photo"),
              titled(stage1, "2. net 1 input: 192 x 192"),
              titled(stage2, "3. net 2 input: 224 x 224"),
              titled(to_height(final, 384), "4. result")], arrows=True)
cv2.imwrite(str(IMG / "HD_pipeline.jpg"), fig)

# 3 · the rock-paper-scissors rule, drawn ---------------------------------------------------
im = cv2.imread(str(next(S.glob("360_F_*.webp"))))
im = cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
out = im.copy()
for hand in dh.detect(det, im)[0]:
    pts = np.array(hand["landmarks_px"])
    w = pts[0]
    up = []
    for f, (p, t) in RPS_FINGERS.items():
        ext = np.linalg.norm(pts[t] - w) > np.linalg.norm(pts[p] - w)
        up.append(int(ext))
        col = (0, 190, 0) if ext else (0, 0, 230)
        cv2.line(out, tuple(w.astype(int)), tuple(pts[t].astype(int)), col, 3, cv2.LINE_AA)
        cv2.circle(out, tuple(pts[p].astype(int)), 7, (255, 255, 255), -1)
        cv2.circle(out, tuple(pts[p].astype(int)), 7, (0, 0, 0), 2)
        cv2.circle(out, tuple(pts[t].astype(int)), 8, col, -1)
    cv2.circle(out, tuple(w.astype(int)), 10, (0, 0, 0), -1)
    name = {(0, 0, 0, 0): "ROCK", (1, 1, 1, 1): "PAPER", (1, 1, 0, 0): "SCISSORS"}.get(tuple(up), "?")
    x, y = pts.min(0).astype(int)
    put(out, name, (int(x), int(max(30, y - 14))), 1.1, 3, dark=True)
cv2.imwrite(str(IMG / "HD_rps_rule.jpg"), out)

# 4 · scale: padding rescues the close-up and loses the small hands -------------------------
panels = []
for name, pad in [("01_fist_plain.jpg", 0), ("01_fist_plain.jpg", 0.5),
                  ("05_high_five_two_people.jpg", 0), ("05_high_five_two_people.jpg", 0.5)]:
    im = cv2.imread(str(S / name))
    hands, _ = dh.detect(det, im, pad)
    b = int(pad * max(im.shape[:2]))
    shown = cv2.copyMakeBorder(im, b, b, b, b, cv2.BORDER_CONSTANT, value=(128, 128, 128)) if b else im
    shifted = [dict(h, landmarks_px=(np.array(h["landmarks_px"]) + b).tolist(),
                    bbox_xyxy=[v + b for v in h["bbox_xyxy"]]) for h in hands]
    shown = to_height(dh.draw(shown, shifted), 330)
    panels.append(titled(shown, f"photo {name[:2]}, pad {pad}: {len(hands)} hand(s)", scale=0.6))
cv2.imwrite(str(IMG / "HD_scale_padding.jpg"), hstack(panels))

det.close()
print("figures:", ", ".join(sorted(p.name for p in IMG.iterdir())))
