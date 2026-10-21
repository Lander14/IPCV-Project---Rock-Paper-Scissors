"""Sunglasses AR overlay for the duel: pose eye keypoints -> a sunglasses sprite warped onto the face.

Ported from the group's face_tracking_final.py, with its separate face-landmark model replaced by the
pose eye keypoints PlayerTracker already extracts (no extra inference, no extra model file to ship).
The source JPGs have no alpha channel: make_transparent keys the background out by saturation, once,
at load time (this is the group's own technique, not a generic border flood-fill).

    overlay = SunglassesOverlay()                   # loads game/assets/sunglasses_picture*.jpg, keyed once
    overlay.draw(frame, players, t_ms)               # players: {pid: Player}; drawn where p.eyes is set
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"

GLASSES_FILES = {1: ("sunglasses_picture.jpg", 0.49), 2: ("sunglasses_picture2.jpg", 0.45)}
GLASSES_WIDTH = 1.6     # how much wider the sprite is than its own eye-to-eye distance
EYE_SMOOTH = 0.3        # EMA toward the new eye positions each frame: lower = smoother, more lag
HOLD_MS = 400           # keep the last position this long after the eyes are lost, then fade out
FADE_MS = 250


def make_transparent(img):
    """A plain photo of glasses on a plain background -> RGBA, cropped to the glasses blob."""
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32)
    alpha = np.clip((sat - 30) / 50, 0, 1)
    solid = cv2.morphologyEx((alpha > 0.5).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(solid)
    if n < 2:
        raise ValueError("No glasses found in the image")
    biggest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    alpha[labels != biggest] = 0
    ys, xs = np.where(alpha > 0.05)
    img = np.dstack([img, (alpha * 255).astype(np.uint8)])
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def load_glasses(assets_dir=ASSETS, files=GLASSES_FILES):
    """{pid: (rgba_sprite, src)}: `src` is the sprite's own left/right eye points, for the warp below.
    A missing or unreadable asset is skipped (that player plays without glasses, not a crash)."""
    glasses = {}
    for pid, (name, eye_y) in files.items():
        img = cv2.imread(str(Path(assets_dir) / name), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 3:
            img = make_transparent(img)
        gh, gw = img.shape[:2]
        half = gw / (2 * GLASSES_WIDTH)
        src = np.float32([[gw / 2 - half, eye_y * gh], [gw / 2 + half, eye_y * gh]])
        glasses[pid] = (img, src)
    return glasses


def _ema(old, new, a):
    return new if old is None else tuple(a * n + (1 - a) * o for n, o in zip(new, old))


class SunglassesOverlay:
    def __init__(self, assets_dir=ASSETS):
        self.glasses = load_glasses(assets_dir)
        self._state = {}          # pid -> (left_eye, right_eye, last_seen_ms), smoothed

    def draw(self, frame, players, t_ms):
        """players: {pid: Player} from PlayerTracker. Draws only for players this overlay has sprites
        for, anchored on p.eyes (set by PlayerTracker from confident pose eye keypoints)."""
        for pid in self.glasses:
            p = players.get(pid)
            st = self._state.get(pid)
            eyes = getattr(p, "eyes", None) if p is not None else None
            if eyes is not None and p.present:
                left, right = eyes
                if st is not None:
                    left = _ema(st[0], left, EYE_SMOOTH)
                    right = _ema(st[1], right, EYE_SMOOTH)
                self._state[pid] = st = (left, right, t_ms)
                alpha_mul = 1.0
            elif st is not None and t_ms - st[2] <= HOLD_MS + FADE_MS:
                left, right, seen = st                       # briefly hold the last position, then fade
                age = max(0.0, t_ms - seen - HOLD_MS)
                alpha_mul = max(0.0, 1 - age / FADE_MS)
            else:
                self._state.pop(pid, None)
                continue
            if alpha_mul > 0:
                self._draw_one(frame, pid, left, right, alpha_mul)

    def _draw_one(self, frame, pid, left, right, alpha_mul):
        img, src = self.glasses[pid]
        height, width = frame.shape[:2]
        dst = np.float32([left, right])
        M, _ = cv2.estimateAffinePartial2D(src, dst)         # one similarity transform: position + roll + scale
        if M is None:
            return
        gh, gw = img.shape[:2]
        corners = cv2.transform(np.float32([[[0, 0], [gw, 0], [0, gh], [gw, gh]]]), M)[0]
        x0, y0 = np.floor(corners.min(axis=0)).astype(int)
        x1, y1 = np.ceil(corners.max(axis=0)).astype(int)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, width), min(y1, height)
        if x1 <= x0 or y1 <= y0:
            return
        M = M.copy()
        M[:, 2] -= (x0, y0)
        warped = cv2.warpAffine(img, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        alpha = (warped[:, :, 3:4].astype(np.float32) / 255.0) * alpha_mul
        roi = frame[y0:y1, x0:x1]
        roi[:] = (alpha * warped[:, :, :3] + (1 - alpha) * roi).astype(np.uint8)
