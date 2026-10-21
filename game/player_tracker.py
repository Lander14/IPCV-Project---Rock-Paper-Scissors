"""Player tracking for the rock-paper-scissors duel (task T3), cut down to what the game needs.

What the game uses, per player and per frame:
    present   is the player in view? (the game pauses after 0.4 s without)
    head      where the HUD is drawn and where the lanes end
    wrists    which hand (from the hand module) belongs to which player; the raised one picks the lane
    identity  PLAYER 1 = left, PLAYER 2 = right, for the whole match, also after occlusions and re-entries
Not needed by the game: legs, eyes and ears, the skeleton drawing, a console print every frame.

Built on two_player_pose_tracking.py (YOLOv8n-pose + ByteTrack), with these changes:
  - input size 480 instead of 640: about 45 % less time; head moves ~5 px, wrists ~20 px (measured; a lane is ~115 px)
  - the pose model runs on every 2nd camera frame (people move slowly); results are kept in between
  - only the 2 biggest people count: players stand closest to the camera, spectators behind are ignored
  - identities fixed (IdentityManager): no 2-s gap after a new tracker id, left = PLAYER 1, re-entry keeps the id
  - ByteTrack keeps lost tracks longer (bytetrack_players.yaml), so fewer new ids in the first place
  - output ready for the game: PlayerInput(present, gesture, hand_xy, body_xy)

    tracker = PlayerTracker()                     # device="cuda" if PyTorch with CUDA is installed (RTX 3050 Ti)
    players = tracker.update(mirrored_frame, t_ms)
    inputs = {pid: tracker.game_input(pid, gesture) for pid in players}

Demo:  python player_tracker.py                 (webcam)   --video clip.mp4   --imgsz 640   --every 1   --device cuda
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SH, R_SH, L_WR, R_WR = 0, 1, 2, 3, 4, 5, 6, 9, 10
SHOULDER_M = 0.40          # typical adult shoulder width, for the rough distance estimate
CAMERA_FOV_DEG = 70.0      # horizontal field of view of a usual laptop webcam


# ------------------------------------------------------------------ identities (no YOLO needed: tested on its own)
APPEARANCE_WEIGHT = 0.5     # how much a clothing-colour mismatch adds to the position cost, when enabled


def appearance_distance(h1, h2):
    """Bhattacharyya distance between two colour histograms (0 = similar, up to ~1 = very different).
    Lazy cv2 import: IdentityManager stays importable and testable without OpenCV when this is unused."""
    if h1 is None or h2 is None:
        return 0.0
    import cv2
    return float(cv2.compareHist(h1.astype("float32"), h2.astype("float32"), cv2.HISTCMP_BHATTACHARYYA))


def appearance_descriptor(frame, box):
    """Normalized HSV histogram of the upper-body/clothing region of `box` (x1, y1, x2, y2) in `frame`.
    Ported from the group's re-ID script. Lazy cv2 import, same reason as above."""
    import cv2
    import numpy as np
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    h, w = frame.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw < 8 or bh < 12:
        return None
    crop = frame[y1 + int(0.20 * bh):y1 + int(0.72 * bh), x1 + int(0.15 * bw):x1 + int(0.85 * bw)]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist


class IdentityManager:
    """Tracker ids -> PLAYER 1 / PLAYER 2.

    - a known tracker id keeps its player
    - a new tracker id goes AT ONCE to a player that is not visible in this frame, if it is close to where that
      player was (or that player has been away longer than `forget_s`: re-entry anywhere)
    - the very first time, players are given by side: left = PLAYER 1, right = PLAYER 2
    - never more than n players: anyone else is ignored
    - `appearance=True` (default off, from the group's re-ID script) also matches on a clothing-colour
      histogram passed into `update()`; with none passed, behaviour is identical to `appearance=False`.
    """

    def __init__(self, n=2, width=1280, near=0.35, forget_s=3.0, appearance=False):
        self.w, self.near, self.forget = float(width), near, forget_s * 1000
        self.appearance = appearance
        self.slots = {p: {"tid": None, "c": None, "seen": None} for p in range(1, n + 1)}
        self.hist = {p: None for p in range(1, n + 1)}

    def _cost(self, p, c, t, hist=None):
        s = self.slots[p]
        if s["c"] is None:                                      # never seen: left player -> 1, right -> 2
            side = c[0] / self.w
            return side if p == 1 else 1 - side
        d = math.dist(c, s["c"]) / self.w
        if t - s["seen"] < self.forget and d > self.near:       # seen recently, far away: not this player
            return math.inf
        if self.appearance and hist is not None and self.hist[p] is not None:
            d += APPEARANCE_WEIGHT * appearance_distance(self.hist[p], hist)
        return d

    def update(self, dets, t_ms, appearances=None):
        """dets: [(tracker_id, (x, y))] of this frame. `appearances`: optional {tracker_id: histogram},
        only used when `appearance=True`. Returns {tracker_id: player} (ignored people are left out)."""
        hists = appearances or {}
        out = {tid: p for tid, _ in dets for p, s in self.slots.items() if s["tid"] == tid}
        new = [(tid, c) for tid, c in dets if tid not in out]
        free = [p for p in self.slots if p not in out.values()]
        best = None
        for k in range(min(len(new), len(free)), 0, -1):      # as many assignments as possible, then cheapest
            for subset in itertools.combinations(new, k):
                for perm in itertools.permutations(free, k):
                    cost = sum(self._cost(p, c, t_ms, hists.get(tid)) for (tid, c), p in zip(subset, perm))
                    if cost < math.inf and (best is None or cost < best[0]):
                        best = (cost, list(zip(subset, perm)))
            if best:
                break
        for (tid, _), p in (best[1] if best else []):
            out[tid] = p
        centres = dict(dets)
        for tid, p in out.items():
            self.slots[p] = {"tid": tid, "c": centres[tid], "seen": t_ms}
            if self.appearance and hists.get(tid) is not None:
                old = self.hist[p]
                if old is None:
                    self.hist[p] = hists[tid].copy()
                else:
                    import cv2
                    blended = 0.8 * old + 0.2 * hists[tid]
                    cv2.normalize(blended, blended, alpha=1.0, norm_type=cv2.NORM_L1)
                    self.hist[p] = blended
        return out


# ------------------------------------------------------------------ what the game gets per player
@dataclass
class Player:
    present: bool = False
    head: tuple | None = None            # pixels
    wrists: list = field(default_factory=list)   # [(x, y)] of the confident wrists, pixels
    box: tuple | None = None             # x1, y1, x2, y2
    shoulder_px: float | None = None
    distance_m: float | None = None      # rough distance to the camera (shoulder width), T3-08
    track_id: int | None = None
    seen_ms: float | None = None         # time of the last detection
    eyes: tuple | None = None            # ((x, y), (x, y)) image-left / image-right eye, for the AR overlay

    def raised_hand(self):
        """The wrist that is highest in the picture: the casting hand, if the hand module does not say otherwise."""
        return min(self.wrists, key=lambda w: w[1]) if self.wrists else None


def _ema(old, new, a):
    return new if old is None or new is None else tuple(a * n + (1 - a) * o for n, o in zip(new, old))


class PlayerTracker:
    def __init__(self, model=None, imgsz=480, every=2, device=None, conf=0.35, kp_conf=0.35, min_height=0.25,
                 smooth=0.6, keep_ms=300, tracker=None, appearance=False):
        from ultralytics import YOLO                            # here, so the rest can be used without it
        local = HERE / "models" / "yolov8n-pose.pt"                 # put the model next to the code for the ZIP
        self.model = YOLO(str(model) if model else str(local) if local.exists() else "yolov8n-pose.pt")
        self.tracker = str(tracker or HERE / "bytetrack_players.yaml")
        self.imgsz, self.every, self.device, self.conf, self.kp_conf = imgsz, every, device, conf, kp_conf
        self.min_height, self.smooth, self.keep_ms, self.appearance = min_height, smooth, keep_ms, appearance
        self.players = {1: Player(), 2: Player()}
        self.ids, self.n, self.ms = None, 0, 0.0

    def update(self, frame, t_ms):
        """Run (every `every` frames) and return {1: Player, 2: Player}. `frame` as the game shows it (mirrored)."""
        h, w = frame.shape[:2]
        if self.ids is None:
            self.ids = IdentityManager(2, w, appearance=self.appearance)
        self.n += 1
        if (self.n - 1) % self.every == 0:
            t0 = time.perf_counter()
            r = self.model.track(frame, imgsz=self.imgsz, persist=True, tracker=self.tracker, conf=self.conf,
                                 device=self.device, verbose=False)[0]
            self.ms = (time.perf_counter() - t0) * 1000
            self._read(r, w, h, t_ms, frame)
        for p in self.players.values():
            p.present = p.seen_ms is not None and t_ms - p.seen_ms <= self.keep_ms
        return self.players

    def _read(self, r, w, h, t_ms, frame):
        if r.boxes is None or r.boxes.id is None or r.keypoints is None or not len(r.boxes):
            return
        boxes = r.boxes.xyxy.cpu().numpy()
        tids = r.boxes.id.int().cpu().tolist()
        kxy, kc = r.keypoints.xy.cpu().numpy(), r.keypoints.conf.cpu().numpy()
        people = [i for i in range(len(boxes)) if boxes[i, 3] - boxes[i, 1] >= self.min_height * h]
        people = sorted(people, key=lambda i: boxes[i, 3] - boxes[i, 1], reverse=True)[:2]   # the 2 biggest
        dets = [(tids[i], ((boxes[i, 0] + boxes[i, 2]) / 2, (boxes[i, 1] + boxes[i, 3]) / 2)) for i in people]
        appearances = {tids[i]: appearance_descriptor(frame, boxes[i]) for i in people} \
            if self.appearance else None
        mapping = self.ids.update(dets, t_ms, appearances)
        f = (w / 2) / math.tan(math.radians(CAMERA_FOV_DEG / 2))
        for i in people:
            pid = mapping.get(tids[i])
            if pid is None:
                continue
            p, pts, c = self.players[pid], kxy[i], kc[i]
            ok = c >= self.kp_conf
            gap = p.seen_ms is None or t_ms - p.seen_ms > self.keep_ms       # long gap: do not smooth with old data
            a = 1.0 if gap else self.smooth
            face = [j for j in (NOSE, L_EYE, R_EYE, L_EAR, R_EAR) if ok[j]]
            head = tuple(pts[NOSE]) if ok[NOSE] else tuple(pts[face].mean(0)) if face else \
                ((boxes[i, 0] + boxes[i, 2]) / 2, boxes[i, 1] + 0.12 * (boxes[i, 3] - boxes[i, 1]))
            p.head = _ema(p.head, tuple(map(float, head)), a)
            p.wrists = [tuple(map(float, pts[j])) for j in (L_WR, R_WR) if ok[j]]
            p.box = tuple(map(float, boxes[i]))
            if ok[L_EYE] and ok[R_EYE]:
                e1, e2 = tuple(map(float, pts[L_EYE])), tuple(map(float, pts[R_EYE]))
                p.eyes = (e1, e2) if e1[0] <= e2[0] else (e2, e1)    # (image-left point, image-right point)
            else:
                p.eyes = None
            if ok[L_SH] and ok[R_SH]:
                p.shoulder_px = float(np.linalg.norm(pts[L_SH] - pts[R_SH]))
                p.distance_m = f * SHOULDER_M / max(p.shoulder_px, 1.0)
            p.track_id, p.seen_ms = tids[i], t_ms

    def game_input(self, pid, gesture=None, hand_xy=None):
        """PlayerInput for game_logic: hand from the hand module if given, else the raised wrist."""
        from game_logic import PlayerInput
        p = self.players[pid]
        return PlayerInput(p.present, gesture, hand_xy or p.raised_hand(), p.head)


# ------------------------------------------------------------------ demo / check on the webcam or a video
def draw(img, players, tracker, fps):
    import cv2
    col = {1: (230, 110, 250), 2: (110, 230, 110)}
    for pid, p in players.items():
        if not p.present:
            continue
        x1, y1, x2, y2 = map(int, p.box)
        cv2.rectangle(img, (x1, y1), (x2, y2), col[pid], 1)
        cv2.circle(img, tuple(map(int, p.head)), 8, col[pid], -1, cv2.LINE_AA)
        for wr in p.wrists:
            cv2.circle(img, tuple(map(int, wr)), 10, col[pid], 2, cv2.LINE_AA)
        d = f"  {p.distance_m:.1f} m" if p.distance_m else ""
        cv2.putText(img, f"PLAYER {pid}{d}", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col[pid], 2)
    cv2.putText(img, f"{fps:.0f} FPS  pose {tracker.ms:.0f} ms (every {tracker.every}. frame, imgsz {tracker.imgsz})",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def main():
    import argparse
    import cv2
    ap = argparse.ArgumentParser(description="Player tracking for the duel (T3)")
    ap.add_argument("--video", help="a recorded clip instead of the webcam")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--every", type=int, default=2, help="run the pose model on every N-th frame")
    ap.add_argument("--device", default=None, help="cpu, cuda, ...")
    ap.add_argument("--no-window", action="store_true", help="print identity changes instead of showing a window")
    args = ap.parse_args()
    tracker = PlayerTracker(imgsz=args.imgsz, every=args.every, device=args.device)
    if args.video:
        cap = cv2.VideoCapture(args.video)
        vfps = cap.get(cv2.CAP_PROP_FPS) or 30
        read = lambda: cap.read()[1]
    else:
        from camera import Camera
        cam = Camera(args.camera, fast=True)
        read = lambda: cam.latest()[0]
        while read() is None:
            time.sleep(0.05)
    t0, n, fps, last, poses, last_t = time.monotonic(), 0, 0.0, None, [], None
    while True:
        frame = read()
        if frame is None:
            break
        frame = cv2.flip(cv2.resize(frame, (1280, 720)), 1)
        t_ms = n * 1000 / vfps if args.video else (time.monotonic() - t0) * 1000
        n += 1
        players = tracker.update(frame, t_ms)
        if (n - 1) % tracker.every == 0:
            poses.append(tracker.ms)
        now = time.perf_counter()
        if last_t is not None:                                     # loop rate (with drawing and window)
            fps = 0.9 * fps + 0.1 / max(1e-3, now - last_t) if fps else 1 / max(1e-3, now - last_t)
        last_t = now
        state = tuple((pid, p.present, p.track_id) for pid, p in players.items())
        if args.no_window:
            if state != last:
                desc = "  ".join(f"P{pid}: {'track ' + str(tid) if pr else '-'}" for pid, pr, tid in state)
                print(f"{t_ms / 1000:6.2f} s  {desc}")
            last = state
            continue
        draw(frame, players, tracker, fps)
        cv2.imshow("Player tracking (T3)", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break
    if poses:
        print(f"pose model: {np.mean(poses[2:]):.0f} ms per run, on every {tracker.every}. frame "
              f"-> {np.mean(poses[2:]) / tracker.every:.0f} ms per frame on average")
    if not args.video:
        cam.release()


if __name__ == "__main__":
    main()
