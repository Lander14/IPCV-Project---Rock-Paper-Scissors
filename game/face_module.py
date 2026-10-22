"""Face tracking + sunglasses AR, wrapping the group's face_tracking_final.py unchanged.

This module is the only place that imports ../face_tracking/face_tracking_final.py. It reuses its
FaceDetector (MediaPipe FaceLandmarker), FaceKalman (temporal filtering, T1-03) and FaceTracker
(multi-face association + appearance re-ID, T1-01/T1-02) exactly as written - including its own
coasting through brief misses (T1-07, LOST_TIMEOUT) - and calls its own draw_sunglasses/draw_hp_bars
drawing functions unchanged too.

What this module adds, since face_tracking_final.py was written as a standalone demo, not a library:
- `model_path` is passed to FaceDetector explicitly (game/models/face_landmarker.task), instead of the
  original's hardcoded Windows constant (MODEL = C:\\cvdata\\face_landmarker.task), which nothing here
  ever reads: face_tracking_final.py itself is unmodified.
- Face tracks are linked to game player IDs (not FaceTracker's own track IDs, which are assigned by its
  own appearance re-ID and can disagree with the pose-based identity PlayerTracker already settled) by
  nearest pose head, each frame (T1-06: prevents glasses/HP from switching between players).
- The HP bar is driven by the real game snapshot instead of the demo's own key-press test damage
  (T1-08/T1-09): draw_hp_bars is called unchanged, with a game_state dict built fresh each frame from
  game.snapshot(), using "hit" events for the flash.
- An optional detect-every-Nth-frame cadence (`every`), letting the Kalman filter coast (predict, no new
  detection) on the skipped frames - this is FaceTracker.step()'s own existing "no detections this call"
  path (empty list in, predict + damp only), not new logic.

    faces = FaceModule()                                   # loads face_landmarker.task + the sprites
    tracks = faces.update(frame, dt, t_s)                   # raw FaceTracker.Track list, this call's state
    views = faces.assign_to_players(tracks, players)        # {pid: view} - view.id is the PLAYER id
    faces.draw(frame, views, snap, events, t_s)              # sunglasses + the real-HP bar, unchanged drawing
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
FACE_DIR = HERE.parent / "face_tracking"
DEFAULT_MODEL = HERE / "models" / "face_landmarker.task"

MAX_ASSIGN_DIST_FRAC = 0.35     # a face farther than this fraction of frame width from every head is unassigned


def _ft():
    """Imports face_tracking_final.py lazily (needs mediapipe), from its own unmodified folder."""
    if str(FACE_DIR) not in sys.path:
        sys.path.insert(0, str(FACE_DIR))
    import face_tracking_final as ft
    return ft


def assign_to_players(tracks, players, frame_w=1280):
    """tracks: FaceTracker.step()'s output. players: {pid: Player} from PlayerTracker. One track per
    player at most, nearest face-centre to player head, gated by distance. Returns {pid: track}."""
    gate = MAX_ASSIGN_DIST_FRAC * frame_w
    candidates = []
    for ti, tr in enumerate(tracks):
        cx, cy = tr.kf.center
        for pid, p in players.items():
            if not p.present or p.head is None:
                continue
            d = math.hypot(cx - p.head[0], cy - p.head[1])
            if d <= gate:
                candidates.append((d, ti, pid))
    candidates.sort(key=lambda c: c[0])
    out, used_tracks, used_players = {}, set(), set()
    for d, ti, pid in candidates:
        if ti in used_tracks or pid in used_players:
            continue
        used_tracks.add(ti)
        used_players.add(pid)
        out[pid] = tracks[ti]
    return out


class FaceModule:
    def __init__(self, model_path=None, max_faces=2, every=1, detector=None, tracker=None, glasses=None):
        ft = _ft()
        self.ft = ft
        self.detector = detector if detector is not None else ft.FaceDetector(str(model_path or DEFAULT_MODEL))
        self.tracker = tracker if tracker is not None else ft.FaceTracker()
        self.glasses = glasses if glasses is not None else ft.load_glasses()
        self.every, self._n = max(1, every), 0
        self._hp_state = {}                               # pid -> last_hit (seconds), for the flash

    def update(self, frame, dt, t_s):
        """Runs face detection every `every`-th call; on the other calls the Kalman filter only predicts
        and coasts (FaceTracker's own built-in behaviour for a call with no detections). Returns the
        current FaceTracker.Track list."""
        import cv2
        h, w = frame.shape[:2]
        do_detect = self._n % self.every == 0
        self._n += 1
        if do_detect:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = self.detector.detect_full(rgb)
            detect_region = self.detector.detect_region
        else:
            detections, detect_region = [], None
        return self.tracker.step(detections, dt, t_s, (w, h), detect_region, self._n)

    def assign_to_players(self, tracks, players, frame_w=1280):
        return assign_to_players(tracks, players, frame_w)

    def _views(self, assigned):
        """Lightweight view objects with `.id` = the PLAYER id (not the FaceTracker track id), `.kf` and
        `.eyes` from the real track - so the group's own draw_sunglasses()/draw_hp_bars() can be called
        completely unmodified while keying glasses and HP by player identity, not FaceTracker's own id."""
        return [SimpleNamespace(id=pid, kf=tr.kf, eyes=tr.eyes) for pid, tr in assigned.items()]

    def draw(self, frame, assigned, snap, events, t_s, glasses=True, hp_bar=True):
        """assigned: {pid: track} from assign_to_players(). snap: game.snapshot(). events: this frame's
        game.update() events (for the hit flash). t_s: this frame's time, in seconds."""
        views = self._views(assigned)
        if glasses:
            self.ft.draw_sunglasses(frame, views, self.glasses)
        if hp_bar:
            for e in events:
                if e["type"] == "hit":
                    self._hp_state[e["target"]] = t_s
            game_state = {
                pid: {"hp": p["hp"], "max_hp": p["max_hp"], "last_hit": self._hp_state.get(pid, -1.0)}
                for pid, p in snap["players"].items() if pid in assigned
            }
            self.ft.draw_hp_bars(frame, views, game_state, t_s)

    def close(self):
        self.detector.close()
