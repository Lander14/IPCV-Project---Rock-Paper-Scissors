"""Hand gestures for the duel, from MediaPipe Hands landmarks (ported from the group's task2_hands.py).

Pure, model-free gesture classification plus hand-to-player assignment live here, so this file is
testable without a camera or MediaPipe installed. HandModel (the only class that imports MediaPipe) is
the thin backend a Vision facade drives; inject a fake one in tests.

The classifier emits FIRE / WATER / LIGHTNING, the names game_logic.GESTURE_TO_ACTION already maps to
ROCK / PAPER / SCISSORS ("the names of the first hand script"). Hold-to-confirm is not duplicated here:
game_logic's own stable_frames debounce already does that, so this module reports the raw per-frame
gesture (or None for an unclear pose) and lets the game decide when it is stable.

    adapter = HandAdapter()                             # HandModel() built lazily on first use
    hand_inputs = adapter.update(frame, players, t_ms)   # {pid: (gesture_or_None, wrist_xy)}
"""
from __future__ import annotations

import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
WRIST = 0
TIPS = (8, 12, 16, 20)         # index, middle, ring, pinky fingertips
PIPS = (6, 10, 14, 18)         # middle joint of each of those fingers
EXTEND_RATIO = 1.1             # finger "up" if the tip is this much farther from the wrist than the PIP joint

MAX_ASSIGN_DIST_FRAC = 0.22    # a hand farther than this fraction of frame width from every wrist is unassigned


def finger_states(pts):
    """True per finger (index, middle, ring, pinky) if extended, by distance from the wrist.
    `pts`: 21 (x, y) landmarks in image pixels. Works at any hand rotation (no finger-direction math)."""
    wx, wy = pts[WRIST]
    states = []
    for tip, pip in zip(TIPS, PIPS):
        d_tip = math.hypot(pts[tip][0] - wx, pts[tip][1] - wy)
        d_pip = math.hypot(pts[pip][0] - wx, pts[pip][1] - wy)
        states.append(d_tip > d_pip * EXTEND_RATIO)
    return states


def classify(pts):
    """Gesture name (FIRE/WATER/LIGHTNING) from 21 (x, y) landmarks, or None for an in-between pose."""
    states = finger_states(pts)
    if not any(states):
        return "FIRE"           # fist / rock
    if all(states):
        return "WATER"          # open hand / paper
    if states == [True, True, False, False]:
        return "LIGHTNING"      # index + middle / scissors
    return None


def assign(hands, players, frame_w):
    """hands: [(gesture_or_None, wrist_xy), ...] detected this frame. players: {pid: Player}.
    Each player is matched against their *raised* wrist only (the casting hand, same convention as
    Player.raised_hand() / the player's lane choice) - not whichever of their two wrists is nearest,
    which would pick arbitrarily between a resting and a raised hand. One hand per player at most, gated
    by distance. Returns {pid: (gesture, wrist_xy)} only for players with a matched hand; a hand with
    nothing close enough is left unassigned (no fallback)."""
    gate = MAX_ASSIGN_DIST_FRAC * frame_w
    candidates = []
    for hi, (_gesture, wrist_xy) in enumerate(hands):
        for pid, p in players.items():
            if not p.present:
                continue
            pw = p.raised_hand()
            if pw is None:
                continue
            d = math.hypot(wrist_xy[0] - pw[0], wrist_xy[1] - pw[1])
            if d <= gate:
                candidates.append((d, hi, pid))
    candidates.sort(key=lambda c: c[0])
    out, used_hands, used_players = {}, set(), set()
    for d, hi, pid in candidates:
        if hi in used_hands or pid in used_players:
            continue
        used_hands.add(hi)
        used_players.add(pid)
        out[pid] = hands[hi]
    return out


class HandModel:
    """Wraps MediaPipe's HandLandmarker Task API (mediapipe >= 1.0 has no mp.solutions any more - see
    hand_detection/requirements.txt). The only part of this module that needs mediapipe installed.
    Uses VIDEO mode, like the group's face tracker did for its own landmarker: it needs the caller's
    frame-clock timestamps (`t_ms`), strictly increasing, not wall-clock time."""

    def __init__(self, model_path=None, max_hands=4, min_conf=0.5):
        import mediapipe as mp                                        # here, so the rest can be used without it
        from mediapipe.tasks.python import BaseOptions, vision
        path = model_path or HERE.parent / "hand_detection" / "models" / "hand_landmarker.task"
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_conf,
            min_hand_presence_confidence=min_conf,
        )
        self._mp = mp
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def detect(self, frame, t_ms):
        """`frame`: BGR image, already mirrored (as the game shows it). Returns [(gesture_or_None, wrist_xy)]."""
        import cv2
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts = max(int(t_ms), self._last_ts + 1)                        # VIDEO mode: strictly increasing
        self._last_ts = ts
        result = self._landmarker.detect_for_video(image, ts)
        out = []
        for lm in result.hand_landmarks:
            pts = [(p.x * w, p.y * h) for p in lm]
            out.append((classify(pts), pts[WRIST]))
        return out

    def close(self):
        self._landmarker.close()


class HandAdapter:
    """Per-frame hand gestures assigned to players. Owns the HandModel (built lazily) so Vision only
    constructs the pose model up front; inject `model` (anything with a `.detect(frame, t_ms)` method)
    in tests."""

    def __init__(self, model=None):
        self._model = model

    def _ensure_model(self):
        if self._model is None:
            self._model = HandModel()
        return self._model

    def update(self, frame, players, t_ms):
        """`frame`: the mirrored BGR frame Vision is processing this call. `players`: {pid: Player} from
        PlayerTracker. Returns {pid: (gesture_or_None, hand_xy)} for players with an assigned hand."""
        if not any(p.present for p in players.values()):
            return {}                                                 # nobody to cast: skip the model entirely
        hands = self._ensure_model().detect(frame, t_ms)
        return assign(hands, players, frame.shape[1])

    def close(self):
        if self._model is not None and hasattr(self._model, "close"):
            self._model.close()
