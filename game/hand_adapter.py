"""Hand gestures for the duel: MediaPipe landmarks -> the group's task2_hands.py filtering -> a player.

Pipeline, matching task2_hands.py's own docstring:
    webcam -> MediaPipe Hands (21 landmarks/hand)
           -> HandSlot (task2_hands.py, unmodified): jump rejection, min-confidence gating, grace frames
              for a brief dropout, One Euro filtering (T2-04/05/06)
           -> classify(slot.pts): a fresh per-frame gesture from the *filtered* landmarks
           -> assign(): which player's raised hand this is (nearest raised wrist, one hand per player)

Hold-to-confirm is not duplicated on top of HandSlot's own: game_logic's stable_frames debounce is the
single place that decides when a gesture is "confirmed" (T4-04), so classify() here reports the raw
per-frame gesture, not HandSlot's own slot.gesture (which would stack a second, redundant hold delay on
top and hurt T2-07's "no excessive delay").

The classifier emits FIRE / WATER / LIGHTNING, the names game_logic.GESTURE_TO_ACTION already maps to
ROCK / PAPER / SCISSORS ("the names of the first hand script").

    adapter = HandAdapter()                             # HandModel() + 2 HandSlots, built lazily
    hand_inputs = adapter.update(frame, players, t_ms)   # {pid: (gesture_or_None, wrist_xy)}
    adapter.last_signals                                 # [HandSlot.signal(), ...] of this call, for --log
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HAND_DIR = HERE.parent / "hand_gestures"
WRIST = 0
TIPS = (8, 12, 16, 20)         # index, middle, ring, pinky fingertips
PIPS = (6, 10, 14, 18)         # middle joint of each of those fingers
EXTEND_RATIO = 1.1             # finger "up" if the tip is this much farther from the wrist than the PIP joint

MAX_ASSIGN_DIST_FRAC = 0.22    # a hand farther than this fraction of frame width from every wrist is unassigned
N_SLOTS = 2                    # one hand tracked per player, matching task2_hands.py's own HandSlot(0), HandSlot(1)


def _th():
    """Imports task2_hands.py lazily (needs mediapipe), from its own unmodified folder.

    task2_hands.py reads the legacy `mp.solutions.hands.HAND_CONNECTIONS` at import time, only to build
    a drawing helper (draw_slot) this module never calls. mediapipe removed `mp.solutions` entirely
    before it had wheels for this Python (3.14) - the same fact hand_detection/requirements.txt notes -
    so there is no installable mediapipe version where the unmodified file would otherwise import. A
    minimal shim supplies just that one constant; nothing this module actually uses (HandSlot, OneEuro,
    assign) ever touches mp.solutions, so the shim's contents don't affect any real behaviour."""
    import mediapipe as mp
    if not hasattr(mp, "solutions"):
        from types import SimpleNamespace
        hand_connections = frozenset({                                # the 21-landmark skeleton, for draw_slot
            (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
            (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
        })
        mp.solutions = SimpleNamespace(hands=SimpleNamespace(HAND_CONNECTIONS=hand_connections))
    if str(HAND_DIR) not in sys.path:
        sys.path.insert(0, str(HAND_DIR))
    import task2_hands as th
    return th


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
    """hands: [(gesture_or_None, wrist_xy), ...] this frame. players: {pid: Player}.
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

    def __init__(self, model_path=None, max_hands=4, min_detection_conf=0.6, min_tracking_conf=0.5):
        import mediapipe as mp                                        # here, so the rest can be used without it
        from mediapipe.tasks.python import BaseOptions, vision
        path = model_path or HERE.parent / "hand_detection" / "models" / "hand_landmarker.task"
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_conf,
            min_hand_presence_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf,
        )
        self._mp = mp
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def detect_raw(self, frame, t_ms):
        """`frame`: BGR image, already mirrored (as the game shows it). Returns [(pts, confidence), ...],
        pts = 21 (x, y) landmarks in image pixels, confidence = the handedness classification score (the
        same signal task2_hands.py's HandSlot.update() expects as `conf`)."""
        import cv2
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts = max(int(t_ms), self._last_ts + 1)                        # VIDEO mode: strictly increasing
        self._last_ts = ts
        result = self._landmarker.detect_for_video(image, ts)
        out = []
        for lm, handedness in zip(result.hand_landmarks, result.handedness):
            pts = [(p.x * w, p.y * h) for p in lm]
            out.append((pts, handedness[0].score))
        return out

    def close(self):
        self._landmarker.close()


class HandAdapter:
    """Per-frame hand gestures assigned to players, filtered through task2_hands.py's HandSlot (One Euro,
    grace frames, jump rejection, min confidence). Owns the HandModel (built lazily) and 2 HandSlots;
    inject `model` (anything with a `.detect_raw(frame, t_ms)` method) in tests."""

    def __init__(self, model=None, use_filter=True, n_slots=N_SLOTS):
        self._model = model
        self.use_filter, self.n_slots = use_filter, n_slots
        self._th, self.slots = None, None                            # built lazily: see _ensure_slots()
        self.last_signals, self.last_raw_wrists = [], [None] * n_slots

    def _ensure_model(self):
        if self._model is None:
            self._model = HandModel()
        return self._model

    def _ensure_slots(self):
        if self.slots is None:
            self._th = _th()
            self.slots = [self._th.HandSlot(i) for i in range(self.n_slots)]
        return self.slots

    def update(self, frame, players, t_ms):
        """`frame`: the mirrored BGR frame Vision is processing this call. `players`: {pid: Player} from
        PlayerTracker. Returns {pid: (gesture_or_None, hand_xy)} for players with an assigned hand."""
        if not any(p.present for p in players.values()):
            return {}                                                 # nobody to cast: skip the model entirely
        slots = self._ensure_slots()
        frame_w = frame.shape[1]
        detected = self._ensure_model().detect_raw(frame, t_ms)       # [(pts, conf), ...], unfiltered
        t_s = t_ms / 1000.0
        slot_hits = self._th.assign(detected, slots, frame_w)         # which slot each detection belongs to
        for s in slots:
            if s.idx in slot_hits:
                pts, conf = slot_hits[s.idx]
                s.update(pts, conf, t_s, self.use_filter, frame_w)
            elif s.pts is not None:
                s.mark_missing()
        self.last_raw_wrists = [s.raw_wrist for s in slots]           # before One Euro: for --log's before/after
        self.last_signals = [s.signal() for s in slots]
        hands = [(classify(s.pts), tuple(s.pts[WRIST])) for s in slots if s.pts is not None and s.missing == 0]
        return assign(hands, players, frame_w)

    def close(self):
        if self._model is not None and hasattr(self._model, "close"):
            self._model.close()
