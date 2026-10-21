"""Vision facade for the camera game mode: the one place that owns the heavy models.

Runs the pose tracker (PlayerTracker, every `every`-th frame, its own cadence) and the hand adapter
(every frame) at most once per frame, and turns their output into the game's PlayerInput contract.
Heavy imports (ultralytics, mediapipe) only happen when the real backends are built, so this module -
and its tests - work without them installed; inject fakes for `tracker` / `hands` to test without either.

    vision = Vision()                                 # PlayerTracker + HandAdapter, built lazily
    vision.warm_up()                                  # optional: load the real models now, not mid-game
    players, inputs = vision.process(frame, t_ms, n)   # frame: mirrored BGR; n: Camera.latest()'s frame count
"""
from __future__ import annotations

from hand_adapter import HandAdapter
from player_tracker import PlayerTracker


class Vision:
    def __init__(self, tracker=None, hands=None, use_hands=True, **tracker_kwargs):
        self._tracker = tracker if tracker is not None else PlayerTracker(**tracker_kwargs)
        self._hands = (hands if hands is not None else HandAdapter()) if use_hands else None
        self._last_n, self._last_hand_out = None, {}

    def warm_up(self):
        """Builds the real hand model now (import + load the .task file), so a missing dependency or
        model file fails at startup, not mid-game, and the first camera frame isn't stalled loading it."""
        if self._hands is not None and hasattr(self._hands, "_ensure_model"):
            self._hands._ensure_model()

    def process(self, frame, t_ms, n=None):
        """Returns ({pid: Player}, {pid: PlayerInput}) for this frame. The hand gesture (if any) overrides
        PlayerTracker's own raised-wrist fallback for both the gesture and the hand position.
        `n`: the camera's own frame counter (Camera.latest()'s second value). When given and unchanged
        since the last call, the hand model is not re-run on what is bit-for-bit the same picture."""
        players = self._tracker.update(frame, t_ms)
        if self._hands is None:
            hand_out = {}
        elif n is not None and n == self._last_n:
            hand_out = self._last_hand_out
        else:
            hand_out = self._hands.update(frame, players, t_ms)
            self._last_n, self._last_hand_out = n, hand_out
        inputs = {}
        for pid in players:
            gesture, hand_xy = hand_out.get(pid, (None, None))
            inputs[pid] = self._tracker.game_input(pid, gesture, hand_xy)
        return players, inputs

    def close(self):
        if self._hands is not None:
            self._hands.close()
