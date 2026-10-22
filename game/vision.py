"""Vision facade for the camera game mode: the one place that owns the heavy models.

Runs the pose tracker (PlayerTracker), the hand adapter and the face module at most once per *camera*
frame each (not once per render-loop tick, which can be faster than the camera's own FPS) and turns
their output into the game's PlayerInput contract. Heavy imports (ultralytics, mediapipe) only happen
when the real backends are built, so this module - and its tests - work without them installed; inject
fakes for `tracker` / `hands` / `faces` to test without any of them.

    vision = Vision()                                       # PlayerTracker + HandAdapter + FaceModule
    vision.warm_up()                                        # optional: load the hand model now, not mid-game
    players, inputs, faces = vision.process(frame, t_ms, n)  # frame: mirrored BGR; n: Camera.latest()'s count
    ...
    events = game.update(t_ms, inputs)
    vision.draw_faces(frame, faces, game.snapshot(), events, t_ms / 1000.0)   # sunglasses + the real HP bar
"""
from __future__ import annotations

from face_module import FaceModule
from hand_adapter import HandAdapter
from player_tracker import PlayerTracker


class Vision:
    def __init__(self, tracker=None, hands=None, faces=None, use_hands=True, use_faces=True,
                 hands_kwargs=None, faces_kwargs=None, **tracker_kwargs):
        self._tracker = tracker if tracker is not None else PlayerTracker(**tracker_kwargs)
        self._hands = (hands if hands is not None else HandAdapter(**(hands_kwargs or {}))) if use_hands else None
        self._faces = (faces if faces is not None else FaceModule(**(faces_kwargs or {}))) if use_faces else None
        self._last_n = None
        self._last_players, self._last_hand_out, self._last_faces = {}, {}, {}
        self._last_t_s = None

    def warm_up(self):
        """Builds the real hand model now (import + load the .task file), so a missing dependency or
        model file fails at startup, not mid-game, and the first camera frame isn't stalled loading it.
        The pose model (PlayerTracker.__init__) and the face model (FaceModule.__init__) already build
        eagerly, at construction time, for the same reason - there is nothing left to warm up for them."""
        if self._hands is not None and hasattr(self._hands, "_ensure_model"):
            self._hands._ensure_model()

    def process(self, frame, t_ms, n=None):
        """Returns ({pid: Player}, {pid: PlayerInput}, {pid: face_track}) for this frame. The hand
        gesture (if any) overrides PlayerTracker's own raised-wrist fallback for both the gesture and
        the hand position. `n`: the camera's own frame counter (Camera.latest()'s second value). When
        given and unchanged since the last call, none of the three models are re-run on what is
        bit-for-bit the same picture (the render loop can run faster than the camera's own FPS)."""
        same_frame = n is not None and n == self._last_n

        if same_frame:
            players = self._last_players
        else:
            players = self._tracker.update(frame, t_ms)
            self._last_players = players

        if self._hands is None:
            hand_out = {}
        elif same_frame:
            hand_out = self._last_hand_out
        else:
            hand_out = self._hands.update(frame, players, t_ms)
            self._last_hand_out = hand_out

        if self._faces is None:
            face_assigned = {}
        elif same_frame:
            face_assigned = self._last_faces
        else:
            t_s = t_ms / 1000.0
            dt = 0.0 if self._last_t_s is None else max(0.0, t_s - self._last_t_s)
            self._last_t_s = t_s
            tracks = self._faces.update(frame, dt, t_s)
            face_assigned = self._faces.assign_to_players(tracks, players, frame.shape[1])
            self._last_faces = face_assigned

        self._last_n = n

        inputs = {}
        for pid in players:
            gesture, hand_xy = hand_out.get(pid, (None, None))
            inputs[pid] = self._tracker.game_input(pid, gesture, hand_xy)
        return players, inputs, face_assigned

    def hand_signals(self):
        """([HandSlot.signal(), ...], [raw_wrist_or_None, ...]) from the most recent hand-model call,
        for --log (task2_hands.py's own CSV metrics, plus the pre-filter wrist for a before/after
        comparison). Both empty when hands are off or the model hasn't run yet."""
        if self._hands is None:
            return [], []
        return list(self._hands.last_signals), list(self._hands.last_raw_wrists)

    def draw_faces(self, frame, face_assigned, snap, events, t_s, **kwargs):
        """Draws the sunglasses + the real-game HP bar for the tracks process() just returned."""
        if self._faces is not None:
            self._faces.draw(frame, face_assigned, snap, events, t_s, **kwargs)

    def close(self):
        if self._hands is not None:
            self._hands.close()
        if self._faces is not None:
            self._faces.close()
