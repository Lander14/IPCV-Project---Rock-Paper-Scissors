"""Tests of the camera-mode fusion (no camera, no YOLO, no MediaPipe needed).   python test_vision.py"""
import subprocess
import sys

from hand_adapter import HandAdapter, assign, classify
from player_tracker import Player
from vision import Vision

TIPS, PIPS = (8, 12, 16, 20), (6, 10, 14, 18)


class FakeFrame:
    """Stand-in for a real frame where only `.shape` (Vision reads frame.shape[1] for frame width) matters."""
    shape = (720, 1280, 3)


def hand_landmarks(extended):
    """21 (x, y) landmarks around a wrist at (100, 100): `extended` says which of the 4 fingers stick out."""
    pts = [(100.0, 100.0)] * 21
    for t, p, ext in zip(TIPS, PIPS, extended):
        pts[p] = (100.0, 60.0)                                 # pip: 40 px above the wrist
        pts[t] = (100.0, 0.0) if ext else (100.0, 80.0)        # tip far (extended) or near the wrist (curled)
    return pts


def test_classify_fist_is_fire():
    assert classify(hand_landmarks([False] * 4)) == "FIRE"


def test_classify_open_palm_is_water():
    assert classify(hand_landmarks([True] * 4)) == "WATER"


def test_classify_index_and_middle_is_lightning():
    assert classify(hand_landmarks([True, True, False, False])) == "LIGHTNING"


def test_classify_inbetween_pose_is_none():
    assert classify(hand_landmarks([True, False, False, False])) is None


def test_gesture_names_feed_the_existing_game_logic_map():
    from game_logic import GESTURE_TO_ACTION, PAPER, ROCK, SCISSORS
    assert GESTURE_TO_ACTION["FIRE"] == ROCK
    assert GESTURE_TO_ACTION["WATER"] == PAPER
    assert GESTURE_TO_ACTION["LIGHTNING"] == SCISSORS


# ------------------------------------------------------------------ hand-to-player assignment
def player(wrists, present=True):
    return Player(present=present, wrists=wrists)


def test_hand_assigned_to_the_nearest_wrist():
    players = {1: player([(100.0, 100.0)]), 2: player([(900.0, 100.0)])}
    out = assign([("FIRE", (110.0, 100.0))], players, frame_w=1280)
    assert out == {1: ("FIRE", (110.0, 100.0))}


def test_hand_near_the_player_s_resting_wrist_is_ignored_when_a_raised_wrist_exists():
    # the player has a high (raised, casting) wrist and a low (resting) one; only the raised one counts
    p = player([(100.0, 100.0), (100.0, 400.0)])
    out = assign([("WATER", (100.0, 400.0))], {1: p}, frame_w=1280)
    assert out == {}


def test_hand_near_the_player_s_raised_wrist_is_assigned():
    p = player([(100.0, 100.0), (100.0, 400.0)])
    out = assign([("FIRE", (105.0, 100.0))], {1: p}, frame_w=1280)
    assert out == {1: ("FIRE", (105.0, 100.0))}


def test_two_hands_assigned_one_to_one_not_both_to_the_closer_player():
    players = {1: player([(100.0, 100.0)]), 2: player([(900.0, 100.0)])}
    hands = [("FIRE", (110.0, 100.0)), ("WATER", (890.0, 100.0))]
    out = assign(hands, players, frame_w=1280)
    assert out == {1: ("FIRE", (110.0, 100.0)), 2: ("WATER", (890.0, 100.0))}


def test_hand_far_from_every_wrist_is_unassigned():
    players = {1: player([(100.0, 100.0)])}
    out = assign([("FIRE", (900.0, 100.0))], players, frame_w=1280)
    assert out == {}


def test_absent_player_gets_no_hand():
    players = {1: player([(100.0, 100.0)], present=False)}
    out = assign([("FIRE", (100.0, 100.0))], players, frame_w=1280)
    assert out == {}


# ------------------------------------------------------------------ call counts (at most once per model per frame)
class FakeTracker:
    """Pose model stand-in: counts calls, returns one present player with a wrist and a head."""

    def __init__(self):
        self.calls = 0

    def update(self, frame, t_ms):
        self.calls += 1
        return {1: Player(present=True, wrists=[(50.0, 50.0)], head=(50.0, 20.0))}

    def game_input(self, pid, gesture=None, hand_xy=None):
        from game_logic import PlayerInput
        p = {1: (50.0, 50.0)}[pid]
        return PlayerInput(True, gesture, hand_xy or p, (50.0, 20.0))


class FakeHands:
    def __init__(self):
        self.calls = 0

    def update(self, frame, players, t_ms):
        self.calls += 1
        return {1: ("FIRE", (50.0, 50.0))} if 1 in players else {}


class FakeFaces:
    """Face model stand-in: counts calls, returns no tracks (most tests don't need any)."""

    def __init__(self):
        self.calls = 0

    def update(self, frame, dt, t_s):
        self.calls += 1
        return []

    def assign_to_players(self, tracks, players, frame_w=1280):
        return {}

    def draw(self, frame, assigned, snap, events, t_s, **kwargs):
        pass

    def close(self):
        pass


def test_vision_calls_each_model_at_most_once_per_frame():
    tracker, hands, faces = FakeTracker(), FakeHands(), FakeFaces()
    vision = Vision(tracker=tracker, hands=hands, faces=faces)
    for n in range(5):
        vision.process(frame=FakeFrame(), t_ms=n * 33.0)
    assert tracker.calls == 5
    assert hands.calls == 5
    assert faces.calls == 5


def test_vision_does_not_rerun_hands_or_faces_on_the_same_camera_frame_number():
    # a faster render loop than the camera's own FPS sees the same (frame, n) more than once
    tracker, hands, faces = FakeTracker(), FakeHands(), FakeFaces()
    vision = Vision(tracker=tracker, hands=hands, faces=faces)
    vision.process(frame=FakeFrame(), t_ms=0.0, n=7)
    vision.process(frame=FakeFrame(), t_ms=16.0, n=7)          # render tick, no new camera picture yet
    _, inputs, _ = vision.process(frame=FakeFrame(), t_ms=33.0, n=8)   # a real new camera frame
    assert hands.calls == 2                                  # not 3
    assert faces.calls == 2
    assert tracker.calls == 2                                # pose is skipped on the duplicate frame too
    assert inputs[1].gesture == "FIRE"                        # the cached result is still used meanwhile


def test_vision_player_input_contract():
    vision = Vision(tracker=FakeTracker(), hands=FakeHands(), faces=FakeFaces())
    players, inputs, faces = vision.process(frame=FakeFrame(), t_ms=0.0)
    assert set(players) == {1}
    assert faces == {}
    inp = inputs[1]
    assert inp.present is True
    assert inp.gesture == "FIRE"
    assert inp.hand_xy == (50.0, 50.0)
    assert inp.body_xy == (50.0, 20.0)


def test_vision_warm_up_builds_the_hand_model_eagerly():
    built = []

    class FakeHandAdapterCtor:
        def _ensure_model(self):
            built.append(True)

    vision = Vision(tracker=FakeTracker(), hands=FakeHandAdapterCtor(), faces=FakeFaces())
    assert built == []
    vision.warm_up()
    assert built == [True]


def test_vision_without_hands_falls_back_to_the_raised_wrist():
    vision = Vision(tracker=FakeTracker(), use_hands=False, faces=FakeFaces())
    _, inputs, _ = vision.process(frame=FakeFrame(), t_ms=0.0)
    assert inputs[1].gesture is None
    assert inputs[1].hand_xy == (50.0, 50.0)                   # PlayerTracker.game_input's own fallback


def test_vision_without_faces_returns_an_empty_face_dict():
    vision = Vision(tracker=FakeTracker(), hands=FakeHands(), use_faces=False)
    _, _, faces = vision.process(frame=FakeFrame(), t_ms=0.0)
    assert faces == {}


# ------------------------------------------------------------------ face tracks -> players
def test_face_track_assigned_to_the_nearest_head():
    from face_module import assign_to_players

    class FakeTrack:
        def __init__(self, cx, cy):
            self.kf = type("KF", (), {"center": (cx, cy)})()

    players = {1: player([], present=True), 2: player([], present=True)}
    players[1].head, players[2].head = (100.0, 100.0), (900.0, 100.0)
    tracks = [FakeTrack(110.0, 100.0), FakeTrack(890.0, 100.0)]
    out = assign_to_players(tracks, players, frame_w=1280)
    assert out == {1: tracks[0], 2: tracks[1]}


# ------------------------------------------------------------------ import hygiene
def test_vision_module_does_not_import_mediapipe_or_ultralytics_at_import_time():
    here = __file__.rsplit("/", 1)[0]
    code = (f"import sys; sys.path.insert(0, {here!r}); import vision; "
            "print('mediapipe' in sys.modules or 'ultralytics' in sys.modules)")
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


def test_hand_adapter_close_is_safe_before_any_model_is_built():
    HandAdapter().close()                                      # no model constructed yet: must not raise


def test_hand_adapter_skips_the_model_when_no_player_is_present():
    class FakeModel:
        def __init__(self):
            self.calls = 0

        def detect(self, frame, t_ms):
            self.calls += 1
            return []

    model = FakeModel()
    adapter = HandAdapter(model=model)
    out = adapter.update(frame=object(), players={1: Player(present=False)}, t_ms=0.0)
    assert out == {}
    assert model.calls == 0


def test_play_camera_help_works_without_a_camera():
    out = subprocess.run([sys.executable, "play_camera.py", "--help"],
                         cwd=__file__.rsplit("/", 1)[0], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    assert "camera" in out.stdout.lower()


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, f in tests:
        f()
        print(f"ok   {name}")
    print(f"\n{len(tests)} tests passed")
