"""Tests of the face-tracking fusion (no camera needed; most need no MediaPipe either).

python test_face_module.py

A few tests need the real face_tracking_final.py + MediaPipe (requirements-vision.txt) and are skipped,
not failed, when those are not installed - matching the rest of the suite's "testable without the heavy
vision libraries" design.
"""
from face_module import assign_to_players
from player_tracker import Player


class FakeKF:
    def __init__(self, cx, cy, radius=60.0, roll=0.0):
        self.center, self.radius, self.roll = (cx, cy), radius, roll


class FakeTrack:
    def __init__(self, cx, cy, tid=1):
        self.id, self.kf, self.eyes = tid, FakeKF(cx, cy), (0.0, 0.0, 1.0)


def player(head, present=True):
    return Player(present=present, head=head)


# ------------------------------------------------------------------ face track -> player assignment
def test_face_assigned_to_the_nearest_head():
    players = {1: player((100.0, 100.0)), 2: player((900.0, 100.0))}
    tracks = [FakeTrack(110.0, 100.0, tid=5), FakeTrack(890.0, 100.0, tid=6)]
    out = assign_to_players(tracks, players, frame_w=1280)
    assert out == {1: tracks[0], 2: tracks[1]}


def test_two_faces_assigned_one_to_one_not_both_to_the_closer_player():
    # regardless of FaceTracker's own track ids, each player gets at most one face
    players = {1: player((100.0, 100.0)), 2: player((900.0, 100.0))}
    tracks = [FakeTrack(120.0, 100.0, tid=1), FakeTrack(140.0, 100.0, tid=2)]   # both near player 1
    out = assign_to_players(tracks, players, frame_w=1280)
    assert len(out) == 1 and 1 in out                           # only the nearer one is claimed; no double-grab


def test_face_far_from_every_head_is_unassigned():
    players = {1: player((100.0, 100.0))}
    out = assign_to_players([FakeTrack(1200.0, 100.0)], players, frame_w=1280)
    assert out == {}


def test_absent_player_gets_no_face():
    players = {1: player((100.0, 100.0), present=False)}
    out = assign_to_players([FakeTrack(100.0, 100.0)], players, frame_w=1280)
    assert out == {}


def test_player_with_no_head_yet_gets_no_face():
    players = {1: player(head=None)}
    out = assign_to_players([FakeTrack(100.0, 100.0)], players, frame_w=1280)
    assert out == {}


# ------------------------------------------------------------------ real-game HP bar wiring (needs the real module)
def _face_module_or_skip():
    try:
        from face_module import FaceModule
        return FaceModule()
    except ImportError:
        return None


def test_hp_bar_state_updates_on_a_hit_event_and_draws_without_crashing():
    faces = _face_module_or_skip()
    if faces is None:
        return                                                   # mediapipe not installed: skip, don't fail
    import numpy as np
    frame = np.zeros((480, 640, 3), dtype="uint8")
    assigned = {1: FakeTrack(100.0, 100.0, tid=7)}
    faces.glasses = {}                                            # no sprite needed for this check
    snap = {"players": {1: {"hp": 70, "max_hp": 100}}}
    faces.draw(frame, assigned, snap, events=[{"type": "hit", "target": 1}], t_s=1.0, glasses=False)
    assert faces._hp_state[1] == 1.0
    faces.close()


# ------------------------------------------------------------------ real-model regression: group-photo faces
def test_face_tracker_recovers_a_frontal_face_from_a_real_group_photo():
    """Reproduces a measured limitation: MediaPipe's face detector finds nothing on the *full* 1280-wide
    photo in one shot (faces are small relative to the frame), but FaceTracker's own half-frame search
    (SEARCH_EVERY, unmodified) recovers the frontal face within a handful of calls. The profile face in
    the same photo is never confirmed - a real, not-fixed-here limitation; see game/README.md."""
    try:
        import cv2
        from face_module import FaceModule
        faces = FaceModule()
    except ImportError:
        return                                                   # mediapipe/cv2 not installed: skip
    from pathlib import Path
    img_path = Path(__file__).resolve().parent.parent / "hand_detection" / "samples" / "05_high_five_two_people.jpg"
    img = cv2.imread(str(img_path))
    assert img is not None
    confirmed = []
    for i in range(12):
        tracks = faces.update(img, dt=1 / 30, t_s=i / 30)
        confirmed = [t.id for t in tracks if t.confirmed]
    assert len(confirmed) == 1, f"expected exactly one confirmed (frontal) face, got {confirmed}"
    faces.close()


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, f in tests:
        f()
        print(f"ok   {name}")
    print(f"\n{len(tests)} tests passed")
