"""Tests of the identity rules of player_tracker.py (no camera, no YOLO needed).   python test_player_tracker.py"""
from player_tracker import IdentityManager

L, R = (300.0, 360.0), (980.0, 360.0)              # where the two players stand (1280 px wide picture)


def test_new_tracker_id_keeps_the_player_at_once():
    m = IdentityManager(2, 1280)
    assert m.update([(1, L), (2, R)], 0) == {1: 1, 2: 2}
    m.update([(2, R)], 500)                          # player 1 hidden for a moment
    assert m.update([(7, (310.0, 360.0)), (2, R)], 1000) == {7: 1, 2: 2}      # new id -> still player 1, no gap


def test_left_is_player_1_even_if_the_right_one_is_seen_first():
    m = IdentityManager(2, 1280)
    assert m.update([(4, R)], 0) == {4: 2}
    assert m.update([(4, R), (5, L)], 100) == {4: 2, 5: 1}


def test_coming_back_after_a_long_time_keeps_the_identity():
    m = IdentityManager(2, 1280)
    m.update([(1, L), (2, R)], 0)
    m.update([(2, R)], 1000)
    assert m.update([(9, (400.0, 360.0)), (2, R)], 6000) == {9: 1, 2: 2}     # 6 s away, back on the left side


def test_single_player_leaving_and_coming_back_stays_the_same_player():
    m = IdentityManager(2, 1280)
    assert m.update([(1, L)], 0) == {1: 1}
    assert m.update([(9, (330.0, 360.0))], 3500) == {9: 1}                   # the teammate's version gave PLAYER 2


def test_spectator_is_ignored_while_both_players_are_there():
    m = IdentityManager(2, 1280)
    m.update([(1, L), (2, R)], 0)
    assert m.update([(1, L), (2, R), (3, (640.0, 300.0))], 100) == {1: 1, 2: 2}


def test_spectator_far_from_a_missing_player_does_not_take_his_place():
    m = IdentityManager(2, 1280)
    m.update([(1, L), (2, R)], 0)
    assert m.update([(2, R), (3, (1200.0, 300.0))], 800) == {2: 2}           # player 1 hidden; stranger far right


def test_both_new_ids_at_once_do_not_swap():
    m = IdentityManager(2, 1280)
    m.update([(1, L), (2, R)], 0)
    assert m.update([(11, (950.0, 360.0)), (10, (320.0, 360.0))], 700) == {11: 2, 10: 1}


def test_crossing_with_kept_tracker_ids_keeps_identities():
    m = IdentityManager(2, 1280)
    m.update([(1, L), (2, R)], 0)
    for k in range(1, 11):                           # they swap places slowly, the tracker keeps its ids
        x = 300 + 68 * k
        assert m.update([(1, (x, 360.0)), (2, (1280 - x, 360.0))], 100 * k) == {1: 1, 2: 2}


def test_solo_forces_the_first_detection_onto_player_1_regardless_of_side():
    m = IdentityManager(2, 1280, solo_pid=1)
    assert m.update([(1, R)], 0) == {1: 1}                                    # lone person on the right: still P1
    assert m.update([(1, (900.0, 360.0))], 100) == {1: 1}                     # and stays P1 as they move


def test_solo_a_second_person_is_ignored_not_treated_as_player_2():
    # solo play expects exactly one real player; a bystander should never take the bot's slot
    m = IdentityManager(2, 1280, solo_pid=1)
    m.update([(1, R)], 0)
    assert m.update([(1, R), (2, L)], 100) == {1: 1}


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, f in tests:
        f()
        print(f"ok   {name}")
    print(f"\n{len(tests)} tests passed")
