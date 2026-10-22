"""Tests of the game mechanics, without camera.   Run:  python test_game_logic.py   (or: pytest)"""
from bot import Bot
from game_logic import (BEATS, FIGHT, LOBBY, MATCH_OVER, PAPER, PAUSED, POTION, RESUME, ROCK, ROUND_END,
                        ROUND_START, SCISSORS, SPELLS, Config, Game, PlayerInput)

FPS = 30
W, H = 1280, 720
BODY = {1: (256.0, 290.0), 2: (1024.0, 290.0), 3: (640.0, 290.0)}
HAND_X = {1: 384.0, 2: 896.0, 3: 700.0}


class Sim:
    """Feeds the game frame by frame at 30 fps, like the camera loop would."""

    def __init__(self, ids=(1, 2), bots=None, **cfg):
        self.game, self.t, self.events = Game(ids, Config(**cfg), (W, H)), 0.0, []
        self.bots = bots or {}                          # {pid: Bot}: the computer plays these players
        self.lane = {pid: 1 for pid in ids}             # which lane each hand is at (0 top, 1 middle, 2 bottom)
        self.absent = set()

    def hand(self, pid):
        return HAND_X[pid], self.game.lane_y[self.lane[pid]]

    def step(self, gestures=None, n=1):
        for _ in range(n):
            g = dict(gestures or {})
            if self.bots:
                snap = self.game.snapshot()
                for pid, bot in self.bots.items():
                    g[pid], self.lane[pid] = bot.act(self.t, snap)
            inputs = {pid: PlayerInput(pid not in self.absent, g.get(pid), self.hand(pid), BODY[pid])
                      for pid in self.game.ids}
            self.events += self.game.update(self.t, inputs)
            self.t += 1000 / FPS

    def secs(self, s, gestures=None):
        self.step(gestures, n=round(s * FPS))

    def until(self, phase, gestures=None, limit_s=40):
        for _ in range(int(limit_s * FPS)):
            if self.game.phase == phase:
                return
            self.step(gestures)
        raise AssertionError(f"never reached {phase}, stuck in {self.game.phase}")

    def fight(self):
        """Lobby -> both ready -> countdown -> FIGHT."""
        self.until(ROUND_START, {pid: PAPER for pid in self.game.ids})
        self.until(FIGHT)

    def tap(self, pid, gesture, s=0.4, lane=None):
        """Show a gesture for s seconds with the hand at a lane."""
        if lane is not None:
            self.lane[pid] = lane
        self.secs(s, {pid: gesture})

    def count(self, kind, **kw):
        return sum(1 for e in self.events if e["type"] == kind and all(e.get(k) == v for k, v in kw.items()))

    def hp(self):
        return {pid: p.hp for pid, p in self.game.players.items()}


# ------------------------------------------------------------------ rules and inputs
def test_rules_are_a_cycle():
    for s in SPELLS:
        assert list(BEATS.values()).count(s) == 1, s
    assert BEATS[PAPER] == ROCK and BEATS[SCISSORS] == PAPER and BEATS[ROCK] == SCISSORS


def test_old_gesture_names_still_work():
    s = Sim()
    s.fight()
    s.tap(1, "FIRE")                                  # the first hand script says FIRE for a fist
    assert s.count("cast", player=1, action=ROCK) == 1


def test_lobby_needs_everyone_ready_then_countdown():
    s = Sim()
    s.secs(2, {1: PAPER})
    assert s.game.phase == LOBBY
    s.secs(1.2, {1: PAPER, 2: PAPER})
    assert s.game.phase == ROUND_START and s.game.round == 1
    s.secs(3.1)
    assert s.game.phase == FIGHT


def test_gesture_held_in_countdown_does_not_fire():
    s = Sim()
    s.until(ROUND_START, {1: PAPER, 2: PAPER})
    s.until(FIGHT, {1: PAPER, 2: PAPER})
    s.secs(2, {1: PAPER, 2: PAPER})
    assert s.count("cast") == 0


# ------------------------------------------------------------------ lanes
def test_hand_height_chooses_the_lane():
    s = Sim()
    s.fight()
    s.tap(1, ROCK, lane=0)
    s.tap(1, None, 0.6)
    s.tap(1, SCISSORS, lane=2)
    assert [e["lane"] for e in s.events if e["type"] == "cast"] == [0, 2]


def test_spells_are_slow():
    s = Sim()
    s.fight()
    s.tap(1, ROCK, 0.5)
    s.secs(1.5)
    assert s.hp()[2] == 100 and len(s.game.spells) == 1       # still travelling after 2 s
    s.secs(2.0)
    assert s.hp() == {1: 100, 2: 75} and s.count("hit", target=2) == 1


def test_counter_in_the_same_lane():
    s = Sim()
    s.fight()
    s.tap(1, ROCK, 0.3, lane=1)
    s.tap(2, PAPER, 0.3, lane=1)                     # paper wraps rock and goes on
    s.secs(5)
    assert s.count("counter", player=2, beaten=ROCK) == 1 and s.hp() == {1: 75, 2: 100}


def test_spells_in_different_lanes_pass_each_other():
    s = Sim()
    s.fight()
    s.tap(1, ROCK, 0.3, lane=0)
    s.tap(2, PAPER, 0.3, lane=2)
    s.secs(5)
    assert s.count("counter") == 0 and s.hp() == {1: 75, 2: 75}


def test_same_spells_cancel_out():
    s = Sim()
    s.fight()
    s.secs(0.3, {1: SCISSORS, 2: SCISSORS})
    s.secs(5)
    assert s.count("clash") == 1 and s.hp() == {1: 100, 2: 100}


def test_shield_gesture_is_now_ignored():
    # dropped (22 Oct 2026 team decision): a "SHIELD" gesture is unknown input, exactly like a typo or "?"
    s = Sim()
    s.fight()
    s.tap(1, "SHIELD", 0.3, lane=2)
    s.secs(2.0)
    assert s.count("cast") == 0 and s.count("invalid") == 0 and s.hp() == {1: 100, 2: 100}


def test_countering_in_the_same_lane_is_the_only_defence():
    # same mechanic as test_counter_in_the_same_lane, restated here since it now has no alternative
    s = Sim()
    s.fight()
    s.tap(1, ROCK, 0.3, lane=2)
    s.secs(2.0)
    s.tap(2, PAPER, 0.3, lane=2)                     # paper wraps the incoming rock: no hit, paper flies on
    s.secs(5)
    assert s.count("counter", player=2, beaten=ROCK) == 1 and s.count("hit", target=2) == 0


# ------------------------------------------------------------------ anti-spam
def test_holding_a_gesture_casts_only_once():
    s = Sim()
    s.fight()
    s.secs(4, {1: ROCK})
    assert s.count("cast", player=1) == 1


def test_flicker_does_not_cast():
    s = Sim()
    s.fight()
    for _ in range(10):
        s.step({1: ROCK}, n=2)                       # 2 frames of ROCK, 2 of nothing: never stable
        s.step(n=2)
    assert s.count("cast") == 0


def test_release_and_cast_again_after_cooldown():
    s = Sim()
    s.fight()
    s.tap(1, ROCK, 0.3)
    s.tap(1, None, 0.2)
    s.tap(1, ROCK, 0.3)                              # 0.5 s after the first: cooldown (0.8 s) not over ...
    assert s.count("cast", player=1) == 1
    s.tap(1, ROCK, 0.5)                              # ... same held gesture: it fires when the cooldown ends
    assert s.count("cast", player=1) == 2


def test_mana_runs_out_and_comes_back():
    s = Sim(mana_regen=0)
    s.fight()
    for g in (ROCK, PAPER, SCISSORS, ROCK):
        s.tap(1, g, 0.9)
    assert s.count("cast", player=1) == 3 and s.count("invalid", player=1, reason="no mana") == 1
    s.game.cfg.mana_regen = 15
    s.secs(2)
    s.tap(1, PAPER, 0.5)
    assert s.count("cast", player=1) == 4


def test_potion_once_per_round():
    s = Sim()
    s.fight()
    s.tap(1, ROCK)
    s.secs(3.5)
    s.tap(2, POTION)
    assert s.hp()[2] == 95
    s.tap(2, None)
    s.tap(2, POTION)
    assert s.count("invalid", player=2, reason="no potion left") == 1


# ------------------------------------------------------------------ pause, rounds, robustness
def test_pause_freezes_the_spells_in_their_lanes():
    s = Sim()
    s.fight()
    s.tap(1, ROCK, 0.3)
    s.secs(1.0)
    s.absent = {2}
    s.secs(1.0)                                      # P2 leaves while the rock travels
    assert s.game.phase == PAUSED
    x = s.game.spells[0].x
    s.secs(3.0)
    assert s.game.phase == PAUSED and s.game.spells[0].x == x
    s.absent = set()
    s.step()
    assert s.game.phase == RESUME
    s.until(FIGHT)
    s.secs(3.0)
    assert s.hp()[2] == 75                           # the rock continues and lands


def test_rounds_and_match():
    s = Sim()
    s.fight()
    for g in (ROCK, PAPER, SCISSORS, ROCK):           # 4 hits = K.O.
        s.tap(1, g, 1.0)
        s.tap(1, None, 0.2)
    s.secs(4)
    assert s.game.phase == ROUND_END and s.game.round_winner == 1 and s.game.players[1].rounds_won == 1
    s.until(FIGHT)                                    # round 2: HP back to 100
    assert s.game.round == 2 and s.hp() == {1: 100, 2: 100}
    for g in (ROCK, PAPER, SCISSORS, ROCK):
        s.tap(1, g, 1.0)
        s.tap(1, None, 0.2)
    s.until(MATCH_OVER)
    assert s.game.match_winner == 1 and s.game.players[1].matches_won == 1
    s.secs(1.5, {1: PAPER, 2: PAPER})                # too early: winner screen stays 3 s
    assert s.game.phase == MATCH_OVER
    s.until(ROUND_START, {1: PAPER, 2: PAPER})
    assert s.game.round == 1 and s.game.players[1].rounds_won == 0


def test_time_limit_more_hp_wins():
    s = Sim(round_ms=6000)
    s.fight()
    s.tap(1, ROCK)
    s.until(ROUND_END)
    assert s.game.end_reason == "TIME" and s.game.round_winner == 1


def test_missing_input_counts_as_out_of_view():
    g = Game()
    g.update(0, {1: PlayerInput(True, PAPER, None, (300, 300))})
    assert g.players[2].present is False and g.phase == LOBBY



# ------------------------------------------------------------------ single-player mode (the bot)
def test_bot_beats_a_player_who_does_nothing():
    s = Sim(bots={2: Bot(2, "normal", seed=1)})
    s.until(ROUND_START, {1: PAPER})                 # the bot is always ready
    s.until(MATCH_OVER, limit_s=300)
    assert s.game.match_winner == 2 and s.count("cast", player=2) >= 8


def test_bot_answers_incoming_spells():
    s = Sim(bots={2: Bot(2, "hard", seed=3)})
    s.fight()
    for i in range(8):
        s.tap(1, ROCK, 0.4, lane=i % 3)
        s.secs(1.8)
    s.secs(4)
    answered = s.count("counter", player=2)
    assert answered >= 5, answered                   # hard: answers most of the 8 rocks


def test_two_bots_play_a_whole_match():
    s = Sim(bots={1: Bot(1, "normal", seed=1), 2: Bot(2, "normal", seed=2)})
    s.until(MATCH_OVER, limit_s=400)
    assert s.game.match_winner in (1, 2)
    assert s.count("counter") + s.count("clash") > 0


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, f in tests:
        f()
        print(f"ok   {name}")
    print(f"\n{len(tests)} tests passed")
