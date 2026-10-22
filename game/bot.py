"""Computer opponent for playing alone (single-player mode).

The bot plays through the same input as a person: every frame it says which gesture it shows and in which lane its
hand is. So it follows exactly the same rules (4 stable frames, change the gesture to cast again, cooldown, mana).
It reads the game state, answers incoming spells with a counter-spell in the same lane after a human-like
reaction time, and attacks at random moments in random lanes.

    bot = Bot(pid=2, level="normal")
    every frame:
        gesture, lane = bot.act(t_ms, game.snapshot())
        inputs[2] = PlayerInput(True, gesture, hand_xy_of(lane), body_xy)
"""
import random

from game_logic import BEATS, FIGHT, LOBBY, MATCH_OVER, PAPER, POTION, SPELLS

LEVELS = {   # chance to answer a spell, reaction time (s), time between own attacks (s)
    "easy":   {"react": 0.35, "delay": (0.9, 1.5), "attack": (2.8, 4.5)},
    "normal": {"react": 0.65, "delay": (0.6, 1.0), "attack": (2.0, 3.2)},
    "hard":   {"react": 0.9, "delay": (0.35, 0.6), "attack": (1.3, 2.2)},
}
COUNTER = {loser: winner for winner, loser in BEATS.items()}       # spell -> the spell that beats it
HOLD_MS, RELEASE_MS = 400, 500                                      # like a person: show, then drop the hand


class Bot:
    def __init__(self, pid, level="normal", seed=None):
        self.pid, self.lv = pid, LEVELS[level]
        self.rng = random.Random(seed)
        self.plan = []            # actions waiting: (start_ms, gesture, lane, hold_ms)
        self.doing = None         # (gesture, lane, end_ms) being shown now
        self.free_at = 0.0        # hand down until then (so the next gesture is a new command)
        self.lane = 1
        self.seen, self.next_attack, self.round, self.potion_tried = set(), None, None, False

    def act(self, t, snap):
        """Gesture shown in this frame (None = hand down) and the lane of the hand."""
        if snap["phase"] in (LOBBY, MATCH_OVER):
            return PAPER, 1                                              # always ready to play
        if snap["round"] != self.round:                                   # new round: forget the old plans
            self.round, self.plan, self.seen, self.next_attack, self.potion_tried = snap["round"], [], set(), None, False
        if snap["phase"] == FIGHT:
            self._think(t, snap)
        if self.doing and t >= self.doing[2]:                             # done: drop the hand for a moment
            self.doing, self.free_at = None, t + RELEASE_MS
        if self.doing is None and t >= self.free_at and self.plan and self.plan[0][0] <= t:
            _, g, lane, hold = self.plan.pop(0)
            self.doing, self.lane = (g, lane, t + hold), lane
        return (self.doing[0] if self.doing else None), self.lane

    def _later(self, start, gesture, lane, hold=HOLD_MS):
        self.plan.append((start, gesture, lane, hold))
        self.plan.sort(key=lambda a: a[0])

    def _think(self, t, snap):
        me = snap["players"][self.pid]
        body = me["body_xy"]
        speed = snap["spell_speed"]                                       # pixels per second
        for s in snap["spells"]:                                          # 1. defend against incoming spells
            if s["target"] != self.pid or s["owner"] == self.pid or s["id"] in self.seen:
                continue
            self.seen.add(s["id"])
            if body is None or self.rng.random() > self.lv["react"]:
                continue
            arrive = t + abs(body[0] - s["xy"][0]) / speed * 1000
            react = t + 1000 * self.rng.uniform(*self.lv["delay"])
            if react < arrive - 600:
                self._later(react, COUNTER[s["action"]], s["lane"])
        if self.next_attack is None:                                      # 2. attack now and then
            self.next_attack = t + 1000 * self.rng.uniform(*self.lv["attack"])
        elif t >= self.next_attack:
            self.next_attack = t + 1000 * self.rng.uniform(*self.lv["attack"])
            if me["mana"] >= 45 and not self.plan:
                self._later(t, self.rng.choice(SPELLS), self.rng.randrange(len(snap["lanes"])))
        if me["hp"] <= 50 and me["potions"] > 0 and not self.potion_tried:   # 3. drink when hurt
            self.potion_tried = True
            self._later(t + 300, POTION, self.lane)
