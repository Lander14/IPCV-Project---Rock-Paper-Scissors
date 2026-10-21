"""Rock-paper-scissors duel - game mechanics (task T4: player interaction and game control).

A real-time fight in rounds. Between the two players there are THREE LANES (top, middle, bottom):
  - a stable gesture (rock, paper or scissors) launches a spell into the lane at the height of the casting hand;
    it travels slowly along that lane towards the opponent (about 3.5 s)
  - two spells that meet in the same lane follow rock-paper-scissors: the winner goes on, the loser is destroyed,
    equal spells destroy each other. Spells in different lanes pass each other.
  - a spell that reaches the end of its lane hits the opponent, unless the opponent holds a shield in that lane
  - mana and a cooldown stop spamming; to cast again you first have to change the gesture
A round ends at K.O. (or when the round time runs out). Best of 3 rounds wins the match.

No image processing in this file: every frame the vision side says, per player ID, whether the player is in view,
which gesture they show, where the casting hand is (its height chooses the lane) and where the player is.

    game = Game([1, 2], frame_size=(1280, 720))
    every frame:
        events = game.update(t_ms, {1: PlayerInput(True, "ROCK", hand_xy=(400, 390), body_xy=(260, 300)), ...})
        state = game.snapshot()

Phases:  LOBBY -> ROUND_START (3-2-1) -> FIGHT -> ROUND_END -> ROUND_START ... -> MATCH_OVER -> (open palm) -> ...
         PAUSED when a player leaves the view during ROUND_START / FIGHT: everything freezes, then RESUME (3-2-1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# ------------------------------------------------------------------ actions and rules
ROCK, PAPER, SCISSORS = "ROCK", "PAPER", "SCISSORS"
SHIELD, POTION = "SHIELD", "POTION"
SPELLS = (ROCK, PAPER, SCISSORS)
ACTIONS = SPELLS + (SHIELD, POTION)

BEATS = {PAPER: ROCK, SCISSORS: PAPER, ROCK: SCISSORS}          # spell -> the spell it beats
WHY = {PAPER: "Paper wraps Rock", SCISSORS: "Scissors cut Paper", ROCK: "Rock smashes Scissors"}

# names the gesture module may send -> game action. Anything else ("?", None, typos) counts as no action.
GESTURE_TO_ACTION = {
    "ROCK": ROCK, "PAPER": PAPER, "SCISSORS": SCISSORS, "SHIELD": SHIELD, "POTION": POTION,
    "rock": ROCK, "paper": PAPER, "scissors": SCISSORS,
    "FIRE": ROCK, "WATER": PAPER, "LIGHTNING": SCISSORS,          # names of the first hand script
}
READY_GESTURES = {"PAPER", "paper", "WATER"}   # open palm = "I'm ready" in the lobby and after a match

LOBBY, ROUND_START, FIGHT, ROUND_END, PAUSED, RESUME, MATCH_OVER = (
    "LOBBY", "ROUND_START", "FIGHT", "ROUND_END", "PAUSED", "RESUME", "MATCH_OVER")


@dataclass
class Config:
    """Every number of the game in one place (thresholds, cooldowns, timing: T4-05)."""
    max_hp: int = 100
    spell_damage: int = 25           # 4 hits = K.O.
    lanes: tuple = (0.36, 0.54, 0.72)   # height of the 3 lanes (x frame height); the hand height picks one
    lane_half: float = 0.08          # half thickness of a lane (x frame height), for drawing
    spell_speed: float = 0.16        # x frame width per second: ~3.5 s from one player to the other
    max_mana: float = 100
    mana_regen: float = 15           # mana per second
    spell_cost: float = 30
    shield_cost: float = 30
    cast_cooldown_ms: int = 800      # minimum time between two spells of the same player
    shield_max_ms: int = 2000        # a shield lasts at most this long ...
    shield_cooldown_ms: int = 4000   # ... and then needs this long to come back
    potion_heal: int = 20
    potions_per_round: int = 1
    stable_frames: int = 4           # a gesture counts after this many frames in a row
    round_ms: int = 90000            # time limit of a round (0 = none)
    rounds_to_win: int = 2           # best of 3
    ready_ms: int = 1000             # everyone holds an open palm this long to start
    countdown_ms: int = 3000         # 3 - 2 - 1 before a round and after a pause
    round_end_ms: int = 3000         # "K.O." screen
    match_over_ms: int = 3000        # winner screen at least this long before a restart is accepted
    lost_ms: int = 400               # out of view longer than this -> pause


@dataclass
class PlayerInput:
    """What the vision modules report for one player in one frame (T3: ID and position, T2: gesture + hand)."""
    present: bool = False                        # the player is tracked in this frame
    gesture: str | None = None                   # name from the gesture module; None = no clear gesture
    hand_xy: tuple[float, float] | None = None   # casting hand, pixels: its height chooses the lane
    body_xy: tuple[float, float] | None = None   # head / torso centre, pixels: where the player's lanes end


@dataclass
class Spell:
    sid: int
    owner: int
    target: int
    action: str
    lane: int
    x: float                                     # position along the lane (pixels)
    dir: int                                     # +1 = to the right, -1 = to the left
    t0: float                                    # fight-clock time of the launch


@dataclass
class PlayerState:
    pid: int
    hp: int = 0
    mana: float = 0.0
    potions: int = 0
    rounds_won: int = 0
    matches_won: int = 0
    present: bool = False
    gone_since: float | None = None
    hand_xy: tuple[float, float] | None = None
    body_xy: tuple[float, float] | None = None
    gesture: str | None = None           # raw gesture of the last frame
    run: str | None = None               # action seen in the last frames ...
    run_len: int = 0                     # ... for this many frames in a row
    stable: str | None = None            # current stable action (None = no clear gesture)
    armed: str | None = None             # stable action not used yet: a new gesture = a new command
    said: bool = False                   # "no mana" / "recharging" already reported for this gesture
    last_cast: float = -1e9              # fight-clock time of the last spell
    shield_up: bool = False
    shield_since: float = 0.0
    shield_ready_at: float = 0.0         # fight-clock time when the shield can be raised again

    @property
    def alive(self) -> bool:
        return self.hp > 0


# ------------------------------------------------------------------ the game
class Game:
    """State machine of one match. Made for 2 players (with more, a spell goes to the nearest opponent on the side
    of the casting hand). All timing uses the frame time t_ms given by the caller, so the game is deterministic and
    can be tested without a camera. The fight clock `gt` only runs during FIGHT: a pause freezes the spells in their
    lanes, the cooldowns, the mana and the round time."""

    def __init__(self, player_ids=(1, 2), config: Config | None = None, frame_size=(1280, 720)):
        if len(player_ids) < 2:
            raise ValueError("a duel needs at least two players")
        self.cfg = c = config or Config()
        self.ids = list(player_ids)
        self.w, self.h = float(frame_size[0]), float(frame_size[1])
        self.lane_y = [f * self.h for f in c.lanes]
        self.players = {pid: PlayerState(pid, c.max_hp, c.max_mana, c.potions_per_round) for pid in self.ids}
        self.phase, self.t_phase, self.t, self.t_last = LOBBY, 0.0, 0.0, None
        self.gt, self.round, self.round_gt0 = 0.0, 0, 0.0
        self.spells: list[Spell] = []
        self._sid = 0
        self.ready_since, self.lost, self.paused_from, self._tick = None, [], None, None
        self.round_winner, self.end_reason, self.match_winner = None, None, None

    def lane_of(self, y):
        """Lane index (0 = top) for a height in pixels: the nearest lane."""
        return min(range(len(self.lane_y)), key=lambda i: abs(self.lane_y[i] - y))

    # ---------------------------------------------------------------- main entry, once per frame
    def update(self, t_ms: float, inputs: dict) -> list[dict]:
        """Advance the game to time t_ms with this frame's inputs {pid: PlayerInput}. Returns this frame's events."""
        cfg = self.cfg
        dt = 0.0 if self.t_last is None else min(max(t_ms - self.t_last, 0.0), 100.0)   # a frozen camera can't jump
        self.t = self.t_last = t_ms
        ev: list[dict] = []
        inputs = {pid: inputs.get(pid) or PlayerInput() for pid in self.ids}         # missing = not in view
        self._presence(t_ms, inputs)
        self._read_gestures(inputs, ev)
        el = t_ms - self.t_phase
        lost = self._lost_players(t_ms)

        if self.phase == LOBBY or (self.phase == MATCH_OVER and el >= cfg.match_over_ms):
            if self._everyone_ready(t_ms):
                if self.phase == MATCH_OVER:
                    self._new_match()
                self._start_round(t_ms, ev)
        elif self.phase in (ROUND_START, FIGHT, RESUME) and lost:
            self.lost, self.paused_from = lost, self.phase
            self._set(PAUSED, t_ms, ev, lost=lost)
        elif self.phase == PAUSED:
            if all(self.players[pid].present for pid in self._alive_ids()):
                self._set(ROUND_START if self.paused_from == ROUND_START else RESUME, t_ms, ev)
        elif self.phase in (ROUND_START, RESUME):
            if el >= cfg.countdown_ms:
                for p in self.players.values():
                    p.armed = None                   # gestures held during the countdown do not fire
                self._set(FIGHT, t_ms, ev)
            else:
                tick = max(1, math.ceil((cfg.countdown_ms - el) / 1000))
                if tick != self._tick:
                    self._tick = tick
                    ev.append(self._event("tick", value=tick))
        elif self.phase == FIGHT:
            self.gt += dt
            self._fight(dt, ev)
            alive = self._alive_ids()
            if len(alive) <= 1:
                self._end_round(t_ms, ev, alive[0] if alive else None, "K.O.")
            elif cfg.round_ms and self.gt - self.round_gt0 >= cfg.round_ms:
                hp = sorted(((self.players[pid].hp, pid) for pid in alive), reverse=True)
                self._end_round(t_ms, ev, hp[0][1] if hp[0][0] > hp[1][0] else None, "TIME")
        elif self.phase == ROUND_END and el >= cfg.round_end_ms:
            best = max(self.players.values(), key=lambda p: p.rounds_won)
            if best.rounds_won >= cfg.rounds_to_win:
                self.match_winner = best.pid
                best.matches_won += 1
                self._set(MATCH_OVER, t_ms, ev, winner=best.pid)
            else:
                self._start_round(t_ms, ev)
        return ev

    # ---------------------------------------------------------------- what the HUD / other modules read
    def snapshot(self) -> dict:
        cfg, el = self.cfg, self.t - self.t_phase
        return {
            "phase": self.phase, "round": self.round, "phase_ms": el,
            "countdown": max(1, math.ceil((cfg.countdown_ms - el) / 1000)) if self.phase in (ROUND_START, RESUME) else None,
            "round_left_s": max(0.0, (cfg.round_ms - (self.gt - self.round_gt0)) / 1000) if cfg.round_ms else None,
            "ready_progress": 0.0 if self.ready_since is None else min(1.0, (self.t - self.ready_since) / cfg.ready_ms),
            "lost": list(self.lost) if self.phase == PAUSED else [],
            "round_winner": self.round_winner, "end_reason": self.end_reason, "match_winner": self.match_winner,
            "rounds_to_win": cfg.rounds_to_win,
            "lanes": list(self.lane_y), "lane_half": cfg.lane_half * self.h,
            "spell_speed": cfg.spell_speed * self.w,                       # pixels per second
            "spells": [{"id": s.sid, "owner": s.owner, "target": s.target, "action": s.action, "lane": s.lane,
                        "xy": (s.x, self.lane_y[s.lane]), "dir": s.dir, "age_ms": self.gt - s.t0} for s in self.spells],
            "players": {pid: {
                "hp": p.hp, "max_hp": cfg.max_hp, "mana": p.mana, "max_mana": cfg.max_mana, "alive": p.alive,
                "present": p.present, "ready": p.present and p.gesture in READY_GESTURES, "gesture": p.stable,
                "lane": self.lane_of(p.hand_xy[1]) if p.hand_xy else None,
                "shield_up": p.shield_up, "shield_in_s": max(0.0, (p.shield_ready_at - self.gt) / 1000),
                "potions": p.potions, "rounds_won": p.rounds_won, "matches_won": p.matches_won,
                "cooldown": max(0.0, 1 - (self.gt - p.last_cast) / cfg.cast_cooldown_ms),
                "hand_xy": p.hand_xy, "body_xy": p.body_xy,
            } for pid, p in self.players.items()},
        }

    # ---------------------------------------------------------------- internals: bookkeeping
    def _event(self, kind, **kw):
        return {"type": kind, "t": self.t, "round": self.round, **kw}

    def _set(self, phase, t, ev, **kw):
        self.phase, self.t_phase, self._tick = phase, t, None
        ev.append(self._event("phase", phase=phase, **kw))

    def _alive_ids(self):
        return [pid for pid in self.ids if self.players[pid].alive]

    def _presence(self, t, inputs):
        for pid, p in self.players.items():
            inp = inputs[pid]
            p.present = bool(inp.present)
            p.gesture = inp.gesture if p.present else None
            if p.present:
                p.gone_since = None
                if inp.body_xy is not None:
                    p.body_xy = tuple(inp.body_xy)
                if inp.hand_xy is not None:
                    p.hand_xy = tuple(inp.hand_xy)
            elif p.gone_since is None:
                p.gone_since = t

    def _lost_players(self, t):
        return [pid for pid in self._alive_ids()
                if self.players[pid].gone_since is not None and t - self.players[pid].gone_since > self.cfg.lost_ms]

    def _everyone_ready(self, t):
        if not all(p.present and p.gesture in READY_GESTURES for p in self.players.values()):
            self.ready_since = None
            return False
        if self.ready_since is None:
            self.ready_since = t
        return t - self.ready_since >= self.cfg.ready_ms

    def _new_match(self):
        for p in self.players.values():
            p.rounds_won = 0
        self.round, self.match_winner = 0, None

    def _start_round(self, t, ev):
        self.round += 1
        self.round_gt0, self.spells = self.gt, []
        self.ready_since, self.lost, self.round_winner, self.end_reason = None, [], None, None
        for p in self.players.values():
            p.hp, p.mana, p.potions = self.cfg.max_hp, self.cfg.max_mana, self.cfg.potions_per_round
            p.armed, p.said, p.last_cast = None, False, -1e9
            p.shield_up, p.shield_ready_at = False, self.gt
        self._set(ROUND_START, t, ev)

    def _end_round(self, t, ev, winner, reason):
        self.spells = []
        for p in self.players.values():
            p.shield_up = False
        self.round_winner, self.end_reason = winner, reason
        if winner is not None:
            self.players[winner].rounds_won += 1
        self._set(ROUND_END, t, ev, winner=winner, reason=reason)

    # ---------------------------------------------------------------- internals: gestures -> commands
    def _read_gestures(self, inputs, ev):
        """Confirmation rule (T4-04/05): a gesture counts after `stable_frames` equal frames. Only a *change* of the
        stable gesture is a new command, so holding a gesture never repeats it, and a flicker never triggers one."""
        for pid, p in self.players.items():
            a = GESTURE_TO_ACTION.get(inputs[pid].gesture) if p.present else None
            if a == p.run:
                p.run_len += 1
            else:
                p.run, p.run_len = a, 1
            if p.run_len == self.cfg.stable_frames and a != p.stable:
                p.stable, p.armed, p.said = a, a, False
                if a is not None and self.phase == FIGHT:
                    ev.append(self._event("aim", player=pid, action=a, xy=p.hand_xy))

    def _target(self, p):
        """The opponent a spell goes to: the only one, or the nearest one on the side of the casting hand."""
        others = [self.players[pid] for pid in self._alive_ids() if pid != p.pid and self.players[pid].body_xy]
        if not others or p.body_xy is None:
            return None
        side = 0.0 if p.hand_xy is None else p.hand_xy[0] - p.body_xy[0]
        facing = [o for o in others if (o.body_xy[0] - p.body_xy[0]) * side > 0] or others
        return min(facing, key=lambda o: abs(o.body_xy[0] - p.body_xy[0]))

    def _lane(self, p):
        return self.lane_of(p.hand_xy[1]) if p.hand_xy else len(self.lane_y) // 2

    def _fight(self, dt, ev):
        cfg, gt = self.cfg, self.gt
        for pid in self._alive_ids():
            p = self.players[pid]
            p.mana = min(cfg.max_mana, p.mana + cfg.mana_regen * dt / 1000)
            if p.shield_up and (p.stable != SHIELD or gt - p.shield_since >= cfg.shield_max_ms):
                p.shield_up, p.shield_ready_at = False, gt + cfg.shield_cooldown_ms
                ev.append(self._event("shield_down", player=pid))
            a = p.armed
            if a is None or not p.present:
                continue
            if a in SPELLS:
                if gt - p.last_cast < cfg.cast_cooldown_ms:
                    continue                                  # wait; fires when the cooldown is over
                if p.mana < cfg.spell_cost:
                    self._refuse(p, ev, "no mana")
                    continue
                target = self._target(p)
                if target is None:
                    continue
                lane = self._lane(p)
                x0 = p.hand_xy[0] if p.hand_xy else p.body_xy[0]
                self._sid += 1
                s = Spell(self._sid, pid, target.pid, a, lane, x0, 1 if target.body_xy[0] > p.body_xy[0] else -1, gt)
                self.spells.append(s)
                p.mana, p.last_cast, p.armed = p.mana - cfg.spell_cost, gt, None
                ev.append(self._event("cast", player=pid, action=a, spell=s.sid, target=target.pid, lane=lane,
                                      xy=(x0, self.lane_y[lane])))
            elif a == SHIELD:
                if gt < p.shield_ready_at:
                    self._refuse(p, ev, f"shield back in {(p.shield_ready_at - gt) / 1000:.1f} s")
                    continue
                if p.mana < cfg.shield_cost:
                    self._refuse(p, ev, "no mana")
                    continue
                p.shield_up, p.shield_since, p.mana, p.armed = True, gt, p.mana - cfg.shield_cost, None
                ev.append(self._event("shield_up", player=pid, lane=self._lane(p)))
            elif a == POTION:
                p.armed = None
                if p.potions <= 0:
                    ev.append(self._event("invalid", player=pid, action=a, reason="no potion left"))
                    continue
                heal = min(cfg.potion_heal, cfg.max_hp - p.hp)
                p.hp, p.potions = p.hp + heal, p.potions - 1
                ev.append(self._event("heal", player=pid, amount=heal))
        self._move_spells(dt, ev)

    def _refuse(self, p, ev, reason):
        """The player asks for something that is not possible right now: say it once per gesture."""
        if not p.said:
            p.said = True
            ev.append(self._event("invalid", player=p.pid, action=p.armed, reason=reason))

    # ---------------------------------------------------------------- internals: spells in the lanes
    def _move_spells(self, dt, ev):
        """Move every spell along its lane; spells that meet in a lane fight (T4-09); spells at the end hit."""
        cfg = self.cfg
        step = cfg.spell_speed * self.w * dt / 1000
        before = {s.sid: s.x for s in self.spells}
        for s in self.spells:
            s.x += s.dir * step
        gone = set()
        for i, a in enumerate(self.spells):
            for b in self.spells[i + 1:]:
                if a.lane != b.lane or a.owner == b.owner or a.dir == b.dir or a.sid in gone or b.sid in gone:
                    continue
                if (before[a.sid] - before[b.sid]) * (a.x - b.x) > 0:      # not crossed during this frame
                    continue
                xy = ((a.x + b.x) / 2, self.lane_y[a.lane])
                if a.action == b.action:
                    gone |= {a.sid, b.sid}
                    ev.append(self._event("clash", player=a.owner, target=b.owner, action=a.action, lane=a.lane, xy=xy))
                else:
                    win, lose = (a, b) if BEATS[a.action] == b.action else (b, a)
                    gone.add(lose.sid)
                    ev.append(self._event("counter", player=win.owner, target=lose.owner, action=win.action,
                                          beaten=lose.action, why=WHY[win.action], lane=a.lane, xy=xy))
        for s in self.spells:
            if s.sid in gone:
                continue
            t = self.players[s.target]
            if t.body_xy is None or (t.body_xy[0] - s.x) * s.dir > 0:      # not at the end of the lane yet
                continue
            gone.add(s.sid)
            if not t.alive:
                continue
            xy = (t.body_xy[0], self.lane_y[s.lane])
            if t.shield_up and self._lane(t) == s.lane:
                ev.append(self._event("blocked", player=s.owner, target=t.pid, action=s.action, lane=s.lane, xy=xy))
            else:
                t.hp = max(0, t.hp - cfg.spell_damage)
                ev.append(self._event("hit", player=s.owner, target=t.pid, action=s.action, lane=s.lane,
                                      amount=cfg.spell_damage, xy=xy))
                if not t.alive:
                    ev.append(self._event("ko", player=t.pid))
        self.spells = [s for s in self.spells if s.sid not in gone]
