"""Plays a scripted match (no camera, no keyboard) over a real background picture and saves it as a video,
to show every mechanic to the team.

    python simulate.py                                    -> simulated_game.mp4 (+ the event log)
    python simulate.py --background my_webcam_clip.mp4    # any picture or video as the "camera"
"""
import argparse
from pathlib import Path

import cv2

from game_logic import FIGHT, LOBBY, MATCH_OVER, PAPER, POTION, ROCK, SCISSORS, SHIELD, Config, Game, PlayerInput
from hud import Hud, shade
from play_keyboard import VirtualPlayers, log

FPS = 30
TOP, MID, LOW = 0, 1, 2
# per round: (seconds of fight time, player, gesture or "leave", lane of the hand, how long in s; default 0.4)
SCRIPT = {
    1: [(0.5, 1, ROCK, MID),                          # rock rolls down the middle lane       -> P2 75
        (2.0, 2, SCISSORS, TOP), (2.6, 1, ROCK, TOP),  # rock smashes the scissors, goes on    -> P2 50
        (4.0, 2, PAPER, LOW), (6.5, 1, SHIELD, LOW, 1.5),   # P1 shields the bottom lane in time: blocked
        (8.5, 1, SCISSORS, MID), (9.0, 2, SCISSORS, MID),   # scissors meet scissors: both gone
        (10.0, 2, POTION, MID),                       # P2 +20                                -> P2 70
        (10.5, 1, PAPER, TOP), (11.5, 1, ROCK, LOW), (13.0, 1, SCISSORS, MID)],   # -> 45, 20, 0: K.O.
    2: [(0.5, 2, PAPER, MID),                         #                                       -> P1 75
        (1.5, 2, ROCK, TOP), (2.5, 1, "leave", MID, 1.5),   # P1 leaves: pause, the rock waits in its lane -> P1 50
        (5.0, 2, SCISSORS, LOW),                      #                                       -> P1 25
        (6.0, 2, PAPER, MID), (6.5, 1, SCISSORS, MID),      # scissors cut the paper, go on        -> P2 75
        (7.0, 2, ROCK, TOP)],                         #                                       -> P1 0: K.O.
    3: [(0.5, 1, ROCK, TOP), (0.8, 2, SCISSORS, TOP),  # counter                               -> P2 75
        (1.5, 1, PAPER, MID), (4.0, 2, SHIELD, MID, 1.5),   # blocked
        (2.5, 1, SCISSORS, LOW), (3.5, 1, ROCK, MID), (4.5, 1, PAPER, LOW)],     # -> 50, 25, 0: P1 wins
}


class Background:
    def __init__(self, path, w, h):
        self.w, self.h = w, h
        self.cap = cv2.VideoCapture(str(path)) if Path(path).suffix.lower() in (".mp4", ".avi", ".mov", ".mkv") else None
        self.img = None if self.cap else cv2.resize(cv2.imread(str(path)), (w, h))

    def frame(self):
        if self.cap is None:
            return self.img.copy()
        ok, f = self.cap.read()
        if not ok:                                                     # loop the clip
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, f = self.cap.read()
        return cv2.resize(f, (self.w, self.h))


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="simulated_game.mp4")
    ap.add_argument("--size", default="960x540")
    ap.add_argument("--background", default=str(here / "assets" / "demo_background.jpg"))
    ap.add_argument("--dim", type=float, default=0.15)
    args = ap.parse_args()
    W, H = map(int, args.size.split("x"))
    cfg = Config()
    game, hud, bg = Game([1, 2], cfg, (W, H)), Hud(), Background(args.background, W, H)
    vp = VirtualPlayers(W, H, cfg.lanes)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    t, end_at, away_until, left = 0.0, None, {1: -1.0, 2: -1.0}, set()
    while end_at is None or t < end_at:
        gestures = {1: None, 2: None}
        if game.phase == LOBBY and t > 500:
            gestures = {1: PAPER, 2: PAPER}
        elif game.phase == FIGHT:
            rel = (game.gt - game.round_gt0) / 1000
            for item in SCRIPT.get(game.round, []):
                t0, pid, what, lane = item[:4]
                dur = item[4] if len(item) > 4 else 0.4
                if what == "leave":
                    if rel >= t0 and (game.round, t0) not in left:
                        left.add((game.round, t0))
                        away_until[pid] = t + dur * 1000
                elif t0 <= rel < t0 + dur:
                    gestures[pid] = what
                    vp.lane[pid] = lane
        present = {pid: t >= away_until[pid] for pid in (1, 2)}
        inputs = {pid: PlayerInput(present[pid], gestures[pid], vp.hand(pid), vp.body[pid]) for pid in (1, 2)}
        events = game.update(t, inputs)
        for e in events:
            log(t, e)
            if e["type"] == "phase" and e["phase"] == MATCH_OVER:
                end_at = t + 4000
        frame = bg.frame()
        if args.dim > 0:
            shade(frame, 0, 0, W, H, args.dim)
        vp.draw(frame, present)
        snap = game.snapshot()
        hud.add(events, t, snap)
        hud.draw(frame, snap, t)
        writer.write(frame)
        t += 1000 / FPS
        if t > 240_000:
            break
    writer.release()
    hp = {pid: p.hp for pid, p in game.players.items()}
    rounds = {pid: p.rounds_won for pid, p in game.players.items()}
    print(f"\n{t / 1000:.1f} s -> {args.out}   rounds won {rounds}   last HP {hp}   match winner: P{game.match_winner}")


if __name__ == "__main__":
    main()
