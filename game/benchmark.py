"""Time every stage of a game frame (no window, no camera): plays the scripted match of simulate.py with a
640x480 picture as the "webcam" and prints the mean / 95th-percentile milliseconds per stage.

    python benchmark.py                 # 1280x720
    python benchmark.py --size 960x540
"""
import argparse
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import hud as H
from game_logic import FIGHT, LOBBY, MATCH_OVER, PAPER, Config, Game, PlayerInput
from hand_adapter import assign as hand_assign
from play_keyboard import VirtualPlayers
from player_tracker import Player
from simulate import SCRIPT
from sunglasses_overlay import SunglassesOverlay

HERE = Path(__file__).resolve().parent

# Vision stages that need YOLO / MediaPipe (not installed by default) are not timed here: only the
# post-processing that runs regardless, on stand-in players/hands of the right shape (T5's "mock mode").
MOCK_PLAYERS = {
    1: Player(present=True, wrists=[(400.0, 390.0)], head=(260.0, 300.0), eyes=((245.0, 290.0), (275.0, 290.0))),
    2: Player(present=True, wrists=[(880.0, 390.0)], head=(1020.0, 300.0), eyes=((1005.0, 290.0), (1035.0, 290.0))),
}
MOCK_HANDS = [((400.0, 390.0), "FIRE"), ((880.0, 390.0), "WATER")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="1280x720")
    W, HH = map(int, ap.parse_args().size.split("x"))
    cam = cv2.resize(cv2.imread(str(HERE / "assets" / "demo_background.jpg")), (640, 480))
    cfg = Config()
    game, hud, vp = Game([1, 2], cfg, (W, HH)), H.Hud(profile=True), VirtualPlayers(W, HH, cfg.lanes)
    overlay = SunglassesOverlay()
    times, n_spells, t, end_at = defaultdict(list), [], 0.0, None

    while end_at is None or t < end_at:
        gestures = {1: None, 2: None}
        if game.phase == LOBBY and t > 500:
            gestures = {1: PAPER, 2: PAPER}
        elif game.phase == FIGHT:
            rel = (game.gt - game.round_gt0) / 1000
            for item in SCRIPT.get(game.round, []):
                t0, pid, what, lane = item[:4]
                if what != "leave" and t0 <= rel < t0 + (item[4] if len(item) > 4 else 0.4):
                    gestures[pid], vp.lane[pid] = what, lane
        c = [time.perf_counter()]

        def tick(name):
            now = time.perf_counter()
            times[name].append((now - c[0]) * 1000)
            c[0] = now

        frame = cv2.resize(cv2.flip(cam, 1), (W, HH))
        tick("camera flip + resize")
        H.shade(frame, 0, 0, W, HH, 0.15)
        tick("dim picture")
        inputs = {pid: PlayerInput(True, gestures[pid], vp.hand(pid), vp.body[pid]) for pid in (1, 2)}
        events = game.update(t, inputs)
        snap = game.snapshot()
        tick("game update")
        hud.add(events, t, snap)
        hud.draw(frame, snap, t)
        tick("drawing (hud)")
        hand_assign(MOCK_HANDS, MOCK_PLAYERS, frame.shape[1])
        tick("hand assignment")
        overlay.draw(frame, MOCK_PLAYERS, t)
        tick("glasses overlay")
        n_spells.append(len(snap["spells"]))
        if any(e["type"] == "phase" and e["phase"] == MATCH_OVER for e in events):
            end_at = t + 3000
        t += 1000 / 30

    total = np.sum([np.array(v) for v in times.values()], axis=0)
    print(f"{len(total)} frames at {W}x{HH}, {sum(n > 1 for n in n_spells)} of them with 2+ spells in the air")
    print(f"{'stage':24s} {'mean ms':>8s} {'p95 ms':>8s}")
    for k, v in list(times.items()) + [("TOTAL", total)]:
        print(f"{k:24s} {np.mean(v):8.2f} {np.percentile(v, 95):8.2f}")
    print("inside the drawing:")
    for k, v in hud.prof.items():
        print(f"  {k:22s} {np.mean(v):8.2f}")
    print(f"=> the game part alone could run at {1000 / total.mean():.0f} FPS")


if __name__ == "__main__":
    main()
