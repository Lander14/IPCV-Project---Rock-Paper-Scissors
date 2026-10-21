"""Play the rock-paper-scissors duel with the keyboard, drawn on the webcam picture.
Tests the game mechanics and the effects before the hand / player tracking is plugged in.

  Player 1 (left):   A ROCK   S PAPER   D SCISSORS   F SHIELD   G POTION   Q / W / E = hand to top / middle / bottom lane
  Player 2 (right):  J ROCK   K PAPER   L SCISSORS   H SHIELD   N POTION   U / I / O = hand to top / middle / bottom lane
  1 / 2 = player leaves / comes back.   Esc = quit.
  --solo: you are player 1 (left), the computer plays player 2 (--level easy / normal / hard).
  A key press = the gesture shown for 0.6 s; holding the key holds it (shield, ready).
  Start: both hold PAPER (open palm) for 1 s.

    python play_keyboard.py                  # webcam picture as background (mirrored)
    python play_keyboard.py --no-camera      # plain background
    python play_keyboard.py --solo           # single player against the computer
    python play_keyboard.py --solo --level hard
    python play_keyboard.py --size 960x540   # lighter, for a slow laptop
    python play_keyboard.py --cam-auto       # leave the webcam on auto exposure (may drop to 10 FPS in a dim room)
"""
import argparse
import time

import cv2
import numpy as np

from bot import LEVELS, Bot
from camera import Camera
from game_logic import PAPER, POTION, ROCK, SCISSORS, SHIELD, Config, Game, PlayerInput
from hud import PLAYER_COL, Hud, shade, text

KEYS = {"a": (1, ROCK), "s": (1, PAPER), "d": (1, SCISSORS), "f": (1, SHIELD), "g": (1, POTION),
        "j": (2, ROCK), "k": (2, PAPER), "l": (2, SCISSORS), "h": (2, SHIELD), "n": (2, POTION)}
LANE_KEYS = {"q": (1, 0), "w": (1, 1), "e": (1, 2), "u": (2, 0), "i": (2, 1), "o": (2, 2)}
HOLD_MS = 600          # one key press holds the gesture this long; the key repeat keeps it while the key is down
HELP_SOLO = "You: A rock  S paper  D scissors  F shield  G potion  Q/W/E lane  |  1 leave  |  Esc quit"
HELP = "P1: A rock  S paper  D scissors  F shield  G potion  Q/W/E lane  |  P2: J K L  H N  U/I/O  |  1/2 leave  Esc"


class VirtualPlayers:
    """Where the two keyboard players 'stand' on the picture: head position and a hand that moves between lanes."""

    def __init__(self, w, h, lanes):
        self.w, self.h, self.lanes = w, h, lanes
        self.body = {1: (0.18 * w, 0.40 * h), 2: (0.82 * w, 0.40 * h)}
        self.lane = {1: 1, 2: 1}

    def hand(self, pid):
        return (0.30 if pid == 1 else 0.70) * self.w, self.lanes[self.lane[pid]] * self.h

    def draw(self, img, present):
        """Outline of each virtual player (with real players in front of the camera this is not needed)."""
        s = self.h / 720
        for pid, (x, y) in self.body.items():
            if not present.get(pid, True):
                continue
            col = tuple(int(0.8 * c) for c in PLAYER_COL[pid])
            cv2.circle(img, (int(x), int(y)), int(42 * s), col, 1, cv2.LINE_AA)
            cv2.ellipse(img, (int(x), int(y + 220 * s)), (int(80 * s), int(170 * s)), 0, 0, 360, col, 1, cv2.LINE_AA)
            hx, hy = self.hand(pid)
            cv2.line(img, (int(x), int(y + 110 * s)), (int(hx), int(hy)), col, 1, cv2.LINE_AA)


def plain_background(w, h):
    y = np.linspace(0, 1, h)[:, None, None]
    return np.repeat(((1 - y) * np.array([60, 45, 40]) + y * np.array([95, 80, 70])).astype(np.uint8), w, axis=1)


def log(t_ms, e):
    if e["type"] in ("aim", "tick"):
        return
    rest = ", ".join(f"{k}={v}" for k, v in e.items() if k not in ("type", "t", "round", "xy"))
    print(f"{t_ms / 1000:7.2f} s  round {e['round']}  {e['type']:11s} {rest}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Rock-paper-scissors duel - keyboard test on the webcam picture")
    ap.add_argument("--no-camera", action="store_true", help="plain background instead of the webcam")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--dim", type=float, default=0.15, help="darken the camera picture a little (0 = not at all)")
    ap.add_argument("--solo", action="store_true", help="single player: you are player 1, the computer player 2")
    ap.add_argument("--level", choices=list(LEVELS), default="normal", help="how good the computer is (--solo)")
    ap.add_argument("--size", default="1280x720", help="window / drawing size, e.g. 960x540 on a slow laptop")
    ap.add_argument("--cam-auto", action="store_true", help="auto exposure (default: fixed 30 ms + gain = 30 FPS)")
    args = ap.parse_args()

    W, H = map(int, args.size.split("x"))
    cfg = Config()
    game, hud = Game([1, 2], cfg, (W, H)), Hud({2: "CPU"} if args.solo else None)
    bot = Bot(2, args.level) if args.solo else None
    vp = VirtualPlayers(W, H, cfg.lanes)
    cam = None if args.no_camera else Camera(args.camera, fast=not args.cam_auto)   # reads in its own thread
    fps, work_ms, n_frames, t_start = 0.0, 0.0, 0, time.monotonic()
    held = {1: (None, 0.0), 2: (None, 0.0)}
    present = {1: True, 2: True}
    background = plain_background(W, H)                                 # made once, copied every frame
    t0, t_ms = time.monotonic(), 0.0
    print(__doc__)
    while True:
        t_prev = t_ms if n_frames else None
        t_ms = (time.monotonic() - t0) * 1000
        if t_prev is not None:                                         # frames per second, smoothed
            fps = 0.9 * fps + 0.1 * 1000 / max(1e-3, t_ms - t_prev) if fps else 1000 / max(1e-3, t_ms - t_prev)
        n_frames += 1
        frame = None
        if cam is not None:
            picture, _ = cam.latest()                                  # never waits for the camera
            if picture is not None:
                frame = cv2.resize(cv2.flip(picture, 1), (W, H))       # mirror: players see themselves like a mirror
                if args.dim > 0:
                    shade(frame, 0, 0, W, H, args.dim)
        if frame is None:
            frame = background.copy()
        t_work = time.perf_counter()
        gestures = {pid: held[pid][0] if t_ms < held[pid][1] else None for pid in (1, 2)}
        if bot is not None:                                            # the computer decides for player 2
            gestures[2], vp.lane[2] = bot.act(t_ms, game.snapshot())
        vp.draw(frame, present)
        inputs = {pid: PlayerInput(present[pid], gestures[pid], vp.hand(pid), vp.body[pid]) for pid in (1, 2)}
        events = game.update(t_ms, inputs)
        for e in events:
            log(t_ms, e)
        snap = game.snapshot()
        hud.add(events, t_ms, snap)
        hud.draw(frame, snap, t_ms)
        work_ms = 0.9 * work_ms + 0.1 * (time.perf_counter() - t_work) * 1000
        text(frame, HELP_SOLO if bot else HELP, (W / 2, H - 52 * H / 720), 0.45 * H / 720, (190, 190, 190), 1,
             center=True)
        cam_txt = f"camera {cam.fps:3.0f} FPS   " if cam is not None else ""
        text(frame, f"{fps:3.0f} FPS   {cam_txt}game {work_ms:4.1f} ms", (W - 380 * H / 720, H - 12 * H / 720),
             0.5 * H / 720, (170, 170, 170), 1)
        cv2.imshow("Rock-paper-scissors duel - keyboard test", frame)
        time.sleep(max(0.0, 0.016 - (time.perf_counter() - t_work)))  # about 60 FPS; the camera never blocks us
        k = cv2.waitKey(1) & 0xFF                                       # 1 ms: do not slow the loop down
        if k == 27:
            break
        ch = chr(k).lower() if 32 <= k < 128 else ""
        pid = KEYS.get(ch, LANE_KEYS.get(ch, (int(ch) if ch in ("1", "2") else 0, None)))[0]
        if bot is not None and pid == 2:
            continue                                                   # player 2 is the computer
        if ch in KEYS:
            held[pid] = (KEYS[ch][1], t_ms + HOLD_MS)
        elif ch in LANE_KEYS:
            vp.lane[pid] = LANE_KEYS[ch][1]
        elif ch in ("1", "2"):
            present[pid] = not present[pid]
    if cam is not None:
        cam.release()                                                  # camera settings back as before
    cv2.destroyAllWindows()
    print(f"{n_frames} frames, {n_frames / (time.monotonic() - t_start):.1f} FPS on average, "
          f"game + drawing {work_ms:.1f} ms per frame")


if __name__ == "__main__":
    main()
