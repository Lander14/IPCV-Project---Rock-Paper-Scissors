"""Play the rock-paper-scissors duel with the camera: pose tracking picks the players, hand gestures cast
spells (T2 + T3 plugged into the T4 game). Mirrors play_keyboard.py's loop; see its docstring for the
shared camera flags (--size, --cam-auto, --dim).

  Fist = FIRE (rock)   open palm = WATER (paper)   index + middle finger = LIGHTNING (scissors)
  The raised hand picks the lane, same as in play_keyboard.py.
  SHIELD / POTION have no camera gesture yet (see game/README.md "to decide as a team"), so they stay on
  the keyboard:  P1 F shield  G potion   |   P2 H shield  N potion.
  --solo: you are player 1 (left) against the computer, player 2 (--level easy / normal / hard).
  --no-glasses: turn off the sunglasses AR overlay.   Esc = quit.

    python play_camera.py
    python play_camera.py --solo --level hard
    python play_camera.py --no-glasses
    python play_camera.py --every 3 --imgsz 384       # a slower laptop: pose every 3rd frame, smaller input
"""
import argparse
import time

import cv2

from bot import LEVELS, Bot
from camera import Camera
from game_logic import POTION, SHIELD, Config, Game, PlayerInput
from hud import Hud, text
from play_keyboard import VirtualPlayers, log, plain_background
from sunglasses_overlay import SunglassesOverlay
from vision import Vision

KEYBOARD_ACTIONS = {"f": (1, SHIELD), "g": (1, POTION), "h": (2, SHIELD), "n": (2, POTION)}
HOLD_MS = 600
HELP = ("Fist=FIRE  open palm=WATER  index+middle=LIGHTNING  |  shield/potion: P1 F/G  P2 H/N  |  Esc quit")


def main():
    ap = argparse.ArgumentParser(description="Rock-paper-scissors duel - camera (pose + hand tracking)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--size", default="1280x720", help="window / drawing size, e.g. 960x540 on a slow laptop")
    ap.add_argument("--dim", type=float, default=0.0, help="darken the camera picture a little (0 = not at all)")
    ap.add_argument("--imgsz", type=int, default=480, help="YOLO pose input size")
    ap.add_argument("--every", type=int, default=2, help="run the pose model on every N-th frame")
    ap.add_argument("--device", default=None, help="cpu, cuda, ... for the pose model")
    ap.add_argument("--glasses", dest="glasses", action="store_true", default=True)
    ap.add_argument("--no-glasses", dest="glasses", action="store_false")
    ap.add_argument("--no-hands", dest="hands", action="store_false", default=True,
                    help="pose tracking only, no gesture recognition (debug)")
    ap.add_argument("--solo", action="store_true", help="single player: you are player 1, the computer player 2")
    ap.add_argument("--level", choices=list(LEVELS), default="normal", help="how good the computer is (--solo)")
    ap.add_argument("--cam-auto", action="store_true", help="auto exposure (default: fixed 30 ms + gain = 30 FPS)")
    args = ap.parse_args()

    W, H = map(int, args.size.split("x"))
    cfg = Config()
    game, hud = Game([1, 2], cfg, (W, H)), Hud({2: "CPU"} if args.solo else None)
    bot = Bot(2, args.level) if args.solo else None
    vp = VirtualPlayers(W, H, cfg.lanes) if bot is not None else None
    vision = Vision(use_hands=args.hands, imgsz=args.imgsz, every=args.every, device=args.device)
    vision.warm_up()                                                   # fail fast, not on the first camera frame
    overlay = SunglassesOverlay() if args.glasses else None
    cam = Camera(args.camera, fast=not args.cam_auto)
    held = {1: (None, 0.0), 2: (None, 0.0)}
    fps, work_ms, n_frames, t_start = 0.0, 0.0, 0, time.monotonic()
    background = plain_background(W, H)                                 # shown until the first camera picture
    t0, t_ms = time.monotonic(), 0.0
    print(__doc__)
    try:
        while True:
            t_prev = t_ms if n_frames else None
            t_ms = (time.monotonic() - t0) * 1000
            if t_prev is not None:
                fps = 0.9 * fps + 0.1 * 1000 / max(1e-3, t_ms - t_prev) if fps else 1000 / max(1e-3, t_ms - t_prev)
            n_frames += 1
            picture, cam_n = cam.latest()
            frame = background.copy() if picture is None else cv2.resize(cv2.flip(picture, 1), (W, H))
            if picture is not None and args.dim > 0:
                from hud import shade
                shade(frame, 0, 0, W, H, args.dim)
            t_work = time.perf_counter()
            players, inputs = vision.process(frame, t_ms, cam_n if picture is not None else None)
            if bot is not None:
                gesture, vp.lane[2] = bot.act(t_ms, game.snapshot())
                inputs[2] = PlayerInput(True, gesture, vp.hand(2), vp.body[2])
            for pid in (1, 2):
                if bot is not None and pid == 2:
                    continue                                           # player 2 is the computer
                action, until = held[pid]
                if t_ms < until:                                       # a keyboard shield/potion override, held
                    inp = inputs[pid]
                    inputs[pid] = PlayerInput(inp.present, action, inp.hand_xy, inp.body_xy)
            events = game.update(t_ms, inputs)
            for e in events:
                log(t_ms, e)
            if overlay is not None:
                overlay.draw(frame, players, t_ms)                     # under the HUD: never covers HP/mana/feed
            snap = game.snapshot()
            hud.add(events, t_ms, snap)
            hud.draw(frame, snap, t_ms)
            work_ms = 0.9 * work_ms + 0.1 * (time.perf_counter() - t_work) * 1000
            text(frame, HELP, (W / 2, H - 52 * H / 720), 0.42 * H / 720, (190, 190, 190), 1, center=True)
            text(frame, f"{fps:3.0f} FPS   camera {cam.fps:3.0f} FPS   game {work_ms:4.1f} ms",
                (W - 380 * H / 720, H - 12 * H / 720), 0.5 * H / 720, (170, 170, 170), 1)
            cv2.imshow("Rock-paper-scissors duel - camera", frame)
            time.sleep(max(0.0, 0.016 - (time.perf_counter() - t_work)))
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            ch = chr(k).lower() if 32 <= k < 128 else ""
            if ch in KEYBOARD_ACTIONS:
                pid, action = KEYBOARD_ACTIONS[ch]
                held[pid] = (action, t_ms + HOLD_MS)
    finally:
        vision.close()
        cam.release()
        cv2.destroyAllWindows()
    print(f"{n_frames} frames, {n_frames / (time.monotonic() - t_start):.1f} FPS on average, "
          f"game + drawing {work_ms:.1f} ms per frame")


if __name__ == "__main__":
    main()
