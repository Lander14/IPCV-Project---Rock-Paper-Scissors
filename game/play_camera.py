"""Play the rock-paper-scissors duel with the camera: pose tracking picks the players (T3), hand
gestures cast spells (T2), face tracking + sunglasses + a real HP bar follow each player (T1). Mirrors
play_keyboard.py's loop; see its docstring for the shared camera flags (--size, --cam-auto, --dim).

  Fist = FIRE (rock)   open palm = WATER (paper)   index + middle finger = LIGHTNING (scissors)
  The raised hand picks the lane, same as in play_keyboard.py. Defence is countering in the same lane -
  there is no shield (dropped, 22 Oct 2026 team decision).
  POTION's camera trigger is still pending (see game/README.md "to decide as a team"), so it stays on
  the keyboard:  P1 G   |   P2 N.
  --solo: you are always player 1 (even standing right of centre) against the computer, player 2
  (--level easy / normal / hard).
  --no-glasses / --no-hp-bar: turn off the face effects.   --log metrics.csv: per-frame hand metrics,
  task2_hands.py's own CSV columns (plus the pre-filter wrist, for a before/after comparison).
  Esc = quit.

    python play_camera.py
    python play_camera.py --solo --level hard
    python play_camera.py --no-glasses --log session.csv
    python play_camera.py --every 3 --imgsz 384 --face-every 2   # a slower laptop
"""
import argparse
import csv
import time

import cv2

from bot import LEVELS, Bot
from camera import Camera
from game_logic import POTION, Config, Game, PlayerInput
from hud import Hud, text
from play_keyboard import VirtualPlayers, log, plain_background
from vision import Vision

KEYBOARD_ACTIONS = {"g": (1, POTION), "n": (2, POTION)}
HOLD_MS = 600
HELP = ("Fist=FIRE  open palm=WATER  index+middle=LIGHTNING  |  potion (camera trigger pending): P1 G  P2 N  |  Esc quit")
LOG_HEADER = ["t", "frame", "cam_frame", "slot", "present", "stale", "conf", "raw_gesture", "gesture",
             "wrist_raw_x", "wrist_raw_y", "wrist_sm_x", "wrist_sm_y", "speed_px_s", "filter_on"]


def main():
    ap = argparse.ArgumentParser(description="Rock-paper-scissors duel - camera (pose + hand + face tracking)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--size", default="1280x720", help="window / drawing size, e.g. 960x540 on a slow laptop")
    ap.add_argument("--dim", type=float, default=0.0, help="darken the camera picture a little (0 = not at all)")
    ap.add_argument("--imgsz", type=int, default=480, help="YOLO pose input size")
    ap.add_argument("--every", type=int, default=2, help="run the pose model on every N-th camera frame")
    ap.add_argument("--face-every", type=int, default=1,
                    help="run face detection on every N-th camera frame; the Kalman filter coasts in between")
    ap.add_argument("--device", default=None, help="cpu, cuda, ... for the pose model")
    ap.add_argument("--glasses", dest="glasses", action="store_true", default=True)
    ap.add_argument("--no-glasses", dest="glasses", action="store_false")
    ap.add_argument("--hp-bar", dest="hp_bar", action="store_true", default=True)
    ap.add_argument("--no-hp-bar", dest="hp_bar", action="store_false")
    ap.add_argument("--no-hands", dest="hands", action="store_false", default=True,
                    help="pose + face tracking only, no gesture recognition (debug)")
    ap.add_argument("--no-faces", dest="faces", action="store_false", default=True,
                    help="pose + hand tracking only, no face tracking (debug)")
    ap.add_argument("--no-filter", dest="filter", action="store_false", default=True,
                    help="disable the One Euro filter on hand landmarks (task2_hands.py's 'f' toggle)")
    ap.add_argument("--log", default=None, help="CSV file for per-frame hand metrics (task2_hands.py's columns)")
    ap.add_argument("--solo", action="store_true", help="single player: you are always player 1, the computer P2")
    ap.add_argument("--level", choices=list(LEVELS), default="normal", help="how good the computer is (--solo)")
    ap.add_argument("--cam-auto", action="store_true", help="auto exposure (default: fixed 30 ms + gain = 30 FPS)")
    args = ap.parse_args()

    W, H = map(int, args.size.split("x"))
    cfg = Config()
    game, hud = Game([1, 2], cfg, (W, H)), Hud({2: "CPU"} if args.solo else None)
    bot = Bot(2, args.level) if args.solo else None
    vp = VirtualPlayers(W, H, cfg.lanes) if bot is not None else None
    vision = Vision(use_hands=args.hands, use_faces=args.faces, imgsz=args.imgsz, every=args.every,
                    device=args.device, solo=args.solo, hands_kwargs={"use_filter": args.filter},
                    faces_kwargs={"every": args.face_every})
    vision.warm_up()                                                   # fail fast, not on the first camera frame
    cam = Camera(args.camera, fast=not args.cam_auto)
    held = {1: (None, 0.0), 2: (None, 0.0)}
    fps, work_ms, n_frames, t_start = 0.0, 0.0, 0, time.monotonic()
    background = plain_background(W, H)                                 # shown until the first camera picture
    t0, t_ms = time.monotonic(), 0.0
    log_f = writer = None
    if args.log:
        log_f = open(args.log, "w", newline="")
        writer = csv.writer(log_f)
        writer.writerow(LOG_HEADER)
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
            players, inputs, faces = vision.process(frame, t_ms, cam_n if picture is not None else None)
            if bot is not None:
                gesture, vp.lane[2] = bot.act(t_ms, game.snapshot())
                inputs[2] = PlayerInput(True, gesture, vp.hand(2), vp.body[2])
            for pid in (1, 2):
                if bot is not None and pid == 2:
                    continue                                           # player 2 is the computer
                action, until = held[pid]
                if t_ms < until:                                       # a keyboard potion override, held
                    inp = inputs[pid]
                    inputs[pid] = PlayerInput(inp.present, action, inp.hand_xy, inp.body_xy)
            events = game.update(t_ms, inputs)
            for e in events:
                log(t_ms, e)
            snap = game.snapshot()
            vision.draw_faces(frame, faces, snap, events, t_ms / 1000.0, glasses=args.glasses, hp_bar=args.hp_bar)
            hud.add(events, t_ms, snap)
            hud.draw(frame, snap, t_ms)
            if writer:
                signals, raw_wrists = vision.hand_signals()
                for s, rw in zip(signals, raw_wrists):
                    sm = s["keypoints"][0] if s["keypoints"] else (None, None)   # WRIST landmark, after filtering
                    rw = rw or (None, None)
                    writer.writerow([round(t_ms / 1000, 4), n_frames, cam_n, s["slot"], int(s["present"]),
                                     int(s["stale"]), round(s["confidence"], 3), s["raw_gesture"], s["gesture"],
                                     rw[0], rw[1], sm[0], sm[1], round(s["speed"], 1), int(args.filter)])
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
        if log_f:
            log_f.close()
    print(f"{n_frames} frames, {n_frames / (time.monotonic() - t_start):.1f} FPS on average, "
          f"game + drawing {work_ms:.1f} ms per frame")


if __name__ == "__main__":
    main()
