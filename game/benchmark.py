"""Time every stage of a game frame (no window, no camera): plays the scripted match of simulate.py with a
640x480 picture as the "webcam" and prints the mean / 95th-percentile milliseconds per stage.

The game+HUD stages (and hand assignment, which needs no model) always run, with no extra dependencies.
--real additionally times the real pose/hand/face models (needs requirements-vision.txt installed) on a
real frame, comparing CPU and CUDA for pose (YOLO/torch); hands and faces run on MediaPipe's CPU
(XNNPACK) delegate only - a GPU delegate needs extra OpenGL/EGL setup this script does not attempt.

    python benchmark.py                 # 1280x720, game + HUD + hand-assignment only
    python benchmark.py --size 960x540
    python benchmark.py --real          # + real pose/hands/faces, CPU and CUDA where available
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

HERE = Path(__file__).resolve().parent

# Hand assignment needs no model (pure python): timed in the default run, on stand-in players/hands of
# the right shape. Pose/hands/faces themselves need the real models; see --real below.
MOCK_PLAYERS = {
    1: Player(present=True, wrists=[(400.0, 390.0)], head=(260.0, 300.0)),
    2: Player(present=True, wrists=[(880.0, 390.0)], head=(1020.0, 300.0)),
}
MOCK_HANDS = [("FIRE", (400.0, 390.0)), ("WATER", (880.0, 390.0))]


def run_mock(W, HH):
    cam = cv2.resize(cv2.imread(str(HERE / "assets" / "demo_background.jpg")), (640, 480))
    cfg = Config()
    game, hud, vp = Game([1, 2], cfg, (W, HH)), H.Hud(profile=True), VirtualPlayers(W, HH, cfg.lanes)
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


def _timed(fn, n=30):
    ms = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ms.append((time.perf_counter() - t0) * 1000)
    ms = np.array(ms[2:]) if len(ms) > 2 else np.array(ms)       # drop warm-up calls
    return float(np.mean(ms)), float(np.percentile(ms, 95))


def run_real(W, HH):
    """Times the real models on a real frame. Needs requirements-vision.txt; prints what it can and
    says what it skipped, rather than failing the whole benchmark over one missing stage."""
    try:
        import torch
    except ImportError:
        torch = None
    frame = cv2.resize(cv2.imread(str(HERE / "assets" / "demo_background.jpg")), (W, HH))
    cuda = bool(torch and torch.cuda.is_available())
    print(f"\n--real: CUDA {'available (' + torch.cuda.get_device_name(0) + ')' if cuda else 'not available'}")

    print(f"\n{'stage':28s} {'mean ms':>8s} {'p95 ms':>8s}")
    try:
        from player_tracker import PlayerTracker
        for device in (["cpu", "cuda"] if cuda else ["cpu"]):
            tracker = PlayerTracker(imgsz=W if W < 640 else 480, every=1, device=device)
            mean, p95 = _timed(lambda: tracker.update(frame, 0.0))
            print(f"{'pose (' + device + ')':28s} {mean:8.2f} {p95:8.2f}")
    except Exception as e:
        print(f"{'pose':28s} skipped: {e}")

    try:
        from hand_adapter import HandAdapter
        adapter = HandAdapter()
        players = {1: Player(present=True, wrists=[(W / 2, HH / 2)])}
        mean, p95 = _timed(lambda: adapter.update(frame, players, 0.0))
        print(f"{'hands (cpu)':28s} {mean:8.2f} {p95:8.2f}")
        adapter.close()
    except Exception as e:
        print(f"{'hands':28s} skipped: {e}")

    try:
        from face_module import FaceModule
        faces = FaceModule()
        mean, p95 = _timed(lambda: faces.update(frame, 1 / 30, 0.0))
        print(f"{'faces (cpu)':28s} {mean:8.2f} {p95:8.2f}")
        faces.close()
    except Exception as e:
        print(f"{'faces':28s} skipped: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--real", action="store_true", help="also time the real pose/hand/face models")
    args = ap.parse_args()
    W, HH = map(int, args.size.split("x"))
    run_mock(W, HH)
    if args.real:
        run_real(W, HH)


if __name__ == "__main__":
    main()
