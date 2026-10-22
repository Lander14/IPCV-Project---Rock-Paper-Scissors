"""Run the real camera pipeline (pose + hands + faces + HUD) on a recorded video file - no camera, no
bot, no keyboard: both players come purely from vision, like a real two-player game. For offline review
of a clip recorded with the live-demo command in the README, and for collecting T1-T5 evidence numbers
without needing a live camera in the room.

    python run_on_clip.py --video clip_2p.mp4 --out clip_2p_annotated.mp4
    python run_on_clip.py --video clip_2p.mp4 --out clip_2p_annotated.mp4 --device cuda

Writes the annotated video (HUD, glasses, HP bars, a player-id label at each tracked head, and the raw
detected hand gesture per player) and prints + saves <video stem>_stats.json: frame count, per-stage ms
(mean/p95), and per player: % present, % with a face track, % with an assigned hand, gesture counts, and
identity changes (how often the pose tracker's own track id behind that player id changes).
"""
import argparse
import json
import time
from pathlib import Path

import cv2

from game_logic import Config, Game
from hud import Hud, text
from vision import Vision

PLAYER_COL = {1: (230, 110, 250), 2: (110, 230, 110)}


def process(video_path, out_path, stats_path=None, size="1280x720", imgsz=480, every=2, face_every=1,
           device=None, glasses=True, hp_bar=True):
    W, H = map(int, size.split("x"))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (W, H))
    cfg = Config()
    game, hud = Game([1, 2], cfg, (W, H)), Hud()
    vision = Vision(use_hands=True, use_faces=True, imgsz=imgsz, every=every, device=device,
                    faces_kwargs={"every": face_every})
    vision.warm_up()

    stage_ms = {"pose": [], "hands": [], "faces": [], "drawing": []}
    pstats = {pid: {"present": 0, "face": 0, "hand": 0, "gestures": {}, "id_changes": 0, "last_track_id": None}
             for pid in (1, 2)}
    n_frames, n = 0, 0
    try:
        while True:
            ok, picture = cap.read()
            if not ok:
                break
            t_ms = n_frames * 1000.0 / fps_in
            frame = cv2.resize(cv2.flip(picture, 1), (W, H))          # mirrored, like the live game shows it
            n += 1

            t0 = time.perf_counter()
            players = vision._tracker.update(frame.copy(), t_ms)
            t1 = time.perf_counter()
            hand_out = vision._hands.update(frame, players, t_ms) if vision._hands else {}
            t2 = time.perf_counter()
            tracks = vision._faces.update(frame, 1 / fps_in, t_ms / 1000.0) if vision._faces else []
            faces = vision._faces.assign_to_players(tracks, players, W) if vision._faces else {}
            t3 = time.perf_counter()

            inputs = {}
            for pid, p in players.items():
                gesture, hand_xy = hand_out.get(pid, (None, None))
                inputs[pid] = vision._tracker.game_input(pid, gesture, hand_xy)
                st = pstats[pid]
                if p.present:
                    st["present"] += 1
                    if p.track_id != st["last_track_id"] and st["last_track_id"] is not None:
                        st["id_changes"] += 1
                    st["last_track_id"] = p.track_id
                if pid in faces:
                    st["face"] += 1
                if pid in hand_out:
                    st["hand"] += 1
                    g = hand_out[pid][0] or "none"
                    st["gestures"][g] = st["gestures"].get(g, 0) + 1

            events = game.update(t_ms, inputs)
            snap = game.snapshot()
            vision.draw_faces(frame, faces, snap, events, t_ms / 1000.0, glasses=glasses, hp_bar=hp_bar)
            hud.add(events, t_ms, snap)
            hud.draw(frame, snap, t_ms)
            for pid, p in players.items():
                if p.present and p.head is not None:
                    col = PLAYER_COL.get(pid, (255, 255, 255))
                    cv2.circle(frame, tuple(map(int, p.head)), 6, col, -1, cv2.LINE_AA)
                    g = hand_out.get(pid, (None, None))[0]
                    text(frame, f"P{pid} {g or ''}".strip(), (p.head[0] + 12, p.head[1] - 10), 0.55, col, 2)
            t4 = time.perf_counter()
            stage_ms["pose"].append((t1 - t0) * 1000)
            stage_ms["hands"].append((t2 - t1) * 1000)
            stage_ms["faces"].append((t3 - t2) * 1000)
            stage_ms["drawing"].append((t4 - t3) * 1000)

            writer.write(frame)
            n_frames += 1
    finally:
        cap.release()
        writer.release()
        vision.close()

    import numpy as np

    def ms_stats(vals):
        if len(vals) < 2:
            return {"mean": 0.0, "p95": 0.0}
        a = np.array(vals[2:]) if len(vals) > 2 else np.array(vals)
        return {"mean": round(float(np.mean(a)), 2), "p95": round(float(np.percentile(a, 95)), 2)}

    stats = {
        "video": str(video_path), "frames": n_frames, "fps_in": fps_in, "size": [W, H],
        "stage_ms": {k: ms_stats(v) for k, v in stage_ms.items()},
        "players": {
            pid: {
                "pct_present": round(100 * st["present"] / max(1, n_frames), 1),
                "pct_face_tracked": round(100 * st["face"] / max(1, n_frames), 1),
                "pct_hand_assigned": round(100 * st["hand"] / max(1, n_frames), 1),
                "gesture_counts": st["gestures"],
                "identity_changes": st["id_changes"],
            } for pid, st in pstats.items()
        },
    }
    stats_path = Path(stats_path) if stats_path else Path(video_path).with_suffix("").with_name(
        Path(video_path).stem + "_stats.json")
    stats_path.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"\nwrote {out_path}")
    print(f"wrote {stats_path}")
    return stats


def main():
    ap = argparse.ArgumentParser(description="Run the camera pipeline on a recorded clip (no camera, no bot)")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats", default=None, help="default: <video stem>_stats.json next to the video")
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--face-every", type=int, default=1)
    ap.add_argument("--device", default=None, help="cpu, cuda, ... for the pose model")
    ap.add_argument("--no-glasses", dest="glasses", action="store_false", default=True)
    ap.add_argument("--no-hp-bar", dest="hp_bar", action="store_false", default=True)
    args = ap.parse_args()
    process(args.video, args.out, args.stats, args.size, args.imgsz, args.every, args.face_every,
            args.device, args.glasses, args.hp_bar)


if __name__ == "__main__":
    main()
