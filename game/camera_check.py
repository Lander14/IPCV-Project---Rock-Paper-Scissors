"""How many pictures per second does the webcam really give? Tries a few modes for ~3 s each.
Nothing is shown or saved: it only counts frames.

    python camera_check.py              # camera 0
    python camera_check.py --camera 1

Many laptop webcams give only ~10 FPS at 1280x720 in the default (uncompressed YUYV) mode, and lower the frame rate
in a dim room to expose longer. MJPG (compressed) usually keeps 30 FPS. The fastest mode is the one to use in the game.
"""
import argparse
import time

import cv2

MODES = [("default", None, None), ("YUYV 640x480", "YUYV", (640, 480)), ("MJPG 640x480", "MJPG", (640, 480)),
         ("MJPG 1280x720", "MJPG", (1280, 720))]


def measure(index, fourcc, size, seconds=3.0):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    if size:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened() or not cap.read()[0]:
        cap.release()
        return None
    for _ in range(5):                                   # let auto exposure settle
        cap.read()
    got = int(cap.get(cv2.CAP_PROP_FOURCC))
    real = "".join(chr((got >> 8 * i) & 0xFF) for i in range(4))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n, t0, worst = 0, time.perf_counter(), 0.0
    while time.perf_counter() - t0 < seconds:
        t = time.perf_counter()
        ok, _ = cap.read()
        worst = max(worst, time.perf_counter() - t)
        n += ok
    fps = n / (time.perf_counter() - t0)
    cap.release()
    return real, w, h, fps, worst * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()
    print(f"{'mode asked':16s} {'mode got':16s} {'FPS':>6s} {'slowest read':>13s}")
    best = None
    for name, fourcc, size in MODES:
        r = measure(args.camera, fourcc, size)
        if r is None:
            print(f"{name:16s} could not open")
            continue
        real, w, h, fps, worst = r
        print(f"{name:16s} {real} {w}x{h:<9d} {fps:6.1f} {worst:10.0f} ms", flush=True)
        if best is None or fps > best[1] + 1 or (abs(fps - best[1]) <= 1 and w > best[2]):
            best = (name, fps, w)
    if best:
        print(f"\nfastest: {best[0]} ({best[1]:.0f} FPS)")
    from camera import Camera                                # the mode the game uses: fixed 30 ms exposure + gain
    cam = Camera(args.camera, fast=True)
    time.sleep(2.5)
    n0, t0 = cam.latest()[1], time.perf_counter()
    time.sleep(3.0)
    fps = (cam.latest()[1] - n0) / (time.perf_counter() - t0)
    frame = cam.latest()[0]
    bright = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean() if frame is not None else float("nan")
    cam.release()
    print(f"game mode (MJPG, exposure 30 ms, gain {cam.gain:.0f}): {fps:.1f} FPS, mean brightness {bright:.0f}")


if __name__ == "__main__":
    main()
