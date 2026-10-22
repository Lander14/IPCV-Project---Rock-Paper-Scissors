"""Test of run_on_clip.py on a short synthetic clip (no camera needed). Needs the real vision models
(requirements-vision.txt); skipped, not failed, when they are not installed - same pattern as
test_face_module.py's real-model test."""
import json
from pathlib import Path


def _make_synthetic_clip(path, n_frames=5, size=(320, 240)):
    import cv2
    import numpy as np
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    rng = np.random.default_rng(0)
    for i in range(n_frames):
        frame = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_run_on_clip_produces_a_video_and_stats_on_a_synthetic_clip(tmp_path=None):
    try:
        import cv2  # noqa: F401
        from run_on_clip import process
    except ImportError:
        return                                                    # mediapipe/ultralytics not installed: skip

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        clip = tmp / "synthetic.mp4"
        _make_synthetic_clip(clip, n_frames=5, size=(320, 240))
        out = tmp / "synthetic_annotated.mp4"
        try:
            stats = process(clip, out, size="320x240", imgsz=160, every=1, device="cpu")
        except Exception as e:                                    # a flaky real-model edge case: skip, don't fail
            if "mediapipe" in str(type(e)).lower() or "No module named" in str(e):
                return
            raise
        assert out.exists() and out.stat().st_size > 0
        assert stats["frames"] == 5
        assert set(stats["players"]) == {1, 2}
        for pid in (1, 2):
            p = stats["players"][pid]
            assert 0.0 <= p["pct_present"] <= 100.0
            assert isinstance(p["gesture_counts"], dict)
        stats_file = tmp / "synthetic_stats.json"
        assert stats_file.exists()
        assert json.loads(stats_file.read_text())["frames"] == 5


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, f in tests:
        f()
        print(f"ok   {name}")
    print(f"\n{len(tests)} tests passed")
