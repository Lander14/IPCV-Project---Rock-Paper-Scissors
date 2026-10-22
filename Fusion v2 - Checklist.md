# Fusion v2 – Checklist

Review of the second fusion (Claude Code, 22 Oct 2026). Teammates' files are integrated unchanged, 62 tests pass, solo fix and frame dedup work. What is left, in order.

## 1. Code fixes (Claude Code – prompt in section 5)

- [ ] **Critical – player 2's hand is dropped.** `HandAdapter` uses `task2_hands.assign()` (marked *placeholder* by its author): with 3 or 4 hands visible it keeps only the 2 left-most, so P2 gets no gesture while P1 shows both hands. Reproduced: 3 hands → `{1: FIRE}`, P2 missing.
  *Done when:* hands are assigned to players first (pose raised wrist), one `HandSlot` per player id; tests for 2 / 3 / 4 visible hands give both players their gesture.
- [ ] **Faces at playing distance.** Small faces are not found on the full frame (sample `04_arms_raised_stage.jpg`: no glasses; v1 with pose eyes had them).
  *Done when:* face search is seeded from each player's pose head (`detect_region` around `Player.head`); sample 04 gets a face track.
- [ ] **Kalman `dt`** clamped to [1/120, 0.2] s, as the original `face_tracking_final.main()` does.
- [ ] **Speed with two people.** Measured: pose 4 ms (CUDA) + hands ≈25 ms + faces ≈20 ms (p95 40) ≈ 50 ms/frame ≈ 20 FPS. README says ≈38 FPS (measured with fewer people).
  *Done when:* ≤ 33 ms per camera frame with 2 people and ≥ 3 hands, or a clear explanation of why not.
- [ ] **Lobby:** the over-head "P1 100/100" label overlaps the title → hide the over-head bar during LOBBY.
- [ ] **README** updated with the 2-person numbers and the fixes; one commit per fix.

## 2. Review after Claude Code (ask Claude in chat)

- [ ] Re-run the 3- and 4-hand check (P2 must get a gesture)
- [ ] Re-run the still-photo check with timings (samples 03, 04, 05, 06)
- [ ] Look at the glasses / HP bar picture again

## 3. Live test with two people

- [ ] P1 is the one on the left at the start
- [ ] Both players cast rock / paper / scissors and change lane; **P2 casts while P1 has both hands visible**
- [ ] Glasses and HP bar stay on the right player when crossing slowly
- [ ] One hides behind the other for 1 s → same player afterwards
- [ ] One leaves the frame 4 s and comes back → same player
- [ ] A third person walks behind → ignored
- [ ] FPS on screen ≥ 25
- [ ] Record a 30 s clip of all the above for offline comparison:
  `ffmpeg -f v4l2 -input_format mjpeg -video_size 1280x720 -framerate 30 -i /dev/video0 -t 30 clip_2p.mp4`
- [ ] One minute with `python play_camera.py --log session.csv` → T2 jitter metric (raw vs filtered wrist)

## 4. Team

- [ ] **T2:** `task2_hands.py` does not run on its own with Python 3.14 / MediaPipe 1.0.1 (`mp.solutions` was removed). His standalone demo needs his own environment or the Tasks API.
- [ ] **T1:** the face detector misses small faces on the full frame; the game now searches around the pose head (once fix 2 is in).
- [ ] **T3:** validate the colour re-ID option (`appearance=True`) on the two-person clip.
- [ ] Where do the sunglasses images come from (licence)?
- [ ] Still open: shield gesture, potion gesture, lane heights, demo laptop and its environment, shared git repo, owner name per module.

## 5. Prompt for Claude Code

```
Fusion v2 follow-up in game/. Keep everything that works: the teammates' files in face_tracking/,
hand_gestures/ and pose_reid/ stay byte-identical; game_logic.py and hud.py stay untouched; all current
tests keep passing. First commit the current working tree as "Fusion v2: teammates' modules integrated",
then make one commit per fix below.

1. CRITICAL - player 2's hand is dropped. HandAdapter feeds HandSlots through task2_hands.assign(), which
   its author marked as a placeholder ("Task 3 should replace this"): with 2+ detections it keeps only the
   two left-most hands. With P1 showing both hands and P2 one raised hand, P2 gets no gesture (reproduced
   with a fake model: 3 hands -> {1: FIRE}, P2 missing; 4 hands -> same). Fix: assign the raw MediaPipe
   detections to players FIRST (raised-wrist match against PlayerTracker, i.e. the game-side assign()),
   then feed one unmodified task2_hands.HandSlot per player id (slots keyed by pid, never by left-to-right
   order); detections not assigned to a player are ignored. Tests: with 2, 3 and 4 visible hands both
   players get the gesture of their raised hand, and a player's slot never jumps to another player's hand
   when a resting hand appears or disappears.

2. Faces at playing distance. The short-range FaceLandmarker misses faces of ~100-200 px on the full
   1280-wide frame (README already notes it); hand_detection/samples/04_arms_raised_stage.jpg currently
   gets no face track at all. Seed the search from pose: for each present player without a confirmed face
   track, call FaceDetector.detect_region() on a box around Player.head (size from the shoulder width or
   the person box), and pass those detections into FaceTracker.step(). Keep the half-frame search as a
   fallback. Do this in face_module.py, not in face_tracking_final.py. Report before/after on samples
   03, 04, 05, 06.

3. FaceModule.update(): clamp dt to [1/120, 0.2] s, as face_tracking_final.main() does (the first frame
   currently gets dt=0 and a stall gives an unbounded dt).

4. Performance with TWO people and 3+ hands in the frame (samples 03, 05, 06 letterboxed to 1280x720,
   30 frames each, CPU and CUDA): per-stage mean and p95 for pose, hands, faces and drawing. My own
   measurement: pose 4 ms on CUDA, hands ~25 ms, faces ~20 ms (p95 ~40) -> ~50 ms per camera frame,
   ~20 FPS, not the ~38 FPS in the README. If the total is above 33 ms, make the cheapest change that
   brings it under (e.g. --face-every 2 by default, ROI-only face detection once tracks exist) and measure
   again. Ask me before adding a worker thread.

5. Lobby: the over-head "P1 100/100" label overlaps the title. Hide the over-head HP bar during LOBBY and
   show it from ROUND_START on.

6. game/README.md: replace the performance table with the two-person numbers (state the test
   conditions), document fixes 1-5, keep the known-risks list current.

Finish with: all tests passing, the list of commits, and what is still unvalidated on real two-player
footage.
```
