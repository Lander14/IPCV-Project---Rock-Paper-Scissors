import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
draw = mp.solutions.drawing_utils

TIPS = [8, 12, 16, 20]   # index, middle, ring, pinky fingertips
PIPS = [6, 10, 14, 18]   # the joint below each fingertip

def classify(lm):
    # finger is "up" if tip is above its joint (y grows downward in image coords)
    up = [lm[t].y < lm[p].y for t, p in zip(TIPS, PIPS)]
    if not any(up):
        return "FIRE"
    if all(up):
        return "WATER"
    if up == [True, True, False, False]:
        return "LIGHTNING"
    return "?"

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    label = "no hand"
    if result.multi_hand_landmarks:
        hl = result.multi_hand_landmarks[0]
        draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
        label = classify(hl.landmark)

    cv2.putText(frame, label, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    cv2.imshow("Elements", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Esc to quit
        break

cap.release()
cv2.destroyAllWindows()