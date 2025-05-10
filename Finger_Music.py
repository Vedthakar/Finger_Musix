import cv2
import numpy as np
import mediapipe as mp
import math
from imutils import resize
import simpleaudio as sa

# ------------------------------
# CONFIGURATION
# ------------------------------
PINCH_THRESH = 22          # px – thumb–index distance that counts as a pinch
RADIUS       = 60          # radius of the orange control ball
BALL_ANGLE   = 0.0         # current angle of the white line on the ball
ANGLE_PREV   = 0.0         # previous frame's hand‑twist angle
CX, CY       = 320, 240    # ball centre (updated when you pinch‑drag it)
IS_PINCHED   = False       # becomes True for the duration of a pinch
CURRENT_MODE = "drum"       # "drum" or "guitar"
MODE_LOCK    = False       # prevents rapid mode flip‑flop while still pinched

# ------------------------------
# PRE‑LOAD SOUNDS (no disk I/O during loop)
# ------------------------------
# Drum kit
snare = sa.WaveObject.from_wave_file("673492__theendofacycle__edm-snare-drum.wav")
kick  = sa.WaveObject.from_wave_file("91600__suicidity__dirty-tonys-kick-drum-mx-028.wav")
base  = sa.WaveObject.from_wave_file("626147__jeremy123__dirt_base.wav")
crash = sa.WaveObject.from_wave_file("45101__matiasreccius__crasha.wav")

# Guitar notes
guitar_g = sa.WaveObject.from_wave_file("Guitar_G.wav")
guitar_a = sa.WaveObject.from_wave_file("Guitar_A.wav")
guitar_d = sa.WaveObject.from_wave_file("Guitar_D.wav")
guitar_b = sa.WaveObject.from_wave_file("Guitar_B.wav")

play = lambda w: w.play()            # fire‑and‑forget helper

# ------------------------------
# MEDIAPIPE HANDS
# ------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

# ------------------------------
# CAMERA – capture low‑res to save CPU
# ------------------------------
vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ------------------------------
# 20‑px overlay icon (purely decorative)
# ------------------------------
OVERLAY_PATH = "8-80320_snare-drumhead-hd-png-download.png"
raw = cv2.imread(OVERLAY_PATH, cv2.IMREAD_UNCHANGED)
if raw is None:
    raise FileNotFoundError(f"Missing overlay: {OVERLAY_PATH}")
raw = resize(raw, width=20)
if raw.shape[2] == 4:
    overlay_bgr, overlay_mask = raw[..., :3], raw[..., 3]
else:
    overlay_bgr = raw
    g = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2GRAY)
    _, overlay_mask = cv2.threshold(g, 1, 255, cv2.THRESH_BINARY)
ov_h, ov_w = overlay_mask.shape
x_off, y_off = 10, 10

# ------------------------------
# MAIN LOOP
# ------------------------------
frame_idx = 0
while True:
    ret, frame = vid.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # ---------- run Mediapipe on half‑sized frame (every other frame) ----------
    small = cv2.resize(frame, (w // 2, h // 2))
    rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    if frame_idx % 2 == 0:
        results = hands.process(rgb)
    frame_idx += 1

    # ---------- draw overlay icon ----------
    roi = frame[y_off:y_off + ov_h, x_off:x_off + ov_w]
    cv2.copyTo(overlay_bgr, overlay_mask, roi)

    # ---------- extract landmarks ----------
    pts = []
    if results and results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0]
        pts = [(int(p.x * w), int(p.y * h)) for p in lm.landmark]
        mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

    if pts:
        # fingertip distances
        d_index  = math.hypot(pts[4][0] - pts[8][0],  pts[4][1] - pts[8][1])
        d_middle = math.hypot(pts[4][0] - pts[12][0], pts[4][1] - pts[12][1])
        d_ring   = math.hypot(pts[4][0] - pts[16][0], pts[4][1] - pts[16][1])
        d_pinky  = math.hypot(pts[4][0] - pts[20][0], pts[4][1] - pts[20][1])
        d_ball   = math.hypot(CX - pts[8][0], CY - pts[8][1])

        # ---------- toggle mode on a pinch *inside* the ball ----------
        if d_ball < RADIUS and d_index < PINCH_THRESH:
            if not MODE_LOCK:
                CURRENT_MODE = "guitar" if CURRENT_MODE == "drum" else "drum"
                MODE_LOCK = True
        else:
            MODE_LOCK = False

        # ---------- play sound based on mode ----------
        if CURRENT_MODE == "drum":
            if d_index < PINCH_THRESH and not IS_PINCHED:
                play(snare); IS_PINCHED = True
            elif d_middle < PINCH_THRESH and not IS_PINCHED:
                play(kick); IS_PINCHED = True
            elif d_ring < PINCH_THRESH and not IS_PINCHED:
                play(base); IS_PINCHED = True
            elif d_pinky < PINCH_THRESH and not IS_PINCHED:
                play(crash); IS_PINCHED = True
            else:
                IS_PINCHED = False
        else:  # guitar
            if d_index < PINCH_THRESH and not IS_PINCHED:
                play(guitar_a); IS_PINCHED = True
            elif d_middle < PINCH_THRESH and not IS_PINCHED:
                play(guitar_g); IS_PINCHED = True
            elif d_ring < PINCH_THRESH and not IS_PINCHED:
                play(guitar_b); IS_PINCHED = True
            elif d_pinky < PINCH_THRESH and not IS_PINCHED:
                play(guitar_d); IS_PINCHED = True
            else:
                IS_PINCHED = False

        # ---------- move / rotate ball ----------
        v1 = np.array(pts[9])  - np.array(pts[0])
        v2 = np.array(pts[17]) - np.array(pts[5])
        ang_curr  = math.degrees(math.atan2(np.cross(v1, v2), np.dot(v1, v2)))
        ang_delta = ang_curr - ANGLE_PREV
        if abs(ang_delta) < 90:
            BALL_ANGLE = (BALL_ANGLE + ang_delta) % 360
        ANGLE_PREV = ang_curr

        if d_index < PINCH_THRESH and (pts[8][0] - CX) ** 2 + (pts[8][1] - CY) ** 2 < RADIUS ** 2:
            CX, CY = pts[8]

    # ---------- draw ball ----------
    center = (int(CX), int(CY))
    end_x  = int(CX + RADIUS * math.cos(math.radians(BALL_ANGLE)))
    end_y  = int(CY + RADIUS * math.sin(math.radians(BALL_ANGLE)))
    cv2.circle(frame, center, RADIUS, (0, 128, 255), -1)
    cv2.line(frame, center, (end_x, end_y), (255, 255, 255), 4)

    # ---------- display ----------
    cv2.imshow("Hand‑Ball Instrument Switcher", frame)
    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
        break

vid.release()
cv2.destroyAllWindows()
