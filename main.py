"""
╔══════════════════════════════════════════════════════════════════════╗
║   HYBRID DRIVER DROWSINESS DETECTION SYSTEM  –  v4.0               ║
║   EAR/PERCLOS  +  MAR  +  Head Pose  +  Auto-Calibration           ║
╚══════════════════════════════════════════════════════════════════════╝

v4.0 critical fixes:
  ✦ EAR smoothing alpha raised (0.30→0.65) — reacts fast to eye closure
  ✦ EAR score formula fixed — uses full open→closed range, not just thresh
  ✦ PERCLOS window tightened to 2 s, fires faster
  ✦ Head pose smoothing alpha raised (0.35→0.6) — reacts in <0.5 s
  ✦ Consecutive frame gates halved — alarm fires in ~0.5 s not ~1 s
  ✦ Score weights rebalanced — EAR alone can trigger alarm
  ✦ SCORE_ALARM lowered 65→55 — more sensitive
  ✦ HEAD weight raised 10→20, can now trigger alarm alone on sustained drop
  ✦ EAR + PERCLOS combo can now reach alarm threshold without other signals
  ✦ Calibration uses 4 s (faster start), prints live debug values
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import (
    FaceLandmarkerOptions, FaceLandmarker, RunningMode
)
import pygame
import threading
import time
import urllib.request
import os
import collections

from ear_mar_utils import (
    compute_EAR, compute_MAR,
    LEFT_EYE_IDX, RIGHT_EYE_IDX, MOUTH_IDX,
    RollingMetric
)
from head_pose import HeadPoseEstimator

# ══════════════════════════════════════════════════════════════════════
#  PATHS
# ══════════════════════════════════════════════════════════════════════
MODEL_TASK_PATH = "models/face_landmarker.task"
MODEL_TASK_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
ALARM_PATH      = "alarm.wav"

# ══════════════════════════════════════════════════════════════════════
#  THRESHOLDS  (base values, may be offset by calibration)
# ══════════════════════════════════════════════════════════════════════
EAR_BASE_THRESH   = 0.21    # eyes closing (overridden by calibration)
PERCLOS_THRESH    = 0.20    # 20% closure in rolling window → drowsy
MAR_BASE_THRESH   = 0.50    # yawning (3-pair formula; closed ≈ 0.0–0.12)
HEAD_PITCH_THRESH = 14.0    # degrees (increased from 10)
HEAD_ROLL_THRESH  = 18.0    # degrees (increased from 15)
HEAD_DEADZONE_PITCH = 5.0   # ignore changes smaller than this
HEAD_DEADZONE_ROLL  = 8.0

# Consecutive-frame gate (at 30 fps)
# These are SHORT — we want fast detection. Smoothing handles false positives.
EAR_FRAMES  = 10            # ~0.33 s  — eyes must stay closed this long
YAWN_FRAMES = 18            # ~0.6 s   — mouth must stay open
HEAD_FRAMES = 15            # ~0.5 s   — head must stay dropped

# ── Fatigue score weights ──────────────────────────────────────────
# IMPORTANT: EAR + PERCLOS alone must be able to breach SCORE_ALARM.
# With weights below: EAR(40) + PERCLOS(30) = 70 > SCORE_ALARM(55). ✓
# Head alone sustained: HEAD(20) = only 20, but combined with EAR it fires.
W_EAR     = 40   # primary signal — most reliable
W_PERCLOS = 30   # secondary — sustained closure over time
W_YAWN    = 10   # supplementary
W_HEAD    = 20   # head drop — raised so sustained nod can contribute

# Alarm thresholds on fatigue score
SCORE_WARNING = 35   # lowered — show warning earlier
SCORE_ALARM   = 55   # lowered — easier to trigger (was 65, too hard)

# Hysteresis: score must drop below this before alarm clears
SCORE_CLEAR   = 25

# Calibration
CALIB_SECONDS     = 4       # 4 s is enough for stable baseline
CALIB_EAR_MARGIN  = -0.04   # threshold = your_baseline − 0.04 (slightly more aggressive)
CALIB_MAR_MARGIN  = +0.18   # threshold = your_baseline + 0.18

# EMA smoothing alphas
# HIGHER alpha = faster response (less smoothing).
# Was 0.30 — too slow. Eyes closing would barely move the smoothed value.
EAR_ALPHA  = 0.65   # fast response to eye closure  (was 0.30 — too slow!)
MAR_ALPHA  = 0.40   # moderate — yawns are sustained
HEAD_ALPHA = 0.50   # was 0.60 – more smoothing for distant/ noisy cases   

# ══════════════════════════════════════════════════════════════════════
#  COLOURS  (BGR)
# ══════════════════════════════════════════════════════════════════════
C_GREEN   = (0,   220,  80)
C_ORANGE  = (0,   165, 255)
C_RED     = (0,    40, 220)
C_WHITE   = (240, 240, 240)
C_GRAY    = (130, 130, 130)
C_DARK    = ( 18,  18,  22)
C_PANEL   = ( 28,  28,  34)
C_ACCENT  = ( 60, 180, 255)
C_YELLOW  = (0,   210, 255)

# ══════════════════════════════════════════════════════════════════════
#  SETUP
# ══════════════════════════════════════════════════════════════════════
os.makedirs("models", exist_ok=True)

if not os.path.exists(MODEL_TASK_PATH):
    print("[INFO] Downloading face landmarker model (one-time, ~3 MB)…")
    urllib.request.urlretrieve(MODEL_TASK_URL, MODEL_TASK_PATH)
    print("[INFO] Model saved.")

options = FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_TASK_PATH),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = FaceLandmarker.create_from_options(options)
poser = HeadPoseEstimator(smooth_alpha=0.50)   

pygame.mixer.init()

# ══════════════════════════════════════════════════════════════════════
#  STATE
# ══════════════════════════════════════════════════════════════════════
class State:
    # Calibration
    calibrating        = True
    calib_start        = None
    calib_ear_samples  = []
    calib_mar_samples  = []
    ear_thresh         = EAR_BASE_THRESH
    mar_thresh         = MAR_BASE_THRESH

    # Smoothed metric values
    ear_smooth         = 0.30
    mar_smooth         = 0.10
    pitch              = 0.0
    yaw                = 0.0
    roll               = 0.0

    # PERCLOS rolling windows (2 sec × 30 fps = 60 frames — tighter = faster)
    perclos_win        = RollingMetric(maxlen=60)

    # Consecutive counters
    ear_cnt   = 0
    yawn_cnt  = 0
    head_cnt  = 0

    # Active flags (latched until counter resets)
    ear_flag  = False
    yawn_flag = False
    head_flag = False

    # Fatigue score (0–100)
    fatigue_score      = 0.0
    score_smooth       = RollingMetric(maxlen=6)   # small window = fast response

    # Alarm
    alarm_on           = False
    alarm_latch        = False   # hysteresis latch

    # Session stats
    total_blinks       = 0
    total_yawns        = 0
    total_head_events  = 0
    session_start      = time.time()

    # Event log (last N events)
    event_log          = collections.deque(maxlen=5)

    # EAR graph buffer
    ear_graph          = collections.deque([0.30] * 120, maxlen=120)
    mar_graph          = collections.deque([0.10] * 120, maxlen=120)
    score_graph        = collections.deque([0.0]  * 120, maxlen=120)

    # Blink detection (for blink count)
    _was_closed        = False

S = State()

# ══════════════════════════════════════════════════════════════════════
#  ALARM
# ══════════════════════════════════════════════════════════════════════
def _alarm_thread():
    try:
        pygame.mixer.music.load(ALARM_PATH)
        pygame.mixer.music.play(-1)
    except Exception as e:
        print(f"[ALARM] {e}")

def trigger_alarm():
    if not S.alarm_on:
        S.alarm_on = True
        threading.Thread(target=_alarm_thread, daemon=True).start()

def stop_alarm():
    if S.alarm_on:
        S.alarm_on = False
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass

# ══════════════════════════════════════════════════════════════════════
#  EYE CROP
# ══════════════════════════════════════════════════════════════════════
def eye_crop(frame, lm, indices, pad=6):
    h, w = frame.shape[:2]
    pts  = np.array([(int(lm[i].x * w), int(lm[i].y * h)) for i in indices])
    x1   = max(0, pts[:, 0].min() - pad)
    y1   = max(0, pts[:, 1].min() - pad)
    x2   = min(w, pts[:, 0].max() + pad)
    y2   = min(h, pts[:, 1].max() + pad)
    return frame[y1:y2, x1:x2]

# ══════════════════════════════════════════════════════════════════════
#  FATIGUE SCORE  (0–100)
# ══════════════════════════════════════════════════════════════════════
def compute_fatigue_score(ear, perclos, yawn, head_norm) -> float:
    """
    Weighted fatigue score 0–100.

    Key design:
      • EAR sub-score uses the FULL open→closed range (0→thresh mapped to 0→1),
        not just the fraction below thresh. This means a fully-closed eye gives
        score=1.0, not a tiny value near 0.
      • EAR(40) + PERCLOS(30) = 70 alone → exceeds SCORE_ALARM(55). ✓
      • HEAD(20) sustained + any EAR dip → easily reaches warning/alarm. ✓
    """
    # EAR sub-score: linearly maps [open_baseline … 0] → [0 … 1]
    # open_baseline estimated as thresh + 0.07 (calibrated offset back)
    """
    head_norm is already computed (0..1) in main loop.
    """
    # EAR sub-score
    ear_open = S.ear_thresh + 0.07
    ear_norm = max(0.0, min(1.0, (ear_open - ear) / max(ear_open, 1e-5)))

    # PERCLOS sub-score
    perc_norm = min(perclos / max(PERCLOS_THRESH, 1e-5), 1.0)

    # Yawn sub-score
    if yawn and S.mar_thresh > 0:
        yawn_norm = min(1.0, max(0.0,
                        (S.mar_smooth - S.mar_thresh) / (S.mar_thresh * 0.5 + 1e-5)))
    else:
        yawn_norm = 0.0

    score = (W_EAR     * ear_norm +
             W_PERCLOS * perc_norm +
             W_YAWN    * yawn_norm +
             W_HEAD    * head_norm)
    return min(100.0, max(0.0, score))

# ══════════════════════════════════════════════════════════════════════
#  HUD DRAWING
# ══════════════════════════════════════════════════════════════════════
FW = 640   # frame width
FH = 480   # frame height
PANEL_W = 200

def draw_hud(canvas, score, status, status_color):
    """Draw full overlay: status banner, metrics, graphs, score bar, log."""
    h, w = canvas.shape[:2]

    # ── Right panel background ──────────────────────────────────────
    overlay = canvas.copy()
    cv2.rectangle(overlay, (w - PANEL_W, 0), (w, h), C_PANEL, -1)
    cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0, canvas)

    # ── Top status bar ──────────────────────────────────────────────
    bar_h = 50
    cv2.rectangle(canvas, (0, 0), (w - PANEL_W, bar_h), C_DARK, -1)
    # Status text
    font_scale = 0.85 if len(status) < 14 else 0.7
    cv2.putText(canvas, status, (12, 34),
                cv2.FONT_HERSHEY_DUPLEX, font_scale, status_color, 2, cv2.LINE_AA)

    # Session timer
    elapsed = int(time.time() - S.session_start)
    mm, ss  = divmod(elapsed, 60)
    cv2.putText(canvas, f"{mm:02d}:{ss:02d}", (w - PANEL_W - 90, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_GRAY, 1, cv2.LINE_AA)

    # ── Metric rows ─────────────────────────────────────────────────
    PX = w - PANEL_W + 10
    def mrow(label, value, y, flag=False, fmt=".3f"):
        col = C_RED if flag else C_WHITE
        cv2.putText(canvas, label, (PX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_GRAY, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{value:{fmt}}", (PX + 90, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

    mrow("EAR",      S.ear_smooth,  80,  S.ear_flag)
    mrow("Thresh",   S.ear_thresh,  98)
    perclos_val = S.perclos_win.ratio_below(S.ear_thresh)
    mrow("PERCLOS",  perclos_val, 116,
         perclos_val > PERCLOS_THRESH, ".1%")
    mrow("MAR",      S.mar_smooth, 138, S.yawn_flag)
    mrow("MARthr",   S.mar_thresh, 156)
    mrow("Pitch",    S.pitch,      174, S.head_flag, ".1f")
    mrow("Roll",     S.roll,       192, S.head_flag, ".1f")

    # Active signal badges
    badge_y = 212
    badge_items = [
        ("EAR",  S.ear_flag,  C_RED),
        ("YAWN", S.yawn_flag, C_ORANGE),
        ("HEAD", S.head_flag, C_RED),
    ]
    bx = PX
    for label_b, active_b, col_b in badge_items:
        bg_col = col_b if active_b else (50, 50, 50)
        txt_col = C_WHITE if active_b else C_GRAY
        cv2.rectangle(canvas, (bx, badge_y - 10), (bx + 34, badge_y + 6), bg_col, -1)
        cv2.putText(canvas, label_b, (bx + 2, badge_y + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, txt_col, 1, cv2.LINE_AA)
        bx += 38

    # Divider
    cv2.line(canvas, (PX, 228), (w - 10, 228), C_GRAY, 1)

    # Session stats
    cv2.putText(canvas, "SESSION STATS", (PX, 246),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_ACCENT, 1, cv2.LINE_AA)
    mrow("Blinks",  S.total_blinks,       262, fmt="d")
    mrow("Yawns",   S.total_yawns,        280, fmt="d")
    mrow("HeadDrop",S.total_head_events,  298, fmt="d")

    # Divider
    cv2.line(canvas, (PX, 312), (w - 10, 312), C_GRAY, 1)

    # Event log
    cv2.putText(canvas, "EVENT LOG", (PX, 328),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_ACCENT, 1, cv2.LINE_AA)
    for k, evt in enumerate(reversed(list(S.event_log))):
        alpha_col = tuple(int(c * (1.0 - k * 0.18)) for c in C_WHITE)
        cv2.putText(canvas, evt, (PX, 346 + k * 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, alpha_col, 1, cv2.LINE_AA)

    # ── Fatigue score bar ───────────────────────────────────────────
    bar_y = h - 30
    bar_x0, bar_x1 = 10, w - PANEL_W - 10
    bar_total = bar_x1 - bar_x0

    cv2.rectangle(canvas, (bar_x0, bar_y - 10), (bar_x1, bar_y + 10),
                  (50, 50, 50), -1)
    fill_w = int(bar_total * score / 100.0)
    if fill_w > 0:
        # Gradient: green → yellow → red
        bar_color = (
            C_GREEN  if score < SCORE_WARNING else
            C_ORANGE if score < SCORE_ALARM   else
            C_RED
        )
        cv2.rectangle(canvas, (bar_x0, bar_y - 10),
                      (bar_x0 + fill_w, bar_y + 10), bar_color, -1)

    # Threshold markers
    for thresh, col in [(SCORE_WARNING, C_YELLOW), (SCORE_ALARM, C_RED)]:
        mx = bar_x0 + int(bar_total * thresh / 100)
        cv2.line(canvas, (mx, bar_y - 14), (mx, bar_y + 14), col, 2)

    cv2.putText(canvas, f"FATIGUE: {score:.0f}/100", (bar_x0, bar_y - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_WHITE, 1, cv2.LINE_AA)

    # ── EAR mini-graph ──────────────────────────────────────────────
    gx0, gy0 = 10, bar_y - 80
    gx1, gy1 = (w - PANEL_W - 10) // 2 - 5, bar_y - 35
    _draw_graph(canvas, list(S.ear_graph), gx0, gy0, gx1, gy1,
                S.ear_thresh, C_ACCENT, "EAR", 0.0, 0.50)

    # ── Score mini-graph ────────────────────────────────────────────
    gx0b = (w - PANEL_W - 10) // 2 + 5
    _draw_graph(canvas, list(S.score_graph), gx0b, gy0, gx1 + gx0b - gx0, gy1,
                SCORE_WARNING, C_ORANGE, "SCORE", 0, 100)

    # ── Calibration overlay ─────────────────────────────────────────
    if S.calibrating:
        elapsed_c = time.time() - (S.calib_start or time.time())
        pct       = min(elapsed_c / CALIB_SECONDS, 1.0)
        cv2.rectangle(canvas, (0, 0), (w - PANEL_W, h), (10, 10, 10), -1)
        cx, cy = (w - PANEL_W) // 2, h // 2
        cv2.putText(canvas, "CALIBRATING", (cx - 110, cy - 30),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, C_ACCENT, 2, cv2.LINE_AA)
        cv2.putText(canvas, "Look straight ahead, keep eyes open",
                    (cx - 155, cy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_WHITE, 1, cv2.LINE_AA)
        bar_len = 280
        bx0 = cx - bar_len // 2
        cv2.rectangle(canvas, (bx0, cy + 30), (bx0 + bar_len, cy + 52),
                      (50, 50, 50), -1)
        cv2.rectangle(canvas, (bx0, cy + 30),
                      (bx0 + int(bar_len * pct), cy + 52), C_GREEN, -1)
        cv2.putText(canvas, f"{int(pct * 100)}%", (cx - 15, cy + 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_DARK, 1, cv2.LINE_AA)


def _draw_graph(canvas, data, x0, y0, x1, y1, threshold, color, label,
                v_min=0.0, v_max=1.0):
    """Draw a mini line graph in the given bounding box."""
    bg = canvas.copy()
    cv2.rectangle(bg, (x0, y0), (x1, y1), (35, 35, 40), -1)
    cv2.addWeighted(bg, 0.7, canvas, 0.3, 0, canvas)

    gw = x1 - x0 - 4
    gh = y1 - y0 - 4

    # Threshold line
    if v_min <= threshold <= v_max:
        ty = y1 - 2 - int((threshold - v_min) / (v_max - v_min) * gh)
        cv2.line(canvas, (x0 + 2, ty), (x1 - 2, ty), (80, 80, 80), 1)

    # Data line
    pts = []
    n   = len(data)
    for i, v in enumerate(data):
        px = x0 + 2 + int(i / max(n - 1, 1) * gw)
        py = y1 - 2 - int(np.clip((v - v_min) / max(v_max - v_min, 1e-5), 0, 1) * gh)
        pts.append((px, py))

    for i in range(1, len(pts)):
        cv2.line(canvas, pts[i - 1], pts[i], color, 1, cv2.LINE_AA)

    cv2.putText(canvas, label, (x0 + 3, y0 + 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_GRAY, 1, cv2.LINE_AA)

# ══════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FW)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FH)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    print("═" * 60)
    print("  HYBRID DROWSINESS DETECTION SYSTEM  v4.0")
    print("  Calibrating… keep eyes open, look straight ahead.")
    print(f"  Alarm fires when fatigue score > {SCORE_ALARM}")
    print("  Press  Q  to quit.")
    print("═" * 60)

    ts_ms = 0
    S.calib_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        H, W  = frame.shape[:2]
        canvas = frame.copy()

        # ── Mediapipe detection ────────────────────────────────────
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms  += 33
        results = landmarker.detect_for_video(mp_img, ts_ms)

        face_detected = (results.face_landmarks and
                         len(results.face_landmarks) > 0)

        if face_detected:
            lm = results.face_landmarks[0]

            # ── Raw metrics ───────────────────────────────────────
            raw_ear = (compute_EAR(lm, LEFT_EYE_IDX,  W, H) +
                       compute_EAR(lm, RIGHT_EYE_IDX, W, H)) / 2.0
            raw_mar  = compute_MAR(lm, MOUTH_IDX, W, H)

            # ── EMA smoothing ──────────────────────────────────────
            # HIGH alpha (0.65) means new values dominate — fast reaction.
            S.ear_smooth = (EAR_ALPHA * raw_ear +
                            (1 - EAR_ALPHA) * S.ear_smooth)
            S.mar_smooth = (MAR_ALPHA * raw_mar +
                            (1 - MAR_ALPHA) * S.mar_smooth)

            # ── Calibration phase ──────────────────────────────────
            if S.calibrating:
                S.calib_ear_samples.append(raw_ear)
                S.calib_mar_samples.append(raw_mar)
                elapsed_c = time.time() - S.calib_start
                if elapsed_c >= CALIB_SECONDS:
                    if S.calib_ear_samples:
                        # Use 20th percentile for EAR (captures slight natural variation)
                        mean_ear = np.percentile(S.calib_ear_samples, 20)
                        mean_mar = np.mean(S.calib_mar_samples)
                        S.ear_thresh = max(0.13, mean_ear + CALIB_EAR_MARGIN)
                        S.mar_thresh = min(0.65, mean_mar + CALIB_MAR_MARGIN)
                        # Reset smoothed values to calibrated baseline
                        S.ear_smooth = mean_ear
                        S.mar_smooth = mean_mar
                        print(f"\n[CALIB] ✓ EAR  baseline={mean_ear:.3f}  → alarm threshold={S.ear_thresh:.3f}")
                        print(f"[CALIB] ✓ MAR  baseline={mean_mar:.3f}  → yawn  threshold={S.mar_thresh:.3f}")
                        print(f"[CALIB] If you close your eyes the EAR will drop below {S.ear_thresh:.3f}")
                        print(f"[CALIB] Alarm fires when fatigue score > {SCORE_ALARM}\n")
                    S.calibrating = False

                draw_hud(canvas, 0.0, "CALIBRATING", C_ACCENT)
                cv2.imshow("Drowsiness Detection", canvas)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # ── PERCLOS ────────────────────────────────────────────
            S.perclos_win.push(S.ear_smooth)
            perclos = S.perclos_win.ratio_below(S.ear_thresh)

            # ── Head pose ──────────────────────────────────────────
            S.pitch, S.yaw, S.roll = poser.estimate(lm, W, H)

            # --- Distance-adaptive thresholds ---
            # Estimate face width using leftmost & rightmost landmarks (indices 33 & 263)
            lx = lm[33].x * W
            rx = lm[263].x * W
            face_width = abs(rx - lx)
            # Adaptive scaling: small face (far) -> higher tolerance
            if face_width < 150:
                scale = min(2.0, 150 / max(face_width, 40))
            else:
                scale = 1.0
            adaptive_pitch_thresh = HEAD_PITCH_THRESH * scale
            adaptive_roll_thresh  = HEAD_ROLL_THRESH  * scale

            # --- Deadzone: ignore small movements ---
            abs_pitch = abs(S.pitch)
            abs_roll  = abs(S.roll)
            if abs_pitch < HEAD_DEADZONE_PITCH:
                abs_pitch = 0.0
            if abs_roll < HEAD_DEADZONE_ROLL:
                abs_roll = 0.0

            head_drop = (abs_pitch > adaptive_pitch_thresh or
                        abs_roll  > adaptive_roll_thresh)

            # If face is very small, head pose is unreliable → reduce its influence later
            head_unreliable = (face_width < 80)

            # ── Counter logic (debounced) ──────────────────────────
            # EAR
            if S.ear_smooth < S.ear_thresh:
                S.ear_cnt += 1
                if S.ear_cnt >= EAR_FRAMES:
                    S.ear_flag = True
            else:
                if S.ear_flag and S.ear_cnt >= EAR_FRAMES:
                    # was a genuine blink/closure
                    S.total_blinks += 1
                    S.event_log.append(f"Blink #{S.total_blinks}")
                S.ear_cnt  = 0
                S.ear_flag = False

            # Yawn
            if S.mar_smooth > S.mar_thresh:
                S.yawn_cnt += 1
                if S.yawn_cnt >= YAWN_FRAMES:
                    if not S.yawn_flag:
                        S.total_yawns += 1
                        S.event_log.append(f"Yawn #{S.total_yawns}")
                    S.yawn_flag = True
            else:
                S.yawn_cnt  = 0
                S.yawn_flag = False

            # 1. Event counter & flag (your original logic)
            if head_drop:
                S.head_cnt += 1
                if S.head_cnt >= HEAD_FRAMES:
                    if not S.head_flag:
                        S.total_head_events += 1
                        S.event_log.append(f"HeadDrop #{S.total_head_events}")
                    S.head_flag = True
            else:
                S.head_cnt  = 0
                S.head_flag = False

            # 2. Continuous head_norm for fatigue score (new block)
            if head_drop:
                max_angle = max(abs_pitch, abs_roll)
                ref_angle = max(adaptive_pitch_thresh, adaptive_roll_thresh)
                head_norm = min(1.0, max_angle / (ref_angle * 1.5))
                if head_unreliable:
                    head_norm *= 0.5
            else:
                head_norm = 0.0

            # ── Fatigue score ──────────────────────────────────────
            raw_score = compute_fatigue_score(
                S.ear_smooth, perclos, S.yawn_flag, head_norm
            )
            S.score_smooth.push(raw_score)
            score = S.score_smooth.mean()

            # ── Graph buffers ──────────────────────────────────────
            S.ear_graph.append(S.ear_smooth)
            S.mar_graph.append(S.mar_smooth)
            S.score_graph.append(score)

            # ── DIRECT ALARM BYPASS ────────────────────────────────
            # Safety net: if eyes have been closed for >= 0.5 s OR
            # head has been dropped for >= 1 s, alarm regardless of score.
            # This ensures the score formula never silently swallows a real event.
            direct_alarm = (
                (S.ear_flag  and S.ear_cnt  >= EAR_FRAMES)  or
                (S.head_flag and S.head_cnt >= HEAD_FRAMES)
            )

            # ── Alarm / hysteresis logic ───────────────────────────
            if score >= SCORE_ALARM or direct_alarm:
                S.alarm_latch = True
            if S.alarm_latch and score < SCORE_CLEAR and not direct_alarm:
                S.alarm_latch = False

            if S.alarm_latch:
                trigger_alarm()
            else:
                stop_alarm()

            # ── Status text — show which signals are active ────────
            active = []
            if S.ear_flag:   active.append("EYES")
            if S.yawn_flag:  active.append("YAWN")
            if S.head_flag:  active.append("HEAD")

            if S.alarm_latch:
                sig_str = "+".join(active) if active else "SCORE"
                status  = f"DROWSY [{sig_str}] WAKE UP!"
                s_col   = C_RED
            elif score >= SCORE_WARNING or active:
                sig_str = "+".join(active) if active else ""
                status  = f"WARNING {sig_str}".strip()
                s_col   = C_ORANGE
            else:
                status  = "ALERT — Driver OK"
                s_col   = C_GREEN

        else:
            # No face detected
            score  = 0.0
            status = "No Face Detected"
            s_col  = C_GRAY
            stop_alarm()
            S.alarm_latch = False

        # ── Draw HUD ───────────────────────────────────────────────
        draw_hud(canvas, score, status, s_col)

        cv2.imshow("Drowsiness Detection", canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ── Cleanup ────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    stop_alarm()

    elapsed = int(time.time() - S.session_start)
    mm, ss  = divmod(elapsed, 60)
    print(f"\n{'═'*45}")
    print(f"  SESSION SUMMARY  ({mm:02d}:{ss:02d})")
    print(f"  Blinks detected : {S.total_blinks}")
    print(f"  Yawns detected  : {S.total_yawns}")
    print(f"  Head drops      : {S.total_head_events}")
    print(f"{'═'*45}\n")


if __name__ == "__main__":
    main()
