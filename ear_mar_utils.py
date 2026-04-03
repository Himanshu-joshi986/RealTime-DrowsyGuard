"""
ear_mar_utils.py  ─  v2.0
─────────────────────────────────────────────────────────────────────
EAR  –  Eye Aspect Ratio   (Soukupová & Čech, 2016)
MAR  –  Mouth Aspect Ratio (3-pair vertical / horizontal)

Mediapipe Face Mesh 468-landmark indices verified against the
official canonical face model topology map.
"""

import numpy as np
from scipy.spatial.distance import euclidean

# ──────────────────────────────────────────────────────────────────
#  EYE LANDMARK INDICES
#  Order: [outer, upper1, upper2, inner, lower2, lower1]
#          p1      p2      p3      p4     p5      p6
# ──────────────────────────────────────────────────────────────────
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]

# ──────────────────────────────────────────────────────────────────
#  MOUTH LANDMARK INDICES  (corrected, verified)
#
#  We use 6 specific points for a stable 3-pair MAR:
#    [left_corner, right_corner,
#     upper_top, lower_bottom,
#     upper_mid_left, lower_mid_left,
#     upper_mid_right, lower_mid_right]
#
#  Simplified to 4 key points for clarity and stability:
#    idx 0 = left corner   (61)
#    idx 1 = right corner  (291)
#    idx 2 = upper lip top (0)
#    idx 3 = lower lip bot (17)
# ──────────────────────────────────────────────────────────────────
MOUTH_IDX = [61, 291, 0, 17, 39, 181, 269, 405]
#             left  right  top  bot  ul   ll    ur   lr


def _px(lm, idx: int, w: int, h: int) -> np.ndarray:
    """Mediapipe NormalizedLandmark → pixel (x, y)."""
    return np.array([lm[idx].x * w, lm[idx].y * h], dtype=np.float64)


# ──────────────────────────────────────────────────────────────────
#  EAR  –  Eye Aspect Ratio
# ──────────────────────────────────────────────────────────────────
def compute_EAR(landmarks, eye_idx: list, W: int, H: int) -> float:
    """
    EAR = (||p2−p6|| + ||p3−p5||) / (2 × ||p1−p4||)
    Typical awake: 0.28–0.38  |  Drowsy threshold: < 0.21
    """
    p = [_px(landmarks, i, W, H) for i in eye_idx]
    A = euclidean(p[1], p[5])   # upper-outer  ↔  lower-outer
    B = euclidean(p[2], p[4])   # upper-inner  ↔  lower-inner
    C = euclidean(p[0], p[3])   # left corner  ↔  right corner
    return float((A + B) / (2.0 * C)) if C > 0 else 0.0


# ──────────────────────────────────────────────────────────────────
#  MAR  –  Mouth Aspect Ratio  (3-pair average for stability)
# ──────────────────────────────────────────────────────────────────
def compute_MAR(landmarks, mouth_idx: list, W: int, H: int) -> float:
    """
    Uses 3 vertical pairs averaged over mouth width.

    Pairs (from MOUTH_IDX above):
      Pair A : upper_lip_top (idx2)  ↔  lower_lip_bot (idx3)   — centre
      Pair B : upper_left    (idx4)  ↔  lower_left    (idx5)   — left third
      Pair C : upper_right   (idx6)  ↔  lower_right   (idx7)   — right third

    MAR = (A + B + C) / (3 × mouth_width)

    Typical closed: 0.0–0.15  |  Yawn threshold: > 0.45
    This formulation is MUCH more stable than single-pair MAR and
    eliminates the false-positive yawn that a single centre pair
    produces when the face is slightly tilted.
    """
    left   = _px(landmarks, mouth_idx[0], W, H)
    right  = _px(landmarks, mouth_idx[1], W, H)
    top_c  = _px(landmarks, mouth_idx[2], W, H)
    bot_c  = _px(landmarks, mouth_idx[3], W, H)
    top_l  = _px(landmarks, mouth_idx[4], W, H)
    bot_l  = _px(landmarks, mouth_idx[5], W, H)
    top_r  = _px(landmarks, mouth_idx[6], W, H)
    bot_r  = _px(landmarks, mouth_idx[7], W, H)

    A = euclidean(top_c, bot_c)   # centre pair
    B = euclidean(top_l, bot_l)   # left pair
    C = euclidean(top_r, bot_r)   # right pair

    width = euclidean(left, right)
    return float((A + B + C) / (3.0 * width)) if width > 0 else 0.0


# ──────────────────────────────────────────────────────────────────
#  PERCLOS  –  helper for rolling window (used in main.py)
# ──────────────────────────────────────────────────────────────────
class RollingMetric:
    """
    Keeps a fixed-length sliding window of float values.
    compute_mean() and compute_ratio_below() used for PERCLOS.
    """
    def __init__(self, maxlen: int = 90):
        self._buf   = np.zeros(maxlen, dtype=np.float32)
        self._maxlen = maxlen
        self._idx    = 0
        self._full   = False

    def push(self, value: float):
        self._buf[self._idx] = value
        self._idx = (self._idx + 1) % self._maxlen
        if self._idx == 0:
            self._full = True

    def _active(self):
        return self._buf if self._full else self._buf[:self._idx]

    def mean(self) -> float:
        a = self._active()
        return float(np.mean(a)) if len(a) else 0.0

    def ratio_below(self, thresh: float) -> float:
        """Fraction of frames where value < thresh (≈ PERCLOS)."""
        a = self._active()
        return float(np.mean(a < thresh)) if len(a) else 0.0

    def count(self) -> int:
        return self._maxlen if self._full else self._idx
