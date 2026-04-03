"""
head_pose.py  ─  v2.0
─────────────────────────────────────────────────────────────────────
Head Pose Estimation via OpenCV solvePnP + Mediapipe face landmarks.
Returns Euler angles in degrees (Pitch, Yaw, Roll).
Includes exponential smoothing to reduce jitter.
"""

import cv2
import numpy as np

# Generic 3-D face model (millimetres, origin at nose tip)
_MODEL_3D = np.array([
    (  0.0,    0.0,    0.0),   # 0 – Nose tip        lm 1
    (  0.0, -330.0,  -65.0),   # 1 – Chin            lm 152
    (-225.0,  170.0, -135.0),  # 2 – L eye outer     lm 263
    ( 225.0,  170.0, -135.0),  # 3 – R eye outer     lm 33
    (-150.0, -150.0, -125.0),  # 4 – L mouth corner  lm 287
    ( 150.0, -150.0, -125.0),  # 5 – R mouth corner  lm 57
], dtype=np.float64)

_LM_IDX = [1, 152, 263, 33, 287, 57]

_DIST = np.zeros((4, 1), dtype=np.float64)


def _euler_from_rvec(rvec: np.ndarray):
    rmat, _ = cv2.Rodrigues(rvec)
    proj    = np.hstack((rmat, np.zeros((3, 1))))
    _, _, _, _, _, _, ea = cv2.decomposeProjectionMatrix(proj)
    return float(ea[0]), float(ea[1]), float(ea[2])


class HeadPoseEstimator:
    """
    Parameters
    ----------
    smooth_alpha : float
        EMA smoothing factor (0 = no update, 1 = no smoothing).
        0.4 gives good balance between responsiveness and stability.
    """

    def __init__(self, smooth_alpha: float = 0.4):
        self._cam   = None
        self._shape = (0, 0)
        self._alpha = smooth_alpha
        self._pitch = 0.0
        self._yaw   = 0.0
        self._roll  = 0.0

    def _camera_matrix(self, w: int, h: int) -> np.ndarray:
        if (w, h) != self._shape:
            f = float(w)
            self._cam   = np.array([[f, 0, w/2],
                                    [0, f, h/2],
                                    [0, 0,   1]], dtype=np.float64)
            self._shape = (w, h)
        return self._cam

    def estimate(self, landmarks, img_w: int, img_h: int):
        """
        Returns (pitch, yaw, roll) smoothed, in degrees.
        Positive pitch = head drooping forward (drowsy).
        """
        try:
            img_pts = np.array(
                [(landmarks[i].x * img_w, landmarks[i].y * img_h)
                 for i in _LM_IDX],
                dtype=np.float64
            )
            ok, rvec, _ = cv2.solvePnP(
                _MODEL_3D, img_pts,
                self._camera_matrix(img_w, img_h), _DIST,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not ok:
                return self._pitch, self._yaw, self._roll

            p, y, r = _euler_from_rvec(rvec)
            a = self._alpha
            self._pitch = a * p + (1 - a) * self._pitch
            self._yaw   = a * y + (1 - a) * self._yaw
            self._roll  = a * r + (1 - a) * self._roll
        except Exception:
            pass

        return self._pitch, self._yaw, self._roll
