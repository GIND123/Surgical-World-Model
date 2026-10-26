"""Endoscope ego-motion between two frames as a 4-DoF similarity (dx, dy, log-scale, rotation).

Sparse Lucas-Kanade tracks with a forward-backward check, then a RANSAC similarity fit. Instruments move
independently of the camera and are rejected as outliers; ``mask`` can exclude them explicitly.
Translations are returned as fractions of the frame width/height so they are resolution independent.
"""

from __future__ import annotations

import cv2
import numpy as np

NAN4 = np.full(4, np.nan, dtype=np.float32)


def estimate(prev: np.ndarray, curr: np.ndarray, mask: np.ndarray | None = None,
             max_corners: int = 400) -> tuple[np.ndarray, float, int]:
    """prev, curr: uint8 grayscale of equal size. Returns (params[4], inlier_ratio, n_tracks).

    params = (dx / W, dy / H, log(scale), angle in radians) of the map prev -> curr; NaN when unreliable.
    """
    h, w = prev.shape
    pts = cv2.goodFeaturesToTrack(prev, maxCorners=max_corners, qualityLevel=0.01, minDistance=7,
                                  mask=mask, blockSize=7)
    if pts is None or len(pts) < 12:
        return NAN4, 0.0, 0 if pts is None else len(pts)
    lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, curr, pts, None, **lk)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(curr, prev, nxt, None, **lk)
    ok = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm(back - pts, axis=2)[:, 0] < 1.0)
    if ok.sum() < 12:
        return NAN4, 0.0, int(ok.sum())
    a, b = pts[ok].reshape(-1, 2), nxt[ok].reshape(-1, 2)
    m, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=1.5, maxIters=500,
                                         confidence=0.99)
    if m is None:
        return NAN4, 0.0, len(a)
    ratio = float(inl.mean())
    scale = float(np.hypot(m[0, 0], m[1, 0]))
    angle = float(np.arctan2(m[1, 0], m[0, 0]))
    # translation of the image centre (not of the origin) so zoom does not leak into dx, dy
    c = np.array([w / 2, h / 2])
    tc = m[:, :2] @ c + m[:, 2] - c
    return np.array([tc[0] / w, tc[1] / h, np.log(scale), angle], dtype=np.float32), ratio, len(a)
