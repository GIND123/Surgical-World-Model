"""Per-arm end-effector motion between frames, from Cartesian poses (position + quaternion) and jaw angle."""

from __future__ import annotations

import numpy as np

# where each dataset keeps PSM1/PSM2 Cartesian poses: (signal, xyz start, quat start [x, y, z, w]), jaw (signal, idx)
LAYOUTS = {
    "suturebot": {"psm1": (("state", 0, 3), ("state", 7)), "psm2": (("state", 8, 11), ("state", 15))},
    "jigsaws": {"psm1": (("state", 0, 3), ("state", 7)), "psm2": (("state", 8, 11), ("state", 15))},
    "srth_chole": {"psm1": (("state", 0, 3), ("state", 7)), "psm2": (("state", 8, 11), ("state", 15))},
    "crcd": {"psm1": (("action", 0, 3), ("state", 6)), "psm2": (("action", 7, 10), ("state", 13))},
}


def quat_delta_axis_angle(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    """Rotation taking q0 to q1 (both [..., x, y, z, w]) as an axis-angle vector."""
    x0, y0, z0, w0 = np.moveaxis(q0, -1, 0)
    x1, y1, z1, w1 = np.moveaxis(q1, -1, 0)
    # q1 * conj(q0)
    w = w1 * w0 + x1 * x0 + y1 * y0 + z1 * z0
    x = -w1 * x0 + x1 * w0 - y1 * z0 + z1 * y0
    y = -w1 * y0 + x1 * z0 + y1 * w0 - z1 * x0
    z = -w1 * z0 - x1 * y0 + y1 * x0 + z1 * w0
    v = np.stack([x, y, z], -1)
    sign = np.where(w < 0, -1.0, 1.0)[..., None]                 # shortest rotation
    v, w = v * sign, np.abs(w)
    s = np.linalg.norm(v, axis=-1, keepdims=True)
    angle = 2 * np.arctan2(s, w[..., None])
    return np.where(s > 1e-9, v / np.maximum(s, 1e-9) * angle, 2 * v)


def arm_motion(signals: dict, dataset: str, arm: str) -> np.ndarray:
    """(N, T, ...) per-window signals -> (N, T-1, 7): dpos (3), drot axis-angle (3), djaw (1) per frame transition."""
    (sig, p0, q0), (jsig, j) = LAYOUTS[dataset][arm]
    s = signals[sig]
    pos, quat, jaw = s[..., p0:p0 + 3], s[..., q0:q0 + 4], signals[jsig][..., j:j + 1]
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True).clip(1e-9)
    return np.concatenate([np.diff(pos, axis=-2), quat_delta_axis_angle(quat[..., :-1, :], quat[..., 1:, :]),
                           np.diff(jaw, axis=-2)], -1)


def per_latent_step(motion: np.ndarray, step: int = 4, n_steps: int = 9) -> tuple[np.ndarray, np.ndarray]:
    """Frame transitions (N, 32, D) -> (concatenated (N, 9, 4*D), summed net motion (N, 9, D)); step 0 is zero."""
    n, _, d = motion.shape
    cat = np.zeros((n, n_steps, step * d), np.float32)
    net = np.zeros((n, n_steps, d), np.float32)
    for t in range(1, n_steps):
        m = motion[:, step * (t - 1):step * t]                      # transitions into frames 4t-3 .. 4t
        cat[:, t] = m.reshape(n, -1)
        net[:, t] = m.sum(1)
    return cat, net
