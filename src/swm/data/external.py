"""Windows of the external datasets with kinematics (prepared by scripts/data/prepare_lerobot.py).

Same item format as swm.data.windows.LatentWindows plus ``kin`` (9, 4, 28): for each latent step and arm slot the
four frame transitions of (dpos, drot, djaw), normalised with training-split statistics (zeros for slots without
a PSM), and ``kin_net`` (9, 4, 7): the net motion over the step (targets of the grounding probes).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from swm.data.kinematics import arm_motion, per_latent_step
from swm.data.windows import camera_scale, latent_stats, normalise_camera
from swm.paths import PROCESSED

SLOTS = {"psm1": 0, "psm2": 2}


def calibrate_sides(cond: dict, net: dict, train: np.ndarray) -> bool:
    """True if the image side assignment must be swapped (slot 0 = right-most box assumed to be PSM1).
    Uses training windows only: the side whose box centre moves with an arm's position wins."""
    def corr(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        return 0.0 if ok.sum() < 50 else abs(np.corrcoef(a[ok], b[ok])[0, 1])

    cx = (cond["boxes"][..., 0] + cond["boxes"][..., 2]) / 2                  # (N, 9, 4)
    dcx = np.diff(cx, axis=1)[train]                                           # (n, 8, 4)
    score = {}
    for arm in SLOTS:
        for side in (0, 2):
            m = net[arm][train, 1:, :3]                                       # (n, 8, 3)
            score[(arm, side)] = max(corr(dcx[..., side].ravel(), m[..., k].ravel()) for k in range(3))
    keep = score[("psm1", 0)] + score[("psm2", 2)]
    swap = score[("psm1", 2)] + score[("psm2", 0)]
    return bool(swap > keep)


class ExternalWindows(Dataset):
    def __init__(self, name: str, split: str, dataset_layout: str | None = None, fraction: float = 1.0, seed: int = 0):
        root = PROCESSED / name
        self.lat = np.load(root / "latents.npy", mmap_mode="r")
        self.rows = pd.read_csv(root / "windows.csv")
        cond = dict(np.load(root / "conditioning.npz"))
        sig = dict(np.load(root / "signals.npz"))
        layout = dataset_layout or name
        cat, net = {}, {}
        for arm in SLOTS:
            c, n_ = per_latent_step(arm_motion(sig, layout, arm))
            cat[arm], net[arm] = c, n_
        train = (self.rows["split"] == "train").to_numpy()
        self.swapped = calibrate_sides(cond, net, train)
        if self.swapped:
            b = cond["boxes"].copy()
            cond["boxes"][:, :, 0], cond["boxes"][:, :, 2] = b[:, :, 2], b[:, :, 0]
        self.kin = np.zeros((len(self.rows), 9, 4, cat["psm1"].shape[-1]), np.float32)
        self.kin_net = np.zeros((len(self.rows), 9, 4, net["psm1"].shape[-1]), np.float32)
        for arm, slot in SLOTS.items():
            scale = np.abs(cat[arm][train][:, 1:]).reshape(-1, cat[arm].shape[-1]).std(0) + 1e-8
            self.kin[:, :, slot] = cat[arm] / scale
            self.kin_net[:, :, slot] = net[arm]
        self.cond = cond
        self.mean, self.std = latent_stats()
        self.cam_scale = camera_scale()
        idx = np.flatnonzero((self.rows["split"] == split).to_numpy())
        if fraction < 1.0:                       # data-efficiency curves: subsample whole episodes
            eps = self.rows.loc[idx, "episode"].unique()
            rng = np.random.default_rng(seed)
            keep = set(rng.choice(eps, max(1, round(fraction * len(eps))), replace=False).tolist())
            idx = np.array([i for i in idx if self.rows.loc[i, "episode"] in keep])
        self.index = idx

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        w = int(self.index[i])
        c = self.cond
        z = (np.asarray(self.lat[w], np.float32) - self.mean[None, :, None, None]) / self.std[None, :, None, None]
        cam = normalise_camera(c["camera"][w], self.cam_scale) * c["cam_ok"][w][:, None]
        return dict(z=torch.from_numpy(z), occupant=torch.from_numpy(c["occupant"][w].astype(np.int64)),
                    hand=torch.zeros(9, 4, dtype=torch.int64), boxes=torch.from_numpy(c["boxes"][w].astype(np.float32)),
                    camera=torch.from_numpy(cam.astype(np.float32)), kin=torch.from_numpy(self.kin[w]),
                    kin_net=torch.from_numpy(self.kin_net[w]), window=torch.tensor([0, w]))
