"""Training windows: Wan2.2 latents of 33-frame clips with their robot state, arm boxes and camera actions."""

from __future__ import annotations

import json

import numpy as np
import torch
from torch.utils.data import Dataset

from swm.paths import CHECKPOINTS, PROCESSED, ROOT

CACHE = PROCESSED / "surgvu"


def latent_stats() -> tuple[np.ndarray, np.ndarray]:
    cfg = json.loads((CHECKPOINTS / "vae" / "wan22" / "vae" / "config.json").read_text())
    return np.asarray(cfg["latents_mean"], np.float32), np.asarray(cfg["latents_std"], np.float32)


def camera_scale() -> np.ndarray:
    """Per-parameter scale (90th percentile of |ego-motion|), tiled over the 4 transitions of a latent step."""
    st = json.loads((ROOT / "configs" / "stats" / "egomotion.json").read_text())
    return np.tile(np.asarray(st["abs_q90"], np.float32), 4)


def normalise_camera(cam: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Signed log compresses the heavy tail: the endoscope is still most of the time, then moves quickly."""
    return np.sign(cam) * np.log1p(np.abs(cam) / scale)


class LatentWindows(Dataset):
    def __init__(self, split: str, cases: list[str] | None = None, max_windows: int | None = None, seed: int = 0):
        sp = json.loads((ROOT / "configs" / "splits" / "surgvu_v1.json").read_text())
        allowed = set(cases if cases is not None else sp[split])
        self.mean, self.std = latent_stats()
        self.cam_scale = camera_scale()
        self.parts, index = [], []
        for cond in sorted((CACHE / "conditioning").glob("*.npz")):
            stem = cond.stem
            if stem.rsplit("_p", 1)[0] not in allowed:
                continue
            c = dict(np.load(cond))
            if len(c["occupant"]) == 0:
                continue
            self.parts.append(dict(stem=stem, cond=c, lat=None))
            index.append(np.stack([np.full(len(c["occupant"]), len(self.parts) - 1), np.arange(len(c["occupant"]))], 1))
        self.index = np.concatenate(index) if index else np.zeros((0, 2), np.int64)
        if max_windows is not None and len(self.index) > max_windows:
            rng = np.random.default_rng(seed)
            self.index = self.index[np.sort(rng.choice(len(self.index), max_windows, replace=False))]

    def __len__(self) -> int:
        return len(self.index)

    def _lat(self, p: dict) -> np.ndarray:
        if p["lat"] is None:                       # open lazily so every DataLoader worker has its own memmap
            p["lat"] = np.load(CACHE / "latents" / f"{p['stem']}.npy", mmap_mode="r")
        return p["lat"]

    def __getitem__(self, i: int) -> dict:
        pi, w = self.index[i]
        p = self.parts[pi]
        c = p["cond"]
        z = (np.asarray(self._lat(p)[w], np.float32) - self.mean[None, :, None, None]) / self.std[None, :, None, None]
        cam = normalise_camera(c["camera"][w], self.cam_scale) * c["cam_ok"][w][:, None]
        return dict(z=torch.from_numpy(z), occupant=torch.from_numpy(c["occupant"][w].astype(np.int64)),
                    hand=torch.zeros(9, 4, dtype=torch.int64), boxes=torch.from_numpy(c["boxes"][w].astype(np.float32)),
                    camera=torch.from_numpy(cam.astype(np.float32)), window=torch.tensor([pi, w]))
