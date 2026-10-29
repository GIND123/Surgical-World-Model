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


def camera_input(cam: np.ndarray, scale: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """(9, 16) raw ego-motion -> (9, 20): normalised parameters of the 4 transitions of each latent step followed by
    one validity flag per transition, so that "not estimated" is not confused with "endoscope still".
    The conditioning builders store unreliable transitions as exact zeros, which defines ``valid`` by default.
    A step used to be zeroed when any of its transitions failed (37 % of CRCD steps); on CRCD this lowered the
    detection of real endoscope motion from AUC 0.84 to 0.63."""
    if valid is None:
        valid = (cam.reshape(len(cam), 4, 4) != 0).any(-1)
    return np.concatenate([normalise_camera(cam, scale), valid.astype(np.float32)], -1)


class LatentWindows(Dataset):
    def __init__(self, split: str, cases: list[str] | None = None, max_windows: int | None = None, seed: int = 0,
                 stems: list[str] | None = None, drop_no_signal: bool = False):
        """``stems`` restricts the parts (e.g. case_012_p002) so that runs started at different times of the
        encoding see the same data. ``drop_no_signal`` leaves out windows showing the recorder's colour bars
        (scripts/data/flag_no_signal.py); the S-size interface comparison keeps them (0.44 % of training)."""
        sp = json.loads((ROOT / "configs" / "splits" / "surgvu_v1.json").read_text())
        allowed = set(cases if cases is not None else sp[split])
        self.mean, self.std = latent_stats()
        self.cam_scale = camera_scale()
        self.parts, index = [], []
        no_signal = np.load(CACHE / "no_signal.npz") if drop_no_signal else None
        for cond in sorted((CACHE / "conditioning").glob("*.npz")):
            stem = cond.stem
            if stem.rsplit("_p", 1)[0] not in allowed or (stems is not None and stem not in stems):
                continue
            c = dict(np.load(cond))
            if len(c["occupant"]) == 0:
                continue
            ui = CACHE / "ui" / f"{stem}.npz"
            if ui.exists():
                u = np.load(ui)
                c["hand"], c["energy"] = u["hand"], u["energy"]
            else:
                c["hand"] = c["energy"] = np.zeros_like(c["occupant"])
            self.parts.append(dict(stem=stem, cond=c, lat=None))
            w = np.arange(len(c["occupant"]))
            if drop_no_signal:
                w = w[~no_signal[stem]]
            index.append(np.stack([np.full(len(w), len(self.parts) - 1), w], 1))
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
        cam = camera_input(c["camera"][w], self.cam_scale)
        return dict(z=torch.from_numpy(z), occupant=torch.from_numpy(c["occupant"][w].astype(np.int64)),
                    hand=torch.from_numpy(c["hand"][w].astype(np.int64)),
                    energy=torch.from_numpy(c["energy"][w].astype(np.int64)),
                    boxes=torch.from_numpy(c["boxes"][w].astype(np.float32)),
                    camera=torch.from_numpy(cam.astype(np.float32)), window=torch.tensor([pi, w]))
