"""Frechet Video Distance with the I3D TorchScript port used by StyleGAN-V (matches the original TF FVD)."""

from __future__ import annotations

import numpy as np
import scipy.linalg
import torch

from swm.paths import CHECKPOINTS

I3D = CHECKPOINTS / "fvd" / "i3d_torchscript.pt"   # sha256 bec6519f...e5860a5e2


class I3DFeatures:
    def __init__(self, device: str = "cuda"):
        self.net = torch.jit.load(str(I3D)).eval().to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, video: torch.Tensor) -> np.ndarray:
        """video: (B, T, H, W, 3) uint8 or float in [0, 255] -> (B, 400) logits-layer features."""
        x = video.to(self.device).float().permute(0, 4, 1, 2, 3).contiguous()   # B 3 T H W, in [0, 255]
        f = self.net(x, rescale=True, resize=True, return_features=True)
        return f.cpu().numpy().astype(np.float64)


def frechet(a: np.ndarray, b: np.ndarray) -> float:
    mu_a, mu_b = a.mean(0), b.mean(0)
    ca, cb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean = scipy.linalg.sqrtm(ca @ cb)
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(ca + cb - 2 * covmean.real))
