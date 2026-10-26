"""Per-frame fidelity metrics on tensors in [0, 1], shape (N, 3, H, W)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def psnr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mse = ((x.float() - y.float()) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return 10 * torch.log10(1.0 / mse)


def _gauss(win: int = 11, sigma: float = 1.5, device=None) -> torch.Tensor:
    g = torch.exp(-((torch.arange(win, device=device) - win // 2) ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Standard SSIM (Gaussian 11x11, sigma 1.5) on luma-weighted RGB, averaged over channels."""
    x, y = x.float(), y.float()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    k = _gauss(device=x.device).expand(x.shape[1], 1, 11, 11)
    mu_x = F.conv2d(x, k, groups=x.shape[1])
    mu_y = F.conv2d(y, k, groups=y.shape[1])
    sxx = F.conv2d(x * x, k, groups=x.shape[1]) - mu_x ** 2
    syy = F.conv2d(y * y, k, groups=y.shape[1]) - mu_y ** 2
    sxy = F.conv2d(x * y, k, groups=x.shape[1]) - mu_x * mu_y
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return s.flatten(1).mean(1)


class LPIPS:
    """Thin wrapper so callers do not depend on the lpips package's [-1, 1] convention."""

    def __init__(self, device: str = "cuda", net: str = "alex"):
        import lpips

        self.model = lpips.LPIPS(net=net, verbose=False).to(device).eval()

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.model(x.float() * 2 - 1, y.float() * 2 - 1).flatten()
