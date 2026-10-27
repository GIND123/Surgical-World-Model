"""Action interfaces of the world model.

* CameraEncoder  - explicit endoscope ego-motion (4 frame transitions x 4 params per latent step) -> vector.
* ArmTokens      - one token per robot arm: arm id + occupant (instrument / endoscope) + driving hand + action.
* Inverse-dynamics heads that turn a latent transition (z_{t-1}, z_t) into latent actions, sharing one trunk:
    ArmIDM    (ours)     one action per arm, pooled only inside that arm's instrument region
    GlobalIDM (baseline) one monolithic action per transition (Genie / SurgWM style)
    SlotIDM   (baseline) K unsupervised slots, one action per slot (FLAM style)
  Every head outputs a diagonal Gaussian; the KL to N(0, I) is the information bottleneck.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from swm.data.surgvu import ARMS, OCCUPANTS
from swm.models.layers import Attention, RMSNorm, SwiGLU, axial_rope_2d

N_HANDS = 3   # 0 unknown / not driven, 1 left, 2 right
N_ENERGY = 3  # 0 none, 1 cut, 2 coag


class CameraEncoder(nn.Module):
    def __init__(self, dim: int, n_params: int = 20):           # 4 transitions x (dx, dy, log s, angle) + 4 valid
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_params, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, cam: torch.Tensor) -> torch.Tensor:          # (B, T, 20) -> (B, T, D)
        return self.net(cam)


class ArmTokens(nn.Module):
    """Builds (B, T, 4, D) tokens; ``u`` (B, T, 4, du) may be None (robot state only)."""

    def __init__(self, dim: int, du: int):
        super().__init__()
        self.arm = nn.Embedding(len(ARMS), dim)
        self.occ = nn.Embedding(len(OCCUPANTS), dim)
        self.hand = nn.Embedding(N_HANDS, dim)
        self.energy = nn.Embedding(N_ENERGY, dim)
        self.u = nn.Linear(du, dim) if du else None

    def forward(self, occupant: torch.Tensor, hand: torch.Tensor, u: torch.Tensor | None,
                energy: torch.Tensor | None = None) -> torch.Tensor:
        arm_ids = torch.arange(len(ARMS), device=occupant.device)
        tok = self.arm(arm_ids)[None, None] + self.occ(occupant) + self.hand(hand)
        if energy is not None:
            tok = tok + self.energy(energy)
        if u is not None and self.u is not None:
            tok = tok + self.u(u.to(tok.dtype))
        return tok


class Trunk(nn.Module):
    """Per-position features of a latent transition: [z_prev, z_next, z_next - z_prev] -> small transformer."""

    def __init__(self, in_ch: int, dim: int, depth: int, heads: int, height: int, width: int):
        super().__init__()
        self.proj = nn.Linear(3 * in_ch, dim)
        self.layers = nn.ModuleList(nn.ModuleList([RMSNorm(dim), Attention(dim, heads), RMSNorm(dim), SwiGLU(dim)])
                                    for _ in range(depth))
        self.register_buffer("rope", axial_rope_2d(dim // heads, height, width, "cpu"), persistent=False)

    def forward(self, z_prev: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        """(N, C, H, W) x2 -> (N, H*W, D)."""
        n, c, h, w = z_prev.shape
        x = torch.cat([z_prev, z_next, z_next - z_prev], dim=1).flatten(2).transpose(1, 2)
        x = self.proj(x)
        rope = self.rope.to(x.device)
        for n1, att, n2, mlp in self.layers:
            x = x + att(n1(x), freqs_q=rope, freqs_k=rope)
            x = x + mlp(n2(x))
        return x


class GaussianHead(nn.Module):
    def __init__(self, dim: int, du: int):
        super().__init__()
        self.net = nn.Sequential(RMSNorm(dim), nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 2 * du))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.net(h).float().chunk(2, dim=-1)
        return mu, logvar.clamp(-10, 6)


def sample_gaussian(mu, logvar, stochastic: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """-> (sample, per-element KL to N(0, I))."""
    kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)
    if not stochastic:
        return mu, kl
    return mu + torch.randn_like(mu) * (0.5 * logvar).exp(), kl


class PooledQuery(nn.Module):
    """Masked attention pooling of trunk features by one or more query vectors."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.att = Attention(dim, heads)

    def forward(self, q: torch.Tensor, feats: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """q: (N, K, D); feats: (N, S, D); mask: (N, K, S) bool -> (N, K, D)."""
        m = None if mask is None else mask[:, None]
        return self.att(q, ctx=feats, mask=m)


class ArmIDM(nn.Module):
    """One latent action per arm, inferred only from the latents inside that arm's region."""

    def __init__(self, trunk: Trunk, dim: int, heads: int, du: int):
        super().__init__()
        self.trunk = trunk
        self.arm = nn.Embedding(len(ARMS), dim)
        self.occ = nn.Embedding(len(OCCUPANTS), dim)
        self.pool = PooledQuery(dim, heads)
        self.head = GaussianHead(dim, du)

    def forward(self, z_prev, z_next, occupant, region, stochastic: bool = True):
        """z_*: (N, C, H, W); occupant: (N, 4); region: (N, 4, H*W) bool. Returns u (N, 4, du), kl (N, 4, du),
        active (N, 4) -- arms with a visible instrument; inactive arms get u = 0 and no KL."""
        feats = self.trunk(z_prev, z_next)
        q = self.arm(torch.arange(len(ARMS), device=feats.device))[None] + self.occ(occupant)
        active = region.any(-1) & (occupant >= 2)
        safe = region | ~active[..., None]          # avoid all-masked rows; their output is discarded
        mu, logvar = self.head(self.pool(q, feats, safe))
        u, kl = sample_gaussian(mu, logvar, stochastic)
        a = active[..., None].to(u.dtype)
        return u * a, kl * a, active


class GlobalIDM(nn.Module):
    """One monolithic latent action per transition, pooled over the whole frame."""

    def __init__(self, trunk: Trunk, dim: int, heads: int, du: int):
        super().__init__()
        self.trunk = trunk
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pool = PooledQuery(dim, heads)
        self.head = GaussianHead(dim, du)

    def forward(self, z_prev, z_next, stochastic: bool = True):
        feats = self.trunk(z_prev, z_next)
        mu, logvar = self.head(self.pool(self.query.expand(feats.shape[0], -1, -1), feats, None))
        u, kl = sample_gaussian(mu, logvar, stochastic)
        return u, kl                                 # (N, 1, du)


class VQ(nn.Module):
    """Vector quantiser with a straight-through estimator (van den Oord et al., 2017)."""

    def __init__(self, n_codes: int, dim: int, beta: float = 0.25):
        super().__init__()
        self.codebook = nn.Embedding(n_codes, dim)
        nn.init.uniform_(self.codebook.weight, -1 / n_codes, 1 / n_codes)
        self.beta = beta

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = h.float()
        e = self.codebook.weight.float()
        d = h.pow(2).sum(-1, keepdim=True) - 2 * h @ e.t() + e.pow(2).sum(-1)
        idx = d.argmin(-1)
        q = e[idx]
        loss = (q - h.detach()).pow(2).mean() + self.beta * (q.detach() - h).pow(2).mean()
        return h + (q - h).detach(), loss, idx


class VQGlobalIDM(nn.Module):
    """Discrete monolithic latent action (Genie / SurgWM: a small codebook of actions per transition)."""

    def __init__(self, trunk: Trunk, dim: int, heads: int, du: int, n_codes: int = 12):
        super().__init__()
        self.trunk = trunk
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pool = PooledQuery(dim, heads)
        self.proj = nn.Sequential(RMSNorm(dim), nn.Linear(dim, du))
        self.vq = VQ(n_codes, du)

    def forward(self, z_prev, z_next, stochastic: bool = True):
        feats = self.trunk(z_prev, z_next)
        h = self.proj(self.pool(self.query.expand(feats.shape[0], -1, -1), feats, None))
        u, loss, idx = self.vq(h)
        return u, loss, idx                           # (N, 1, du), scalar, (N, 1)


class SlotIDM(nn.Module):
    """K slots from slot attention over the transition features (no arm supervision), one action each."""

    def __init__(self, trunk: Trunk, dim: int, du: int, k: int = 4, iters: int = 3):
        super().__init__()
        self.trunk, self.k, self.iters = trunk, k, iters
        self.slots = nn.Parameter(torch.randn(1, k, dim) * 0.02)
        self.to_q, self.to_k, self.to_v = (nn.Linear(dim, dim, bias=False) for _ in range(3))
        self.norm_in, self.norm_s = RMSNorm(dim), RMSNorm(dim)
        self.gru = nn.GRUCell(dim, dim)
        self.head = GaussianHead(dim, du)

    def forward(self, z_prev, z_next, stochastic: bool = True):
        feats = self.norm_in(self.trunk(z_prev, z_next))
        n, d = feats.shape[0], feats.shape[-1]
        kk, vv = self.to_k(feats), self.to_v(feats)
        s = self.slots.expand(n, -1, -1)
        for _ in range(self.iters):
            attn = torch.einsum("nkd,nsd->nks", self.to_q(self.norm_s(s)), kk) / d ** 0.5
            attn = attn.float().softmax(dim=1).to(vv.dtype)           # slots compete for positions
            attn = attn / attn.sum(-1, keepdim=True).clamp_min(1e-6)
            upd = torch.einsum("nks,nsd->nkd", attn, vv)
            s = self.gru(upd.reshape(-1, d).float(), s.reshape(-1, d).float()).view(n, self.k, d).to(feats.dtype)
        mu, logvar = self.head(s)
        u, kl = sample_gaussian(mu, logvar, stochastic)
        return u, kl                                  # (N, K, du)


def region_from_boxes(boxes: torch.Tensor, height: int, width: int, dilate: int = 1) -> torch.Tensor:
    """boxes: (..., 4) normalised x0, y0, x1, y1 in [0, 1] (NaN = absent) -> (..., H*W) bool grid mask."""
    ys = (torch.arange(height, device=boxes.device) + 0.5) / height
    xs = (torch.arange(width, device=boxes.device) + 0.5) / width
    pad_y, pad_x = dilate / height, dilate / width
    x0, y0, x1, y1 = boxes.unbind(-1)
    inside_y = (ys >= (y0[..., None] - pad_y)) & (ys <= (y1[..., None] + pad_y))
    inside_x = (xs >= (x0[..., None] - pad_x)) & (xs <= (x1[..., None] + pad_x))
    return (inside_y[..., :, None] & inside_x[..., None, :]).flatten(-2)     # NaN boxes compare False
