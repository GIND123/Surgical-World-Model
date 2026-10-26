"""Latent spatio-temporal diffusion transformer (ST-DiT) used as the world model's dynamics.

Input: noisy latents (B, T, C, H, W) with a noise level per frame, a conditioning vector per frame
(noise level + camera action + context) and, optionally, action tokens per frame (B, T, K, D).
Each block: spatial self-attention (per frame) -> causal temporal self-attention (per location) ->
cross-attention to the frame's action tokens -> MLP. Modulation is adaLN-single: one projection of the
per-frame conditioning shared by all blocks plus a learned per-block offset; residual gates start at zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt

from swm.models.layers import Attention, RMSNorm, SwiGLU, axial_rope_2d, modulate, rope_freqs


@dataclass
class DiTConfig:
    in_channels: int = 48
    height: int = 14
    width: int = 20
    dim: int = 768
    depth: int = 12
    heads: int = 12
    cross_attention: bool = True
    grad_checkpoint: bool = False


SIZES = {
    "S": dict(dim=512, depth=8, heads=8),
    "B": dict(dim=768, depth=12, heads=12),
    "L": dict(dim=1024, depth=16, heads=16),
}


class STBlock(nn.Module):
    N_MOD = 9   # (shift, scale, gate) for spatial attention, temporal attention and MLP

    def __init__(self, dim: int, heads: int, cross: bool):
        super().__init__()
        self.n1, self.n2, self.n4 = (RMSNorm(dim, affine=False) for _ in range(3))
        self.spatial = Attention(dim, heads)
        self.temporal = Attention(dim, heads)
        self.cross = Attention(dim, heads) if cross else None
        self.n3 = RMSNorm(dim) if cross else None
        self.mlp = SwiGLU(dim)
        self.offset = nn.Parameter(torch.zeros(self.N_MOD, dim))
        if cross:
            nn.init.zeros_(self.cross.o.weight)

    def forward(self, x, mod, rope_s, rope_t, tokens, token_mask):
        b, t, s, d = x.shape
        sh1, sc1, g1, sh2, sc2, g2, sh4, sc4, g4 = (m[:, :, None] for m in (mod + self.offset).unbind(2))
        h = modulate(self.n1(x), sh1, sc1).reshape(b * t, s, d)
        x = x + g1 * self.spatial(h, freqs_q=rope_s, freqs_k=rope_s).view(b, t, s, d)
        h = modulate(self.n2(x), sh2, sc2).transpose(1, 2).reshape(b * s, t, d)
        h = self.temporal(h, freqs_q=rope_t, freqs_k=rope_t, is_causal=True)
        x = x + g2 * h.view(b, s, t, d).transpose(1, 2)
        if self.cross is not None and tokens is not None:
            k = tokens.shape[2]
            h = self.n3(x).reshape(b * t, s, d)
            mask = token_mask.reshape(b * t, 1, 1, k)
            x = x + self.cross(h, ctx=tokens.reshape(b * t, k, d), mask=mask).view(b, t, s, d)
        h = modulate(self.n4(x), sh4, sc4)
        return x + g4 * self.mlp(h)


class STDiT(nn.Module):
    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.dim
        self.embed = nn.Linear(cfg.in_channels, d)
        self.blocks = nn.ModuleList(STBlock(d, cfg.heads, cfg.cross_attention) for _ in range(cfg.depth))
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(d, STBlock.N_MOD * d))
        self.final_norm = RMSNorm(d, affine=False)
        self.final_mod = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.out = nn.Linear(d, cfg.in_channels)
        self.null_token = nn.Parameter(torch.zeros(1, 1, 1, d))    # always-valid key for cross-attention
        nn.init.zeros_(self.mod[1].weight)
        nn.init.zeros_(self.mod[1].bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        hd = d // cfg.heads
        self.register_buffer("rope_s", axial_rope_2d(hd, cfg.height, cfg.width, "cpu"), persistent=False)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, tokens: torch.Tensor | None = None,
                token_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, T, C, H, W); cond: (B, T, D); tokens: (B, T, K, D); token_mask: (B, T, K) bool -> (B, T, C, H, W)."""
        b, t, c, h, w = x.shape
        z = self.embed(x.permute(0, 1, 3, 4, 2).reshape(b, t, h * w, c))
        mod = self.mod(cond).view(b, t, STBlock.N_MOD, -1)
        rope_t = rope_freqs(self.cfg.dim // self.cfg.heads, torch.arange(t, device=x.device))
        rope_s = self.rope_s.to(x.device)
        if tokens is not None:
            null = self.null_token.expand(b, t, 1, -1).to(tokens.dtype)
            tokens = torch.cat([null, tokens], dim=2)
            token_mask = torch.cat([torch.ones(b, t, 1, dtype=torch.bool, device=x.device), token_mask], dim=2)
        for blk in self.blocks:
            if self.cfg.grad_checkpoint and self.training:
                z = ckpt.checkpoint(blk, z, mod, rope_s, rope_t, tokens, token_mask, use_reentrant=False)
            else:
                z = blk(z, mod, rope_s, rope_t, tokens, token_mask)
        sh, sc = self.final_mod(cond).chunk(2, dim=-1)
        z = self.out(modulate(self.final_norm(z), sh[:, :, None], sc[:, :, None]))
        return z.view(b, t, h, w, c).permute(0, 1, 4, 2, 3)
