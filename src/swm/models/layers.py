"""Transformer building blocks: RMSNorm, rotary embeddings, attention, SwiGLU and AdaLN modulation."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = None if self.weight is None else self.weight.to(x.dtype)
        return F.rms_norm(x, (x.shape[-1],), w, self.eps)


def rope_freqs(dim: int, positions: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    """positions: (L,) -> (2, L, dim) cos/sin tables for the half-split rotary convention."""
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=positions.device, dtype=torch.float32) / dim))
    return _tables(positions.float()[:, None] * inv[None])


def _tables(angles: torch.Tensor) -> torch.Tensor:
    """angles (L, d/2) -> (2, L, d): dimension i and i + d/2 rotate together by angles[:, i]."""
    a = torch.cat([angles, angles], dim=-1)
    return torch.stack([a.cos(), a.sin()])


def axial_rope_2d(head_dim: int, h: int, w: int, device, theta: float = 10000.0) -> torch.Tensor:
    """(2, h*w, head_dim): half of the rotary pairs encode rows, half encode columns."""
    ys, xs = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    inv = 1.0 / (theta ** (torch.arange(0, head_dim // 2, 2, device=device, dtype=torch.float32) / (head_dim // 2)))
    ang = torch.cat([ys.flatten().float()[:, None] * inv, xs.flatten().float()[:, None] * inv], dim=-1)
    return _tables(ang)


def apply_rope(x: torch.Tensor, cs: torch.Tensor) -> torch.Tensor:
    """x: (..., L, head_dim); cs: (2, L, head_dim) cos/sin tables."""
    cos, sin = cs[0].to(x.dtype), cs[1].to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat([-x2, x1], dim=-1) * sin


class Attention(nn.Module):
    """Multi-head attention (self or cross) on (B, L, D) with optional RoPE on queries/keys and QK-norm."""

    def __init__(self, dim: int, heads: int, kv_dim: int | None = None):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(kv_dim or dim, 2 * dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        self.q_norm, self.k_norm = RMSNorm(self.head_dim), RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor | None = None, freqs_q=None, freqs_k=None,
                mask: torch.Tensor | None = None, is_causal: bool = False) -> torch.Tensor:
        b, lq, _ = x.shape
        ctx = x if ctx is None else ctx
        q = self.q(x).view(b, lq, self.heads, self.head_dim).transpose(1, 2)
        k, v = self.kv(ctx).view(b, ctx.shape[1], 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q), self.k_norm(k)
        if freqs_q is not None:
            q = apply_rope(q, freqs_q)
        if freqs_k is not None:
            k = apply_rope(k, freqs_k)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal)
        return self.o(out.transpose(1, 2).reshape(b, lq, -1))


class SwiGLU(nn.Module):
    def __init__(self, dim: int, mult: float = 8 / 3):
        super().__init__()
        hidden = int(dim * mult / 64 + 0.999) * 64
        self.w12 = nn.Linear(dim, 2 * hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding of continuous noise levels t in [0, 1] (scaled to [0, 1000])."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = (t.float() * 1000)[..., None] * freqs
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift
