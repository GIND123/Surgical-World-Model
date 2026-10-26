"""Re-implementation of the Surgical Vision World Model (Koju et al., MICCAI-DEMI 2025; code unreleased).

Three components trained in sequence, following the paper's stated hyper-parameters:
  VideoVQVAE      ST-Transformer VQ-VAE tokenizer: frames 120x180, 4x4 patches, 1024 codes of dim 32,
                  encoder 4 / decoder 6 layers, width 384, 12 heads
  LatentActionModel  ST-Transformer encoder 8 / decoder 12 layers, width 384, 12 latent actions (dim 32),
                  4x4 patches at 40x60; trained to reconstruct the next frame from the past and the action
  MaskGITDynamics ST-Transformer, 12 layers, width 512, 8 heads (about 62.5 M parameters); predicts the masked
                  tokens of future frames from past tokens and latent actions; MaskGIT decoding at inference
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from swm.models.actions import VQ
from swm.models.layers import Attention, RMSNorm, SwiGLU, axial_rope_2d, rope_freqs


class STBlockPlain(nn.Module):
    """Spatial attention, causal temporal attention, MLP (pre-norm, no conditioning)."""

    def __init__(self, dim: int, heads: int, causal: bool = True):
        super().__init__()
        self.n1, self.n2, self.n3 = RMSNorm(dim), RMSNorm(dim), RMSNorm(dim)
        self.sa, self.ta, self.mlp = Attention(dim, heads), Attention(dim, heads), SwiGLU(dim)
        self.causal = causal

    def forward(self, x, rope_s, rope_t):
        b, t, s, d = x.shape
        x = x + self.sa(self.n1(x).reshape(b * t, s, d), freqs_q=rope_s, freqs_k=rope_s).view(b, t, s, d)
        h = self.n2(x).transpose(1, 2).reshape(b * s, t, d)
        h = self.ta(h, freqs_q=rope_t, freqs_k=rope_t, is_causal=self.causal)
        x = x + h.view(b, s, t, d).transpose(1, 2)
        return x + self.mlp(self.n3(x))


class STTransformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, h: int, w: int, causal: bool = True):
        super().__init__()
        self.blocks = nn.ModuleList(STBlockPlain(dim, heads, causal) for _ in range(depth))
        self.norm = RMSNorm(dim)
        self.heads, self.h, self.w = heads, h, w
        self.register_buffer("rope_s", axial_rope_2d(dim // heads, h, w, "cpu"), persistent=False)

    def forward(self, x):                     # (B, T, S, D)
        rope_t = rope_freqs(x.shape[-1] // self.heads, torch.arange(x.shape[1], device=x.device))
        rope_s = self.rope_s.to(x.device)
        for blk in self.blocks:
            x = blk(x, rope_s, rope_t)
        return self.norm(x)


def patchify(x: torch.Tensor, p: int) -> torch.Tensor:
    """(B, T, 3, H, W) -> (B, T, H/p * W/p, 3*p*p)."""
    b, t, c, h, w = x.shape
    x = x.view(b, t, c, h // p, p, w // p, p).permute(0, 1, 3, 5, 2, 4, 6)
    return x.reshape(b, t, (h // p) * (w // p), c * p * p)


def unpatchify(x: torch.Tensor, p: int, h: int, w: int) -> torch.Tensor:
    b, t = x.shape[:2]
    x = x.view(b, t, h // p, w // p, 3, p, p).permute(0, 1, 4, 2, 5, 3, 6)
    return x.reshape(b, t, 3, h, w)


@dataclass
class SurgWMConfig:
    height: int = 120
    width: int = 180
    patch: int = 4
    n_codes: int = 1024
    code_dim: int = 32
    tok_dim: int = 384
    tok_heads: int = 12
    enc_depth: int = 4
    dec_depth: int = 6
    lam_height: int = 40
    lam_width: int = 60
    lam_enc_depth: int = 8
    lam_dec_depth: int = 12
    n_actions: int = 12
    dyn_dim: int = 512
    dyn_depth: int = 12
    dyn_heads: int = 8


class VideoVQVAE(nn.Module):
    def __init__(self, c: SurgWMConfig):
        super().__init__()
        self.c = c
        gh, gw = c.height // c.patch, c.width // c.patch
        self.inp = nn.Linear(3 * c.patch ** 2, c.tok_dim)
        self.enc = STTransformer(c.tok_dim, c.enc_depth, c.tok_heads, gh, gw)
        self.to_code = nn.Linear(c.tok_dim, c.code_dim)
        self.vq = VQ(c.n_codes, c.code_dim)
        self.from_code = nn.Linear(c.code_dim, c.tok_dim)
        self.dec = STTransformer(c.tok_dim, c.dec_depth, c.tok_heads, gh, gw)
        self.out = nn.Linear(c.tok_dim, 3 * c.patch ** 2)

    def encode(self, x):                      # x (B, T, 3, H, W) in [-1, 1] -> codes (B, T, S), quantised (B, T, S, d)
        h = self.to_code(self.enc(self.inp(patchify(x, self.c.patch))))
        q, loss, idx = self.vq(h)
        return idx, q, loss

    def decode_codes(self, idx):
        q = self.vq.codebook(idx)
        return self.decode(q)

    def decode(self, q):
        y = self.out(self.dec(self.from_code(q.to(self.from_code.weight.dtype))))
        return unpatchify(y, self.c.patch, self.c.height, self.c.width)

    def forward(self, x):
        idx, q, vq_loss = self.encode(x)
        rec = self.decode(q)
        return rec, vq_loss, idx


class LatentActionModel(nn.Module):
    """Encoder sees frames 0..T and infers a discrete action per transition; decoder rebuilds frame t+1 from
    frames 0..t and action t. Only the encoder (and codebook) is used afterwards."""

    def __init__(self, c: SurgWMConfig):
        super().__init__()
        self.c = c
        gh, gw = c.lam_height // c.patch, c.lam_width // c.patch
        d = c.tok_dim
        self.inp = nn.Linear(3 * c.patch ** 2, d)
        self.enc = STTransformer(d, c.lam_enc_depth, c.tok_heads, gh, gw, causal=False)
        self.to_act = nn.Linear(d, c.code_dim)
        self.vq = VQ(c.n_actions, c.code_dim)
        self.act_in = nn.Linear(c.code_dim, d)
        self.dec = STTransformer(d, c.lam_dec_depth, c.tok_heads, gh, gw, causal=True)
        self.out = nn.Linear(d, 3 * c.patch ** 2)

    def actions(self, x):                     # x (B, T, 3, h, w) -> per transition: idx (B, T-1), q (B, T-1, dim)
        h = self.enc(self.inp(patchify(x, self.c.patch)))          # (B, T, S, d)
        pooled = (h[:, 1:] - h[:, :-1]).mean(2)                    # transition summary
        q, loss, idx = self.vq(self.to_act(pooled))
        return idx, q, loss

    def forward(self, x):
        idx, q, vq_loss = self.actions(x)
        tok = self.inp(patchify(x[:, :-1], self.c.patch)) + self.act_in(q)[:, :, None]
        pred = unpatchify(self.out(self.dec(tok)), self.c.patch, self.c.lam_height, self.c.lam_width)
        return F.mse_loss(pred, x[:, 1:]), vq_loss, idx


class MaskGITDynamics(nn.Module):
    def __init__(self, c: SurgWMConfig):
        super().__init__()
        self.c = c
        gh, gw = c.height // c.patch, c.width // c.patch
        self.mask_id = c.n_codes
        self.tok = nn.Embedding(c.n_codes + 1, c.dyn_dim)
        self.act = nn.Embedding(c.n_actions + 1, c.dyn_dim)             # last id = no action (frame 0)
        self.net = STTransformer(c.dyn_dim, c.dyn_depth, c.dyn_heads, gh, gw, causal=True)
        self.head = nn.Linear(c.dyn_dim, c.n_codes)

    def logits(self, tokens, actions):        # tokens (B, T, S) with mask ids, actions (B, T) (action into frame t)
        x = self.tok(tokens) + self.act(actions)[:, :, None]
        return self.head(self.net(x))

    def loss(self, tokens, actions, n_ctx: int):
        b, t, s = tokens.shape
        r = torch.rand(b, t, 1, device=tokens.device)
        ratio = torch.cos(r * math.pi / 2)                             # MaskGIT cosine schedule
        mask = torch.rand(b, t, s, device=tokens.device) < ratio
        mask[:, :n_ctx] = False
        inp = torch.where(mask, torch.full_like(tokens, self.mask_id), tokens)
        lg = self.logits(inp, actions)
        return F.cross_entropy(lg[mask].float(), tokens[mask])

    @torch.no_grad()
    def generate(self, tokens, actions, n_ctx: int, steps: int = 12, temperature: float = 1.0, generator=None):
        """Frame-by-frame MaskGIT decoding of frames n_ctx.. given context tokens and actions."""
        tokens = tokens.clone()
        b, t, s = tokens.shape
        for f in range(n_ctx, t):
            cur = tokens[:, :f + 1].clone()
            cur[:, f] = self.mask_id
            known = torch.zeros(b, s, dtype=torch.bool, device=tokens.device)
            for k in range(steps):
                lg = self.logits(cur, actions[:, :f + 1])[:, f].float() / max(temperature, 1e-4)
                probs = lg.softmax(-1)
                sample = torch.multinomial(probs.view(-1, probs.shape[-1]), 1, generator=generator).view(b, s)
                conf = probs.gather(-1, sample[..., None])[..., 0]
                conf = torch.where(known, torch.full_like(conf, float("inf")), conf)
                n_keep = int(s * (1 - math.cos(math.pi / 2 * (k + 1) / steps)))
                if k == steps - 1:
                    n_keep = s
                thresh = conf.topk(max(n_keep, 1), dim=-1).values[:, -1:]
                newly = (conf >= thresh) & ~known
                cur[:, f] = torch.where(newly, sample, cur[:, f])
                known |= newly
            tokens[:, f] = cur[:, f]
        return tokens
