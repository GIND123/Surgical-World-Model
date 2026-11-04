"""Re-implementation of the Surgical Vision World Model (SurgWM; Koju et al., MICCAI-DEMI 2025, arXiv:2503.02904).

Three components trained in sequence, following the paper's stated hyper-parameters:
  VideoVQVAE      ST-Transformer VQ-VAE tokenizer: frames 120x180, 4x4 patches, 1024 codes of dim 32,
                  encoder 4 / decoder 6 layers, width 384, 12 heads
  LatentActionModel  ST-Transformer encoder 8 / decoder 12 layers, width 384, 12 latent actions (dim 32),
                  4x4 patches at 40x60; trained to reconstruct the next frame from the past and the action
  MaskGITDynamics ST-Transformer, 12 layers, width 512, 8 heads (about 62.5 M parameters); predicts the masked
                  tokens of future frames from past tokens and latent actions; MaskGIT decoding at inference
                  (25 steps, temperature 1, as in the paper)
Both quantisers move their codes by exponential moving averages and train the encoder with the commitment loss
only, as the paper states (EMAVQ). The authors released only the latent action model (github.com/bhattarailab/
Surgical-Vision-World-Model, action/model/discrete.py); its design (an action prompt token per frame, causal over
time) and its quantiser settings are followed here. Every model trains on 16-frame clips, as in the paper: longer
clips are tokenised in 16-frame chunks, and each predicted frame sees at most the 15 frames before it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from swm.models.layers import Attention, RMSNorm, SwiGLU, apply_rope, axial_rope_2d, rope_freqs


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
    """extra: tokens per frame ahead of the h*w patch tokens (action prompts), without a spatial position."""

    def __init__(self, dim: int, depth: int, heads: int, h: int, w: int, causal: bool = True, extra: int = 0):
        super().__init__()
        self.blocks = nn.ModuleList(STBlockPlain(dim, heads, causal) for _ in range(depth))
        self.norm = RMSNorm(dim)
        self.heads, self.h, self.w = heads, h, w
        self.grad_checkpoint = False
        rope = axial_rope_2d(dim // heads, h, w, "cpu")
        ident = torch.stack([torch.ones(extra, rope.shape[-1]), torch.zeros(extra, rope.shape[-1])])   # no rotation
        self.register_buffer("rope_s", torch.cat([ident, rope], 1), persistent=False)

    def forward(self, x):                     # (B, T, S, D)
        rope_t = rope_freqs(x.shape[-1] // self.heads, torch.arange(x.shape[1], device=x.device))
        rope_s = self.rope_s.to(x.device)
        for blk in self.blocks:
            if self.grad_checkpoint and self.training:
                x = checkpoint(blk, x, rope_s, rope_t, use_reentrant=False)
            else:
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
    clip: int = 16                            # frames per training clip; longer clips: chunks / a window of this size


class EMAVQ(nn.Module):
    """Vector quantiser whose codes follow exponential moving averages of the encoder outputs assigned to them, with
    the commitment loss as the only loss on the encoder (SurgWM: momentum updates of the codebook, van den Oord et
    al., 2017), and a straight-through estimator. Settings of the authors' released latent action model, which uses
    vector_quantize_pytorch: decay 0.8, commitment weight 2, and every code whose moving-average count falls below 2
    restarts from an encoder output of the current batch. update() applies the averages after the optimiser step."""

    def __init__(self, n_codes: int, dim: int, decay: float = 0.8, beta: float = 2.0, dead: float = 2.0,
                 eps: float = 1e-5):
        super().__init__()
        self.decay, self.beta, self.dead, self.eps = decay, beta, dead, eps
        embed = nn.init.kaiming_uniform_(torch.empty(n_codes, dim))
        self.register_buffer("embed", embed)
        self.register_buffer("count", torch.zeros(n_codes))
        self.register_buffer("total", embed.clone())
        self.last: tuple[torch.Tensor, torch.Tensor] | None = None

    def codes(self, idx: torch.Tensor) -> torch.Tensor:
        return self.embed[idx]

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.autocast(h.device.type, enabled=False):                   # exact nearest codes
            h = h.float()
            d = h.pow(2).sum(-1, keepdim=True) - 2 * h @ self.embed.t() + self.embed.pow(2).sum(-1)
            idx = d.argmin(-1)
            q = self.embed[idx]
            loss = self.beta * (q - h).pow(2).mean()
        if self.training:
            self.remember(h.detach(), idx)
        return h + (q - h).detach(), loss, idx

    @torch.compiler.disable
    def remember(self, h: torch.Tensor, idx: torch.Tensor) -> None:
        self.last = (h.reshape(-1, h.shape[-1]), idx.flatten())

    @torch.no_grad()
    def update(self) -> int:
        """Move the codes towards the encoder outputs of the last training batch; returns the number restarted."""
        if self.last is None:
            return 0
        h, idx = self.last
        self.last = None
        n = torch.bincount(idx, minlength=len(self.embed)).float()
        self.count.mul_(self.decay).add_(n, alpha=1 - self.decay)
        self.total.mul_(self.decay).add_(torch.zeros_like(self.total).index_add_(0, idx, h), alpha=1 - self.decay)
        size = (self.count + self.eps) / (self.count.sum() + len(self.count) * self.eps) * self.count.sum()
        self.embed.copy_(self.total / size[:, None])
        dead = (self.count < self.dead).nonzero().flatten()
        if len(dead):
            new = h[torch.randint(0, len(h), (len(dead),), device=h.device)]
            self.embed[dead], self.total[dead], self.count[dead] = new, new * self.dead, self.dead
        return len(dead)


class VideoVQVAE(nn.Module):
    def __init__(self, c: SurgWMConfig):
        super().__init__()
        self.c = c
        gh, gw = c.height // c.patch, c.width // c.patch
        self.inp = nn.Linear(3 * c.patch ** 2, c.tok_dim)
        self.enc = STTransformer(c.tok_dim, c.enc_depth, c.tok_heads, gh, gw)
        self.to_code = nn.Linear(c.tok_dim, c.code_dim)
        self.vq = EMAVQ(c.n_codes, c.code_dim)
        self.from_code = nn.Linear(c.code_dim, c.tok_dim)
        self.dec = STTransformer(c.tok_dim, c.dec_depth, c.tok_heads, gh, gw)
        self.out = nn.Linear(c.tok_dim, 3 * c.patch ** 2)

    def encode(self, x):                      # x (B, T, 3, H, W) in [-1, 1] -> codes (B, T, S), quantised (B, T, S, d)
        if x.shape[1] > self.c.clip:          # longer than the training clips: independent chunks, as in training
            parts = [self.encode(x[:, i:i + self.c.clip]) for i in range(0, x.shape[1], self.c.clip)]
            return (torch.cat([p[0] for p in parts], 1), torch.cat([p[1] for p in parts], 1),
                    torch.stack([p[2] for p in parts]).mean())
        h = self.to_code(self.enc(self.inp(patchify(x, self.c.patch))))
        q, loss, idx = self.vq(h)
        return idx, q, loss

    def decode_codes(self, idx):
        return self.decode(self.vq.codes(idx))

    def decode(self, q):
        if q.shape[1] > self.c.clip:
            return torch.cat([self.decode(q[:, i:i + self.c.clip]) for i in range(0, q.shape[1], self.c.clip)], 1)
        y = self.out(self.dec(self.from_code(q.to(self.from_code.weight.dtype))))
        return unpatchify(y, self.c.patch, self.c.height, self.c.width)

    def forward(self, x):
        idx, q, vq_loss = self.encode(x)
        rec = self.decode(q)
        return rec, vq_loss, idx


class LatentActionModel(nn.Module):
    """As the authors' released model: a learned action prompt token ahead of each frame's patch tokens, causal over
    time, so that the prompt's output at frame t+1 sees frames 0..t+1 and is quantised to the action of transition
    t. The decoder rebuilds frame t+1 (sigmoid, in [0, 1]) from frames 0..t with the action of transition t as the
    prompt token of frame t. Only the encoder (and codebook) is used afterwards."""

    def __init__(self, c: SurgWMConfig):
        super().__init__()
        self.c = c
        gh, gw = c.lam_height // c.patch, c.lam_width // c.patch
        d = c.tok_dim
        self.inp = nn.Linear(3 * c.patch ** 2, d)
        self.prompt = nn.Parameter(torch.empty(d).uniform_(-1, 1))
        self.enc = STTransformer(d, c.lam_enc_depth, c.tok_heads, gh, gw, causal=True, extra=1)
        self.to_act = nn.Linear(d, c.code_dim)
        self.vq = EMAVQ(c.n_actions, c.code_dim)
        self.act_in = nn.Linear(c.code_dim, d)
        self.dec = STTransformer(d, c.lam_dec_depth, c.tok_heads, gh, gw, causal=True, extra=1)
        self.out = nn.Linear(d, 3 * c.patch ** 2)

    def actions(self, x):                     # x (B, T, 3, h, w) -> per transition: idx (B, T-1), q (B, T-1, dim)
        b, t, w = x.shape[0], x.shape[1], self.c.clip
        if t > w:                             # windows of the training length; each transition from the first that
            idx, q, done = [], [], 0          # contains it, the last window ending at the last frame
            while done < t - 1:
                s = min(done, t - w)
                i, qq, _ = self.actions(x[:, s:s + w])                 # transitions s .. s + w - 2
                idx.append(i[:, done - s:])
                q.append(qq[:, done - s:])
                done = s + w - 1
            return torch.cat(idx, 1), torch.cat(q, 1), None
        tok = self.inp(patchify(x, self.c.patch))
        h = self.enc(torch.cat([self.prompt.to(tok.dtype).expand(b, t, 1, -1), tok], 2))[:, 1:, 0]  # frames 1..T-1
        q, loss, idx = self.vq(self.to_act(h))
        return idx, q, loss

    def forward(self, x):                     # x in [-1, 1]; the reconstruction loss in [0, 1], as released
        idx, q, vq_loss = self.actions(x)
        tok = self.inp(patchify(x[:, :-1], self.c.patch))
        y = self.dec(torch.cat([self.act_in(q)[:, :, None].to(tok.dtype), tok], 2))[:, :, 1:]
        pred = torch.sigmoid(unpatchify(self.out(y), self.c.patch, self.c.lam_height, self.c.lam_width).float())
        return F.mse_loss(pred, x[:, 1:].float().add(1).div(2)), vq_loss, idx


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

    def new_cache(self, length: int) -> dict:
        """Keys and values of every block's temporal attention for `length` frames, allocated on first use."""
        return {"length": length, "k": [None] * len(self.net.blocks), "v": [None] * len(self.net.blocks)}

    def frame_logits(self, frame, actions, f: int, cache: dict) -> torch.Tensor:
        """Logits of frame f (tokens (B, S)) given the cached temporal-attention keys and values of frames 0..f-1
        of every block (new_cache; frame f's are written at position f). Under causal temporal attention a frame's
        hidden states never depend on later frames, so this equals logits(tokens[:, :f + 1], actions[:, :f + 1])
        [:, f] while running only frame f (the block's attention, written out to keep the keys and values)."""
        net = self.net
        b, s = frame.shape
        heads, hd = self.c.dyn_heads, self.c.dyn_dim // self.c.dyn_heads
        rope_s = net.rope_s.to(frame.device)
        rope_t = rope_freqs(hd, torch.arange(f, f + 1, device=frame.device))       # frame f's position
        x = (self.tok(frame) + self.act(actions[:, f])[:, None])[:, None]           # (B, 1, S, D)
        for i, blk in enumerate(net.blocks):
            x = x + blk.sa(blk.n1(x).reshape(b, s, -1), freqs_q=rope_s, freqs_k=rope_s).view(b, 1, s, -1)
            h, ta = blk.n2(x).reshape(b * s, 1, -1), blk.ta
            q = apply_rope(ta.q_norm(ta.q(h).view(b * s, 1, heads, hd).transpose(1, 2)), rope_t)
            k, v = ta.kv(h).view(b * s, 1, 2, heads, hd).permute(2, 0, 3, 1, 4)
            k = apply_rope(ta.k_norm(k), rope_t)
            if cache["k"][i] is None:
                cache["k"][i] = k.new_zeros(b * s, heads, cache["length"], hd)
                cache["v"][i] = v.new_zeros(b * s, heads, cache["length"], hd)
            cache["k"][i][:, :, f], cache["v"][i][:, :, f] = k[:, :, 0], v[:, :, 0]
            y = F.scaled_dot_product_attention(q, cache["k"][i][:, :, :f + 1], cache["v"][i][:, :, :f + 1])
            x = x + ta.o(y.transpose(1, 2).reshape(b * s, 1, -1)).view(b, 1, s, -1)
            x = x + blk.mlp(blk.n3(x))
        return self.head(net.norm(x))[:, 0]

    @torch.no_grad()
    def generate(self, tokens, actions, n_ctx: int, steps: int = 25, temperature: float = 1.0, generator=None,
                 window: int | None = None):
        """Frame-by-frame MaskGIT decoding of frames n_ctx.. given context tokens and actions (B, T). Frame f sees
        frames max(0, f - window + 1)..f, as a model trained on clips of `window` frames does, the first of them
        without an action; the keys and values of the finished frames of a window are cached (frame_logits). A new
        window rewrites the cache from its first frame, so no frame outside it is ever attended to."""
        tokens = tokens.clone()
        b, t, s = tokens.shape
        window = min(window or t, t)
        lo, acts, past = -1, actions, self.new_cache(window)
        for f in range(n_ctx, t):
            if max(0, f + 1 - window) != lo:                          # a new window: the states of its frames before f
                lo = max(0, f + 1 - window)
                acts = actions[:, lo:lo + window].clone()
                acts[:, 0] = self.c.n_actions                         # no action into the first frame, as in training
                for j in range(lo, f):
                    self.frame_logits(tokens[:, j], acts, j - lo, past)
            cur = torch.full_like(tokens[:, f], self.mask_id)
            known = torch.zeros(b, s, dtype=torch.bool, device=tokens.device)
            for k in range(steps):
                lg = self.frame_logits(cur, acts, f - lo, past).float() / max(temperature, 1e-4)
                probs = lg.softmax(-1)
                sample = torch.multinomial(probs.view(-1, probs.shape[-1]), 1, generator=generator).view(b, s)
                conf = probs.gather(-1, sample[..., None])[..., 0]
                conf = torch.where(known, torch.full_like(conf, float("inf")), conf)
                n_keep = int(s * (1 - math.cos(math.pi / 2 * (k + 1) / steps)))
                if k == steps - 1:
                    n_keep = s
                thresh = conf.topk(max(n_keep, 1), dim=-1).values[:, -1:]
                newly = (conf >= thresh) & ~known
                cur = torch.where(newly, sample, cur)
                known |= newly
            tokens[:, f] = cur
            if f + 1 < t and max(0, f + 2 - window) == lo:            # the next frame shares the window: cache it
                self.frame_logits(cur, acts, f - lo, past)
        return tokens
