"""EG-SWM and its baselines: one dynamics transformer, different action interfaces.

action = "none"    no tokens at all (unconditional video prediction)
         "state"   robot-state tokens only (arm id + occupant + hand), no learned action
         "global"  robot-state tokens + one monolithic latent action per transition (continuous)
         "global_vq" as "global" but a discrete action from a 12-entry codebook (Genie / SurgWM style)
         "slots"   robot-state tokens + K unsupervised slot actions (FLAM style)
         "arms"    robot-state tokens carrying one latent action per instrument arm (EG-SWM)
         "kin"     as "arms" but each arm's action is mapped from its recorded kinematics by a small adapter
                   (kinematics-conditioned; initialise from an "arms" model to transfer unlabeled pre-training)
camera = True adds the explicit endoscope ego-motion to the per-frame conditioning.

Training: rectified flow on normalised VAE latents, x_tau = (1 - tau) x0 + tau eps, target v = eps - x0, with a
noise level per frame. Mixed schedule: clean (or slightly noised) context + future frames with a shared or
independent tau, or pure diffusion forcing (independent tau everywhere). Actions for every transition are
inferred from the clean latents by the inverse-dynamics head and bottlenecked by a KL penalty.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn

from swm.models.actions import ArmIDM, ArmTokens, CameraEncoder, GlobalIDM, SlotIDM, Trunk, VQGlobalIDM
from swm.models.dynamics import SIZES, DiTConfig, STDiT
from swm.models.layers import timestep_embedding

ACTIONS = ("none", "state", "global", "global_vq", "slots", "arms", "kin")


@dataclass
class WMConfig:
    size: str = "B"
    action: str = "arms"
    camera: bool = True
    latent_shape: tuple = (48, 14, 20)
    du: int = 8                    # per-arm / per-slot action dimension
    dg: int = 24                   # monolithic action dimension (= 3 instrument arms x du)
    n_slots: int = 4
    n_codes: int = 12              # codebook size of the discrete monolithic action
    vq_weight: float = 1.0
    kin_dim: int = 28              # 4 frame transitions x (dpos 3, drot 3, djaw 1) per latent step
    kl_weight: float = 1e-3
    idm_dim: int = 256
    idm_depth: int = 2
    idm_heads: int = 4
    p_drop_actions: float = 0.1    # classifier-free guidance on actions (camera + learned actions)
    p_context: float = 0.75        # clean-context training mode, otherwise pure diffusion forcing
    max_context: int = 3           # latent frames of context in context mode
    p_shared_tau: float = 0.5      # future frames share one tau (vs independent) in context mode
    p_noisy_context: float = 0.3   # context frames get tau ~ U(0, ctx_noise) instead of 0
    ctx_noise: float = 0.1
    grad_checkpoint: bool = False
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class WorldModel(nn.Module):
    def __init__(self, cfg: WMConfig):
        super().__init__()
        assert cfg.action in ACTIONS, cfg.action
        self.cfg = cfg
        c, h, w = cfg.latent_shape
        uses_tokens = cfg.action != "none"
        self.dit = STDiT(DiTConfig(in_channels=c, height=h, width=w, cross_attention=uses_tokens,
                                   grad_checkpoint=cfg.grad_checkpoint, **SIZES[cfg.size]))
        d = self.dit.cfg.dim
        self.time = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
        self.cam = CameraEncoder(d) if cfg.camera else None
        self.arm_tokens = ArmTokens(d, cfg.du if cfg.action in ("arms", "kin") else 0) if uses_tokens else None
        if cfg.action == "kin":
            self.kin_adapter = nn.Sequential(nn.Linear(cfg.kin_dim, 256), nn.SiLU(), nn.Linear(256, cfg.du))
        if cfg.action in ("global", "global_vq", "slots", "arms"):
            trunk = Trunk(c, cfg.idm_dim, cfg.idm_depth, cfg.idm_heads, h, w)
            if cfg.action == "arms":
                self.idm = ArmIDM(trunk, cfg.idm_dim, cfg.idm_heads, cfg.du)
            elif cfg.action == "global":
                self.idm = GlobalIDM(trunk, cfg.idm_dim, cfg.idm_heads, cfg.dg)
                self.extra_proj, self.extra_role = nn.Linear(cfg.dg, d), nn.Parameter(torch.zeros(1, 1, 1, d))
            elif cfg.action == "global_vq":
                self.idm = VQGlobalIDM(trunk, cfg.idm_dim, cfg.idm_heads, cfg.dg, cfg.n_codes)
                self.extra_proj, self.extra_role = nn.Linear(cfg.dg, d), nn.Parameter(torch.zeros(1, 1, 1, d))
            else:
                self.idm = SlotIDM(trunk, cfg.idm_dim, cfg.du, k=cfg.n_slots)
                self.extra_proj, self.extra_role = nn.Linear(cfg.du, d), nn.Parameter(torch.zeros(1, 1, cfg.n_slots, d))
        else:
            self.idm = None

    # ------------------------------------------------------------------ actions
    def infer_actions(self, z: torch.Tensor, batch: dict, stochastic: bool) -> dict:
        """Latent actions for every transition (t-1 -> t), placed at frame t; frame 0 gets zeros."""
        b, t = z.shape[:2]
        out = {"kl": z.new_zeros(())}
        if self.cfg.action == "kin":
            active = (batch["occupant"] >= 2) & (batch["kin"].abs().sum(-1) > 0)
            out["u_arms"] = self.kin_adapter(batch["kin"].to(z.dtype)) * active[..., None].to(z.dtype)
            return out
        if self.idm is None:
            return out
        zp, zn = z[:, :-1].flatten(0, 1), z[:, 1:].flatten(0, 1)
        if self.cfg.action == "arms":
            occ = batch["occupant"][:, 1:].flatten(0, 1)
            region = batch["region"][:, 1:].flatten(0, 1)
            u, kl, active = self.idm(zp, zn, occ, region, stochastic)
            n_active = active.sum().clamp_min(1)
            out["kl"] = kl.sum(-1).sum() / n_active
            out["u_arms"] = torch.cat([u.new_zeros(b, 1, *u.shape[1:]), u.view(b, t - 1, *u.shape[1:])], 1)
            out["active"] = active.view(b, t - 1, -1)
        elif self.cfg.action == "global_vq":
            u, vq_loss, idx = self.idm(zp, zn, stochastic)
            out["vq"] = vq_loss
            out["codes"] = idx.view(b, t - 1)
            u = u.view(b, t - 1, *u.shape[1:])
            out["u_extra"] = torch.cat([u.new_zeros(b, 1, *u.shape[2:]), u], 1)
        else:
            u, kl = self.idm(zp, zn, stochastic)
            out["kl"] = kl.sum(-1).mean()
            u = u.view(b, t - 1, *u.shape[1:])
            out["u_extra"] = torch.cat([u.new_zeros(b, 1, *u.shape[2:]), u], 1)
        return out

    def tokens(self, batch: dict, acts: dict, keep: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """keep: (B,) float, 0 drops every action (CFG) but keeps the robot state."""
        if self.arm_tokens is None:
            return None, None
        k = keep[:, None, None, None]
        u_arms = acts.get("u_arms")
        tok = self.arm_tokens(batch["occupant"], batch["hand"], None if u_arms is None else u_arms * k)
        mask = batch["occupant"] != 0
        if "u_extra" in acts:
            u = acts["u_extra"]
            if u.dim() == 3:
                u = u[:, :, None]
            extra = self.extra_proj(u.to(tok.dtype)) * k.to(tok.dtype) + self.extra_role
            b, t = extra.shape[:2]
            frame_has = torch.ones(b, t, extra.shape[2], dtype=torch.bool, device=tok.device)
            frame_has[:, 0] = False                                   # no transition into frame 0
            tok, mask = torch.cat([tok, extra], 2), torch.cat([mask, frame_has], 2)
        return tok, mask

    def cond(self, tau: torch.Tensor, batch: dict, keep: torch.Tensor) -> torch.Tensor:
        c = self.time(timestep_embedding(tau, 256).to(self.time[0].weight.dtype))
        if self.cam is not None:
            c = c + self.cam(batch["camera"].to(c.dtype) * keep[:, None, None].to(c.dtype))
        return c

    # ------------------------------------------------------------------ training
    def sample_tau(self, b: int, t: int, device) -> torch.Tensor:
        cfg = self.cfg
        tau = torch.sigmoid(torch.randn(b, t, device=device))                     # logit-normal, independent
        ctx_mode = torch.rand(b, device=device) < cfg.p_context
        n_ctx = torch.randint(1, cfg.max_context + 1, (b,), device=device)
        is_ctx = (torch.arange(t, device=device)[None] < n_ctx[:, None]) & ctx_mode[:, None]
        shared = (torch.rand(b, device=device) < cfg.p_shared_tau)[:, None] & ctx_mode[:, None]
        tau = torch.where(shared & ~is_ctx, tau[:, -1:].expand(-1, t), tau)
        noisy = (torch.rand(b, device=device) < cfg.p_noisy_context)[:, None]
        ctx_tau = torch.where(noisy, torch.rand(b, t, device=device) * cfg.ctx_noise, torch.zeros_like(tau))
        return torch.where(is_ctx, ctx_tau, tau), is_ctx

    def loss(self, batch: dict, kl_scale: float = 1.0) -> tuple[torch.Tensor, dict]:
        z = batch["z"]
        b, t = z.shape[:2]
        tau, is_ctx = self.sample_tau(b, t, z.device)
        eps = torch.randn_like(z)
        tt = tau[:, :, None, None, None].to(z.dtype)
        x = (1 - tt) * z + tt * eps
        acts = self.infer_actions(z, batch, stochastic=True)
        keep = (torch.rand(b, device=z.device) >= self.cfg.p_drop_actions).float()
        tok, mask = self.tokens(batch, acts, keep)
        v = self.dit(x, self.cond(tau, batch, keep), tok, mask)
        err = (v.float() - (eps - z).float()).pow(2).mean(dim=(2, 3, 4))           # (B, T)
        w = (~is_ctx).float()
        fm = (err * w).sum() / w.sum().clamp_min(1)
        loss = fm + self.cfg.kl_weight * kl_scale * acts["kl"]
        logs = {"fm": fm.item(), "kl": float(acts["kl"])}
        if "vq" in acts:
            loss = loss + self.cfg.vq_weight * acts["vq"]
            counts = torch.bincount(acts["codes"].flatten(), minlength=self.cfg.n_codes).float()
            p = counts / counts.sum()
            logs.update(vq=acts["vq"].item(), code_perplexity=float(torch.exp(-(p * (p + 1e-10).log()).sum())))
        return loss, {"loss": loss.item(), **logs}

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def generate(self, batch: dict, n_ctx: int, steps: int = 20, cfg_scale: float = 1.0,
                 actions: dict | None = None, ctx_tau: float = 0.0, generator=None) -> torch.Tensor:
        """Predict frames n_ctx.. of batch["z"] from its first n_ctx latents. Actions default to those inferred
        from the ground-truth latents (action replay); pass ``actions`` (same format as infer_actions) to
        control the rollout. Returns the full latent sequence (context kept)."""
        z = batch["z"]
        b, t = z.shape[:2]
        acts = actions if actions is not None else self.infer_actions(z, batch, stochastic=False)
        ones = torch.ones(b, device=z.device)
        tok, mask = self.tokens(batch, acts, ones)
        tok0, mask0 = self.tokens(batch, acts, torch.zeros_like(ones)) if cfg_scale != 1.0 else (None, None)
        x = torch.randn(z.shape, device=z.device, dtype=z.dtype, generator=generator)
        x[:, :n_ctx] = z[:, :n_ctx]
        future = torch.arange(t, device=z.device) >= n_ctx
        grid = torch.linspace(1.0, 0.0, steps + 1, device=z.device)
        for i in range(steps):
            tau = torch.where(future[None], grid[i], torch.full_like(grid[i], ctx_tau)).expand(b, t)
            v = self.dit(x, self.cond(tau, batch, ones), tok, mask)
            if cfg_scale != 1.0:
                v0 = self.dit(x, self.cond(tau, batch, torch.zeros_like(ones)), tok0, mask0)
                v = v0 + cfg_scale * (v - v0)
            step = (grid[i + 1] - grid[i]).to(x.dtype)
            x = torch.where(future[None, :, None, None, None], x + step * v.to(x.dtype), x)
        return x
