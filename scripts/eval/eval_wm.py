#!/usr/bin/env python
"""E1 pixel-space evaluation of a trained world model on the fixed held-out SurgVU windows.

From the first n_ctx latents (n_ctx=2 -> 5 real frames, 0.5 s) the model predicts the remaining frames of the
33-frame window (2.8 s) with actions inferred from the ground truth (action replay) and, for controllability,
with actions taken from another window. Frames are decoded with the Wan2.2 VAE and compared with the real
frames. Reference rows: repeat-last-context-frame and VAE reconstruction of the ground truth.

    python scripts/eval/eval_wm.py --run eg_swm_B [--limit 1000] [--fvd]
    python scripts/eval/eval_wm.py --reference      # baselines that need no model
Writes outputs/eval/<run>/<tag>.json (+ per-window .npz).
"""

from __future__ import annotations

import argparse
import json
import time

import diffusers
import numpy as np
import torch

from swm.data.windows import CACHE, LatentWindows, latent_stats
from swm.metrics.fvd import I3DFeatures, frechet
from swm.metrics.image import LPIPS, psnr, ssim
from swm.models.actions import region_from_boxes
from swm.models.world_model import WMConfig, WorldModel
from swm.paths import CHECKPOINTS, OUTPUTS, ROOT


class EvalWindows(LatentWindows):
    """The fixed evaluation windows, in the order of configs/eval/surgvu_<split>_windows.json."""

    def __init__(self, split: str):
        super().__init__(split)
        order = json.loads((ROOT / "configs" / "eval" / f"surgvu_{split}_windows.json").read_text())
        pos = {p["stem"]: i for i, p in enumerate(self.parts)}
        self.index = np.array([[pos[o["stem"]], o["window"]] for o in order], np.int64)


def load_model(run: str, ckpt: str | None) -> tuple[WorldModel, dict]:
    path = OUTPUTS / "runs" / run / (ckpt or "ckpt_last.pt")
    st = torch.load(path, map_location="cuda", weights_only=False)
    cfg = WMConfig(**{k: (tuple(v) if k == "latent_shape" else v) for k, v in st["config"].items()})
    model = WorldModel(cfg).cuda().eval()
    model.load_state_dict(st["model"])
    for p, e in zip(model.parameters(), st["ema"], strict=True):
        p.data.copy_(e)
    return model, {"step": st["step"], "ckpt": str(path)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--ckpt")
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-ctx", type=int, default=2)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--fvd", action="store_true")
    ap.add_argument("--tag", default="replay")
    args = ap.parse_args()
    torch.manual_seed(0)
    ds = EvalWindows(args.split)
    real_all = np.load(CACHE / f"eval_frames_{args.split}.npy", mmap_mode="r")
    n = min(len(ds), args.limit or len(ds))
    vae = diffusers.AutoencoderKLWan.from_pretrained(CHECKPOINTS / "vae/wan22/vae", torch_dtype=torch.bfloat16)
    vae = vae.cuda().eval()
    mean, std = (torch.from_numpy(a).cuda()[None, None, :, None, None] for a in latent_stats())
    lp = LPIPS()
    i3d = I3DFeatures() if args.fvd else None
    model, meta = (None, {}) if args.reference else load_model(args.run, args.ckpt)
    first = 4 * (args.n_ctx - 1) + 1                         # first predicted frame index
    names = ["recon", "copy_last"] if args.reference else ["pred", "pred_shuffled"]
    per = {k: {m: [] for m in ("psnr", "ssim", "lpips")} for k in names}
    feats = {k: [] for k in names + ["real"]}
    gen = torch.Generator(device="cuda").manual_seed(0)
    t0 = time.time()
    for s in range(0, n, args.bs):
        items = [ds[i] for i in range(s, min(n, s + args.bs))]
        b = {k: torch.stack([it[k] for it in items]).cuda() for k in items[0]}
        b["region"] = region_from_boxes(b["boxes"], 14, 20, dilate=1)
        real = torch.from_numpy(np.ascontiguousarray(real_all[s:s + len(items)])).cuda().float() / 255  # B T H W 3
        outs = {}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            if args.reference:
                outs["recon"] = b["z"]
                ctx_last = real[:, first - 1:first]
            else:
                outs["pred"] = model.generate(b, args.n_ctx, args.steps, args.cfg, generator=gen)
                acts = model.infer_actions(b["z"], b, stochastic=False)
                shuf = {k: (v.roll(1, 0) if k.startswith("u_") else v) for k, v in acts.items()}
                b_shuf = dict(b, camera=b["camera"].roll(1, 0))
                outs["pred_shuffled"] = model.generate(b_shuf, args.n_ctx, args.steps, args.cfg, actions=shuf,
                                                       generator=gen)
            frames = {}
            for k, z in outs.items():
                lat = (z.float() * std + mean).transpose(1, 2).to(torch.bfloat16)          # B C T h w
                frames[k] = vae.decode(lat).sample.float().clamp(-1, 1).add(1).div(2).permute(0, 2, 3, 4, 1)
        if args.reference:
            frames["copy_last"] = torch.cat([real[:, :first], ctx_last.expand(-1, 33 - first, -1, -1, -1)], 1)
        for k, f in frames.items():
            a = f[:, first:].flatten(0, 1).permute(0, 3, 1, 2)
            r = real[:, first:].flatten(0, 1).permute(0, 3, 1, 2)
            bt = (len(items), 33 - first)
            per[k]["psnr"].append(psnr(a, r).view(bt).cpu())
            per[k]["ssim"].append(ssim(a, r).view(bt).cpu())
            per[k]["lpips"].append(lp(a, r).view(bt).cpu())
            if i3d is not None:
                feats[k].append(i3d((f * 255).round()))
        if i3d is not None:
            feats["real"].append(i3d(real * 255))
        if s // args.bs % 25 == 0:
            print(f"{s + len(items)}/{n} windows, {time.time() - t0:.0f}s", flush=True)
    summary = dict(meta, split=args.split, n_windows=n, n_ctx=args.n_ctx, steps=args.steps, cfg=args.cfg,
                   first_pred_frame=first)
    arrays = {}
    for k in names:
        for m in ("psnr", "ssim", "lpips"):
            x = torch.cat(per[k][m]).numpy()                       # (N, horizon)
            arrays[f"{k}_{m}"] = x
            summary[f"{k}_{m}"] = float(x.mean())
            summary[f"{k}_{m}_by_frame"] = x.mean(0).round(4).tolist()
        if i3d is not None:
            summary[f"{k}_fvd"] = frechet(np.concatenate(feats["real"]), np.concatenate(feats[k]))
    if not args.reference:
        summary["delta_psnr"] = summary["pred_psnr"] - summary["pred_shuffled_psnr"]
    out = OUTPUTS / "eval" / ("reference" if args.reference else args.run)
    out.mkdir(parents=True, exist_ok=True)
    tag = "reference" if args.reference else args.tag
    (out / f"{tag}.json").write_text(json.dumps(summary, indent=1))
    np.savez_compressed(out / f"{tag}.npz", **arrays)
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith("by_frame")}, indent=1))


if __name__ == "__main__":
    main()
