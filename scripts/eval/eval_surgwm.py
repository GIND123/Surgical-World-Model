#!/usr/bin/env python
"""E1 fidelity of the SurgWM re-implementation (scripts/train/train_surgwm.py) on the fixed held-out SurgVU windows.

Same windows, context and horizon as eval_wm.py: from the first 5 real frames (0.5 s) the dynamics model predicts
the remaining 28 frames (2.8 s), with the latent actions that its action model infers from the real clip (action
replay) and, for controllability, with the actions of another window. SurgWM works at 120x180: real frames are
resized down for its input, and its predictions are resized up to 224x320 and compared with the real frames, like
every other row. Its models were trained on 16-frame clips: the clip is tokenised and its actions inferred in
16-frame windows, and each predicted frame sees at most the 15 frames before it (25 MaskGIT steps per frame,
temperature 1, as in the paper). Also reports the reconstruction of its own tokenizer (its ceiling). Writes
outputs/eval/<name>/replay.json (+ per-window .npz) in the format of eval_wm.py.

With --control, the tests of eval_control.py that apply to SurgWM, on its control windows: locality (its
frame-wide action replaced by another window's, sampled with the reference's seed; the change is measured in every
instrument's region, as for the other frame-wide actions) and embodiment consistency of the replayed future
(control.json and control_rows.json in the format of eval_control.py). It has no endoscope action to command.

    python scripts/eval/eval_surgwm.py [--name surgwm] [--limit 1000] [--fvd]
    python scripts/eval/eval_surgwm.py --control --limit 300 --tag control
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_control import embodiment, summarise, where_change_lands  # noqa: E402
from eval_wm import EvalWindows, spread  # noqa: E402

from swm.baselines.surgwm import LatentActionModel, MaskGITDynamics, SurgWMConfig, VideoVQVAE  # noqa: E402
from swm.data.windows import CACHE  # noqa: E402
from swm.metrics.fvd import I3DFeatures, frechet  # noqa: E402
from swm.metrics.image import LPIPS, psnr, ssim  # noqa: E402
from swm.paths import DETECTOR, OUTPUTS  # noqa: E402


def load(cls, c: SurgWMConfig, name: str, stage: str):
    st = torch.load(OUTPUTS / "runs" / f"{name}_{stage}" / "ckpt_last.pt", map_location="cuda", weights_only=False)
    model = cls(c).cuda().eval()
    model.load_state_dict(st["model"])
    return model, st["step"]


def inputs(real: torch.Tensor, c: SurgWMConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Real frames (B, T, H, W, 3) in [0, 1] -> the tokenizer's and the action model's inputs in [-1, 1]."""
    bsz, t = real.shape[:2]
    x = F.interpolate(real.permute(0, 1, 4, 2, 3).flatten(0, 1) * 2 - 1, size=(c.height, c.width), mode="area")
    small = F.interpolate(x, size=(c.lam_height, c.lam_width), mode="area")
    return x.view(bsz, t, 3, c.height, c.width), small.view(bsz, t, 3, c.lam_height, c.lam_width)


def frames(tok: VideoVQVAE, codes: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Token codes (B, T, ...) -> decoded frames resized to the evaluation size, (B, T, h, w, 3) in [0, 1]."""
    bsz, t = codes.shape[:2]
    y = tok.decode_codes(codes).float().flatten(0, 1)                                   # (B*T, 3, 120, 180)
    y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
    return y.clamp(-1, 1).add(1).div(2).view(bsz, t, 3, h, w).permute(0, 1, 3, 4, 2)


def control(args, c: SurgWMConfig, tok, lam, dyn, ds, real_all, step: dict) -> None:
    """Locality and embodiment consistency on the windows of eval_control.py (see the module docstring)."""
    from ultralytics import YOLO
    det = YOLO(str(DETECTOR))
    idx = spread(len(ds), args.limit)
    n, first = len(idx), args.n_ctx
    n_lat = (first - 1) // 4 + 1                          # context latents of eval_control.py (their boxes are skipped)
    res = {"locality": [], "ecs": [], "ecs_windows": 0}
    for s in range(0, n, args.bs):
        items = [ds[i] for i in idx[s:s + args.bs]]
        boxes = np.stack([it["boxes"].numpy() for it in items])
        occ = np.stack([it["occupant"].numpy() for it in items])
        real = torch.from_numpy(np.ascontiguousarray(real_all[idx[s:s + args.bs]])).cuda().float() / 255
        bsz, _, h, w, _ = real.shape
        where = [dict(window=int(i), case=ds.parts[ds.index[i, 0]]["stem"].rsplit("_p", 1)[0])
                 for i in idx[s:s + bsz]]
        x, small = inputs(real, c)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            codes, _, _ = tok.encode(x)
            a, _, _ = lam.actions(small)
            acts = torch.cat([torch.full_like(a[:, :1], c.n_actions), a], 1)
            px = {}
            for k, ak in (("ref", acts), ("alt", torch.cat([acts[:, :1], acts[:, 1:].roll(1, 0)], 1))):
                z = dyn.generate(codes, ak, first, args.mask_steps, args.temperature, window=c.clip,
                                 generator=torch.Generator(device="cuda").manual_seed(s))     # the reference's seed
                px[k] = frames(tok, z, h, w)
        diff = (px["alt"] - px["ref"]).abs().mean(-1)[:, first:].cpu().numpy()            # B T H W
        ref, real = px["ref"].cpu().numpy(), real.cpu().numpy()
        for i in range(bsz):
            res["locality"] += [dict(r, **where[i]) for r in where_change_lands(diff[i], boxes[i], occ[i], range(4),
                                                                                 n_lat)]
            rows = embodiment(det, ref[i][first:], real[i][first:], occ[i, -1].tolist())
            if rows is not None:
                res["ecs_windows"] += 1
                res["ecs"] += [dict(r, **where[i]) for r in rows]
        print(f"{s + bsz}/{n}", flush=True)
    summary = dict(step=step, run=args.name, n_windows=n, tests=["ecs", "locality"], **summarise(res))
    out = OUTPUTS / "eval" / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.tag}.json").write_text(json.dumps(summary, indent=1))
    (out / f"{args.tag}_rows.json").write_text(json.dumps(res))
    print(json.dumps(summary, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="surgwm")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-ctx", type=int, default=5, help="real context frames (5 = the 2 context latents of eval_wm)")
    ap.add_argument("--mask-steps", type=int, default=25, help="MaskGIT decoding steps per frame")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--fvd", action="store_true")
    ap.add_argument("--tag", default="replay")
    ap.add_argument("--control", action="store_true", help="locality and embodiment consistency instead of fidelity")
    args = ap.parse_args()
    torch.manual_seed(0)
    c = SurgWMConfig()
    tok, s_tok = load(VideoVQVAE, c, args.name, "tokenizer")
    lam, s_lam = load(LatentActionModel, c, args.name, "lam")
    dyn, s_dyn = load(MaskGITDynamics, c, args.name, "dynamics")
    ds = EvalWindows(args.split)
    real_all = np.load(CACHE / f"eval_frames_{args.split}.npy", mmap_mode="r")
    if args.control:
        control(args, c, tok, lam, dyn, ds, real_all, {"tokenizer": s_tok, "lam": s_lam, "dynamics": s_dyn})
        return
    idx = spread(len(ds), args.limit)
    n, first = len(idx), args.n_ctx
    lp = LPIPS()
    i3d = I3DFeatures() if args.fvd else None
    names = ["pred", "pred_shuffled", "recon"]
    per = {k: {m: [] for m in ("psnr", "ssim", "lpips")} for k in names}
    feats = {k: [] for k in names + ["real"]}
    gen = torch.Generator(device="cuda").manual_seed(0)
    t0 = time.time()
    for s in range(0, n, args.bs):
        real = torch.from_numpy(np.ascontiguousarray(real_all[idx[s:s + args.bs]])).cuda().float() / 255  # B T H W 3
        bsz, t, h, w, _ = real.shape
        x, small = inputs(real, c)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            codes, _, _ = tok.encode(x)
            a, _, _ = lam.actions(small)
            acts = torch.cat([torch.full_like(a[:, :1], c.n_actions), a], 1)           # action into frame t
            shuf = torch.cat([acts[:, :1], acts[:, 1:].roll(1, 0)], 1)                   # another window's actions
            out = {"recon": codes}
            for k, ak in (("pred", acts), ("pred_shuffled", shuf)):
                out[k] = dyn.generate(codes, ak, first, args.mask_steps, args.temperature, generator=gen,
                                      window=c.clip)
            px = {k: frames(tok, z, h, w) for k, z in out.items()}
        for k, f in px.items():
            a_ = f[:, first:].flatten(0, 1).permute(0, 3, 1, 2)
            r = real[:, first:].flatten(0, 1).permute(0, 3, 1, 2)
            bt = (bsz, t - first)
            per[k]["psnr"].append(psnr(a_, r).view(bt).cpu())
            per[k]["ssim"].append(ssim(a_, r).view(bt).cpu())
            per[k]["lpips"].append(lp(a_, r).view(bt).cpu())
            if i3d is not None:
                feats[k].append(i3d((f * 255).round()))
        if i3d is not None:
            feats["real"].append(i3d(real * 255))
        if s // args.bs % 25 == 0:
            print(f"{s + bsz}/{n} windows, {time.time() - t0:.0f}s", flush=True)
    summary = dict(step={"tokenizer": s_tok, "lam": s_lam, "dynamics": s_dyn}, data="surgvu", split=args.split,
                   n_windows=n, n_ctx_frames=first, mask_steps=args.mask_steps, temperature=args.temperature,
                   window=c.clip, first_pred_frame=first, resolution=[c.height, c.width],
                   seconds=round(time.time() - t0), peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1))
    arrays = {"window": idx,
              "case": np.array([ds.parts[ds.index[i, 0]]["stem"].rsplit("_p", 1)[0] for i in idx])}
    for k in names:
        for m in ("psnr", "ssim", "lpips"):
            v = torch.cat(per[k][m]).numpy()                                              # (N, horizon)
            arrays[f"{k}_{m}"] = v
            summary[f"{k}_{m}"] = float(v.mean())
            summary[f"{k}_{m}_by_frame"] = v.mean(0).round(4).tolist()
        if i3d is not None:
            summary[f"{k}_fvd"] = frechet(np.concatenate(feats["real"]), np.concatenate(feats[k]))
    summary["delta_psnr"] = summary["pred_psnr"] - summary["pred_shuffled_psnr"]
    out_dir = OUTPUTS / "eval" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.tag}.json").write_text(json.dumps(summary, indent=1))
    np.savez_compressed(out_dir / f"{args.tag}.npz", **arrays)
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith("by_frame")}, indent=1))


if __name__ == "__main__":
    main()
