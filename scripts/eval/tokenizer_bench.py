#!/usr/bin/env python
"""Reconstruction quality and throughput of candidate latent tokenizers on held-out SurgVU clips.

Clips: 9 consecutive 10 fps frames of the 224x320 endoscope field from validation cases (frames the audit
classed as tissue). Image VAEs encode frames independently; video VAEs encode the whole clip.
Writes outputs/tokenizer_bench/results.json.
"""

from __future__ import annotations

import argparse
import json
import time

import av
import cv2
import numpy as np
import pandas as pd
import torch

from swm.data import surgvu as sv
from swm.data.geometry import CACHE_H, CACHE_W, FIELD_ROWS, MODEL_H, MODEL_W, SRC_CROP
from swm.metrics.image import LPIPS, psnr, ssim
from swm.paths import CHECKPOINTS, OUTPUTS, ROOT, SURGVU24_VIDEOS


def field(rgb: np.ndarray) -> np.ndarray:
    x0, x1, y0, y1 = SRC_CROP
    img = cv2.resize(rgb[y0:y1, x0:x1], (CACHE_W, CACHE_H), interpolation=cv2.INTER_AREA)
    return cv2.resize(img[FIELD_ROWS[0]:FIELD_ROWS[1]], (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA)


def load_clips(n: int, length: int, seed: int) -> torch.Tensor:
    split = json.loads((ROOT / "configs" / "splits" / "surgvu_v1.json").read_text())
    fr = pd.read_csv(OUTPUTS / "audit" / "frames.csv")
    cand = fr[fr["case"].isin(split["val"]) & (fr["cls"] == "tissue")].sample(n, random_state=seed)
    clips = []
    for _, r in cand.iterrows():
        path = next(d["path"] for d in sv.parts_table()[r["case"]] if d["part"] == r["part"])
        want = [float(r["t"]) + i / 10 for i in range(length)]
        got = []
        with av.open(str(SURGVU24_VIDEOS / path)) as c:
            c.seek(int(max(0.0, want[0] - 2) * 1e6), backward=True)
            for f in c.decode(video=0):
                if f.time is not None and f.time + 1e-3 >= want[len(got)]:
                    got.append(field(f.to_ndarray(format="rgb24")))
                    if len(got) == length:
                        break
        if len(got) == length:
            clips.append(np.stack(got))
    return torch.from_numpy(np.stack(clips)).permute(0, 1, 4, 2, 3).float() / 255   # B T 3 H W


def build(name: str):
    import diffusers
    root = CHECKPOINTS / "vae" / name
    if name == "flux2":
        return diffusers.AutoencoderKLFlux2.from_pretrained(root / "vae"), "image"
    if name == "sdvae":
        return diffusers.AutoencoderKL.from_pretrained(root), "image"
    if name == "dcae":
        return diffusers.AutoencoderDC.from_pretrained(root), "image"
    if name in ("wan21", "wan22"):
        return diffusers.AutoencoderKLWan.from_pretrained(root / "vae"), "video"
    raise KeyError(name)


@torch.no_grad()
def roundtrip(vae, kind: str, x: torch.Tensor) -> tuple[torch.Tensor, tuple]:
    """x: B T 3 H W in [0, 1] -> reconstruction, latent shape per clip."""
    b, t = x.shape[:2]
    if kind == "image":
        flat = x.flatten(0, 1) * 2 - 1
        enc = vae.encode(flat)
        z = enc.latent if hasattr(enc, "latent") else enc.latent_dist.mode()
        rec = vae.decode(z).sample
        return ((rec.clamp(-1, 1) + 1) / 2).unflatten(0, (b, t)), (t, *z.shape[1:])
    vid = x.permute(0, 2, 1, 3, 4) * 2 - 1                                   # B 3 T H W
    z = vae.encode(vid).latent_dist.mode()
    rec = vae.decode(z).sample
    return ((rec.clamp(-1, 1) + 1) / 2).permute(0, 2, 1, 3, 4), tuple(z.shape[1:])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="*", default=["flux2", "sdvae", "wan21", "wan22", "dcae"])
    ap.add_argument("--clips", type=int, default=64)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()
    out = OUTPUTS / "tokenizer_bench"
    out.mkdir(parents=True, exist_ok=True)
    clips = load_clips(args.clips, 9, seed=0)
    print("clips", tuple(clips.shape), flush=True)
    lp = LPIPS()
    results = {}
    for name in args.names:
        vae, kind = build(name)
        vae = vae.to("cuda", torch.bfloat16).eval()
        ps, ss, ls = [], [], []
        torch.cuda.synchronize()
        t0 = time.time()
        for i in range(0, len(clips), args.batch):
            x = clips[i:i + args.batch].to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                rec, zshape = roundtrip(vae, kind, x.to(torch.bfloat16))
            a, b = rec.flatten(0, 1).float(), x.flatten(0, 1)
            ps.append(psnr(a, b).cpu())
            ss.append(ssim(a, b).cpu())
            ls.append(lp(a, b).cpu())
        torch.cuda.synchronize()
        fps = clips.shape[0] * clips.shape[1] / (time.time() - t0)
        r = dict(kind=kind, latent_shape_per_clip=list(zshape), psnr=float(torch.cat(ps).mean()),
                 ssim=float(torch.cat(ss).mean()), lpips=float(torch.cat(ls).mean()),
                 roundtrip_fps=fps, latent_numbers_per_frame=float(np.prod(zshape)) / clips.shape[1])
        results[name] = r
        print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
        del vae
        torch.cuda.empty_cache()
    (out / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
