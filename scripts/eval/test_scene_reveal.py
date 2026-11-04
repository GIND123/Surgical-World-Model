#!/usr/bin/env python
"""Does a frame-wide learned action carry the appearance of what the moving endoscope reveals?

For each test window the measured endoscope motion of the predicted steps is applied to a map of ones on the grid of
the last context latent (stabilise(..., padding="zeros")); grid cells it no longer covers show content that the
context did not contain ("revealed"). The future is generated twice with the same noise: with the window's own
actions and with one learned action taken from another window. The share of the resulting latent change that falls
in the revealed cells, divided by their share of the grid, is near 1 for an action whose effect ignores the camera
motion and above 1 for one that describes newly revealed content. Measured for EG-SWM's scene action and its arm
actions, and for the monolithic and slot actions, on the windows whose revealed share is at least --min-area.
Writes outputs/eval/scene_reveal.json.

    python scripts/eval/test_scene_reveal.py [--limit 300]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_wm import EvalWindows, load_model, spread  # noqa: E402

from swm.data.windows import camera_scale  # noqa: E402
from swm.models.actions import region_from_boxes, stabilise  # noqa: E402
from swm.paths import OUTPUTS  # noqa: E402

TESTS = [  # run, learned action replaced, label
    ("S_arms_scene", "u_extra", "EG-SWM + scene: scene action"),
    ("S_arms_scene", "u_arms", "EG-SWM + scene: arm actions"),
    ("S_arms", "u_arms", "EG-SWM: arm actions"),
    ("S_global", "u_extra", "Monolithic latent + camera"),
    ("S_slots", "u_extra", "Slots + camera"),
]


def revealed(camera: torch.Tensor, n_ctx: int, h: int, w: int) -> torch.Tensor:
    """(B, T, 20) camera input -> (B, T - n_ctx, h, w): cells of each predicted step not covered by the content of
    the last context step, following the measured endoscope motion step by step."""
    scale = torch.from_numpy(camera_scale()).to(camera.device)
    cover = torch.ones(camera.shape[0], 1, h, w, device=camera.device)
    out = []
    for t in range(n_ctx, camera.shape[1]):
        cover = stabilise(cover, camera[:, t].float(), scale, padding="zeros")
        out.append(cover[:, 0] < 0.5)
    return torch.stack(out, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--n-ctx", type=int, default=2)
    ap.add_argument("--min-area", type=float, default=0.02)
    args = ap.parse_args()
    ds = EvalWindows("test")
    idx = spread(len(ds), args.limit)
    out = {"windows": int(len(idx)), "min_area": args.min_area, "tests": {}}
    for run, key, label in TESTS:
        model, meta = load_model(run, None)
        rows = []
        for s in range(0, len(idx), args.bs):
            items = [ds[int(i)] for i in idx[s:s + args.bs]]
            b = {k: torch.stack([it[k] for it in items]).cuda() for k in items[0]}
            b["region"] = region_from_boxes(b["boxes"], 14, 20, dilate=1)
            mask = revealed(b["camera"], args.n_ctx, 14, 20)                                 # B T' h w
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                acts = model.infer_actions(b["z"], b, stochastic=False)
                alt = dict(acts, **{key: acts[key].roll(1, 0)})
                g1, g2 = (torch.Generator(device="cuda").manual_seed(s) for _ in range(2))  # same noise for both
                z1 = model.generate(b, args.n_ctx, args.steps, actions=acts, generator=g1)
                z2 = model.generate(b, args.n_ctx, args.steps, actions=alt, generator=g2)
            d = (z1 - z2).float().abs().mean(2)[:, args.n_ctx:]                              # B T' h w
            for i in range(len(items)):
                area = float(mask[i].float().mean())
                if area >= args.min_area:
                    share = float((d[i] * mask[i]).sum() / d[i].sum().clamp_min(1e-8))
                    rows.append(dict(area=area, share=share, ratio=share / area))
        r = np.array([x["ratio"] for x in rows])
        a = np.array([x["area"] for x in rows])
        big = a >= np.median(a) if len(a) else a.astype(bool)
        out["tests"][label] = dict(run=run, action=key, n=len(rows), ratio_median=float(np.median(r)),
                                   ratio_iqr=np.quantile(r, [0.25, 0.75]).tolist(), area_median=float(np.median(a)),
                                   ratio_median_large_motion=float(np.median(r[big])),
                                   ratio_median_small_motion=float(np.median(r[~big])))
        print(label, json.dumps(out["tests"][label]), flush=True)
        del model
        torch.cuda.empty_cache()
    (OUTPUTS / "eval" / "scene_reveal.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
