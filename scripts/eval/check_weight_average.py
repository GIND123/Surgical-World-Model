#!/usr/bin/env python
"""Does the weight average still lag at the end of a short run? Latent error of action-replay generation on the
first validation windows of an external dataset, with a run's final weights and with its weight average (the same
noise and 10 Euler steps as the validation during training). Runs on the CPU, next to the GPU queue.

    CUDA_VISIBLE_DEVICES="" python scripts/eval/check_weight_average.py --data suturebot SB_kin_scratch_f0.1
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from swm.data.external import ExternalWindows
from swm.models.world_model import WMConfig, WorldModel
from swm.paths import OUTPUTS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--data", default="suturebot")
    ap.add_argument("--windows", type=int, default=64)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    val = ExternalWindows(args.data, "val", synced_only=True)
    for name in args.runs:
        st = torch.load(OUTPUTS / "runs" / name / "ckpt_last.pt", map_location="cpu", weights_only=False)
        model = WorldModel(WMConfig.from_dict(st["config"]))
        res = {}
        for which in ("final", "average"):
            model.load_state_dict(st["model"])
            if which == "average":
                for p, e in zip(model.parameters(), st["ema"], strict=True):
                    p.data.copy_(e)
            model.eval()
            g = torch.Generator().manual_seed(1234)
            err, k = 0.0, 0
            for b in DataLoader(val, batch_size=16, shuffle=False):
                with torch.no_grad():
                    out = model.generate(b, n_ctx=2, steps=10, generator=g)
                err += (out[:, 2:].float() - b["z"][:, 2:]).pow(2).mean().item() * len(b["z"])
                k += len(b["z"])
                if k >= args.windows:
                    break
            res[which] = round(err / k, 4)
        print(name, "step", st["step"], res, f"average/final {res['average'] / res['final']:.2f}", flush=True)


if __name__ == "__main__":
    main()
