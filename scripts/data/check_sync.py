#!/usr/bin/env python
"""Estimate the video-kinematics lag of every episode of a prepared external dataset and write
data/processed/<name>/sync.json and signals_synced.npz (lag-corrected signals plus a per-window ``synced`` flag).
See swm.data.sync for the method and the acceptance rule.

    python scripts/data/check_sync.py crcd suturebot
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from swm.data import sync
from swm.paths import PROCESSED


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    args = ap.parse_args()
    for name in args.names:
        root = PROCESSED / name
        rows = pd.read_csv(root / "windows.csv")
        sig = dict(np.load(root / "signals.npz"))
        est = sync.estimate(rows, np.load(root / "latents.npy", mmap_mode="r"), sig["state"])
        shifted, synced = sync.apply(rows, sig, est)
        np.savez_compressed(root / "signals_synced.npz", **shifted, synced=synced)
        (root / "sync.json").write_text(json.dumps({str(k): v for k, v in est.items()}, indent=1))
        print(f"== {name}: {sum(v['ok'] for v in est.values())}/{len(est)} episodes synced, "
              f"{synced.mean():.2f} of windows")
        pooled = [v for v in est.values() if v["pooled"]]
        if pooled:
            v = pooled[0]
            print(f"{len(pooled)} short episodes pooled: lag {v['lag'] / 10:+.1f}s r={v['pooled_corr']:.2f}, "
                  f"{v['pooled_agreement']:.2f} of episodes peak within 0.5 s -> {'ok' if v['ok'] else 'BAD'}")
        for ep, v in est.items():
            if v["pooled"]:
                continue
            split = rows.loc[rows["episode"] == ep, "split"].iloc[0]
            state = "ok " if v["ok"] else "BAD"
            print(f"ep {ep:4d} {split:5s} {v['minutes']:5.1f} min  {state} lag {v['lag'] / 10:+5.1f}s "
                  f"(global {v['global_lag'] / 10:+5.1f}s r={v['global_corr']:.2f}, r@0={v['corr_at_0']:.2f}, "
                  f"chunks agree {v['chunk_agreement']:.2f})")


if __name__ == "__main__":
    main()
