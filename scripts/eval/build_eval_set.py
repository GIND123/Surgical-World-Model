#!/usr/bin/env python
"""Fix the held-out evaluation windows once and cache their real frames.

Samples the same number of windows from every test case (seeded), stores the list in
configs/eval/surgvu_<split>_windows.json, and extracts the real 224x320 endoscope-field frames of each window
from the 10 fps cache into data/processed/surgvu/eval_frames_<split>.npy (N, 33, 224, 320, 3) uint8, so every
method is scored against identical ground truth (real frames, not VAE reconstructions).

    python scripts/eval/build_eval_set.py --split test --per-case 100
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import av
import cv2
import numpy as np

from swm.data.geometry import FIELD_ROWS, MODEL_H, MODEL_W
from swm.paths import PROCESSED, ROOT

CACHE = PROCESSED / "surgvu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--per-case", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sp = json.loads((ROOT / "configs" / "splits" / "surgvu_v1.json").read_text())
    rng = np.random.default_rng(args.seed)
    by_case = defaultdict(list)
    for f in sorted((CACHE / "conditioning").glob("*.npz")):
        case = f.stem.rsplit("_p", 1)[0]
        if case in sp[args.split]:
            n = len(np.load(CACHE / "latents" / f"{f.stem}.starts.npy"))
            by_case[case] += [(f.stem, w) for w in range(n)]
    missing = sorted(set(sp[args.split]) - set(by_case))
    if missing:
        raise SystemExit(f"cases not processed yet: {missing}")
    chosen = []
    for case in sorted(by_case):
        pool = by_case[case]
        idx = rng.choice(len(pool), min(args.per_case, len(pool)), replace=False)
        chosen += [pool[i] for i in sorted(idx)]
    out = ROOT / "configs" / "eval" / f"surgvu_{args.split}_windows.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([{"stem": s, "window": int(w)} for s, w in chosen], indent=0))
    frames = np.lib.format.open_memmap(CACHE / f"eval_frames_{args.split}.npy", mode="w+", dtype=np.uint8,
                                       shape=(len(chosen), 33, MODEL_H, MODEL_W, 3))
    by_stem = defaultdict(list)
    for k, (stem, w) in enumerate(chosen):
        by_stem[stem].append((k, w))
    for stem, items in by_stem.items():
        starts = np.load(CACHE / "latents" / f"{stem}.starts.npy")
        want = {int(starts[w]) + j: (k, j) for k, w in items for j in range(33)}
        with av.open(str(CACHE / "frames10" / f"{stem}.mp4")) as c:
            c.streams.video[0].thread_type = "AUTO"
            last = max(want)
            for i, fr in enumerate(c.decode(video=0)):
                if i in want:
                    k, j = want[i]
                    img = fr.to_ndarray(format="rgb24")[FIELD_ROWS[0]:FIELD_ROWS[1]]
                    frames[k, j] = cv2.resize(img, (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA)
                if i >= last:
                    break
        print(f"{stem}: {len(items)} windows", flush=True)
    frames.flush()
    print(f"{len(chosen)} windows from {len(by_case)} cases -> {out}")


if __name__ == "__main__":
    main()
