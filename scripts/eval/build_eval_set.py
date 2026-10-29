#!/usr/bin/env python
"""Fix the held-out evaluation windows once and cache their real frames.

Samples the same number of windows from every test case (seeded), stores the list in
configs/eval/surgvu_<split>_windows.json, and extracts the real 224x320 endoscope-field frames of each window
from the 10 fps cache into data/processed/surgvu/eval_frames_<split>.npy (N, 33, 224, 320, 3) uint8, so every
method is scored against identical ground truth (real frames, not VAE reconstructions). Drawn windows that show
the recorder's colour bars (scripts/data/flag_no_signal.py) are replaced by other windows of the same case, so
every other window stays as first drawn; frames of windows already in the previous list are reused. Cases in
EXCLUDE contain no surgery and are left out of the evaluation (the split itself is unchanged).

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
EXCLUDE = {  # checked by eye
    "case_126": "no surgery: 78 % of the windows show colour bars, the rest the endoscope idle against a drape "
                "or a still training board",
}


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
    no_signal = np.load(CACHE / "no_signal.npz")
    chosen = []
    for case in sorted(by_case):
        pool = by_case[case]
        idx = set(rng.choice(len(pool), min(args.per_case, len(pool)), replace=False).tolist())
        if case in EXCLUDE:                  # still drawn, so that the windows of later cases do not change
            print(f"{case}: excluded, {EXCLUDE[case]}", flush=True)
            continue
        bad = {i for i in idx if no_signal[pool[i][0]][pool[i][1]]}
        if bad:
            spare = [i for i in range(len(pool)) if i not in idx and not no_signal[pool[i][0]][pool[i][1]]]
            new = np.random.default_rng([args.seed, int(case.split("_")[1])]).choice(spare, len(bad), replace=False)
            idx = (idx - bad) | set(new.tolist())
            print(f"{case}: {len(bad)} colour-bar windows replaced", flush=True)
        chosen += [pool[i] for i in sorted(idx)]
    out = ROOT / "configs" / "eval" / f"surgvu_{args.split}_windows.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    dst = CACHE / f"eval_frames_{args.split}.npy"
    prev = json.loads(out.read_text()) if out.exists() and dst.exists() else []
    reuse = {(p["stem"], p["window"]): k for k, p in enumerate(prev)}
    old_frames = np.load(dst, mmap_mode="r") if prev else None
    tmp = dst.with_suffix(".tmp.npy")
    frames = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(chosen), 33, MODEL_H, MODEL_W, 3))
    by_stem = defaultdict(list)
    for k, (stem, w) in enumerate(chosen):
        if (stem, w) in reuse:
            frames[k] = old_frames[reuse[(stem, w)]]
        else:
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
    del frames, old_frames
    tmp.replace(dst)
    out.write_text(json.dumps([{"stem": s, "window": int(w)} for s, w in chosen], indent=0))
    reused = len(chosen) - sum(map(len, by_stem.values()))
    print(f"{len(chosen)} windows from {len({s.rsplit('_p', 1)[0] for s, _ in chosen})} cases -> {out} "
          f"({reused} frame sets reused)")


if __name__ == "__main__":
    main()
