#!/usr/bin/env python
"""Export a held-out sample window (context + future frames, robot state, ego-motion) for README/paper figures.

Picks a test-split case and a minute that the frame audit classed as tissue, with >= 2 instruments mounted,
then saves 224x320 endoscope-field crops (no console UI) to docs/figures/samples/ and the metadata to
sample.json. Frames are from the SurgVU dataset (Zia et al., 2025) and are shown for illustration only.
"""

from __future__ import annotations

import argparse
import json

import av
import cv2
import numpy as np
import pandas as pd

from swm.data import egomotion
from swm.data import surgvu as sv
from swm.data.geometry import CACHE_H, CACHE_W, FIELD_ROWS, SRC_CROP
from swm.paths import OUTPUTS, ROOT, SURGVU24_VIDEOS

OUT = ROOT / "docs" / "figures" / "samples"


def field(rgb: np.ndarray) -> np.ndarray:
    x0, x1, y0, y1 = SRC_CROP
    img = cv2.resize(rgb[y0:y1, x0:x1], (CACHE_W, CACHE_H), interpolation=cv2.INTER_AREA)
    return cv2.resize(img[FIELD_ROWS[0]:FIELD_ROWS[1]], (320, 224), interpolation=cv2.INTER_AREA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-context", type=int, default=4)
    ap.add_argument("--n-future", type=int, default=4)
    ap.add_argument("--stride", type=float, default=0.4, help="seconds between exported frames")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    split = json.loads((ROOT / "configs" / "splits" / "surgvu_v1.json").read_text())
    frames = pd.read_csv(OUTPUTS / "audit" / "frames.csv")
    cand = frames[frames["case"].isin(split["test"]) & (frames["cls"] == "tissue") & (frames["sharp"] > 80)
                  & (frames["sat"] > 70)]
    rng = np.random.default_rng(args.seed)
    for _ in range(200):
        row = cand.iloc[rng.integers(len(cand))]
        case, part, t_local = row["case"], int(row["part"]), float(row["t"])
        tl = sv.timeline(case)
        t_glob = tl.offsets[part] + t_local
        tools, _ = sv.load_tools(case)
        live = tools[(tools["t0"] <= t_glob) & (tools["t1"] >= t_glob)]
        state = {arm: live[live["arm"] == arm].sort_values("t0")["occupant"].iloc[-1]
                 for arm in sv.ARMS if (live["arm"] == arm).any()}
        if sum(o not in (sv.CAMERA, sv.UNKNOWN) for o in state.values()) >= 2 and sv.CAMERA in state.values():
            break
    n = args.n_context + args.n_future
    want = [t_local + i * args.stride for i in range(n)]
    path = next(d["path"] for d in sv.parts_table()[case] if d["part"] == part)
    got = []
    with av.open(str(SURGVU24_VIDEOS / path)) as c:
        c.seek(int(max(0.0, t_local - 2.0) * 1e6), backward=True)
        for f in c.decode(video=0):
            if f.time is not None and f.time + 1e-3 >= want[len(got)]:
                got.append(field(f.to_ndarray(format="rgb24")))
                if len(got) == n:
                    break
    OUT.mkdir(parents=True, exist_ok=True)
    names = [f"ctx{i}" for i in range(args.n_context)] + [f"fut{i}" for i in range(args.n_future)]
    for name, img in zip(names, got, strict=True):
        cv2.imwrite(str(OUT / f"{name}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    grays = [cv2.cvtColor(g, cv2.COLOR_RGB2GRAY) for g in got]
    ego = [egomotion.estimate(a, b)[0].round(4).tolist() for a, b in zip(grays[:-1], grays[1:], strict=True)]
    tasks = sv.load_tasks(case)
    task = tasks[(tasks["t0"] <= t_glob) & (tasks["t1"] >= t_glob)]["task"].tolist()
    meta = dict(case=case, part=part, t_local=t_local, t_global=t_glob, stride_s=args.stride, frames=names,
                robot_state=state, task=task[0] if task else None, egomotion_between_frames=ego,
                source="SurgVU dataset (Zia et al., arXiv:2501.09209), test split, endoscope field only")
    (OUT / "sample.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
