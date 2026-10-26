#!/usr/bin/env python
"""Index the training windows: 33 consecutive cached frames (3.3 s at 10 fps) that are all usable.

A frame is usable when it is not black (luma >= 25), not blank (variance of Laplacian >= 15) and either
coloured tissue (saturation >= 45) or Firefly fluorescence (> 2 % saturated green). Windows are
non-overlapping and aligned to the start of each part. Writes
  data/processed/surgvu/windows.csv        one row per window (case, part, start frame, split, stats)
  configs/stats/egomotion.json             robust scale of each ego-motion parameter (for normalisation)
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from swm.paths import PROCESSED, ROOT

META = PROCESSED / "surgvu" / "meta10"
WIN = 33


def usable(m) -> np.ndarray:
    tissue = m["sat"] >= 45
    fluo = m["green"] > 0.02
    return (m["luma"] >= 25) & (m["sharp"] >= 15) & (tissue | fluo)


def main() -> None:
    split = json.loads((ROOT / "configs" / "splits" / "surgvu_v1.json").read_text())
    which = {c: s for s in ("train", "val", "test") for c in split[s]}
    rows, ego_all = [], []
    for f in sorted(META.glob("*.npz")):
        case, part = f.stem.rsplit("_p", 1)
        m = np.load(f)
        ok = usable(m)
        ego = m["ego"]
        good_ego = ok & np.isfinite(ego).all(1) & (m["ego_inlier"] > 0.5)
        ego_all.append(ego[good_ego])
        for s in range(0, len(ok) - WIN + 1, WIN):
            if not ok[s:s + WIN].all():
                continue
            e = ego[s + 1:s + WIN]
            rows.append(dict(case=case, part=int(part), start=s, t_global=float(m["t_global"][s]), split=which[case],
                             fluo=float((m["green"][s:s + WIN] > 0.02).mean()), luma=float(m["luma"][s:s + WIN].mean()),
                             ego_valid=float(np.isfinite(e).all(1).mean()),
                             cam_speed=float(np.nanmean(np.abs(e[:, :2]).sum(1))) if np.isfinite(e).any() else np.nan,
                             ui_bottom=float(((m["bot_y"][s:s + WIN] < 90) & (m["bot_e"][s:s + WIN] > 0.06)).mean())))
    df = pd.DataFrame(rows)
    out = PROCESSED / "surgvu" / "windows.csv"
    df.to_csv(out, index=False)
    ego = np.concatenate(ego_all)
    q = np.nanquantile(np.abs(ego), [0.5, 0.9, 0.99], axis=0)
    mad = 1.4826 * np.median(np.abs(ego - np.median(ego, 0)), 0)
    stats = dict(params=["dx", "dy", "log_scale", "angle"], mad=mad.tolist(), abs_q50=q[0].tolist(),
                 abs_q90=q[1].tolist(), abs_q99=q[2].tolist(), n=int(len(ego)))
    (ROOT / "configs" / "stats").mkdir(parents=True, exist_ok=True)
    (ROOT / "configs" / "stats" / "egomotion.json").write_text(json.dumps(stats, indent=1))
    summ = df.groupby("split").agg(windows=("case", "size"), cases=("case", "nunique"))
    summ["hours"] = summ["windows"] * WIN / 10 / 3600
    print(summ.to_string())
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
