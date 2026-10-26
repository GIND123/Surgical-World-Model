#!/usr/bin/env python
"""Case-level SurgVU split, stratified by camera arm, case length and dominant task.

Writes configs/splits/surgvu_v1.json. Frames within a case are strongly correlated, so cases never cross
splits. case_071 (missing part 1) is kept in train.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from swm.data import surgvu as sv
from swm.paths import ROOT

N_TEST, N_VAL, SEED = 20, 12, 0


def main() -> None:
    rows = []
    for case in sv.cases():
        tl = sv.timeline(case)
        tools, _ = sv.load_tools(case)
        cam = tools[tools["occupant"] == sv.CAMERA].assign(d=lambda d: d["t1"] - d["t0"])
        cam_arm = cam.groupby("arm")["d"].sum().idxmax() if len(cam) else "none"
        tasks = sv.load_tasks(case).assign(d=lambda d: d["t1"] - d["t0"])
        main_task = tasks.groupby("task")["d"].sum().idxmax() if len(tasks) else "none"
        rows.append(dict(case=case, hours=tl.duration / 3600, cam_arm=cam_arm, main_task=main_task))
    df = pd.DataFrame(rows)
    df["len_bin"] = pd.qcut(df["hours"], 3, labels=["short", "mid", "long"])
    rng = np.random.default_rng(SEED)
    df["stratum"] = df["cam_arm"] + "|" + df["len_bin"].astype(str)
    df = df.sample(frac=1.0, random_state=SEED).sort_values("stratum", kind="stable")
    # systematic sampling over the stratum-sorted list spreads test/val evenly across strata
    order = df["case"].tolist()
    eligible = [c for c in order if c != "case_071"]
    step_t = len(eligible) / N_TEST
    test = [eligible[int(i * step_t + rng.uniform(0, step_t))] for i in range(N_TEST)]
    rest = [c for c in eligible if c not in test]
    step_v = len(rest) / N_VAL
    val = [rest[int(i * step_v + rng.uniform(0, step_v))] for i in range(N_VAL)]
    train = sorted(c for c in order if c not in test and c not in val)
    split = dict(version="surgvu_v1", seed=SEED, train=train, val=sorted(val), test=sorted(test))
    out = ROOT / "configs" / "splits" / "surgvu_v1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(split, indent=1))
    for name in ("train", "val", "test"):
        sub = df[df["case"].isin(split[name])]
        print(f"{name:5s} n={len(sub):3d} hours={sub['hours'].sum():6.1f} "
              f"cam={sub['cam_arm'].value_counts().to_dict()} tasks={sub['main_task'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
