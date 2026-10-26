#!/usr/bin/env python
"""Robot-configuration and task statistics on the global case timeline (outputs/audit/timeline_summary.json)."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd

from swm.data import surgvu as sv
from swm.paths import OUTPUTS

OUT = OUTPUTS / "audit"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    qc_tot, cam_arm_s, n_inst_s, inst_s, task_s = Counter(), Counter(), Counter(), Counter(), Counter()
    per_case, descriptions, raw_names = [], set(), Counter()
    exchanges = cam_moves = 0
    total_s = labelled_task_s = 0.0
    for case in sv.cases():
        tl = sv.timeline(case)
        tools, qc = sv.load_tools(case)
        qc_tot.update(qc)
        tasks = sv.load_tasks(case)
        occ = sv.arm_occupancy(tools, tl.duration, hz=1.0)          # [T, 4]
        trk = sv.task_track(tasks, tl.duration, hz=1.0)
        total_s += len(occ)
        cam = occ == sv.OCC_INDEX[sv.CAMERA]
        n_cam = cam.sum(1)
        for a, arm in enumerate(sv.ARMS):
            cam_arm_s[arm] += int(cam[:, a].sum())
        cam_arm_s["none"] += int((n_cam == 0).sum())
        cam_arm_s["multiple"] += int((n_cam > 1).sum())
        inst = (occ >= 2)
        for k, v in Counter(inst.sum(1).tolist()).items():
            n_inst_s[k] += v
        for name, i in sv.OCC_INDEX.items():
            if i >= 2:
                inst_s[name] += int((occ == i).sum())
        # an exchange = an arm's instrument identity changes between two non-empty states
        cam_arm = np.where(n_cam == 1, cam.argmax(1), -1)
        moves = int(((cam_arm[1:] != cam_arm[:-1]) & (cam_arm[1:] >= 0) & (cam_arm[:-1] >= 0)).sum())
        cam_moves += moves
        ex_case = 0
        for a in range(4):
            seq = occ[:, a]
            nz = seq[seq >= 2]
            ex_case += int((nz[1:] != nz[:-1]).sum())
        exchanges += ex_case
        for k, v in Counter(trk[trk >= 0].tolist()).items():
            task_s[sv.TASKS[k]] += v
        labelled_task_s += float((trk >= 0).sum())
        descriptions.update(tasks["description"].dropna().unique())
        raw_names.update(tasks["raw_name"].tolist())
        per_case.append(dict(case=case, hours=tl.duration / 3600, camera_arm_changes=moves, exchanges=ex_case,
                             main_camera_arm=sv.ARMS[int(cam.sum(0).argmax())] if cam.any() else None,
                             camera_logged_frac=float((n_cam >= 1).mean()), task_frac=float((trk >= 0).mean()),
                             **{f"qc_{k}": v for k, v in qc.items()}))
    pc = pd.DataFrame(per_case)
    pc.to_csv(OUT / "timeline_cases.csv", index=False)
    summary = dict(
        total_hours=total_s / 3600, qc=dict(qc_tot),
        camera_arm_hours={k: round(v / 3600, 2) for k, v in cam_arm_s.items()},
        main_camera_arm_cases=pc["main_camera_arm"].value_counts().to_dict(),
        camera_arm_changes_total=cam_moves, cases_with_camera_arm_change=int((pc["camera_arm_changes"] > 0).sum()),
        instruments_installed_hours={int(k): round(v / 3600, 2) for k, v in sorted(n_inst_s.items())},
        instrument_hours={k: round(v / 3600, 2) for k, v in sorted(inst_s.items(), key=lambda x: -x[1])},
        instrument_exchanges_total=exchanges, exchanges_per_hour=exchanges / (total_s / 3600),
        task_hours={k: round(v / 3600, 2) for k, v in sorted(task_s.items(), key=lambda x: -x[1])},
        task_labelled_hours=labelled_task_s / 3600, task_labelled_frac=labelled_task_s / total_s,
        n_unique_task_descriptions=len(descriptions), n_raw_task_names=len(raw_names),
        top_raw_task_names=dict(raw_names.most_common(25)),
        camera_logged_frac_median=float(pc["camera_logged_frac"].median()),
    )
    (OUT / "timeline_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
