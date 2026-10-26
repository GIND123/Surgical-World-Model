#!/usr/bin/env python
"""Does the explicit camera action measure real endoscope motion? Validation on CRCD (dVRK ECM kinematics + pedals).

Per 10 fps frame transition: the video-estimated ego-motion (dx, dy, log s, angle) vs the change of the four
ECM joints, and the console pedals. Reports which pedal channel is the camera pedal (the one during which the
ECM actually moves), per-axis correlations, linear R^2 between ego-motion and ECM motion on held-out
episodes, and how well ego-motion magnitude detects camera-pedal transitions (ROC AUC).

    python scripts/eval/validate_camera_action.py --data crcd
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, roc_auc_score

from swm.paths import OUTPUTS, PROCESSED


def transitions(cond: dict, sig: dict) -> dict:
    """Flatten windows into per-transition arrays (frame j-1 -> j, j = 1..32)."""
    cam = cond["camera"].reshape(len(cond["camera"]), 9, 4, 4)            # (N, step, sub-transition, param)
    ok = np.repeat(cond["cam_ok"][:, 1:], 4, axis=1)                      # (N, 32)
    ego = cam[:, 1:].reshape(len(cam), 32, 4)                            # transitions 1..32 in order
    decm = np.diff(sig["ecm"], axis=1)                                   # (N, 32, 4)
    ped = sig["pedals"][:, 1:] > 0.5                                     # (N, 32, 3)
    return dict(ego=ego, ok=ok, decm=decm, ped=ped)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="crcd")
    args = ap.parse_args()
    root = PROCESSED / args.data
    rows = pd.read_csv(root / "windows.csv")
    tr = transitions(dict(np.load(root / "conditioning.npz")), dict(np.load(root / "signals.npz")))
    split = rows["split"].to_numpy()
    out = {"data": args.data}
    mag = np.abs(tr["decm"]).sum(-1)                                    # ECM motion magnitude per transition
    ratios = [float(mag[tr["ped"][..., c]].mean() / max(mag[~tr["ped"][..., c]].mean(), 1e-12))
              for c in range(tr["ped"].shape[-1])]
    cam_col = int(np.argmax(ratios))
    out["pedal_ecm_motion_ratio"] = ratios
    out["camera_pedal_column"] = cam_col
    out["camera_pedal_fraction"] = float(tr["ped"][..., cam_col].mean())

    def flat(mask_split):
        m = (split == mask_split)[:, None] & tr["ok"]
        return tr["ego"][m], tr["decm"][m], tr["ped"][..., cam_col][m]

    e_tr, d_tr, p_tr = flat("train")
    e_te, d_te, p_te = flat("test")
    out["n_transitions"] = {"train": int(len(e_tr)), "test": int(len(e_te))}
    corr = np.corrcoef(np.concatenate([e_te, d_te], 1).T)[:4, 4:]
    out["corr_ego_vs_ecm_joints"] = np.round(corr, 3).tolist()            # rows dx, dy, log s, angle
    reg = RidgeCV(alphas=np.logspace(-6, 2, 9)).fit(e_tr, d_tr)
    out["r2_ego_to_ecm"] = float(r2_score(d_te, reg.predict(e_te), multioutput="variance_weighted"))
    reg2 = RidgeCV(alphas=np.logspace(-6, 2, 9)).fit(d_tr, e_tr[:, :3])
    out["r2_ecm_to_ego"] = float(r2_score(e_te[:, :3], reg2.predict(d_te), multioutput="variance_weighted"))
    if p_te.any() and (~p_te).any():
        out["auc_camera_pedal_from_ego"] = float(roc_auc_score(p_te, np.abs(e_te[:, :3]).sum(1)))
        out["auc_camera_pedal_from_ecm"] = float(roc_auc_score(p_te, np.abs(d_te).sum(1)))
    dst = OUTPUTS / "eval" / "camera_validation"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / f"{args.data}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
