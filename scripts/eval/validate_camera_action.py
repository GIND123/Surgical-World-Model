#!/usr/bin/env python
"""Does the explicit camera action measure real endoscope motion? Validation on CRCD (dVRK ECM kinematics + pedals).

Per 10 fps frame transition: the video-estimated ego-motion (dx, dy, log s, angle) vs the change of the four ECM
joints. Only episodes whose video and kinematics are verified to be in sync are used (scripts/data/check_sync.py,
lag-corrected signals); about a third of the CRCD conversion drifts or is offset, which otherwise hides any
agreement. Reported:
  * detection of real endoscope motion (ECM moving) from the ego-motion magnitude, ROC AUC per transition and per
    latent step (4 transitions, the granularity of the camera action), on all and on reliably estimated steps;
  * the same against the camera pedal (a weaker label: the pedal can be held while the endoscope is still);
  * per-axis correlations, sign agreement and leave-one-episode-out ridge R^2 on transitions with ECM motion.

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

MOVING = 3e-3           # sum |d joint| per 0.1 s; 10x the 99th percentile while the camera pedal is released


def transitions(cond: dict, sig: dict) -> dict:
    """Flatten windows into per-transition arrays (frame j-1 -> j, j = 1..32)."""
    cam = cond["camera"].reshape(len(cond["camera"]), 9, 4, 4)            # (N, step, sub-transition, param)
    ego = cam[:, 1:].reshape(len(cam), 32, 4)                            # transitions 1..32 in order
    ok = (ego != 0).any(-1)                                              # unreliable transitions are stored as 0
    decm = np.diff(sig["ecm"], axis=1)                                   # (N, 32, 4)
    ped = sig["pedals"][:, 1:]                                           # (N, 32, 3), NaN when not recorded
    return dict(ego=ego, ok=ok, decm=decm, ped=ped)


def auc(label: np.ndarray, score: np.ndarray) -> float | None:
    return float(roc_auc_score(label, score)) if 0 < label.mean() < 1 else None


def detection(ego, ok, label) -> dict:
    """AUC of |ego| (translation + zoom) for a binary label, per transition and per latent step."""
    mag = np.where(ok, np.abs(ego[..., :2]).sum(-1) + np.abs(ego[..., 2]), 0)
    step_mag, step_lab = mag.reshape(-1, 8, 4).sum(-1), label.reshape(-1, 8, 4).any(-1)
    step_ok = ok.reshape(-1, 8, 4).all(-1)                               # all 4 transitions estimated
    return dict(auc_transition=auc(label.ravel(), mag.ravel()), auc_step=auc(step_lab.ravel(), step_mag.ravel()),
                auc_step_reliable=auc(step_lab[step_ok], step_mag[step_ok]),
                positive_rate=float(label.mean()), reliable_step_fraction=float(step_ok.mean()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="crcd")
    args = ap.parse_args()
    root = PROCESSED / args.data
    rows = pd.read_csv(root / "windows.csv")
    sig = dict(np.load(root / "signals_synced.npz"))
    synced = sig.pop("synced")
    sync = json.loads((root / "sync.json").read_text())
    tr = transitions(dict(np.load(root / "conditioning.npz")), sig)
    out = {"data": args.data, "moving_threshold": MOVING,
           "episodes_synced": sorted(int(e) for e, v in sync.items() if v["ok"]),
           "episodes_excluded": sorted(int(e) for e, v in sync.items() if not v["ok"]),
           "windows_synced": int(synced.sum()), "windows_total": len(rows)}
    w = np.flatnonzero(synced)
    ego, ok, decm, ped = tr["ego"][w], tr["ok"][w], tr["decm"][w], tr["ped"][w]
    ep = rows["episode"].to_numpy()[w]
    dmag = np.abs(decm).sum(-1)
    moving = dmag > MOVING
    out["ecm_moving"] = detection(ego, ok, moving)
    # camera pedal: the channel during which the ECM moves most, on episodes that recorded pedals
    has_ped = np.isfinite(ped).all(axis=(1, 2))
    ratios = [float(dmag[has_ped][ped[has_ped][..., c] > 0.5].mean()
                    / max(dmag[has_ped][~(ped[has_ped][..., c] > 0.5)].mean(), 1e-12)) for c in range(ped.shape[-1])]
    cam_col = int(np.argmax(ratios))
    out["pedal_ecm_motion_ratio"], out["camera_pedal_column"] = ratios, cam_col
    pedal = ped[has_ped][..., cam_col] > 0.5
    out["camera_pedal"] = detection(ego[has_ped], ok[has_ped], pedal)
    out["camera_pedal"]["auc_transition_from_ecm"] = auc(pedal.ravel(), dmag[has_ped].ravel())
    out["pedal_while_ecm_still"] = float((~moving[has_ped][pedal]).mean())
    # geometry on transitions with real ECM motion and a reliable estimate
    m = ok & moving
    X, Y, g = ego[m], decm[m], np.repeat(ep[:, None], 32, axis=1)[m]
    out["n_moving_transitions"] = int(m.sum())
    out["corr_ego_vs_ecm_joints"] = np.round(np.corrcoef(np.concatenate([X, Y], 1).T)[:4, 4:], 3).tolist()
    p_ecm, p_ego = np.zeros_like(Y), np.zeros_like(X[:, :3])
    for e in np.unique(g):                                               # leave one episode out
        a, b = g != e, g == e
        p_ecm[b] = RidgeCV(alphas=np.logspace(-6, 2, 9)).fit(X[a], Y[a]).predict(X[b])
        p_ego[b] = RidgeCV(alphas=np.logspace(-6, 2, 9)).fit(Y[a], X[a, :3]).predict(Y[b])
    out["r2_ego_to_ecm_loeo"] = float(r2_score(Y, p_ecm, multioutput="variance_weighted"))
    out["r2_ego_to_ecm_per_joint"] = np.round(r2_score(Y, p_ecm, multioutput="raw_values"), 3).tolist()
    out["r2_ecm_to_ego_loeo"] = float(r2_score(X[:, :3], p_ego, multioutput="variance_weighted"))
    sign = {}
    for axis, joint in (("dx", 0), ("dy", 1), ("log_s", 2)):
        k = ["dx", "dy", "log_s"].index(axis)
        big = np.abs(Y[:, joint]) > np.median(np.abs(Y[:, joint]))
        sign[f"{axis}_vs_joint{joint + 1}"] = float((np.sign(X[big, k]) == np.sign(Y[big, joint])).mean())
    out["sign_agreement"] = sign
    out["per_episode"] = {}
    for e in np.unique(ep):
        s = ep == e
        out["per_episode"][int(e)] = dict(split=rows.loc[rows["episode"] == e, "split"].iloc[0],
                                          lag_s=sync[str(e)]["lag"] / 10,
                                          **{k: v for k, v in detection(ego[s], ok[s], moving[s]).items()
                                             if k.startswith("auc")})
    dst = OUTPUTS / "eval" / "camera_validation"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / f"{args.data}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
