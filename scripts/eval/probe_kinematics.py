#!/usr/bin/env python
"""E2: do learned latent actions encode real robot motion? Linear probes on datasets with kinematics.

For every latent step of every window, features come from a frozen model (posterior-mean latent actions) or
from model-free baselines; targets are the net PSM1/PSM2 motion over the step (dpos 3, drot 3, djaw 1).
Ridge regression (alpha chosen by cross-validation on training episodes) is fitted on the training split and
R^2 is reported on held-out test episodes, per arm and per target group.

    python scripts/eval/probe_kinematics.py --data suturebot --runs eg_swm_B mono_B slots_B --baselines
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

from swm.data.external import SLOTS, ExternalWindows
from swm.models.actions import region_from_boxes
from swm.paths import OUTPUTS

GROUPS = {"pos": slice(0, 3), "rot": slice(3, 6), "jaw": slice(6, 7)}


@torch.no_grad()
def model_features(run: str, ds: ExternalWindows, bs: int = 32) -> dict[str, np.ndarray]:
    """{arm: (N, 8, F)} posterior-mean action features for latent steps 1..8."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_wm import load_model

    model, _ = load_model(run, None)
    feats = {arm: [] for arm in SLOTS}
    for b in DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4):
        b = {k: v.cuda() for k, v in b.items()}
        b["region"] = region_from_boxes(b["boxes"], 14, 20, dilate=1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            acts = model.infer_actions(b["z"], b, stochastic=False)
        for arm, slot in SLOTS.items():
            if "u_arms" in acts:
                f = acts["u_arms"][:, 1:, slot]
            else:
                f = acts["u_extra"][:, 1:].flatten(2)            # monolithic or all slots, same for both arms
            feats[arm].append(f.float().cpu().numpy())
    return {arm: np.concatenate(v) for arm, v in feats.items()}


def baseline_features(ds: ExternalWindows, kind: str) -> dict[str, np.ndarray]:
    """Model-free features: pooled latent difference (inside the arm region or over the frame), or camera."""
    from swm.models.actions import region_from_boxes as rfb
    out = {arm: [] for arm in SLOTS}
    for b in DataLoader(ds, batch_size=64, shuffle=False, num_workers=4):
        z = b["z"]                                                        # (B, 9, C, H, W)
        dz = (z[:, 1:] - z[:, :-1]).flatten(3)                           # (B, 8, C, HW)
        reg = rfb(b["boxes"], 14, 20, dilate=1)[:, 1:].float()          # (B, 8, 4, HW)
        for arm, slot in SLOTS.items():
            if kind == "dz_region":
                w = reg[:, :, slot][:, :, None]                          # (B, 8, 1, HW)
                f = (dz * w).sum(-1) / w.sum(-1).clamp_min(1)
                f = torch.cat([f, (dz.abs() * w).sum(-1) / w.sum(-1).clamp_min(1)], -1)
            elif kind == "dz_frame":
                f = torch.cat([dz.mean(-1), dz.abs().mean(-1)], -1)
            else:                                                         # explicit camera action
                f = b["camera"][:, 1:]
            out[arm].append(f.numpy())
    return {arm: np.concatenate(v) for arm, v in out.items()}


def probe(train_x, train_y, test_x, test_y) -> dict:
    res = {}
    keep_tr = np.isfinite(train_x).all(1) & np.isfinite(train_y).all(1)
    keep_te = np.isfinite(test_x).all(1) & np.isfinite(test_y).all(1)
    mu, sd = train_x[keep_tr].mean(0), train_x[keep_tr].std(0) + 1e-8
    for g, sl in GROUPS.items():
        reg = RidgeCV(alphas=np.logspace(-3, 4, 15)).fit((train_x[keep_tr] - mu) / sd, train_y[keep_tr, sl])
        pred = reg.predict((test_x[keep_te] - mu) / sd)
        res[g] = float(r2_score(test_y[keep_te, sl], pred, multioutput="variance_weighted"))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--baselines", action="store_true")
    args = ap.parse_args()
    tr, te = ExternalWindows(args.data, "train"), ExternalWindows(args.data, "test")
    targets = {s: {arm: d.kin_net[d.index][:, 1:, slot].reshape(-1, 7) for arm, slot in SLOTS.items()}
               for s, d in (("train", tr), ("test", te))}
    sources = {r: (lambda r=r, d=None: None) for r in args.runs}
    results = {"data": args.data, "n_train_steps": int(len(targets["train"]["psm1"])),
               "n_test_steps": int(len(targets["test"]["psm1"])), "sides_swapped": tr.swapped, "probes": {}}
    names = list(args.runs) + (["dz_region", "dz_frame", "camera"] if args.baselines else [])
    for name in names:
        if name in sources:
            ftr, fte = model_features(name, tr), model_features(name, te)
        else:
            ftr, fte = baseline_features(tr, name), baseline_features(te, name)
        results["probes"][name] = {
            arm: probe(ftr[arm].reshape(-1, ftr[arm].shape[-1]), targets["train"][arm],
                       fte[arm].reshape(-1, fte[arm].shape[-1]), targets["test"][arm]) for arm in SLOTS}
        print(name, json.dumps(results["probes"][name]), flush=True)
    out = OUTPUTS / "eval" / "probes"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.data}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
