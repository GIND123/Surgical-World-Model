#!/usr/bin/env python
"""Per-window conditioning: robot state, arm regions and camera actions for every Wan2.2 latent step.

For a window starting at cache frame s, latent step t covers frames 4t-3..4t (t = 0 is frame s alone).
  occupant (N, 9, 4)    occupant index per arm (swm.data.surgvu.OCCUPANTS) at frame s+4t, from the system logs
  boxes    (N, 9, 4, 4) arm-k instrument box at step t: union of its detections at frames s+4(t-1) and s+4t,
                        normalised to the endoscope field (x0, y0, x1, y1 in [0, 1]); NaN if not found
  camera   (N, 9, 16)   ego-motion of the 4 frame transitions inside step t (dx, dy, log s, angle), 0 at t = 0
  cam_ok   (N, 9)       all 4 transitions of the step were estimated reliably
Detections are bound to arms by Hungarian matching: cost = class term (detected class vs the arm's logged
instrument) + spatial term (distance to the arm's typical image position in this case, learned from frames
where every mounted instrument class is unique). Writes data/processed/surgvu/conditioning/<stem>.npz.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from swm.data import surgvu as sv
from swm.data.geometry import CACHE_H, CACHE_W, FIELD_ROWS
from swm.data.io import savez, settled
from swm.paths import PROCESSED

CACHE = PROCESSED / "surgvu"
OUT = CACHE / "conditioning"
STEP, N_STEPS = 4, 9
MISMATCH, SPATIAL_W, MAX_COST = 1.5, 2.0, 3.5


def det_by_frame(d) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out = {}
    order = np.argsort(d["frame"], kind="stable")
    fr, bx, cf, cl = d["frame"][order], d["box"][order], d["conf"][order].astype(np.float32), d["cls"][order]
    cuts = np.flatnonzero(np.diff(fr)) + 1
    for f, b, c, k in zip(np.split(fr, cuts), np.split(bx, cuts), np.split(cf, cuts), np.split(cl, cuts), strict=True):
        if len(f):
            out[int(f[0])] = (b, c, k.astype(np.int64) + 2)            # detector class -> occupant index
    return out


def centre(b: np.ndarray) -> np.ndarray:
    return np.stack([(b[:, 0] + b[:, 2]) / 2 / CACHE_W, (b[:, 1] + b[:, 3]) / 2 / CACHE_H], 1)


def learn_priors(dets, occ_at) -> np.ndarray:
    """(4, 2) median image position of each arm's instrument from unambiguous frames; NaN when unseen."""
    pts = [[] for _ in sv.ARMS]
    for f, (b, c, k) in dets.items():
        occ = occ_at(f)
        inst = [(a, o) for a, o in enumerate(occ) if o >= 2]
        if len({o for _, o in inst}) != len(inst):
            continue
        cen = centre(b)
        for a, o in inst:
            hit = np.flatnonzero((k == o) & (c > 0.5))
            if len(hit) == 1:
                pts[a].append(cen[hit[0]])
    return np.array([np.median(p, 0) if len(p) >= 20 else [np.nan, np.nan] for p in pts])


def assign(b, c, k, occ, prior) -> dict[int, np.ndarray]:
    arms = [a for a, o in enumerate(occ) if o >= 2]
    if not arms or len(b) == 0:
        return {}
    cen = centre(b)
    cost = np.zeros((len(arms), len(b)))
    for i, a in enumerate(arms):
        cost[i] = -np.log(np.clip(c, 1e-3, 1)) + MISMATCH * (k != occ[a])
        if np.isfinite(prior[a]).all():
            cost[i] += SPATIAL_W * np.linalg.norm(cen - prior[a], axis=1)
    rows, cols = linear_sum_assignment(cost)
    return {arms[r]: b[j] for r, j in zip(rows, cols, strict=True) if cost[r, j] < MAX_COST}


def to_field(box: np.ndarray) -> np.ndarray:
    f0, f1 = FIELD_ROWS
    x0, y0, x1, y1 = box
    return np.clip([x0 / CACHE_W, (y0 - f0) / (f1 - f0), x1 / CACHE_W, (y1 - f0) / (f1 - f0)], 0, 1)


def build(stem: str) -> str:
    case, part = stem.rsplit("_p", 1)
    starts = np.load(CACHE / "latents" / f"{stem}.starts.npy")
    meta = np.load(CACHE / "meta10" / f"{stem}.npz")
    dets = det_by_frame(np.load(CACHE / "detections" / f"{stem}.npz"))
    tl = sv.timeline(case)
    tools, _ = sv.load_tools(case)
    occ10 = sv.arm_occupancy(tools, tl.duration, hz=10.0)
    tg = meta["t_global"]

    def occ_at(f: int) -> np.ndarray:
        return occ10[min(int(round(tg[f] * 10)), len(occ10) - 1)]

    prior = learn_priors(dets, occ_at)
    n = len(starts)
    occupant = np.zeros((n, N_STEPS, 4), np.int8)
    boxes = np.full((n, N_STEPS, 4, 4), np.nan, np.float16)
    camera = np.zeros((n, N_STEPS, 16), np.float32)
    cam_ok = np.zeros((n, N_STEPS), bool)
    ego, inl = meta["ego"], meta["ego_inlier"]
    per_frame: dict[int, dict[int, np.ndarray]] = {}
    for w, s in enumerate(starts):
        for t in range(N_STEPS):
            f = int(s) + STEP * t
            occupant[w, t] = occ_at(f)
            if f not in per_frame:
                per_frame[f] = assign(*dets[f], occupant[w, t], prior) if f in dets else {}
            for a in range(4):
                bs = [per_frame[f].get(a)]
                if t > 0:
                    bs.append(per_frame.get(f - STEP, {}).get(a))
                bs = [b for b in bs if b is not None]
                if bs:
                    u = np.array([min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs),
                                  max(b[3] for b in bs)])
                    boxes[w, t, a] = to_field(u)
            if t > 0:
                e = ego[f - STEP + 1:f + 1]
                ok = np.isfinite(e).all(1) & (inl[f - STEP + 1:f + 1] > 0.5)
                camera[w, t] = np.where(ok[:, None], e, 0).ravel()
                cam_ok[w, t] = ok.all()
    OUT.mkdir(parents=True, exist_ok=True)
    savez(OUT / f"{stem}.npz", occupant=occupant, boxes=boxes, camera=camera, cam_ok=cam_ok, prior=prior)
    found = np.isfinite(boxes[..., 0]).sum() / max(1, (occupant >= 2).sum())
    return f"{stem}: {n} windows, arm-box coverage {found:.2f}, priors {np.round(prior, 2).tolist()}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait", action="store_true")
    args = ap.parse_args()
    while True:
        todo = [p.stem for p in sorted((CACHE / "detections").glob("*.npz"))
                if settled(p, 5) and not (OUT / p.name).exists()]
        if not todo:
            if not args.wait or len(list(OUT.glob("*.npz"))) >= 280:
                break
            time.sleep(120)
            continue
        for stem in todo:
            print(time.strftime("%H:%M:%S"), build(stem), flush=True)


if __name__ == "__main__":
    main()
