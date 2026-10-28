#!/usr/bin/env python
"""Map JIGSAWS (da Vinci bench-top suturing, knot tying, needle passing) onto the world-model representation.

Left-camera video (640x480, 30 fps) is resampled to 10 fps and centre-cropped to 448x640; the 76-column
kinematics give the patient-side manipulators: slave right -> PSM1 (slot 0, right of the image), slave left ->
PSM2 (slot 2). Rotation matrices become quaternions so the per-arm state has the SutureBot layout
[pos 3, quat 4, gripper 1] x 2. Surgeons D and I are held out (leave-users-out). Please cite Gao et al.,
M2CAI 2014 when using JIGSAWS.

    python scripts/data/prepare_jigsaws.py
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import av
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_lerobot import WIN, bounded_imap, build, crop_resize  # noqa: E402

from swm.data import egomotion  # noqa: E402
from swm.data import surgvu as sv  # noqa: E402
from swm.data.geometry import MODEL_H, MODEL_W  # noqa: E402
from swm.paths import EXTERNAL, PROCESSED  # noqa: E402

TEST_USERS = {"D", "I"}
TASKS = ("Suturing", "Knot_Tying", "Needle_Passing")


def arm_state(kin: np.ndarray, c0: int) -> np.ndarray:
    pos = kin[:, c0:c0 + 3]
    quat = Rotation.from_matrix(kin[:, c0 + 3:c0 + 12].reshape(-1, 3, 3)).as_quat()       # x, y, z, w
    return np.concatenate([pos, quat, kin[:, c0 + 18:c0 + 19]], 1)


def job(args) -> dict:
    idx, task, trial, stride = args
    cv2.setNumThreads(1)
    root = EXTERNAL / "jigsaws" / f"{task}_x" / task
    kin = np.loadtxt(root / "kinematics" / "AllGestures" / f"{trial}.txt", dtype=np.float32)
    frames = []
    with av.open(str(root / "video" / f"{trial}_capture1.avi")) as c:
        for i, f in enumerate(c.decode(video=0)):
            if i % 3 == 0:
                frames.append(crop_resize(f.to_ndarray(format="rgb24"), 2 * MODEL_H, 2 * MODEL_W))
    n = min(len(frames), (len(kin) + 2) // 3)
    frames = np.stack(frames[:n])
    k = kin[::3][:n]
    state = np.concatenate([arm_state(k, 57), arm_state(k, 38)], 1).astype(np.float32)   # PSM1 = slave right
    small = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA)
             for f in frames]
    ego = np.full((n, 4), np.nan, np.float32)
    inl = np.zeros(n, np.float32)
    for i in range(1, n):
        ego[i], inl[i], _ = egomotion.estimate(small[i - 1], small[i])
    nd = sv.OCC_INDEX["needle driver"]
    return dict(ep=idx, frames=frames, sig={"state": state, "action": None}, ego=ego, inl=inl, tools=[nd, nd],
                starts=list(range(0, n - WIN + 1, stride)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    trials = []
    for task in TASKS:                    # 17 of the 120 trials have kinematics but no video; they are skipped
        root = EXTERNAL / "jigsaws" / f"{task}_x" / task
        trials += [(task, p.stem) for p in sorted((root / "kinematics" / "AllGestures").glob("*.txt"))
                   if (root / "video" / f"{p.stem}_capture1.avi").exists()]
    user = [trial.split("_")[-1][0] for _, trial in trials]                 # e.g. Suturing_B001 -> B
    split_of = {i: ("test" if u in TEST_USERS else "train") for i, u in enumerate(user)}
    out = PROCESSED / "jigsaws"
    out.mkdir(parents=True, exist_ok=True)
    (out / "episodes.tsv").write_text("".join(f"{i}\t{t}\t{tr}\t{split_of[i]}\n" for i, (t, tr) in enumerate(trials)))
    jobs = [(i, t, tr, args.stride) for i, (t, tr) in enumerate(trials)]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        build("jigsaws", bounded_imap(pool, job, jobs, args.workers + 1), split_of)


if __name__ == "__main__":
    main()
