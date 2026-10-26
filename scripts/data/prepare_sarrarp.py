#!/usr/bin/env python
"""Map SAR-RARP50 (clinical robot-assisted radical prostatectomy) onto the world-model representation.

Left video (60 fps) -> 10 fps windows (448x640 for detection, 224x320 for the VAE); the 10 Hz surgical gesture
labels (action_discrete.txt, 8 classes) are kept per frame as the signal ``gesture`` for probing. No robot
kinematics or logs exist, so instrument identities are unknown and the two most confident detections are
bound to arm slots by image side. The official train (40 videos) / test (10 videos) split is used.

    python scripts/data/prepare_sarrarp.py
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import zipfile
from pathlib import Path

import av
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_lerobot import WIN, build, crop_resize  # noqa: E402

from swm.data import egomotion  # noqa: E402
from swm.data import surgvu as sv  # noqa: E402
from swm.data.geometry import MODEL_H, MODEL_W  # noqa: E402
from swm.paths import EXTERNAL  # noqa: E402

ROOT = EXTERNAL / "sar_rarp50"
EXTRACT = ROOT / "extracted"


def extract(zpath: Path) -> Path:
    out = EXTRACT / zpath.stem
    if not (out / "video_left.avi").exists():
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath) as z:
            for name in ("video_left.avi", "action_discrete.txt"):
                if name in z.namelist():
                    z.extract(name, out)
    return out


def job(args) -> dict:
    idx, zpath, stride = args
    cv2.setNumThreads(2)
    d = extract(Path(zpath))
    labels = {}
    if (d / "action_discrete.txt").exists():
        for line in (d / "action_discrete.txt").read_text().split():
            f, g = line.split(",")
            labels[int(f)] = int(g)
    frames, gest = [], []
    with av.open(str(d / "video_left.avi")) as c:
        c.streams.video[0].thread_type = "AUTO"
        for i, f in enumerate(c.decode(video=0)):
            if i % 6 == 0:
                frames.append(crop_resize(f.to_ndarray(format="rgb24"), 2 * MODEL_H, 2 * MODEL_W))
                gest.append(labels.get(i, -1))
    frames = np.stack(frames)
    n = len(frames)
    small = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA)
             for f in frames]
    ego = np.full((n, 4), np.nan, np.float32)
    inl = np.zeros(n, np.float32)
    for i in range(1, n):
        ego[i], inl[i], _ = egomotion.estimate(small[i - 1], small[i])
    unk = sv.OCC_INDEX[sv.UNKNOWN]
    return dict(ep=idx, frames=frames, sig={"gesture": np.asarray(gest, np.float32)[:, None]}, ego=ego, inl=inl,
                tools=[unk, unk], starts=list(range(0, n - WIN + 1, stride)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=33)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    zips = [(z, "train") for z in sorted((ROOT / "SAR-RARP50_train_set").glob("video_*.zip"))]
    zips += [(z, "test") for z in sorted((ROOT / "SAR-RARP50_test_set").glob("video_*.zip"))]
    split_of = {i: s for i, (_, s) in enumerate(zips)}
    out = Path(__file__).resolve().parents[2] / "data" / "processed" / "sar_rarp50"
    out.mkdir(parents=True, exist_ok=True)
    (out / "episodes.tsv").write_text("".join(f"{i}\t{z.stem}\t{s}\n" for i, (z, s) in enumerate(zips)))
    jobs = [(i, str(z), args.stride) for i, (z, _) in enumerate(zips)]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        build("sar_rarp50", pool.imap(job, jobs), split_of)


if __name__ == "__main__":
    main()
