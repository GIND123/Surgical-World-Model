#!/usr/bin/env python
"""Flag windows that show the recorder's no-signal screen (colour bars) instead of the endoscope image.

Colour bars pass the usable-frame rules of build_windows.py (bright, saturated, sharp). 0.84 % of the windows
show them, most in case_126 (78 % of its first part), case_140 and case_099. Two tests on the raw Wan2.2
latents, so no video is decoded:
  * column uniformity: in the upper 9 of the 14 latent rows, bars are constant down every column. The mean
    standard deviation down the columns is below 0.075 for the first latent (one frame) and below 0.085 for the
    others (four frames each) only for bars, also shifted ones or ones under a dialog box; tissue stays above;
  * template: the first or last latent lies within 0.2 (mean absolute difference) of the median first latent of
    the clearly uniform windows, which catches standard bars under a moving overlay.
A window is flagged when any test fires. Writes data/processed/surgvu/no_signal.npz ({part: bool per window})
and configs/stats/no_signal.json (counts per split).

    python scripts/data/flag_no_signal.py [--workers 8]
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool

import numpy as np

from swm.paths import PROCESSED, ROOT

LAT = PROCESSED / "surgvu" / "latents"
COL_MAX = np.array([0.075] + [0.085] * 8, np.float32)       # per latent position
TEMPLATE_MAX = 0.2


def column_std(stem: str) -> tuple[str, np.ndarray]:
    lat = np.load(LAT / f"{stem}.npy", mmap_mode="r")      # (N, 9, 48, 14, 20)
    col = np.empty(lat.shape[:2], np.float32)
    for i in range(0, len(lat), 256):
        z = np.asarray(lat[i:i + 256], np.float32)
        col[i:i + 256] = z[:, :, :, :9].std(3).mean((2, 3))
    return stem, col


def template_dist(args: tuple[str, np.ndarray]) -> tuple[str, np.ndarray]:
    stem, template = args
    lat = np.load(LAT / f"{stem}.npy", mmap_mode="r")
    d = [np.abs(np.asarray(lat[:, t], np.float32) - template).mean((1, 2, 3)) for t in (0, 8)]
    return stem, np.minimum(*d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    stems = sorted(p.stem for p in LAT.glob("*.npy") if not p.stem.endswith(".starts"))
    with Pool(args.workers) as pool:
        col = dict(pool.imap_unordered(column_std, stems))
        clear = [(s, np.flatnonzero(c[:, 0] < 0.07)) for s, c in col.items()]
        template = np.median(np.concatenate([np.asarray(np.load(LAT / f"{s}.npy", mmap_mode="r")[w, 0], np.float32)
                                             for s, w in clear if len(w)]), 0)
        dist = dict(pool.imap_unordered(template_dist, [(s, template) for s in stems]))
    flags = {s: (col[s] < COL_MAX).any(1) | (dist[s] < TEMPLATE_MAX) for s in stems}
    np.savez(PROCESSED / "surgvu" / "no_signal.npz", **flags)
    split = json.loads((ROOT / "configs" / "splits" / "surgvu_v1.json").read_text())
    which = {c: s for s in ("train", "val", "test") for c in split[s]}
    summary = {s: {"windows": 0, "no_signal": 0} for s in ("train", "val", "test")}
    for stem, f in flags.items():
        row = summary[which[stem.rsplit("_p", 1)[0]]]
        row["windows"] += len(f)
        row["no_signal"] += int(f.sum())
    summary["parts_over_10pct"] = {s: round(float(f.mean()), 3) for s, f in flags.items()
                                   if len(f) and f.mean() > 0.1}
    (ROOT / "configs" / "stats" / "no_signal.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
