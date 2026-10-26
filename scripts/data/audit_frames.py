#!/usr/bin/env python
"""Frame-level quality audit on keyframes sampled every ``--every`` seconds of every video.

Per frame: luma, saturation, sharpness (variance of Laplacian), green-fluorescence share and whether the
console UI bands (top warning banner, bottom arm panels) are burned in. Writes outputs/audit/frames.csv and
outputs/audit/frames_summary.json.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import av
import cv2
import numpy as np
import pandas as pd

from swm.data import surgvu as sv
from swm.paths import OUTPUTS, SURGVU24_VIDEOS

OUT = OUTPUTS / "audit"
X0, X1 = 193, 1087                      # active (non-pillarbox) columns of the 1280x720 frame
TOP, BOTTOM = (8, 50), (684, 718)       # console banner / arm-panel rows
TISSUE = (60, 680)                      # rows that are never covered by the console UI


def band_stats(band_bgr: np.ndarray) -> tuple[float, float]:
    """(brightness, edge density). Console UI bands are dark and edge-dense (text, panel outlines);
    tissue in the same rows is bright with few strong edges, black frames have neither."""
    v = cv2.cvtColor(band_bgr, cv2.COLOR_BGR2HSV)[..., 2].mean()
    edges = cv2.Canny(cv2.cvtColor(band_bgr, cv2.COLOR_BGR2GRAY), 60, 160).mean() / 255
    return float(v), float(edges)


def is_ui(v: float, edges: float) -> bool:
    return v < 90 and edges > 0.06


def frame_stats(bgr: np.ndarray) -> dict:
    act = bgr[:, X0:X1]
    tis = act[TISSUE[0]:TISSUE[1]]
    hsv = cv2.cvtColor(tis, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(tis, cv2.COLOR_BGR2GRAY)
    h, s, v = hsv[..., 0].astype(np.float32), hsv[..., 1].astype(np.float32), hsv[..., 2].astype(np.float32)
    green = ((h > 35) & (h < 85) & (s > 80) & (v > 60)).mean()
    red = (((h < 12) | (h > 160)) & (s > 60) & (v > 40)).mean()
    top_v, top_e = band_stats(act[TOP[0]:TOP[1], 10:-10])
    bot_v, bot_e = band_stats(act[BOTTOM[0]:BOTTOM[1], 10:-10])
    return dict(luma=float(v.mean()), sat=float(s.mean()), sharp=float(cv2.Laplacian(gray, cv2.CV_32F).var()),
                green=float(green), red=float(red), dark_frac=float((v < 20).mean()),
                pillar_luma=float(np.concatenate([bgr[:, :X0 - 5], bgr[:, X1 + 5:]], 1).mean()),
                top_v=top_v, top_edges=top_e, bottom_v=bot_v, bottom_edges=bot_e,
                ui_top=is_ui(top_v, top_e), ui_bottom=is_ui(bot_v, bot_e))


def audit_video(args) -> list[dict]:
    case, part, rel, every = args
    rows = []
    with av.open(str(SURGVU24_VIDEOS / rel)) as c:
        dur = c.duration / 1e6
        for t in np.arange(every / 2, dur, every):
            try:
                c.seek(int(t * 1e6), backward=True)
                frame = next(c.decode(video=0))
            except (StopIteration, av.error.FFmpegError):
                continue
            st = frame_stats(frame.to_ndarray(format="bgr24"))
            rows.append(dict(case=case, part=part, t=float(frame.time if frame.time is not None else t), **st))
    return rows


def classify(df: pd.DataFrame) -> pd.Series:
    cls = pd.Series("tissue", index=df.index)
    cls[df["green"] > 0.02] = "fluorescence"
    cls[(df["sat"] < 45) & (df["green"] <= 0.02)] = "low_colour"        # grey: out of body, smoke, drapes
    cls[df["sharp"] < 15] = "blank_or_blurred"
    cls[df["luma"] < 25] = "black"
    return cls


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()
    jobs = [(case, d["part"], d["path"], args.every) for case, parts in sv.parts_table().items() for d in parts]
    with ProcessPoolExecutor(args.workers) as ex:
        df = pd.DataFrame([r for rows in ex.map(audit_video, jobs) for r in rows])
    df["cls"] = classify(df)
    df.to_csv(OUT / "frames.csv", index=False)
    summary = dict(
        n_frames=len(df), every_s=args.every,
        class_frac=df["cls"].value_counts(normalize=True).round(4).to_dict(),
        ui_top_frac=float(df["ui_top"].mean()), ui_bottom_frac=float(df["ui_bottom"].mean()),
        cases_with_ui_bottom=int(df.groupby("case")["ui_bottom"].mean().gt(0.5).sum()),
        cases_without_ui_bottom=int(df.groupby("case")["ui_bottom"].mean().lt(0.05).sum()),
        pillarbox_luma_p99=float(df["pillar_luma"].quantile(0.99)),
        sharp_quantiles=df["sharp"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(1).to_dict(),
    )
    (OUT / "frames_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
