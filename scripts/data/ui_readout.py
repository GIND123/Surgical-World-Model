#!/usr/bin/env python
"""Read the da Vinci console overlay: per arm, energy activation and the driving hand, for every latent step.

The bottom band of the cache (rows 486-511) holds one panel per arm (USM1-4, 160 px wide each). On instruments
with energy or grip functions the right side of the panel shows labels with a small hand badge: yellow
(GRIP / CUT) on top, cyan (COAG) below, each carrying "L" or "R". A label that lights up (filled) means the
corresponding energy is being applied. Instruments without such functions show no badge, so the hand is
unknown (0) for them.
  hand   (N, 9, 4) 0 unknown, 1 left, 2 right   (from the boundary frame of each latent step)
  energy (N, 9, 4) 0 none, 1 cut (yellow), 2 coag (cyan)   (any of the step's 4 frames)
  ui     (N, 9)    the band is visible
Writes data/processed/surgvu/ui/<stem>.npz; follows build_conditioning.py with --wait.
"""

from __future__ import annotations

import argparse
import time

import av
import cv2
import numpy as np

from swm.data.io import savez, settled
from swm.paths import PROCESSED

CACHE = PROCESSED / "surgvu"
OUT = CACHE / "ui"
STEP, WIN = 4, 33


def masks(hsv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    yellow = (h > 15) & (h < 35) & (s > 100) & (v > 140)
    cyan = (h > 80) & (h < 105) & (s > 80) & (v > 120)
    return yellow, cyan


def badge_hand(hsv: np.ndarray, colour: np.ndarray) -> int:
    """Badge = small coloured square with a dark letter; an R fills the upper-right quadrant, an L does not.
    Calibrated on labelled panels (right-hand monopolar vs left-hand force bipolar): upper-right darkness is
    0.00 for every L and >= 0.11 for every R."""
    if not 0.08 < colour.mean() < 0.8:
        return 0
    ys, xs = np.nonzero(colour)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    if y1 - y0 < 4 or x1 - x0 < 4:
        return 0
    dark = hsv[y0:y1, x0:x1, 2] < 110
    hh, ww = dark.shape
    return 2 if dark[:hh // 2, ww // 2:].mean() > 0.05 else 1


def read_frame(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    band = bgr[486:512]
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(cv2.cvtColor(band, cv2.COLOR_BGR2GRAY), 60, 160).mean() / 255
    visible = bool(hsv[..., 2].mean() < 90 and edges > 0.06)
    hand, energy = np.zeros(4, np.int8), np.zeros(4, np.int8)
    if not visible:
        return hand, energy, False
    for p in range(4):
        x0 = 160 * p
        label = hsv[:, x0 + 114:x0 + 158]
        y, c = masks(label)
        if y.mean() > 0.45:
            energy[p] = 1
        elif c.mean() > 0.45:
            energy[p] = 2
        top = hsv[1:12, x0 + 114:x0 + 128]
        bot = hsv[16:26, x0 + 114:x0 + 128]
        hand[p] = badge_hand(top, masks(top)[0]) or badge_hand(bot, masks(bot)[1])
    return hand, energy, True


def process(stem: str) -> str:
    starts = np.load(CACHE / "latents" / f"{stem}.starts.npy")
    n = len(starts)
    hand = np.zeros((n, 9, 4), np.int8)
    energy = np.zeros((n, 9, 4), np.int8)
    ui = np.zeros((n, 9), bool)
    if n:
        need = {}                                     # frame -> list of (window, step, is_boundary)
        for w, s in enumerate(starts):
            for t in range(9):
                frames = [int(s)] if t == 0 else range(int(s) + STEP * (t - 1) + 1, int(s) + STEP * t + 1)
                for f in frames:
                    need.setdefault(f, []).append((w, t, f == int(s) + STEP * t))
        last = max(need)
        with av.open(str(CACHE / "frames10" / f"{stem}.mp4")) as c:
            c.streams.video[0].thread_type = "AUTO"
            for i, fr in enumerate(c.decode(video=0)):
                if i in need:
                    h, e, vis = read_frame(fr.to_ndarray(format="bgr24"))
                    for w, t, boundary in need[i]:
                        energy[w, t] = np.maximum(energy[w, t], e)
                        if boundary:
                            hand[w, t], ui[w, t] = h, vis
                if i >= last:
                    break
    OUT.mkdir(parents=True, exist_ok=True)
    savez(OUT / f"{stem}.npz", hand=hand, energy=energy, ui=ui)
    return (f"{stem}: {n} windows, ui {ui.mean() if n else 0:.2f}, hand known {(hand > 0).mean() if n else 0:.3f}, "
            f"energy {(energy > 0).mean() if n else 0:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait", action="store_true")
    args = ap.parse_args()
    cv2.setNumThreads(2)
    while True:
        todo = [p.stem for p in sorted((CACHE / "conditioning").glob("*.npz"))
                if settled(p, 5) and not (OUT / p.name).exists()]
        if not todo:
            if not args.wait or len(list(OUT.glob("*.npz"))) >= 280:
                break
            time.sleep(120)
            continue
        for stem in todo:
            print(time.strftime("%H:%M:%S"), process(stem), flush=True)


if __name__ == "__main__":
    main()
