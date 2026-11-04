#!/usr/bin/env python
"""Fidelity over the prediction horizon (E1): PSNR of every predicted frame, 0.1-2.8 s ahead, for the main action
interfaces and for copying the last context frame (mean over the 1,000 test windows of the E1 evaluation).

    python scripts/docs/figure_horizon.py     # -> docs/figures/e1_horizon.{pdf,png}
"""

from __future__ import annotations

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from swm.paths import OUTPUTS, ROOT  # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = [  # run (or reference prefix), label, colour
    ("S_slots", "Slots + endoscope (FLAM-style)", "#1baf7a"),
    ("S_global", "Monolithic latent + endoscope (Genie-style)", "#6250d6"),
    ("S_arms", "EG-SWM", "#2a78d6"),
    ("S_global_vq", "12-code VQ + endoscope (SurgWM-style)", "#eb6834"),
    ("S_state", "Robot state + endoscope", "#eda100"),
    ("S_none", "No actions", "#b4b2a9"),
    ("copy_last", "Copy last context frame", INK),
]


def main() -> None:
    ref = json.loads((OUTPUTS / "eval" / "reference" / "reference.json").read_text())
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, ax = plt.subplots(figsize=(3.5, 3.3))
    for run, label, color in SERIES:
        if run == "copy_last":
            y = np.array(ref["copy_last_psnr_by_frame"])
        else:
            p = OUTPUTS / "eval" / run / "replay.json"
            if not p.exists():
                continue
            y = np.array(json.loads(p.read_text())["pred_psnr_by_frame"])
        t = np.arange(1, len(y) + 1) / 10                                   # seconds ahead of the last context frame
        ax.plot(t, y, color=color, lw=1.5, ls="--" if run == "copy_last" else "-", label=label,
                zorder=3 if run == "S_arms" else 2)
    ax.set_xlabel("time ahead (s)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_xlim(0, 2.9)
    ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5])
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2,
              handlelength=2.2, columnspacing=1.0)
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / "docs" / "figures" / f"e1_horizon.{ext}", dpi=220)
    print("wrote docs/figures/e1_horizon.{pdf,png}")


if __name__ == "__main__":
    main()
