#!/usr/bin/env python
"""Figure: the robot's embodiment is observable without kinematics (one SurgVU case on one time axis).

(a) arm occupancy from the system log, per universal surgical manipulator, coloured by instrument family
(b) annotated task segments
(c) endoscope ego-motion (translation per 0.1 s, 10 s median)
Palette: validated categorical order (dataviz reference); light-contrast slots carry a legend and direct labels.

    python scripts/docs/figure_embodiment.py --case case_120
"""

from __future__ import annotations

import argparse

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from swm.data import surgvu as sv  # noqa: E402
from swm.paths import PROCESSED, ROOT  # noqa: E402

FAMILIES = {   # validated order: blue, orange, aqua, violet, yellow (adjacent CVD dE >= 9.2)
    "endoscope": ("#2a78d6", {sv.CAMERA}),
    "energy": ("#eb6834", {"bipolar forceps", "force bipolar", "bipolar dissector", "monopolar curved scissors",
                           "permanent cautery hook/spatula", "vessel sealer", "synchroseal", "curved scissors"}),
    "grasper": ("#1baf7a", {"cadiere forceps", "prograsp forceps", "grasping retractor", "tip-up fenestrated grasper",
                            "crocodile grasper", "tenaculum forceps", "potts scissors", "suction irrigator"}),
    "needle driver": ("#6250d6", {"needle driver"}),
    "clip / stapler": ("#eda100", {"clip applier", "stapler"}),
}
UNKNOWN_COLOR = "#b4b2a9"
SHORT = {"needle driver": "needle drv", "monopolar curved scissors": "monopolar", "bipolar forceps": "bipolar",
         "force bipolar": "force bip.", "cadiere forceps": "cadiere", "prograsp forceps": "prograsp",
         "grasping retractor": "retractor", "permanent cautery hook/spatula": "hook", "vessel sealer": "sealer"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def family(occupant: str) -> tuple[str, str]:
    for name, (color, members) in FAMILIES.items():
        if occupant in members:
            return name, color
    return "unknown", UNKNOWN_COLOR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="case_120")
    args = ap.parse_args()
    case = args.case
    tl = sv.timeline(case)
    tools, _ = sv.load_tools(case)
    tasks = sv.load_tasks(case)
    occ = sv.arm_occupancy(tools, tl.duration, hz=1.0)                          # resolved occupancy, 1 s grid
    ego_t, ego_m = [], []
    for part in sv.parts_table()[case]:
        f = PROCESSED / "surgvu" / "meta10" / f"{case}_p{part['part']:03d}.npz"
        if f.exists():
            m = np.load(f)
            ego_t.append(m["t_global"])
            mag = np.abs(m["ego"][:, :2]).sum(1)
            ego_m.append(np.where(m["ego_inlier"] > 0.5, mag, np.nan))          # reliable estimates only
    t_ego = np.concatenate(ego_t) / 60
    m_ego = np.concatenate(ego_m) * 100                                        # % of frame size per 0.1 s
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(3, 1, figsize=(7.16, 3.6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0, 1.2]})
    ax = axes[0]
    minutes = np.arange(len(occ)) / 60
    for a in range(len(sv.ARMS)):
        row = occ[:, a]
        change = np.flatnonzero(np.diff(row)) + 1
        bounds = np.r_[0, change, len(row)]
        for s, e in zip(bounds[:-1], bounds[1:], strict=True):
            name = sv.OCCUPANTS[row[s]]
            if name == sv.EMPTY:
                continue
            fam, color = family(name)
            ax.barh(3 - a, (e - s) / 60, left=minutes[s], height=0.7, color=color, edgecolor="white", linewidth=0.6)
            if (e - s) / 60 > 28 and name != sv.CAMERA:                          # direct label on long segments
                ax.text(minutes[s] + (e - s) / 120, 3 - a, SHORT.get(name, name), ha="center", va="center",
                        fontsize=6, color="white" if fam in ("endoscope", "needle driver", "energy") else INK)
    ax.set_yticks(range(4), [f"USM{4 - i}" for i in range(4)])
    ax.set_title(f"(a) arm occupancy from the robot log — {case}", loc="left", fontsize=8, color=INK)
    ax.legend(handles=[Patch(color=c, label=n) for n, (c, _) in FAMILIES.items()] + [Patch(color=UNKNOWN_COLOR,
              label="unknown")], ncol=6, fontsize=6.5, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.08))
    ax = axes[1]
    for _, r in tasks.iterrows():
        ax.barh(0, (r.t1 - r.t0) / 60, left=r.t0 / 60, height=0.6, color="#d7d5ce", edgecolor="white")
        if (r.t1 - r.t0) / 60 > 12:                                            # label only segments that fit
            ax.text((r.t0 + r.t1) / 120, 0, r.task.split(" ")[0].split("/")[0], ha="center", va="center",
                    fontsize=6, color=INK, clip_on=True)
    ax.set_yticks([])
    ax.set_title("(b) annotated tasks", loc="left", fontsize=8, color=INK)
    ax = axes[2]
    order = np.argsort(t_ego)
    t_ego, m_ego = t_ego[order], m_ego[order]
    k = 600                                                                    # 1-minute bins at 10 fps
    share, ts = [], []
    for i in range(0, len(m_ego) - k, k):
        seg = m_ego[i:i + k]
        ok = np.isfinite(seg)
        share.append(100 * (seg[ok] > 1.0).mean() if ok.sum() > k // 4 else np.nan)
        ts.append(t_ego[i] + 0.5)
    ax.plot(ts, share, color=FAMILIES["endoscope"][0], linewidth=1.2)
    ax.set_ylim(0, 60)
    ax.set_ylabel("% of frames", fontsize=7)
    ax.set_title("(c) endoscope moving (> 1 % of the frame per 0.1 s), measured from the video, per minute", loc="left",
                 fontsize=8, color=INK)
    ax.set_xlabel("case time (min)")
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="x", color=GRID, linewidth=0.5)
        a.set_axisbelow(True)
    fig.tight_layout(h_pad=0.6)
    out = ROOT / "docs" / "figures"
    fig.savefig(out / "embodiment_timeline.pdf")
    fig.savefig(out / "embodiment_timeline.png", dpi=220)
    print("saved", out / "embodiment_timeline.png")


if __name__ == "__main__":
    main()
