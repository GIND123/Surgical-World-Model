#!/usr/bin/env python
"""Fidelity against control (E1): PSNR of each action interface against the mean gain of its endoscope action and
against the locality of its learned actions, with 95 % intervals over test cases on the control axes, and the
copied last context frame as the fidelity reference. Reads the generated tables in docs/results (e1_main,
e6_ablations, e1_locality, e1_control), so runs without results are left out.

    python scripts/docs/figure_pareto.py     # -> docs/figures/e1_pareto.{pdf,png}
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from swm.paths import ROOT  # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
BLUE = "#2a78d6"
POINTS = [  # run, label, colour, marker, filled; colours as in e1_horizon
    ("S_arms", "EG-SWM", BLUE, "o", True),
    ("B_arms", "EG-SWM (B)", BLUE, "D", True),
    ("S_arms_scene", "EG-SWM + scene", BLUE, "o", False),
    ("S_global", "Monolithic", "#6250d6", "s", True),
    ("B_global", "Monolithic (B)", "#6250d6", "D", True),
    ("S_slots", "Slots", "#1baf7a", "^", True),
    ("S_global_vq", "VQ", "#eb6834", "v", True),
    ("surgwm", "SurgWM", "#eb6834", "v", False),
    ("S_state", "Robot state", "#eda100", "P", True),
]
LABEL_AT = {  # panel, run -> label offset in points and alignment
    ("gain", "S_arms"): (-4, 5, "right"), ("gain", "B_arms"): (0, 7, "center"),
    ("gain", "S_arms_scene"): (5, -2, "left"), ("gain", "S_global"): (5, 5, "left"),
    ("gain", "S_slots"): (5, -1, "left"), ("gain", "S_global_vq"): (-5, -3, "right"),
    ("gain", "S_state"): (-5, -3, "right"),
    ("loc", "S_arms"): (0, -10, "center"), ("loc", "B_arms"): (0, 7, "center"),
    ("loc", "S_arms_scene"): (0, -9, "center"), ("loc", "S_global"): (0, -9, "center"),
    ("loc", "S_slots"): (5, -1, "left"), ("loc", "S_global_vq"): (5, -3, "left"),
    ("loc", "surgwm"): (5, -3, "left"),
}


def rows(name: str) -> dict:
    p = ROOT / "docs" / "results" / f"{name}.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    return {r["run"]: r for r in (d["rows"] if isinstance(d, dict) else d) if "run" in r}


def main() -> None:
    main_rows = {**rows("e6_ablations"), **rows("e1_main")}
    loc, ctl = rows("e1_locality"), rows("e1_control")
    copy = main_rows["copy_last"]["pred_psnr"]
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 7, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, (ax_g, ax_l) = plt.subplots(1, 2, figsize=(3.5, 2.15), sharey=True)
    for run, label, color, marker, filled in POINTS:
        r = main_rows.get(run)
        if r is None or "pred_psnr" not in r:
            continue
        y = r["pred_psnr"]
        style = dict(color=color, marker=marker, ms=5, mew=1.0, mfc=color if filled else "white", ls="none",
                     elinewidth=0.8, capsize=0, zorder=3)
        if isinstance(r.get("cam_gain_mean"), float):          # endoscope gain, interval once e1_control has rows
            c = ctl.get(run, {})
            x = c.get("gain", r["cam_gain_mean"])
            err = [[x - c["gain_ci"][0]], [c["gain_ci"][1] - x]] if "gain_ci" in c else None
            ax_g.errorbar(x, y, xerr=err, **style)
            dx, dy, ha = LABEL_AT.get(("gain", run), (5, -1, "left"))
            ax_g.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                          color=INK, fontsize=6)
        if run in loc:                                          # learned actions only
            x, (lo, hi) = loc[run]["locality"], loc[run]["locality_ci"]
            ax_l.errorbar(x, y, xerr=[[x - lo], [hi - x]], **style)
            dx, dy, ha = LABEL_AT.get(("loc", run), (5, -1, "left"))
            ax_l.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                          color=INK, fontsize=6)
    for ax in (ax_g, ax_l):
        ax.axhline(copy, color=MUTED, lw=0.8, ls="--", zorder=1)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    ax_l.annotate("copied frame", (1.0, copy), xycoords=("axes fraction", "data"), xytext=(0, 5),
                  textcoords="offset points", ha="right", va="center", color=MUTED, fontsize=6)
    ax_g.set_xlim(-0.05, 1.0)
    ax_g.set_xlabel("endoscope gain")
    ax_g.set_ylabel("PSNR (dB)")
    ax_l.set_xlim(1.0, 4.2)
    ax_l.set_xlabel("locality")
    fig.tight_layout(pad=0.3, w_pad=0.6)
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / "docs" / "figures" / f"e1_pareto.{ext}", dpi=220)
    print("wrote docs/figures/e1_pareto.{pdf,png}")


if __name__ == "__main__":
    main()
