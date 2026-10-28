#!/usr/bin/env python
"""Collect evaluation outputs into result tables for the README (Markdown) and the manuscript (LaTeX).

Reads outputs/eval/<run>/{replay,control}.json and outputs/eval/reference/{reference,control}.json; writes
docs/results/e1_main.md, docs/results/e1_main.json and paper/tmrb/tables/e1_main.tex. Missing runs are skipped.
"""

from __future__ import annotations

import json

from swm.paths import OUTPUTS, ROOT

ROWS = [  # run, label, action interface, camera
    ("S_none", "None (unconditional)", "none", False),
    ("S_state_nocam", "Robot state", "state", False),
    ("S_state", "Robot state + camera", "state", True),
    ("S_global_nocam", "Monolithic latent (Genie)", "global", False),
    ("S_global", "Monolithic latent + camera", "global", True),
    ("S_global_vq", "Monolithic VQ + camera (SurgWM-style)", "global_vq", True),
    ("S_slots", "Slots + camera (FLAM-style)", "slots", True),
    ("S_arms_nocam", "EG-SWM, no camera action", "arms", False),
    ("S_arms", "EG-SWM", "arms", True),
    ("S_arms_scene", "EG-SWM + scene action", "arms", True),
    ("S_arms_nofwd", "EG-SWM, diffusion loss only", "arms", True),
    ("S_arms_poolglobal", "EG-SWM, arm actions not grounded", "arms", True),
    ("S_arms_shuffled", "EG-SWM, wrong arm binding", "arms", True),
    ("S_arms_ui", "EG-SWM + console readout", "arms", True),
]
COLS = [  # key, header, format, higher-is-better
    ("pred_psnr", "PSNR", "{:.2f}", True), ("pred_ssim", "SSIM", "{:.3f}", True),
    ("pred_lpips", "LPIPS", "{:.3f}", False), ("pred_fvd", "FVD", "{:.0f}", False),
    ("delta_psnr", "ΔPSNR", "{:.2f}", True), ("cam_gain_mean", "Cam. gain", "{:.2f}", True),
    ("locality_ratio_median", "Locality", "{:.2f}", True), ("ecs_precision", "ECS-P", "{:.3f}", True),
    ("ecs_recall", "ECS-R", "{:.3f}", True), ("ecs_hallucination_rate", "Halluc.", "{:.3f}", False),
]


def load(run: str) -> dict:
    out = {}
    for name in ("replay", "control"):
        p = OUTPUTS / "eval" / run / f"{name}.json"
        if p.exists():
            out.update(json.loads(p.read_text()))
    gains = [v for k, v in out.items() if k.startswith("cam_") and k.endswith("_gain")]
    if gains:  # sign accuracy saturates at 1.0, so the table reports how much of the commanded motion is realised
        out["cam_gain_mean"] = sum(gains) / len(gains)
    return out


def main() -> None:
    table = []
    ref = OUTPUTS / "eval" / "reference" / "reference.json"
    if ref.exists():
        r = json.loads(ref.read_text())
        ecs = {k: v for k, v in load("reference").items() if k.startswith("ecs_")}   # eval_control.py --reference
        for prefix, label in (("copy_last", "Copy last context frame"), ("recon", "VAE reconstruction (ceiling)")):
            table.append(dict(run=prefix, label=label, **{k.replace(prefix, "pred"): v for k, v in r.items()
                                                          if k.startswith(prefix) and not k.endswith("by_frame")},
                              **(ecs if prefix == "recon" else {})))
        if "ecs_real_hallucination_rate" in ecs:
            table.append(dict(run="real", label="Real frames (detector floor)",
                              ecs_hallucination_rate=ecs["ecs_real_hallucination_rate"]))
    for run, label, action, cam in ROWS:
        d = load(run)
        if d:
            table.append(dict(run=run, label=label, action=action, camera=cam,
                              **{k: d[k] for k, *_ in COLS if k in d}, step=d.get("step")))
    out_md = ROOT / "docs" / "results" / "e1_main.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "results" / "e1_main.json").write_text(json.dumps(table, indent=1))
    hdr = "| Model | " + " | ".join(h + (" ↑" if hi else " ↓") for _, h, _, hi in COLS) + " |"
    lines = [hdr, "|" + "---|" * (len(COLS) + 1)]
    for row in table:
        cells = [fmt.format(row[k]) if isinstance(row.get(k), (int, float)) else "–" for k, _, fmt, _ in COLS]
        lines.append(f"| {row['label']} | " + " | ".join(cells) + " |")
    out_md.write_text("\n".join(lines) + "\n")
    tex = ROOT / "paper" / "tmrb" / "tables" / "e1_main.tex"
    tex.parent.mkdir(parents=True, exist_ok=True)
    keep = COLS[:8]
    rows_tex = []
    for row in table:
        cells = [fmt.format(row[k]) if isinstance(row.get(k), (int, float)) else "--" for k, _, fmt, _ in keep]
        rows_tex.append(row["label"].replace("Δ", r"$\Delta$") + " & " + " & ".join(cells) + r" \\")
    head = " & ".join(h.replace("Δ", r"$\Delta$") + (r"$\uparrow$" if hi else r"$\downarrow$") for _, h, _, hi in keep)
    tex.write_text("\\begin{tabular}{l" + "c" * len(keep) + "}\n\\toprule\nModel & " + head + r" \\" + "\n\\midrule\n"
                   + "\n".join(rows_tex) + "\n\\bottomrule\n\\end{tabular}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
