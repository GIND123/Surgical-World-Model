#!/usr/bin/env python
"""Collect evaluation outputs into result tables for the README (Markdown) and the manuscript (LaTeX).

E1 reads outputs/eval/<run>/{replay,control}.json and outputs/eval/reference/{reference,control}.json; PSNR, SSIM
and LPIPS are means over test cases of the per-case means, with 95 % bootstrap intervals over cases (from the
per-window arrays in <run>/replay.npz). E2 reads the kinematic probes outputs/eval/probes/<data>_{zeroshot,adapted}
.json; E3 reads outputs/eval/<run>/suturebot/replay.json. Each experiment writes docs/results/<name>.{md,json} and
paper/tmrb/tables/<name>.tex; missing runs are skipped and an experiment without any result writes nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

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
LABEL = {run: label for run, label, *_ in ROWS}
E2_DATA = [("suturebot", "SutureBot"), ("crcd", "CRCD"), ("jigsaws", "JIGSAWS")]
E2_ROWS = [  # probe name ({d}: dataset), probe file, label, setting
    ("dz_region", "zeroshot", "Latent change in the arm region", "model-free"),
    ("dz_frame", "zeroshot", "Latent change over the frame", "model-free"),
    ("camera", "zeroshot", "Endoscope action", "model-free"),
    *((run, "zeroshot", label, "zero-shot") for run, label, *_ in ROWS),
    *((f"{{d}}_adapt_{a}", "adapted", LABEL[f"S_{a}"], "adapted") for a in ("global", "global_vq", "slots", "arms")),
]
TARGETS = ("pos", "rot", "jaw")                     # end-effector translation, rotation, jaw of the same arm
E3_INITS = [("scratch", "None (from scratch)"), ("arms", LABEL["S_arms"]), ("global", LABEL["S_global"])]
E3_FRACTIONS = ("0.1", "0.25", "1.0")               # share of the SutureBot training episodes with kinematics


PER_WINDOW = ("psnr", "ssim", "lpips")


def by_case(npz: Path, prefix: str, n_boot: int = 2000) -> dict:
    """Per-window metrics -> mean over test cases and a 95 % interval from resampling cases."""
    if not npz.exists():
        return {}
    d = np.load(npz)
    if "case" not in d:
        return {}
    case = d["case"]
    cases = sorted(set(case.tolist()))
    pick = np.random.default_rng(0).integers(0, len(cases), (n_boot, len(cases)))
    out = {}
    for m in PER_WINDOW:
        x = d[f"{prefix}_{m}"].mean(1)
        per = np.array([x[case == c].mean() for c in cases])
        out[f"pred_{m}"] = float(per.mean())
        out[f"pred_{m}_ci"] = np.quantile(per[pick].mean(1), [0.025, 0.975]).tolist()
    return out


def cell(row: dict, key: str, fmt: str, tex: bool = False) -> str:
    if not isinstance(row.get(key), (int, float)):
        return "--" if tex else "–"
    v = fmt.format(row[key])
    if f"{key}_ci" in row:
        lo, hi = (fmt.format(x) for x in row[f"{key}_ci"])
        v += rf"\,{{\scriptsize[{lo}, {hi}]}}" if tex else f" [{lo}, {hi}]"
    return v.replace("-", "$-$") if tex else v


def load(run: str) -> dict:
    out = {}
    for name in ("replay", "control"):
        p = OUTPUTS / "eval" / run / f"{name}.json"
        if p.exists():
            out.update(json.loads(p.read_text()))
    out.update(by_case(OUTPUTS / "eval" / run / "replay.npz", "pred"))
    gains = [v for k, v in out.items() if k.startswith("cam_") and k.endswith("_gain")]
    if gains:  # sign accuracy saturates at 1.0, so the table reports how much of the commanded motion is realised
        out["cam_gain_mean"] = sum(gains) / len(gains)
    return out


def reference(ref: Path) -> list[dict]:
    """Rows for copying the last context frame and for the VAE reconstruction (eval_wm.py --reference)."""
    r = json.loads(ref.read_text())
    rows = []
    for prefix, label in (("copy_last", "Copy last context frame"), ("recon", "VAE reconstruction (ceiling)")):
        row = {k.replace(prefix, "pred"): v for k, v in r.items()
               if k.startswith(prefix) and not k.endswith("by_frame")}
        row.update(by_case(ref.with_suffix(".npz"), prefix))
        rows.append(dict(run=prefix, label=label, **row))
    return rows


def write(name: str, data: list | dict, md: list[str], tex: str) -> None:
    """docs/results/<name>.{json,md} and paper/tmrb/tables/<name>.tex."""
    for d in (ROOT / "docs" / "results", ROOT / "paper" / "tmrb" / "tables"):
        d.mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "results" / f"{name}.json").write_text(json.dumps(data, indent=1))
    (ROOT / "docs" / "results" / f"{name}.md").write_text("\n".join(md) + "\n")
    (ROOT / "paper" / "tmrb" / "tables" / f"{name}.tex").write_text(tex)
    print("\n".join(md) + "\n")


def tabular(cols: str, head: list[str], rows: list[str]) -> str:
    return (f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n" + "\n".join(head) + "\n\\midrule\n" + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n")


def e1() -> None:
    """Fidelity and control of each action interface on the SurgVU test cases."""
    table = []
    ref = OUTPUTS / "eval" / "reference" / "reference.json"
    if ref.exists():
        table += reference(ref)
        ecs = {k: v for k, v in load("reference").items() if k.startswith("ecs_")}   # eval_control.py --reference
        table[-1].update(ecs)
        if "ecs_real_hallucination_rate" in ecs:
            table.append(dict(run="real", label="Real frames (detector floor)",
                              ecs_hallucination_rate=ecs["ecs_real_hallucination_rate"]))
    for run, label, action, cam in ROWS:
        d = load(run)
        if d:
            table.append(dict(run=run, label=label, action=action, camera=cam, step=d.get("step"),
                              **{k: d[k] for k in d if k.removesuffix("_ci") in {c for c, *_ in COLS}}))
    if not table:
        return
    hdr = "| Model | " + " | ".join(h + (" ↑" if hi else " ↓") for _, h, _, hi in COLS) + " |"
    lines = [hdr, "|" + "---|" * (len(COLS) + 1)]
    for row in table:
        cells = [cell(row, k, fmt) for k, _, fmt, _ in COLS]
        lines.append(f"| {row['label']} | " + " | ".join(cells) + " |")
    keep = COLS[:7] + COLS[9:]                  # the paper keeps the hallucination rate, which has a real-frame floor
    rows_tex = []
    for row in table:
        cells = [cell(row, k, fmt, tex=True) for k, _, fmt, _ in keep]
        rows_tex.append(row["label"].replace("Δ", r"$\Delta$") + " & " + " & ".join(cells) + r" \\")
    head = " & ".join(h.replace("Δ", r"$\Delta$") + (r"$\uparrow$" if hi else r"$\downarrow$") for _, h, _, hi in keep)
    write("e1_main", table, lines, tabular("l" + "c" * len(keep), ["Model & " + head + r" \\"], rows_tex))


def e2() -> None:
    """Kinematic grounding: R^2 of the linear probes on held-out episodes per target, averaged over the two arms
    (per arm in the JSON), for model-free features, frozen SurgVU models and models adapted without kinematics."""
    probes = {}
    for d, _ in E2_DATA:
        for src in ("zeroshot", "adapted"):
            p = OUTPUTS / "eval" / "probes" / f"{d}_{src}.json"
            if p.exists():
                probes[d, src] = json.loads(p.read_text())
    if not probes:
        return
    table = []
    for name, src, label, setting in E2_ROWS:
        row = dict(probe=name.format(d="<data>"), label=label, setting=setting)
        for d, _ in E2_DATA:
            arms = probes.get((d, src), {}).get("probes", {}).get(name.format(d=d))
            if arms:
                row[d] = dict(arms, mean={t: float(np.mean([a[t] for a in arms.values()])) for t in TARGETS})
        if any(d in row for d, _ in E2_DATA):
            table.append(row)
    flat = [{f"{d}_{t}": row[d]["mean"][t] for d, _ in E2_DATA if d in row for t in TARGETS} for row in table]
    keys = [f"{d}_{t}" for d, _ in E2_DATA for t in TARGETS]
    lines = ["| Features | Setting | " + " | ".join(f"{n} {t}" for _, n in E2_DATA for t in TARGETS) + " |",
             "|" + "---|" * (len(keys) + 2)]
    rows_tex, last = [], None
    for row, f in zip(table, flat, strict=True):
        lines.append(f"| {row['label']} | {row['setting']} | " + " | ".join(cell(f, k, "{:.2f}") for k in keys) + " |")
        if last not in (None, row["setting"]):
            rows_tex.append(r"\midrule")
        last = row["setting"]
        rows_tex.append(f"{row['label']} & {row['setting']} & "
                        + " & ".join(cell(f, k, "{:.2f}", tex=True) for k in keys) + r" \\")
    head = [" & & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{n}}}" for _, n in E2_DATA) + r" \\",
            "".join(rf"\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(len(E2_DATA))),
            "Features & Setting & " + " & ".join(t for _ in E2_DATA for t in TARGETS) + r" \\"]
    data = {"datasets": {f"{d}_{s}": {k: v for k, v in p.items() if k != "probes"} for (d, s), p in probes.items()},
            "rows": table}
    write("e2_probes", data, lines, tabular("ll" + "c" * len(keys), head, rows_tex))


def e3() -> None:
    """Kinematics-conditioned prediction on the SutureBot test episodes by initialisation and by the share of the
    training episodes used."""
    table = []
    for init, label in E3_INITS:
        for f in E3_FRACTIONS:
            p = OUTPUTS / "eval" / f"SB_kin_{init}_f{f}" / "suturebot" / "replay.json"
            if p.exists():
                d = json.loads(p.read_text())
                table.append(dict(run=f"SB_kin_{init}_f{f}", label=label, fraction=float(f), step=d.get("step"),
                                  **{k: d[k] for k, *_ in COLS[:5] if k in d}))
    if not table:
        return
    ref = OUTPUTS / "eval" / "reference" / "suturebot" / "reference.json"
    table = (reference(ref) if ref.exists() else []) + table
    cols = COLS[:5]                                                     # PSNR, SSIM, LPIPS, FVD, ΔPSNR
    lines = ["| Initialisation | Training episodes | " + " | ".join(h + (" ↑" if hi else " ↓") for _, h, _, hi in cols)
             + " |", "|" + "---|" * (len(cols) + 2)]
    rows_tex = []
    for row in table:
        share = f"{row['fraction']:.0%}" if "fraction" in row else "–"
        share_tex = share.replace("%", r"\,\%").replace("–", "--")
        lines.append(f"| {row['label']} | {share} | " + " | ".join(cell(row, k, fmt) for k, _, fmt, _ in cols) + " |")
        rows_tex.append(f"{row['label']} & {share_tex} & "
                        + " & ".join(cell(row, k, fmt, tex=True) for k, _, fmt, _ in cols) + r" \\")
    head = " & ".join(h.replace("Δ", r"$\Delta$") + (r"$\uparrow$" if hi else r"$\downarrow$") for _, h, _, hi in cols)
    write("e3_kinematics", table, lines,
          tabular("ll" + "c" * len(cols), ["Initialisation & Training episodes & " + head + r" \\"], rows_tex))


def main() -> None:
    e1()
    e2()
    e3()


if __name__ == "__main__":
    main()
