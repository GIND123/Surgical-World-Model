#!/usr/bin/env python
"""Collect evaluation outputs into result tables for the README (Markdown) and the manuscript (LaTeX).

E1 (action interfaces) and E6 (ablations of EG-SWM) read outputs/eval/<run>/{replay,control}.json and
outputs/eval/reference/{reference,control}.json; PSNR, SSIM and LPIPS are means over test cases of the per-case
means, with 95 % bootstrap intervals over cases (from the per-window arrays in <run>/replay.npz); locality has its
interval and its difference to EG-SWM from <run>/{locality,control}_rows.json (e1_locality), endoscope gain and
embodiment consistency theirs from <run>/{control,camera_ecs}_rows.json (e1_control); e1_contrasts pairs
other models the same way (CONTRASTS), and numbers() writes every number the manuscript's text quotes as a \\val{key}
macro (paper/tmrb/tables/numbers.tex, docs/results/numbers.json). Seeds: mean and
standard deviation over the runs trained with seeds 0-2. Sensitivity: sampler, context length and endoscope-command
scale on 200 (100) of the test windows. E2 reads the kinematic probes outputs/eval/probes/<data>_{zeroshot,allarms,
adapted}.json; E3 and E4 read outputs/eval/<run>/<data>/replay.json, with intervals over test episodes. Each table is
written to docs/results/<name>.{md,json} and paper/tmrb/tables/<name>.tex; missing runs are skipped and a table
without any result is not written.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

from swm.paths import OUTPUTS, ROOT

ROWS = [  # E1 action interfaces: run, label, action interface, camera
    ("S_none", "None (unconditional)", "none", False),
    ("S_state_nocam", "Robot state", "state", False),
    ("S_state", "Robot state + camera", "state", True),
    ("S_global_nocam", "Monolithic latent (Genie)", "global", False),
    ("S_global", "Monolithic latent + camera", "global", True),
    ("surgwm", "SurgWM (re-implementation)", "surgwm", False),
    ("S_global_vq", "Monolithic VQ + camera (SurgWM-style)", "global_vq", True),
    ("S_slots", "Slots + camera (FLAM-style)", "slots", True),
    ("S_arms_nocam", "EG-SWM, no camera action", "arms", False),
    ("S_arms", "EG-SWM", "arms", True),
    ("S_arms_scene", "EG-SWM + scene action", "arms", True),
]
ABLATIONS = [  # E6: EG-SWM (S, 60k steps) with one part changed
    ("S_arms_nofwd", "EG-SWM, diffusion loss only", "arms", True),
    ("S_arms_poolglobal", "EG-SWM, arm actions not grounded", "arms", True),
    ("S_arms_shuffled", "EG-SWM, wrong arm binding", "arms", True),
    ("S_arms_ui", "EG-SWM + console readout", "arms", True),
    ("S_arms_hold", "EG-SWM, boxes held across detector gaps", "arms", True),
    ("S_arms_kl1e-3", "EG-SWM, KL weight 10⁻³", "arms", True),
    ("S_arms_kl1e-5", "EG-SWM, KL weight 10⁻⁵", "arms", True),
    ("S_arms_data0.1", "EG-SWM, 10 % of the training cases", "arms", True),
    ("S_arms_data0.3", "EG-SWM, 30 % of the training cases", "arms", True),
]
SCALE = [  # E6: EG-SWM at a larger size, the recipe of size S without the colour-bar windows
    ("B_arms", "EG-SWM, size B", "arms", True),
]
SEEDS = ("S_arms", "S_global", "S_slots", "S_global_vq")    # also trained with seeds 1 and 2 (<run>_seed<n>)
SENS_RUNS = ("S_arms", "S_global", "S_slots", "S_global_vq", "S_arms_scene", "B_arms")
SAMPLER = (("1.0", "20"), ("1.0", "50"), ("1.0", "100"), ("2.0", "50"))   # guidance scale, Euler steps
CAM_RUNS, CAM_SCALES = ("S_arms", "S_global_vq", "S_state", "B_arms"), ("0.5", "1.0", "2.0")
LONG_RUNS = ("S_arms", "S_global", "S_slots", "S_global_vq", "S_state", "S_none", "S_arms_scene", "B_arms")
DROPS = ("0.25", "0.5", "1.0")       # share of the instrument detections removed at test time (job 335)
CONTRASTS = [  # (a, b): a minus b, paired over the test cases on the E1 windows (e1_contrasts; quoted in the text)
    ("S_arms", "copy_last"), ("S_arms", "S_none"), ("S_slots", "S_arms"), ("S_global", "S_arms"),
    ("S_global_nocam", "S_arms"), ("S_arms_scene", "S_arms"), ("S_arms", "S_global_vq"), ("S_arms", "S_state"),
    ("S_global_vq", "copy_last"), ("S_arms", "S_arms_nocam"), ("S_arms", "S_arms_nofwd"),
    ("S_arms_nocam", "S_state_nocam"), ("S_state_nocam", "S_none"), ("S_state", "S_state_nocam"),
    ("S_global_nocam", "S_global"),
    ("S_arms", "surgwm"), ("surgwm", "copy_last"),
    *((run, "S_arms") for run, *_ in ABLATIONS + SCALE),
    ("B_arms", "copy_last"), ("S_global", "B_arms"), ("S_slots", "B_arms"), ("B_arms", "S_global_vq"),
    ("B_arms", "surgwm"),
]
COLS = [  # key, header, format, higher-is-better
    ("pred_psnr", "PSNR", "{:.2f}", True), ("pred_ssim", "SSIM", "{:.3f}", True),
    ("pred_lpips", "LPIPS", "{:.3f}", False), ("pred_fvd", "FVD", "{:.0f}", False),
    ("delta_psnr", "ΔPSNR", "{:.2f}", True), ("cam_gain_mean", "Cam. gain", "{:.2f}", True),
    ("locality_ratio_median", "Locality", "{:.2f}", True), ("ecs_precision", "ECS-P", "{:.3f}", True),
    ("ecs_recall", "ECS-R", "{:.3f}", True), ("ecs_hallucination_rate", "Halluc.", "{:.3f}", False),
]
LABEL = {run: label for run, label, *_ in ROWS + ABLATIONS + SCALE}
E2_DATA = [("suturebot", "SutureBot"), ("crcd", "CRCD"), ("jigsaws", "JIGSAWS")]
E2_ROWS = [  # probe name ({d}: dataset), probe file, label, setting
    ("dz_region", "zeroshot", "Latent change in the arm region", "model-free"),
    ("dz_frame", "zeroshot", "Latent change over the frame", "model-free"),
    ("camera", "zeroshot", "Endoscope action", "model-free"),
    *((run, "zeroshot", label, "zero-shot") for run, label, *_ in ROWS + ABLATIONS + SCALE),
    ("S_arms_all", "allarms", "EG-SWM, every arm's action", "zero-shot"),  # as many features as a frame-wide action
    ("B_arms_all", "allarms", "EG-SWM, size B, every arm's action", "zero-shot"),
    *((f"{{d}}_adapt_{a}", "adapted", LABEL[f"S_{a}"], "adapted") for a in ("global", "global_vq", "slots", "arms")),
]
TARGETS = ("pos", "rot", "jaw")                     # end-effector translation, rotation, jaw of the same arm
E3_DATA = [("suturebot", "SB", "SutureBot"), ("srth", "SRTH", "SRT-H")]   # dataset, run prefix, name
E3_INITS = [("scratch", "None (from scratch)"), ("arms", LABEL["S_arms"]), ("global", LABEL["S_global"])]
E3_FRACTIONS = ("0.1", "0.25", "1.0")               # share of the training episodes with kinematics
E4_ROWS = [  # SAR-RARP50: run, label, setting
    *((run, LABEL[run], "zero-shot") for run in ("S_none", "S_global", "S_global_vq", "S_slots", "S_arms",
                                                 "S_arms_scene", "B_arms")),
    *((f"SAR_adapt_{a}", LABEL[f"S_{a}"], "adapted") for a in ("global", "global_vq", "slots", "arms")),
    ("SAR_scratch_arms", LABEL["S_arms"], "from scratch"),
]
TEX = {"Δ": r"$\Delta$", "−": r"$-$", "β": r"$\beta$", "10⁻³": r"$10^{-3}$", "10⁻⁵": r"$10^{-5}$", " %": r"\,\%",
       "±": r"$\pm$"}


def tex(s: str) -> str:
    for k, v in TEX.items():
        s = s.replace(k, v)
    return re.sub(r"(?<!\\)%", r"\\%", s)                         # "10%" too, not only "10 %"


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


def paired_psnr(a: Path, b: Path, pa: str = "pred", pb: str = "pred", n_boot: int = 2000) -> dict:
    """PSNR of a minus b on the same windows: mean over test cases (episodes) of the per-case difference, its 95 %
    interval from resampling cases, and the number of cases in which a is better."""
    if not (a.exists() and b.exists()):
        return {}
    da, db = np.load(a), np.load(b)
    if "case" not in da or "case" not in db or not np.array_equal(da["window"], db["window"]):
        return {}
    cases = sorted(set(da["case"].tolist()))
    x = da[f"{pa}_psnr"].mean(1) - db[f"{pb}_psnr"].mean(1)
    d = np.array([x[da["case"] == c].mean() for c in cases])
    pick = np.random.default_rng(0).integers(0, len(cases), (n_boot, len(cases)))
    return dict(d_psnr=float(d.mean()), d_psnr_ci=np.quantile(d[pick].mean(1), [0.025, 0.975]).tolist(),
                wins=int((d > 0).sum()), n_cases=len(cases))


def cell(row: dict, key: str, fmt: str, tex: bool = False) -> str:
    if not isinstance(row.get(key), (int, float)):
        return "--" if tex else "–"
    v = fmt.format(row[key])
    if f"{key}_ci" in row:
        lo, hi = (fmt.format(x) for x in row[f"{key}_ci"])
        v += rf"\,{{\scriptsize[{lo}, {hi}]}}" if tex else f" [{lo}, {hi}]"
    return v.replace("-", "$-$") if tex else v


def test_rows(run: str, test: str) -> list | dict | None:
    """Per-window results of one test of eval_control.py (camera: per command; locality: per window and arm; ecs: per
    window and frame) with the test case: from the control evaluation, else from the re-evaluations of the runs
    evaluated before eval_control.py recorded them (job 306: locality; job 336: camera and ecs, whose noise draws
    are those of the full evaluation)."""
    for tag in ("control", "locality", "camera_ecs"):
        p = OUTPUTS / "eval" / run / f"{tag}_rows.json"
        if p.exists():
            rows = json.loads(p.read_text())[test]
            first = next(iter(rows.values())) if isinstance(rows, dict) else rows
            if first and "case" in first[0]:
                return rows
    return None


def locality_rows(run: str) -> list[dict] | None:
    return test_rows(run, "locality")


def load(run: str) -> dict:
    out = {}
    for name in ("replay", "control", "camera_ecs"):         # camera_ecs: the re-evaluation with rows (job 336)
        p = OUTPUTS / "eval" / run / f"{name}.json"
        if p.exists():
            out.update(json.loads(p.read_text()))
    rows = locality_rows(run)
    if rows:                                         # the same source as the intervals of locality()
        out["locality_ratio_median"] = float(np.median([r["ratio"] for r in rows]))
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


def interface_table(name: str, runs: list[tuple], table: list[dict]) -> None:
    """E1/E6 table: `table` (reference rows) followed by every run of `runs` that has results."""
    for run, label, action, cam in runs:
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
        rows_tex.append(tex(row["label"]) + " & " + " & ".join(cells) + r" \\")
    head = " & ".join(tex(h) + (r"$\uparrow$" if hi else r"$\downarrow$") for _, h, _, hi in keep)
    write(name, table, lines, tabular("l" + "c" * len(keep), ["Model & " + head + r" \\"], rows_tex))


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
    interface_table("e1_main", ROWS, table)


def e6() -> None:
    """Ablations: EG-SWM with one part changed, next to EG-SWM itself."""
    if any(load(run) for run, *_ in ABLATIONS + SCALE):
        interface_table("e6_ablations", [r for r in ROWS if r[0] == "S_arms"] + ABLATIONS + SCALE, [])


def seeds() -> None:
    """Run-to-run variation: mean and standard deviation over seeds 0 (the E1 run), 1 and 2."""
    table = []
    for run in SEEDS:
        got = [d for d in (load(r) for r in (run, f"{run}_seed1", f"{run}_seed2")) if "pred_psnr" in d]
        if len(got) < 2:
            continue
        row = dict(run=run, label=LABEL[run], n_seeds=len(got))
        for k, *_ in COLS:
            v = [d[k] for d in got if isinstance(d.get(k), (int, float))]
            if len(v) == len(got):
                row[k], row[f"{k}_sd"] = float(np.mean(v)), float(np.std(v, ddof=1))
        table.append(row)
    if not table:
        return
    cols = COLS[:7]

    def pm(row: dict, k: str, fmt: str) -> str:
        return f"{fmt.format(row[k])} ± {fmt.format(row[k + '_sd'])}" if k in row else "–"

    lines = ["| Model | Seeds | " + " | ".join(h + (" ↑" if hi else " ↓") for _, h, _, hi in cols) + " |",
             "|" + "---|" * (len(cols) + 2)]
    lines += [f"| {r['label']} | {r['n_seeds']} | " + " | ".join(pm(r, k, f) for k, _, f, _ in cols) + " |"
              for r in table]
    rows_tex = [tex(f"{r['label']} & {r['n_seeds']} & " + " & ".join(pm(r, k, f) for k, _, f, _ in cols))
                .replace("–", "--") + r" \\" for r in table]
    head = " & ".join(tex(h) + (r"$\uparrow$" if hi else r"$\downarrow$") for _, h, _, hi in cols)
    write("e1_seeds", table, lines, tabular("lc" + "c" * len(cols), ["Model & Seeds & " + head + r" \\"], rows_tex))


def clean(tag_dir: Path, tag: str) -> dict | None:
    """A sensitivity result whose windows are known to leave out the excluded test case (job 204 ran before the
    exclusion and recorded no cases)."""
    js, npz = tag_dir / f"{tag}.json", tag_dir / f"{tag}.npz"
    if not (js.exists() and npz.exists()):
        return None
    cases = np.load(npz)
    if "case" not in cases or "case_126" in set(cases["case"].tolist()):
        return None
    return json.loads(js.read_text())


def sensitivity() -> None:
    """Sampler (Euler steps, guidance), context length and endoscope-command scale, on 200 (camera: 100) of the
    test windows spread over the 19 cases."""
    rows = []
    for run in SENS_RUNS:
        for g, n in SAMPLER:
            d = clean(OUTPUTS / "eval" / run, f"sampler_cfg{g}_s{n}")
            if d:
                rows.append(dict(run=run, label=LABEL[run], guidance=float(g), steps=int(n), pred_psnr=d["pred_psnr"],
                                 pred_ssim=d["pred_ssim"], pred_lpips=d["pred_lpips"],
                                 psnr_end=d["pred_psnr_by_frame"][-1], delta_psnr=d["delta_psnr"]))
    if rows:
        cols = [("pred_psnr", "PSNR ↑", "{:.2f}"), ("pred_ssim", "SSIM ↑", "{:.3f}"),
                ("pred_lpips", "LPIPS ↓", "{:.3f}"), ("psnr_end", "PSNR at 2.8 s ↑", "{:.2f}"),
                ("delta_psnr", "ΔPSNR ↑", "{:.2f}")]
        lines = ["| Model | Euler steps | Guidance | " + " | ".join(h for _, h, _ in cols) + " |",
                 "|" + "---|" * (len(cols) + 3)]
        lines += [f"| {r['label']} | {r['steps']} | {r['guidance']:g} | "
                  + " | ".join(cell(r, k, f) for k, _, f in cols) + " |" for r in rows]
        rows_tex = [tex(f"{r['label']} & {r['steps']} & {r['guidance']:g} & ")
                    + " & ".join(cell(r, k, f, tex=True) for k, _, f in cols) + r" \\" for r in rows]
        head = "Model & Euler steps & Guidance & " + " & ".join(
            tex(h).replace("↑", r"$\uparrow$").replace("↓", r"$\downarrow$") for _, h, _ in cols) + r" \\"
        write("e1_sampler", rows, lines, tabular("lcc" + "c" * len(cols), [head], rows_tex))
    ctx = []                   # PSNR over the frames that every context length predicts (0.9-3.2 s)
    for run in SENS_RUNS:
        row = dict(run=run, label=LABEL[run])
        for n_ctx, tag in ((1, "ctx1"), (2, "sampler_cfg1.0_s50"), (3, "ctx3")):
            d = clean(OUTPUTS / "eval" / run, tag)
            if d:
                first = 4 * (n_ctx - 1) + 1
                row[f"ctx{n_ctx}"] = float(np.mean(d["pred_psnr_by_frame"][9 - first:]))
        if len(row) > 2:
            ctx.append(row)
    if ctx:
        heads = [("ctx1", "1 latent (0.1 s)"), ("ctx2", "2 latents (0.5 s)"), ("ctx3", "3 latents (0.9 s)")]
        lines = ["| Model | " + " | ".join(h for _, h in heads) + " |", "|" + "---|" * (len(heads) + 1)]
        lines += [f"| {r['label']} | " + " | ".join(cell(r, k, "{:.2f}") for k, _ in heads) + " |" for r in ctx]
        rows_tex = [tex(r["label"]) + " & " + " & ".join(cell(r, k, "{:.2f}", tex=True) for k, _ in heads) + r" \\"
                    for r in ctx]
        write("e1_context", ctx, lines, tabular("lccc", ["Model & " + " & ".join(h for _, h in heads) + r" \\"],
                                                rows_tex))
    cam = []
    for run in CAM_RUNS:
        for sc in CAM_SCALES:
            p = OUTPUTS / "eval" / run / f"control_scale{sc}.json"
            if p.exists():
                d = json.loads(p.read_text())
                gains = [v for k, v in d.items() if k.startswith("cam_") and k.endswith("_gain")]
                cam.append(dict(run=run, label=LABEL[run], scale=float(sc), cam_gain_mean=float(np.mean(gains)),
                                **{k: v for k, v in d.items() if k.startswith("cam_")}))     # gain, direction
    if cam:
        lines = ["| Model | Command scale | Mean gain | Right direction |", "|---|---|---|---|"]
        lines += [f"| {r['label']} | ×{r['scale']:g} | {r['cam_gain_mean']:.2f} | {r['cam_sign_acc_mean']:.1%} |"
                  for r in cam]
        rows_tex = [tex(f"{r['label']} & $\\times${r['scale']:g} & {r['cam_gain_mean']:.2f} & "
                        f"{100 * r['cam_sign_acc_mean']:.1f} %") + r" \\" for r in cam]
        write("e1_camera_scale", cam, lines,
              tabular("lccc", [r"Model & Command scale & Mean gain & Right direction \\"], rows_tex))


def paired() -> None:
    """Each model against EG-SWM on the same test windows: per-case differences (model minus EG-SWM), their mean with
    a 95 % bootstrap interval over cases, and the number of cases in which EG-SWM is better."""
    base = OUTPUTS / "eval" / "S_arms" / "replay.npz"
    if not base.exists():
        return
    b = np.load(base)
    cases = sorted(set(b["case"].tolist()))
    pick = np.random.default_rng(0).integers(0, len(cases), (2000, len(cases)))

    def per_case(d, prefix: str, m: str) -> np.ndarray:
        x = d[f"{prefix}_{m}"].mean(1)
        return np.array([x[d["case"] == c].mean() for c in cases])

    srcs = [("copy_last", "Copy last context frame", OUTPUTS / "eval" / "reference" / "reference.npz", "copy_last")]
    srcs += [(run, label, OUTPUTS / "eval" / run / "replay.npz", "pred") for run, label, *_ in ROWS + ABLATIONS + SCALE
             if run != "S_arms"]
    table = []
    for run, label, npz, prefix in srcs:
        if not npz.exists():
            continue
        d = np.load(npz)
        if "case" not in d or not np.array_equal(d["window"], b["window"]):
            continue                                                # not the E1 windows
        row = dict(run=run, label=label)
        for m, better in (("psnr", 1), ("ssim", 1), ("lpips", -1)):
            diff = per_case(d, prefix, m) - per_case(b, "pred", m)
            row[f"d_{m}"] = float(diff.mean())
            row[f"d_{m}_ci"] = np.quantile(diff[pick].mean(1), [0.025, 0.975]).tolist()
            row[f"egswm_better_{m}"] = int((better * diff < 0).sum())
        row["n_cases"] = len(cases)
        table.append(row)
    if not table:
        return
    cols = [("d_psnr", "ΔPSNR", "{:.2f}"), ("d_ssim", "ΔSSIM", "{:.3f}"), ("d_lpips", "ΔLPIPS", "{:.3f}")]
    lines = ["| Model | " + " | ".join(f"{h} vs EG-SWM | EG-SWM better ({h[1:]})" for _, h, _ in cols) + " |",
             "|" + "---|" * (2 * len(cols) + 1)]
    rows_tex = []
    for r in table:
        n = r["n_cases"]
        lines.append(f"| {r['label']} | " + " | ".join(f"{cell(r, k, f)} | {r['egswm_better_' + k[2:]]}/{n}"
                                                       for k, _, f in cols) + " |")
        rows_tex.append(tex(r["label"]) + " & " + " & ".join(f"{cell(r, k, f, tex=True)} & {r['egswm_better_' + k[2:]]}"
                                                             for k, _, f in cols) + r" \\")
    head = "Model & " + " & ".join(rf"{tex(h)} & better" for _, h, _ in cols) + r" \\"
    write("e1_paired", table, lines, tabular("l" + "cc" * len(cols), [head], rows_tex))


def contrasts() -> None:
    """The pairs of CONTRASTS (a minus b) on the E1 windows: per-case differences, their mean with a 95 % bootstrap
    interval over cases (the resampling of paired()), and the number of cases in which a is better."""
    base = OUTPUTS / "eval" / "S_arms" / "replay.npz"
    if not base.exists():
        return
    window = np.load(base)["window"]
    src = {"copy_last": (OUTPUTS / "eval" / "reference" / "reference.npz", "copy_last")}
    label = dict(LABEL, copy_last="Copy last context frame")
    table, cases, pick = [], None, None
    for a, b in CONTRASTS:
        per = []
        for run in (a, b):
            npz, prefix = src.get(run, (OUTPUTS / "eval" / run / "replay.npz", "pred"))
            d = np.load(npz) if npz.exists() else None
            if d is None or "case" not in d or not np.array_equal(d["window"], window):
                break
            if cases is None:
                cases = sorted(set(d["case"].tolist()))
                pick = np.random.default_rng(0).integers(0, len(cases), (2000, len(cases)))
            per.append({m: np.array([d[f"{prefix}_{m}"].mean(1)[d["case"] == c].mean() for c in cases])
                        for m in PER_WINDOW})
        if len(per) < 2:
            continue
        row = dict(a=a, b=b, label=f"{label[a]} − {label[b]}", n_cases=len(cases))
        for m, sign in (("psnr", 1), ("ssim", 1), ("lpips", -1)):
            diff = per[0][m] - per[1][m]
            row[f"d_{m}"] = float(diff.mean())
            row[f"d_{m}_ci"] = np.quantile(diff[pick].mean(1), [0.025, 0.975]).tolist()
            row[f"a_better_{m}"] = int((sign * diff > 0).sum())
        table.append(row)
    if not table:
        return
    cols = [("d_psnr", "ΔPSNR", "{:.2f}"), ("d_ssim", "ΔSSIM", "{:.3f}"), ("d_lpips", "ΔLPIPS", "{:.3f}")]
    lines = ["| a − b | " + " | ".join(f"{h} | a better ({h[1:]})" for _, h, _ in cols) + " |",
             "|" + "---|" * (2 * len(cols) + 1)]
    lines += [f"| {r['label']} | " + " | ".join(f"{cell(r, k, f)} | {r['a_better_' + k[2:]]}/{r['n_cases']}"
                                               for k, _, f in cols) + " |" for r in table]
    rows_tex = [tex(r["label"]) + " & " + " & ".join(f"{cell(r, k, f, tex=True)} & {r['a_better_' + k[2:]]}"
                                                     for k, _, f in cols) + r" \\" for r in table]
    head = r"$a - b$ & " + " & ".join(rf"{tex(h)} & $a$ better" for _, h, _ in cols) + r" \\"
    write("e1_contrasts", table, lines, tabular("l" + "cc" * len(cols), [head], rows_tex))


def locality() -> None:
    """Locality of each interface's learned actions (median over windows and instruments) with a 95 % interval from
    resampling test cases, all rows of a drawn case together; and paired against EG-SWM on the same windows and
    instruments: the difference of the medians over the same resampled cases, and the number of cases in which
    EG-SWM's arm actions are the more local."""
    base = locality_rows("S_arms")
    if base is None:
        return
    cases = sorted({r["case"] for r in base})
    pick = np.random.default_rng(0).integers(0, len(cases), (2000, len(cases)))

    def per_case(rows: list[dict]) -> list[np.ndarray]:
        return [np.array([r["ratio"] for r in rows if r["case"] == c]) for c in cases]

    def boot(per: list[np.ndarray]) -> np.ndarray:
        return np.array([np.median(np.concatenate([per[j] for j in p])) for p in pick])

    b_per = per_case(base)
    b_med, b_boot = float(np.median(np.concatenate(b_per))), boot(b_per)
    table = []
    for run, label, *_ in ROWS + ABLATIONS + SCALE:
        rows = locality_rows(run)
        if rows is None:
            continue
        per = per_case(rows)
        boots = boot(per)
        row = dict(run=run, label=label, n_rows=len(rows), locality=float(np.median(np.concatenate(per))),
                   locality_ci=np.quantile(boots, [0.025, 0.975]).tolist())
        if run != "S_arms":
            both = [(np.median(x), np.median(y)) for x, y in zip(per, b_per, strict=True) if len(x) and len(y)]
            row.update(d_locality=row["locality"] - b_med,
                       d_locality_ci=np.quantile(boots - b_boot, [0.025, 0.975]).tolist(),
                       egswm_better=int(sum(m < e for m, e in both)), n_cases=len(both))
        table.append(row)

    def better(r: dict) -> str:
        return f"{r['egswm_better']}/{r['n_cases']}" if "egswm_better" in r else "–"

    lines = ["| Model | Locality ↑ | ΔLocality vs EG-SWM | EG-SWM more local (cases) |", "|---|---|---|---|"]
    lines += [f"| {r['label']} | {cell(r, 'locality', '{:.2f}')} | {cell(r, 'd_locality', '{:.2f}')} | {better(r)} |"
              for r in table]
    rows_tex = [tex(r["label"]) + f" & {cell(r, 'locality', '{:.2f}', tex=True)} & "
                f"{cell(r, 'd_locality', '{:.2f}', tex=True)} & {better(r).replace('–', '--')}" + r" \\"
                for r in table]
    head = r"Model & Locality$\uparrow$ & $\Delta$ vs EG-SWM & EG-SWM more local \\"
    write("e1_locality", table, lines, tabular("lccc", [head], rows_tex))


def control_intervals() -> None:
    """Endoscope gain and embodiment consistency with 95 % intervals from resampling test cases (all windows of a
    drawn case together, as in locality()), and paired against EG-SWM on the same windows: the difference over the
    same resampled cases and the number of cases in which EG-SWM is better. The gain is the mean over the six
    commands of the median realised/commanded motion along the commanded axis (eval_control.py); the embodiment
    rates are ratios of counts summed over the frames of the drawn cases."""
    base_cam, base_ecs = test_rows("S_arms", "camera"), test_rows("S_arms", "ecs")
    if base_cam is None or base_ecs is None:
        return
    cases = sorted({r["case"] for rows in base_cam.values() for r in rows})
    draws = np.concatenate([np.arange(len(cases))[None],                  # row 0: every case once
                            np.random.default_rng(0).integers(0, len(cases), (2000, len(cases)))])

    def ci(x: np.ndarray) -> list:
        return np.quantile(x[1:], [0.025, 0.975]).tolist()

    def gains(cam: dict) -> tuple[np.ndarray, np.ndarray]:
        """-> the gain over each draw of cases, and per case."""
        per = []                                                          # per command: the ratios of each case
        for rows in cam.values():
            c, r = np.array([x["commanded"] for x in rows]), np.array([x["realised"] for x in rows])
            ax = int(np.argmax(np.abs(c[0])))
            ratio, case = r[:, ax] / c[:, ax], np.array([x["case"] for x in rows])
            per.append([ratio[(case == k) & np.isfinite(ratio)] for k in cases])
        draw = np.array([np.mean([np.median(np.concatenate([p[j] for j in d])) for p in per]) for d in draws])
        return draw, np.array([np.mean([np.median(p[j]) for p in per]) for j in range(len(cases))])

    def rates(ecs: list) -> tuple[dict, np.ndarray]:
        """-> each rate over each draw of cases, and the hallucination rate per case (NaN without detections)."""
        n = np.zeros((len(cases), 5))
        at = {k: j for j, k in enumerate(cases)}
        for x in ecs:
            n[at[x["case"]]] += [x["tp"], x["gen"], x["real"], x["halluc"], x["real_halluc"]]
        tp, gen, real, hal, real_hal = n[draws].sum(1).T
        with np.errstate(invalid="ignore", divide="ignore"):
            return (dict(ecs_p=tp / gen, ecs_r=tp / real, halluc=100 * hal / gen, real_halluc=100 * real_hal / real),
                    100 * n[:, 3] / n[:, 1])

    b_gain, b_gain_case = gains(base_cam)
    b_rate, b_hal_case = rates(base_ecs)
    table = []
    for run, label, *_ in ROWS + ABLATIONS + SCALE:
        cam, ecs = test_rows(run, "camera"), test_rows(run, "ecs")
        if cam is None and ecs is None:
            continue
        row = dict(run=run, label=label, n_cases=len(cases))
        if cam:
            g, g_case = gains(cam)
            row.update(gain=float(g[0]), gain_ci=ci(g))
            if run != "S_arms":
                row.update(d_gain=float(b_gain[0] - g[0]), d_gain_ci=ci(b_gain - g),
                           egswm_higher=int((b_gain_case > g_case).sum()),
                           egswm_lower=int((b_gain_case < g_case).sum()))
        if ecs:
            rate, hal_case = rates(ecs)
            for k, x in rate.items():
                row.update({k: float(x[0]), f"{k}_ci": ci(x)})
            if run != "S_arms":
                d, ok = b_rate["halluc"] - rate["halluc"], np.isfinite(b_hal_case) & np.isfinite(hal_case)
                row.update(d_halluc=float(d[0]), d_halluc_ci=ci(d), n_halluc=int(ok.sum()),
                           egswm_fewer=int((b_hal_case[ok] < hal_case[ok]).sum()),
                           egswm_more=int((b_hal_case[ok] > hal_case[ok]).sum()))
        summary = load(run)                               # the same windows and noise: the summaries must reproduce
        for k, s, scale in (("gain", "cam_gain_mean", 1), ("halluc", "ecs_hallucination_rate", 100)):
            if k in row and s in summary and abs(row[k] - scale * summary[s]) > 1e-6 * scale:
                print(f"### {run}: {k} {row[k]:.4f} from the rows, {scale * summary[s]:.4f} in the summary")
        table.append(row)

    def vs(r: dict, k: str, n: str, of: str, tex_: bool = False) -> str:
        if n not in r:
            return "--" if tex_ else "–"
        return f"{cell(r, k, '{:.2f}' if k == 'd_gain' else '{:.1f}', tex=tex_)} ({r[n]}/{r[of]})"

    cols = [("gain", "Cam. gain ↑", "{:.2f}"), ("ecs_p", "ECS-P ↑", "{:.2f}"), ("ecs_r", "ECS-R ↑", "{:.2f}"),
            ("halluc", "Halluc. % ↓", "{:.1f}")]
    lines = ["| Model | " + " | ".join(h for _, h, _ in cols) + " | EG-SWM minus model: gain (cases higher) "
             "| halluc. (cases fewer) |", "|" + "---|" * (len(cols) + 3)]
    lines += [f"| {r['label']} | " + " | ".join(cell(r, k, f) for k, _, f in cols)
              + f" | {vs(r, 'd_gain', 'egswm_higher', 'n_cases')} | {vs(r, 'd_halluc', 'egswm_fewer', 'n_halluc')} |"
              for r in table]
    rows_tex = [tex(r["label"]) + " & " + " & ".join(cell(r, k, f, tex=True) for k, _, f in cols)
                + f" & {vs(r, 'd_gain', 'egswm_higher', 'n_cases', True)}"
                + f" & {vs(r, 'd_halluc', 'egswm_fewer', 'n_halluc', True)}" + r" \\" for r in table]
    head = ("Model & " + " & ".join(tex(h).replace("↑", r"$\uparrow$").replace("↓", r"$\downarrow$")
                                    for _, h, _ in cols) + r" & $\Delta$gain & $\Delta$halluc. \\")
    write("e1_control", dict(cases=cases, rows=table), lines,
          tabular("l" + "c" * (len(cols) + 2), [head], rows_tex))


def detector() -> None:
    """EG-SWM with a weaker instrument detector at test time: a share of the detections removed at random, so that
    the arm actions of the steps that lose their region are zero. Mean over cases with intervals, and the change
    against all detections on the same windows, paired over cases."""
    base = OUTPUTS / "eval" / "S_arms" / "replay.npz"
    if not base.exists():
        return
    b = np.load(base)
    cases = sorted(set(b["case"].tolist()))
    pick = np.random.default_rng(0).integers(0, len(cases), (2000, len(cases)))

    def per_case(d) -> np.ndarray:
        x = d["pred_psnr"].mean(1)
        return np.array([x[d["case"] == c].mean() for c in cases])

    rows = []
    for drop in ("0", *DROPS):
        npz = base if drop == "0" else base.with_name(f"replay_drop{drop}.npz")
        if not npz.exists() or not np.array_equal(np.load(npz)["window"], b["window"]):
            continue
        row = dict(drop=float(drop), kept=f"{100 * (1 - float(drop)):.0f} %", **by_case(npz, "pred"),
                   delta_psnr=json.loads(npz.with_suffix(".json").read_text())["delta_psnr"])
        if drop != "0":
            diff = per_case(np.load(npz)) - per_case(b)
            row.update(d_psnr=float(diff.mean()), d_psnr_ci=np.quantile(diff[pick].mean(1), [0.025, 0.975]).tolist(),
                       cases_worse=f"{int((diff < 0).sum())}/{len(cases)}")
        rows.append(row)
    if len(rows) < 2:
        return
    cols = [("pred_psnr", "PSNR ↑", "{:.2f}"), ("pred_ssim", "SSIM ↑", "{:.3f}"), ("pred_lpips", "LPIPS ↓", "{:.3f}"),
            ("delta_psnr", "ΔPSNR ↑", "{:.2f}"), ("d_psnr", "PSNR change", "{:.2f}")]
    lines = ["| Detections kept | " + " | ".join(h for _, h, _ in cols) + " | Cases worse |",
             "|" + "---|" * (len(cols) + 2)]
    lines += [f"| {r['kept']} | " + " | ".join(cell(r, k, f) for k, _, f in cols) + f" | {r.get('cases_worse', '–')} |"
              for r in rows]
    rows_tex = [tex(r["kept"]) + " & " + " & ".join(cell(r, k, f, tex=True) for k, _, f in cols)
                + f" & {r.get('cases_worse', '--')}" + r" \\" for r in rows]
    head = ("Detections kept & " + " & ".join(tex(h).replace("↑", r"$\uparrow$").replace("↓", r"$\downarrow$")
                                              for _, h, _ in cols) + r" & Cases worse \\")
    write("e6_detector", rows, lines, tabular("l" + "c" * (len(cols) + 1), [head], rows_tex))


def binding() -> None:
    """E1 fidelity by how many of a window's mounted instruments are bound to an arm region (detected and matched
    through the robot log) over the predicted steps; EG-SWM's arm actions only reach bound instruments."""
    sys.path.insert(0, str(ROOT / "scripts" / "eval"))
    from eval_wm import EvalWindows, spread

    ev = EvalWindows("test")
    idx = spread(len(ev), 1000)
    n_bound, seen = np.zeros(len(idx), int), np.zeros(len(idx), bool)
    bound = []                                       # per mounted instrument: share of the window's steps bound
    for j, i in enumerate(idx):
        p, w = ev.index[i]
        c = ev.parts[p]["cond"]
        inst = (c["occupant"][w] >= 2).all(0)                            # instrument arms (not empty, not endoscope)
        bound += np.isfinite(c["boxes"][w][..., 0]).mean(0)[inst].tolist()
        seen[j] = bool(np.isfinite(c["boxes"][w][..., 0]).any(0)[inst].any())   # some instrument detected at all
        occ, box = c["occupant"][w][2:], c["boxes"][w][2:]               # predicted latent steps
        n_bound[j] = int(((occ >= 2).all(0) & (np.isfinite(box[..., 0]).mean(0) >= 0.5)).sum())
    bound = np.array(bound)
    coverage = dict(steps=float(bound.mean()), every=float((bound == 1).mean()),
                    some=float(((bound > 0) & (bound < 1)).mean()), none=float((bound == 0).mean()),
                    instruments=len(bound))
    groups = [("none bound", n_bound == 0), ("one bound", n_bound == 1), ("two or more bound", n_bound >= 2)]
    srcs = [("copy_last", "Copy last context frame", OUTPUTS / "eval" / "reference" / "reference.npz", "copy_last")]
    srcs += [(run, label, OUTPUTS / "eval" / run / "replay.npz", "pred") for run, label, *_ in ROWS + SCALE]
    table = []
    for run, label, npz, prefix in srcs:
        if not npz.exists():
            continue
        d = np.load(npz)
        if not np.array_equal(d["window"], idx):
            continue
        x = d[f"{prefix}_psnr"].mean(1)
        table.append(dict(run=run, label=label, **{g: float(x[m].mean()) for g, m in groups if m.any()}))
    if not table:
        return
    heads = [f"{g} ({int(m.sum())} windows)" for g, m in groups]
    lines = ["| Model | " + " | ".join(heads) + " |", "|" + "---|" * (len(heads) + 1)]
    lines += [f"| {r['label']} | " + " | ".join(cell(r, g, "{:.2f}") for g, _ in groups) + " |" for r in table]
    rows_tex = [tex(r["label"]) + " & " + " & ".join(cell(r, g, "{:.2f}", tex=True) for g, _ in groups) + r" \\"
                for r in table]
    head = [rf" & \multicolumn{{{len(groups)}}}{{c}}{{Instruments bound (windows)}} \\",
            rf"\cmidrule(lr){{2-{len(groups) + 1}}}",
            "Model & " + " & ".join(f"{g.split()[0]} ({int(m.sum())})".replace("two", r"$\geq$2")
                                    for g, m in groups) + r" \\"]
    data = dict(windows={g: int(m.sum()) for g, m in groups}, coverage=coverage, rows=table,
                none_undetected=int((~seen & (n_bound == 0)).sum()))     # no box to hold across detector gaps
    write("e1_binding", data, lines, tabular("l" + "c" * len(groups), head, rows_tex))


def cost() -> None:
    """Parameters, training time and memory (from the training logs) and the time of one 2.8 s prediction
    (outputs/eval/timing.json, written by scripts/docs/make_rollouts.py)."""
    timing = OUTPUTS / "eval" / "timing.json"
    timing = json.loads(timing.read_text()) if timing.exists() else {}
    table = []
    for run, label, *_ in ROWS + ABLATIONS + SCALE:
        log, cfg = OUTPUTS / "runs" / run / "log.jsonl", OUTPUTS / "runs" / run / "config.json"
        if not (log.exists() and cfg.exists() and (OUTPUTS / "eval" / run / "replay.json").exists()):
            continue
        train = json.loads(cfg.read_text())["train"]
        recs = []
        for line in log.read_text(errors="replace").splitlines():
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:                                 # a line cut short by the power cut
                continue
        recs = [r for r in recs if "win_per_s" in r]
        speed = float(np.median([r["win_per_s"] for r in recs]))
        steps = max(r["step"] for r in recs)
        row = dict(run=run, label=label, size=train["size"], steps=steps, train_h=steps * train["bs"] / speed / 3600,
                   win_per_s=speed, train_mem_gb=float(max(r["mem_gb"] for r in recs)))
        if run in timing:
            row.update(parameters=timing[run]["parameters"], gen_s=timing[run]["generate_s"],
                       decode_s=timing[run]["decode_s"], infer_mem_gb=timing[run]["peak_gb"])
        table.append(row)
    if not table:
        return
    cols = [("train_h", "Training (h)", "{:.1f}"), ("win_per_s", "Windows/s", "{:.0f}"),
            ("train_mem_gb", "Training memory (GB)", "{:.1f}"), ("gen_s", "Prediction (s)", "{:.2f}"),
            ("decode_s", "VAE decoding (s)", "{:.2f}"), ("infer_mem_gb", "Inference memory (GB)", "{:.1f}")]
    lines = ["| Model | Size | Steps | " + " | ".join(h for _, h, _ in cols) + " |", "|" + "---|" * (len(cols) + 3)]
    lines += [f"| {r['label']} | {r['size']} | {r['steps']} | " + " | ".join(cell(r, k, f) for k, _, f in cols)
              + " |" for r in table]
    rows_tex = [tex(f"{r['label']} & {r['size']} & {r['steps']} & ")
                + " & ".join(cell(r, k, f, tex=True) for k, _, f in cols) + r" \\" for r in table]
    write("cost", table, lines, tabular("lcc" + "c" * len(cols),
                                        ["Model & Size & Steps & " + " & ".join(h for _, h, _ in cols) + r" \\"],
                                        rows_tex))


def long_rollouts() -> None:
    """H4: 14 s autoregressive rollouts (scripts/eval/eval_long.py): fidelity per 2.8 s chunk, and over the first and
    the last chunk the share of generated instruments that the robot log does not mount and the share of real
    instruments not found in the generated frames (per-second rates averaged; for the copied frame, the detector on
    real frames)."""
    srcs = [("reference", "Copy last context frame")] + [(r, LABEL[r]) for r in LONG_RUNS]
    table = []
    for run, label in srcs:
        p = OUTPUTS / "eval" / run / "long.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        x, lp_ = np.array(d["psnr_by_frame"]), np.array(d["lpips_by_frame"])
        row = dict(run=run, label=label, n_clips=d["n_clips"])
        for c in range(0, len(x) // 28):
            row[f"psnr_c{c}"] = float(x[28 * c:28 * c + 28].mean())
            row[f"lpips_c{c}"] = float(lp_[28 * c:28 * c + 28].mean())
        sec = np.array(d["detector_seconds"])
        for name, key in (("halluc", "halluc_rate_by_second"), ("missing", "missing_rate_by_second")):
            rate = np.array(d[key])
            row[f"{name}_first"] = float(rate[sec < 2.8].mean())
            row[f"{name}_last"] = float(rate[sec >= 2.8 * (len(x) // 28 - 1)].mean())
        table.append(row)
    if not table or len(table) == 1 and table[0]["run"] == "reference":
        return
    n = max(int(k[6:]) for k in table[0] if k.startswith("psnr_c")) + 1
    cols = [(f"psnr_c{c}", f"PSNR {2.8 * c:.1f}–{2.8 * (c + 1):.1f} s", "{:.2f}") for c in range(n)]
    cols += [(f"lpips_c{n - 1}", f"LPIPS {2.8 * (n - 1):.1f}–{2.8 * n:.1f} s", "{:.3f}"),
             ("halluc_first", "Unmounted 0–2.8 s", "{:.3f}"),
             ("halluc_last", f"Unmounted {2.8 * (n - 1):.1f}–{2.8 * n:.1f} s", "{:.3f}"),
             ("missing_last", f"Missing {2.8 * (n - 1):.1f}–{2.8 * n:.1f} s", "{:.3f}")]
    lines = ["| Model | " + " | ".join(h for _, h, _ in cols) + " |", "|" + "---|" * (len(cols) + 1)]
    lines += [f"| {r['label']} | " + " | ".join(cell(r, k, f) for k, _, f in cols) + " |" for r in table]
    rows_tex = [tex(r["label"]) + " & " + " & ".join(cell(r, k, f, tex=True) for k, _, f in cols) + r" \\"
                for r in table]
    head = "Model & " + " & ".join(h.replace("–", "--") for _, h, _ in cols) + r" \\"
    write("e1_long", table, lines, tabular("l" + "c" * len(cols), [head], rows_tex))


def scene_reveal() -> None:
    """Where a learned action's effect lands relative to the content the moving endoscope reveals
    (scripts/eval/test_scene_reveal.py): share of the change in revealed grid cells over their area share."""
    p = OUTPUTS / "eval" / "scene_reveal.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    table = [dict(label=k, **v) for k, v in d["tests"].items()]
    cols = [("n", "Windows", "{:d}"), ("ratio_median", "Revealed-cell ratio", "{:.2f}"),
            ("ratio_median_small_motion", "smaller motion", "{:.2f}"),
            ("ratio_median_large_motion", "larger motion", "{:.2f}")]
    lines = ["| Action replaced | " + " | ".join(h for _, h, _ in cols) + " |", "|" + "---|" * (len(cols) + 1)]
    lines += [f"| {r['label']} | " + " | ".join(cell(r, k, f) for k, _, f in cols) + " |" for r in table]
    rows_tex = [tex(r["label"]) + " & " + " & ".join(cell(r, k, f, tex=True) for k, _, f in cols) + r" \\"
                for r in table]
    write("e1_scene_reveal", d, lines,
          tabular("l" + "c" * len(cols), ["Action replaced & " + " & ".join(h for _, h, _ in cols) + r" \\"],
                  rows_tex))


def e2() -> None:
    """Kinematic grounding: R^2 of the linear probes on held-out episodes per target, averaged over the two arms
    (per arm in the JSON), for model-free features, frozen SurgVU models and models adapted without kinematics."""
    probes = {}
    for d, _ in E2_DATA:
        for src in ("zeroshot", "allarms", "adapted"):
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
    pos, rows_pos, last = [f"{d}_pos" for d, _ in E2_DATA], [], None    # the main paper's table: translation only
    for row, f in zip(table, flat, strict=True):
        if last not in (None, row["setting"]):
            rows_pos.append(r"\midrule")
        last = row["setting"]
        rows_pos.append(f"{row['label']} & " + " & ".join(cell(f, k, "{:.2f}", tex=True) for k in pos) + r" \\")
    (ROOT / "paper" / "tmrb" / "tables" / "e2_probes_pos.tex").write_text(
        tabular("l" + "c" * len(pos), ["Features & " + " & ".join(n for _, n in E2_DATA) + r" \\"], rows_pos))


def external(run: str, data: str) -> dict:
    """eval_wm.py --data <data> results of a run, with intervals over test episodes."""
    p = OUTPUTS / "eval" / run / data / "replay.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    return dict({k: d[k] for k, *_ in COLS[:5] if k in d}, step=d.get("step"), weights=d.get("weights", "ema"),
                **by_case(p.with_suffix(".npz"), "pred"))


def grouped(name: str, groups: list[tuple[str, list[dict]]], first: str, extra: str) -> None:
    """A table of reference and model rows in groups (datasets or settings) separated by rules in LaTeX; `extra`
    is a per-row column (key of the row) named `first`."""
    cols = COLS[:5]                                                     # PSNR, SSIM, LPIPS, FVD, ΔPSNR
    lines = [f"| {first} | Model | " + " | ".join(h + (" ↑" if hi else " ↓") for _, h, _, hi in cols) + " |",
             "|" + "---|" * (len(cols) + 2)]
    rows_tex = []
    for g, rows in groups:
        if rows_tex:
            rows_tex.append(r"\midrule")
        for row in rows:
            lines.append(f"| {row.get(extra, g)} | {row['label']} | "
                         + " | ".join(cell(row, k, fmt) for k, _, fmt, _ in cols) + " |")
            rows_tex.append(tex(f"{row.get(extra, g)} & {row['label']} & ").replace("–", "--")
                            + " & ".join(cell(row, k, fmt, tex=True) for k, _, fmt, _ in cols) + r" \\")
    head = " & ".join(tex(h) + (r"$\uparrow$" if hi else r"$\downarrow$") for _, h, _, hi in cols)
    write(name, [r for _, rows in groups for r in rows], lines,
          tabular("ll" + "c" * len(cols), [f"{first} & Model & " + head + r" \\"], rows_tex))


def e3() -> None:
    """Kinematics-conditioned prediction on the SutureBot and SRT-H test episodes by initialisation and by the share
    of the training episodes with kinematics."""
    groups = []
    for data, prefix, dname in E3_DATA:
        rows = []
        for init, label in E3_INITS:
            for f in E3_FRACTIONS:
                d = external(f"{prefix}_kin_{init}_f{f}", data)
                if d:
                    rows.append(dict(run=f"{prefix}_kin_{init}_f{f}", data=data, init=init, f=f, label=label,
                                     fraction=float(f), share=f"{dname}, {float(f):.0%}", **d))
        if len({r["weights"] for r in rows}) > 1:   # E3 is scored on the final weights (8k steps: the average lags)
            print(f"### E3 on {dname} mixes weight-average and final-weight results")
        ev = OUTPUTS / "eval"                       # paired on the same test windows, over test episodes
        for r in rows:
            npz = ev / r["run"] / data / "replay.npz"
            others = {"arms": ("scratch", "global"), "global": ("scratch",)}.get(r["init"], ())
            for other in others:
                pr = paired_psnr(npz, ev / f"{prefix}_kin_{other}_f{r['f']}" / data / "replay.npz")
                r.update({f"vs_{other}_{k}": v for k, v in pr.items()})
            pr = paired_psnr(npz, ev / "reference" / data / "reference.npz", pb="copy_last")
            r.update({f"vs_copy_{k}": v for k, v in pr.items()})
            if r["f"] != "0.1":                     # more training episodes, the same initialisation and steps
                pr = paired_psnr(npz, ev / f"{prefix}_kin_{r['init']}_f0.1" / data / "replay.npz")
                r.update({f"vs_tenth_{k}": v for k, v in pr.items()})
            elif r["init"] != "scratch":            # a tenth of the episodes against all of them from scratch
                pr = paired_psnr(npz, ev / f"{prefix}_kin_scratch_f1.0" / data / "replay.npz")
                r.update({f"vs_scratch_all_{k}": v for k, v in pr.items()})
        if rows:
            ref = OUTPUTS / "eval" / "reference" / data / "reference.json"
            refs = [dict(r, data=data, share=f"{dname}, –") for r in (reference(ref) if ref.exists() else [])]
            groups.append((dname, refs + rows))
    if groups:
        grouped("e3_kinematics", groups, "Data, training episodes", "share")


def e4() -> None:
    """Transfer to clinical surgery (SAR-RARP50 test videos): SurgVU models zero-shot, after action-free adaptation
    to the SAR-RARP50 training videos, and EG-SWM trained on them alone with the same budget."""
    groups = {}
    for run, label, setting in E4_ROWS:
        d = external(run, "sar_rarp50")
        if d:
            groups.setdefault(setting, []).append(dict(run=run, label=label, setting=setting, **d))
    if not groups:
        return
    ref = OUTPUTS / "eval" / "reference" / "sar_rarp50" / "reference.json"
    refs = [dict(r, setting="–") for r in (reference(ref) if ref.exists() else [])]
    grouped("e4_clinical", ([("reference", refs)] if refs else []) + list(groups.items()), "Setting", "setting")


def numbers() -> None:
    """Every number the manuscript's text quotes, as \\val{key} (paper/tmrb/tables/numbers.tex, read by main.tex) and in
    docs/results/numbers.json: main/<run>/<metric>[_ci], ctl/<run>/... (control and embodiment), diff/<a>-<b>/...
    (e1_contrasts), use/<run>/... (action-use diagnostics on the validation windows: replay error and its rise in %
    when the learned action (swap), the endoscope action (cam_swap) or both are replaced, inside and outside the arm
    regions, and at noise level 0.9), diff/S_arms-<run>/locality[_ci, _wins, _losses] (e1_locality),
    ctl/<run>/{gain,ecs_p,ecs_r,halluc,real_halluc}_ci and diff/S_arms-<run>/{gain,halluc}[_ci, _wins, _of]
    (e1_control), bind/... (e1_binding), cost/<size>/... and horizon/... (EG-SWM minus the copied frame)."""
    res, val = ROOT / "docs" / "results", {}

    def f(x: float, nd: int) -> str:
        return f"{x:.{nd}f}"

    def pct(x: float) -> str:                                  # a percentage: no decimals from 10 % up
        return f"{x:.0f}" if abs(x) >= 9.95 else f"{x:.1f}"

    def put(key: str, x: float, nd: int, ci: list | None = None) -> None:
        val[key] = f(x, nd)
        if ci is not None:
            val[f"{key}_ci"] = f"[{f(ci[0], nd)}, {f(ci[1], nd)}]"

    rows = [r for name in ("e1_main", "e6_ablations") if (res / f"{name}.json").exists()
            for r in json.loads((res / f"{name}.json").read_text())]
    for r in rows:
        for k, nd in (("pred_psnr", 2), ("pred_ssim", 3), ("pred_lpips", 3), ("pred_fvd", 0), ("delta_psnr", 2)):
            if isinstance(r.get(k), (int, float)):
                put(f"main/{r['run']}/{k.removeprefix('pred_')}", r[k], nd, r.get(f"{k}_ci"))
    for run in sorted({r for r, *_ in ROWS + ABLATIONS + SCALE} | {"reference"}):
        d = load(run)
        gains = [v for k, v in d.items() if k.startswith("cam_") and k.endswith("_gain")]
        if gains:
            put(f"ctl/{run}/gain", float(np.mean(gains)), 2)
            val[f"ctl/{run}/gain_pct"] = f"{100 * np.mean(gains):.0f}"
            val[f"ctl/{run}/gain_range"] = f"{100 * min(gains):.0f}--{100 * max(gains):.0f}"
            val[f"ctl/{run}/sign_min"] = f(100 * min(v for k, v in d.items() if k.endswith("_sign_acc")
                                                 and k != "cam_sign_acc_mean"), 1)
        if "locality_ratio_median" in d:
            put(f"ctl/{run}/locality", d["locality_ratio_median"], 2)
        for k in ("locality_share_mean", "locality_area_mean"):
            if k in d:
                val[f"ctl/{run}/{k.split('_')[1]}"] = pct(100 * d[k])
        for k, key in (("ecs_precision", "ecs_p"), ("ecs_recall", "ecs_r")):
            if k in d:
                put(f"ctl/{run}/{key}", d[k], 2)
        for k, key in (("ecs_hallucination_rate", "halluc"), ("ecs_real_hallucination_rate", "real_halluc")):
            if k in d:
                put(f"ctl/{run}/{key}", 100 * d[k], 1)
        step = d.get("step")
        p = OUTPUTS / "eval" / "action_use" / f"{run}_{step}.json"
        if not p.exists():
            continue
        u = json.loads(p.read_text())
        rp, gap = u["replay"], u["replay_gap_pct"]
        put(f"use/{run}/replay", rp["real"]["all"], 3)
        for name, key in (("shuffled", "swap"), ("camera_shuffled", "cam_swap"), ("zero", "zeroed"),
                          ("zero_camera_shuffled", "cam_swap_zeroed")):
            if name not in gap:
                continue
            val[f"use/{run}/{key}"] = pct(gap[name])
            for side, s in (("inside", "in"), ("outside", "out")):
                val[f"use/{run}/{key}_{s}"] = pct(100 * (rp[name][side] / rp["real"][side] - 1))
                if f"gap_{name}_{side}_pct" in u and u["taus"][-1] == 0.9:
                    val[f"use/{run}/{key}_t9_{s}"] = pct(u[f"gap_{name}_{side}_pct"][-1])
        if u.get("posterior", {}).get("kl_per_arm") is not None:
            put(f"use/{run}/kl", u["posterior"]["kl_per_arm"], 1)
            val[f"use/{run}/active"] = str(u["posterior"]["active_units"])      # of the 8 dimensions of an arm action
    for run in ("S_arms",):
        for name in ("replay", "control"):
            p = OUTPUTS / "eval" / run / f"{name}.json"
            if p.exists():
                n = json.loads(p.read_text())["n_windows"]
                val[f"{'main' if name == 'replay' else 'ctl'}/n_windows"] = f"{n:,}"
    if (res / "e1_contrasts.json").exists():
        for r in json.loads((res / "e1_contrasts.json").read_text()):
            k = f"diff/{r['a']}-{r['b']}"
            for m, nd in (("psnr", 2), ("ssim", 3), ("lpips", 3)):
                put(f"{k}/{m}", r[f"d_{m}"], nd, r[f"d_{m}_ci"])
            val[f"{k}/wins"], val[f"{k}/losses"] = str(r["a_better_psnr"]), str(r["n_cases"] - r["a_better_psnr"])
            val["main/n_cases"] = str(r["n_cases"])
    if (res / "e1_locality.json").exists():                    # interval of each median; EG-SWM minus each run, paired
        for r in json.loads((res / "e1_locality.json").read_text()):
            put(f"ctl/{r['run']}/locality", r["locality"], 2, r["locality_ci"])
            if "d_locality" in r:
                k = f"diff/S_arms-{r['run']}/locality"
                put(k, -r["d_locality"], 2, [-x for x in r["d_locality_ci"][::-1]])
                val[f"{k}_wins"], val[f"{k}_losses"] = str(r["egswm_better"]), str(r["n_cases"] - r["egswm_better"])
    if (res / "e1_control.json").exists():                     # intervals of gain and ECS; EG-SWM minus each run
        for r in json.loads((res / "e1_control.json").read_text())["rows"]:
            run = r["run"]
            if "gain" in r:
                put(f"ctl/{run}/gain", r["gain"], 2, r["gain_ci"])
                val[f"ctl/{run}/gain_pct_ci"] = "[{:.0f}, {:.0f}]".format(*(100 * x for x in r["gain_ci"]))
            for k, nd in (("ecs_p", 2), ("ecs_r", 2), ("halluc", 1), ("real_halluc", 1)):
                if k in r:
                    put(f"ctl/{run}/{k}", r[k], nd, r[f"{k}_ci"])
            for k, n, m, of, nd in (("gain", "egswm_higher", "egswm_lower", "n_cases", 2),
                                    ("halluc", "egswm_fewer", "egswm_more", "n_halluc", 1)):
                if f"d_{k}" in r:
                    key = f"diff/S_arms-{run}/{k}"
                    put(key, r[f"d_{k}"], nd, r[f"d_{k}_ci"])
                    val[f"{key}_wins"], val[f"{key}_losses"], val[f"{key}_of"] = str(r[n]), str(r[m]), str(r[of])
    if (res / "e1_sampler.json").exists():                     # Euler steps and guidance on 200 test windows
        at = {(r["run"], r["steps"], r["guidance"]): r for r in json.loads((res / "e1_sampler.json").read_text())}
        runs = sorted({run for run, *_ in at})
        for run in runs:
            if (run, 50, 1.0) not in at:
                continue
            b = at[(run, 50, 1.0)]
            put(f"sens/{run}/delta_g1", b["delta_psnr"], 2)
            if (run, 20, 1.0) in at:                          # what 20 steps lose, over the horizon and at 2.8 s
                put(f"sens/{run}/s20_loss", b["pred_psnr"] - at[(run, 20, 1.0)]["pred_psnr"], 2)
                put(f"sens/{run}/end20_loss", b["psnr_end"] - at[(run, 20, 1.0)]["psnr_end"], 2)
            if (run, 50, 2.0) in at:
                put(f"sens/{run}/delta_g2", at[(run, 50, 2.0)]["delta_psnr"], 2)
        inter = [r for r in SENS_RUNS if r.startswith("S_") and all((r, *s) in at for s in ((50, 1.0), (100, 1.0),
                                                                                        (20, 1.0), (50, 2.0)))]
        if inter:                                                   # across the size-S interfaces
            val["sens/s100_max"] = f(max(abs(at[(r, 100, 1.0)]["pred_psnr"] - at[(r, 50, 1.0)]["pred_psnr"])
                                         for r in inter), 2)
            g2 = [at[(r, 50, 2.0)]["pred_psnr"] - at[(r, 50, 1.0)]["pred_psnr"] for r in inter]
            val["sens/g2_range"] = f"{f(min(g2), 2)}--{f(max(g2), 2)}"
            orders = {tuple(sorted(inter, key=lambda r, s=s: -at[(r, *s)]["pred_psnr"]))
                      for s in ((20, 1.0), (50, 1.0), (100, 1.0), (50, 2.0))}
            if len(orders) > 1:
                print("### the order of the interfaces depends on the sampler (text: it does not)")
    if (res / "e6_detector.json").exists():                    # EG-SWM with instrument detections removed at test time
        for r in json.loads((res / "e6_detector.json").read_text()):
            if "d_psnr" in r:
                k = f"det/{100 - 100 * r['drop']:.0f}"                # share of the detections kept
                put(f"{k}/psnr", r["d_psnr"], 2, r["d_psnr_ci"])
                put(f"{k}/delta_psnr", r["delta_psnr"], 2)
                val[f"{k}/worse"], val[f"{k}/of"] = r["cases_worse"].split("/")
    if (res / "e1_scene_reveal.json").exists():                # change in the content the moving endoscope reveals
        d = json.loads((res / "e1_scene_reveal.json").read_text())
        for t in d["tests"].values():
            put(f"reveal/{t['run']}/{t['action']}", t["ratio_median"], 2)
            val["reveal/n"] = str(t["n"])
        val["reveal/min_area"] = f"{100 * d['min_area']:.0f}"
    if (res / "e1_long.json").exists():                        # 14 s rollouts: PSNR per chunk, instruments early, late
        for r in json.loads((res / "e1_long.json").read_text()):
            for k, v in r.items():
                if k.startswith("psnr_c"):
                    put(f"long/{r['run']}/{k}", v, 2)
                elif k.startswith(("halluc_", "missing_")):
                    put(f"long/{r['run']}/{k}", 100 * v, 0)
    if (res / "e1_camera_scale.json").exists():                # endoscope commands at 0.5-2x the protocol's size
        rows = json.loads((res / "e1_camera_scale.json").read_text())
        for run in sorted({r["run"] for r in rows}):
            mine = [r for r in rows if r["run"] == run]
            for r in mine:
                put(f"scale/{run}/{r['scale']:g}", r["cam_gain_mean"], 2)
            sign = [v for r in mine for k, v in r.items() if k.endswith("_sign_acc")]
            val[f"scale/{run}/sign_min"] = f(100 * min(sign), 1)
            for name, keys in (("zoom", ("zoom_in", "zoom_out")), ("pantilt", ("pan_left", "pan_right", "tilt_up",
                                                                                "tilt_down"))):
                g = [r[f"cam_{k}_gain"] for r in mine for k in keys]
                val[f"scale/{run}/{name}_range"] = f"{100 * min(g):.0f}--{100 * max(g):.0f}"
    if (res / "e1_context.json").exists():                     # 1-3 context latents: PSNR over the frames all predict
        ctx = {r["run"]: r for r in json.loads((res / "e1_context.json").read_text())}
        inter = [r for r in SENS_RUNS if r.startswith("S_") and all(f"ctx{n}" in ctx.get(r, {}) for n in (1, 2, 3))]
        if inter:
            step = [(ctx[r]["ctx3"] - ctx[r]["ctx1"]) / 2 for r in inter]
            val["ctx/per_latent"] = f"{f(min(step), 1)}--{f(max(step), 1)}"
            for other in ("S_global", "S_slots"):
                if "S_arms" in inter and other in inter:
                    for n in (1, 2, 3):
                        put(f"ctx/{other}-S_arms/{n}", ctx[other][f"ctx{n}"] - ctx["S_arms"][f"ctx{n}"], 2)
            if len({tuple(sorted(inter, key=lambda r, n=n: -ctx[r][f"ctx{n}"])) for n in (1, 2, 3)}) > 1:
                print("### the order of the interfaces depends on the context length (text: it does not)")
    if (res / "e2_probes.json").exists():                      # R^2 of the linear probes, mean over the two arms
        rows = json.loads((res / "e2_probes.json").read_text())["rows"]
        for r in rows:
            name = r["probe"].replace("<data>_", "")                  # a run, a model-free feature or adapt_<action>
            got = {d: r[d]["mean"] for d, _ in E2_DATA if d in r}
            for d, m in got.items():
                for t in TARGETS:
                    put(f"kin/{d}/{name}/{t}", m[t], 2)
            if len(got) == len(E2_DATA):
                for t in TARGETS:
                    put(f"kin/mean/{name}/{t}", float(np.mean([m[t] for m in got.values()])), 2)
        # how far any feature gets where the probes find little: translation on CRCD and JIGSAWS, rotation and jaw
        pos = [r[d]["mean"]["pos"] for r in rows for d in ("crcd", "jigsaws") if d in r]
        rot_jaw = [r[d]["mean"][t] for r in rows for d, _ in E2_DATA if d in r for t in ("rot", "jaw")]
        if pos:
            put("kin/max/crcd_jigsaws_pos", max(pos), 2)
        if rot_jaw:
            put("kin/max/rot_jaw", max(rot_jaw), 2)
    if (res / "e3_kinematics.json").exists():                  # kinematics-conditioned prediction, test episodes
        e3_rows = json.loads((res / "e3_kinematics.json").read_text())
        for r in e3_rows:
            if "init" not in r:                                       # copied frame, VAE reconstruction
                put(f"e3/{r['data']}/{r['run']}/psnr", r["pred_psnr"], 2, r.get("pred_psnr_ci"))
                continue
            k = f"e3/{r['data']}/{r['init']}/{r['f']}"
            put(f"{k}/psnr", r["pred_psnr"], 2, r.get("pred_psnr_ci"))
            for other in ("scratch", "global", "copy", "tenth", "scratch_all"):
                if f"vs_{other}_d_psnr" in r:
                    put(f"{k}/vs_{other}", r[f"vs_{other}_d_psnr"], 2, r[f"vs_{other}_d_psnr_ci"])
                    val[f"{k}/vs_{other}_wins"] = str(r[f"vs_{other}_wins"])
                    val[f"{k}/vs_{other}_of"] = str(r[f"vs_{other}_n_cases"])
        # ranges over datasets, shares and initialisations: all episodes against a tenth, an initialised model
        # against training from scratch, and EG-SWM against the monolithic model as the starting point
        for name, key, nd in (("tenth", "vs_tenth_d_psnr", 2), ("init", "vs_scratch_d_psnr", 2)):
            x = [r[key] for r in e3_rows if key in r]
            if x:
                put(f"e3/{name}/min", min(x), nd)
                put(f"e3/{name}/max", max(x), nd)
        iface = [abs(r["vs_global_d_psnr"]) for r in e3_rows if "vs_global_d_psnr" in r]
        if iface:
            put("e3/iface/max_abs", max(iface), 2)
    if (res / "e4_clinical.json").exists():                    # SAR-RARP50 test videos
        for r in json.loads((res / "e4_clinical.json").read_text()):
            if isinstance(r.get("pred_psnr"), float):
                put(f"e4/{r['run']}/psnr", r["pred_psnr"], 2, r.get("pred_psnr_ci"))
    if (res / "e1_binding.json").exists():
        d = json.loads((res / "e1_binding.json").read_text())
        for k, v in d.get("coverage", {}).items():
            if k != "instruments":
                val[f"bind/{k}"] = f"{100 * v:.0f}"
        if "none_undetected" in d:
            val["bind/n/none_undetected"] = str(d["none_undetected"])
        if "none" in d.get("coverage", {}):           # steps with a region when boxes are held across detector gaps
            val["bind/held"] = f"{100 * (1 - d['coverage']['none']):.0f}"
        by_run = {r["run"]: r for r in d["rows"]}
        for g, n in d["windows"].items():
            gk = g.split()[0]                                             # none, one, two (or more)
            val[f"bind/n/{gk}"] = f"{n:,}"
            for run, r in by_run.items():
                if g in r:
                    put(f"bind/{gk}/{run}", r[g], 2)
                    if run != "S_arms" and g in by_run.get("S_arms", {}):
                        put(f"bind/{gk}/{run}-S_arms", r[g] - by_run["S_arms"][g], 2)
    a, c = OUTPUTS / "eval" / "S_arms" / "replay.json", OUTPUTS / "eval" / "reference" / "reference.json"
    if a.exists() and c.exists():
        d = (np.array(json.loads(a.read_text())["pred_psnr_by_frame"])
             - np.array(json.loads(c.read_text())["copy_last_psnr_by_frame"]))
        t = np.arange(1, len(d) + 1) / 10                                 # seconds after the last context frame
        put("horizon/first", d[0], 1)
        val["horizon/trail"] = f(t[d < 0].max(), 1) if (d < 0).any() else "0"
        late = d[t >= 0.5]
        val["horizon/lead"] = f"{f(late.min(), 1)}--{f(late.max(), 1)}"
    if (res / "cost.json").exists():
        cost = json.loads((res / "cost.json").read_text())
        for size in sorted({r["size"] for r in cost}):
            for k, key, nd in (("train_h", "train_h", 1), ("win_per_s", "speed", 0), ("train_mem_gb", "mem", 0)):
                lo, hi = (f(g(r[k] for r in cost if r["size"] == size), nd) for g in (min, max))
                val[f"cost/{size}/{key}"] = lo if lo == hi else f"{lo}--{hi}"
        for r in cost:
            if "gen_s" in r:
                put(f"cost/{r['run']}/gen_s", r["gen_s"], 2)
                put(f"cost/{r['run']}/decode_s", r["decode_s"], 2)
                put(f"cost/{r['run']}/infer_mem", r["infer_mem_gb"], 1)
                put(f"cost/{r['run']}/realtime", 2.8 / (r["gen_s"] + r["decode_s"]), 1)   # predicted seconds per second
    loc = {run: load(run).get("locality_ratio_median") for run in ("S_arms", "S_arms_poolglobal", "S_global")}
    if all(isinstance(v, float) for v in loc.values()):                   # how far ungrounded pooling goes
        way = (loc["S_arms"] - loc["S_arms_poolglobal"]) / (loc["S_arms"] - loc["S_global"])
        val["derived/pool_way"] = pct(100 * way)
    if not val:
        return
    lines = ["% Numbers quoted in the manuscript, generated by scripts/eval/aggregate.py (do not edit): \\val{key}"]
    lines += [rf"\expandafter\def\csname val@{k}\endcsname{{{re.sub(r'(?<![0-9-])-(?=[0-9])', '$-$', v)}}}"
              for k, v in sorted(val.items())]
    (ROOT / "paper" / "tmrb" / "tables" / "numbers.tex").write_text("\n".join(lines) + "\n")
    (res / "numbers.json").write_text(json.dumps(dict(sorted(val.items())), indent=1))
    print(f"{len(val)} numbers -> paper/tmrb/tables/numbers.tex")


def main() -> None:
    e1()
    paired()
    contrasts()
    locality()
    control_intervals()
    binding()
    detector()
    cost()
    e6()
    seeds()
    sensitivity()
    long_rollouts()
    scene_reveal()
    e2()
    e3()
    e4()
    numbers()


if __name__ == "__main__":
    main()
