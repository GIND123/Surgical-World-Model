#!/usr/bin/env python
"""Collect evaluation outputs into result tables for the README (Markdown) and the manuscript (LaTeX).

E1 (action interfaces) and E6 (ablations of EG-SWM) read outputs/eval/<run>/{replay,control}.json and
outputs/eval/reference/{reference,control}.json; PSNR, SSIM and LPIPS are means over test cases of the per-case
means, with 95 % bootstrap intervals over cases (from the per-window arrays in <run>/replay.npz). Seeds: mean and
standard deviation over the runs trained with seeds 0-2. Sensitivity: sampler, context length and endoscope-command
scale on 200 (100) of the test windows. E2 reads the kinematic probes outputs/eval/probes/<data>_{zeroshot,adapted}
.json; E3 and E4 read outputs/eval/<run>/<data>/replay.json, with intervals over test episodes. Each table is
written to docs/results/<name>.{md,json} and paper/tmrb/tables/<name>.tex; missing runs are skipped and a table
without any result is not written.
"""

from __future__ import annotations

import json
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
SEEDS = ("S_arms", "S_global", "S_slots", "S_global_vq")    # also trained with seeds 1 and 2 (<run>_seed<n>)
SENS_RUNS = ("S_arms", "S_global", "S_slots", "S_global_vq", "S_arms_scene")
SAMPLER = (("1.0", "20"), ("1.0", "50"), ("1.0", "100"), ("2.0", "50"))   # guidance scale, Euler steps
CAM_RUNS, CAM_SCALES = ("S_arms", "S_global_vq", "S_state"), ("0.5", "1.0", "2.0")
LONG_RUNS = ("S_arms", "S_global", "S_slots", "S_global_vq", "S_state", "S_none", "S_arms_scene")
COLS = [  # key, header, format, higher-is-better
    ("pred_psnr", "PSNR", "{:.2f}", True), ("pred_ssim", "SSIM", "{:.3f}", True),
    ("pred_lpips", "LPIPS", "{:.3f}", False), ("pred_fvd", "FVD", "{:.0f}", False),
    ("delta_psnr", "ΔPSNR", "{:.2f}", True), ("cam_gain_mean", "Cam. gain", "{:.2f}", True),
    ("locality_ratio_median", "Locality", "{:.2f}", True), ("ecs_precision", "ECS-P", "{:.3f}", True),
    ("ecs_recall", "ECS-R", "{:.3f}", True), ("ecs_hallucination_rate", "Halluc.", "{:.3f}", False),
]
LABEL = {run: label for run, label, *_ in ROWS + ABLATIONS}
E2_DATA = [("suturebot", "SutureBot"), ("crcd", "CRCD"), ("jigsaws", "JIGSAWS")]
E2_ROWS = [  # probe name ({d}: dataset), probe file, label, setting
    ("dz_region", "zeroshot", "Latent change in the arm region", "model-free"),
    ("dz_frame", "zeroshot", "Latent change over the frame", "model-free"),
    ("camera", "zeroshot", "Endoscope action", "model-free"),
    *((run, "zeroshot", label, "zero-shot") for run, label, *_ in ROWS + ABLATIONS),
    *((f"{{d}}_adapt_{a}", "adapted", LABEL[f"S_{a}"], "adapted") for a in ("global", "global_vq", "slots", "arms")),
]
TARGETS = ("pos", "rot", "jaw")                     # end-effector translation, rotation, jaw of the same arm
E3_DATA = [("suturebot", "SB", "SutureBot"), ("srth", "SRTH", "SRT-H")]   # dataset, run prefix, name
E3_INITS = [("scratch", "None (from scratch)"), ("arms", LABEL["S_arms"]), ("global", LABEL["S_global"])]
E3_FRACTIONS = ("0.1", "0.25", "1.0")               # share of the training episodes with kinematics
E4_ROWS = [  # SAR-RARP50: run, label, setting
    *((run, LABEL[run], "zero-shot") for run in ("S_none", "S_global", "S_global_vq", "S_slots", "S_arms",
                                                 "S_arms_scene")),
    *((f"SAR_adapt_{a}", LABEL[f"S_{a}"], "adapted") for a in ("global", "global_vq", "slots", "arms")),
    ("SAR_scratch_arms", LABEL["S_arms"], "from scratch"),
]
TEX = {"Δ": r"$\Delta$", "β": r"$\beta$", "10⁻³": r"$10^{-3}$", "10⁻⁵": r"$10^{-5}$", " %": r"\,\%", "±": r"$\pm$"}


def tex(s: str) -> str:
    for k, v in TEX.items():
        s = s.replace(k, v)
    return s


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
    if any(load(run) for run, *_ in ABLATIONS):
        interface_table("e6_ablations", [r for r in ROWS if r[0] == "S_arms"] + ABLATIONS, [])


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
                                cam_sign_acc_mean=d["cam_sign_acc_mean"],
                                **{k: v for k, v in d.items() if k.startswith("cam_") and k.endswith("_gain")}))
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
    srcs += [(run, label, OUTPUTS / "eval" / run / "replay.npz", "pred") for run, label, *_ in ROWS + ABLATIONS
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


def binding() -> None:
    """E1 fidelity by how many of a window's mounted instruments are bound to an arm region (detected and matched
    through the robot log) over the predicted steps; EG-SWM's arm actions only reach bound instruments."""
    sys.path.insert(0, str(ROOT / "scripts" / "eval"))
    from eval_wm import EvalWindows, spread

    ev = EvalWindows("test")
    idx = spread(len(ev), 1000)
    n_bound = np.zeros(len(idx), int)
    for j, i in enumerate(idx):
        p, w = ev.index[i]
        c = ev.parts[p]["cond"]
        occ, box = c["occupant"][w][2:], c["boxes"][w][2:]               # predicted latent steps
        inst = (occ >= 2).all(0)                                         # instrument arms (not empty, not endoscope)
        n_bound[j] = int((inst & (np.isfinite(box[..., 0]).mean(0) >= 0.5)).sum())
    groups = [("none bound", n_bound == 0), ("one bound", n_bound == 1), ("two or more bound", n_bound >= 2)]
    srcs = [("copy_last", "Copy last context frame", OUTPUTS / "eval" / "reference" / "reference.npz", "copy_last")]
    srcs += [(run, label, OUTPUTS / "eval" / run / "replay.npz", "pred") for run, label, *_ in ROWS]
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
    data = dict(windows={g: int(m.sum()) for g, m in groups}, rows=table)
    write("e1_binding", data, lines, tabular("l" + "c" * len(groups), head, rows_tex))


def cost() -> None:
    """Parameters, training time and memory (from the training logs) and the time of one 2.8 s prediction
    (outputs/eval/timing.json, written by scripts/docs/make_rollouts.py)."""
    timing = OUTPUTS / "eval" / "timing.json"
    timing = json.loads(timing.read_text()) if timing.exists() else {}
    table = []
    for run, label, *_ in ROWS + ABLATIONS:
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
    """H4: 14 s autoregressive rollouts (scripts/eval/eval_long.py): fidelity per 2.8 s chunk and the share of
    generated instruments that the robot log does not mount, early and late."""
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
        if run != "reference":
            row["halluc_first"], row["halluc_last"] = d["halluc_rate_by_second"][0], d["halluc_rate_by_second"][-1]
            first_s, last_s = d["detector_seconds"][0], d["detector_seconds"][-1]
        table.append(row)
    if not table or len(table) == 1 and table[0]["run"] == "reference":
        return
    n = max(int(k[6:]) for k in table[0] if k.startswith("psnr_c")) + 1
    cols = [(f"psnr_c{c}", f"PSNR {2.8 * c:.1f}–{2.8 * (c + 1):.1f} s", "{:.2f}") for c in range(n)]
    cols += [(f"lpips_c{n - 1}", f"LPIPS {2.8 * (n - 1):.1f}–{2.8 * n:.1f} s", "{:.3f}"),
             ("halluc_first", f"Unmounted at {first_s:.1f} s", "{:.3f}"),
             ("halluc_last", f"Unmounted at {last_s:.1f} s", "{:.3f}")]
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


def external(run: str, data: str) -> dict:
    """eval_wm.py --data <data> results of a run, with intervals over test episodes."""
    p = OUTPUTS / "eval" / run / data / "replay.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    return dict({k: d[k] for k, *_ in COLS[:5] if k in d}, step=d.get("step"),
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
                    rows.append(dict(run=f"{prefix}_kin_{init}_f{f}", data=data, label=label, fraction=float(f),
                                     share=f"{dname}, {float(f):.0%}", **d))
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


def main() -> None:
    e1()
    paired()
    binding()
    cost()
    e6()
    seeds()
    sensitivity()
    long_rollouts()
    scene_reveal()
    e2()
    e3()
    e4()


if __name__ == "__main__":
    main()
