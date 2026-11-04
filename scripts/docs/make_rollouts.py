#!/usr/bin/env python
"""Model rollouts for the README and paper figures, and the time and memory of one prediction.

    python scripts/docs/make_rollouts.py

sample    the case_139 window whose start is nearest the frames of docs/figures/samples/sample.json: EG-SWM sees
          four context latents (1.3 s, the four input frames of the architecture figure, 0.4 s apart) and predicts
          the rest with the actions inferred from the real clip. Input, real and predicted frames at the figure's
          times are written to docs/figures/samples/ (ctx*, fut*, pred*.jpg) and sample.json is updated.
compare   docs/figures/rollouts_compare.png: real future and every interface's prediction (protocol: 2 context
          latents, action replay, 50 Euler steps) on two test windows, chosen as those with two instruments bound to
          arms whose EG-SWM PSNR is nearest the median of such windows, from different cases.
control   docs/figures/rollouts_control.png: EG-SWM on the first of these windows with its own actions, one arm's
          action from another window (with the change it causes), the arm actions zeroed, and the endoscope commanded
          to pan right and zoom in (learned actions zeroed) for EG-SWM, the monolithic latent and EG-SWM + scene.
failures  docs/figures/rollouts_failures.png: the four test windows (from different cases) where EG-SWM's PSNR is
          lowest, real and predicted.
timing    outputs/eval/timing.json: seconds and peak GPU memory for one 2.8 s prediction (batch 1, 50 Euler steps),
          the world model and the VAE decoder separately.
Frames are from the SurgVU dataset (Zia et al., 2025), endoscope field only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import av
import cv2
import diffusers
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from eval_control import command, decode  # noqa: E402
from eval_wm import EvalWindows, load_model, spread  # noqa: E402

from swm.data.geometry import FIELD_ROWS, MODEL_H, MODEL_W  # noqa: E402
from swm.data.windows import CACHE, LatentWindows, latent_stats  # noqa: E402
from swm.models.actions import region_from_boxes  # noqa: E402
from swm.paths import CHECKPOINTS, OUTPUTS, ROOT  # noqa: E402

SAMPLES = ROOT / "docs" / "figures" / "samples"
FIG = ROOT / "docs" / "figures"
INTERFACES = [("S_arms", "EG-SWM"), ("S_global", "Monolithic latent"), ("S_slots", "Slots"),
              ("S_global_vq", "VQ (SurgWM-style)"), ("S_state", "Robot state"), ("S_none", "No actions")]
SHOW = (4, 12, 20, 32)                 # frames shown: last context frame (0.4 s), 1.2 s, 2.0 s, 3.2 s


def batch(items: list[dict]) -> dict:
    b = {k: torch.stack([it[k] for it in items]).cuda() for k in items[0]}
    b["region"] = region_from_boxes(b["boxes"], 14, 20, dilate=1)
    return b


def real_frames(stem: str, start: int) -> np.ndarray:
    """33 real 224x320 endoscope-field frames from the 10 fps cache, as scripts/eval/build_eval_set.py cuts them."""
    out = []
    with av.open(str(CACHE / "frames10" / f"{stem}.mp4")) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i >= start:
                img = fr.to_ndarray(format="rgb24")[FIELD_ROWS[0]:FIELD_ROWS[1]]
                out.append(cv2.resize(img, (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA))
            if len(out) == 33:
                break
    return np.stack(out)


def grid(rows: list[tuple[str, list[np.ndarray]]], cols: list[str], path: Path, scale: float = 1.6) -> None:
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(scale * len(cols), scale * 0.7 * len(rows) + 0.3),
                             squeeze=False)
    for r, (label, imgs) in enumerate(rows):
        for c, img in enumerate(imgs):
            ax = axes[r, c]
            ax.imshow(img, cmap="magma" if img.ndim == 2 else None, vmin=0, vmax=None)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=8)
            if c == 0:
                ax.set_ylabel(label, fontsize=7, rotation=0, ha="right", va="center")
    fig.subplots_adjust(wspace=0.03, hspace=0.05, left=0.2, right=0.995, top=0.94, bottom=0.01)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--only", default="sample,compare,control,failures,timing")
    args = ap.parse_args()
    parts = set(args.only.split(","))
    torch.manual_seed(0)
    vae = diffusers.AutoencoderKLWan.from_pretrained(CHECKPOINTS / "vae/wan22/vae", torch_dtype=torch.bfloat16)
    vae = vae.cuda().eval()
    mean, std = (torch.from_numpy(a).cuda()[None, None, :, None, None] for a in latent_stats())
    models = {run: load_model(run, None)[0] for run, _ in INTERFACES + [("S_arms_scene", "")]}

    def gen(run: str, b: dict, n_ctx: int = 2, actions: dict | None = None, seed: int = 0) -> np.ndarray:
        g = torch.Generator(device="cuda").manual_seed(seed)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            z = models[run].generate(b, n_ctx, args.steps, actions=actions, generator=g)
        return decode(vae, z, mean, std).cpu().numpy()                   # B T H W 3 in [0, 1]

    def acts(run: str, b: dict) -> dict:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return models[run].infer_actions(b["z"], b, stochastic=False)

    if "sample" in parts:
        meta = json.loads((SAMPLES / "sample.json").read_text())
        stem = f"{meta['case']}_p{int(meta['part']):03d}"
        ds = LatentWindows("test", stems=[stem])
        starts = np.load(CACHE / "latents" / f"{stem}.starts.npy")
        w = int(np.argmin(np.abs(starts / 10.0 - meta["t_local"])))
        k = int(np.flatnonzero(ds.index[:, 1] == w)[0])
        real = real_frames(stem, int(starts[w]))
        pred = gen("S_arms", batch([ds[k]]), n_ctx=4)[0]
        for i, t in enumerate((0, 4, 8, 12)):
            cv2.imwrite(str(SAMPLES / f"ctx{i}.jpg"), cv2.cvtColor(real[t], cv2.COLOR_RGB2BGR))
        for i, t in enumerate((16, 20, 24, 28)):
            cv2.imwrite(str(SAMPLES / f"fut{i}.jpg"), cv2.cvtColor(real[t], cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(SAMPLES / f"pred{i}.jpg"),
                        cv2.cvtColor((pred[t] * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR))
        meta.update(t_local=float(starts[w] / 10.0), window=w, stem=stem,
                    prediction=dict(run="S_arms", n_ctx_latents=4, euler_steps=args.steps,
                                    actions="inferred from the real clip (replay)", frames_shown=[16, 20, 24, 28]))
        meta.pop("egomotion_between_frames", None)                        # was measured on the earlier frames
        (SAMPLES / "sample.json").write_text(json.dumps(meta, indent=1))
        print(f"sample: {stem} window {w} at {starts[w] / 10:.1f} s", flush=True)

    ev = EvalWindows("test")
    real_all = np.load(CACHE / "eval_frames_test.npy", mmap_mode="r")
    rep = np.load(OUTPUTS / "eval" / "S_arms" / "replay.npz")
    idx = spread(len(ev), 1000)                                          # the E1 windows, as in replay.npz
    assert (rep["window"] == idx).all()
    psnr, case = rep["pred_psnr"].mean(1), rep["case"]
    two = np.array([(np.isfinite(ev[int(i)]["boxes"][-1, :, 0]).sum() >= 2).item() for i in idx])
    med = np.median(psnr[two])
    picks = []
    for j in np.argsort(np.abs(psnr - med) + 1e3 * ~two):
        if case[j] not in {case[p] for p in picks}:
            picks.append(j)
        if len(picks) == 2:
            break
    times = [f"{t / 10:.1f} s" for t in SHOW]

    if "compare" in parts:
        rows = []
        for j in picks:
            b = batch([ev[int(idx[j])]])
            real = np.asarray(real_all[idx[j]], np.float32) / 255
            rows.append((f"Real ({case[j]})", [real[t] for t in SHOW]))
            for run, label in INTERFACES:
                rows.append((label, [gen(run, b)[0][t] for t in SHOW]))
        grid(rows, times, FIG / "rollouts_compare.png")

    if "control" in parts:
        j = picks[0]
        arm = int(np.flatnonzero(np.isfinite(ev[int(idx[j])]["boxes"][-1, :, 0].numpy()))[0])
        src = next(int(idx[(j + d) % len(idx)]) for d in range(1, len(idx))     # the same arm bound elsewhere
                   if np.isfinite(ev[int(idx[(j + d) % len(idx)])]["boxes"][-1, arm, 0].item()))
        b = batch([ev[int(idx[j])], ev[src]])
        b1 = {k: v[:1] for k, v in b.items()}
        a = acts("S_arms", b)
        a1 = {k: (v[:1] if v.ndim else v) for k, v in a.items()}             # "kl" is a scalar
        own = gen("S_arms", b1, actions=a1)[0]
        u = a["u_arms"][:1].clone()
        u[:, :, arm] = a["u_arms"][1:, :, arm]
        swap = gen("S_arms", b1, actions=dict(a1, u_arms=u))[0]
        zero = {k: (torch.zeros_like(v) if k.startswith("u_") else v) for k, v in a1.items()}
        rows = [("Real", [np.asarray(real_all[idx[j]], np.float32)[t] / 255 for t in SHOW]),
                ("EG-SWM, own actions", [own[t] for t in SHOW]),
                (f"arm {arm + 1}: another window's action", [swap[t] for t in SHOW]),
                ("change caused", [np.abs(swap[t] - own[t]).mean(-1) for t in SHOW]),
                ("arm actions zeroed", [gen("S_arms", b1, actions=zero)[0][t] for t in SHOW])]
        stats = json.loads((ROOT / "configs" / "stats" / "egomotion.json").read_text())
        mag = np.asarray(stats["abs_q99"], np.float32)
        for run, label in (("S_arms", "EG-SWM"), ("S_global", "Monolithic latent"),
                           ("S_arms_scene", "EG-SWM + scene")):
            ar = acts(run, b1)
            zr = {k: (torch.zeros_like(v) if k.startswith("u_") else v) for k, v in ar.items()}
            for name, verb in (("pan_right", "pan right"), ("zoom_in", "zoom in")):
                bc, _ = command(b1, name, mag, 2)
                rows.append((f"{label}: {verb}", [gen(run, bc, actions=zr)[0][t] for t in SHOW]))
        grid(rows, times, FIG / "rollouts_control.png")

    if "failures" in parts:
        worst = []
        for j in np.argsort(psnr):
            if case[j] not in {case[p] for p in worst}:
                worst.append(j)
            if len(worst) == 4:
                break
        rows = []
        for j in worst:
            pred = gen("S_arms", batch([ev[int(idx[j])]]))[0]
            real = np.asarray(real_all[idx[j]], np.float32) / 255
            rows += [(f"Real ({case[j]})", [real[t] for t in SHOW]),
                     (f"EG-SWM, {psnr[j]:.1f} dB", [pred[t] for t in SHOW])]
        grid(rows, times, FIG / "rollouts_failures.png")
        (OUTPUTS / "eval" / "figure_windows.json").write_text(json.dumps(
            {"typical": [dict(case=str(case[j]), window=int(idx[j]), psnr=float(psnr[j])) for j in picks],
             "failures": [dict(case=str(case[j]), window=int(idx[j]), psnr=float(psnr[j])) for j in worst]}, indent=1))

    if "timing" in parts:
        b = batch([ev[int(idx[0])]])
        out = {}
        for run in ("S_arms", "S_global"):
            ts, td = [], []
            for rep_i in range(6):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    z = models[run].generate(b, 2, args.steps)
                torch.cuda.synchronize()
                t1 = time.time()
                decode(vae, z, mean, std)
                torch.cuda.synchronize()
                if rep_i:                                                      # the first run warms up
                    ts.append(t1 - t0)
                    td.append(time.time() - t1)
            out[run] = dict(generate_s=float(np.median(ts)), decode_s=float(np.median(td)), euler_steps=args.steps,
                            peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
                            parameters=sum(p.numel() for p in models[run].parameters()))
        (OUTPUTS / "eval" / "timing.json").write_text(json.dumps(out, indent=1))
        print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
