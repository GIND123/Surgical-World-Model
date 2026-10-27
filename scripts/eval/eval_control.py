#!/usr/bin/env python
"""E1 controllability and embodiment metrics on the fixed held-out SurgVU windows.

camera   commanded endoscope motions (pan/tilt/zoom, arm actions zeroed); the realised motion of the generated
         video is re-estimated with the same LK+RANSAC estimator -> gain (realised/commanded) and sign accuracy
locality the action of one arm (or the monolithic action / one slot) is replaced by another window's; the share
         of the resulting pixel change inside the arm's region, divided by the region's area share
ecs      the instrument detector on generated vs real future frames: class-set precision/recall per frame,
         hallucinations (classes not mounted per the robot log) and disappearances

    python scripts/eval/eval_control.py --run eg_swm_S --limit 300
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import diffusers
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_wm import EvalWindows, load_model  # noqa: E402

from swm.data import egomotion  # noqa: E402
from swm.data.geometry import CACHE_H, CACHE_W, FIELD_ROWS  # noqa: E402
from swm.data.windows import CACHE, camera_input, camera_scale, latent_stats  # noqa: E402
from swm.models.actions import region_from_boxes  # noqa: E402
from swm.paths import CHECKPOINTS, DETECTOR, OUTPUTS  # noqa: E402

COMMANDS = {"pan_left": (-1, 0, 0), "pan_right": (1, 0, 0), "tilt_up": (0, -1, 0), "tilt_down": (0, 1, 0),
            "zoom_in": (0, 0, 1), "zoom_out": (0, 0, -1)}


def decode(vae, z, mean, std) -> torch.Tensor:
    lat = (z.float() * std + mean).transpose(1, 2).to(torch.bfloat16)
    with torch.no_grad():
        return vae.decode(lat).sample.float().clamp(-1, 1).add(1).div(2).permute(0, 2, 3, 4, 1)   # B T H W 3


def realised_motion(frames: np.ndarray, first: int) -> np.ndarray:
    """(T, H, W, 3) in [0,1] -> mean (dx, dy, log s) over the predicted transitions."""
    g = [cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) for f in frames]
    est = [egomotion.estimate(g[i - 1], g[i])[0] for i in range(first, len(g))]
    est = np.array([e for e in est if np.isfinite(e).all()])
    return est[:, :3].mean(0) if len(est) else np.full(3, np.nan)


def to_detector_canvas(frames: np.ndarray) -> list[np.ndarray]:
    """Model-field frames (H, W, 3) in [0,1] -> 640x512 BGR canvases matching the detector's training geometry."""
    out = []
    for f in frames:
        canvas = np.zeros((CACHE_H, CACHE_W, 3), np.uint8)
        fld = cv2.resize((f * 255).astype(np.uint8), (CACHE_W, FIELD_ROWS[1] - FIELD_ROWS[0]))
        canvas[FIELD_ROWS[0]:FIELD_ROWS[1]] = cv2.cvtColor(fld, cv2.COLOR_RGB2BGR)
        out.append(canvas)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--n-ctx", type=int, default=2)
    ap.add_argument("--magnitude", type=float, default=0.99, help="quantile of real camera motion to command")
    args = ap.parse_args()
    model, meta = load_model(args.run, None)
    cfg = model.cfg
    ds = EvalWindows(args.split)
    real_all = np.load(CACHE / f"eval_frames_{args.split}.npy", mmap_mode="r")
    vae = diffusers.AutoencoderKLWan.from_pretrained(CHECKPOINTS / "vae/wan22/vae", torch_dtype=torch.bfloat16)
    vae = vae.cuda().eval()
    mean, std = (torch.from_numpy(a).cuda()[None, None, :, None, None] for a in latent_stats())
    from ultralytics import YOLO
    det = YOLO(str(DETECTOR))
    stats = json.loads((Path(__file__).resolve().parents[2] / "configs/stats/egomotion.json").read_text())
    q = "abs_q99" if args.magnitude >= 0.99 else "abs_q90"
    mag = np.asarray(stats[q], np.float32)
    first = 4 * (args.n_ctx - 1) + 1
    gen = torch.Generator(device="cuda").manual_seed(0)
    res = {"camera": {k: [] for k in COMMANDS}, "locality": [], "ecs": []}
    n = min(len(ds), args.limit)
    for s in range(0, n, args.bs):
        items = [ds[i] for i in range(s, min(n, s + args.bs))]
        b = {k: torch.stack([it[k] for it in items]).cuda() for k in items[0]}
        b["region"] = region_from_boxes(b["boxes"], 14, 20, dilate=1)
        bsz = len(items)
        real = np.asarray(real_all[s:s + bsz], np.float32) / 255
        with torch.autocast("cuda", dtype=torch.bfloat16):
            acts = model.infer_actions(b["z"], b, stochastic=False)
            base = model.generate(b, args.n_ctx, args.steps, actions=acts, generator=gen)
        base_px = decode(vae, base, mean, std).cpu().numpy()
        # ---- camera compliance (only models with an explicit camera action)
        if cfg.camera:
            zero = {k: (torch.zeros_like(v) if k.startswith("u_") else v) for k, v in acts.items()}
            for name, (sx, sy, sz) in COMMANDS.items():
                raw = np.zeros((9, 16), np.float32)
                cmd = np.array([sx * mag[0], sy * mag[1], sz * mag[2], 0.0], np.float32)
                raw[1:] = np.tile(cmd, 4)
                valid = np.zeros((9, 4), bool)
                valid[1:] = True                                    # a commanded still camera is valid, not unknown
                cam = torch.from_numpy(camera_input(raw, camera_scale(), valid)).cuda()
                bc = dict(b, camera=cam[None].expand(bsz, -1, -1).clone())
                bc["camera"][:, :args.n_ctx] = b["camera"][:, :args.n_ctx]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model.generate(bc, args.n_ctx, args.steps, actions=zero, generator=gen)
                px = decode(vae, out, mean, std).cpu().numpy()
                for i in range(bsz):
                    r = realised_motion(px[i], first)
                    res["camera"][name].append(dict(commanded=cmd[:3].tolist(), realised=r.tolist()))
        # ---- locality: replace one action, measure where the change lands
        alt = {k: v.roll(1, 0) for k, v in acts.items() if k.startswith("u_")}
        boxes = b["boxes"].cpu().numpy()
        targets = []
        if "u_arms" in acts:
            for arm in range(4):
                a2 = dict(acts)
                u = acts["u_arms"].clone()
                u[:, :, arm] = alt["u_arms"][:, :, arm]
                a2["u_arms"] = u
                targets.append((arm, a2))
        elif "u_extra" in acts:
            targets.append((None, dict(acts, u_extra=alt["u_extra"])))
        for arm, a2 in targets:
            g1, g2 = (torch.Generator(device="cuda").manual_seed(s) for _ in range(2))   # same noise for both
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.generate(b, args.n_ctx, args.steps, actions=a2, generator=g1)
                ref = model.generate(b, args.n_ctx, args.steps, actions=acts, generator=g2)
            diff = (decode(vae, out, mean, std) - decode(vae, ref, mean, std)).abs().mean(-1)[:, first:]  # B T H W
            h, w = diff.shape[-2:]
            for i in range(bsz):
                for k in ([arm] if arm is not None else range(4)):
                    bx = boxes[i, args.n_ctx:, k]
                    bx = bx[np.isfinite(bx[:, 0])]
                    if not len(bx) or b["occupant"][i, -1, k] < 2:
                        continue
                    m = np.zeros((h, w), np.float32)
                    for x0, y0, x1, y1 in bx:
                        m[int(y0 * h):int(np.ceil(y1 * h)), int(x0 * w):int(np.ceil(x1 * w))] = 1
                    m = cv2.dilate(m, np.ones((9, 9), np.uint8))
                    dm = diff[i].sum(0).cpu().numpy()
                    share = float((dm * m).sum() / max(dm.sum(), 1e-8))
                    area = float(m.mean())
                    res["locality"].append(dict(arm=int(k), share=share, area=area, ratio=share / max(area, 1e-6),
                                                change=float(dm.mean())))
        # ---- embodiment consistency
        for i in range(bsz):
            kw = dict(imgsz=640, conf=0.3, quantize=16, verbose=False)                   # FP16 inference
            gen_dets = det.predict(to_detector_canvas(base_px[i][first:]), **kw)
            real_dets = det.predict(to_detector_canvas(real[i][first:]), **kw)
            mounted = {int(o) - 2 for o in b["occupant"][i, -1].tolist() if 2 <= o < 16}
            for g, r in zip(gen_dets, real_dets, strict=True):
                gc = {int(c) for c in g.boxes.cls.tolist()}
                rc = {int(c) for c in r.boxes.cls.tolist()}
                res["ecs"].append(dict(tp=len(gc & rc), gen=len(gc), real=len(rc), halluc=len(gc - mounted),
                                       missing=len(rc - gc)))
        print(f"{s + bsz}/{n}", flush=True)
    summary = dict(meta, run=args.run, n_windows=n)
    if cfg.camera:
        for name, rows in res["camera"].items():
            c = np.array([r["commanded"] for r in rows])
            r = np.array([r["realised"] for r in rows])
            ax = int(np.argmax(np.abs(c[0])))
            ok = np.isfinite(r[:, ax])
            summary[f"cam_{name}_gain"] = float(np.median(r[ok, ax] / c[ok, ax]))
            summary[f"cam_{name}_sign_acc"] = float((np.sign(r[ok, ax]) == np.sign(c[ok, ax])).mean())
        summary["cam_sign_acc_mean"] = float(np.mean([summary[f"cam_{k}_sign_acc"] for k in COMMANDS]))
    if res["locality"]:
        loc = res["locality"]
        summary["locality_ratio_median"] = float(np.median([x["ratio"] for x in loc]))
        summary["locality_share_mean"] = float(np.mean([x["share"] for x in loc]))
        summary["locality_area_mean"] = float(np.mean([x["area"] for x in loc]))
    e = res["ecs"]
    tp, ng, nr = sum(x["tp"] for x in e), sum(x["gen"] for x in e), sum(x["real"] for x in e)
    summary.update(ecs_precision=tp / max(ng, 1), ecs_recall=tp / max(nr, 1),
                   ecs_hallucination_rate=sum(x["halluc"] for x in e) / max(ng, 1),
                   ecs_missing_rate=sum(x["missing"] for x in e) / max(nr, 1))
    out = OUTPUTS / "eval" / args.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "control.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
