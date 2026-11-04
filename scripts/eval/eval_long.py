#!/usr/bin/env python
"""H4: long autoregressive rollouts on the SurgVU test cases.

From 5 real frames (0.5 s) the model predicts 2.8 s; the last 5 generated frames are re-encoded with the VAE as
the context of the next 2.8 s, for --chunks chunks (5: 14 s ahead). Actions are replayed: in every chunk the learned
actions are inferred from the real frames on that chunk's latent grid (encoded on the fly, as the training latents
were), the endoscope action is the measured ego-motion of the same frame transitions, and robot state, console
readout and arm boxes come from the nearest latent step of the precomputed windows (at most 2 frames away).
Clips are runs of consecutive usable windows of the test cases (no colour bars), --per-case per case spread evenly,
fixed in configs/eval/surgvu_test_long.json. Per predicted frame: PSNR, SSIM, LPIPS against the real frames; once per
second the instrument detector on generated and real frames (classes not mounted per the robot log; real classes
missing), on clips with a usable log. Reference: the last context frame copied. Writes outputs/eval/<run>/long.json
(+ .npz).

    python scripts/eval/eval_long.py --run S_arms [--chunks 5]
    python scripts/eval/eval_long.py --reference
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import av
import cv2
import diffusers
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_eval_set import EXCLUDE  # noqa: E402
from eval_control import decode, to_detector_canvas  # noqa: E402
from eval_wm import load_model  # noqa: E402

from swm.data.geometry import FIELD_ROWS, MODEL_H, MODEL_W  # noqa: E402
from swm.data.surgvu import CAMERA, DETECTOR_CLASSES, OCC_INDEX, UNKNOWN  # noqa: E402
from swm.data.windows import CACHE, camera_input, camera_scale, latent_stats  # noqa: E402
from swm.metrics.image import LPIPS, psnr, ssim  # noqa: E402
from swm.models.actions import region_from_boxes  # noqa: E402
from swm.paths import CHECKPOINTS, DETECTOR, OUTPUTS, ROOT  # noqa: E402

WIN, STEP, CHUNK = 33, 4, 28                 # frames per window, per latent step, predicted per chunk
CLIPS = ROOT / "configs" / "eval" / "surgvu_test_long.json"


def select_clips(per_case: int, n_frames: int, seed: int = 0) -> list[dict]:
    """Starts of `per_case` runs of consecutive usable windows per test case covering `n_frames` frames."""
    sp = json.loads((ROOT / "configs" / "splits" / "surgvu_v1.json").read_text())
    no_signal = np.load(CACHE / "no_signal.npz")
    n_win = -(-n_frames // WIN)
    pool = defaultdict(list)
    for cond in sorted((CACHE / "conditioning").glob("*.npz")):
        stem = cond.stem
        case = stem.rsplit("_p", 1)[0]
        if case not in sp["test"] or case in EXCLUDE:
            continue
        starts, bars = np.load(CACHE / "latents" / f"{stem}.starts.npy"), no_signal[stem]
        for w in range(len(starts) - n_win + 1):
            if starts[w + n_win - 1] - starts[w] == WIN * (n_win - 1) and not bars[w:w + n_win].any():
                pool[case].append((stem, w))
    rng = np.random.default_rng(seed)
    out = []
    for case in sorted(pool):
        cand = pool[case][::n_win]                                    # clips that do not overlap
        pick = sorted(rng.choice(len(cand), min(per_case, len(cand)), replace=False).tolist())
        out += [dict(case=case, stem=cand[i][0], window=int(cand[i][1])) for i in pick]
    return out


def read_frames(stem: str, start: int, n: int) -> np.ndarray:
    """n real 224x320 endoscope-field frames from the 10 fps cache, cut as in build_eval_set.py."""
    out = []
    with av.open(str(CACHE / "frames10" / f"{stem}.mp4")) as c:
        c.streams.video[0].thread_type = "AUTO"
        for i, fr in enumerate(c.decode(video=0)):
            if i >= start:
                img = fr.to_ndarray(format="rgb24")[FIELD_ROWS[0]:FIELD_ROWS[1]]
                out.append(cv2.resize(img, (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA))
            if len(out) == n:
                break
    return np.stack(out)


class Part:
    """Conditioning of one part, looked up at arbitrary frames of a clip."""

    def __init__(self, stem: str):
        c = np.load(CACHE / "conditioning" / f"{stem}.npz")
        self.occupant, self.boxes = c["occupant"], c["boxes"]
        ui = CACHE / "ui" / f"{stem}.npz"
        u = np.load(ui) if ui.exists() else None
        self.hand = u["hand"] if u is not None else np.zeros_like(self.occupant)
        self.energy = u["energy"] if u is not None else np.zeros_like(self.occupant)
        m = np.load(CACHE / "meta10" / f"{stem}.npz")
        self.ego, self.inlier = m["ego"], m["ego_inlier"]
        self.starts = np.load(CACHE / "latents" / f"{stem}.starts.npy")

    def step(self, w0: int, f: int) -> tuple[int, int]:
        """Window and latent step whose boundary frame is nearest frame f (windows w0, w0+1, ... are consecutive)."""
        w = w0 + int(f - self.starts[w0]) // WIN
        return w, (int(f - self.starts[w]) + 1) // STEP                 # a frame midway goes to the earlier step

    def item(self, w0: int, f0: int, scale: np.ndarray) -> dict:
        """Conditioning of the 9 latent steps of a chunk whose first frame is f0."""
        occ, hand, energy = (np.zeros((9, 4), np.int64) for _ in range(3))
        boxes = np.full((9, 4, 4), np.nan, np.float32)
        raw = np.zeros((9, 16), np.float32)
        for t in range(9):
            f = f0 + STEP * t
            w, k = self.step(w0, f)
            occ[t], hand[t], energy[t] = self.occupant[w, k], self.hand[w, k], self.energy[w, k]
            boxes[t] = self.boxes[w, k]
            if t > 0:
                e = self.ego[f - STEP + 1:f + 1]
                ok = np.isfinite(e).all(1) & (self.inlier[f - STEP + 1:f + 1] > 0.5)
                raw[t] = np.where(ok[:, None], e, 0).ravel()
        return dict(occupant=torch.from_numpy(occ), hand=torch.from_numpy(hand), energy=torch.from_numpy(energy),
                    boxes=torch.from_numpy(boxes),
                    camera=torch.from_numpy(camera_input(raw, scale).astype(np.float32)),
                    window=torch.tensor([0, w0]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--chunks", type=int, default=5)
    ap.add_argument("--per-case", type=int, default=5)
    ap.add_argument("--limit", type=int, help="first clips only (smoke test)")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--tag", default="long")
    args = ap.parse_args()
    assert args.reference != bool(args.run), "give --run or --reference"
    torch.manual_seed(0)
    n_frames = 5 + CHUNK * args.chunks
    if not CLIPS.exists():
        CLIPS.write_text(json.dumps(select_clips(args.per_case, n_frames), indent=0))
    clips = json.loads(CLIPS.read_text())[:args.limit]
    vae = diffusers.AutoencoderKLWan.from_pretrained(CHECKPOINTS / "vae/wan22/vae", torch_dtype=torch.bfloat16)
    vae = vae.cuda().eval()
    mean, std = (torch.from_numpy(a).cuda()[None, None, :, None, None] for a in latent_stats())
    model, meta = (None, {}) if args.reference else load_model(args.run, None)
    lp = LPIPS()
    from ultralytics import YOLO
    det = YOLO(str(DETECTOR))
    kw = dict(imgsz=640, conf=0.3, quantize=16, verbose=False)
    scale = camera_scale()
    parts: dict[str, Part] = {}
    gen = torch.Generator(device="cuda").manual_seed(0)

    def encode(frames: torch.Tensor) -> torch.Tensor:
        """(B, T, H, W, 3) in [0, 1] -> normalised latents (B, T', 48, 14, 20), as encode_latents.py."""
        x = (frames * 2 - 1).permute(0, 4, 1, 2, 3).to(torch.bfloat16)
        with torch.no_grad():
            z = vae.encode(x).latent_dist.mode().transpose(1, 2).float()
        return (z - mean) / std

    per = {m: [] for m in ("psnr", "ssim", "lpips")}
    seconds = list(range(10, n_frames, 10))                         # detector frames: 1.0 s, 2.0 s, ...
    ecs = {k: np.zeros(len(seconds)) for k in ("gen", "halluc", "real", "missing")}
    n_log, t0 = 0, time.time()
    for s in range(0, len(clips), args.bs):
        batch_clips = clips[s:s + args.bs]
        for c in batch_clips:
            parts.setdefault(c["stem"], Part(c["stem"]))
        real = np.stack([read_frames(c["stem"], int(parts[c["stem"]].starts[c["window"]]), n_frames)
                         for c in batch_clips])
        real_t = torch.from_numpy(real).cuda().float() / 255                                    # B T H W 3
        pred = torch.empty_like(real_t)
        pred[:, :5] = real_t[:, :5]
        for ch in range(args.chunks):
            f0 = CHUNK * ch                                                                     # clip frame
            if args.reference:
                pred[:, f0 + 5:f0 + 5 + CHUNK] = real_t[:, 4:5]
                continue
            items = [parts[c["stem"]].item(c["window"], int(parts[c["stem"]].starts[c["window"]]) + f0, scale)
                     for c in batch_clips]
            b = {k: torch.stack([it[k] for it in items]).cuda() for k in items[0]}
            b["region"] = region_from_boxes(b["boxes"], 14, 20, dilate=1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z_real = encode(real_t[:, f0:f0 + WIN])
                ctx = z_real[:, :2] if ch == 0 else encode(pred[:, f0:f0 + 5])
                with torch.no_grad():
                    acts = model.infer_actions(z_real, dict(b, z=z_real), stochastic=False)
                    z = model.generate(dict(b, z=torch.cat([ctx, z_real[:, 2:]], 1)), 2, args.steps, actions=acts,
                                       generator=gen)
            pred[:, f0 + 5:f0 + 5 + CHUNK] = decode(vae, z, mean, std)[:, 5:]
        a = pred[:, 5:].flatten(0, 1).permute(0, 3, 1, 2)
        r = real_t[:, 5:].flatten(0, 1).permute(0, 3, 1, 2)
        bt = (len(batch_clips), n_frames - 5)
        per["psnr"].append(psnr(a, r).view(bt).cpu())
        per["ssim"].append(ssim(a, r).view(bt).cpu())
        per["lpips"].append(lp(a, r).view(bt).cpu())
        pred_np = pred.cpu().numpy()
        for i, c in enumerate(batch_clips):
            p = parts[c["stem"]]
            occ = p.occupant[c["window"], 0].tolist()
            if OCC_INDEX[CAMERA] not in occ or OCC_INDEX[UNKNOWN] in occ:
                continue                                                # no usable robot log
            n_log += 1
            gd = det.predict(to_detector_canvas(pred_np[i][seconds]), **kw)
            rd = det.predict(to_detector_canvas(real[i][seconds].astype(np.float32) / 255), **kw)
            for j, (g, rr) in enumerate(zip(gd, rd, strict=True)):
                w, k = p.step(c["window"], int(p.starts[c["window"]]) + seconds[j])
                mounted = {o - 2 for o in p.occupant[w, k].tolist() if 2 <= o < 2 + len(DETECTOR_CLASSES)}
                gc, rc = {int(x) for x in g.boxes.cls.tolist()}, {int(x) for x in rr.boxes.cls.tolist()}
                ecs["gen"][j] += len(gc)
                ecs["halluc"][j] += len(gc - mounted)
                ecs["real"][j] += len(rc)
                ecs["missing"][j] += len(rc - gc)
        print(f"{s + len(batch_clips)}/{len(clips)} clips, {time.time() - t0:.0f}s", flush=True)
    summary = dict(meta, run="reference" if args.reference else args.run, n_clips=len(clips), chunks=args.chunks,
                   steps=args.steps, horizon_s=(n_frames - 5) / 10, ecs_clips=n_log,
                   detector_seconds=[x / 10 - 0.4 for x in seconds])
    arrays = {"case": np.array([c["case"] for c in clips])}
    for m, v in per.items():
        x = torch.cat(v).numpy()                                        # (clips, frames)
        arrays[m] = x
        summary[f"{m}_by_frame"] = x.mean(0).round(4).tolist()
        summary[f"{m}_by_second"] = [float(x[:, i:i + 10].mean()) for i in range(0, x.shape[1], 10)]
    summary["halluc_rate_by_second"] = (ecs["halluc"] / np.maximum(ecs["gen"], 1)).round(4).tolist()
    summary["missing_rate_by_second"] = (ecs["missing"] / np.maximum(ecs["real"], 1)).round(4).tolist()
    out = OUTPUTS / "eval" / ("reference" if args.reference else args.run)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.tag}.json").write_text(json.dumps(summary, indent=1))
    np.savez_compressed(out / f"{args.tag}.npz", **arrays)
    print(json.dumps({k: v for k, v in summary.items() if "by_frame" not in k}, indent=1))


if __name__ == "__main__":
    main()
