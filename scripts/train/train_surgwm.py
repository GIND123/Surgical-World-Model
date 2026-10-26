#!/usr/bin/env python
"""Train the SurgWM re-implementation (swm.baselines.surgwm) in three stages on the SurgVU windows.

    python scripts/train/train_surgwm.py --stage tokenizer --steps 60000
    python scripts/train/train_surgwm.py --stage lam --steps 60000
    python scripts/train/train_surgwm.py --stage dynamics --steps 60000

Frames are decoded on the fly from the 10 fps cache (same windows as EG-SWM) and resized to 120x180
(40x60 for the latent action model). Outputs: outputs/runs/surgwm_<stage>/.
"""

from __future__ import annotations

import argparse
import json
import time

import av
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from swm.baselines.surgwm import LatentActionModel, MaskGITDynamics, SurgWMConfig, VideoVQVAE
from swm.data.geometry import FIELD_ROWS
from swm.paths import OUTPUTS, PROCESSED, ROOT

CACHE = PROCESSED / "surgvu"


class PixelWindows(Dataset):
    def __init__(self, split: str, n_frames: int = 33, size=(120, 180), max_windows: int | None = None, seed=0):
        sp = json.loads((ROOT / "configs/splits/surgvu_v1.json").read_text())
        rows = []
        for f in sorted((CACHE / "latents").glob("*.starts.npy")):
            stem = f.name.replace(".starts.npy", "")
            if stem.rsplit("_p", 1)[0] in sp[split]:
                rows += [(stem, int(s)) for s in np.load(f)]
        rng = np.random.default_rng(seed)
        if max_windows and len(rows) > max_windows:
            rows = [rows[i] for i in sorted(rng.choice(len(rows), max_windows, replace=False))]
        self.rows, self.n, self.size = rows, n_frames, size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> torch.Tensor:
        stem, s = self.rows[i]
        h, w = self.size
        out = []
        with av.open(str(CACHE / "frames10" / f"{stem}.mp4")) as c:
            st = c.streams.video[0]
            c.seek(int((s / 10.0) / st.time_base), stream=st, backward=True)
            for f in c.decode(st):
                idx = int(round(float(f.time) * 10))
                if idx < s:
                    continue
                img = f.to_ndarray(format="rgb24")[FIELD_ROWS[0]:FIELD_ROWS[1]]
                out.append(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA))
                if len(out) == self.n:
                    break
        while len(out) < self.n:                      # should not happen; keep shapes fixed
            out.append(out[-1])
        x = torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).float() / 127.5 - 1
        return x                                      # (T, 3, h, w)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["tokenizer", "lam", "dynamics"], required=True)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--frames", type=int, default=16, help="clip length for tokenizer/LAM training")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    c = SurgWMConfig()
    run = OUTPUTS / "runs" / f"surgwm_{args.stage}"
    run.mkdir(parents=True, exist_ok=True)
    lr = {"tokenizer": 1e-4, "lam": 1e-5, "dynamics": 1e-4}[args.stage]       # as reported for SurgWM
    n_frames = 33 if args.stage == "dynamics" else args.frames
    ds = PixelWindows("train", n_frames=n_frames)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, drop_last=True, num_workers=args.workers,
                    persistent_workers=True, pin_memory=True)
    if args.stage == "tokenizer":
        model = VideoVQVAE(c).cuda()
    elif args.stage == "lam":
        model = LatentActionModel(c).cuda()
    else:
        model = MaskGITDynamics(c).cuda()
        tok = VideoVQVAE(c).cuda().eval()
        tok.load_state_dict(torch.load(OUTPUTS / "runs/surgwm_tokenizer/ckpt_last.pt", weights_only=False)["model"])
        lam = LatentActionModel(c).cuda().eval()
        lam.load_state_dict(torch.load(OUTPUTS / "runs/surgwm_lam/ckpt_last.pt", weights_only=False)["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.01, fused=True)
    step = 0
    ckpt = run / "ckpt_last.pt"
    if args.resume and ckpt.exists():
        st = torch.load(ckpt, map_location="cuda", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        step = st["step"]
    model.compile()
    log = (run / "log.jsonl").open("a")
    t0 = time.time()
    while step < args.steps:
        for x in dl:
            if step >= args.steps:
                break
            x = x.cuda(non_blocking=True)
            for g in opt.param_groups:
                g["lr"] = lr * min(1.0, (step + 1) / 1000)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if args.stage == "tokenizer":
                    rec, vq_loss, _ = model(x)
                    rec_loss = F.mse_loss(rec.float(), x)
                    loss = rec_loss + vq_loss
                    logs = {"rec": rec_loss.item(), "vq": vq_loss.item()}
                elif args.stage == "lam":
                    small = F.interpolate(x.flatten(0, 1), size=(c.lam_height, c.lam_width), mode="area")
                    rec_loss, vq_loss, idx = model(small.view(*x.shape[:2], 3, c.lam_height, c.lam_width))
                    loss = rec_loss + vq_loss
                    counts = torch.bincount(idx.flatten(), minlength=c.n_actions).float()
                    p = counts / counts.sum()
                    logs = {"rec": rec_loss.item(), "vq": vq_loss.item(),
                            "perplexity": float(torch.exp(-(p * (p + 1e-10).log()).sum()))}
                else:
                    with torch.no_grad():
                        codes, _, _ = tok.encode(x)
                        small = F.interpolate(x.flatten(0, 1), size=(c.lam_height, c.lam_width), mode="area")
                        a, _, _ = lam.actions(small.view(*x.shape[:2], 3, c.lam_height, c.lam_width))
                    acts = torch.cat([torch.full_like(a[:, :1], c.n_actions), a], 1)
                    loss = model.loss(codes, acts, n_ctx=int(np.random.randint(1, 6)))
                    logs = {"ce": loss.item()}
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 100 == 0:
                rec = dict(step=step, sec_per_step=(time.time() - t0) / 100, **logs)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                t0 = time.time()
                if step % 1000 == 0:
                    print(json.dumps(rec), flush=True)
            if step % 5000 == 0 or step == args.steps:
                tmp = ckpt.with_suffix(".tmp")
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, tmp)
                tmp.replace(ckpt)
    print("done", flush=True)


if __name__ == "__main__":
    main()
