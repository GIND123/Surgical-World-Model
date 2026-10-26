#!/usr/bin/env python
"""Encode every usable 33-frame window of the 10 fps cache with the frozen Wan2.2 VAE.

Per cached part writes (to data/processed/surgvu/latents/, a symlink to NVMe storage)
  <case>_p<part>.npy        (N, 9, 48, 14, 20) float16 raw latents (normalise with the VAE's latents_mean/std)
  <case>_p<part>.starts.npy (N,) start frame of each window in the cache
Parts are processed as soon as their cache metadata exists, val and test cases first, so it can run while the
cache is still being built; re-running skips finished parts. Decoding runs in background processes.

    python scripts/data/encode_latents.py --decoders 6 --batch 8 [--wait]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import av
import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_windows import WIN, usable  # noqa: E402

from swm.data.geometry import FIELD_ROWS, MODEL_H, MODEL_W  # noqa: E402
from swm.data.io import save_npy, settled, try_load  # noqa: E402
from swm.paths import CHECKPOINTS, PROCESSED, ROOT  # noqa: E402

CACHE = PROCESSED / "surgvu"
OUT = CACHE / "latents"


def window_starts(npz: Path) -> np.ndarray | None:
    """None while the cache metadata is still being written."""
    m = try_load(npz) if settled(npz) else None
    if m is None:
        return None
    ok = usable(m)
    return np.array([s for s in range(0, len(ok) - WIN + 1, WIN) if ok[s:s + WIN].all()], dtype=np.int64)


def decode_part(stem: str, starts: np.ndarray, q, batch: int) -> None:
    """Decode the cached part once and push (stem, window indices, frames uint8 [n, 33, H, W, 3])."""
    cv2.setNumThreads(1)
    want = {int(s) + k for s in starts for k in range(WIN)}
    frames, buf, idx = {}, [], []
    next_w = 0
    with av.open(str(CACHE / "frames10" / f"{stem}.mp4")) as c:
        c.streams.video[0].thread_type = "AUTO"
        for i, f in enumerate(c.decode(video=0)):
            if i in want:
                img = f.to_ndarray(format="rgb24")[FIELD_ROWS[0]:FIELD_ROWS[1]]
                frames[i] = cv2.resize(img, (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA)
            while next_w < len(starts) and i >= starts[next_w] + WIN - 1:
                s = int(starts[next_w])
                buf.append(np.stack([frames.pop(s + k) for k in range(WIN)]))
                idx.append(next_w)
                next_w += 1
                if len(buf) == batch:
                    q.put((stem, idx, np.stack(buf)))
                    buf, idx = [], []
    if buf:
        q.put((stem, idx, np.stack(buf)))
    q.put((stem, None, None))                                       # end-of-part marker


def worker(jobs, q, batch: int) -> None:
    for stem, starts in iter(jobs.get, None):
        decode_part(stem, starts, q, batch)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoders", type=int, default=6)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--wait", action="store_true", help="keep polling for newly cached parts until all 280 exist")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    split = json.loads((ROOT / "configs" / "splits" / "surgvu_v1.json").read_text())
    rank = {c: i for i, c in enumerate(split["val"] + split["test"] + split["train"])}

    import diffusers
    torch.backends.cudnn.benchmark = True
    vae = diffusers.AutoencoderKLWan.from_pretrained(CHECKPOINTS / "vae" / "wan22" / "vae", torch_dtype=torch.bfloat16)
    vae = vae.to("cuda").eval()
    encode = torch.compile(vae.encode)

    ctx = mp.get_context("spawn")
    jobs, q = ctx.Queue(), ctx.Queue(maxsize=24)
    procs = [ctx.Process(target=worker, args=(jobs, q, args.batch), daemon=True) for _ in range(args.decoders)]
    for p in procs:
        p.start()
    queued, pending, results = set(), {}, {}
    t0, n_win = time.time(), 0
    while True:
        metas = sorted((CACHE / "meta10").glob("*.npz"), key=lambda p: rank[p.stem.rsplit("_p", 1)[0]])
        for npz in metas:
            stem = npz.stem
            if stem in queued or (OUT / f"{stem}.npy").exists():
                continue
            starts = window_starts(npz)
            if starts is None:
                continue
            queued.add(stem)
            if len(starts) == 0:
                save_npy(OUT / f"{stem}.npy", np.zeros((0, 9, 48, 14, 20), np.float16))
                save_npy(OUT / f"{stem}.starts.npy", starts)
                continue
            pending[stem] = starts
            results[stem] = np.zeros((len(starts), 9, 48, 14, 20), np.float16)
            jobs.put((stem, starts))
        if not pending:
            done_all = len(list((CACHE / "meta10").glob("*.npz"))) >= 280
            if not args.wait or done_all:
                break
            time.sleep(60)
            continue
        stem, idx, frames = q.get()
        if idx is None:
            save_npy(OUT / f"{stem}.npy", results.pop(stem))           # latents first ...
            save_npy(OUT / f"{stem}.starts.npy", pending.pop(stem))    # ... the window list marks the part done
            rate = n_win / (time.time() - t0)
            print(f"{time.strftime('%H:%M:%S')} saved {stem} ({len(list(OUT.glob('*.starts.npy')))} parts, "
                  f"{n_win} windows, {rate:.1f} win/s)", flush=True)
            continue
        x = torch.from_numpy(frames).cuda(non_blocking=True).permute(0, 4, 1, 2, 3)
        x = x.to(torch.bfloat16) / 127.5 - 1
        if x.shape[0] < args.batch:                                 # keep one compiled shape
            x = torch.cat([x, x[-1:].expand(args.batch - x.shape[0], -1, -1, -1, -1)])
        with torch.no_grad():
            z = encode(x).latent_dist.mode()[:len(idx)]            # (n, 48, 9, 14, 20)
        results[stem][idx] = z.transpose(1, 2).to(torch.float16).cpu().numpy()
        n_win += len(idx)
    for _ in procs:
        jobs.put(None)


if __name__ == "__main__":
    main()
