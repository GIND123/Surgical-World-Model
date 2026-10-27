#!/usr/bin/env python
"""Map an Open-H (LeRobot v2.1) dataset with kinematics onto the SurgVU world-model representation.

Per episode the left endoscope video is resampled to 10 fps, centre-cropped to the model aspect (320:224) and
cut into 33-frame windows. Each window gets Wan2.2 latents, camera ego-motion per latent step, instrument
detections at the latent-step boundary frames (two largest-confidence boxes, bound to PSM1/PSM2 by image
side) and the robot signals of every frame (PSM states/actions, ECM joints, pedals when present).
Arm slots follow the SurgVU convention: PSM1 -> slot 0, endoscope -> slot 1, PSM2 -> slot 2, slot 3 empty.

Writes data/processed/<name>/{latents.npy, windows.csv, signals.npz, conditioning.npz, eval_frames.npy}.

    python scripts/data/prepare_lerobot.py --name suturebot \
        --root data/external/open_h_suturebot/Surgical/jhu/imerse/suturebot
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from collections import deque

import av
import cv2
import numpy as np
import pandas as pd
import torch

from swm.data import egomotion
from swm.data import surgvu as sv
from swm.data.geometry import MODEL_H, MODEL_W
from swm.paths import CHECKPOINTS, DETECTOR, PROCESSED, ROOT

WIN, STEP = 33, 4
TOOL_MAP = {"large needle driver": "needle driver", "debakey forceps": "needle driver",
            "needle driver": "needle driver",
            "maryland bipolar forceps": "bipolar forceps", "fenestrated bipolar forceps": "bipolar forceps",
            "permanent cautery hook": "permanent cautery hook/spatula", "cadiere forceps": "cadiere forceps",
            "prograsp forceps": "prograsp forceps", "monopolar curved scissors": "monopolar curved scissors",
            "curved scissors": "monopolar curved scissors"}


def crop_resize(img: np.ndarray, h: int, w: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    target = w / h
    if iw / ih > target:                       # too wide: crop columns
        cw = int(round(ih * target))
        x0 = (iw - cw) // 2
        img = img[:, x0:x0 + cw]
    else:
        ch = int(round(iw / target))
        y0 = (ih - ch) // 2
        img = img[y0:y0 + ch]
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def col(df: pd.DataFrame, name: str) -> np.ndarray | None:
    return np.stack(df[name].values).astype(np.float32) if name in df else None


def episode_job(args) -> dict:
    """Decode one episode at 10 fps -> frames (n, 448, 640, 3), signals (n, .), ego-motion (n, 4)."""
    root, info, ep, cam_key, stride = args
    cv2.setNumThreads(1)
    chunk = ep["episode_index"] // info["chunks_size"]
    vpath = root / info["video_path"].format(episode_chunk=chunk, video_key=cam_key, episode_index=ep["episode_index"])
    dpath = root / info["data_path"].format(episode_chunk=chunk, episode_index=ep["episode_index"])
    df = pd.read_parquet(dpath)
    sub = max(1, round(info["fps"] / 10))
    frames = []
    with av.open(str(vpath)) as c:
        c.streams.video[0].thread_type = "AUTO"
        for i, f in enumerate(c.decode(video=0)):
            if i % sub == 0:
                frames.append(crop_resize(f.to_ndarray(format="rgb24"), 2 * MODEL_H, 2 * MODEL_W))
    n = min(len(frames), (len(df) + sub - 1) // sub)
    frames = np.stack(frames[:n]) if n else np.zeros((0, 2 * MODEL_H, 2 * MODEL_W, 3), np.uint8)
    rows = df.iloc[::sub].iloc[:n]
    sig = {k: col(rows, key) for k, key in [("state", "observation.state"), ("action", "action"),
                                          ("ecm", "observation.meta.ecm_measured_js"),
                                          ("pedals", "observation.meta.pedals")]}
    small = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA)
             for f in frames]
    ego = np.full((n, 4), np.nan, np.float32)
    inl = np.zeros(n, np.float32)
    for i in range(1, n):
        ego[i], inl[i], _ = egomotion.estimate(small[i - 1], small[i])
    tools = []
    for k in ("observation.meta.tool.psm1", "observation.meta.tool.psm2"):
        name = str(rows[k].iloc[0]).strip().lower() if k in rows and n else ""
        tools.append(sv.OCC_INDEX.get(TOOL_MAP.get(name, ""), sv.OCC_INDEX[sv.UNKNOWN]))
    starts = list(range(0, n - WIN + 1, stride))
    return dict(ep=ep["episode_index"], frames=frames, sig=sig, ego=ego, inl=inl, tools=tools, starts=starts)


def bounded_imap(pool, fn, jobs, ahead: int):
    """Ordered pool.imap with at most ``ahead`` results in flight: decoded episodes are large, and an
    unbounded imap lets the workers outrun the GPU consumer until memory runs out."""
    it = iter(jobs)
    pending = deque(pool.apply_async(fn, (j,)) for _, j in zip(range(ahead), it, strict=False))
    while pending:
        res = pending.popleft().get()
        nxt = next(it, None)
        if nxt is not None:
            pending.append(pool.apply_async(fn, (nxt,)))
        yield res


def build(name: str, results, split_of: dict) -> None:
    """Encode, detect and assemble windows from an iterator of decoded episodes (see episode_job)."""
    out = PROCESSED / name
    out.mkdir(parents=True, exist_ok=True)

    import diffusers
    from ultralytics import YOLO
    vae = diffusers.AutoencoderKLWan.from_pretrained(CHECKPOINTS / "vae" / "wan22" / "vae", torch_dtype=torch.bfloat16)
    vae = vae.cuda().eval()
    det = YOLO(str(DETECTOR))

    rows, lat_chunks, cond, sig_chunks, eval_frames = [], [], [], {}, []
    for res in results:
        frames, starts = res["frames"], res["starts"]
        if not starts:
            continue
        small = np.stack([cv2.resize(f, (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA) for f in frames])
        for i in range(0, len(starts), 8):
            ss = starts[i:i + 8]
            x = torch.from_numpy(np.stack([small[s:s + WIN] for s in ss])).cuda().permute(0, 4, 1, 2, 3)
            with torch.no_grad():
                z = vae.encode(x.to(torch.bfloat16) / 127.5 - 1).latent_dist.mode()
            lat_chunks.append(z.transpose(1, 2).to(torch.float16).cpu().numpy())
        # detections at latent-step boundary frames, bound to arms by image side
        bidx = sorted({s + STEP * t for s in starts for t in range(9)})
        boxes_at = {}
        for i in range(0, len(bidx), 32):
            ims = [cv2.cvtColor(frames[j], cv2.COLOR_RGB2BGR) for j in bidx[i:i + 32]]
            for j, r in zip(bidx[i:i + 32], det.predict(ims, imgsz=640, conf=0.25, verbose=False), strict=True):
                b = r.boxes.xyxy.cpu().numpy()[np.argsort(-r.boxes.conf.cpu().numpy())][:2]
                b = b / np.array([2 * MODEL_W, 2 * MODEL_H, 2 * MODEL_W, 2 * MODEL_H])
                boxes_at[j] = b[np.argsort(b[:, 0] + b[:, 2])]          # left to right
        for s in starts:
            occ = np.zeros((9, 4), np.int8)
            occ[:, 0], occ[:, 1], occ[:, 2] = res["tools"][0], sv.OCC_INDEX[sv.CAMERA], res["tools"][1]
            bx = np.full((9, 4, 4), np.nan, np.float32)        # side assignment calibrated later per dataset
            for t in range(9):
                for f in ([s + STEP * t] if t == 0 else [s + STEP * (t - 1), s + STEP * t]):
                    b = boxes_at.get(f, np.zeros((0, 4)))
                    for slot, k in ((0, -1), (2, 0)):          # right-most -> slot 0, left-most -> slot 2
                        if len(b) == 2 or (len(b) == 1 and slot == 0):
                            v = b[k]
                            old = bx[t, slot]
                            bx[t, slot] = v if np.isnan(old[0]) else np.r_[np.minimum(old[:2], v[:2]),
                                                                         np.maximum(old[2:], v[2:])]
            cam_a = np.zeros((9, 16), np.float32)
            cam_ok = np.zeros(9, bool)
            for t in range(1, 9):
                e = res["ego"][s + STEP * (t - 1) + 1:s + STEP * t + 1]
                ok = np.isfinite(e).all(1) & (res["inl"][s + STEP * (t - 1) + 1:s + STEP * t + 1] > 0.5)
                cam_a[t] = np.where(ok[:, None], e, 0).ravel()
                cam_ok[t] = ok.all()
            cond.append((occ, bx, cam_a, cam_ok))
            split = split_of.get(res["ep"], "train")
            rows.append(dict(episode=res["ep"], start=s, split=split))
            for k, v in res["sig"].items():
                if v is not None:
                    sig_chunks.setdefault(k, []).append(v[s:s + WIN])
            if split == "test":
                eval_frames.append(small[s:s + WIN])
        print(f"episode {res['ep']}: {len(starts)} windows (total {len(rows)})", flush=True)
    np.save(out / "latents.npy", np.concatenate(lat_chunks))
    pd.DataFrame(rows).to_csv(out / "windows.csv", index=False)
    np.savez_compressed(out / "signals.npz", **{k: np.stack(v) for k, v in sig_chunks.items()})
    occ, bx, cam_a, cam_ok = (np.stack(x) for x in zip(*cond, strict=True))
    np.savez_compressed(out / "conditioning.npz", occupant=occ, boxes=bx, camera=cam_a, cam_ok=cam_ok)
    if eval_frames:
        np.save(out / "eval_frames.npy", np.stack(eval_frames))
    print(f"{name}: {len(rows)} windows -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--camera", default=None, help="video feature key (default: first endoscope-left stream)")
    ap.add_argument("--stride", type=int, default=16, help="window stride in 10 fps frames")
    ap.add_argument("--test-frac", type=float, default=0.2, help="episode fraction held out when no split exists")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-episodes", type=int)
    args = ap.parse_args()
    root = ROOT / args.root
    info = json.loads((root / "meta" / "info.json").read_text())
    eps = [json.loads(line) for line in (root / "meta" / "episodes.jsonl").read_text().splitlines() if line.strip()]
    if args.max_episodes:
        eps = eps[:args.max_episodes]
    cam = args.camera or next(k for k in info["features"] if "endoscope" in k and "left" in k)
    split_of = {}
    for name, rng in info.get("splits", {}).items():
        if name in ("train", "val", "test"):
            a, b = map(int, rng.split(":"))
            split_of.update({i: name for i in range(a, b)})
    if not any(v == "test" for v in split_of.values()):        # deterministic episode-level hold-out
        order = np.random.default_rng(0).permutation([e["episode_index"] for e in eps])
        n_test = max(1, round(args.test_frac * len(order)))
        split_of = {int(e): ("test" if i < n_test else "train") for i, e in enumerate(order)}
    jobs = [(root, info, e, cam, args.stride) for e in eps]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        build(args.name, bounded_imap(pool, episode_job, jobs, args.workers + 1), split_of)


if __name__ == "__main__":
    main()
