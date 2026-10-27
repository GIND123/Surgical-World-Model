#!/usr/bin/env python
"""One decode pass over every SurgVU video part -> 10 fps cache + per-frame metadata.

For each part writes
  data/processed/surgvu/frames10/<case>_p<part>.mp4   640x512 H.264, pillarbox removed, console UI kept
  data/processed/surgvu/meta10/<case>_p<part>.npz     per cached frame: local/global time, quality stats,
                                                      UI-band stats, endoscope ego-motion from the previous frame
Frames stay in YUV 4:2:0 throughout (crop + resize per plane, encode from planes); only the small
224x320 endoscope field is converted to RGB for the statistics. Resumable: parts with an .npz are skipped.

    python scripts/data/build_surgvu_cache.py --workers 8 --encoder nvenc
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import av
import cv2
import numpy as np

from swm.data import egomotion, surgvu
from swm.data.geometry import CACHE_FPS, CACHE_H, CACHE_W, FIELD_ROWS, SRC_CROP, UI_BOTTOM_ROWS, UI_TOP_ROWS
from swm.data.io import savez
from swm.paths import PROCESSED, SURGVU24_VIDEOS

OUT = PROCESSED / "surgvu"
ENCODERS = {
    "x264": ("libx264", {"crf": "18", "preset": "fast", "threads": "2",
                         "x264-params": "keyint=20:min-keyint=20:scenecut=0"}),
    "nvenc": ("h264_nvenc", {"preset": "p4", "tune": "hq", "rc": "vbr", "cq": "19", "b": "0", "g": "20", "bf": "0"}),
}


def planes(frame: av.VideoFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = frame.to_ndarray(format="yuv420p")              # (3h/2, w): Y, then U and V packed row-wise
    h, w = frame.height, frame.width
    y = a[:h]
    u = a[h:h + h // 4].reshape(h // 2, w // 2)
    v = a[h + h // 4:].reshape(h // 2, w // 2)
    return y, u, v


def i420(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.concatenate([y.ravel(), u.ravel(), v.ravel()]).reshape(y.shape[0] * 3 // 2, y.shape[1])


def edge_density(gray: np.ndarray) -> float:
    return float(cv2.Canny(gray, 60, 160).mean() / 255)


def cpu_frames(path: str, max_seconds: float | None, hwaccel: bool = False):
    """Yield (t, Y 512x640, U 256x320, V 256x320) uint8 planes at 10 fps, decoded and resized on the CPU."""
    x0, x1, y0, y1 = SRC_CROP
    kw = {}
    if hwaccel:
        from av.codec.hwaccel import HWAccel
        kw["hwaccel"] = HWAccel(device_type="cuda", allow_software_fallback=True)
    with av.open(path, **kw) as src:
        vs = src.streams.video[0]
        vs.thread_type, vs.thread_count = "AUTO", 2
        next_t = 0.0
        for frame in src.decode(vs):
            t = float(frame.time)
            if max_seconds is not None and t > max_seconds:
                break
            if t + 1e-4 < next_t:
                continue
            next_t = (np.floor(t * CACHE_FPS + 1e-4) + 1) / CACHE_FPS
            y, u, v = planes(frame)
            yield (t, cv2.resize(y[y0:y1, x0:x1], (CACHE_W, CACHE_H), interpolation=cv2.INTER_AREA),
                   cv2.resize(u[y0 // 2:y1 // 2, x0 // 2:x1 // 2], (CACHE_W // 2, CACHE_H // 2),
                              interpolation=cv2.INTER_AREA),
                   cv2.resize(v[y0 // 2:y1 // 2, x0 // 2:x1 // 2], (CACHE_W // 2, CACHE_H // 2),
                              interpolation=cv2.INTER_AREA))


def area_matrix(n_in: int, n_out: int) -> np.ndarray:
    """Weights of OpenCV's INTER_AREA downscale along one axis: output i averages input [i*s, (i+1)*s), s = n_in/n_out,
    counting partially covered pixels by their overlap. X_out = A_rows @ X @ A_cols.T reproduces cv2.resize."""
    s = n_in / n_out
    a = np.zeros((n_out, n_in), np.float64)
    for i in range(n_out):
        lo, hi = i * s, (i + 1) * s
        for j in range(int(np.floor(lo)), min(n_in, int(np.ceil(hi)))):
            a[i, j] = (min(hi, j + 1) - max(lo, j)) / s
    return a


def nvdec_frames(path: str, max_seconds: float | None, fps: float):
    """Same as cpu_frames, but decoded by NVDEC straight into GPU memory and cropped / area-resized with CUDA;
    only the small 10 fps planes are copied back to the host (keeps the CPU free for the other workers)."""
    import PyNvVideoCodec as nvc
    import torch

    x0, x1, y0, y1 = SRC_CROP
    step = int(round(fps / CACHE_FPS))
    dec = nvc.ThreadedDecoder(path, buffer_size=120, gpu_id=0, use_device_memory=True,
                              output_color_type=nvc.OutputColorType.NATIVE)
    dev = "cuda"
    ry, cy = (torch.tensor(area_matrix(*d), device=dev, dtype=torch.float32)
              for d in ((y1 - y0, CACHE_H), (x1 - x0, CACHE_W)))
    ruv, cuv = (torch.tensor(area_matrix(*d), device=dev, dtype=torch.float32)
                for d in (((y1 - y0) // 2, CACHE_H // 2), ((x1 - x0) // 2, CACHE_W // 2)))
    i = 0
    while True:
        batch = dec.get_batch_frames(120)
        if not batch:
            break
        keep = []
        for f in batch:
            if i % step == 0:
                keep.append((i, torch.from_dlpack(f).clone()))       # NV12: 720 luma rows, 360 interleaved UV
            i += 1
        if keep:
            nv = torch.stack([k[1] for k in keep])
            h = nv.shape[1] * 2 // 3
            yy = nv[:, y0:y1, x0:x1].float()
            uv = nv[:, h + y0 // 2:h + y1 // 2, x0:x1].float()
            uv = uv.view(len(keep), uv.shape[1], uv.shape[2] // 2, 2).permute(0, 3, 1, 2)
            yc = ry @ yy @ cy.T                                       # exact INTER_AREA, on the GPU
            uvc = ruv @ uv @ cuv.T
            yc = yc.round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            uvc = uvc.round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            for j, (idx, _) in enumerate(keep):
                yield idx / fps, yc[j], uvc[j, 0], uvc[j, 1]
        if max_seconds is not None and i / fps > max_seconds:
            break


def process_part(case: str, part: dict, offset: float, encoder: str, max_seconds: float | None,
                 hwaccel: bool = False, nvdec: bool = False) -> str:
    cv2.setNumThreads(1)          # OpenCV otherwise spawns a pool per worker and oversubscribes the CPU
    stem = f"{case}_p{part['part']:03d}"
    mp4, npz = OUT / "frames10" / f"{stem}.mp4", OUT / "meta10" / f"{stem}.npz"
    lock = npz.with_suffix(".lock")
    if npz.exists():
        return f"skip {stem}"
    try:                                  # several builder runs can share the work without collisions
        lock.touch(exist_ok=False)
    except FileExistsError:
        return f"locked {stem}"
    t0 = time.time()
    f0, f1 = FIELD_ROWS
    rec = {k: [] for k in ("t", "luma", "sat", "sharp", "green", "red", "top_y", "top_e", "bot_y", "bot_e",
                           "ego", "ego_inlier", "ego_n")}
    tmp = mp4.with_suffix(".tmp.mp4")
    codec, opts = ENCODERS[encoder]
    path = str(SURGVU24_VIDEOS / part["path"])
    source = nvdec_frames(path, max_seconds, part["fps"]) if nvdec else cpu_frames(path, max_seconds, hwaccel)
    with av.open(str(tmp), "w") as dst:
        enc = dst.add_stream(codec, rate=int(CACHE_FPS))
        enc.width, enc.height, enc.pix_fmt = CACHE_W, CACHE_H, "yuv420p"
        enc.options = opts
        prev_gray = None
        for t, yc, uc, vc in source:
            for pkt in enc.encode(av.VideoFrame.from_ndarray(i420(yc, uc, vc), format="yuv420p")):
                dst.mux(pkt)
            # endoscope field at 224x320 (luma for tracking, RGB for colour statistics)
            gray = cv2.resize(yc[f0:f1], (320, 224), interpolation=cv2.INTER_AREA)
            uf = cv2.resize(uc[f0 // 2:f1 // 2], (160, 112), interpolation=cv2.INTER_AREA)
            vf = cv2.resize(vc[f0 // 2:f1 // 2], (160, 112), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(i420(gray, uf, vf), cv2.COLOR_YUV2RGB_I420)
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
            rec["t"].append(t)
            rec["luma"].append(vv.mean())
            rec["sat"].append(ss.mean())
            rec["sharp"].append(cv2.Laplacian(gray, cv2.CV_32F).var())
            rec["green"].append(((hh > 35) & (hh < 85) & (ss > 80) & (vv > 60)).mean())
            rec["red"].append((((hh < 12) | (hh > 160)) & (ss > 60) & (vv > 40)).mean())
            top = yc[UI_TOP_ROWS[0]:UI_TOP_ROWS[1], 10:-10]
            bot = yc[UI_BOTTOM_ROWS[0]:UI_BOTTOM_ROWS[1], 10:-10]
            rec["top_y"].append(top.mean())
            rec["top_e"].append(edge_density(top))
            rec["bot_y"].append(bot.mean())
            rec["bot_e"].append(edge_density(bot))
            if prev_gray is None or vv.mean() < 15:
                p, r, n = egomotion.NAN4, 0.0, 0
            else:
                p, r, n = egomotion.estimate(prev_gray, gray)
            rec["ego"].append(p)
            rec["ego_inlier"].append(r)
            rec["ego_n"].append(n)
            prev_gray = gray
        for pkt in enc.encode():
            dst.mux(pkt)
    tmp.rename(mp4)
    arrs = {k: np.asarray(v, dtype=np.float32) for k, v in rec.items()}
    arrs["t_global"] = arrs["t"].astype(np.float64) + offset
    arrs["ego_n"] = arrs["ego_n"].astype(np.int16)
    savez(npz, **arrs)
    lock.unlink(missing_ok=True)
    n, dt = len(arrs["t"]), time.time() - t0
    return f"done {stem}: {n} frames ({n / CACHE_FPS / 3600:.2f} h) in {dt:.0f}s = {n / CACHE_FPS / dt:.1f}x realtime"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--encoder", choices=sorted(ENCODERS), default="nvenc")
    ap.add_argument("--cases", nargs="*", help="restrict to these cases (default: all)")
    ap.add_argument("--max-seconds", type=float, help="benchmarking: stop each part after this many seconds")
    ap.add_argument("--hwaccel", action="store_true", help="NVDEC through FFmpeg (frames copied to the host)")
    ap.add_argument("--nvdec", action="store_true", help="NVDEC into GPU memory, crop/resize with CUDA")
    ap.add_argument("--reverse", action="store_true", help="shortest parts first (for a second concurrent run)")
    args = ap.parse_args()
    (OUT / "frames10").mkdir(parents=True, exist_ok=True)
    (OUT / "meta10").mkdir(parents=True, exist_ok=True)
    jobs = []
    for case in args.cases or surgvu.cases():
        tl = surgvu.timeline(case)
        for part in surgvu.parts_table()[case]:
            jobs.append((part["duration"], case, part, tl.offsets[part["part"]]))
    jobs.sort(key=lambda j: j[0] if args.reverse else -j[0])   # longest first for load balance
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(process_part, c, p, o, args.encoder, args.max_seconds, args.hwaccel, args.nvdec)
                for _, c, p, o in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            print(f"[{i}/{len(futs)}] {f.result()}", flush=True)


if __name__ == "__main__":
    main()
