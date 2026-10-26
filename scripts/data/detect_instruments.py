#!/usr/bin/env python
"""Run the SurgVU instrument detector on the latent-step boundary frames of every encoded window.

For each window starting at cache frame s the frames s, s+4, ..., s+32 are detected (one per Wan2.2 latent
step). Frames get the console bands blacked out exactly as in the detector's training. Writes per part
  data/processed/surgvu/detections/<case>_p<part>.npz  frame (N,), box (N, 4) in 640x512 px, conf (N,), cls (N,)
Follows encode_latents.py: parts are processed once their window list exists; re-running skips finished parts.

    python scripts/data/detect_instruments.py --wait
"""

from __future__ import annotations

import argparse
import threading
import time
from queue import Queue

import av
import numpy as np

from swm.data.geometry import UI_BOTTOM_ROWS, UI_TOP_ROWS
from swm.data.io import savez, settled
from swm.paths import DETECTOR, PROCESSED

CACHE = PROCESSED / "surgvu"
OUT = CACHE / "detections"
STEP = 4


def frames_of(stem: str, wanted: np.ndarray, q: Queue, batch: int) -> None:
    want = set(wanted.tolist())
    buf, idx = [], []
    with av.open(str(CACHE / "frames10" / f"{stem}.mp4")) as c:
        c.streams.video[0].thread_type = "AUTO"
        for i, f in enumerate(c.decode(video=0)):
            if i in want:
                a = f.to_ndarray(format="bgr24")
                a[UI_TOP_ROWS[0]:UI_TOP_ROWS[1]] = 0
                a[UI_BOTTOM_ROWS[0]:UI_BOTTOM_ROWS[1]] = 0
                buf.append(a)
                idx.append(i)
                if len(buf) == batch:
                    q.put((idx, buf))
                    buf, idx = [], []
            if i >= wanted[-1]:
                break
    if buf:
        q.put((idx, buf))
    q.put(None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--wait", action="store_true")
    args = ap.parse_args()
    from ultralytics import YOLO

    OUT.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(DETECTOR))
    while True:
        todo = [p for p in sorted((CACHE / "latents").glob("*.starts.npy"))
                if settled(p, 5) and not (OUT / p.name.replace(".starts.npy", ".npz")).exists()]
        if not todo:
            if not args.wait or len(list((CACHE / "latents").glob("*.starts.npy"))) >= 280:
                break
            time.sleep(120)
            continue
        for sp in todo:
            stem = sp.name.replace(".starts.npy", "")
            starts = np.load(sp)
            t0 = time.time()
            rec = {"frame": [], "box": [], "conf": [], "cls": []}
            if len(starts):
                wanted = np.unique((starts[:, None] + np.arange(0, 33, STEP)[None]).ravel())
                q: Queue = Queue(maxsize=8)
                th = threading.Thread(target=frames_of, args=(stem, wanted, q, args.batch), daemon=True)
                th.start()
                for item in iter(q.get, None):
                    idx, imgs = item
                    res = model.predict(imgs, imgsz=args.imgsz, conf=0.1, iou=0.65, agnostic_nms=True, max_det=10,
                                        verbose=False, batch=args.batch)
                    for i, r in zip(idx, res, strict=True):
                        b = r.boxes
                        rec["frame"].append(np.full(len(b), i, np.int32))
                        rec["box"].append(b.xyxy.cpu().numpy().astype(np.float32))
                        rec["conf"].append(b.conf.cpu().numpy().astype(np.float16))
                        rec["cls"].append(b.cls.cpu().numpy().astype(np.int8))
                th.join()
            arrs = {k: (np.concatenate(v) if v else np.zeros((0, 4) if k == "box" else 0)) for k, v in rec.items()}
            savez(OUT / f"{stem}.npz", **arrs)
            print(f"{time.strftime('%H:%M:%S')} {stem}: {len(starts)} windows, {len(arrs['frame'])} boxes "
                  f"in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
