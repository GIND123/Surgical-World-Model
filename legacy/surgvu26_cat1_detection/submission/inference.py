"""SurgVU 2026 Category 1 - surgical tool detection & localisation.

Reads   : /input/endoscopic-robotic-surgery-video.mp4   (socket endoscopic-robotic-surgery-video)
Writes  : /output/surgical-tools.json                   ("Multiple 2D bounding boxes")

Frame numbering: the official evaluator (isi-challenges/surgvu24-category-1-eval)
subtracts 1 from GT slice numbers but NOT from prediction slice numbers, so a
prediction's slice_nr must equal the 0-based frame index.
"""
import os, sys, json, time, glob, traceback
from pathlib import Path

INPUT_PATH  = Path(os.environ.get("SURGVU_INPUT",  "/input"))
OUTPUT_PATH = Path(os.environ.get("SURGVU_OUTPUT", "/output"))
MODEL_DIR   = Path(os.environ.get("SURGVU_MODELS", "/opt/app/model"))
TIME_BUDGET = float(os.environ.get("SURGVU_BUDGET", "480"))   # 10 min hard limit; stay well under

os.environ.setdefault("YOLO_CONFIG_DIR", "/opt/app/.cfg")
os.environ.setdefault("YOLO_OFFLINE", "True")
os.environ.setdefault("OMP_NUM_THREADS", "4")

T_START = time.time()

def log(*a):
    print(f"[{time.time()-T_START:7.1f}s]", *a, flush=True)

def write_json(content):
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH / "surgical-tools.json", "w") as f:
        json.dump(content, f)
    log("wrote", OUTPUT_PATH / "surgical-tools.json", f'boxes={len(content["boxes"])}')

EMPTY = {"name": "Regions of interest", "type": "Multiple 2D bounding boxes",
         "boxes": [], "version": {"major": 1, "minor": 0}}

def find_video():
    cands = [INPUT_PATH / "endoscopic-robotic-surgery-video.mp4"]
    try:
        inputs = json.load(open(INPUT_PATH / "inputs.json"))
        for sv in inputs:
            itf = sv.get("interface", {})
            if itf.get("slug") == "endoscopic-robotic-surgery-video" and itf.get("relative_path"):
                cands.insert(0, INPUT_PATH / itf["relative_path"])
    except Exception as e:
        log("inputs.json not usable:", e)
    for c in cands:
        if c.is_file():
            return str(c)
    for pat in ("*.mp4", "*/*.mp4", "*.avi", "*/*.avi", "*.mkv", "*/*.mkv"):
        g = sorted(glob.glob(str(INPUT_PATH / pat)))
        if g:
            return g[0]
    raise FileNotFoundError(f"no video found under {INPUT_PATH}")

def main():
    import cv2, numpy as np, torch
    sys.path.insert(0, str(Path(__file__).parent))
    from infer_core import detect, mask_ui, GC_NAMES, W, H
    from track_core import link_tracks, vote, emit, to_gc

    cfg = json.load(open(MODEL_DIR / "config.json"))
    log("config:", cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    half = bool(cfg.get("half", True)) and device == "cuda"
    log("device:", device, "half:", half)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    from ultralytics import YOLO
    models = []
    for w in cfg["weights"]:
        m = YOLO(str(MODEL_DIR / w)).model.to(device).eval()
        m = m.half() if half else m.float()
        for p in m.parameters():
            p.requires_grad_(False)
        models.append(m)
    log(f"loaded {len(models)} model(s)")

    vid = find_video()
    log("video:", vid)
    cap = cv2.VideoCapture(vid)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if fr.shape[0] != H or fr.shape[1] != W:
            fr = cv2.resize(fr, (W, H), interpolation=cv2.INTER_LINEAR)
        frames.append(mask_ui(fr))
    cap.release()
    n = len(frames)
    log(f"decoded {n} frames")
    if n == 0:
        write_json(EMPTY); return 0

    # ---- adaptive TTA: one measurement, then extrapolate cost analytically ----
    plans = cfg["plans"]                       # richest first
    batch = int(cfg.get("batch", 16))

    # `unit` below is measured on plans[-1], so costs are expressed relative to it.
    REF_IMGSZ = float(plans[-1].get("imgsz", 640))

    def cost_units(p):
        # each forward pass costs ~ (imgsz * scale)^2; a flip doubles passes; so
        # does each extra model. imgsz must be in here or a ladder that mixes
        # resolutions under-estimates its own high-res plans and busts the budget.
        nm = len(p.get("use_models", range(len(models))))
        r = (float(p.get("imgsz", 640)) / REF_IMGSZ) ** 2
        return r * sum(sc * sc for sc in p["scales"]) * (2 if p.get("flip") else 1) * nm

    cheap = plans[-1]
    probe = frames[: min(batch, n)]
    try:
        detect(models, probe, device=device, half=half, **cheap)   # warm up
    except Exception:
        traceback.print_exc()
    t0 = time.time()
    _ = detect(models, probe, device=device, half=half, **cheap)
    unit = (time.time() - t0) / len(probe) / max(cost_units(cheap), 1e-6)
    log(f"probe: {unit*1000:.1f} ms/frame per cost-unit")

    chosen = plans[-1]
    for pi, plan in enumerate(plans):
        est = unit * cost_units(plan) * n * 1.5 + 25   # probe runs on warm frames; keep margin
        rem = TIME_BUDGET - (time.time() - T_START)
        log(f"plan[{pi}] units={cost_units(plan):.2f} est={est:.0f}s remaining={rem:.0f}s")
        if est <= rem:
            chosen = plan
            break
    log("chosen plan:", chosen)

    dets = []
    for s in range(0, n, batch):
        dets += detect(models, frames[s:s + batch], device=device, half=half, **chosen)
        if time.time() - T_START > TIME_BUDGET:
            log("BUDGET HIT during detection at frame", s + batch, "- padding remainder")
            while len(dets) < n:
                dets.append((np.zeros((0, 4), np.float32), np.zeros((0, 14), np.float32)))
            break
    log("detection done")

    sc = [s for _, s in dets]
    if cfg.get("track", True):
        try:
            tr = link_tracks(dets, iou_thr=cfg.get("iou_link", 0.30))
            sc = vote(dets, tr, alpha=cfg.get("alpha", 0.65))
            log(f"tracking: {len(tr)} tracks")
        except Exception:
            log("tracking failed, using raw scores"); traceback.print_exc()
    pf = emit(dets, sc, topk=cfg.get("topk", 3), min_conf=cfg.get("min_conf", 0.001),
              max_per_frame=cfg.get("max_per_frame", 30))
    write_json(to_gc(pf, GC_NAMES))
    return 0

if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        try:
            write_json(EMPTY)
        except Exception:
            traceback.print_exc()
        rc = 0
    log("done")
    raise SystemExit(rc)
