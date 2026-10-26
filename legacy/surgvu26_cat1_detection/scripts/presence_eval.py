"""An UNCONTAMINATED checkpoint-ranking signal.

The round-3 v5 holdout is contaminated (a single teacher that saw all 7 videos
produced every pseudo-label), and worse, it appears to track *teacher similarity*
rather than quality. This metric avoids the teacher entirely:

  frames  : keyframes from the 280 surgvu24 training videos that were NEVER
            selected for pseudo-labelling, so no checkpoint has trained on them
  labels  : the installed-instrument multiset from tools.csv -- real ground
            truth, and never used as box supervision for these frames

It measures detection + classification, NOT box IoU, so it cannot settle the
high-IoU precision question. It can rank classification quality and detect a
collapse, which is what the contaminated v5 curve can no longer do.

Two numbers, both macro-averaged over frames:
  f1_topk   threshold-free: keep exactly len(installed) top-scoring boxes, so
            confidence calibration differences between epochs cannot skew it.
            This is the primary ranking metric.
  f1_conf   at a fixed confidence, which also reflects calibration/count.
"""
import os, sys, json, pickle, argparse, time, collections
import numpy as np, torch, cv2
sys.path.insert(0, os.path.dirname(__file__))
from infer_core import detect, mask_ui, NAMES, W, H
from pseudo import frame_installed
from torch.utils.data import Dataset, DataLoader

ROOT = '/mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image/work'
FR = f'{ROOT}/data/train_frames'


class FrameDS(Dataset):
    def __init__(self, rec, idxs): self.rec = rec; self.idxs = idxs
    def __len__(self): return len(self.idxs)
    def __getitem__(self, i):
        j = self.idxs[i]
        img = cv2.imread(f'{FR}/{self.rec[j][0]}')
        if img is None:
            img = np.zeros((H, W, 3), np.uint8)
        elif img.shape[0] != H or img.shape[1] != W:
            img = cv2.resize(img, (W, H))
        return torch.from_numpy(mask_ui(img)), j


def coll(b): return [x[0] for x in b], [x[1] for x in b]


def multiset_f1(pred, inst):
    if len(pred) == 0 and len(inst) == 0: return 1.0
    if len(pred) == 0 or len(inst) == 0: return 0.0
    cp, ci = collections.Counter(pred), collections.Counter(inst)
    tp = sum(min(cp[k], ci[k]) for k in cp)
    p, r = tp / len(pred), tp / len(inst)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def evaluate_ckpt(weights, idxs, rec, imgsz, conf, batch, workers):
    from ultralytics import YOLO
    m = YOLO(weights).model.cuda().eval().half()
    for p in m.parameters(): p.requires_grad_(False)
    dl = DataLoader(FrameDS(rec, idxs), batch_size=batch, num_workers=workers,
                    collate_fn=coll, shuffle=False, pin_memory=False)
    f1t, f1c, cnt_ok, n = [], [], 0, 0
    for imgs, js in dl:
        frames = [x.numpy() for x in imgs]
        # low conf so the top-k branch has candidates to rank
        dets = detect([m], frames, imgsz=imgsz, conf=0.01, iou=0.6, max_det=8)
        for (bx, sv), j in zip(dets, js):
            inst = frame_installed(rec[j][4:8])
            if inst is None: continue
            inst_c = list(inst[0]); n += 1
            if len(bx) == 0:
                f1t.append(0.0); f1c.append(0.0); continue
            best = sv.max(1)
            cls = sv.argmax(1)
            order = np.argsort(-best)
            k = min(len(inst_c), len(order))
            f1t.append(multiset_f1([int(cls[i]) for i in order[:k]], inst_c))
            keep = [i for i in order if best[i] >= conf]
            f1c.append(multiset_f1([int(cls[i]) for i in keep], inst_c))
            cnt_ok += int(len(keep) == len(inst_c))
    del m; torch.cuda.empty_cache()
    return dict(f1_topk=float(np.mean(f1t)), f1_conf=float(np.mean(f1c)),
                count_match=cnt_ok / max(n, 1), n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', nargs='+', required=True)
    ap.add_argument('--n', type=int, default=4000)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--sel', default='selected_frames_xl')
    ap.add_argument('--out', default=f'{ROOT}/runs/presence_scores.json')
    a = ap.parse_args()

    rec = pickle.load(open(f'{ROOT}/data/frame_index.pkl', 'rb'))
    used = set(pickle.load(open(f'{ROOT}/data/{a.sel}.pkl', 'rb')))
    rng = np.random.default_rng(0)
    pool = np.array([j for j in range(len(rec)) if j not in used], dtype=np.int64)
    rng.shuffle(pool)
    # keep only frames whose arm state is usable, then take the first n
    idxs = []
    for j in pool:
        if frame_installed(rec[j][4:8]) is not None:
            idxs.append(int(j))
            if len(idxs) >= a.n: break
    print(f'held-out unseen frames: {len(idxs)} (pool {len(pool)}, trained-on {len(used)})', flush=True)

    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for c in a.ckpts:
        k = f'{os.path.basename(os.path.dirname(os.path.dirname(c)))}/{os.path.basename(c)}'
        if k in res:
            r = res[k]
        else:
            t0 = time.time()
            r = evaluate_ckpt(c, idxs, rec, a.imgsz, a.conf, a.batch, a.workers)
            r['secs'] = round(time.time() - t0)
            res[k] = r
            json.dump(res, open(a.out, 'w'), indent=1)
        print(f'  {k:28s} f1_topk={r["f1_topk"]:.4f}  f1_conf={r["f1_conf"]:.4f}  '
              f'count_match={r["count_match"]:.3f}  n={r["n"]}', flush=True)
    b = max(res, key=lambda k: res[k]['f1_topk'])
    print(f'\nBEST by f1_topk: {b}  {res[b]["f1_topk"]:.4f}')


if __name__ == '__main__':
    main()
