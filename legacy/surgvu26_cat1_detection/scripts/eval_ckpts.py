"""Score every saved checkpoint of the final run on the held-out video with the
exact container inference pipeline, so the shipped epoch is chosen on evidence
rather than assumed. Optionally sweeps inference resolution on the best one.
"""
import sys, os, re, glob, json, torch
sys.path.insert(0, 'scripts')
from eval_pipeline import load_models, score

ROOT = '/mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image/work'
KW = dict(use_track=True, alpha=0.35, topk=5, conf=0.001, max_det=30,
          scales=(0.8, 1.0, 1.25, 1.5), flip=True, verbose=False)
# smaller batches let this run alongside training without fighting it for VRAM;
# the batch size changes speed only, never the score.
BATCH = int(os.environ.get('EVAL_BATCH', 32))


def ep(p):
    m = re.findall(r'epoch(\d+)', p)
    return int(m[0]) if m else 10**6


def main():
    rundir = sys.argv[1]
    vid = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    out = sys.argv[3] if len(sys.argv) > 3 else f'{ROOT}/runs/ckpt_scores.json'
    cks = sorted(glob.glob(f'{rundir}/weights/epoch*.pt'), key=ep)
    last = f'{rundir}/weights/last.pt'
    # last.pt is only worth scoring when no per-epoch checkpoints exist: with
    # save_period=1 it duplicates the final epoch, and caching it by basename
    # would pin a stale score to a name whose contents change every epoch.
    if os.path.exists(last) and not cks:
        cks.append(last)
    res = {}
    if os.path.exists(out):
        res = json.load(open(out))
    best = (None, -1)
    for c in cks:
        k = os.path.basename(c)
        if k in res:
            s = res[k]
        else:
            m = load_models([c])
            s, _ = score(m, vids=[vid], imgsz=640, batch=BATCH, **KW)
            del m
            torch.cuda.empty_cache()
            res[k] = s
            json.dump(res, open(out, 'w'), indent=1)
        flag = ''
        if s > best[1]:
            best = (c, s); flag = '  <-- best so far'
        print(f'  {k:14s} held-out v{vid} mAP = {s:.4f}{flag}', flush=True)
    print(f'\nBEST: {best[0]}  mAP={best[1]:.4f}')

    if best[0] and int(os.environ.get('EVAL_SWEEP', 1)):
        print('\n=== inference-resolution sweep on the best checkpoint ===')
        m = load_models([best[0]])
        for sz, sc in [(640, (0.8, 1.0, 1.25, 1.5)), (640, (0.8, 1.0, 1.25)),
                       (768, (0.8, 1.0, 1.25)), (832, (0.8, 1.0, 1.25)),
                       (640, (1.0, 1.25, 1.5, 1.75))]:
            kw = dict(KW); kw['scales'] = sc
            s, _ = score(m, vids=[vid], imgsz=sz, batch=BATCH, **kw)
            print(f'  infer {sz} scales={sc} -> {s:.4f}', flush=True)


if __name__ == '__main__':
    main()
