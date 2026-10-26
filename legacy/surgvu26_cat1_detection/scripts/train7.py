"""Final model on ALL SEVEN annotated videos, using the epoch count validated by
the video-5 holdout run.

The holdout run (runs/fin/f5) exists to choose E* honestly; it can never be the
shipped model, because video 5 carries the only ground-truth `force bipolar`
boxes in the public set -- held out, that class scores 0.097 from pseudo-labels
alone, so putting v5 back is worth real macro-mAP.

Two modes:
  scratch  retrain from the pretrained backbone for E* epochs. Identical recipe
           to the validated run, one extra video. Preferred when E* is small
           enough to fit the remaining budget.
  cont     resume from the validated checkpoint and train E more epochs at a low
           LR with almost no warmup. Cheap, but restarts the LR schedule, so use
           only when scratch does not fit.
"""
import os, sys, argparse, time
sys.path.insert(0, 'scripts')
from ds import build_fold

ROOT = '/mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image/work'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['scratch', 'cont'], required=True)
    ap.add_argument('--init', required=True, help='yolo11l.pt for scratch, else the checkpoint to continue')
    ap.add_argument('--epochs', type=int, required=True)
    ap.add_argument('--pdir', default='r3_bal')
    ap.add_argument('--tag', default='fin7')
    ap.add_argument('--imgsz', type=int, default=832)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--oversample', type=int, default=5)
    ap.add_argument('--lr0', type=float, default=0.0)
    ap.add_argument('--warmup', type=float, default=-1.0)
    ap.add_argument('--cos', type=int, default=0)
    ap.add_argument('--skip_fold', type=int, default=0)
    a = ap.parse_args()

    from ultralytics import YOLO
    p = f'{ROOT}/data/pseudo/{a.pdir}'
    t0 = time.time()
    yaml = f'{ROOT}/data/yolo/{a.tag}/data.yaml'
    if a.skip_fold and os.path.exists(yaml):
        print(f'  reusing existing fold -> {yaml}', flush=True)
    else:
        # val_vids is required by the yaml but never read: training runs with val=False.
        yaml = build_fold(a.tag, list(range(1, 8)), [5],
                          extra_img_dirs=[(f'{p}/images', f'{p}/labels')], oversample=a.oversample)
        print(f'  fold built in {time.time()-t0:.0f}s -> {yaml}', flush=True)

    kw = dict(data=yaml, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=0, workers=14,
              project=f'{ROOT}/runs/{a.tag}', name='w', exist_ok=True, verbose=False, val=False,
              plots=False, save_period=1, seed=0)
    if a.mode == 'cont':
        kw['lr0'] = a.lr0 or 0.002
        kw['warmup_epochs'] = 0.1
    elif a.lr0:
        kw['lr0'] = a.lr0
    # The 12-epoch fin run collapsed at epoch 4: a 3-epoch warmup stretched over
    # 12 epochs overshoots to lr 0.025, while the 0.5020 model never exceeded
    # 0.0151. Both are overridable so the schedule can be capped explicitly.
    if a.warmup >= 0:
        kw['warmup_epochs'] = a.warmup
    if a.cos:
        kw['cos_lr'] = True
    print(f'  {a.mode} from {a.init}: {kw}', flush=True)
    YOLO(a.init).train(**kw)
    print(f'=== TRAIN7 DONE {time.time()-t0:.0f}s ===', flush=True)


if __name__ == '__main__':
    main()
