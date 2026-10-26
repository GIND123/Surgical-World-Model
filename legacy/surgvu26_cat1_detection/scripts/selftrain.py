"""Round-2: per-fold teacher -> presence-constrained pseudo-labels -> student.

Fold v's teacher never saw video v, so the pseudo-labels it produces carry no
information about the held-out video and the LOVO estimate stays honest.
"""
import os, sys, json, time, argparse, subprocess, numpy as np
sys.path.insert(0,'scripts')
from ds import build_fold
from evalcoco import coco_gt, evaluate
ROOT='/mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image/work'
PY=f'{ROOT}/env/bin/python'

def gen_pseudo(teacher, out, limit=0, conf=0.25, margin=1.0, batch=64, require_full=0):
    if os.path.exists(f'{out}/stats.json'):
        print(f'  pseudo exists: {out}'); return
    cmd=[PY,f'{ROOT}/scripts/make_pseudo.py','--weights',teacher,'--out',out,
         '--conf',str(conf),'--margin',str(margin),'--batch',str(batch),
         '--require_full',str(require_full)]
    if limit: cmd+=['--limit',str(limit)]
    print('  '+' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--tag', default='st1'); ap.add_argument('--teacher_tag', default='base_m640')
    ap.add_argument('--model', default='yolo11m.pt'); ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--imgsz', type=int, default=640); ap.add_argument('--batch', type=int, default=48)
    ap.add_argument('--folds', default='1,2,3,4,5,6,7')
    ap.add_argument('--plimit', type=int, default=0); ap.add_argument('--pconf', type=float, default=0.25)
    ap.add_argument('--pmargin', type=float, default=1.0); ap.add_argument('--require_full', type=int, default=0); ap.add_argument('--oversample', type=int, default=1); ap.add_argument('--pdir', default=''); ap.add_argument('--save_period', type=int, default=-1)
    a=ap.parse_args()
    from train_cv import predict_video
    from ultralytics import YOLO
    folds=[int(x) for x in a.folds.split(',')]
    res={}
    for v in folds:
        t0=time.time()
        teacher=f'{ROOT}/runs/{a.teacher_tag}/f{v}/weights/last.pt'
        pdir=f'{ROOT}/data/pseudo/{a.pdir}_f{v}' if a.pdir else f'{ROOT}/data/pseudo/{a.tag}_f{v}'
        if not a.pdir: gen_pseudo(teacher, pdir, a.plimit, a.pconf, a.pmargin, require_full=a.require_full)
        tr=[x for x in range(1,8) if x!=v]
        yaml=build_fold(f'{a.tag}_f{v}', tr, [v], extra_img_dirs=[(f'{pdir}/images', f'{pdir}/labels')], oversample=a.oversample)
        m=YOLO(a.model)
        m.train(data=yaml, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=0, workers=14,
                project=f'{ROOT}/runs/{a.tag}', name=f'f{v}', exist_ok=True, verbose=False, val=False,
                plots=False, save_period=a.save_period, seed=0)
        best=YOLO(f'{ROOT}/runs/{a.tag}/f{v}/weights/last.pt')
        d=predict_video(best, v, a.imgsz)
        r=evaluate(coco_gt([v]), d); res[v]=r
        print(f'[{a.tag}] fold v{v}: mAP={r["mAP"]:.4f} mAP50={r["mAP50"]:.4f} ({time.time()-t0:.0f}s) '
              f'{ {k:round(x,3) for k,x in r["per_class"].items()} }', flush=True)
    print(f'\n[{a.tag}] MEAN PER-VIDEO mAP = {np.mean([res[v]["mAP"] for v in folds]):.4f}')
    json.dump({str(k):v for k,v in res.items()}, open(f'{ROOT}/runs/{a.tag}_result.json','w'), indent=1)
if __name__=='__main__': main()
