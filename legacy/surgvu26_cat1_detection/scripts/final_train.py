"""Build the submission model: teacher on ALL public-val videos -> pseudo-labels
over the 280 training videos -> final student on everything."""
import os, sys, json, time, argparse, subprocess, shutil, collections
sys.path.insert(0,'scripts')
from ds import build_fold
ROOT='/mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image/work'
PY=f'{ROOT}/env/bin/python'
ALL=[1,2,3,4,5,6,7]

def sh(cmd):
    print('  $ '+' '.join(cmd), flush=True); subprocess.run(cmd, check=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage', default='all')
    ap.add_argument('--tmodel', default='yolo11s.pt'); ap.add_argument('--tepochs', type=int, default=25)
    ap.add_argument('--model', default='yolo11s.pt'); ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--imgsz', type=int, default=640); ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--oversample', type=int, default=5); ap.add_argument('--name', default='final_s')
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--pdir', default='final_mix')
    a=ap.parse_args()
    from ultralytics import YOLO

    # ---------- 1. teacher on all 7 videos ----------
    tdir=f'{ROOT}/runs/teacher_all/w/weights/last.pt'
    if a.stage in ('all','teacher') and not os.path.exists(tdir):
        y=build_fold('teacher_all', ALL, [1])          # val unused (val=False)
        YOLO(a.tmodel).train(data=y, epochs=a.tepochs, imgsz=a.imgsz, batch=a.batch, device=0,
                             workers=8, cache='ram', project=f'{ROOT}/runs/teacher_all', name='w',
                             exist_ok=True, verbose=False, val=False, plots=False, save_period=-1, seed=0)
        print('teacher done', flush=True)

    # ---------- 2. pseudo-labels (volume + elimination-balanced), merged ----------
    A=f'{ROOT}/data/pseudo/final_a'; B=f'{ROOT}/data/pseudo/final_b'; M=f'{ROOT}/data/pseudo/'+a.pdir
    if a.stage in ('all','pseudo'):
        if not os.path.exists(f'{A}/stats.json'):
            sh([PY,f'{ROOT}/scripts/make_pseudo.py','--weights',tdir,'--out',A,
                '--conf','0.25','--require_full','0','--margin','1.0','--batch','96','--workers','12'])
        if not os.path.exists(f'{B}/stats.json'):
            sh([PY,f'{ROOT}/scripts/make_pseudo.py','--weights',tdir,'--out',B,
                '--conf','0.05','--require_full','1','--margin','1.0','--batch','96','--workers','12'])
        if not os.path.exists(f'{M}/.done'):
            os.makedirs(f'{M}/labels',exist_ok=True); os.makedirs(f'{M}/images',exist_ok=True)
            for src in (A,B):
                for f in os.listdir(f'{src}/labels'):
                    shutil.copyfile(f'{src}/labels/{f}', f'{M}/labels/{f}')
                    st=f[:-4]; lp=f'{M}/images/{st}.jpg'
                    if not os.path.exists(lp): os.symlink(os.path.realpath(f'{src}/images/{st}.jpg'), lp)
            open(f'{M}/.done','w').write('1')
        cc=collections.Counter()
        for f in os.listdir(f'{M}/labels'):
            for line in open(f'{M}/labels/{f}'): cc[int(line.split()[0])]+=1
        print(f'merged pseudo: {len(os.listdir(f"{M}/labels"))} frames, {sum(cc.values())} boxes', flush=True)

    # ---------- 3. final student ----------
    if a.stage in ('all','student'):
        y=build_fold(a.name, ALL, [1], extra_img_dirs=[(f'{M}/images', f'{M}/labels')], oversample=a.oversample)
        YOLO(a.model).train(data=y, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=0, workers=16,
                            project=f'{ROOT}/runs/{a.name}', name='w', exist_ok=True, verbose=False,
                            val=False, plots=False, save_period=-1, seed=a.seed)
        print(f'FINAL MODEL: {ROOT}/runs/{a.name}/w/weights/last.pt', flush=True)
if __name__=='__main__': main()
