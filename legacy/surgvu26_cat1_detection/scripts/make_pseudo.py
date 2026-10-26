"""Generate box-level pseudo-labels on the surgvu24 training keyframes.

teacher detector -> boxes + 14-way class posterior
tools.csv        -> the multiset of instruments installed on the arms at that instant
assignment       -> best injective matching of boxes to installed instruments
                    (spatial prior: USM1..USM4 are usually left->right)
Only confident, fully-explained frames are kept.
"""
import os, sys, json, pickle, argparse, time, numpy as np, torch, cv2
sys.path.insert(0,os.path.dirname(__file__))
from infer_core import detect, mask_ui, NAMES, W, H
from pseudo import assign, frame_installed, adjust
from torch.utils.data import Dataset, DataLoader
ROOT='/mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image/work'
FR=f'{ROOT}/data/train_frames'

class FrameDS(Dataset):
    def __init__(self, rec, idxs): self.rec=rec; self.idxs=idxs
    def __len__(self): return len(self.idxs)
    def __getitem__(self, i):
        j=self.idxs[i]; r=self.rec[j]
        img=cv2.imread(f'{FR}/{r[0]}')
        if img is None: img=np.zeros((H,W,3),np.uint8)
        elif img.shape[0]!=H or img.shape[1]!=W: img=cv2.resize(img,(W,H))
        return torch.from_numpy(mask_ui(img)), j
def coll(b): return [x[0] for x in b], [x[1] for x in b]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--weights', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--batch', type=int, default=64); ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.25); ap.add_argument('--margin', type=float, default=1.0)
    ap.add_argument('--lam', type=float, default=0.35); ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--limit', type=int, default=0); ap.add_argument('--require_full', type=int, default=0)
    ap.add_argument('--sel', default='selected_frames'); ap.add_argument('--tau', type=float, default=0.0)
    ap.add_argument('--prior', default='')
    a=ap.parse_args()
    rec=pickle.load(open(f'{ROOT}/data/frame_index.pkl','rb'))
    sel=pickle.load(open(f'{ROOT}/data/{a.sel}.pkl','rb'))
    PRIOR=np.array(json.load(open(a.prior)),dtype=float) if a.prior else None
    if a.limit: sel=sel[:a.limit]
    os.makedirs(f'{a.out}/labels', exist_ok=True); os.makedirs(f'{a.out}/images', exist_ok=True)
    from ultralytics import YOLO
    m=YOLO(a.weights).model.cuda().eval().half()
    for p in m.parameters(): p.requires_grad_(False)
    dl=DataLoader(FrameDS(rec,sel), batch_size=a.batch, num_workers=a.workers,
                  collate_fn=coll, shuffle=False, pin_memory=False)
    stats=dict(seen=0, no_inst=0, nodet=0, too_many=0, low_margin=0, kept=0)
    clscnt=np.zeros(14,int); t0=time.time()
    for bi,(imgs,js) in enumerate(dl):
        frames=[x.numpy() for x in imgs]
        dets=detect([m], frames, imgsz=a.imgsz, conf=a.conf, iou=0.6, max_det=8)
        for (bx,sv), j in zip(dets, js):
            stats['seen']+=1
            r=rec[j]; fi=frame_installed(r[4:8])
            if fi is None: stats['no_inst']+=1; continue
            inst_c, inst_a = fi
            if len(bx)==0: stats['nodet']+=1; continue
            if len(bx)>len(inst_c): stats['too_many']+=1; continue
            if a.require_full and len(bx)!=len(inst_c): stats['too_many']+=1; continue
            p=sv/np.maximum(sv.sum(1,keepdims=True),1e-9)
            logp=np.log(np.clip(p,1e-8,1))
            if PRIOR is not None and a.tau>0: logp=adjust(logp, PRIOR, a.tau)
            cx=((bx[:,0]+bx[:,2])/2)/W
            res=assign(logp, inst_c, inst_a, cx, lam_spatial=a.lam)
            if res is None: stats['too_many']+=1; continue
            lbl,sc,mg=res
            if mg<a.margin: stats['low_margin']+=1; continue
            stem=r[0].replace('/','__')[:-4]
            with open(f'{a.out}/labels/{stem}.txt','w') as f:
                for (x1,y1,x2,y2),c in zip(bx,lbl):
                    cxn=(x1+x2)/2/W; cyn=(y1+y2)/2/H; wn=(x2-x1)/W; hn=(y2-y1)/H
                    if wn<=0 or hn<=0: continue
                    f.write(f'{int(c)} {cxn:.6f} {cyn:.6f} {wn:.6f} {hn:.6f}\n')
                    clscnt[int(c)]+=1
            lp=f'{a.out}/images/{stem}.jpg'
            if not os.path.exists(lp): os.symlink(f'{FR}/{r[0]}', lp)
            stats['kept']+=1
        if bi%100==0:
            el=time.time()-t0
            print(f'  [{bi*a.batch}/{len(sel)}] {el:.0f}s {stats}', flush=True)
    print('FINAL', stats)
    print('pseudo-label boxes per class:')
    for i,n in enumerate(NAMES): print(f'  {n:34s} {clscnt[i]:7d}')
    json.dump({'stats':stats,'per_class':clscnt.tolist()}, open(f'{a.out}/stats.json','w'), indent=1)
if __name__=='__main__': main()
