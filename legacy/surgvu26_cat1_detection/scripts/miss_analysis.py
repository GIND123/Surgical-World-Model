import sys, json, numpy as np, collections
sys.path.insert(0,'scripts')
from eval_pipeline import load_models, load_frames, run_video
from track_core import link_tracks, vote, iou_mat
from infer_core import NAMES
ROOT='/mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image/work'
m=load_models([sys.argv[1]]); v=int(sys.argv[2])
A=json.load(open(f'{ROOT}/data/val_ann.json'))
fr,ak=load_frames(v)
dets=run_video(m,fr,batch=32,imgsz=640,conf=0.005,max_det=30,scales=(1.0,1.25),flip=True)
tr=link_tracks(dets,iou_thr=0.30); sc=vote(dets,tr,alpha=0.65)
per=collections.defaultdict(lambda: dict(best=[], area=[], areamiss=[], y=[], ymiss=[]))
for f in ak:
    gt=A[str(v)]['ann'][str(f)]; bx=dets[f][0]
    g=np.array([[b[0],b[1],b[0]+b[2],b[1]+b[3]] for _,*b in gt],dtype=np.float32)
    M=iou_mat(g,bx) if len(bx) else np.zeros((len(g),0))
    for i,(c,*b) in enumerate(gt):
        bi=float(M[i].max()) if M.shape[1] else 0.0
        d=per[c]; d['best'].append(bi); d['area'].append(b[2]*b[3]); d['y'].append(b[1]+b[3]/2)
        if bi<0.5: d['areamiss'].append(b[2]*b[3]); d['ymiss'].append(b[1]+b[3]/2)
print(f'=== video {v}: best-IoU achieved per GT box (class-agnostic matching) ===')
for c,d in sorted(per.items()):
    b=np.array(d['best']); a=np.array(d['area']); am=np.array(d['areamiss'])
    print(f'  {NAMES[c]:28s} n={len(b):5d} meanIoU={b.mean():.3f} | IoU=0:{np.mean(b==0):.2f} '
          f'<0.1:{np.mean(b<0.1):.2f} 0.1-0.5:{np.mean((b>=0.1)&(b<0.5)):.2f} >=0.5:{np.mean(b>=0.5):.2f}')
    print(f'      area: all med={np.median(a):7.0f}  missed med={np.median(am) if len(am) else -1:7.0f}   '
          f'ctr-y: all med={np.median(d["y"]):.0f} missed med={np.median(d["ymiss"]) if d["ymiss"] else -1:.0f}')
