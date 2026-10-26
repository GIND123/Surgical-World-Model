"""Run the FULL container pipeline over the 7 public val videos and score it the
way the official evaluator does: mean over videos of COCOeval stats[0]."""
import os, sys, json, time, numpy as np, torch, cv2
sys.path.insert(0,os.path.dirname(__file__))
from infer_core import detect, mask_ui, NAMES, GC_NAMES, W, H
from track_core import link_tracks, vote, emit
from evalcoco import coco_gt, evaluate
ROOT='/mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image/work'

_FCACHE={}
def load_frames(v):
    """Decode the mp4 exactly as the container does (no extra JPEG round-trip)."""
    A=json.load(open(f'{ROOT}/data/val_ann.json'))
    if v not in _FCACHE:
        cap=cv2.VideoCapture(f'/mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image/cat1_test_set_public/{v}_fps1.mp4')
        fr=[]
        while True:
            ok,f=cap.read()
            if not ok: break
            if f.shape[0]!=H or f.shape[1]!=W: f=cv2.resize(f,(W,H))
            fr.append(mask_ui(f))
        cap.release(); _FCACHE[v]=fr
    return _FCACHE[v], sorted(int(k) for k in A[str(v)]['ann'])

def load_models(paths, device='cuda', half=True):
    from ultralytics import YOLO
    ms=[]
    for p in paths:
        y=YOLO(p); m=y.model.to(device).eval()
        m=m.half() if half else m.float()
        for q in m.parameters(): q.requires_grad_(False)
        ms.append(m)
    return ms

def run_video(models, frames, batch=32, **kw):
    dets=[]
    for s in range(0,len(frames),batch):
        dets+=detect(models, frames[s:s+batch], **kw)
    return dets

def score(models, vids=range(1,8), use_track=True, alpha=0.65, topk=3,
          iou_link=0.30, batch=32, verbose=True, **kw):
    per=[]; allper={}
    t0=time.time()
    for v in vids:
        fr, ann_keys = load_frames(v)
        dets=run_video(models, fr, batch=batch, **kw)
        sc=[s for _,s in dets]
        if use_track:
            tr=link_tracks(dets, iou_thr=iou_link)
            sc=vote(dets, tr, alpha=alpha)
        pf=emit(dets, sc, topk=topk)
        ak=set(ann_keys); D=[]
        for f,rows in enumerate(pf):
            if f not in ak: continue
            for (x1,y1,x2,y2),c,s in rows:
                D.append(dict(image_id=v*100000+f, category_id=c,
                              bbox=[float(x1),float(y1),float(x2-x1),float(y2-y1)], score=float(s)))
        r=evaluate(coco_gt([v]), D)
        per.append(r['mAP']); allper[v]=r
        if verbose: print(f'  v{v}: mAP={r["mAP"]:.4f} mAP50={r["mAP50"]:.4f} nfr={len(fr)} ndet={len(D)} '
                          f'{ {k:round(x,3) for k,x in r["per_class"].items()} }', flush=True)
    m=float(np.mean(per))
    if verbose: print(f'  >>> OFFICIAL-STYLE SCORE (mean per-video mAP) = {m:.4f}   [{time.time()-t0:.0f}s]')
    return m, allper
