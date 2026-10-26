"""Cache raw detections once per video, then sweep postprocessing cheaply."""
import sys, os, json, pickle, numpy as np, itertools
sys.path.insert(0,'scripts')
from eval_pipeline import load_models, load_frames, run_video
from track_core import link_tracks, vote, emit
from evalcoco import coco_gt, evaluate
ROOT='/mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image/work'

def cache_dets(weights, vids, tag, **kw):
    out={}
    for v in vids:
        cp=f'{ROOT}/data/detcache/{tag}_v{v}.pkl'
        os.makedirs(f'{ROOT}/data/detcache',exist_ok=True)
        if os.path.exists(cp): out[v]=pickle.load(open(cp,'rb')); continue
        m=load_models([weights(v)] if callable(weights) else [weights])
        fr,ak=load_frames(v)
        d=run_video(m, fr, batch=32, **kw)
        pickle.dump((d,ak), open(cp,'wb')); out[v]=(d,ak)
        print(f'  cached v{v}: {len(d)} frames', flush=True)
        del m
        import torch; torch.cuda.empty_cache()
    return out

def eval_cfg(cache, use_track=True, alpha=0.65, iou_link=0.30, topk=3,
             min_conf=0.001, max_per_frame=30, box_scales=None):
    per=[]
    for v,(dets,ak) in cache.items():
        sc=[s for _,s in dets]
        if use_track:
            tr=link_tracks(dets, iou_thr=iou_link); sc=vote(dets,tr,alpha=alpha)
        pf=emit(dets, sc, topk=topk, min_conf=min_conf, max_per_frame=max_per_frame, box_scales=box_scales)
        aks=set(ak); D=[]
        for f,rows in enumerate(pf):
            if f not in aks: continue
            for (x1,y1,x2,y2),c,s in rows:
                D.append(dict(image_id=v*100000+f, category_id=c,
                              bbox=[float(x1),float(y1),float(x2-x1),float(y2-y1)], score=float(s)))
        per.append(evaluate(coco_gt([v]), D)['mAP'])
    return float(np.mean(per)), per
