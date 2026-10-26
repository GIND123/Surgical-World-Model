import json, numpy as np, sys, os, io, contextlib
sys.path.insert(0,os.path.dirname(__file__)); from common import NAMES
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
ROOT='/mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image/work'
def coco_gt(vids):
    A=json.load(open(f'{ROOT}/data/val_ann.json'))
    imgs=[]; anns=[]; k=0
    for v in vids:
        for key,boxes in A[str(v)]['ann'].items():
            iid=v*100000+int(key)
            imgs.append(dict(id=iid,width=640,height=512,file_name=f'v{v}_{int(key):06d}.jpg'))
            for c,x,y,w,h in boxes:
                anns.append(dict(id=k,image_id=iid,category_id=int(c),bbox=[x,y,w,h],area=w*h,iscrowd=0)); k+=1
    d=dict(images=imgs,annotations=anns,categories=[dict(id=i,name=n) for i,n in enumerate(NAMES)])
    c=COCO(); c.dataset=d;
    with contextlib.redirect_stdout(io.StringIO()): c.createIndex()
    return c
def evaluate(gt, dets, quiet=True):
    """dets: list of dict(image_id, category_id, bbox[xywh], score)"""
    if len(dets)==0: return {'mAP':0.0,'mAP50':0.0,'per_class':{}}
    with contextlib.redirect_stdout(io.StringIO()):
        dt=gt.loadRes([dict(d) for d in dets])
        E=COCOeval(gt,dt,'bbox'); E.params.maxDets=[5,10,100]; E.evaluate(); E.accumulate(); E.summarize()
    # per-class AP@[.5:.95]
    prec=E.eval['precision']            # [T,R,K,A,M]
    per={}
    for k,cid in enumerate(gt.getCatIds()):
        p=prec[:,:,k,0,2]
        p=p[p>-1]
        if p.size: per[NAMES[cid]]=float(p.mean())
    return {'mAP':float(E.stats[0]),'mAP50':float(E.stats[1]),'mAP75':float(E.stats[2]),
            'mAP_s':float(E.stats[3]),'mAP_m':float(E.stats[4]),'mAP_l':float(E.stats[5]),
            'AR100':float(E.stats[8]),'per_class':per}
