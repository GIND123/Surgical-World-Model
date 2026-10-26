import os, sys, json, argparse, numpy as np, torch, time
sys.path.insert(0,'scripts')
from ds import build_fold, load_val
from evalcoco import coco_gt, evaluate
from common import NAMES
os.environ.setdefault('YOLO_VERBOSE','False')
from ultralytics import YOLO
ROOT='/mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image/work'

def predict_video(model, v, imgsz, conf=0.001, augment=False, half=True, batch=32, maxdet=30):
    A=load_val(); keys=sorted(int(k) for k in A[str(v)]['ann'])
    paths=[f'{ROOT}/data/val_masked/v{v}/{i:06d}.jpg' for i in keys]
    dets=[]
    for s in range(0,len(paths),batch):
        chunk=paths[s:s+batch]
        rs=model.predict(chunk, imgsz=imgsz, conf=conf, iou=0.7, augment=augment, half=half,
                         verbose=False, device=0, max_det=maxdet)
        for p,r in zip(chunk,rs):
            i=int(os.path.basename(p)[:-4]); iid=v*100000+i
            b=r.boxes
            if b is None or len(b)==0: continue
            xy=b.xyxy.cpu().numpy(); cf=b.conf.cpu().numpy(); cl=b.cls.cpu().numpy().astype(int)
            for (x1,y1,x2,y2),s_,c_ in zip(xy,cf,cl):
                dets.append(dict(image_id=iid,category_id=int(c_),
                                 bbox=[float(x1),float(y1),float(x2-x1),float(y2-y1)],score=float(s_)))
    return dets

def run(tag, model_name, epochs, imgsz, batch, folds=range(1,8), extra=None, augment_eval=False, workers=6, save_period=-1, **kw):
    res={}; alldets=[]
    for v in folds:
        tr=[x for x in range(1,8) if x!=v]
        yaml=build_fold(f'{tag}_f{v}', tr, [v], extra_img_dirs=extra)
        m=YOLO(model_name)
        t=time.time()
        m.train(data=yaml, epochs=epochs, imgsz=imgsz, batch=batch, device=0, workers=workers, cache='ram',
                project=f'{ROOT}/runs/{tag}', name=f'f{v}', exist_ok=True, verbose=False, val=False,
                plots=False, save_period=save_period, seed=0, **kw)
        best=YOLO(f'{ROOT}/runs/{tag}/f{v}/weights/last.pt')
        d=predict_video(best, v, imgsz, augment=augment_eval)
        alldets+=d
        r=evaluate(coco_gt([v]), d)
        res[v]=r
        print(f'[{tag}] fold v{v}: mAP={r["mAP"]:.4f} mAP50={r["mAP50"]:.4f} ({time.time()-t:.0f}s) '
              f'{ {k:round(x,3) for k,x in r["per_class"].items()} }', flush=True)
    pooled=evaluate(coco_gt(list(folds)), alldets)
    print(f'\n[{tag}] POOLED LOVO: mAP={pooled["mAP"]:.4f} mAP50={pooled["mAP50"]:.4f} mAP75={pooled["mAP75"]:.4f}')
    print('  per-class:', {k:round(x,3) for k,x in pooled['per_class'].items()})
    print('  mean-of-folds mAP=%.4f'%np.mean([res[v]['mAP'] for v in folds]))
    json.dump({'folds':{str(k):v for k,v in res.items()},'pooled':pooled}, open(f'{ROOT}/runs/{tag}_result.json','w'), indent=1)
    return pooled
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--tag',default='base'); ap.add_argument('--model',default='yolo11m.pt')
    ap.add_argument('--epochs',type=int,default=80); ap.add_argument('--imgsz',type=int,default=640)
    ap.add_argument('--batch',type=int,default=48); ap.add_argument('--folds',default='1,2,3,4,5,6,7'); ap.add_argument('--save_period',type=int,default=-1)
    a=ap.parse_args()
    run(a.tag,a.model,a.epochs,a.imgsz,a.batch,[int(x) for x in a.folds.split(',')],save_period=a.save_period)
