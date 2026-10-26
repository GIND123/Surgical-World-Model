import sys, json, numpy as np, collections
sys.path.insert(0,'scripts')
from eval_pipeline import load_models, load_frames, run_video
from track_core import link_tracks, vote, iou_mat
from infer_core import NAMES
ROOT='/mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image/work'
w=sys.argv[1]; vids=[int(x) for x in sys.argv[2].split(',')]
m=load_models([w])
A=json.load(open(f'{ROOT}/data/val_ann.json'))
for v in vids:
    fr,ak=load_frames(v)
    dets=run_video(m,fr,batch=32,imgsz=640,conf=0.005,max_det=30,scales=(1.0,1.25),flip=True)
    tr=link_tracks(dets, iou_thr=0.30); sc=vote(dets,tr,alpha=0.65)
    C=np.zeros((14,15),int)   # last col = missed (no box with IoU>=0.5)
    loc=collections.Counter()
    for f in ak:
        gt=A[str(v)]['ann'][str(f)]
        bx=dets[f][0]; sv=sc[f]
        if len(bx)==0:
            for c,*b in gt: C[c,14]+=1
            continue
        g=np.array([[b[0],b[1],b[0]+b[2],b[1]+b[3]] for _,*b in gt],dtype=np.float32)
        M=iou_mat(g,bx)
        for i,(c,*b) in enumerate(gt):
            j=int(np.argmax(M[i]))
            if M[i,j]<0.5: C[c,14]+=1; continue
            loc['localised']+=1
            C[c, int(np.argmax(sv[j]))]+=1
    print(f'\n=== video {v}: GT class -> predicted class of best-IoU>=0.5 box ({loc["localised"]} localised) ===')
    for c in range(14):
        if C[c].sum()==0: continue
        row=[(NAMES[k],C[c,k]) for k in range(14) if C[c,k]>0]
        row.sort(key=lambda x:-x[1])
        print(f'  GT {NAMES[c]:28s} n={C[c].sum():5d} missed={C[c,14]:5d} | ' +
              ', '.join(f'{n}:{v_}' for n,v_ in row))
