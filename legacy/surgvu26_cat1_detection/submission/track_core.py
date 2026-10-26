"""Track linking + track-level class voting.

At 1 fps the GT boxes still overlap strongly frame-to-frame (median IoU 0.74, only
11% with zero overlap), so greedy IoU linking recovers physical instrument tracks.
A track is one physical instrument, so its class is constant: averaging the class
posterior along a track fixes per-frame classification errors, while still allowing
a class that only appears briefly (e.g. vessel sealer in 1% of frames) to survive.
"""
import numpy as np

def iou_mat(a, b):
    if len(a)==0 or len(b)==0: return np.zeros((len(a),len(b)),np.float32)
    ax1,ay1,ax2,ay2=a[:,0:1],a[:,1:2],a[:,2:3],a[:,3:4]
    bx1,by1,bx2,by2=b[:,0],b[:,1],b[:,2],b[:,3]
    ix=np.clip(np.minimum(ax2,bx2)-np.maximum(ax1,bx1),0,None)
    iy=np.clip(np.minimum(ay2,by2)-np.maximum(ay1,by1),0,None)
    inter=ix*iy
    ua=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return (inter/np.maximum(ua,1e-9)).astype(np.float32)

def link_tracks(dets, iou_thr=0.30, max_gap=2, min_score=0.05):
    """dets: list per frame of (boxes[n,4], scores[n,14]).
    Returns tracks: list of list of (frame_idx, det_idx)."""
    tracks=[]; active=[]          # active: (track_id, last_frame, last_box)
    for f,(bx,sv) in enumerate(dets):
        keep=np.where(sv.max(1)>=min_score)[0] if len(bx) else np.array([],int)
        alive=[a for a in active if f-a[1]<=max_gap]
        if len(keep) and len(alive):
            M=iou_mat(bx[keep], np.stack([a[2] for a in alive]))
            pairs=[]
            Mw=M.copy()
            while True:
                i,j=np.unravel_index(np.argmax(Mw), Mw.shape) if Mw.size else (0,0)
                if Mw.size==0 or Mw[i,j]<iou_thr: break
                pairs.append((i,j)); Mw[i,:]=-1; Mw[:,j]=-1
            used_d=set(); used_a=set()
            for i,j in pairs:
                d=keep[i]; tid=alive[j][0]
                tracks[tid].append((f,d)); used_d.add(d); used_a.add(alive[j][0])
                alive[j]=(tid,f,bx[d])
            for d in keep:
                if d in used_d: continue
                tracks.append([(f,d)]); alive.append((len(tracks)-1,f,bx[d]))
        else:
            for d in keep:
                tracks.append([(f,d)]); alive.append((len(tracks)-1,f,bx[d]))
        active=alive
    return tracks

def vote(dets, tracks, alpha=0.65, min_len=2, power=0.7):
    """Blend each detection's class posterior with its track-level average.
    alpha = weight on the track consensus. Returns new score arrays."""
    new=[s.copy() for _,s in dets]
    for tr in tracks:
        if len(tr)<min_len: continue
        P=np.stack([dets[f][1][d] for f,d in tr])          # (L,14)
        Pn=P/np.maximum(P.sum(1,keepdims=True),1e-9)
        cons=(Pn**power).mean(0)
        cons=cons/max(cons.sum(),1e-9)
        for f,d in tr:
            p=dets[f][1][d]; tot=p.sum()
            pn=p/max(tot,1e-9)
            mix=(1-alpha)*pn + alpha*cons
            new[f][d]=mix*max(tot,1e-9)
    return new

def scale_box(b, f):
    cx=(b[0]+b[2])/2; cy=(b[1]+b[3])/2; w=(b[2]-b[0])*f/2; h=(b[3]-b[1])*f/2
    return np.array([cx-w, cy-h, cx+w, cy+h], dtype=np.float32)

def emit(dets_boxes, scores, topk=3, min_conf=0.001, max_per_frame=30, box_scales=None):
    """-> list per frame of (box, class_id, score), multi-class per box.
    Cross-class false positives are free under COCO per-class AP whenever the class
    is absent from that video's GT, and recall is what limits AP here, so we emit
    the top-k class hypotheses per box rather than just the argmax."""
    out=[]
    for f,(bx,_) in enumerate(dets_boxes):
        sv=scores[f]; rows=[]
        if len(bx):
            idx=np.argsort(-sv,axis=1)[:,:topk]
            for i in range(len(bx)):
                for c in idx[i]:
                    s=float(sv[i,c])
                    if s<min_conf: continue
                    rows.append((bx[i],int(c),s))
                    if box_scales:
                        for f,pen in box_scales:
                            if s*pen>=min_conf:
                                rows.append((scale_box(bx[i],f),int(c),s*pen))
        rows.sort(key=lambda r:-r[2])
        out.append(rows[:max_per_frame])
    return out

def to_gc(per_frame, gc_names):
    boxes=[]
    for f,rows in enumerate(per_frame):
        for (x1,y1,x2,y2),c,s in rows:
            x1,y1,x2,y2=float(x1),float(y1),float(x2),float(y2)
            boxes.append({"name":f"slice_nr_{f}_{gc_names[c]}",
                          "corners":[[x1,y1,0.5],[x2,y1,0.5],[x2,y2,0.5],[x1,y2,0.5]],
                          "probability":float(s)})
    return {"name":"Regions of interest","type":"Multiple 2D bounding boxes",
            "boxes":boxes,"version":{"major":1,"minor":0}}
