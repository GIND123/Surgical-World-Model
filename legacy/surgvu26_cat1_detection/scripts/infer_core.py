"""Shared inference core - identical code runs in local CV and inside the GC container."""
import numpy as np, torch, cv2

NAMES=['grasping retractor','cadiere forceps','bipolar forceps','force bipolar','clip applier','stapler',
       'permanent cautery hook spatula','monopolar curved scissor','vessel sealer','tip up fenestrated grasper',
       'bipolar dissector','needle driver','prograsp forceps','suction irrigator']
GC_NAMES=[n.replace(' ','_') for n in NAMES]
UI_TOP, UI_BOT = 32, 488
FUSE_MODE = 'box_only'
W, H = 640, 512

def mask_ui(img):
    img[:UI_TOP]=0; img[UI_BOT:]=0; return img

# ---------------------------------------------------------------- preprocessing
def prep(frames, imgsz, pad_square=False, device='cuda', half=True):
    """frames: list of HxWx3 BGR uint8 (640x512). -> tensor (B,3,h,w) plus scale/pad info."""
    sc = imgsz / max(W, H)
    nw, nh = int(round(W*sc)), int(round(H*sc))
    nw = (nw+31)//32*32; nh = (nh+31)//32*32
    th, tw = (imgsz, imgsz) if pad_square else (nh, nw)
    out=np.zeros((len(frames), th, tw, 3), dtype=np.uint8)
    dy=(th-nh)//2; dx=(tw-nw)//2
    for i,f in enumerate(frames):
        r=cv2.resize(f,(nw,nh),interpolation=cv2.INTER_LINEAR)
        out[i, dy:dy+nh, dx:dx+nw]=r
    t=torch.from_numpy(out).to(device)
    t=t[...,[2,1,0]].permute(0,3,1,2).contiguous()
    t=(t.half() if half else t.float())/255.0
    return t, (nw/W, nh/H, dx, dy)

# ---------------------------------------------------------------- raw forward
@torch.no_grad()
def forward_scores(model, t):
    """-> boxes (B,A,4) xyxy in network px, scores (B,A,nc)"""
    y=model(t)
    if isinstance(y,(list,tuple)): y=y[0]
    y=y.float().transpose(1,2)                     # (B,A,4+nc)
    xywh=y[...,:4]; sc=y[...,4:]
    cx,cy,w,h=xywh.unbind(-1)
    boxes=torch.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], -1)
    return boxes, sc

def nms_agnostic(boxes, scores_max, iou_thr):
    from torchvision.ops import nms
    return nms(boxes, scores_max, iou_thr)

def fuse(boxes, scores, smax, iou_thr, n_src):
    """Weighted box fusion: cluster by IoU around NMS survivors, then emit the
    score-weighted mean box and the MEAN class-score vector of each cluster.
    Averaging across TTA views / models sharpens both localisation and class posterior,
    whereas plain NMS throws every non-maximal view away."""
    from torchvision.ops import nms, box_iou
    keep = nms(boxes, smax, iou_thr)
    if keep.numel() == 0:
        return boxes[keep], scores[keep]
    anchors = boxes[keep]
    iou = box_iou(anchors, boxes)                        # (K, N)
    member = iou >= iou_thr
    w = (smax[None, :] * member).clamp(min=0)            # (K, N)
    wsum = w.sum(1, keepdim=True).clamp(min=1e-9)
    nb = (w[:, :, None] * boxes[None, :, :]).sum(1) / wsum
    if FUSE_MODE == "box_only":
        # average coordinates across views (sharper localisation) but keep the
        # NMS-survivor's own class posterior, so detection ranking is unchanged
        return nb, scores[keep]
    cnt = member.sum(1, keepdim=True).clamp(min=1)
    ns = (member[:, :, None] * scores[None, :, :]).sum(1) / cnt.clamp(min=1)
    if FUSE_MODE == "mean_penal":
        ns = ns * (cnt.clamp(max=n_src).float() / n_src) ** 0.25
    return nb, ns

@torch.no_grad()
def detect(models, frames, imgsz=640, pad_square=False, conf=0.005, iou=0.65, use_models=None,
           max_det=30, scales=(1.0,), flip=False, device='cuda', half=True, fusion=False):
    """Multi-model / multi-scale / flip TTA.  Returns per-frame list of
       (boxes_xyxy float32 [n,4] in 640x512 coords, class_scores [n,14])."""
    B=len(frames)
    if use_models is not None: models=[models[i] for i in use_models]
    acc_b=[[] for _ in range(B)]; acc_s=[[] for _ in range(B)]
    for sc_mult in scales:
        sz=int(round(imgsz*sc_mult/32))*32
        t,(rx,ry,dx,dy)=prep(frames, sz, pad_square, device, half)
        variants=[(t,False)] + ([(torch.flip(t,[3]),True)] if flip else [])
        for tv,is_flip in variants:
            for m in models:
                bx,scv=forward_scores(m, tv)
                if is_flip:
                    wpx=tv.shape[3]
                    x1=wpx-bx[...,2]; x2=wpx-bx[...,0]
                    bx=torch.stack([x1,bx[...,1],x2,bx[...,3]],-1)
                bx[...,[0,2]]-=dx; bx[...,[1,3]]-=dy
                bx[...,[0,2]]/=rx;  bx[...,[1,3]]/=ry
                for i in range(B):
                    acc_b[i].append(bx[i]); acc_s[i].append(scv[i])
    out=[]
    for i in range(B):
        bx=torch.cat(acc_b[i],0); sv=torch.cat(acc_s[i],0)
        smax=sv.max(1).values
        keep=smax>conf
        bx,sv,smax=bx[keep],sv[keep],smax[keep]
        if bx.numel()==0: out.append((np.zeros((0,4),np.float32), np.zeros((0,14),np.float32))); continue
        bx[:,[0,2]]=bx[:,[0,2]].clamp(0,W); bx[:,[1,3]]=bx[:,[1,3]].clamp(0,H)
        if fusion:
            nsrc=len(scales)*(2 if flip else 1)*len(models)
            nb, ns = fuse(bx, sv, smax, iou, nsrc)
            k = ns.max(1).values.argsort(descending=True)[:max_det]
            out.append((nb[k].cpu().numpy().astype(np.float32), ns[k].cpu().numpy().astype(np.float32)))
        else:
            k=nms_agnostic(bx, smax, iou)[:max_det]
            out.append((bx[k].cpu().numpy().astype(np.float32), sv[k].cpu().numpy().astype(np.float32)))
    return out
