import json, numpy as np, collections, sys
sys.path.insert(0,'scripts'); from common import NAMES
ROOT='/mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image/work'
A=json.load(open(f'{ROOT}/data/val_ann.json'))
def iou(a,b):
    ax2,ay2=a[0]+a[2],a[1]+a[3]; bx2,by2=b[0]+b[2],b[1]+b[3]
    ix=max(0,min(ax2,bx2)-max(a[0],b[0])); iy=max(0,min(ay2,by2)-max(a[1],b[1]))
    inter=ix*iy; u=a[2]*a[3]+b[2]*b[3]-inter
    return inter/u if u>0 else 0
print("=== frame-to-frame GT box IoU (same class, best match) at 1 fps ===")
allious=[]
for v in map(str,range(1,8)):
    ann=A[v]['ann']; ks=sorted(int(k) for k in ann); ious=[]
    for i in range(len(ks)-1):
        if ks[i+1]!=ks[i]+1: continue
        for c,*b in ann[str(ks[i])]:
            ious.append(max([iou(b,b2) for c2,*b2 in ann[str(ks[i+1])] if c2==c], default=0.0))
    allious+=ious
    a=np.array(ious)
    print(f'  v{v}: n={len(ious)} mean={a.mean():.3f} med={np.median(a):.3f} >0.5={np.mean(a>0.5):.3f} >0.3={np.mean(a>0.3):.3f}')
a=np.array(allious); print(f'  ALL: mean={a.mean():.3f} med={np.median(a):.3f} >0.5={np.mean(a>0.5):.3f} >0.3={np.mean(a>0.3):.3f} ==0={np.mean(a==0):.3f}')
print("\n=== per-video class coverage (frames containing the class) ===")
for v in map(str,range(1,8)):
    ann=A[v]['ann']; ks=sorted(int(k) for k in ann); out=[]
    for c in range(14):
        n=sum(1 for k in ks if any(cc==c for cc,*_ in ann[str(k)]))
        if n: out.append(f'{NAMES[c]}:{n}/{len(ks)}({100*n/len(ks):.0f}%)')
    print(f'  v{v} ({len(ks)} fr): ' + ', '.join(out))
