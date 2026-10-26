import json, os, sys, numpy as np
sys.path.insert(0,os.path.dirname(__file__)); from common import NAMES
ROOT='/mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image/work'
W,H=640,512
def load_val():  return json.load(open(f'{ROOT}/data/val_ann.json'))
def write_labels(base):
    """Write YOLO .txt labels for every annotated val frame once."""
    A=load_val()
    for v in map(str,range(1,8)):
        d=f'{base}/v{v}'; os.makedirs(d,exist_ok=True)
        for k,boxes in A[v]['ann'].items():
            with open(f'{d}/{int(k):06d}.txt','w') as f:
                for c,x,y,w,h in boxes:
                    f.write(f'{int(c)} {(x+w/2)/W:.6f} {(y+h/2)/H:.6f} {w/W:.6f} {h/H:.6f}\n')
def build_fold(name, train_vids, val_vids, extra_img_dirs=None, oversample=1):
    A=load_val(); lab=f'{ROOT}/data/val_labels'
    root=f'{ROOT}/data/yolo/{name}'
    for sp,vids in (('train',train_vids),('val',val_vids)):
        di=f'{root}/{sp}/images'; dl=f'{root}/{sp}/labels'
        os.makedirs(di,exist_ok=True); os.makedirs(dl,exist_ok=True)
        for f in os.listdir(di): os.unlink(f'{di}/{f}')
        for f in os.listdir(dl): os.unlink(f'{dl}/{f}')
        reps = oversample if sp=='train' else 1
        for r in range(reps):
            sfx = '' if r==0 else f'_r{r}'
            for v in vids:
                for k in A[str(v)]['ann']:
                    i=int(k); s=f'{ROOT}/data/val_masked/v{v}/{i:06d}.jpg'
                    os.symlink(s, f'{di}/v{v}_{i:06d}{sfx}.jpg')
                    os.symlink(f'{lab}/v{v}/{i:06d}.txt', f'{dl}/v{v}_{i:06d}{sfx}.txt')
    if extra_img_dirs:
        di=f'{root}/train/images'; dl=f'{root}/train/labels'
        n=0
        for imgd,labd in extra_img_dirs:
            for f in sorted(os.listdir(labd)):
                stem=f[:-4]
                src=f'{imgd}/{stem}.jpg'
                if not os.path.exists(src): continue
                os.symlink(src, f'{di}/x_{stem}.jpg'); os.symlink(f'{labd}/{f}', f'{dl}/x_{stem}.txt'); n+=1
        print(f'  + {n} extra pseudo-labelled images')
    y=f"path: {root}\ntrain: train/images\nval: val/images\nnames:\n" + \
      "".join(f"  {i}: {n}\n" for i,n in enumerate(NAMES))
    open(f'{root}/data.yaml','w').write(y)
    return f'{root}/data.yaml'
