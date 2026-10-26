import json, os, sys, shutil, cv2, numpy as np, collections
sys.path.insert(0,'scripts'); from common import NAMES, mask_ui
root='/mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image/work'
A=json.load(open(f'{root}/data/val_ann.json'))
W,H=640,512
print(f'{"video":6s} {"imgs":>5s} {"anns":>5s}  class counts')
allc=collections.Counter()
for v in map(str,range(1,8)):
    c=collections.Counter()
    for k,boxes in A[v]['ann'].items():
        for b in boxes: c[NAMES[b[0]]]+=1; allc[NAMES[b[0]]]+=1
    print(f'v{v:5s} {len(A[v]["ann"]):5d} {sum(c.values()):5d}  ' + ', '.join(f'{k}:{n}' for k,n in c.most_common()))
print('\nTOTAL:', dict(allc.most_common()))
# masked-image cache (UI blacked out) used for ALL training/eval
os.makedirs(f'{root}/data/val_masked',exist_ok=True)
for v in map(str,range(1,8)):
    d=f'{root}/data/val_masked/v{v}'; os.makedirs(d,exist_ok=True)
    for i in range(A[v]['nframes']):
        src=f'{root}/data/val_frames/v{v}/{i:06d}.jpg'
        dst=f'{d}/{i:06d}.jpg'
        if os.path.exists(dst): continue
        cv2.imwrite(dst, mask_ui(cv2.imread(src)), [cv2.IMWRITE_JPEG_QUALITY,95])
    print('masked v'+v, flush=True)
