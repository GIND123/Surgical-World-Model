"""Assemble the final training set: merge pseudo-label sets (elimination labels win),
then oversample frames containing rare classes so the tail is not drowned out."""
import os, shutil, sys, collections, json, math
sys.path.insert(0,'scripts'); from common import NAMES
ROOT='/mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image/work'

def merge(srcs, out):
    os.makedirs(f'{out}/labels',exist_ok=True); os.makedirs(f'{out}/images',exist_ok=True)
    for src in srcs:                       # later sources overwrite earlier ones
        if not os.path.isdir(f'{src}/labels'): print('skip missing', src); continue
        for f in os.listdir(f'{src}/labels'):
            shutil.copyfile(f'{src}/labels/{f}', f'{out}/labels/{f}')
            st=f[:-4]; lp=f'{out}/images/{st}.jpg'
            if not os.path.exists(lp):
                os.symlink(os.path.realpath(f'{src}/images/{st}.jpg'), lp)
    return out

def stats(p):
    cc=collections.Counter()
    for f in os.listdir(f'{p}/labels'):
        for line in open(f'{p}/labels/{f}'): cc[int(line.split()[0])]+=1
    return cc

def balance(src, out, target=None, cap=24):
    """Duplicate (symlink) frames so每 rare class reaches ~target boxes."""
    cc=stats(src)
    if target is None: target=int(sorted(cc.values())[-4])//6   # aim at ~1/6 of a common class
    files=os.listdir(f'{src}/labels')
    byclass=collections.defaultdict(list)
    for f in files:
        cs={int(l.split()[0]) for l in open(f'{src}/labels/{f}')}
        for c in cs: byclass[c].append(f)
    os.makedirs(f'{out}/labels',exist_ok=True); os.makedirs(f'{out}/images',exist_ok=True)
    reps={}
    for c in range(14):
        reps[c]=1 if cc[c]>=target or cc[c]==0 else min(cap, max(1, round(target/max(cc[c],1))))
    # a frame's repeat count = max over the classes it contains
    frep={}
    for c,fs in byclass.items():
        for f in fs: frep[f]=max(frep.get(f,1), reps[c])
    n=0
    for f in files:
        r=frep.get(f,1); st=f[:-4]
        for k in range(r):
            sfx='' if k==0 else f'~d{k}'
            shutil.copyfile(f'{src}/labels/{f}', f'{out}/labels/{st}{sfx}.txt')
            lp=f'{out}/images/{st}{sfx}.jpg'
            if not os.path.exists(lp): os.symlink(os.path.realpath(f'{src}/images/{st}.jpg'), lp)
            n+=1
    print(f'  target={target}  repeats={ {NAMES[c]:reps[c] for c in range(14) if reps[c]>1} }')
    print(f'  frames {len(files)} -> {n}')
    return out

if __name__=='__main__':
    srcs=sys.argv[1].split(','); out=sys.argv[2]; bal=sys.argv[3]
    merge([f'{ROOT}/data/pseudo/{s}' for s in srcs], f'{ROOT}/data/pseudo/{out}')
    c0=stats(f'{ROOT}/data/pseudo/{out}')
    balance(f'{ROOT}/data/pseudo/{out}', f'{ROOT}/data/pseudo/{bal}')
    c1=stats(f'{ROOT}/data/pseudo/{bal}')
    print(f'\n{"class":32s} {"merged":>10s} {"balanced":>10s}')
    for i,n in enumerate(NAMES): print(f'{n:32s} {c0[i]:10d} {c1[i]:10d}')
