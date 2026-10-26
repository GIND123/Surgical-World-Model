"""Pick a class-balanced subset of the 725k training keyframes for pseudo-labelling.
Rarest classes first so their (few) frames are never crowded out."""
import pickle, numpy as np, sys, collections
sys.path.insert(0,'scripts'); from common import NAMES
ROOT='/mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image/work'
rec=pickle.load(open(f'{ROOT}/data/frame_index.pkl','rb'))
QUOTA={ 'default':120000 }
clean=[]
for i,r in enumerate(rec):
    cs=np.array(r[4:8])
    if (cs==-4).any() or (cs==-2).any(): continue
    inst=cs[cs>=0]
    if len(inst)==0: continue
    clean.append((i, frozenset(inst.tolist()), len(inst)))
print('clean frames with >=1 instrument:', len(clean))
bycls=collections.defaultdict(list)
for i,s,n in clean:
    for c in s: bycls[c].append(i)
order=sorted(range(14), key=lambda c: len(bycls[c]))
chosen=set(); got=collections.Counter()
rng=np.random.default_rng(0)
for c in order:
    q=QUOTA.get(NAMES[c], QUOTA['default'])
    have=sum(1 for i in chosen if c in dict((x[0],x[1]) for x in [])) # placeholder
    pool=[i for i in bycls[c] if i not in chosen]
    cur=got[c]
    need=max(0, q-cur)
    if need>0 and pool:
        take=pool if len(pool)<=need else list(rng.choice(pool, need, replace=False))
        for i in take:
            chosen.add(i)
            for cc in dict(((x[0],x[1]) for x in [])) or []: pass
        # recount contributions
        for i in take:
            cs=np.array(rec[i][4:8]);
            for cc in set(cs[cs>=0].tolist()): got[cc]+=1
    print(f'  {NAMES[c]:34s} pool={len(bycls[c]):7d} selected_now={got[c]:7d}')
sel=sorted(chosen)
print(f'\nSELECTED {len(sel)} frames')
print('per-class coverage in selection:')
cov=collections.Counter()
for i in sel:
    cs=np.array(rec[i][4:8])
    for c in set(cs[cs>=0].tolist()): cov[c]+=1
for c in range(14): print(f'  {NAMES[c]:34s} {cov[c]:7d}')
pickle.dump(sel, open(f'{ROOT}/data/selected_frames_xl.pkl','wb'))
