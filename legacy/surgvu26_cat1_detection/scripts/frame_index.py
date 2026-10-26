import os, sys, json, pickle, numpy as np, collections
sys.path.insert(0,'scripts'); from common import NAMES
ROOT='/mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image/work'
def build():
    st=pickle.load(open(f'{ROOT}/data/armstate.pkl','rb'))
    FR=f'{ROOT}/data/train_frames'
    rec=[]   # (relpath, case, part, t_sec, c1,c2,c3,c4)
    for d in sorted(os.listdir(FR)):
        if not os.path.exists(f'{FR}/{d}/.done'): continue
        case,p=d.rsplit('_p',1); part=int(p)
        key=f'{case}|{part}'
        S=st.get(key)
        if S is None: continue
        for f in sorted(os.listdir(f'{FR}/{d}')):
            if not f.endswith('.jpg'): continue
            t=int(f[:-4])/1000.0
            i=int(t)
            if i>=len(S): continue
            rec.append((f'{d}/{f}', case, part, t, *S[i].tolist()))
    return rec
if __name__=='__main__':
    rec=build()
    print('indexed frames:', len(rec))
    pickle.dump(rec, open(f'{ROOT}/data/frame_index.pkl','wb'))
    # stats
    nd=collections.Counter(); cls_frames=np.zeros(14,int); caseA=np.zeros(14,int); clean=0
    for r in rec:
        cs=np.array(r[4:8])
        if (cs==-4).any() or (cs==-2).any(): continue
        clean+=1
        inst=cs[cs>=0]
        u=set(inst.tolist()); nd[len(u)]+=1
        for c in u: cls_frames[c]+=1
        if len(u)==1: caseA[list(u)[0]]+=1
    print('clean frames (no conflict/unknown):', clean, '/', len(rec))
    print('distinct classes per frame:', dict(sorted(nd.items())))
    print(f'\n{"class":34s} {"frames":>8s} {"caseA(1cls)":>12s}')
    for i,n in enumerate(NAMES): print(f'{n:34s} {cls_frames[i]:8d} {caseA[i]:12d}')
