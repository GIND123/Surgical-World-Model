import json, pickle, collections, numpy as np
NAMES=['grasping retractor','cadiere forceps','bipolar forceps','force bipolar','clip applier','stapler',
       'permanent cautery hook spatula','monopolar curved scissor','vessel sealer','tip up fenestrated grasper',
       'bipolar dissector','needle driver','prograsp forceps','suction irrigator']
iv=pickle.load(open('data/intervals.pkl','rb'))
dur={tuple([k.split('|')[0], int(k.split('|')[1])]):v for k,v in json.load(open('data/durations.json')).items()}
ARMS=['USM1','USM2','USM3','USM4']
GRID=1.0
# per (case,part): arm_state[n,4] with values: -3 empty, -1 camera, -2 other, 0..13 class ; -4 = CONFLICT
out={}
conf_tot=0; cell_tot=0
ndist=collections.Counter(); nvis=collections.Counter()
tot_c=np.zeros(14); una_c=np.zeros(14); pair_c=np.zeros(14)
for (case,part),lst in iv.items():
    T=dur[(case,part)]; n=int(T//GRID)
    if n<=0: continue
    st=np.full((n,4),-3,dtype=np.int8)
    for t0,t1,arm,cls in lst:
        if arm not in ARMS: continue
        ai=ARMS.index(arm)
        a=max(0,int(t0//GRID)); b=min(n,int(np.ceil(t1/GRID)))
        if b<=a: continue
        cur=st[a:b,ai]
        conflict=(cur!=-3)&(cur!=cls)
        st[a:b,ai]=np.where(cur==-3, cls, np.where(conflict, -4, cur))
    conf_tot+=int((st==-4).sum()); cell_tot+=st.size
    out[(case,part)]=st
    # stats on instrument (0..13) cells only
    inst=[(st==c) for c in range(14)]
    P=np.stack([x.sum(1) for x in inst],1)          # n x 14 counts
    nd=(P>0).sum(1); nv=P.sum(1)
    bad=(st==-4).any(1)|(st==-2).any(1)
    for k,v in collections.Counter(nd[~bad].tolist()).items(): ndist[k]+=v
    for k,v in collections.Counter(nv[~bad].tolist()).items(): nvis[k]+=v
    tot_c += (P>0).sum(0)
    una_c += (P[(nd==1)&(~bad)]>0).sum(0)
    pair_c += (P[(nd<=2)&(nd>=1)&(~bad)]>0).sum(0)
pickle.dump({f'{k[0]}|{k[1]}':v for k,v in out.items()}, open('data/armstate.pkl','wb'))
print(f'conflict cells {conf_tot}/{cell_tot} = {100*conf_tot/cell_tot:.2f}%')
print('distinct instrument classes per second:', dict(sorted(ndist.items())))
print('num instruments installed per second :', dict(sorted(nvis.items())))
print(f'\n{"class":34s} {"inst_h":>8s} {"1class_h":>9s} {"<=2class_h":>10s}')
for i,nm in enumerate(NAMES):
    print(f'{nm:34s} {tot_c[i]/3600:8.2f} {una_c[i]/3600:9.2f} {pair_c[i]/3600:10.2f}')
