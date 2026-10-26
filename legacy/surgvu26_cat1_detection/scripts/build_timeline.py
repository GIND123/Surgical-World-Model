import json, glob, os, collections, av, numpy as np, pickle
root='/mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image'
rows=json.load(open('data/tools_rows.json'))
MAP={'grasping retractor':0,'cadiere forceps':1,'bipolar forceps':2,'force bipolar':3,'clip applier':4,
     'stapler':5,'permanent cautery hook/spatula':6,'monopolar curved scissors':7,'vessel sealer':8,
     'tip-up fenestrated grasper':9,'bipolar dissector':10,'needle driver':11,'prograsp forceps':12,
     'suction irrigator':13}
NAMES=['grasping retractor','cadiere forceps','bipolar forceps','force bipolar','clip applier','stapler',
       'permanent cautery hook spatula','monopolar curved scissor','vessel sealer','tip up fenestrated grasper',
       'bipolar dissector','needle driver','prograsp forceps','suction irrigator']
CAMERA={'nan(camera in)'}
# --- video durations per (case, part) ---
dur={}
for p in sorted(glob.glob(f'{root}/surgvu24_videos_only/surgvu24/*/*.mp4')):
    b=os.path.basename(p); case=b[:8]; part=int(b.split('part_')[1][:3])
    c=av.open(p); s=c.streams.video[0]
    dur[(case,part)]=float(s.duration*s.time_base) if s.duration else s.frames/float(s.average_rate)
    c.close()
print('parts:', len(dur), 'total hours %.1f'%(sum(dur.values())/3600))
json.dump({f'{k[0]}|{k[1]}':v for k,v in dur.items()}, open('data/durations.json','w'))
# --- build interval list per (case,part): (t0,t1,arm,cls) cls: 0-13, -1 camera, -2 other/unknown ---
iv=collections.defaultdict(list)
skipped=0
for r in rows:
    g=(r['gt'] or '').strip()
    cls = MAP.get(g, -1 if g in CAMERA else -2)
    try: ip=int(float(r['ip'])); up=int(float(r['up']))
    except: skipped+=1; continue
    it,ut=r['it'],r['ut']
    if (case_part:=(r['case'],ip)) not in dur: skipped+=1; continue
    if ip==up:
        if ut<=it: skipped+=1; continue           # inverted -> unusable
        iv[(r['case'],ip)].append((it,ut,r['arm'],cls))
    elif up==ip+1 and (r['case'],up) in dur:      # spans into next part
        iv[(r['case'],ip)].append((it,dur[(r['case'],ip)],r['arm'],cls))
        iv[(r['case'],up)].append((0.0,ut,r['arm'],cls))
    else: skipped+=1
print('skipped rows', skipped, 'parts with intervals', len(iv))
pickle.dump(dict(iv), open('data/intervals.pkl','wb'))

# --- per-class installed-seconds, and "unambiguous" seconds (only 1 distinct instrument class installed) ---
GRID=1.0
tot_c=np.zeros(14); una_c=np.zeros(14); ndist=collections.Counter(); ninst=collections.Counter()
tot_sec=0
for (case,part),lst in iv.items():
    T=dur[(case,part)]; n=int(T//GRID)
    if n<=0: continue
    grid=np.arange(n)*GRID+0.5
    present=np.zeros((n,14),dtype=np.int16); othr=np.zeros(n,dtype=bool)
    for t0,t1,arm,cls in lst:
        a=max(0,int(t0//GRID)); b=min(n,int(np.ceil(t1/GRID)))
        if b<=a: continue
        if cls>=0: present[a:b,cls]+=1
        elif cls==-2: othr[a:b]=True
    nd=(present>0).sum(1); ni=present.sum(1)
    tot_sec+=n
    for k,v in collections.Counter(nd.tolist()).items(): ndist[k]+=v
    for k,v in collections.Counter(ni.tolist()).items(): ninst[k]+=v
    tot_c += (present>0).sum(0)
    good = (nd==1)&(~othr)
    una_c += (present[good]>0).sum(0)
print(f'\ntotal gridded seconds: {tot_sec} ({tot_sec/3600:.1f} h)')
print('distinct instrument classes installed simultaneously:', dict(sorted(ndist.items())))
print('total instruments installed simultaneously:', dict(sorted(ninst.items())))
print(f'\n{"class":34s} {"installed_h":>11s} {"unambig_h":>10s} {"unambig%":>8s}')
for i,nm in enumerate(NAMES):
    print(f'{nm:34s} {tot_c[i]/3600:11.2f} {una_c[i]/3600:10.2f} {100*una_c[i]/max(tot_c[i],1):7.1f}%')
