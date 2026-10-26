import av, numpy as np, glob, os, random
root='/mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image'
def grab_many(p, times):
    out=[]
    c=av.open(p); s=c.streams.video[0]
    for t in times:
        try:
            c.seek(int(t/float(s.time_base)), stream=s)
            for f in c.decode(video=0):
                out.append(f.to_ndarray(format='rgb24')); break
        except Exception: pass
    c.close(); return out
def cbox(frames, thr=12):
    m=np.zeros(frames[0].shape[:2])
    for a in frames: m=np.maximum(m, a.mean(2))
    cols=m.max(0); rows=m.max(1)
    nz=np.where(cols>thr)[0]; nzr=np.where(rows>thr)[0]
    if len(nz)==0: return None
    return (int(nz[0]), int(nz[-1]), int(nzr[0]), int(nzr[-1]))
vids=sorted(glob.glob(f'{root}/surgvu24_videos_only/surgvu24/*/*.mp4'))
random.seed(1); res={}
print("=== TRAIN pillarbox over 20 videos (max-projection of 6 timestamps) ===")
for p in random.sample(vids,20):
    fr=grab_many(p,[300,900,1500,2100,60,1200])
    if not fr: continue
    bb=cbox(fr)
    if bb is None: continue
    res[bb]=res.get(bb,0)+1
    print(f'  {os.path.basename(p):35s} x[{bb[0]},{bb[1]}] y[{bb[2]},{bb[3]}] w={bb[1]-bb[0]+1} h={bb[3]-bb[2]+1}')
print('  DISTRIBUTION:', res)
print()
print("=== TEST 640x512 content bbox (max-proj) ===")
for v in range(1,8):
    fr=grab_many(f'{root}/cat1_test_set_public/{v}_fps1.mp4',[10,50,100,150,200,250])
    print(f'  v{v}: {cbox(fr)}')
