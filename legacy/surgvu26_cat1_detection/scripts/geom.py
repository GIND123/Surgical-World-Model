import av, numpy as np, glob, os, random
root='/mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image'
def grab(p, t):
    c=av.open(p); s=c.streams.video[0]
    try:
        c.seek(int(t/float(s.time_base)), stream=s)
        for f in c.decode(video=0):
            a=f.to_ndarray(format='rgb24'); c.close(); return a
    except Exception as e: pass
    c.close(); return None
def content_bbox(a, thr=12):
    g=a.mean(2)
    cols=g.max(0); rows=g.max(1)
    nz=np.where(cols>thr)[0]; nzr=np.where(rows>thr)[0]
    return (nz[0], nz[-1], nzr[0], nzr[-1])
print("=== TRAIN pillarbox (x0,x1,y0,y1) over random videos/times ===")
vids=sorted(glob.glob(f'{root}/surgvu24_videos_only/surgvu24/*/*.mp4'))
random.seed(0)
res=[]
for p in random.sample(vids,12):
    a=grab(p, 900)
    if a is None: continue
    bb=content_bbox(a); res.append(bb)
    print(f'  {os.path.basename(p):35s} {bb}  w={bb[1]-bb[0]+1} h={bb[3]-bb[2]+1} ar={(bb[1]-bb[0]+1)/(bb[3]-bb[2]+1):.4f}')
r=np.array(res); print('  MEDIAN:', np.median(r,0), ' unique:', {tuple(x) for x in res})
print()
print("=== TEST 640x512 content bbox ===")
for v in range(1,8):
    a=grab(f'{root}/cat1_test_set_public/{v}_fps1.mp4', 50)
    print(f'  v{v}: {content_bbox(a)}  shape={a.shape}')
