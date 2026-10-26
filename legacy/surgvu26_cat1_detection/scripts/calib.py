import av, numpy as np, glob, os, random, cv2
from pathlib import Path
root='/mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image'
def frames_at(p, times, fmt='gray'):
    out=[]; c=av.open(p); s=c.streams.video[0]
    for t in times:
        try:
            c.seek(int(t/float(s.time_base)), stream=s)
            for f in c.decode(video=0):
                out.append(f.to_ndarray(format='rgb24').mean(2)); break
        except Exception: pass
    c.close(); return out
# --- TEST median image ---
te=[]
for v in range(1,8):
    te += frames_at(f'{root}/cat1_test_set_public/{v}_fps1.mp4',[5,60,120,180,240,300,360])
TE=np.median(np.stack(te),0)   # 512x640
print('test median', TE.shape, len(te),'frames')
# --- TRAIN median image ---
vids=sorted(glob.glob(f'{root}/surgvu24_videos_only/surgvu24/*/*.mp4'))
random.seed(3); tr=[]
for p in random.sample(vids,25):
    tr += frames_at(p,[400,1000,1600])
TR=np.median(np.stack(tr),0)   # 720x1280
print('train median', TR.shape, len(tr),'frames')
np.save('/tmp/TE.npy',TE); np.save('/tmp/TR.npy',TR)
output_dir = Path(__file__).resolve().parents[1] / 'outputs' / 'calibration'
output_dir.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(output_dir / 'MED_TEST.png'), TE.astype(np.uint8))
cv2.imwrite(str(output_dir / 'MED_TRAIN.png'), TR.astype(np.uint8))

# --- search crop that best aligns UI ---
def score(x0,x1,y0,y1):
    crop=TR[y0:y1, x0:x1]
    r=cv2.resize(crop,(640,512),interpolation=cv2.INTER_AREA)
    a=r-r.mean(); b=TE-TE.mean()
    return float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()))
best=None
for x0 in range(186,199):
    for x1 in range(1082,1095):
        for y0 in range(0,5):
            for y1 in range(716,721):
                s=score(x0,x1,y0,y1)
                if best is None or s>best[0]: best=(s,x0,x1,y0,y1)
print('BEST NCC crop:', best, ' -> w=%d h=%d ar=%.4f'%(best[2]-best[1], best[4]-best[3], (best[2]-best[1])/(best[4]-best[3])))
print('baseline x192-1088 y0-720 ncc=', score(192,1088,0,720))
