import av, time, glob, random
root='/mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image'
vids=sorted(glob.glob(f'{root}/surgvu24_videos_only/surgvu24/*/*.mp4'))
p=vids[0]
c=av.open(p); s=c.streams.video[0]
s.codec_context.skip_frame='NONKEY'
t=time.time(); ts=[]
for i,f in enumerate(c.decode(video=0)):
    ts.append(float(f.pts*s.time_base))
    if i>=200: break
el=time.time()-t
c.close()
import numpy as np
d=np.diff(ts)
print(f'{p.split("/")[-1]}: decoded {len(ts)} keyframes in {el:.1f}s ({len(ts)/el:.1f} kf/s)')
print(f'  keyframe interval: mean={d.mean():.2f}s median={np.median(d):.2f}s min={d.min():.2f} max={d.max():.2f}')
print(f'  covered {ts[-1]:.0f}s of video in {el:.1f}s wall  => speed {ts[-1]/el:.0f}x realtime')
