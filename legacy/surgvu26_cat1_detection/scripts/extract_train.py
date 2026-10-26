import av, cv2, os, sys, glob, json, numpy as np
from multiprocessing import Pool
root='/mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image'
OUT=f'{root}/work/data/train_frames'
CX0,CX1,CY0,CY1=193,1087,4,717      # calibrated pillarbox crop
W,H=640,512
def job(p):
    b=os.path.basename(p); case=b[:8]; part=int(b.split('part_')[1][:3])
    d=f'{OUT}/{case}_p{part}'; os.makedirs(d,exist_ok=True)
    done=f'{d}/.done'
    if os.path.exists(done): return (b,-1)
    n=0
    try:
        c=av.open(p); s=c.streams.video[0]; s.codec_context.skip_frame='NONKEY'
        tb=float(s.time_base)
        for f in c.decode(video=0):
            t=float(f.pts*tb)
            a=f.to_ndarray(format='bgr24')[CY0:CY1, CX0:CX1]
            a=cv2.resize(a,(W,H),interpolation=cv2.INTER_AREA)
            cv2.imwrite(f'{d}/{int(round(t*1000)):09d}.jpg', a, [cv2.IMWRITE_JPEG_QUALITY,92])
            n+=1
        c.close()
    except Exception as e:
        return (b,-2,str(e))
    open(done,'w').write(str(n))
    return (b,n)
if __name__=='__main__':
    vids=sorted(glob.glob(f'{root}/surgvu24_videos_only/surgvu24/*/*.mp4'))
    os.makedirs(OUT,exist_ok=True)
    print(f'{len(vids)} videos -> {OUT}', flush=True)
    tot=0
    with Pool(20) as pool:
        for i,r in enumerate(pool.imap_unordered(job, vids)):
            if r[1]>=0: tot+=r[1]
            print(f'[{i+1}/{len(vids)}] {r}  running_total={tot}', flush=True)
    print('TOTAL FRAMES', tot)
