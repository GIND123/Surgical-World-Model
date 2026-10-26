import numpy as np, json, av
root='/mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image'
print("=== per-video temporal std of rows (static UI = low std WITHIN a video) ===")
prof=[]
for v in range(1,8):
    fr=[]; c=av.open(f'{root}/cat1_test_set_public/{v}_fps1.mp4')
    for i,f in enumerate(c.decode(video=0)):
        if i%15==0: fr.append(f.to_ndarray(format='rgb24').mean(2))
    c.close()
    A=np.stack(fr); rs=A.std(0).mean(1); prof.append(rs)
    band=np.where(rs < rs[100:400].mean()*0.20)[0]
    print(f'  v{v} nfr={len(fr)}  low-std rows: {band.min() if len(band) else "-"}..{band.max() if len(band) else "-"}  count={len(band)}')
P=np.mean(prof,0)
ref=P[100:400].mean()
print('\n mean row-std (rel to interior):')
for y in range(0,512,4):
    bar='#'*int(40*P[y]/ref)
    mark=' <== STATIC' if P[y]<0.35*ref else ''
    if y<48 or y>460: print(f'  y={y:3d} {P[y]/ref:5.2f} {bar}{mark}')
lowtop=[y for y in range(0,60) if P[y]<0.35*ref]
lowbot=[y for y in range(440,512) if P[y]<0.35*ref]
print('\n TOP static rows:', lowtop)
print(' BOTTOM static rows:', lowbot)
# GT boxes overlapping candidate bands
for TOP,BOT in [(28,481),(30,480),(26,482)]:
    n_t=n_b=0; tot=0
    for v in range(1,8):
        d=json.load(open(f'{root}/cat1_test_set_public/{v}_fps1_coco.json'))
        for a in d['annotations']:
            x,y,w,h=a['bbox']; tot+=1
            if y<TOP: n_t+=1
            if y+h>BOT: n_b+=1
    print(f' band(top<{TOP}, bot>{BOT}): boxes crossing top={n_t} bottom={n_b} of {tot}')
