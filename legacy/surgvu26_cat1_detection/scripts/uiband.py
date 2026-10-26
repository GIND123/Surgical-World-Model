import numpy as np, json, cv2
TE=np.load('/tmp/TE.npy')
# temporal std over test frames reveals which rows are static (UI) vs dynamic (endoscope)
import av
root='/mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image'
fr=[]
for v in range(1,8):
    c=av.open(f'{root}/cat1_test_set_public/{v}_fps1.mp4')
    for i,f in enumerate(c.decode(video=0)):
        if i%40==0: fr.append(f.to_ndarray(format='rgb24').mean(2))
        if i>400: break
    c.close()
A=np.stack(fr); print('frames',A.shape)
rowstd=A.std(0).mean(1)
print('row temporal-std profile (low = static UI):')
for y in list(range(0,40,2))+list(range(470,512,2)):
    print(f'  y={y:3d} std={rowstd[y]:6.2f}')
print('  mid y=250 std=%.2f'%rowstd[250])
# find bands
thr = rowstd[200:400].mean()*0.25
top = 0
while top<60 and rowstd[top]<thr: top+=1
bot = 511
while bot>440 and rowstd[bot]<thr: bot-=1
print(f'\nDETECTED static bands: top rows 0..{top-1}, bottom rows {bot+1}..511  (thr={thr:.2f})')

# GT overlap with those bands
tot=0; hit_top=0; hit_bot=0; ys0=[]; ys1=[]
for v in range(1,8):
    d=json.load(open(f'{root}/cat1_test_set_public/{v}_fps1_coco.json'))
    for a in d['annotations']:
        x,y,w,h=a['bbox']; tot+=1; ys0.append(y); ys1.append(y+h)
        if y < top: hit_top+=1
        if y+h > bot+1: hit_bot+=1
print(f'GT boxes total {tot}: crossing into TOP band {hit_top}, into BOTTOM band {hit_bot}')
print('GT y_min: min=%.0f p1=%.0f | y_max: p99=%.0f max=%.0f'%(min(ys0), np.percentile(ys0,1), np.percentile(ys1,99), max(ys1)))
