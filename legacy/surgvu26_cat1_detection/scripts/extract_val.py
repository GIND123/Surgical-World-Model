import av, cv2, os, json, numpy as np, sys
sys.path.insert(0,'scripts'); from common import NAMES
root='/mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image'
OUT=f'{root}/work/data/val_frames'
allann={}
for v in range(1,8):
    d=f'{OUT}/v{v}'; os.makedirs(d,exist_ok=True)
    c=av.open(f'{root}/cat1_test_set_public/{v}_fps1.mp4')
    n=0
    for i,f in enumerate(c.decode(video=0)):
        cv2.imwrite(f'{d}/{i:06d}.jpg', f.to_ndarray(format='bgr24'), [cv2.IMWRITE_JPEG_QUALITY,95]); n+=1
    c.close()
    co=json.load(open(f'{root}/cat1_test_set_public/{v}_fps1_coco.json'))
    cats={x['id']:x['name'] for x in co['categories']}
    assert [cats[i] for i in range(14)]==NAMES, [cats[i] for i in range(14)]
    per={}
    for a in co['annotations']:
        per.setdefault(a['image_id'],[]).append([a['category_id']]+[float(x) for x in a['bbox']])
    allann[v]={'nframes':n,'img_ids':[im['id'] for im in co['images']],'ann':per}
    print(f'v{v}: decoded {n} frames, annotated_imgs={len(co["images"])}, anns={len(co["annotations"])}')
json.dump(allann, open(f'{root}/work/data/val_ann.json','w'))
