"""Score our container's surgical-tools.json with the ORIGINAL organiser code:
ref/surgvu24-category-1-eval/gc_to_coco.py + the adjust/eval logic of evaluation.py."""
import sys, json, os, contextlib, io
sys.path.insert(0,'ref/surgvu24-category-1-eval')
from gc_to_coco import GC_to_COCO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
ROOT='/mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image'

def official_score(pred_gc_path, gt_coco_path, gt_gc_path, shift=0):
    gt_gc=json.load(open(gt_gc_path))
    gt_frames={int(b['name'].split('slice_nr_')[-1].split('_')[0])-1 for b in gt_gc['boxes']}
    pred=json.load(open(pred_gc_path))
    new=[]
    for b in pred['boxes']:
        n=int(b['name'].split('slice_nr_')[-1].split('_')[0])+shift
        if n in gt_frames:
            b=dict(b); b['name']=b['name'].replace(f"slice_nr_{n-shift}_", f"slice_nr_{n}_",1)
            new.append(b)
    pred=dict(pred); pred['boxes']=new
    conv=GC_to_COCO(gc_json_fname=pred_gc_path, width=640, height=512, set_index_0=False, gc_json=pred)
    dt=conv.convert()
    with contextlib.redirect_stdout(io.StringIO()):
        gt=COCO(gt_coco_path)
        dtc=gt.loadRes([dict(d) for d in dt])
        E=COCOeval(gt,dtc,'bbox'); E.params.maxDets=[5,10,100]
        E.evaluate(); E.accumulate(); E.summarize()
    return E.stats[0], len(new)

if __name__=='__main__':
    # argv: [pred_json] [video_nr]   -- the fixture video differs per submission dir,
    # so both are overridable instead of hardcoded.
    p = sys.argv[1] if len(sys.argv)>1 else 'submission/test/output/interf0/surgical-tools.json'
    v = sys.argv[2] if len(sys.argv)>2 else '2'
    gt_coco=f'{ROOT}/cat1_test_set_public/{v}_fps1_coco.json'
    gt_gc=f'{ROOT}/cat1_test_set_public/{v}_fps1_gc.json'
    for shift in (0,+1,-1):
        s,n=official_score(p, gt_coco, gt_gc, shift)
        print(f'  slice_nr shift {shift:+d}: OFFICIAL mAP = {s:.4f}   (kept {n} boxes)')
