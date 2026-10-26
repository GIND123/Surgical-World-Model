"""How much mAP is recoverable by fixing CLASSIFICATION alone?

Takes the detector's boxes as-is and replaces each box's class posterior with a
one-hot of the class of its best-matching GT box (IoU>=0.5); unmatched boxes keep
their own posterior. The resulting mAP is the ceiling a perfect crop classifier
could reach on top of the current localisation, so it says whether a dedicated
classifier is worth building at all.
"""
import sys, json, numpy as np
sys.path.insert(0, 'scripts')
from eval_pipeline import load_models, load_frames, run_video
from track_core import link_tracks, vote, iou_mat, emit
from evalcoco import coco_gt, evaluate

ROOT = '/mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image/work'


def to_dets(pf, ak, v):
    D = []
    for f, rows in enumerate(pf):
        if f not in ak:
            continue
        for (x1, y1, x2, y2), c, s in rows:
            D.append(dict(image_id=v * 100000 + f, category_id=c,
                          bbox=[float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                          score=float(s)))
    return D


def main():
    w, v = sys.argv[1], int(sys.argv[2])
    A = json.load(open(f'{ROOT}/data/val_ann.json'))
    m = load_models([w])
    fr, ak = load_frames(v)
    dets = run_video(m, fr, batch=32, imgsz=640, conf=0.001, max_det=30,
                     scales=(0.8, 1.0, 1.25, 1.5), flip=True)
    tr = link_tracks(dets, iou_thr=0.30)
    sc = vote(dets, tr, alpha=0.35)
    aks = set(ak)

    base = evaluate(coco_gt([v]), to_dets(emit(dets, sc, topk=5), aks, v))
    print(f'  actual model                : mAP={base["mAP"]:.4f}  mAP50={base["mAP50"]:.4f}')
    print('     per-class:', {k: round(x, 3) for k, x in base['per_class'].items()})

    # oracle class assignment, keeping the detector's own boxes and confidences
    orc = [s.copy() for s in sc]
    for f in ak:
        gt = A[str(v)]['ann'][str(f)]
        bx = dets[f][0]
        if len(bx) == 0 or not gt:
            continue
        g = np.array([[b[0], b[1], b[0] + b[2], b[1] + b[3]] for _, *b in gt], dtype=np.float32)
        M = iou_mat(bx, g)
        for i in range(len(bx)):
            j = int(np.argmax(M[i]))
            if M[i, j] >= 0.5:
                conf = float(orc[f][i].max())
                orc[f][i] = np.full(14, 1e-6, np.float32)
                orc[f][i][int(gt[j][0])] = conf
    oc = evaluate(coco_gt([v]), to_dets(emit(dets, orc, topk=1), aks, v))
    print(f'  ORACLE classification       : mAP={oc["mAP"]:.4f}  mAP50={oc["mAP50"]:.4f}')
    print('     per-class:', {k: round(x, 3) for k, x in oc['per_class'].items()})
    print(f'  >>> headroom from fixing classification only: {oc["mAP"] - base["mAP"]:+.4f}')


if __name__ == '__main__':
    main()
