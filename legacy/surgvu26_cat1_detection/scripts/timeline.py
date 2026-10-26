import json, collections, numpy as np
rows=json.load(open('data/tools_rows.json'))
COCO=['grasping retractor','cadiere forceps','bipolar forceps','force bipolar','clip applier','stapler',
      'permanent cautery hook spatula','monopolar curved scissor','vessel sealer','tip up fenestrated grasper',
      'bipolar dissector','needle driver','prograsp forceps','suction irrigator']
MAP={'grasping retractor':0,'cadiere forceps':1,'bipolar forceps':2,'force bipolar':3,'clip applier':4,
     'stapler':5,'permanent cautery hook/spatula':6,'monopolar curved scissors':7,'vessel sealer':8,
     'tip-up fenestrated grasper':9,'bipolar dissector':10,'needle driver':11,'prograsp forceps':12,
     'suction irrigator':13}
CAMERA={'nan(camera in)'}
OTHER={'synchroseal','curved scissors','tenaculum forceps','crocodile grasper','potts scissors',''}
stat=collections.Counter()
for r in rows:
    g=(r['gt'] or '').strip()
    if g in MAP: stat['mapped']+=1
    elif g in CAMERA: stat['camera']+=1
    elif g in OTHER: stat['other/unknown']+=1
    else: stat['UNMAPPED:'+g]+=1
print(dict(stat))
# per-case per-part duration of usable intervals
cases=collections.defaultdict(list)
for r in rows: cases[r['case']].append(r)
n_ok=n_inv=n_xpart=0
for c,rs in cases.items():
    for r in rs:
        if r['ip']!=r['up']: n_xpart+=1
        elif r['ut']<=r['it']: n_inv+=1
        else: n_ok+=1
print(f'usable intervals={n_ok} inverted={n_inv} cross-part={n_xpart}')
# how many distinct tool types simultaneously installed - the key for unambiguous pseudo-labels
