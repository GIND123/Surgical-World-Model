import csv, glob, os, collections, json, re
root='/mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image'
def t2s(x):
    x=str(x).strip()
    if not x or x=='nan': return None
    m=re.match(r'^(\d+):(\d+):(\d+(?:\.\d+)?)$',x)
    if m: return int(m.group(1))*3600+int(m.group(2))*60+float(m.group(3))
    try: return float(x)
    except: return None
names=collections.Counter(); comm=collections.Counter(); arms=collections.Counter()
rows_all=[]; bad=0; ncase=0
for cd in sorted(glob.glob(f'{root}/surgvu24_labels_updated_v2/labels/case_*')):
    case=os.path.basename(cd); ncase+=1
    f=os.path.join(cd,'tools.csv')
    if not os.path.exists(f): print('MISSING', case); continue
    for r in csv.DictReader(open(f)):
        gt=(r.get('groundtruth_toolname') or '').strip()
        names[gt]+=1; comm[(r.get('commercial_toolname') or '').strip()]+=1
        arms[(r.get('arm') or '').strip()]+=1
        ip,it=r.get('install_case_part'),t2s(r.get('install_case_time'))
        up,ut=r.get('uninstall_case_part'),t2s(r.get('uninstall_case_time'))
        rows_all.append(dict(case=case, ip=ip, it=it, up=up, ut=ut, arm=r.get('arm'), gt=gt, comm=r.get('commercial_toolname')))
        if it is None or ut is None: bad+=1
print(f'cases={ncase} rows={len(rows_all)} bad_times={bad}')
print(f'\n=== groundtruth_toolname ({len(names)} unique) ===')
for k,v in names.most_common(): print(f'  {v:6d}  "{k}"')
print(f'\n=== arms ===', dict(arms))
print(f'\n=== commercial_toolname ({len(comm)} unique) top 25 ===')
for k,v in comm.most_common(25): print(f'  {v:6d}  "{k}"')
# same-part & ordering sanity
same=sum(1 for r in rows_all if r['ip']==r['up']); print(f'\nsame part rows: {same}/{len(rows_all)}')
inv=sum(1 for r in rows_all if r['ip']==r['up'] and r['it'] is not None and r['ut'] is not None and r['ut']<=r['it'])
print(f'INVERTED (uninstall<=install, same part): {inv}')
json.dump(rows_all, open('data/tools_rows.json','w'))
