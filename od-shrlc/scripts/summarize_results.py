"""Summarize result.json KPIs; each completed deployment is one observation."""
import argparse,csv,json,statistics
from collections import defaultdict
from pathlib import Path
def summarize(root):
    groups=defaultdict(list)
    for p in root.rglob('result.json'):
        r=json.loads(p.read_text(encoding='utf-8'));j=r.get('job',{})
        if not all(k in r for k in ['cost','energy','emissions','discomfort']):continue
        key=(j.get('group','single'),r.get('variant',j.get('variant','full')),r.get('scenario',j.get('scenario','apartment')))
        groups[key].append(r)
    rows=[]
    for key,values in sorted(groups.items()):
        for metric in ['cost','energy','emissions','discomfort']:
            xs=[float(r[metric]) for r in values]
            rows.append(dict(group=key[0],variant=key[1],scenario=key[2],metric=metric,n=len(xs),mean=statistics.mean(xs),sample_sd=statistics.stdev(xs) if len(xs)>1 else ''))
    return rows
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--output',type=Path,default=Path('outputs/summary.csv'));a=p.parse_args()
    if not a.root.is_dir():p.error('Result directory not found')
    rows=summarize(a.root)
    if not rows:p.error('No completed result records with the four KPIs found')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
