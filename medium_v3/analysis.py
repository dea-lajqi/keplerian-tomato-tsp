#!/usr/bin/env python3
"""Offline (without PyKEP) physics-informed diagnostics on supplied Medium data."""
import argparse,json,math,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(Path(__file__).resolve().parent))
from topology import long_transfer_analysis,proxy,geometry_cost,candidate_moves


def problem_elements(path):
    lines=Path(path).read_text().splitlines()
    rows=[];header=None
    for line in lines:
        x=line.strip()
        if x.startswith('p '):header=x.split()
        elif x and not x.startswith('c'):
            rows.append([float(a) for a in x.split()])
    if not header or len(rows)!=181:raise ValueError('Expected an original ESA 181-body medium instance')
    return rows,header


def load_sol(path):
    d=json.loads(Path(path).read_text())
    if 'solution' in d:d=d['solution']
    if 'transfers' in d:
        ts=d['transfers']
        return {'path':d['path'],'dep':[x[2] for x in ts],'tof':[x[3] for x in ts]}
    return {'path':d['path'],'dep':d['dep'],'tof':d['tof']}


def diagnostics(elements,sol):
    result=long_transfer_analysis(sol,elements)
    t=sol['tof'];d=sol['dep'];p=sol['path']
    if len(t)!=180 or len(p)!=181 or sorted(p)!=list(range(181)):
        raise ValueError('ESA Medium path counts inconsistent')
    result.update({'arrival_day':d[-1]+t[-1],
      'flight_days':sum(t),
      'waiting_days':sum(d[i+1]-d[i]-t[i] for i in range(179))+d[0],
      'geometry_proxy_total':proxy(p,elements),
      'proxy_is_not_ESA':True})
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--instance',type=Path,default=ROOT/'data/spoc4/kttsp/problems/medium.kttsp')
    p.add_argument('--solution',type=Path,default=ROOT/'results/esa/medium_20261120T080509Z_bc1f8d8f_4_verified.json')
    a=p.parse_args()
    elements,header=problem_elements(a.instance)
    sol=load_sol(a.solution)
    print(json.dumps(diagnostics(elements,sol),indent=2))

if __name__=='__main__':main()
