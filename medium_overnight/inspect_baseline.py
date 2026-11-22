"""Fast source-integrity check requiring no PyKEP; does not claim ESA revalidation."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
BASE=ROOT/'results/esa/medium_20261120T080509Z_bc1f8d8f_4_verified.json'
ORBIT=ROOT/'data/spoc4/kttsp/problems/medium.kttsp'

def main():
    obj=json.loads(BASE.read_text(encoding='utf-8'))
    orbital=[s.strip() for s in ORBIT.read_text().splitlines() if s.strip() and not s.startswith('c')]
    meta=orbital[0].split()
    orbitrows=[list(map(float,s.split())) for s in orbital[1:]]
    n=len(orbitrows);path=obj['path'];ts=obj['transfers']
    assert meta[:2]==['p','kttsp'],meta
    assert n==181 and len(path)==n and len(set(path))==n and len(ts)==n-1
    assert all(t[0]==path[i] and t[1]==path[i+1] for i,t in enumerate(ts))
    assert all(t[2]+t[3]<=ts[i+1][2]+1e-8 for i,t in enumerate(ts[:-1]))
    assert abs(ts[-1][2]+ts[-1][3]-obj['arrival_day'])<1e-7
    ex=sum(t[4]>100 for t in ts)
    assert ex<=5 and all(t[4]<=600+1e-6 for t in ts)
    crossings=sum((orbitrows[ts[i][0]][0]>8e6)!=(orbitrows[ts[i][1]][0]>8e6) for i in range(n-1))
    print('SOURCE INSPECTION (stored ΔV values, not fresh PyKEP calculation)')
    print('Header:',meta,'tomatoes:',n)
    print('ESA saved validated arrival:',obj['arrival_day'])
    print('Flight days:',sum(t[3] for t in ts))
    print('Waiting days:',obj['arrival_day']-sum(t[3] for t in ts)-ts[0][2])
    print('Exception legs:',ex,'radius-shell crossings:',crossings)
    print('Source structure, chronology and saved ΔV constraints: OK')
    print('Run campaign.py verify-best using PyKEP for independent ESA revalidation.')

if __name__=='__main__':main()
