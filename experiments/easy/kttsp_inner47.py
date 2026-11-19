"""Targeted 47 search: insert inner-orbit tomatoes with time/exception DP.
Place next to beam_search.py. Reads validated 46 path; never overwrites it.
"""
import json, math, time
from pathlib import Path
from functools import lru_cache
from beam_search import transfer_dv_cached, tomatoes

ROOT=Path('results'); ROOT.mkdir(exist_ok=True)
SOURCE=ROOT/'KTTSP_46_DAY183_VALIDATED.json'
seed=json.loads(SOURCE.read_text(encoding='utf8'))
base=seed['path']; old=seed['transfers']; missing=sorted(set(range(len(tomatoes)))-set(base))
assert len(base)==46 and missing==[4,11,17]
T0=time.monotonic(); LIMIT=300.; TMAX=200.; NORMAL=100.; MAXDV=600.; MAXEX=5
# Time grids include exact existing leg times, plus half/quarter day windows.
TOFS=(.125,.25,.375,.5,.75,1.,1.5,2.,2.5,3.,3.5,4.,5.,6.,7.,9.,12.)
WAITS=(0.,.125,.25,.5,.75,1.,1.5,2.,3.,4.,6.)
@lru_cache(maxsize=450000)
def dv(i,j,dep,tof):
    try:
        v=float(transfer_dv_cached(i,j,float(dep),float(tof)))
        return v if math.isfinite(v) else math.inf
    except (ValueError,RuntimeError,OverflowError,ZeroDivisionError): return math.inf

def legs(i,j,arr,oldleg=None):
    """Keep alternatives in arrival-time buckets; not only earliest leg."""
    out=[]; seen=set()
    if oldleg is not None and oldleg[0]==i and oldleg[1]==j and oldleg[2]>=arr-1e-8:
        out.append(tuple(oldleg));seen.add((oldleg[2],oldleg[3]))
    # Departure grid from current arrival, rounding both down/up to 1/8 day
    for dep0 in (math.ceil((arr-1e-9)*8)/8, math.ceil((arr-1e-9)*4)/4):
        for wait in WAITS:
            dep=round(dep0+wait,6)
            if dep<arr-1e-8 or dep>=TMAX:continue
            for tof in TOFS:
                end=dep+tof
                if end>TMAX+1e-8 or (dep,tof) in seen:continue
                seen.add((dep,tof));v=dv(i,j,dep,tof)
                if v<=MAXDV+1e-8:out.append((i,j,dep,tof,v))
    # Keep earliest 2 per 0.5-day arrival bucket, separately normal/exception
    bins={}
    for leg in out:
        ex=int(leg[4]>NORMAL+1e-8);end=leg[2]+leg[3]
        key=(ex,int(end*2+1e-7))
        bins.setdefault(key,[]).append(leg)
    chosen=[]
    for group in bins.values():
        group.sort(key=lambda z:(z[2]+z[3],z[4]))
        chosen.extend(group[:2])
    return chosen

def search_at(pos,target):
    route=base[:pos]+[target]+base[pos:]
    # Reuse exact prefix up to predecessor of inserted node
    prefix=old[:max(0,pos-1)]
    t=prefix[-1][2]+prefix[-1][3] if prefix else 0.
    e=sum(x[4]>NORMAL+1e-8 for x in prefix)
    # States (exceptions, time bucket) => (arrival, transfer list)
    states={(e,int(t*4+1e-7)):(t,list(prefix))}
    far=pos
    for k in range(max(0,pos-1),len(route)-1):
        if time.monotonic()-T0>LIMIT:return None,far
        i,j=route[k:k+2]
        oldleg=old[k-1] if k>=pos+1 and k-1<len(old) else None
        new={}
        for (used,_),(arr,ts) in states.items():
            for leg in legs(i,j,arr,oldleg):
                ex=int(leg[4]>NORMAL+1e-8)
                if used+ex>MAXEX:continue
                end=leg[2]+leg[3]; key=(used+ex,int(end*4+1e-7))
                if key not in new or end<new[key][0]:new[key]=(end,ts+[leg])
        if not new:return None,far
        # Global beam with diversity in exception count and time; cap 100
        ranked=sorted(new.items(),key=lambda x:(x[0][0],x[1][0]))
        # retain up to 18 states per exception count
        counts={};selected={}
        for key,val in ranked:
            ex=key[0]
            if counts.get(ex,0)>=18:continue
            selected[key]=val;counts[ex]=counts.get(ex,0)+1
        states=selected;far=k+2
    (_, _),(arr,ts)=min(states.items(),key=lambda z:(z[1][0],z[0][0]))
    result={'path':route,'transfers':ts,'arrival':arr,'exceptions':sum(z[4]>NORMAL+1e-8 for z in ts),'verified':False,'method':'inner_targeted_DP'}
    return result,far

print('START | 46/49 day 183.75 | missing',missing,flush=True)
# Prioritize positions next to inner-orbit tomatoes, then all others
inner=[k for k,x in enumerate(base) if tomatoes[x][0]<5e6] if False else [k for k,x in enumerate(base) if x in (16,23,27,32,34)]
positions=list(range(1,len(base)+1))
positions.sort(key=lambda p:min(abs(p-k) for k in inner))
trials=0;best_reach=0
for target in missing:
    for pos in positions:
        if time.monotonic()-T0>LIMIT:break
        trials+=1
        result,reach=search_at(pos,target)
        best_reach=max(best_reach,reach)
        if result is not None and len(result['path'])==47:
            # Recompute all stored values to catch accidental mismatches
            ts=result['transfers'];prev=0.;exc=0;ok=len(ts)==46
            for idx,(i,j,dep,tof,v) in enumerate(ts):
                vv=dv(i,j,dep,tof);exc+=vv>100+1e-8
                ok &= i==result['path'][idx] and j==result['path'][idx+1] and dep>=prev-1e-8 and tof>=.001 and dep+tof<=200+1e-8 and abs(v-vv)<.001 and vv<=600+1e-8
                prev=dep+tof
            ok &= exc<=5 and prev<=200
            if ok:
                path=ROOT/'KTTSP_47_INNER_CANDIDATE.json'
                path.write_text(json.dumps(result,indent=2),encoding='utf8')
                print('FOUND 47/49 | arrival',round(prev,4),'| exceptions',exc,'| SAVED',path,flush=True)
                raise SystemExit
    print('Checked target',target,'| trials',trials,'| furthest partial',best_reach,'| seconds',int(time.monotonic()-T0),flush=True)
print('NO 47 FOUND | trials',trials,'| max reached',best_reach,'| seconds',int(time.monotonic()-T0),flush=True)
