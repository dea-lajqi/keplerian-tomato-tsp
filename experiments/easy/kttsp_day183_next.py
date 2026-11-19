"""KTTSP 47+ escape: PDF-inspired Or-opt/2-opt, exception DP and suffix re-timing.
Run beside beam_search.py. Does not overwrite validated JSON files.
"""
import json, math, random, time
from pathlib import Path
from functools import lru_cache
from beam_search import transfer_dv_cached, tomatoes

ROOT=Path('results'); N=len(tomatoes); LIMIT=300.; rng=random.Random(47183)
SOURCES=[ROOT/'KTTSP_46_DAY183_VALIDATED.json',ROOT/'KTTSP_46_VALIDATED.json',ROOT/'KTTSP_45_DAY183_VALIDATED.json']
seeds=[json.loads(p.read_text(encoding='utf8')) for p in SOURCES if p.exists()]
if not seeds:raise SystemExit('Missing results/KTTSP_46_DAY183_VALIDATED.json')
START=time.monotonic(); NORMAL=100.; MAXDV=600.; MAXEX=5; TMAX=200.
TOFS=(.125,.25,.375,.5,.75,1.,1.5,2.,2.5,3.,3.5,4.,5.,6.,7.,9.,12.)
WAITS=(0.,.25,.5,1.,2.,3.,5.,7.,10.)
@lru_cache(maxsize=500000)
def dv(i,j,t,dt):
    try:
        x=float(transfer_dv_cached(int(i),int(j),float(t),float(dt)))
        return x if math.isfinite(x) else math.inf
    except (ValueError,RuntimeError,OverflowError,ZeroDivisionError):return math.inf

@lru_cache(maxsize=150000)
def choices(i,j,bucket):
    # PDF principle: earliest-arrival queries with an explicit normal/exception split.
    t0=bucket*.25; best=[None,None]
    for wait in WAITS:
        dep=t0+wait
        for tof in TOFS:
            end=dep+tof
            if end>TMAX+1e-8:continue
            v=dv(i,j,dep,tof)
            if v>MAXDV+1e-7:continue
            exc=int(v>NORMAL+1e-7)
            if best[exc] is None or end<best[exc][0]:best[exc]=(end,[i,j,dep,tof,v])
    return best

def solve(route,seed,keep):
    old=seed['transfers']; prefix=[list(z) for z in old[:keep-1]]
    t=prefix[-1][2]+prefix[-1][3] if prefix else 0.
    used=sum(z[4]>NORMAL+1e-7 for z in prefix)
    if used>MAXEX:return None
    states={used:(t,prefix)}; reached=keep
    for k in range(keep-1,len(route)-1):
        if time.monotonic()-START>=LIMIT:return None
        i,j=route[k:k+2]; nxt={}
        for exc,(arr,legs) in states.items():
            bucket=math.ceil((arr-1e-9)*4)
            for extra,option in enumerate(choices(i,j,bucket)):
                if option is None or exc+extra>MAXEX:continue
                end,leg=option; ne=exc+extra
                if ne not in nxt or end<nxt[ne][0]:nxt[ne]=(end,legs+[leg])
        if not nxt:break
        states={e:s for e,s in nxt.items() if not any(e2<e and s2[0]<=s[0] for e2,s2 in nxt.items())}
        reached=k+2
    if not states:return None
    exc,(arr,legs)=min(states.items(),key=lambda x:(x[1][0],x[0]))
    return {'path':route[:reached],'transfers':legs,'arrival':arr,'exceptions':exc,'verified':False}

def valid(s):
    p=s['path']; ls=s['transfers']; t=0.; ex=0
    if len(set(p))!=len(p) or len(ls)!=len(p)-1:return False
    for k,(i,j,dep,tof,v) in enumerate(ls):
        if i!=p[k] or j!=p[k+1] or dep<t-1e-7 or tof<.001 or dep+tof>TMAX+1e-7:return False
        actual=dv(i,j,dep,tof)
        if actual>MAXDV+1e-7 or abs(actual-v)>.001:return False
        ex+=actual>NORMAL+1e-7;t=dep+tof
    return ex<=MAXEX and abs(t-s['arrival'])<1e-6

for s in seeds:
    if not valid(s):raise SystemExit('Starting seed failed independent transfer validation')
archive=list(seeds); best=max(seeds,key=lambda x:(len(x['path']),-x['arrival']))
print('DAY183 SEARCH | best',len(best['path']),'/',N,'| missing',sorted(set(range(N))-set(best['path'])),flush=True)
print('Moves: Or-opt, 2-opt, swap, insert missing; re-time suffix with exception DP.',flush=True)
trials=0; last=0
try:
    while time.monotonic()-START<LIMIT and len(best['path'])<N:
        trials+=1
        seed=rng.choice(archive[-12:] if rng.random()<.65 else seeds)
        p=seed['path']; n=len(p); missing=list(set(range(N))-set(p))
        # Focus near existing exception legs and at early points to free time/exceptions.
        special=[k for k,z in enumerate(seed['transfers']) if z[4]>NORMAL]
        if rng.random()<.25 and special:
            focus=rng.choice(special); keep=max(1,focus-rng.randint(0,9))
        else:keep=rng.choice([1,2,5,9,13,17,21,25,29,33,37,41,44])
        keep=min(keep,n)
        route=p[:]; op=rng.randrange(5)
        # Preserve prefix, but reconstruct all downstream timings (not fixed suffix).
        if op==0 and n-keep>=2: # Or-opt: relocate 1..3 old tomatoes
            a=rng.randrange(keep,n);size=min(rng.randint(1,3),n-a)
            block=route[a:a+size];del route[a:a+size]
            b=rng.randrange(keep,len(route)+1);route[b:b]=block
        elif op==1 and n-keep>=3: # 2-opt
            a=rng.randrange(keep,n-1);b=min(n,a+rng.randint(2,12));route[a:b]=reversed(route[a:b])
        elif op==2 and n-keep>=2:
            a,b=rng.sample(range(keep,n),2);route[a],route[b]=route[b],route[a]
        elif op==3 and n-keep>=3:
            a=rng.randrange(keep,n);b=min(n,a+rng.randint(3,12));rng.shuffle(route[a:b]) # handled below
            chunk=route[a:b];rng.shuffle(chunk);route[a:b]=chunk
        # Always insert one missing tomato; sometimes two or all three.
        rng.shuffle(missing)
        count=1
        for target in missing[:count]:
            at=rng.randrange(keep,len(route)+1);route.insert(at,target)
        # A little directed 2-opt around an inserted tomato.
        if rng.random()<.4 and len(route)-keep>=3:
            a=rng.randrange(keep,len(route)-1);b=min(len(route),a+rng.randint(2,7))
            route[a:b]=reversed(route[a:b])
        s=solve(route,seed,keep)
        if s and valid(s):
            archive.append(s)
            if len(archive)>50:
                archive=sorted(archive,key=lambda z:(len(z['path']),-z['arrival']),reverse=True)[:25]
            if (len(s['path']),-s['arrival'])>(len(best['path']),-best['arrival']):
                best=s
                out=ROOT/f'KTTSP_{len(best["path"])}_DAY183_NEXT_CANDIDATE.json'
                out.write_text(json.dumps(best,indent=2),encoding='utf8')
                print('NEW',len(best['path']),'/',N,'day',round(best['arrival'],3),'exceptions',best['exceptions'],'saved',out,flush=True)
                print('Missing:',sorted(set(range(N))-set(best['path'])),flush=True)
        if trials%100==0:
            print('Trials',trials,'| best',len(best['path']),'/',N,'|',int(time.monotonic()-START),'sec',flush=True)
except KeyboardInterrupt:print('Stopped by user; saved candidates remain.',flush=True)
print('Finished | best',len(best['path']),'/',N,'|',round(best['arrival'],3),'days',flush=True)
print('Candidates must be checked separately before official submission.',flush=True)
