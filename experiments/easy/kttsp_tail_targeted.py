"""Targeted KTTSP late-segment search. Run: python kttsp_tail_targeted.py
Requires beam_search.py and results/KTTSP_46_VALIDATED.json.
No validated file is overwritten. New solutions must be independently validated.
"""
import json, math, random, time
from functools import lru_cache
from pathlib import Path
from beam_search import transfer_dv_cached, tomatoes

ROOT=Path('results'); SOURCE=ROOT/'KTTSP_46_VALIDATED.json'
if not SOURCE.exists(): raise SystemExit(f'Missing {SOURCE}')
base=json.loads(SOURCE.read_text(encoding='utf8'))
N=len(tomatoes); MAX_T=200.; E=5; NORMAL=100.; SPECIAL=600.
LIMIT=240.; rng=random.Random(460117); start=time.monotonic()
def elapsed():return time.monotonic()-start
@lru_cache(maxsize=250000)
def dv(i,j,t,dt):
    try:return float(transfer_dv_cached(int(i),int(j),float(t),float(dt)))
    except (ValueError,RuntimeError,OverflowError):return math.inf

# Half/quarter-day options plus shorter transfer times, with local waiting.
TOFS=(.125,.25,.375,.5,.75,1.,1.5,2.,2.5,3.,3.5,4.,5.,6.,7.,9.,12.)
WAITS=(0.,.25,.5,1.,2.,3.)
@lru_cache(maxsize=30000)
def choices(i,j,bucket):
    avail=bucket*.25
    out=[None,None]
    for w in WAITS:
        dep=avail+w
        for tof in TOFS:
            arr=dep+tof
            if arr>MAX_T+1e-8:continue
            value=dv(i,j,dep,tof)
            if not math.isfinite(value) or value>SPECIAL+1e-8:continue
            ex=int(value>NORMAL+1e-8)
            if out[ex] is None or arr<out[ex][0]:out[ex]=(arr,[i,j,dep,tof,value])
    return out

def solve(route,keep):
    prefix=[list(x) for x in base['transfers'][:keep-1]]
    arr=prefix[-1][2]+prefix[-1][3] if prefix else 0.
    used=sum(x[4]>NORMAL+1e-8 for x in prefix)
    states={used:(arr,prefix)}
    reached=keep
    for k in range(keep-1,len(route)-1):
        if elapsed()>LIMIT:return None
        i,j=route[k:k+2]; nxt={}
        for e,(t,legs) in states.items():
            bucket=math.ceil((t-1e-9)*4)
            for extra,item in enumerate(choices(i,j,bucket)):
                if item is None or e+extra>E:continue
                end,leg=item
                if e+extra not in nxt or end<nxt[e+extra][0]:
                    nxt[e+extra]=(end,legs+[leg])
        if not nxt:break
        states={e:v for e,v in nxt.items() if not any(q<e and other[0]<=v[0] for q,other in nxt.items())}
        reached=k+2
    if not states:return None
    e,(arr,legs)=min(states.items(),key=lambda x:(x[1][0],x[0]))
    return {'path':route[:reached],'transfers':legs,'arrival':arr,'exceptions':e,'verified':False}

def valid(sol):
    p=sol['path']; legs=sol['transfers']
    if len(set(p))!=len(p) or len(legs)!=len(p)-1:return False
    prev=0.; exc=0
    for k,(a,b,t,dt,v) in enumerate(legs):
        if a!=p[k] or b!=p[k+1] or t<prev-1e-7 or dt<.001 or t+dt>MAX_T+1e-7:return False
        actual=dv(a,b,t,dt)
        if not math.isfinite(actual) or abs(actual-v)>.01 or actual>SPECIAL+1e-7:return False
        exc+=actual>NORMAL+1e-7;prev=t+dt
    return exc<=E and abs(prev-sol['arrival'])<1e-6

best={'path':list(base['path']),'transfers':base['transfers'],'arrival':base['arrival'],'exceptions':base['exceptions']}
if len(best['path'])!=46 or not valid(best):raise SystemExit('46/49 source failed checks')
missing=sorted(set(range(N))-set(best['path']))
print('Targeted tail search | baseline 46/49 day',best['arrival'],'| missing',missing,flush=True)
print('Testing tail reorders, exception reallocation and inserting missing targets; 240s limit.',flush=True)
trials=0
while elapsed()<LIMIT and len(best['path'])<N:
    trials+=1
    # Early cut includes 13->10 special leg, later cut includes 22->18 special leg.
    keep=rng.choice([34,35,36,37,38,39,40,41,42])
    path=best['path'];keep=min(keep,len(path)-1)
    prefix=path[:keep]
    tail=path[keep:]
    # Two candidate modes: preserve most of tail, or rearrange a short window.
    route=prefix[:]
    if rng.random()<.65:
        route+=tail[:]
        if len(tail)>=2:
            a=rng.randrange(keep,len(route))
            b=min(len(route),a+rng.randint(2,6))
            if rng.random()<.5:route[a:b]=reversed(route[a:b])
            else:
                piece=route[a:b];rng.shuffle(piece);route[a:b]=piece
    else:
        route+=tail[:]
        if len(tail)>=2:
            a=rng.randrange(keep,len(route));b=rng.randrange(keep,len(route))
            route[a],route[b]=route[b],route[a]
    # Insert 1-3 missing, prioritizing insertion before the late special transfer.
    take=rng.sample(missing,rng.randint(1,len(missing)))
    for x in take:
        pos=rng.randrange(keep,len(route)+1)
        route.insert(pos,x)
    sol=solve(route,keep)
    if sol and (len(sol['path']),-sol['arrival'])>(len(best['path']),-best['arrival']) and valid(sol):
        best=sol
        out=ROOT/f'KTTSP_{len(best["path"])}_TAIL_CANDIDATE.json'
        out.write_text(json.dumps(best,indent=2),encoding='utf8')
        print(f'NEW CANDIDATE {len(best["path"])}/{N} day {best["arrival"]:.3f} exc {best["exceptions"]} | {out}',flush=True)
    if trials%100==0:
        print(f'Trials {trials} | best {len(best["path"])}/{N} day {best["arrival"]:.3f} | {elapsed():.0f}s',flush=True)
print(f'Finished | trials {trials} | best {len(best["path"])}/{N} day {best["arrival"]:.3f}',flush=True)
print('All new candidates need independent verification. Validated source unchanged.',flush=True)
