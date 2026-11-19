"""KTTSP 46->47+ search, starting from independently checked day-183.75 route.
Run from project root: python kttsp_46_from183.py
Requires beam_search.py and results/KTTSP_46_DAY183_VALIDATED.json.
Does not overwrite any validated file. Candidate files need independent verification.
"""
import json, math, random, time
from pathlib import Path
from functools import lru_cache
from beam_search import transfer_dv_cached, tomatoes

ROOT=Path('results')
SOURCE=ROOT/'KTTSP_46_DAY183_VALIDATED.json'
if not SOURCE.exists():
    raise SystemExit('Missing '+str(SOURCE)+'; first copy KTTSP_46_PDF_ESCAPE_CANDIDATE.json to this filename.')
base=json.loads(SOURCE.read_text(encoding='utf8'))
N=len(tomatoes); CAP=200.; NORMAL=100.; SPECIAL=600.; E=5
LIMIT=240.; rng=random.Random(20261120)
start=time.monotonic()
def elapsed():return time.monotonic()-start
def expired():return elapsed()>=LIMIT

@lru_cache(maxsize=240000)
def dv(i,j,t,dt):
    try:return float(transfer_dv_cached(int(i),int(j),round(float(t),6),round(float(dt),6)))
    except (ValueError,RuntimeError,OverflowError):return math.inf

# More granular short flights than earlier versions; preserve exact fixed prefix.
TOFS=(.125,.25,.375,.5,.75,1.,1.5,2.,2.5,3.,3.5,4.,5.,6.,7.,9.,12.,15.)
WAITS=(0.,.25,.5,1.,2.,3.,5.,7.)
@lru_cache(maxsize=35000)
def choices(i,j,bucket):
    t0=bucket*.25
    best=[None,None]
    for w in WAITS:
        t=t0+w
        if t>=CAP:continue
        for dt in TOFS:
            arr=t+dt
            if arr>CAP+1e-8:continue
            val=dv(i,j,t,dt)
            if not math.isfinite(val) or val>SPECIAL+1e-7:continue
            ex=int(val>NORMAL+1e-7)
            if best[ex] is None or arr<best[ex][0]:best[ex]=(arr,[i,j,t,dt,val])
    return best

def retime(route,keep,reflegs):
    prefix=[list(z) for z in reflegs[:keep-1]]
    t=prefix[-1][2]+prefix[-1][3] if prefix else 0.
    used=sum(z[4]>NORMAL+1e-7 for z in prefix)
    if used>E:return None
    states={used:(t,prefix)}
    reached=keep
    for k in range(keep-1,len(route)-1):
        if expired():break
        i,j=route[k:k+2]; nxt={}
        for e,(arrival,legs) in states.items():
            bucket=math.ceil((arrival-1e-9)*4)
            for extra,option in enumerate(choices(i,j,bucket)):
                if option is None or e+extra>E:continue
                end,leg=option
                if end>CAP+1e-8:continue
                ne=e+extra
                if ne not in nxt or end<nxt[ne][0]:nxt[ne]=(end,legs+[leg])
        if not nxt:break
        states={e:v for e,v in nxt.items() if not any(q<e and u[0]<=v[0] for q,u in nxt.items())}
        reached=k+2
    if not states:return None
    e,(arr,legs)=min(states.items(),key=lambda z:(z[1][0],z[0]))
    return {'path':route[:reached],'transfers':legs,'arrival':arr,'exceptions':e,'verified':False}

def check(sol):
    p=sol['path']; ts=sol['transfers']
    if len(set(p))!=len(p) or any(not 0<=x<N for x in p) or len(ts)!=len(p)-1:return False
    prev=0.;exc=0
    for k,(i,j,t,dt,val) in enumerate(ts):
        if i!=p[k] or j!=p[k+1] or t<prev-1e-7 or dt<.001 or t+dt>CAP+1e-7:return False
        actual=dv(i,j,t,dt)
        if not math.isfinite(actual) or abs(actual-val)>.001 or actual>SPECIAL+1e-7:return False
        exc+=actual>NORMAL+1e-7;prev=t+dt
    return exc<=E and abs(prev-sol['arrival'])<1e-6

def score(s):return (len(s['path']),-s['arrival'],-s['exceptions'])

best={'path':list(map(int,base['path'])),'transfers':base['transfers'],
      'arrival':float(base['arrival']),'exceptions':int(base['exceptions']),'verified':False}
if len(best['path'])!=46 or not check(best):
    raise SystemExit('Starting 46/49 record failed checks; stopping without changing files.')
print(f'Search from verified 46/{N} | day {best["arrival"]:.3f} | {LIMIT:.0f}s',flush=True)
print('Missing:',sorted(set(range(N))-set(best['path'])),flush=True)
print('Validated files are NEVER overwritten.',flush=True)
trials=0
try:
    while not expired() and len(best['path'])<N:
        trials+=1
        path=best['path']; legs=best['transfers']; n=len(path)
        missing=list(set(range(N))-set(path))
        # Usually preserve a reliable prefix, sometimes rebuild almost everything.
        keep=min(n,rng.choice([1,4,8,12,16,20,24,28,32,36,39,41,43,45,46]))
        route=path[:]
        count=min(len(missing),rng.choices([1,2,3],weights=[78,17,5])[0])
        targets=rng.sample(missing,count)
        # Reuse the mutation + retiming method that previously improved 45 to 46.
        # Try all missing nodes across early and late positions.
        if rng.random()<.42:
            suffix=route[keep:]
            if suffix:
                # Reorder a contiguous window, allowing a hard tomato to be visited earlier.
                a=rng.randrange(len(suffix))
                b=min(len(suffix),a+rng.randint(3,14))
                chunk=suffix[a:b]
                if rng.random()<.5:chunk.reverse()
                else:rng.shuffle(chunk)
                suffix[a:b]=chunk
            route=route[:keep]+suffix
        for target in targets:
            at=rng.randrange(keep,len(route)+1)
            route.insert(at,target)
            if rng.random()<.65 and len(route)-keep>=4:
                a=max(keep,at-rng.randint(0,4));b=min(len(route),at+rng.randint(2,7))
                chunk=route[a:b]
                if rng.random()<.5:chunk.reverse()
                else:rng.shuffle(chunk)
                route[a:b]=chunk
        # Sometimes move an existing target across a larger window.
        if rng.random()<.35 and len(route)-keep>=4:
            a=rng.randrange(keep,len(route));v=route.pop(a)
            b=rng.randrange(keep,len(route)+1);route.insert(b,v)
        sol=retime(route,keep,legs)
        if sol and score(sol)>score(best) and check(sol):
            best=sol
            out=ROOT/f'KTTSP_{len(best["path"])}_RESTART47_CANDIDATE.json'
            out.write_text(json.dumps(best,indent=2),encoding='utf8')
            print(f'NEW CANDIDATE {len(best["path"])}/{N} | day {best["arrival"]:.3f} | exc {best["exceptions"]} | {out}',flush=True)
        if trials%100==0:
            print(f'Trials {trials} | best {len(best["path"])}/{N} day {best["arrival"]:.3f} | {elapsed():.0f}s',flush=True)
except KeyboardInterrupt:
    print('Interrupted; saved candidates remain available.',flush=True)
print(f'Finished | trials {trials} | best {len(best["path"])}/{N} day {best["arrival"]:.3f}',flush=True)
print('New candidates require independent PyKEP and constraints validation.',flush=True)
