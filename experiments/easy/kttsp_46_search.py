"""Progressive KTTSP search: 43 -> 44 -> 45 -> 46, with safe checkpoints.
Run in project root: python kttsp_46_search.py
Requires beam_search.py and results/KTTSP_43_VALIDATED.json.
Candidate files are NOT official verification. Existing results are never overwritten.
"""
import json, math, random, time
from pathlib import Path
from functools import lru_cache
from beam_search import transfer_dv_cached, tomatoes

ROOT=Path('results'); ROOT.mkdir(exist_ok=True)
SOURCE=ROOT/'KTTSP_43_VALIDATED.json'
NORMAL,SPECIAL,MAX_EXC,MAX_TIME=100.,600.,5,200.
GOAL=46
TIME_LIMIT=360.0  # up to six minutes; Ctrl+C preserves saved checkpoints
RNG=random.Random(20261119)
N=len(tomatoes)
if not SOURCE.exists():
    raise SystemExit('Missing results/KTTSP_43_VALIDATED.json. First copy your validated 43 candidate to this name.')
base=json.loads(SOURCE.read_text(encoding='utf8'))
start_path=list(map(int,base['path']))
start_legs=[list(x) for x in base['transfers']]
assert len(start_path)==43 and len(start_legs)==42 and len(set(start_path))==43
clock=time.monotonic()
def expired():return time.monotonic()-clock>=TIME_LIMIT

@lru_cache(maxsize=220000)
def dv(i,j,dep,tof):
    try:return float(transfer_dv_cached(int(i),int(j),round(float(dep),6),round(float(tof),6)))
    except (ValueError,RuntimeError,OverflowError):return math.inf

TOFS=(.25,.5,.75,1.,1.5,2.,2.5,3.,3.5,4.,5.,6.,7.,9.,12.,15.)
WAITS=(0.,.25,.5,1.,2.,3.,5.,7.)
@lru_cache(maxsize=32000)
def options(i,j,bucket):
    available=bucket*.25
    result=[None,None]
    for wait in WAITS:
        dep=available+wait
        if dep>=MAX_TIME:continue
        for tof in TOFS:
            arr=dep+tof
            if arr>MAX_TIME+1e-9:continue
            val=dv(i,j,dep,tof)
            if not math.isfinite(val) or val>SPECIAL+1e-8:continue
            ex=int(val>NORMAL+1e-8)
            if result[ex] is None or arr<result[ex][0]:
                result[ex]=(arr,[i,j,dep,tof,val])
    return result

def retime(path,prefix_len,reference_legs):
    prefix=[list(x) for x in reference_legs[:prefix_len-1]]
    start=(prefix[-1][2]+prefix[-1][3]) if prefix else 0.
    used=sum(x[4]>NORMAL+1e-8 for x in prefix)
    states={used:(start,prefix)}
    reached=prefix_len
    for k in range(prefix_len-1,len(path)-1):
        if expired():break
        i,j=path[k:k+2]
        new={}
        for ex,(arr,legs) in states.items():
            bucket=math.ceil((arr-1e-9)*4)
            for extra,op in enumerate(options(i,j,bucket)):
                if op is None or ex+extra>MAX_EXC:continue
                end,leg=op
                key=ex+extra
                if key not in new or end<new[key][0]:new[key]=(end,legs+[leg])
        if not new:break
        states={e:v for e,v in new.items() if not any(q<e and u[0]<=v[0] for q,u in new.items())}
        reached=k+2
    if not states:return None
    e,(arr,legs)=min(states.items(),key=lambda kv:(kv[1][0],kv[0]))
    return {'path':path[:reached],'transfers':legs,'arrival':arr,'exceptions':e,'verified':False}

def check(sol):
    path=sol['path'];legs=sol['transfers']
    if len(path)!=len(set(path)) or any(not 0<=i<N for i in path) or len(legs)!=len(path)-1:return False
    prev=0.;exc=0
    for k,(i,j,dep,tof,stored) in enumerate(legs):
        if i!=path[k] or j!=path[k+1] or dep<prev-1e-7 or tof<.001 or dep+tof>MAX_TIME+1e-7:return False
        actual=dv(i,j,dep,tof)
        if not math.isfinite(actual) or abs(actual-stored)>1e-4 or actual>SPECIAL+1e-7:return False
        exc+=actual>NORMAL+1e-7
        prev=dep+tof
    return exc<=MAX_EXC and abs(prev-sol['arrival'])<1e-6

def score(sol):return (len(sol['path']),-sol['arrival'],-sol['exceptions'])

best={'path':start_path,'transfers':start_legs,'arrival':float(base['arrival']),
      'exceptions':sum(x[4]>NORMAL for x in start_legs),'verified':False}
# The best path becomes a new starting point for subsequent iterations.
print(f'Progressive search | baseline {len(best["path"])}/{N} day {best["arrival"]:.2f} | goal {GOAL}/{N}',flush=True)
print(f'Time limit: {TIME_LIMIT:.0f}s | Missing: {sorted(set(range(N))-set(start_path))}',flush=True)
print('Saved records are never overwritten; new checkpoints require independent validation.',flush=True)
trials=0
try:
    while not expired() and len(best['path'])<GOAL:
        trials+=1
        # Most trials start from the current best; some return to the validated 43 to escape a trap.
        ref=best if RNG.random()<.78 else {'path':start_path,'transfers':start_legs}
        path=list(ref['path']); legs=ref['transfers']
        missing=list(set(range(N))-set(path))
        if not missing:break
        # Search from earlier prefixes too: a late-only insertion may be impossible.
        n=len(path)
        prefix_len=RNG.choice([1,5,10,15,20,25,30,34,37,39,41,42,43,max(1,n-4),n])
        prefix_len=min(prefix_len,n)
        route=path[:]
        # Usually insert one target, sometimes two or three (can cross a difficult barrier).
        count=RNG.choices([1,2,3],weights=[75,20,5])[0]
        count=min(count,len(missing))
        targets=RNG.sample(missing,count)
        for target in targets:
            pos=RNG.randrange(prefix_len,len(route)+1)
            route.insert(pos,target)
            if RNG.random()<.85:
                lo=max(prefix_len,pos-4)
                hi=min(len(route)-1,pos+3)
                if lo<=hi:
                    a=RNG.randrange(lo,hi+1)
                    b=min(len(route),max(a+2,pos+RNG.randint(2,7)))
                    if b-a>=2:
                        chunk=route[a:b];RNG.shuffle(chunk);route[a:b]=chunk
        if RNG.random()<.4 and len(route)-prefix_len>=4:
            a=RNG.randrange(prefix_len,len(route)-2)
            b=min(len(route),a+RNG.randint(2,7))
            route[a:b]=reversed(route[a:b])
        sol=retime(route,prefix_len,legs)
        if sol is not None and score(sol)>score(best) and check(sol):
            best=sol
            # Never overwrite previously validated or candidate files; each count has a distinct output.
            out=ROOT/f'KTTSP_{len(best["path"])}_PROGRESS_CANDIDATE.json'
            # Same count is overwritten only if strictly improved in this run.
            out.write_text(json.dumps(best,indent=2),encoding='utf8')
            print(f'NEW CANDIDATE {len(best["path"])}/{N} | day {best["arrival"]:.3f} | exc {best["exceptions"]} | {out}',flush=True)
        if trials%50==0:
            print(f'Trials {trials} | best {len(best["path"])}/{N} | {time.monotonic()-clock:.0f}s',flush=True)
except KeyboardInterrupt:
    print('\nStopped by user; saved checkpoints remain on disk.',flush=True)
print(f'Finished | trials {trials} | best {len(best["path"])}/{N} day {best["arrival"]:.3f}',flush=True)
print('Any new candidate must pass the independent PyKEP and constraint checks before being called verified.',flush=True)
