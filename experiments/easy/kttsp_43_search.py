"""KTTSP 43-target search from independently checked day-180.75 route.
Run from project root: python kttsp_43_search.py
Requires beam_search.py and results/KTTSP_42_DAY180_VALIDATED.json.
Never overwrites the baseline.
"""
import json, math, random, time
from pathlib import Path
from functools import lru_cache
from beam_search import transfer_dv_cached, tomatoes

ROOT=Path('results'); ROOT.mkdir(exist_ok=True)
SOURCE=ROOT/'KTTSP_42_DAY180_VALIDATED.json'
OUT=ROOT/'KTTSP_43_PDF_CANDIDATE.json'
NORMAL=100.; SPECIAL=600.; MAX_EXC=5; MAX_TIME=200.
TIME_LIMIT=180.
RNG=random.Random(20261118)
N=len(tomatoes)
base=json.loads(SOURCE.read_text(encoding='utf8'))
base_path=list(map(int,base['path']))
base_legs=[list(x) for x in base['transfers']]
assert len(base_path)==42 and len(base_legs)==41
missing=sorted(set(range(N))-set(base_path))
assert len(missing)==7
clock=time.monotonic()
def expired(): return time.monotonic()-clock>=TIME_LIMIT

@lru_cache(maxsize=220000)
def dv(i,j,dep,tof):
    try: return float(transfer_dv_cached(int(i),int(j),round(float(dep),6),round(float(tof),6)))
    except (ValueError,RuntimeError,OverflowError): return math.inf

# Refine the grid around a known good schedule instead of replacing it with a coarse grid.
TOFS=(.25,.5,.75,1.,1.5,2.,2.5,3.,3.5,4.,5.,6.,7.,9.,12.,15.)
WAITS=(0.,.25,.5,1.,2.,3.,5.,7.)
@lru_cache(maxsize=32000)
def options(i,j,bucket):
    available=bucket*.25
    result=[None,None]
    for wait in WAITS:
        dep=available+wait
        if dep>=MAX_TIME: continue
        for tof in TOFS:
            arr=dep+tof
            if arr>MAX_TIME+1e-9: continue
            val=dv(i,j,dep,tof)
            if val>SPECIAL+1e-8: continue
            ex=int(val>NORMAL+1e-8)
            if result[ex] is None or arr<result[ex][0]:
                result[ex]=(arr,[i,j,dep,tof,val])
    return result

def retime(path,prefix_len):
    """Keep verified prefix; DP on number of exception transfers in new suffix."""
    prefix=base_legs[:prefix_len-1]
    start=(prefix[-1][2]+prefix[-1][3]) if prefix else 0.
    used=sum(x[4]>NORMAL+1e-8 for x in prefix)
    states={used:(start,list(prefix))}
    reached=prefix_len
    for k in range(prefix_len-1,len(path)-1):
        if expired(): break
        i,j=path[k:k+2]
        new={}
        for ex,(arr,legs) in states.items():
            bucket=math.ceil((arr-1e-9)*4)
            for extra,op in enumerate(options(i,j,bucket)):
                if op is None or ex+extra>MAX_EXC: continue
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
    if len(path)!=len(set(path)) or len(legs)!=len(path)-1:return False
    prev=0.;exc=0
    for k,(i,j,dep,tof,stored) in enumerate(legs):
        if i!=path[k] or j!=path[k+1] or dep<prev-1e-7 or tof<.001 or dep+tof>MAX_TIME+1e-7:return False
        actual=dv(i,j,dep,tof)
        if not math.isfinite(actual) or abs(actual-stored)>1e-4 or actual>SPECIAL+1e-7:return False
        exc+=actual>NORMAL+1e-7
        prev=dep+tof
    return exc<=MAX_EXC and abs(prev-sol['arrival'])<1e-6

best={'path':base_path,'transfers':base_legs,'arrival':float(base['arrival']),
      'exceptions':sum(x[4]>NORMAL for x in base_legs),'verified':False}
print(f'43 search | baseline {len(base_path)}/{N} day {best["arrival"]:.2f}',flush=True)
print('Missing:',missing,'| max',TIME_LIMIT,'seconds',flush=True)
print('Baseline will NOT be overwritten.',flush=True)
trials=0
while not expired():
    trials+=1
    # Preserve early verified legs; aggressively change the late section.
    prefix_len=RNG.choice([10,15,20,25,30,33,35,37,39,40,41,42])
    prefix_len=min(prefix_len,len(base_path))
    route=base_path[:]
    # One missing tomato inserted near or after the retained prefix.
    target=RNG.choice(missing)
    pos=RNG.randrange(prefix_len,len(route)+1)
    route.insert(pos,target)
    # Allow 2-6 adjacent stops to be reordered, so inserting does not require
    # both original edges to survive unchanged.
    if RNG.random()<.85:
        a=max(prefix_len,RNG.randrange(max(prefix_len,pos-4),min(len(route),pos+1)))
        b=min(len(route),max(a+2,pos+RNG.randint(2,7)))
        if b-a>=2:
            part=route[a:b]
            RNG.shuffle(part)
            route[a:b]=part
    # Occasionally make a second local change in the suffix.
    if RNG.random()<.3 and len(route)-prefix_len>=4:
        a=RNG.randrange(prefix_len,len(route)-2)
        b=min(len(route),a+RNG.randint(2,5))
        route[a:b]=reversed(route[a:b])
    sol=retime(route,prefix_len)
    if sol is not None and (len(sol['path'])>len(best['path']) or
        (len(sol['path'])==len(best['path']) and sol['arrival']<best['arrival']-1e-6)):
        if check(sol):
            best=sol
            OUT.write_text(json.dumps(best,indent=2),encoding='utf8')
            print(f'NEW CANDIDATE {len(best["path"])}/{N} | day {best["arrival"]:.3f} | exceptions {best["exceptions"]}',flush=True)
            if len(best['path'])>=43:
                print('43 reached: candidate saved; validate independently before using.',flush=True)
                break
    if trials%50==0:
        print(f'Trials {trials} | best {len(best["path"])}/{N} | {time.monotonic()-clock:.0f}s',flush=True)
print(f'Finished | trials {trials} | best {len(best["path"])}/{N}',flush=True)
if not OUT.exists():print('No improved candidate saved. Original day-180.75 result is safe.',flush=True)
else:print('Candidate:',OUT,'(not officially verified)',flush=True)
