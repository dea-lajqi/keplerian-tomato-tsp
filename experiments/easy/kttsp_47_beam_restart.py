"""Restart beam from early verified prefixes, prioritize missing inner-orbit tomatoes.
Run next to beam_search.py; preserves all validated JSONs.
"""
import json, time, math, heapq
from pathlib import Path
from functools import lru_cache
from beam_search import transfer_dv_cached, tomatoes

ROOT=Path('results'); src=ROOT/'KTTSP_46_DAY183_VALIDATED.json'
base=json.loads(src.read_text(encoding='utf8'))
P=base['path']; L=base['transfers']; N=len(tomatoes)
assert len(P)==46 and len(L)==45 and len(set(P))==46
M=set(range(N))-set(P); assert M=={4,11,17}
T0=time.monotonic(); LIMIT=240.; WIDTH=90
TOFS=(.125,.25,.5,.75,1.,1.5,2.,2.5,3.,3.5,4.,5.,6.,7.,9.,12.)
WAITS=(0.,.25,.5,1.,2.,3.,5.,8.)
@lru_cache(maxsize=250000)
def dv(i,j,t,dt):
    try:
        z=float(transfer_dv_cached(i,j,round(t,5),dt))
        return z if math.isfinite(z) else math.inf
    except (ValueError,RuntimeError,OverflowError,ZeroDivisionError):return math.inf

@lru_cache(maxsize=120000)
def legs(i,j,bucket):
    t=bucket*.25
    best=[None,None]
    for w in WAITS:
        dep=t+w
        for dt in TOFS:
            end=dep+dt
            if end>200:continue
            v=dv(i,j,dep,dt)
            if v>600:continue
            ex=int(v>100)
            if best[ex] is None or end<best[ex][0]:
                best[ex]=(end,[i,j,dep,dt,v])
    return tuple(best)

def verify(path,ts):
    if len(set(path))!=len(path) or len(ts)!=len(path)-1:return False
    t=0.;ex=0
    for k,(i,j,dep,dt,v) in enumerate(ts):
        if (i,j)!=(path[k],path[k+1]) or dep<t-1e-7 or dt<.001 or dep+dt>200+1e-7:return False
        z=dv(i,j,dep,dt)
        if not math.isfinite(z) or abs(z-v)>.001 or z>600:return False
        ex+=z>100;t=dep+dt
    return ex<=5

def save(path,ts):
    d={'path':path,'transfers':ts,'arrival':ts[-1][2]+ts[-1][3],
       'exceptions':sum(x[4]>100 for x in ts),'verified':False,
       'method':'focused_early_beam_restart'}
    out=ROOT/f'KTTSP_{len(path)}_EARLY_BEAM_CANDIDATE.json'
    out.write_text(json.dumps(d,indent=2),encoding='utf8')
    print('NEW',len(path),'/',N,'day',round(d['arrival'],3),'missing',sorted(set(range(N))-set(path)),flush=True)

# Keep a verified prefix, then prioritize missing targets and nearby inner-orbit nodes.
# Multiple starts allow changes before the existing first three exceptions.
starts=[3,5,8,12,17,23,30]
inner={4,11,16,17,18,23,27,32,34}
record=len(P)
print('START | verified baseline 46/49 day 183.75 | 4-minute cap',flush=True)
for keep in starts:
    if time.monotonic()-T0>=LIMIT:break
    prefix=P[:keep]; ts=L[:keep-1]
    arr=ts[-1][2]+ts[-1][3] if ts else 0.
    used=sum(x[4]>100 for x in ts)
    # State: path, arrival, used, transfers. Initial prefix is valid.
    beams=[(prefix,arr,used,ts)]
    print('Restart from',keep,'tomatoes | day',round(arr,2),flush=True)
    for depth in range(keep,N):
        if time.monotonic()-T0>=LIMIT:break
        candidates=[]
        for path,arr,used,ts in beams:
            if time.monotonic()-T0>=LIMIT:break
            visited=set(path)
            remaining=[x for x in range(N) if x not in visited]
            # Try missing tomatoes first and preserve some of the old good order.
            old=[x for x in P if x not in visited]
            priority=[x for x in remaining if x in M or x in inner]
            targets=list(dict.fromkeys(priority+old[:9]+remaining))
            for j in targets:
                if time.monotonic()-T0>=LIMIT:break
                bucket=math.ceil((arr-1e-9)*4)
                for extra,opt in enumerate(legs(path[-1],j,bucket)):
                    if opt is None or used+extra>5:continue
                    end,leg=opt
                    newpath=path+[j];newts=ts+[leg]
                    if len(newpath)>record and verify(newpath,newts):
                        record=len(newpath);save(newpath,newts)
                        if record>=47:
                            print('FOUND 47; verify candidate independently before submission.',flush=True)
                            raise SystemExit(0)
                    candidates.append((newpath,end,used+extra,newts))
        if not candidates:break
        # Diversity by last node, exception count, arrival-time bin, and whether targets visited.
        buckets={}
        for s in candidates:
            path,arr,used,ts=s
            key=(path[-1],used,int(arr//4),tuple(x in path for x in (4,11,17)))
            # Prefer more missing targets reached, but retain varied timing.
            score=(arr,used)
            if key not in buckets or score<(buckets[key][1],buckets[key][2]):buckets[key]=s
        pool=list(buckets.values())
        pool.sort(key=lambda s:(-sum(x in s[0] for x in M),s[2],s[1]))
        # Ensure diversity: cap 8 states per last node and exception count
        beams=[]; counts={}
        for s in pool:
            key=(s[0][-1],s[2]);
            if counts.get(key,0)>=5:continue
            counts[key]=counts.get(key,0)+1;beams.append(s)
            if len(beams)>=WIDTH:break
        print('  depth',depth+1,'beam',len(beams),'best',record,'elapsed',round(time.monotonic()-T0),'s',flush=True)
print('DONE | best',record,'/49 | seconds',round(time.monotonic()-T0),flush=True)
