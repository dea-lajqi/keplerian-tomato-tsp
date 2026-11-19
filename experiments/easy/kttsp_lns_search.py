"""KTTSP large-neighborhood, time-aware segment repair.
Run from project root: python kttsp_lns_search.py
Requires beam_search.py and results/KTTSP_46_VALIDATED.json.
Never overwrites validated results. All discoveries are CANDIDATES.
"""
import json, math, random, time
from pathlib import Path
from functools import lru_cache
from beam_search import transfer_dv_cached, tomatoes

ROOT = Path('results'); ROOT.mkdir(exist_ok=True)
N = len(tomatoes)
SOURCE = ROOT / 'KTTSP_46_VALIDATED.json'
if not SOURCE.exists(): raise SystemExit(f'Missing {SOURCE}')
base = json.loads(SOURCE.read_text(encoding='utf-8'))
RNG = random.Random(2026111901)
LIMIT = 360.0
start = time.monotonic()
def elapsed(): return time.monotonic()-start

@lru_cache(maxsize=400000)
def cost(i,j,departure,tof):
    try:
        v = float(transfer_dv_cached(int(i),int(j),round(float(departure),5),round(float(tof),5)))
        return v if math.isfinite(v) else math.inf
    except (ValueError, RuntimeError, OverflowError, ZeroDivisionError):
        return math.inf

def check(s):
    p=s['path']; legs=s['transfers']
    if len(p)!=len(set(p)) or len(legs)!=len(p)-1 or any(x<0 or x>=N for x in p): return False
    prev=0.; ex=0
    for k,(i,j,dep,tof,dv) in enumerate(legs):
        if (i,j)!=(p[k],p[k+1]) or dep<prev-1e-7 or tof<.001 or dep+tof>200+1e-7: return False
        actual=cost(i,j,dep,tof)
        if actual>600+1e-7 or abs(actual-dv)>.01: return False
        ex+=actual>100+1e-7;prev=dep+tof
    return ex<=5 and abs(prev-s['arrival'])<1e-6

if not check(base): raise SystemExit('Baseline JSON failed independent transfer/constraint checks')

# Sample many arrival phases rather than retaining only the earliest per exception count.
TOFS=(.125,.25,.375,.5,.75,1.,1.5,2.,2.5,3.,3.5,4.,5.,6.,7.,9.,12.)
WAITS=(0.,.25,.5,1.,2.,3.,4.)
BEAM=9

def extend(states,i,j,deadline,remaining_legs):
    nxt=[]
    for arr,exc,legs in states:
        # Preserve exact arrival, then explore later departure phases.
        for wait in WAITS:
            dep=round(math.ceil((arr+wait-1e-8)*4)/4,5)
            if dep>deadline: continue
            for tof in TOFS:
                end=dep+tof
                if end>deadline+1e-8: continue
                v=cost(i,j,dep,tof)
                if v>600+1e-7: continue
                e=exc+int(v>100+1e-7)
                if e>5: continue
                nxt.append((end,e,legs+[[i,j,dep,tof,v]]))
    if not nxt: return []
    # Keep multiple arrival phases, grouped by exception budget, instead of just earliest.
    by={}
    for item in nxt:
        key=(item[1],int(item[0]*2)) # half-day arrival buckets
        if key not in by or item[0]<by[key][0]: by[key]=item
    candidates=list(by.values())
    candidates.sort(key=lambda z:(z[1]*1.8+z[0]/max(1,remaining_legs+1),z[0]))
    # Keep earliest arrival for each exception budget plus a diverse sample of phases.
    selected=[]; seen=set()
    for e in range(6):
        same=sorted((x for x in candidates if x[1]==e),key=lambda x:x[0])
        if same:
            selected.append(same[0]);seen.add((same[0][1],same[0][0]))
    for item in candidates:
        key=(item[1],item[0])
        if len(selected)>=BEAM:break
        if key not in seen: selected.append(item);seen.add(key)
    return selected[:BEAM]

def repair(seed,a,b,insert):
    """Replace seed path nodes a+1..b-1, keep endpoints a,b and untouched suffix."""
    p=seed['path']; old=seed['transfers']
    inner=p[a+1:b]
    rng_inner=inner[:]
    RNG.shuffle(rng_inner)
    # Sometimes retain original order, sometimes reverse a block.
    if RNG.random()<.20:rng_inner=inner[:]
    if RNG.random()<.15:rng_inner=list(reversed(inner))
    for x in insert:rng_inner.insert(RNG.randrange(len(rng_inner)+1),x)
    new_nodes=[p[a]]+rng_inner+[p[b]]
    prefix=[list(x) for x in old[:a]]
    arrival=prefix[-1][2]+prefix[-1][3] if prefix else 0.
    used=sum(x[4]>100+1e-7 for x in prefix)
    suffix=[list(x) for x in old[b:]]
    # Must rejoin before unchanged suffix's first departure; otherwise final deadline.
    deadline=suffix[0][2] if suffix else 200.
    if deadline-arrival<.001*(len(new_nodes)-1): return None
    states=[(arrival,used,[])]
    for k,(i,j) in enumerate(zip(new_nodes,new_nodes[1:])):
        states=extend(states,i,j,deadline,len(new_nodes)-2-k)
        if not states:return None
        if elapsed()>LIMIT:return None
    suffix_exc=sum(x[4]>100+1e-7 for x in suffix)
    feasible=[x for x in states if x[1]+suffix_exc<=5 and (not suffix or x[0]<=suffix[0][2]+1e-8)]
    if not feasible:return None
    arrival,ex,legs=min(feasible,key=lambda x:(x[0],x[1]))
    new_path=p[:a]+new_nodes+p[b+1:]
    new_legs=prefix+legs+suffix
    final=new_legs[-1][2]+new_legs[-1][3]
    return {'path':new_path,'transfers':new_legs,'arrival':final,'exceptions':ex+suffix_exc,'verified':False}

seeds=[base]
alt=ROOT/'KTTSP_45_DAY183_VALIDATED.json'
if alt.exists():
    try:
        s=json.loads(alt.read_text(encoding='utf-8'))
        if check(s):seeds.append(s)
    except (OSError,ValueError,KeyError,TypeError):pass
best=base
print(f'LNS | baseline {len(base["path"])}/{N} day {base["arrival"]:.3f}; {len(seeds)} seed(s); time limit {LIMIT:.0f}s',flush=True)
print('Missing:',sorted(set(range(N))-set(base['path'])),flush=True)
trials=0
while elapsed()<LIMIT and len(best['path'])<N:
    trials+=1
    seed=RNG.choice(seeds+[best]*3)
    p=seed['path']; missing=list(set(range(N))-set(p))
    if not missing:break
    # Focus on windows around special transfers, plus randomly sampled windows.
    specials=[k for k,x in enumerate(seed['transfers']) if x[4]>100+1e-7]
    if RNG.random()<.65 and specials:
        center=RNG.choice(specials)
        a=max(0,center-RNG.randint(0,4))
    else:a=RNG.randrange(0,len(p)-2)
    span=RNG.choice([2,3,4,5,6,7,8])
    b=min(len(p)-1,a+span)
    if b<=a+1:continue
    k=RNG.randint(1,min(3,len(missing)))
    inserted=RNG.sample(missing,k)
    sol=repair(seed,a,b,inserted)
    if sol and (len(sol['path']),-sol['arrival'])>(len(best['path']),-best['arrival']) and check(sol):
        best=sol;seeds.append(sol)
        seeds=sorted(seeds,key=lambda s:(-len(s['path']),s['arrival']))[:7]
        out=ROOT/f'KTTSP_{len(best["path"])}_LNS_CANDIDATE.json'
        out.write_text(json.dumps(best,indent=2),encoding='utf-8')
        print(f'NEW {len(best["path"])}/{N} | day {best["arrival"]:.3f} | exceptions {best["exceptions"]} | {out}',flush=True)
    if trials%50==0:
        print(f'Trials {trials} | best {len(best["path"])}/{N} day {best["arrival"]:.3f} | {elapsed():.0f}s',flush=True)
print(f'Finished | trials {trials} | best {len(best["path"])}/{N} day {best["arrival"]:.3f}',flush=True)
print('New outputs are CANDIDATES; verify independently before submitting.',flush=True)
