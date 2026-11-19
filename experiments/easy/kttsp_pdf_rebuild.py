"""Reconstruction of key PDF solver ideas for existing KTTSP project.
Not a byte-for-byte reproduction of the PDF. Keeps verified baseline intact.
Run: python kttsp_pdf_rebuild.py
"""
import json, math, random, time
from pathlib import Path
from functools import lru_cache
from beam_search import transfer_dv_cached, tomatoes

ROOT = Path('results'); ROOT.mkdir(exist_ok=True)
SOURCE = ROOT / 'KTTSP_42_verified.json'
OUT = ROOT / 'KTTSP_PDF_REBUILD_CANDIDATE.json'
MAX_TIME = 200.0
TIME_LIMIT = 180.0
NORMAL, SPECIAL, MAX_EXC = 100., 600., 5
RNG = random.Random(20261118)
N = len(tomatoes)

with SOURCE.open(encoding='utf8') as f:
    baseline = json.load(f)
base_path = [int(i) for i in baseline['path']]
base_legs = [list(t) for t in baseline['transfers']]
if len(base_legs) != len(base_path)-1:
    raise ValueError('Baseline path/transfers mismatch')

# PDF: multi-start construction + exception-state DP + local moves.
# Retiming uses a small Pareto frontier over arrival and exception count.
# Unlike the printed PDF's full-permutation-only solver, this also accepts partial paths.
TOFS = (0.25, 0.5, 0.75, 1., 1.5, 2., 2.5, 3., 3.5, 4., 5., 6., 7., 9., 12.)
WAITS = (0., 0.25, 0.5, 1., 2., 3., 5., 7., 10.)
clock = time.monotonic()

def expired(): return time.monotonic() - clock >= TIME_LIMIT

@lru_cache(maxsize=150000)
def cost(i,j,dep,tof):
    try:
        return float(transfer_dv_cached(int(i),int(j),round(dep,5),round(tof,5)))
    except (ValueError, RuntimeError, OverflowError):
        return math.inf

@lru_cache(maxsize=12000)
def options(i,j,available_bucket):
    """Earliest arrival normal/special from a 0.25-day snapped available time."""
    available = available_bucket * 0.25
    best = [None,None]
    for w in WAITS:
        dep = available+w
        for tof in TOFS:
            arr=dep+tof
            if arr > MAX_TIME: continue
            dv=cost(i,j,dep,tof)
            if not math.isfinite(dv) or dv>SPECIAL: continue
            ex=int(dv>NORMAL)
            if best[ex] is None or arr < best[ex][0]:
                best[ex]=(arr,[i,j,dep,tof,dv])
    return best

def schedule(path, keep_prefix=0):
    """DP with at most five exception legs; fixed original prefix can be retained."""
    if keep_prefix:
        prefix = base_legs[:keep_prefix-1]
        initial_arr = prefix[-1][2]+prefix[-1][3] if prefix else 0.
        initial_exc = sum(x[4]>NORMAL for x in prefix)
    else:
        prefix=[];initial_arr=0.;initial_exc=0
    states = {initial_exc:(initial_arr,list(prefix))}
    reached=keep_prefix if keep_prefix else 1
    for k in range(max(0,keep_prefix-1),len(path)-1):
        i,j=path[k:k+2]
        new={}
        for used,(arr,legs) in states.items():
            bucket=int(math.ceil((arr-1e-9)*4))
            for extra,item in enumerate(options(i,j,bucket)):
                if item is None or used+extra>MAX_EXC: continue
                arrival,leg=item
                key=used+extra
                if key not in new or arrival < new[key][0]:
                    new[key]=(arrival,legs+[leg])
        if not new: break
        # Pareto-prune states with later arrival and more exceptions.
        states={e:v for e,v in new.items() if not any(q<e and u[0]<=v[0] for q,u in new.items())}
        reached=k+2
        if expired(): break
    if not states:return None
    e,(arr,legs)=min(states.items(),key=lambda z:(-reached,z[1][0],z[0]))
    return {'path':path[:reached], 'transfers':legs, 'arrival':arr, 'exceptions':e}

def score(s):return (len(s['path']),-s['arrival'],-s['exceptions'])

best={'path':base_path,'transfers':base_legs,
      'arrival':base_legs[-1][2]+base_legs[-1][3],
      'exceptions':sum(x[4]>NORMAL for x in base_legs)}
print(f'PDF reconstruction | verified baseline {len(base_path)}/{N} | {TIME_LIMIT:.0f}s',flush=True)
print('Original verified JSON will not be modified.',flush=True)
trials=0
# Construct suffixes from earlier prefix, with occasional insertion/reversal/swap.
# This is distinct from the earlier window-only local search: suffix is fully re-timed
# by DP with exception budget, and paths are rebuilt from a chosen prefix.
while not expired():
    trials+=1
    prefix_len=RNG.choice([1,5,10,15,20,25,30,35,39,42])
    prefix_len=min(prefix_len,len(base_path))
    candidate=base_path[:prefix_len]
    remaining=[i for i in range(N) if i not in candidate]
    # Blend old good suffix order with novel insertions and reversals.
    if RNG.random()<0.55:
        candidate += [i for i in base_path[prefix_len:] if i not in candidate]
        missing=[i for i in range(N) if i not in candidate]
        RNG.shuffle(missing)
        for i in missing:
            at=RNG.randrange(prefix_len,len(candidate)+1)
            candidate.insert(at,i)
    else:
        old=[i for i in base_path[prefix_len:] if i in remaining]
        RNG.shuffle(remaining)
        candidate+=old[:RNG.randrange(len(old)+1)]
        candidate += [i for i in remaining if i not in candidate]
    if len(candidate)>prefix_len+3 and RNG.random()<0.5:
        a=RNG.randrange(prefix_len,len(candidate)-1)
        b=min(len(candidate),a+RNG.randint(2,7))
        candidate[a:b]=reversed(candidate[a:b])
    # Keep prefix as exact verified legs, allowing DP to reassign exceptions in suffix.
    sol=schedule(candidate,keep_prefix=prefix_len)
    if sol and score(sol)>score(best):
        # Independently re-evaluate every saved leg, avoiding trust in cached values.
        exc=0;prev=0.;ok=True
        for k,(i,j,dep,tof,dv) in enumerate(sol['transfers']):
            if (i!=sol['path'][k] or j!=sol['path'][k+1] or dep<prev-1e-7
                    or tof<0.001 or dep+tof>MAX_TIME+1e-7):ok=False;break
            check=cost(i,j,dep,tof)
            if check>SPECIAL+1e-6:ok=False;break
            exc+=check>NORMAL+1e-6
            prev=dep+tof
        if ok and exc<=MAX_EXC and len(sol['transfers'])==len(sol['path'])-1:
            best=sol
            best['verified']=False  # independent official validator still needed
            OUT.write_text(json.dumps(best,indent=2),encoding='utf8')
            print(f'NEW CANDIDATE: {len(best["path"])}/{N} day {best["arrival"]:.2f} exc {exc}',flush=True)
    if trials%25==0:
        print(f'Trials {trials} | best {len(best["path"])}/{N} | {time.monotonic()-clock:.0f}s',flush=True)
print(f'Finished | trials {trials} | best {len(best["path"])}/{N}',flush=True)
print('Verified 42/49 remains unchanged. New candidates require official validation.',flush=True)
