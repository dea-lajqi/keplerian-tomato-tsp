"""Focused KTTSP 47 search: rearrange 2-6 nodes, insert missing tomato,
reconnect to the verified suffix at its ORIGINAL departure time.
Run next to beam_search.py. Existing validated files are never overwritten.
"""
import json, math, random, time, itertools
from pathlib import Path
from functools import lru_cache
from beam_search import transfer_dv_cached, tomatoes

ROOT=Path('results'); SRC=ROOT/'KTTSP_46_DAY183_VALIDATED.json'
seed=json.loads(SRC.read_text(encoding='utf-8'))
P=seed['path']; L=seed['transfers']; N=len(tomatoes)
assert len(P)==46 and len(L)==45 and len(set(P))==46
missing=sorted(set(range(N))-set(P))
assert missing==[4,11,17],missing
START=time.monotonic(); LIMIT=300.; rng=random.Random(20261119)
TOFS=(.125,.25,.375,.5,.75,1.,1.5,2.,2.5,3.,3.5,4.,5.,6.,7.,9.,12.)
WAITS=(0.,.125,.25,.5,.75,1.,1.5,2.,3.,4.,5.,7.,10.)
@lru_cache(maxsize=500000)
def dv(i,j,t,tof):
    try:
        x=float(transfer_dv_cached(int(i),int(j),round(float(t),6),float(tof)))
        return x if math.isfinite(x) else math.inf
    except (ValueError,RuntimeError,OverflowError,ZeroDivisionError):return math.inf

def check(legs,path):
    if len(legs)!=len(path)-1 or len(set(path))!=len(path):return False
    t=0.; exc=0
    for k,(a,b,dep,tof,v) in enumerate(legs):
        if (a,b)!=(path[k],path[k+1]) or dep<t-1e-7 or tof<.001 or dep+tof>200+1e-7:return False
        actual=dv(a,b,dep,tof)
        if abs(actual-v)>.001 or actual>600+1e-7:return False
        exc+=actual>100+1e-7;t=dep+tof
    return exc<=5

# Precompute prefix and suffix exception counts, so local DP knows its exact budget.
pre=[0]
for leg in L:pre.append(pre[-1]+int(leg[4]>100+1e-7))
suf=[0]*(len(L)+1)
for k in range(len(L)-1,-1,-1):suf[k]=suf[k+1]+int(L[k][4]>100+1e-7)

@lru_cache(maxsize=220000)
def edge_choices(i,j,bucket,deadline):
    """Multiple arrivals retained by half-day buckets, both normal/exception."""
    arr=bucket/8.; out={}
    for wait in WAITS:
        dep=round(arr+wait,6)
        if dep>=deadline:continue
        for tof in TOFS:
            end=dep+tof
            if end>deadline+1e-7:continue
            v=dv(i,j,dep,tof)
            if v>600+1e-7:continue
            ex=int(v>100+1e-7)
            key=(ex,int(end*2+1e-7))
            if key not in out or end<out[key][0]:out[key]=(end,[i,j,dep,tof,v])
    return tuple(out.values())

def test_window(left,right,target,order):
    # Replace P[left:right] with 'order' + target. Keep P[:left] and P[right:] unchanged.
    if time.monotonic()-START>LIMIT:return None
    route=P[:left]+list(order)+P[right:]
    prefix=L[:left-1] if left else []
    begin=prefix[-1][2]+prefix[-1][3] if prefix else 0.
    prefix_ex=pre[left-1] if left else 0
    # right node is rejoin point; old outgoing leg must remain feasible.
    if right<len(P)-1:
        deadline=L[right][2]  # departure from P[right] to P[right+1]
        suffix=L[right:]
        suffix_ex=suf[right]
    else:
        deadline=200.
        suffix=[]
        suffix_ex=0
    budget=5-prefix_ex-suffix_ex
    if budget<0 or begin>=deadline:return None
    # first altered edge starts at left-1, or 0 if left=0
    first=max(0,left-1)
    states={(0,int(begin*8+1e-7)):(begin,[])}
    for k in range(first, min(right+1,len(route)-1)):
        i,j=route[k],route[k+1]
        nxt={}
        for (used,_),(arr,legs) in states.items():
            b=math.ceil((arr-1e-9)*8)
            for end,leg in edge_choices(i,j,b,deadline):
                extra=int(leg[4]>100+1e-7)
                if used+extra>budget:continue
                key=(used+extra,int(end*2+1e-7))
                if key not in nxt or end<nxt[key][0]:nxt[key]=(end,legs+[leg])
        if not nxt:return None
        # Preserve diversity of arrival times and exception budgets
        states=dict(sorted(nxt.items(),key=lambda kv:(kv[0][0],kv[1][0]))[:90])
    if not states:return None
    for (_, _),(arr,legs) in sorted(states.items(),key=lambda kv:(kv[0][0],kv[1][0])):
        if suffix and arr>suffix[0][2]+1e-7:continue
        full=prefix+legs+suffix
        if check(full,route):
            return {'path':route,'transfers':full,'arrival':full[-1][2]+full[-1][3],
                    'exceptions':sum(x[4]>100+1e-7 for x in full),'verified':False,
                    'method':'window_reorder_exact_suffix','target':target,'window':[left,right]}
    return None

print('START | 46/49 day 183.75 | missing',missing,flush=True)
print('Window reorder + insertion; verified prefix/suffix kept exactly.',flush=True)
trials=0
# Focus on inner-orbit group but include all positions, windows sizes 2..6.
inner={4,11,16,17,18,23,27,32,34}
windows=[]
for width in (2,3,4,5,6):
    for left in range(1,len(P)-width+1):
        right=left+width
        priority=0 if any(x in inner for x in P[max(0,left-2):min(len(P),right+2)]) else 1
        windows.append((priority,width,left,right))
windows.sort(key=lambda x:(x[0],x[1],rng.random()))
try:
    for priority,width,left,right in windows:
        if time.monotonic()-START>LIMIT:break
        chunk=P[left:right]
        for target in missing:
            if time.monotonic()-START>LIMIT:break
            # Enumerate short windows; sample larger ones to keep runtime bounded.
            arrangements=[]
            if width<=3:
                arrangements=itertools.permutations(chunk+[target])
            else:
                arrangements=[]
                seen=set()
                for _ in range(20 if width==4 else 35):
                    seq=chunk+[target]
                    rng.shuffle(seq); key=tuple(seq)
                    if key not in seen:arrangements.append(key);seen.add(key)
                for idx in range(width+1):
                    seq=chunk[:];seq.insert(idx,target)
                    if tuple(seq) not in seen:arrangements.append(tuple(seq));seen.add(tuple(seq))
            for seq in arrangements:
                if time.monotonic()-START>LIMIT:break
                trials+=1
                result=test_window(left,right,target,seq)
                if result:
                    out=ROOT/'KTTSP_47_WINDOW_CANDIDATE.json'
                    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
                    print('FOUND 47/49 | day',result['arrival'],'| exceptions',result['exceptions'],
                          '| target',target,'| window',left,right,'| SAVED',out,flush=True)
                    raise SystemExit
                if trials%250==0:
                    print('Trials',trials,'| window',width,'|',int(time.monotonic()-START),'sec',flush=True)
except KeyboardInterrupt:
    print('Interrupted; validated original unchanged.',flush=True)
print('FINISHED | no 47 found in tested windows | trials',trials,'|',int(time.monotonic()-START),'sec',flush=True)
