"""Systematic single-missing-tomato insertion with local retiming.
Run: python kttsp_47_precision.py
Keeps all validated originals intact; new output is a candidate.
"""
import json,math,time,random,itertools,heapq
from pathlib import Path
from functools import lru_cache
from beam_search import transfer_dv_cached,tomatoes
ROOT=Path('results'); N=len(tomatoes)
SOURCE=ROOT/'KTTSP_46_VALIDATED.json'
base=json.loads(SOURCE.read_text(encoding='utf8'))
LIMIT=360.; T0=time.monotonic(); rng=random.Random(47)
TOFS=(.125,.25,.375,.5,.75,1.,1.5,2.,2.5,3.,3.5,4.,5.,6.,7.,9.,12.)
WAITS=(0.,.25,.5,1.,2.,4.,7.)
@lru_cache(maxsize=600000)
def dv(i,j,t,dt):
    try:
        x=float(transfer_dv_cached(i,j,t,dt))
        return x if math.isfinite(x) else math.inf
    except (ValueError,RuntimeError,OverflowError,ZeroDivisionError):return math.inf

def valid(sol):
    p=sol['path']; ls=sol['transfers']; t=0.; ex=0
    if len(set(p))!=len(p) or len(ls)!=len(p)-1:return False
    for k,(i,j,d,dt,v) in enumerate(ls):
        if (i,j)!=(p[k],p[k+1]) or d<t-1e-7 or dt<.001 or d+dt>200+1e-7:return False
        z=dv(i,j,d,dt)
        if z>600+1e-7 or abs(z-v)>.01:return False
        ex+=z>100+1e-7;t=d+dt
    return ex<=5 and abs(t-sol['arrival'])<1e-6

if not valid(base):raise SystemExit('Base 46 solution did not pass validation')
missing=sorted(set(range(N))-set(base['path']))
print('PRECISION SEARCH | missing',missing,'| max 360 sec',flush=True)

# Prioritize small neighborhoods: preserve most of the already validated 46 path.
# First evaluate each insertion at every edge, then grow windows around the insertion.
windows=[]
for radius in (0,1,2,3,4,6,9):
    for edge in range(len(base['path'])-1):
        a=max(0,edge-radius);b=min(len(base['path'])-1,edge+1+radius)
        windows.append((a,b,edge,radius))
# First attempt very short repairs across every edge, and then longer ones.

# For a repaired window, prefix and suffix remain unchanged, and all transfer
# constraints are checked against the real Lambert dv function.
def repair(a,b,edge,missing_node,mode):
    path=base['path']; old=base['transfers']
    prefix=old[:a];suffix=old[b:]
    start=prefix[-1][2]+prefix[-1][3] if prefix else 0.
    deadline=suffix[0][2] if suffix else 200.
    if deadline-start<.001*(b-a+1):return None
    prefix_ex=sum(x[4]>100+1e-7 for x in prefix)
    suffix_ex=sum(x[4]>100+1e-7 for x in suffix)
    budget=5-prefix_ex-suffix_ex
    if budget<0:return None
    inside=path[a+1:b]
    if mode==1 and len(inside)>1:
        inside=inside[:]
        idx=max(0,min(len(inside)-1,edge-a-1))
        if idx+1<len(inside):inside[idx],inside[idx+1]=inside[idx+1],inside[idx]
    elif mode==2 and len(inside)>1:
        inside=list(reversed(inside))
    insert_pos=max(0,min(len(inside),edge-a))
    inside.insert(insert_pos,missing_node)
    nodes=[path[a]]+inside+[path[b]]
    states=[(start,0,[])]
    for i,j in zip(nodes,nodes[1:]):
        new={}
        for arr,ex,legs in states:
            for wait in WAITS:
                dep=round(math.ceil((arr+wait-1e-9)*4)/4,5)
                if dep>=deadline:continue
                for tof in TOFS:
                    end=dep+tof
                    if end>deadline+1e-8:continue
                    v=dv(i,j,dep,tof)
                    if v>600+1e-7:continue
                    ex2=ex+int(v>100+1e-7)
                    if ex2>budget:continue
                    # Quarter-day phases; keep earliest feasible per phase and exception use.
                    key=(ex2,round(end*4))
                    if key not in new or end<new[key][0]:
                        new[key]=(end,ex2,legs+[[i,j,dep,tof,v]])
        if not new:return None
        # Diverse arrival phases, plus best earliest state for each exception count.
        values=list(new.values());values.sort(key=lambda z:z[0])
        keep=[]
        for ex2 in range(budget+1):
            options=[x for x in values if x[1]==ex2]
            if options:keep.append(options[0])
        # Keep late phases too, since Lambert feasibility is phase-dependent.
        by_phase={}
        for x in values:
            key=(x[1],int(x[0]*2))
            if key not in by_phase:by_phase[key]=x
        extra=list(by_phase.values());rng.shuffle(extra)
        keep.extend(extra[:max(0,16-len(keep))])
        states=keep[:16]
        if time.monotonic()-T0>LIMIT:return None
    winner=min(states,key=lambda z:(z[0],z[1]))
    new_path=path[:a]+nodes+path[b+1:]
    new_legs=prefix+winner[2]+suffix
    return {'path':new_path,'transfers':new_legs,'arrival':new_legs[-1][2]+new_legs[-1][3],
            'exceptions':prefix_ex+winner[1]+suffix_ex,'verified':False}

trials=0
for a,b,edge,radius in windows:
    if time.monotonic()-T0>LIMIT:break
    for m in missing:
        if time.monotonic()-T0>LIMIT:break
        for mode in ((0,) if radius<2 else (0,1,2)):
            trials+=1
            sol=repair(a,b,edge,m,mode)
            if sol and len(sol['path'])==47 and valid(sol):
                out=ROOT/'KTTSP_47_PRECISION_CANDIDATE.json'
                out.write_text(json.dumps(sol,indent=2),encoding='utf8')
                print('FOUND 47/49 | arrival',sol['arrival'],'| exceptions',sol['exceptions'],'|',out,flush=True)
                print('Remaining:',sorted(set(range(N))-set(sol['path'])),flush=True)
                raise SystemExit(0)
            if trials%100==0:
                print('Trials',trials,'| radius',radius,'| elapsed',round(time.monotonic()-T0),'sec',flush=True)
print('Finished:',trials,'trials, no 47 found in searched windows.',flush=True)
