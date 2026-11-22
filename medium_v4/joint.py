"""Joint sequence+timing search over a bounded window of the original ESA Medium tour.

The surrounding 181-tomato tour is immutable during a splice; all physical tests
use the unchanged original ESA udp.compute_transfer via the V1 DVCache.
Any proposed complete tour MUST pass udp.fitness through validation.verify.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import random
import time

@dataclass(frozen=True)
class Node:
    arrival: float
    used_ex: int
    visited: int
    order: tuple[int, ...]
    starts: tuple[float, ...]
    flights: tuple[float, ...]


def phases(arrival: float, reference: float, src_reference: float, rng: random.Random, precision: str = 'broad'):
    # Include exact baseline departure, and several earlier/later phases.
    if precision == 'fine':
        shifts = (-.25, -.125, -.0625, -.03125, -.015625, 0, .015625, .03125, .0625, .125)
    else:
        shifts = (-.75, -.5, -.25, -.125, -.0625, 0, .0625, .125, .25, .5)
    values = [arrival,arrival+.03125,arrival+.0625,arrival+.125]
    for s in shifts:
        values.append(reference+s)
    for s in shifts[2:-2:2]:
        values.append(src_reference+s)
    values += [arrival + rng.uniform(0,.2)]
    valid = sorted({round(v, 8) for v in values if v >= arrival-1e-9 and v>=0})
    if reference >= arrival-1e-9 and reference not in valid: valid.append(reference)
    # Keep baseline ALWAYS even with cap.
    valid = sorted(valid, key=lambda d:(min(abs(d-reference),abs(d-arrival)),d))[:12]
    if reference>=arrival-1e-9 and reference not in valid:
        valid[-1]=reference
    return sorted(set(valid))


def times_of_flight(primary: float, secondary: float, minimum: float, rng: random.Random, mode='broad'):
    offsets = (-.5,-.25,-.125,-.0625,-.03125,0,.03125,.0625,.125,.25,.5)
    if mode=='fine':offsets=(-.125,-.0625,-.03125,-.015625,-.0078125,0,.0078125,.015625,.03125,.0625,.125)
    raw=[primary+x for x in offsets]+[secondary+x for x in offsets[::2]]
    raw += [.0625,.125,.25,.375,.5,.75,1.,1.5,2.,3.,4.,5.,6.]
    raw += [primary+rng.uniform(-.08,.08)]
    vals=sorted({round(t,8) for t in raw if minimum-1e-9<=t<=7.5},key=lambda t:(abs(t-primary),t))[:14]
    if primary >= minimum and primary not in vals:
        vals[-1]=primary
    return sorted(set(vals))


def group_shell(udp, idx):
    return int(float(udp.tomato_orbital_parameters[idx][0])>8e6)


def edge_exceptions(sol, cache):
    out=[]
    for i,(d,f) in enumerate(zip(sol['dep'],sol['tof'])):
        v=cache.exception(sol['path'][i],sol['path'][i+1],d,f)
        if v is None:raise ValueError('Cannot classify baseline exception leg '+str(i))
        out.append(v)
    return out


def window_choices(sol, udp, rng, length_range=(2,5), end_bias=.33):
    """Choose an interior-only window in a single shell; sometimes target endgame."""
    path=sol['path'];n=len(path)
    L=rng.randint(*length_range)
    if rng.random()<end_bias: left=n-L-2-(0 if rng.random()<.65 else rng.randrange(1,min(10,n-L-1)))
    else:left=rng.randrange(0,n-L-1)
    left=max(0,min(left,n-L-2));right=left+L+1
    # Do not reorder across shell boundaries (budget of exceptions is tight).
    shell=group_shell(udp,path[left])
    if any(group_shell(udp, t)!=shell for t in path[left:right]):return None
    return left,right


def prune(states, beam:int, baseline_order:tuple, rng, forced=None):
    """Time-expanded beam: quota for baseline, early routes and diverse phases/orders."""
    if not states:return []
    states.sort(key=lambda s:(s.arrival,s.used_ex))
    original=forced if forced is not None else next((s for s in states if s.order==baseline_order[:len(s.order)]),None)
    result=[];seen=set()
    if original:
        result.append(original)
        seen.add((original.visited, original.order, round(original.arrival*8), original.used_ex))
    # Separate partial order and phase buckets to prevent copies of one ordering filling beam.
    per_order={}
    for s in states:
        k=(s.visited,s.order)
        if per_order.get(k,0)>=2:continue
        key=(s.visited,s.order,round(s.arrival*8),s.used_ex)
        if key in seen:continue
        per_order[k]=per_order.get(k,0)+1
        result.append(s);seen.add(key)
        if len(result)>=beam:break
    return result[:beam]


def joint_window(udp,cache,sol,left,right,rng,*,beam=24,call_cap=9000,
                 deadline=None,mode='broad',reorder=True,keep_baseline=True,
                 max_results=12):
    """Bounded dynamic program over a subset of targets AND continuous edge timing.

    `left` / `right` index tomatoes and the return anchor is path[right].
    Exact prefix and suffix retained. We accept only states arriving before the
    original next departure; the old suffix then remains ESA-feasible.
    Return a list of (complete candidate, metadata); no record is claimed here.
    """
    order=sol['path'];dep=sol['dep'];tof=sol['tof'];n=len(dep)
    if not (0<=left<right<=n):return []
    interiors=order[left+1:right]
    if not interiors:return []
    if len(interiors)>7:raise ValueError('window too large')
    K=len(interiors)
    if len(set(interiors))!=K:raise ValueError('invalid source window')
    allowed = edge_exceptions(sol,cache)
    limit=udp.n_dv_exception-sum(allowed[:left])-sum(allowed[right:])
    if limit<0:return []
    arrival=dep[left-1]+tof[left-1] if left else 0.
    original_end = dep[right] if right<n else udp.max_time
    # Do not constrain a no-suffix endgame to existing terminal date.
    original_end=min(original_end,udp.max_time)
    srcdep={order[i]:dep[i] for i in range(n)}
    srctof={order[i]:tof[i] for i in range(n)}
    baseline_seq=tuple(interiors+[order[right]])
    allmask=(1<<K)-1
    states=[Node(arrival,0,0,(),(),())]
    start_calls=cache.stats.calls
    baseline_state=Node(arrival,0,0,(),(),())
    result=[]
    for depth in range(K+1):
        next_states=[]
        for s in states:
            if deadline and time.monotonic()>deadline:return result
            if cache.stats.calls-start_calls >= call_cap:return result
            if depth<K:
                choices=[(j,x) for j,x in enumerate(interiors) if not (s.visited>>j)&1]
                if not reorder:
                    target=interiors[depth]
                    choices=[(j,x) for j,x in choices if x==target]
                else:
                    # Favor baseline and geometrically similar next targets.
                    here=order[left] if depth==0 else s.order[-1]
                    choices.sort(key=lambda p:(p[1]!=interiors[depth],abs(float(udp.tomato_orbital_parameters[here][2])-float(udp.tomato_orbital_parameters[p[1]][2]))))
            else:
                choices=[(-1,order[right])]
            here=order[left] if depth==0 else s.order[-1]
            i=left+depth
            for j,nxt in choices:
                if cache.stats.calls-start_calls >= call_cap:break
                next_mask=s.visited if j==-1 else s.visited|(1<<j)
                departures=phases(s.arrival,dep[i],srcdep.get(here,dep[i]),rng,mode)
                flights=times_of_flight(tof[i],srctof.get(here,tof[i]),udp.min_tof,rng,mode)
                # At the return anchor, leave sufficient time for unchanged suffix.
                for d in departures:
                    if d>original_end+1e-9:continue
                    for f in flights:
                        end=d+f
                        if end>original_end+1e-9:continue
                        e=cache.exception(here,nxt,d,f)
                        if e is None or s.used_ex+e>limit:continue
                        new=Node(end,s.used_ex+e,next_mask,s.order+(nxt,),s.starts+(d,),s.flights+(f,))
                        next_states.append(new)
                        if cache.stats.calls-start_calls>=call_cap:break
                    if cache.stats.calls-start_calls>=call_cap:break
        # Keep an exact original-time reference trajectory in the beam, even if a
        # faster *partial* arrival consumes a different exception allocation.
        # Without this, premature dominance pruning loses the verified fallback.
        i=left+depth
        count=depth+1
        forced=Node(dep[i]+tof[i],sum(allowed[left:i+1]),
                    (1<<min(count,K))-1,
                    tuple(order[left+1:i+2]),
                    tuple(dep[left:i+1]),tuple(tof[left:i+1]))
        if forced not in next_states:next_states.append(forced)
        if depth == K:
            # This is a *full anchored* window; unchanged suffix is feasible,
            # because arrival <= departure of next fixed edge.
            next_states.sort(key=lambda s:(s.arrival,s.used_ex,s.order==baseline_seq))
            per_order={}
            for s in next_states:
                if s.visited!=allmask:continue
                if per_order.get(s.order,0)>=2:continue
                per_order[s.order]=per_order.get(s.order,0)+1
                candidate={'path':order[:left+1]+list(s.order)+order[right+1:],
                           'dep':dep[:left]+list(s.starts)+dep[right:],
                           'tof':tof[:left]+list(s.flights)+tof[right:]}
                assert len(candidate['path'])==len(order)
                result.append((candidate,{'arrival':s.arrival,'slack':(dep[right]-s.arrival if right<n else 0.),
                                           'reordered':tuple(candidate['path'])!=tuple(order),
                                           'window':(left,right),'window_exception_count':s.used_ex}))
                if len(result)>=max_results:break
            return result
        states=prune(next_states,beam,baseline_seq,rng,forced=forced)
    return result


def suffix_polish(udp,cache,candidate,rng,anchor,*,deadline=None):
    """Optional exact-physics V1 suffix compression after a novel complete route."""
    from medium_overnight.search import greedy_suffix
    dep=candidate['dep'];tof=candidate['tof'];path=candidate['path']
    used=0
    for i in range(anchor):
        e=cache.exception(path[i],path[i+1],dep[i],tof[i])
        if e is None:return None
        used+=e
    arr=dep[anchor-1]+tof[anchor-1] if anchor else 0.
    found=greedy_suffix(udp,cache,path,dep,tof,anchor,arr,used,rng,deadline=deadline)
    if found is None:return None
    sd,sf,_=found
    return {'path':path[:],'dep':dep[:anchor]+sd,'tof':tof[:anchor]+sf}


def terminal_crossing(udp,cache,sol,rng,*,deadline=None,eval_limit=2400):
    """Optimize the *last* leg, including its potentially expensive shell crossing.

    The prior 179 transfers and their DV exception budget are unchanged.
    This creates a complete candidate for original ESA fitness validation.
    """
    path=sol['path'];dep=sol['dep'];tof=sol['tof'];n=len(dep)
    i=n-1; arrival=dep[i-1]+tof[i-1] if i else 0.
    used=sum(edge_exceptions(sol,cache)[:i])
    if used>udp.n_dv_exception:return None
    best=(dep[i]+tof[i],dep[i],tof[i])
    start=cache.stats.calls
    src=[arrival,arrival+.015625,arrival+.03125,arrival+.0625,arrival+.125,
         arrival+.25,arrival+.375,arrival+.5,dep[i]]
    src += [dep[i]+o for o in (-.5,-.375,-.25,-.1875,-.125,-.0625,-.03125,
                                 .03125,.0625,.125)]
    src += [arrival+rng.uniform(0,.5) for _ in range(3)]
    for d in sorted({round(x,8) for x in src if x>=arrival-1e-9 and x>=0}):
        if deadline and time.monotonic()>deadline:break
        upper=best[0]-d
        if upper<udp.min_tof:continue
        options=[tof[i]+o for o in (-.75,-.5,-.375,-.25,-.1875,-.125,-.0625,
             -.03125,-.015625,-.0078125,0,.0078125,.015625,.03125,.0625,.125,.25)]
        options += [.0625,.09375,.125,.1875,.25,.3125,.375,.5,.75,1.,1.5,2.,3.,4.]
        trials=sorted({round(v,8) for v in options if udp.min_tof-1e-10<=v<upper-1e-9})
        for f in trials:
            if deadline and time.monotonic()>deadline:break
            e=cache.exception(path[i],path[i+1],d,f)
            if e is None or used+e>udp.n_dv_exception:continue
            # Refine feasible interval via stepwise local trials (no monotonic-DV assumption).
            best=(d+f,d,f)
            for step in (.015625,.0078125,.00390625,.001953125):
                shorter=best[2]-step
                if shorter<udp.min_tof:continue
                ex=cache.exception(path[i],path[i+1],d,shorter)
                if ex is not None and used+ex<=udp.n_dv_exception:
                    best=(d+shorter,d,shorter)
            break
        if cache.stats.calls-start>eval_limit:break
    if best[0]>=dep[-1]+tof[-1]-1e-8:return None
    cand={'path':path[:],'dep':dep[:],'tof':tof[:]}
    cand['dep'][-1]=best[1];cand['tof'][-1]=best[2]
    return cand
