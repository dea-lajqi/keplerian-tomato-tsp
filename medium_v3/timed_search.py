"""Bounded beam repair with exact original ESA transfer calls.

Finds a NEW ordering for a short, inclination-compatible shell segment;
rebuilds its departure/flight schedule and optionally the entire suffix.
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Optional

@dataclass
class State:
    arrival:float
    exceptions:int
    dep:tuple
    tof:tuple


def select_phases(states, beam, rng):
    """Retain earliest and distinct arrival time / exception combinations."""
    states.sort(key=lambda st:(st.arrival,st.exceptions))
    keep=[];seen=set()
    # Keep more of the early frontier than V2, but retain phase diversity.
    for s in states[:max(1,beam//3)]:
        key=(round(s.arrival*16),s.exceptions)
        if key not in seen:keep.append(s);seen.add(key)
    leftovers=states[max(1,beam//3):]
    buckets={}
    for s in leftovers:
        key=(int(round(s.arrival*8)),s.exceptions)
        if key not in buckets or buckets[key].arrival>s.arrival:buckets[key]=s
    items=list(buckets.values());rng.shuffle(items)
    for s in items:
        k=(round(s.arrival*16),s.exceptions)
        if k not in seen:keep.append(s);seen.add(k)
        if len(keep)>=beam:break
    if len(keep)<beam:
        for s in states:
            k=(round(s.arrival*16),s.exceptions)
            if k not in seen:keep.append(s);seen.add(k)
            if len(keep)>=beam:break
    return keep[:beam]


def sample_departures(arrival, olddep, source_dep, rng):
    x=[arrival,arrival+.03125,arrival+.0625,arrival+.125,arrival+.25,arrival+.5,
       olddep,olddep-.125,olddep+.125,olddep-.375,olddep+.375,
       source_dep, source_dep-.125,source_dep+.125,arrival+rng.random()*.3]
    x={round(max(0.,v),8) for v in x if v>=arrival-1e-8}
    a=sorted(x,key=lambda v:min(abs(v-arrival),abs(v-olddep),abs(v-source_dep)))
    return a[:12]


def sample_flights(oldtof, rng, min_tof):
    base=[.0625,.125,.1875,.25,.3125,.375,.5,.625,.75,1,1.25,1.5,2.,
          2.5,3.,3.5,4.,4.5,5.,5.5,6.]
    base += [oldtof,oldtof-.5,oldtof-.25,oldtof-.125,oldtof-.0625,
             oldtof+.125,oldtof+.25,oldtof+.5,oldtof+rng.uniform(-.15,.15)]
    x=sorted({round(v,8) for v in base if v>=min_tof-1e-9 and v<=max(6.5,oldtof+1)})
    if len(x)>22:
        x=sorted(sorted(x,key=lambda v:(min(abs(v-oldtof),abs(v-.25),abs(v-.75),abs(v-2.)),v))[:22])
    return x


def repair_window(udp, cache, old, new_route, left, right, rng, *, beam=10,
                  max_calls=9000, deadline=None):
    """Reconnect [left,right] inclusive; keep a fixed prefix.

    Returns list of (state, prefix_dep, prefix_tof). These are not independently
    full ESA-validated until a suffix is appended and original fitness checked.
    """
    dep=old['dep'];tof=old['tof'];path=old['path'];n=len(dep)
    if not (0<=left<right<=n):return []
    if path[:left+1]!=new_route[:left+1] or path[right:]!=new_route[right:]:
        raise ValueError('Repair must preserve prefix, start and suffix nodes')
    prefix_exc=0
    for i in range(left):
        v=cache.exception(path[i],path[i+1],dep[i],tof[i])
        if v is None:raise ValueError('ESA invalid source prefix')
        prefix_exc+=v
    start=dep[left-1]+tof[left-1] if left else 0.
    states=[State(start,prefix_exc,(),())]
    source_deps={path[i]:dep[i] for i in range(n)}
    start_calls=cache.stats.calls
    upper=min(udp.max_time, (dep[right] if right<n else dep[-1]+tof[-1]) + 1.25)
    for i in range(left,right):
        if deadline is not None and __import__('time').monotonic()>deadline:return []
        candidates=[]
        a,b=new_route[i],new_route[i+1]
        for s in states:
            dt=sample_departures(s.arrival,dep[i],source_deps.get(a,dep[i]),rng)
            fl=sample_flights(tof[i],rng,udp.min_tof)
            # The baseline transfer is included when both its endpoints are unchanged.
            for d in dt:
                for f in fl:
                    end=d+f
                    if end>upper+1e-8:continue
                    ex=cache.exception(a,b,d,f)
                    if ex is None or s.exceptions+ex>udp.n_dv_exception:continue
                    candidates.append(State(end,s.exceptions+ex,s.dep+(d,),s.tof+(f,)))
                    if cache.stats.calls-start_calls>=max_calls:break
                if cache.stats.calls-start_calls>=max_calls:break
            if cache.stats.calls-start_calls>=max_calls:break
        if not candidates:return []
        states=select_phases(candidates,beam,rng)
        if cache.stats.calls-start_calls>=max_calls and i<right-1:return []
    return [(s,dep[:left]+list(s.dep),tof[:left]+list(s.tof)) for s in states]


def join_candidates(udp,cache,old,new_route,right,repaired,rng,*,deadline=None,
                    max_suffix=2):
    """Complete a repaired route with unchanged tail OR adaptively retimed tail.

    Critical: all results must pass original ESA fitness later; this is not
    itself an ESA validation function.
    """
    old_dep=old['dep'];old_tof=old['tof'];n=len(old_dep)
    out=[]
    for s,ds,fs in sorted(repaired,key=lambda p:p[0].arrival)[:max(3,max_suffix)]:
        if right==n:
            out.append({'path':new_route,'dep':ds,'tof':fs});continue
        if s.arrival<=old_dep[right]+1e-9:
            out.append({'path':new_route,'dep':ds+old_dep[right:],
                        'tof':fs+old_tof[right:]})
    # An unchanged tail cannot improve total arrival. Retime the tail instead.
    from medium_overnight.search import greedy_suffix
    for s,ds,fs in sorted(repaired,key=lambda p:p[0].arrival)[:max_suffix]:
        if deadline is not None and __import__('time').monotonic()>deadline:break
        if right==n:break
        more=greedy_suffix(udp,cache,new_route,old_dep,old_tof,right,s.arrival,
                           s.exceptions,rng,deadline=deadline)
        if more:
            td,tf,_=more
            out.append({'path':new_route,'dep':ds+td,'tof':fs+tf})
    return out
