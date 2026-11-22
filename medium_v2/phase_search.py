"""Non-greedy time-window beam retiming for a fixed ESA Medium tomato order.

Every edge in every partial state is checked with the ORIGINAL evaluator's
compute_transfer. Only the original ESA fitness function can certify a record.

The beam explicitly keeps slower intermediate arrivals to explore orbital phase
windows that a greedy earliest-arrival suffix would discard.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import random
from typing import Optional


@dataclass
class State:
    arrival: float
    exceptions: int
    deps: tuple
    tofs: tuple


def exceptions_by_leg(udp, sol, cache):
    out = []
    for a,b,d,f in zip(sol['path'],sol['path'][1:],sol['dep'],sol['tof']):
        v=cache.cost(a,b,d,f)
        if not math.isfinite(v) or v > udp.dv_threshold_exception + 1e-6:
            raise ValueError('Seed contains an invalid transfer')
        out.append(int(v > udp.dv_threshold))
    if sum(out)>udp.n_dv_exception:
        raise ValueError('ESA exception budget exceeded in source')
    return out


def departure_grid(arrival, reference, rng, mode):
    """Orbital phase candidates, allowing both early and delayed departures."""
    if mode=='fine':
        shifts=(-.75,-.5,-.25,-.125,-.0625,-.03125,-.015625,0,
                 .015625,.03125,.0625,.125,.25,.5,.75)
        delay=(0,.015625,.03125,.0625,.125,.25,.5,.75)
    elif mode=='phase':
        shifts=(-2.,-1.5,-1.,-.75,-.5,-.25,0,.25,.5,.75,1.,1.5,2.)
        delay=(0,.125,.25,.5,.75,1.,1.5,2.)
    else:
        shifts=(-1.,-.5,-.25,-.125,-.0625,0,.0625,.125,.25,.5,1.)
        delay=(0,.03125,.0625,.125,.25,.375,.5,1.)
    vals={round(reference+x,8) for x in shifts}
    vals.update(round(arrival+x,8) for x in delay)
    vals.add(round(reference+rng.uniform(-.5,.5),8))
    vals.add(round(arrival+rng.random()*.5,8))
    vals.add(round(reference,8))
    vals={v for v in vals if v>=arrival-1e-8 and v>=0}
    # Keep departure candidates near the reference AND the current arrival.
    vals=sorted(vals,key=lambda v:(min(abs(v-reference),abs(v-arrival)),v))
    return vals[:13]


def flight_grid(reference, rng, mode, min_tof):
    if mode=='fine':
        shifts=(-.5,-.25,-.125,-.0625,-.03125,-.015625,-.0078125,
                 0,.0078125,.015625,.03125,.0625,.125,.25)
        base=(.03125,.0625,.125,.1875,.25,.375,.5,.75,1.,1.5,2.,3.)
    elif mode=='phase':
        shifts=(-1.5,-1.,-.75,-.5,-.25,-.125,0,.125,.25,.5,1.)
        base=(.0625,.125,.25,.375,.5,.75,1.,1.5,2.,2.5,3.,4.,5.)
    else:
        shifts=(-.75,-.5,-.25,-.125,-.0625,-.03125,0,.03125,.0625,.125,.25,.5)
        base=(.0625,.125,.1875,.25,.375,.5,.75,1.,1.5,2.,3.,4.)
    vals={round(reference+x,8) for x in shifts}
    vals.update(base)
    vals.add(round(max(min_tof,reference+rng.uniform(-.1,.1)),8))
    vals.add(round(reference,8))
    vals={v for v in vals if min_tof-1e-9<=v<=max(6.5,reference+1.)}
    vals=sorted(vals,key=lambda v:(min(abs(v-reference),abs(v-.25),abs(v-.5)),v))
    return vals[:15]


def select_states(states, beam, reference_arrival, rng):
    """Pareto+phase diversity: preserve late arrivals for downstream geometry."""
    if not states:return []
    states.sort(key=lambda s:(s.arrival,s.exceptions))
    chosen=[]; keys=set()
    # 0.0625 d phase bins, but also keep other exception use levels.
    def append(s):
        key=(int(round(s.arrival*16)),s.exceptions)
        if key in keys:return
        keys.add(key);chosen.append(s)
    for s in states[:max(1,beam//4)]:append(s)
    # Choose alternatives near the original phase. These can be slower now
    # but lead to faster, cheaper future legs.
    for s in sorted(states,key=lambda s:(abs(s.arrival-reference_arrival),s.exceptions))[:max(2,beam//3)]:
        append(s)
        if len(chosen)>=beam:return chosen[:beam]
    # Diversify over *different* phase bins, not just best arrival times.
    buckets={}
    for s in states:
        phase=int((s.arrival%1.0)*8)
        k=(phase,s.exceptions)
        if k not in buckets or s.arrival<buckets[k].arrival:
            buckets[k]=s
    rest=list(buckets.values()); rng.shuffle(rest)
    for s in rest:
        append(s)
        if len(chosen)>=beam:return chosen[:beam]
    for s in states:
        append(s)
        if len(chosen)>=beam:break
    return chosen[:beam]


def retime_suffix(udp, cache, sol, start, rng, *, beam=10, mode='balanced',
                  exception_flags=None, best_bound=None, max_calls=90000):
    """Retimes [start,n-1) without modifying route, prefix or official rules.

    Returns (solution, metadata) if a complete feasible schedule was constructed.
    Intermediate beam states may finish later than the incumbent at their index;
    only the final arrival determines objective.
    """
    n=len(sol['path']); nedges=n-1
    if not (0<=start<nedges):raise ValueError('Invalid suffix start')
    flags=exception_flags or exceptions_by_leg(udp,sol,cache)
    old=sol['dep']; duration=sol['tof']; route=sol['path']
    prior_ex=sum(flags[:start]);
    arrival=old[start-1]+duration[start-1] if start else 0.
    states=[State(arrival,prior_ex,(),())]
    used0=cache.stats.calls
    stats={'start':start,'mode':mode,'peak_frontier':1,'depth':0,'limit_exceeded':False}
    # Stay within the ESA final time and offer some slack relative to source phases.
    bound=min(udp.max_time, float(best_bound) if best_bound is not None else udp.max_time)
    for i in range(start,nedges):
        if cache.stats.calls-used0>=max_calls:
            stats['limit_exceeded']=True;return None,stats
        candidates=[]
        reference_arrival=old[i]+duration[i]
        shell_ex_reserve=0
        # Reserve known source exception legs for the untouched downstream portion.
        # This is a *heuristic* pruning rule, not an ESA constraint. We allow an
        # extra exception if a changed current leg can save one later.
        # Keep this disabled to allow genuine exception reallocation.
        for st in states:
            if st.arrival>bound-(nedges-i)*udp.min_tof+1e-8:continue
            a,b=route[i],route[i+1]
            departures=departure_grid(st.arrival,old[i],rng,mode)
            flights=flight_grid(duration[i],rng,mode,udp.min_tof)
            # Valid baseline edge remains an explicit fallback, if reachable.
            if old[i]>=st.arrival-1e-8:
                departures.append(old[i]);flights.append(duration[i])
            for d in sorted(set(departures)):
                if d<st.arrival-1e-8 or d>old[i]+2.0+1e-9:continue
                for f in sorted(set(flights)):
                    end=d+f
                    if end>=bound-(nedges-i-1)*udp.min_tof-1e-9:continue
                    v=cache.cost(a,b,d,f)
                    if v>udp.dv_threshold_exception+1e-6 or not math.isfinite(v):continue
                    ex=st.exceptions+int(v>udp.dv_threshold)
                    if ex>udp.n_dv_exception:continue
                    candidates.append(State(end,ex,st.deps+(d,),st.tofs+(f,)))
                if cache.stats.calls-used0>=max_calls:
                    stats['limit_exceeded']=True;break
            if stats['limit_exceeded']:break
        if not candidates:return None,stats
        states=select_states(candidates,beam,reference_arrival,rng)
        stats['depth']=i-start+1
        stats['peak_frontier']=max(stats['peak_frontier'],len(states))
        if stats['limit_exceeded']:return None,stats
    best=min(states,key=lambda s:(s.arrival,s.exceptions))
    result={'path':route[:], 'dep':old[:start]+list(best.deps),
            'tof':duration[:start]+list(best.tofs)}
    stats['arrival']=best.arrival
    stats['exceptions']=best.exceptions
    return result,stats


def choose_start(sol,rng,mode):
    n=len(sol['tof'])
    # 60% tail windows, 30% mid-range, 10% long-range.
    choice=rng.random()
    if choice<.60: return max(0,n-rng.randint(5,min(n,48)))
    if choice<.9:return max(0,n-rng.randint(30,min(n,105)))
    return rng.randint(0,max(0,n-1))
