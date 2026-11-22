"""Experimental local route + timing search for ESA Medium.
Run from project root in conda environment kttsp-medium.
Always preserves the incumbent and ESA evaluator; saves only validated improvements.
"""
import argparse
import json
import math
import random
import runpy
import time
from pathlib import Path

DEFAULT_JSON = 'results/esa/medium_20261120T080509Z_bc1f8d8f_4_verified.json'
DEFAULT_EVAL = 'esa_evaluation/spoc-4-keplerian-tomato-traveling-salesperson-medium-1782203984713.py'

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--solution', default=DEFAULT_JSON)
    p.add_argument('--evaluator', default=DEFAULT_EVAL)
    p.add_argument('--seconds', type=float, default=300)
    p.add_argument('--trials', type=int, default=250)
    p.add_argument('--seed', type=int, default=20261122)
    p.add_argument('--radius', type=int, default=5)
    args = p.parse_args()
    rng = random.Random(args.seed)
    udp = runpy.run_path(args.evaluator)['udp']
    saved = json.loads(Path(args.solution).read_text())
    order = [int(v) for v in saved['path']]
    legs = saved['transfers']
    departures = [float(t[2]) for t in legs]
    durations = [float(t[3]) for t in legs]
    n = len(legs)
    assert len(order) == udp.n_tomatoes == n + 1
    incumbent = udp.fitness(departures + durations + order)
    assert all(abs(float(v)) < 1e-7 for v in incumbent[1:]), incumbent
    best = float(incumbent[0])
    out = Path('results/esa/medium_route_experiments')
    out.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.seconds
    calls = 0
    accepted = 0
    checked = 0
    cache = {}
    normal = udp.dv_threshold
    exceptional = udp.dv_threshold_exception

    def dv(a, b, dep, tof):
        nonlocal calls
        key = (a, b, round(dep, 7), round(tof, 7))
        if key not in cache:
            calls += 1
            cache[key] = float(udp.compute_transfer(a, b, dep, tof))
        return cache[key]

    def solve_block(candidate_order, lo, hi):
        # Change a short segment and reconnect by the incumbent departure at hi+1.
        # The resulting full schedule is subsequently checked by ESA exactly.
        prefix_arrival = departures[lo-1] + durations[lo-1] if lo else 0.0
        target = departures[hi+1] if hi+1 < n else best
        prefix_exceptions = sum(float(t[4]) > normal + 1e-6 for t in legs[:lo])
        suffix_exceptions = sum(float(t[4]) > normal + 1e-6 for t in legs[hi+1:])
        remaining_exceptions = udp.n_dv_exception - prefix_exceptions - suffix_exceptions
        if remaining_exceptions < 0: return None
        # State = (arrival, exception_count, departures, durations)
        states = [(prefix_arrival, 0, [], [])]
        for i in range(lo, hi+1):
            next_states = []
            for arrival, used, ds, fs in states:
                ref_dep, ref_flight = departures[i], durations[i]
                # Anchor candidates to both current encounter and incumbent phase.
                dep_grid = {arrival + d for d in (0, .03125, .0625, .125, .25, .5)}
                dep_grid.update(ref_dep + d for d in (-.5, -.25, -.125, 0, .125, .25, .5))
                tof_grid = {ref_flight, max(udp.min_tof, ref_flight*.75),
                            max(udp.min_tof, ref_flight*.5)}
                tof_grid.update(max(udp.min_tof, ref_flight + d)
                                for d in (-.5, -.25, -.125, -.0625, .0625, .125, .25))
                for dep in sorted(dep_grid):
                    if dep + 1e-8 < arrival or dep < 0 or dep >= target: continue
                    for tof in sorted(tof_grid):
                        if tof < udp.min_tof or dep+tof > target + 1e-8: continue
                        value = dv(candidate_order[i], candidate_order[i+1], dep, tof)
                        if not math.isfinite(value) or value > exceptional + 1e-6: continue
                        exc = used + int(value > normal + 1e-6)
                        if exc > remaining_exceptions: continue
                        next_states.append((dep+tof, exc, ds+[dep], fs+[tof]))
                if time.monotonic() >= deadline: return None
            if not next_states: return None
            # Preserve different phase timings and exception usage.
            next_states.sort(key=lambda s: (s[0] + .15*s[1], s[0]))
            states=[]
            buckets=set()
            for s in next_states:
                bucket = (round(s[0]*16), s[1])
                if bucket in buckets: continue
                buckets.add(bucket)
                states.append(s)
                if len(states) >= 12: break
        return states

    print(f'ESA-validated baseline: {best:.8f} days; route mutation experiment', flush=True)
    for trial in range(1, args.trials+1):
        if time.monotonic() >= deadline: break
        # Swap nearby nodes, reverse a segment, or relocate one node locally.
        lo = rng.randint(1, n-args.radius-2)
        width = rng.randint(2, args.radius)
        hi = min(n-2, lo + width)
        candidate = order[:]
        mode = rng.choice(('swap', 'reverse', 'insert'))
        j, k = sorted(rng.sample(range(lo+1, hi+1), 2))
        if mode == 'swap': candidate[j],candidate[k]=candidate[k],candidate[j]
        elif mode == 'reverse': candidate[j:k+1]=reversed(candidate[j:k+1])
        else:
            node=candidate.pop(k); candidate.insert(j,node)
        # Only arcs lo..hi change; other legs remain exactly the incumbent.
        states=solve_block(candidate, lo, hi)
        if states is None: continue
        accepted += 1
        for arrival, used, ds, fs in states:
            # For a full finish improvement, the unchanged suffix needs a real
            # earlier departure; a short block reconnection alone does not save time.
            # First test feasible alternative route, then shift suffix greedily
            # using feasible Lambert re-evaluation.
            dnew=departures[:lo] + ds + departures[hi+1:]
            fnew=durations[:lo] + fs + durations[hi+1:]
            curr=arrival
            for s in range(hi+1,n):
                old_dep=dnew[s]
                # Search for a feasible earlier departure with existing flight time.
                earliest=max(curr, old_dep-1.0)
                candidates=sorted(set([old_dep, earliest, curr] +
                                     [old_dep-x for x in (.0625,.125,.25,.5,.75,1.)]))
                selected=None
                budget=udp.n_dv_exception - sum(
                    dv(candidate[z],candidate[z+1],dnew[z],fnew[z]) > normal+1e-6
                    for z in range(s))
                for dep in candidates:
                    if dep < curr - 1e-8 or dep < 0:continue
                    value=dv(candidate[s],candidate[s+1],dep,fnew[s])
                    if value <= exceptional+1e-6 and (value <= normal+1e-6 or budget > 0):
                        selected=dep; break
                if selected is None:
                    curr=math.inf;break
                dnew[s]=selected
                curr=selected+fnew[s]
                if curr >= best:break
            if curr >= best - 1e-7:continue
            checked += 1
            fit=udp.fitness(dnew + fnew + candidate)
            if fit[0] < best - 1e-7 and all(abs(float(v)) < 1e-7 for v in fit[1:]):
                best=float(fit[0])
                newlegs=[[candidate[i],candidate[i+1],dnew[i],fnew[i],
                          dv(candidate[i],candidate[i+1],dnew[i],fnew[i])] for i in range(n)]
                payload={'path':candidate,'transfers':newlegs,'arrival_day':best,
                         'exceptions':sum(t[4] > normal+1e-6 for t in newlegs),
                         'validation':{'fitness':fit,'valid':True},
                         'source':args.solution,'method':'local_route_timing_mutation'}
                dest=out/f'medium_route_valid_{best:.9f}.json'
                dest.write_text(json.dumps(payload,indent=2))
                print('NEW ESA-VALID BEST',best,'saved:',dest,flush=True)
        if trial % 25==0:
            print(f'Trials {trial}, candidate blocks {accepted}, full ESA checks {checked}, '
                  f'Lambert calls {calls}, best {best:.8f}',flush=True)
    print(f'Finished. Best ESA-validated result: {best:.9f} days; '
          f'improvement {float(incumbent[0])-best:.9f} days; Lambert calls {calls}',flush=True)

if __name__ == '__main__':main()
